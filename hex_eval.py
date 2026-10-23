import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Arena import Arena
from hex.policies import ClassicPolicy
from hex.HexGame import HexGame
from hex.azalea_mcts import AzaleaStylePolicy
from hex.pytorch.NNet import NNetWrapper
from utils import dotdict


log = logging.getLogger(__name__)


class RandomPlayer:
    def __init__(self, game, seed=0):
        self.game = game
        self.rng = np.random.default_rng(seed)

    def startGame(self):
        pass

    def notify(self, board, action):
        pass

    def __call__(self, canonical_board):
        valids = np.flatnonzero(self.game.getValidMoves(canonical_board, 1))
        return int(self.rng.choice(valids))


MCTSPlayer = ClassicPolicy
EvalAzaleaPlayer = AzaleaStylePolicy


class ModelSpec:
    def __init__(self, name, path=None, nnet=None):
        self.name = name
        self.path = path
        self.nnet = nnet


def build_args(cli_args):
    return dotdict({
        "numMCTSSims": cli_args.num_mcts_sims,
        "cpuct": cli_args.cpuct,
        "heuristicPriorWeight": cli_args.heuristic_prior_weight,
        "mctsBackend": cli_args.mcts_backend,
        "mctsBatchSize": cli_args.mcts_batch_size,
        "reuseSearchTree": cli_args.reuse_search_tree,
        "rootDirichletNoise": False,
        "dirichletAlpha": cli_args.dirichlet_alpha,
        "dirichletFraction": cli_args.dirichlet_fraction,
        "useOpeningBook": cli_args.use_opening_book,
        "useTacticalCheck": cli_args.use_tactical_check,
    })


def load_model_specs(game, checkpoint_paths, include_random):
    specs = []
    if include_random:
        specs.append(ModelSpec("<random>"))

    for path in checkpoint_paths:
        path = Path(path)
        nnet = NNetWrapper(game)
        nnet.load_checkpoint(str(path.parent), path.name)
        specs.append(ModelSpec(path.name, str(path), nnet))
    return specs


def make_player(spec, game, args, seed):
    if spec.nnet is None:
        return RandomPlayer(game, seed=seed)
    if args.mctsBackend == "azalea":
        policy = EvalAzaleaPlayer(game, spec.nnet, args)
        policy.mcts.rng = np.random.default_rng(seed)
        return policy
    return MCTSPlayer(game, spec.nnet, args, seed=seed)


def play_pair(game, first_spec, second_spec, args, games, seed):
    if games < 2 or games % 2:
        raise ValueError("games-per-pair must be a positive even number")
    first_wins = 0
    second_wins = 0
    draws = 0

    for game_idx in range(games):
        pair_seed = seed + game_idx * 1009
        player1 = make_player(first_spec, game, args, pair_seed + 1)
        player2 = make_player(second_spec, game, args, pair_seed + 2)
        swapped = bool(game_idx % 2)
        arena = Arena(player2, player1, game) if swapped else Arena(player1, player2, game)
        result = arena.playGame(verbose=False)
        if result in (-1, 1) and swapped:
            result = -result
        if result == 1:
            first_wins += 1
        elif result == -1:
            second_wins += 1
        else:
            draws += 1

    return [first_wins, draws, second_wins]


def evaluate_round_robin(specs, game, args, games_per_pair, seed):
    outcomes = {}
    total_pairs = len(specs) * (len(specs) - 1) // 2
    pair_iter = (
        (i, j)
        for j in range(len(specs))
        for i in range(j)
    )
    for pair_index, (i, j) in enumerate(tqdm(pair_iter, total=total_pairs, desc="Round Robin"), start=1):
        result = play_pair(
            game,
            specs[i],
            specs[j],
            args,
            games_per_pair,
            seed + pair_index * 100003,
        )
        outcomes[(i, j)] = result
        total = sum(result)
        winrate = result[0] / total if total else 0.0
        log.info(
            "pair %s/%s: %s vs %s outcomes=%s first_winrate=%.3f",
            pair_index,
            total_pairs,
            specs[i].name,
            specs[j].name,
            result,
            winrate,
        )
    return outcomes


def compute_bradley_terry_elo(num_players, outcomes, anchor_index=0, max_iter=10000, tol=1e-10):
    wins = np.zeros(num_players, dtype=np.float64)
    games = defaultdict(float)

    for (i, j), (i_wins, draws, j_wins) in outcomes.items():
        draw_half = 0.5 * draws
        wins[i] += i_wins + draw_half
        wins[j] += j_wins + draw_half
        games[(i, j)] += i_wins + draws + j_wins

    # Small prior keeps ratings finite when one model has only wins or losses.
    strengths = np.ones(num_players, dtype=np.float64)
    prior = 0.5
    wins += prior
    for _ in range(max_iter):
        previous = strengths.copy()
        for i in range(num_players):
            denom = 0.0
            for j in range(num_players):
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                n_ij = games.get(pair, 0.0)
                if n_ij:
                    denom += (n_ij + 2 * prior / max(1, num_players - 1)) / (strengths[i] + strengths[j])
            if denom > 0:
                strengths[i] = max(1e-12, wins[i] / denom)
        strengths /= strengths[anchor_index]
        if np.max(np.abs(strengths - previous)) < tol:
            break

    return 400.0 * np.log10(strengths)


def write_results(path, specs, outcomes, elos, args):
    if not path:
        return
    payload = {
        "models": [{"name": spec.name, "path": spec.path} for spec in specs],
        "outcomes": [
            {"pair": [i, j], "result": result}
            for (i, j), result in outcomes.items()
        ],
        "elo": elos.tolist(),
        "args": dict(args),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hex checkpoints with round-robin Elo ranking.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint files, e.g. temp/hex_11x11/checkpoint_7.pth.tar")
    parser.add_argument("--board-size", type=int, default=11)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--num-mcts-sims", type=int, default=50)
    parser.add_argument("--mcts-backend", choices=("classic", "azalea"), default="azalea")
    parser.add_argument("--mcts-batch-size", type=int, default=16)
    parser.add_argument("--reuse-search-tree", action="store_true")
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--heuristic-prior-weight", type=float, default=0.0)
    parser.add_argument("--win-check-mode", choices=("bfs", "fast", "numba"), default="numba")
    parser.add_argument("--include-random", action="store_true", help="Use random player as 0 Elo reference.")
    parser.add_argument("--seed", type=int, default=0xBAD5EED5)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--use-opening-book", action="store_true")
    parser.add_argument("--use-tactical-check", action="store_true")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-fraction", type=float, default=0.25)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cli_args = parse_args()
    args = build_args(cli_args)
    game = HexGame(cli_args.board_size, win_check_mode=cli_args.win_check_mode)
    specs = load_model_specs(game, cli_args.checkpoints, cli_args.include_random)

    if len(specs) < 2:
        raise ValueError("Need at least two players. Provide two checkpoints or use --include-random.")

    outcomes = evaluate_round_robin(specs, game, args, cli_args.games_per_pair, cli_args.seed)
    elos = compute_bradley_terry_elo(len(specs), outcomes, anchor_index=0)
    write_results(cli_args.output_json, specs, outcomes, elos, args)

    print("\nElo ranking:")
    for idx in np.argsort(elos)[::-1]:
        print("{:8.1f}  {}".format(float(elos[idx]), specs[idx].name))

    print("\nPair outcomes: [first wins, draws, second wins]")
    for (i, j), result in outcomes.items():
        print("{} vs {}: {}".format(specs[i].name, specs[j].name, result))


if __name__ == "__main__":
    main()
