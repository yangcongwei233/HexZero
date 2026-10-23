"""Balanced evaluation with independent, reproducible games in spawn workers."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing as mp
import tempfile

import numpy as np
from tqdm import tqdm

from Arena import Arena
from hex.HexGame import HexGame
from utils import dotdict

log = logging.getLogger(__name__)
_worker_context = None


def policy_args(args, game_id, model_id):
    config = dotdict(dict(args))
    # Seeds depend on the game/model, never the worker or completion order.
    config.seed = int(np.random.SeedSequence([
        args.get('seed', 0), args.get('iteration', 0), 0xA2E0,
        game_id, model_id,
    ]).generate_state(1)[0])
    config.rootDirichletNoise = False
    return config


def play_arena_game(game, previous, candidate, args, game_id):
    if args.get('mctsBackend', 'classic') == 'azalea':
        from hex.azalea_mcts import AzaleaStylePolicy as Policy
    else:
        from hex.policies import ClassicPolicy as Policy
    players = [Policy(game, net, policy_args(args, game_id, model_id))
               for model_id, net in enumerate((previous, candidate))]
    swapped = bool(game_id % 2)
    first, second = players[::-1] if swapped else players
    result = Arena(first, second, game).playGame()
    # Always report from the previous model's perspective, independent of seat.
    winner = result * (-1 if swapped else 1) if result in (-1, 1) else 0
    return game_id, int(winner)


def _initialize_worker(board_size, folder, config, device):
    global _worker_context
    import torch
    from hex.pytorch.NNet import NNetWrapper

    torch.set_num_threads(int(config.get('arenaThreads', 1)))
    game = HexGame(board_size, win_check_mode=config.get('winCheckMode', 'numba'))
    networks = []
    for filename in ('previous.pth.tar', 'candidate.pth.tar'):
        net = NNetWrapper(game, config={'cuda': device == 'cuda'})
        net.load_checkpoint(folder, filename)
        networks.append(net)
    _worker_context = game, networks[0], networks[1], dotdict(config)


def _worker_game(game_id):
    return play_arena_game(*_worker_context, game_id)


def collect_results(results, num_games):
    seen = set()
    previous_wins = candidate_wins = draws = 0
    for game_id, winner in results:
        if not 0 <= game_id < num_games or game_id in seen:
            raise ValueError(f'Invalid or duplicate Arena game ID: {game_id}')
        if winner not in (-1, 0, 1):
            raise ValueError(f'Invalid Arena result: {winner}')
        seen.add(game_id)
        previous_wins += winner == 1
        candidate_wins += winner == -1
        draws += winner == 0
    if len(seen) != num_games:
        raise RuntimeError(f'Arena returned {len(seen)} of {num_games} games')
    return previous_wins, candidate_wins, draws


def _run_pool(game, folder, args, device):
    games = int(args.arenaCompare)
    workers = min(int(args.get('arenaWorkers', 1)), games)
    # Executor propagates initializer/process failures instead of repeatedly
    # respawning a broken initializer. CUDA must use spawn, never fork.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=_initialize_worker,
                             initargs=(game.n, folder, dict(args), device)) as pool:
        futures = [pool.submit(_worker_game, game_id) for game_id in range(games)]
        results = (future.result() for future in tqdm(
            as_completed(futures), total=games, desc='Parallel Arena'))
        return collect_results(results, games)


def run_parallel_arena(game, previous, candidate, args):
    games = int(args.arenaCompare)
    workers = int(args.get('arenaWorkers', 1))
    threads = int(args.get('arenaThreads', 1))
    if games < 2 or games % 2:
        raise ValueError('Arena game count must be a positive even number')
    if workers < 1 or threads < 1:
        raise ValueError('Arena workers and threads must be positive')
    device = args.get('arenaDevice') or ('cuda' if candidate.args.cuda else 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Arena device must be cpu or cuda')
    log.info('Parallel Arena: games=%s workers=%s device=%s; each model starts %s games',
             games, min(workers, games), device, games // 2)
    # Only two inference snapshots per evaluation; no optimizer/replay copies.
    # The directory is removed after workers exit, including on failure.
    with tempfile.TemporaryDirectory(prefix='hex-arena-') as folder:
        previous.save_checkpoint(folder, 'previous.pth.tar', include_training=False)
        candidate.save_checkpoint(folder, 'candidate.pth.tar', include_training=False)
        return _run_pool(game, folder, args, device)
