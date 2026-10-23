import logging
import json
import multiprocessing as mp
import os
import time
import tempfile
from collections import deque

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from hex.HexGame import HexGame
from hex.value_diagnostics import example_value_stats, format_value_stats
from utils import dotdict


log = logging.getLogger(__name__)


def _new_mcts(game, nnet, args):
    if getattr(args, 'mctsBackend', 'classic') == 'azalea':
        from hex.azalea_mcts import AzaleaMCTS
        return AzaleaMCTS(game, nnet, args)
    return MCTS(game, nnet, args)


def _execute_episode(game, nnet, args, seed, worker_id=0, game_id=0, checkpoint_file=None):
    rng = np.random.default_rng(seed)
    train_examples = []
    board = game.getInitBoard()
    cur_player = 1
    episode_step = 0
    last_move = None
    episode_args = dotdict(dict(args))
    episode_args.seed = int(seed)
    mcts = _new_mcts(game, nnet, episode_args)
    moves = []
    policies = []
    search_times = []

    while True:
        episode_step += 1
        canonical_board = game.getCanonicalForm(board, cur_player)
        temp = int(episode_step < args.tempThreshold)
        raw_action = None

        if getattr(args, 'useOpeningBook', False):
            from hex.opening_book import opening_book_action
            raw_action = opening_book_action(board, cur_player, game)
        if raw_action is None and getattr(args, 'useTacticalCheck', False):
            from hex.hex_tactics import tactical_action
            raw_action = tactical_action(board, cur_player, game, last_move=last_move)

        if raw_action is None:
            search_start = time.perf_counter()
            pi = mcts.getActionProb(canonical_board, temp=temp)
            canonical_action = int(rng.choice(len(pi), p=pi))
            raw_action = game.actionFromCanonical(canonical_action, cur_player)
            search_times.append(time.perf_counter() - search_start)
        else:
            canonical_action = game.actionToCanonical(raw_action, cur_player)
            pi = np.zeros(game.getActionSize(), dtype=np.float32)
            pi[canonical_action] = 1.0

        for b, p in game.getSymmetries(canonical_board, pi):
            train_examples.append([b, cur_player, p, None])

        if hasattr(mcts, 'update_with_move'):
            mcts.update_with_move(canonical_action)
        moves.append(int(raw_action))
        if getattr(args, 'savePolicyInJsonl', False):
            policies.append(np.asarray(pi, dtype=float).tolist())
        board, cur_player = game.getNextState(board, cur_player, raw_action)
        last_move = raw_action
        result = game.getGameEnded(board, cur_player)

        if result != 0:
            examples = [(x[0], x[2], result * ((-1) ** (x[1] != cur_player))) for x in train_examples]
            winner = int(cur_player * result)
            record = {
                "board_size": int(game.n),
                "worker_id": int(worker_id),
                "game_id": int(game_id),
                "iteration": int(getattr(args, 'iteration', 0)),
                "policy_coordinates": "canonical",
                "seed": int(seed),
                "checkpoint": checkpoint_file,
                "mcts_backend": getattr(args, 'mctsBackend', 'classic'),
                "mcts_batch_size": int(getattr(args, 'mctsBatchSize', 1)),
                "winner": winner,
                "moves": moves,
                "final_board": np.asarray(board, dtype=np.int8).tolist(),
                "avg_search_time": float(np.mean(search_times)) if search_times else 0.0,
                "heuristic_prior_weight": float(getattr(args, 'heuristicPriorWeight', 0.0)),
            }
            if getattr(args, 'savePolicyInJsonl', False):
                record["policies"] = policies
            return examples, record


def _worker(task):
    board_size, checkpoint_folder, checkpoint_file, args_dict, num_games, seed, worker_id = task

    if args_dict.get('selfplayDevice', 'cpu') == 'cpu':
        os.environ['HEX_FORCE_CPU'] = '1'
    else:
        os.environ.pop('HEX_FORCE_CPU', None)
    import torch
    from hex.pytorch.NNet import NNetWrapper

    torch.set_num_threads(int(args_dict.get('selfplayThreads', 1)))
    game = HexGame(board_size, win_check_mode=args_dict.get('winCheckMode', 'numba'))
    nnet = NNetWrapper(game, config={'cuda': args_dict.get('selfplayDevice', 'cpu') == 'cuda'})
    nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
    args = dotdict(args_dict)

    examples = []
    records = []
    for game_id in range(num_games):
        game_examples, record = _execute_episode(
            game, nnet, args, episode_seed(seed, args.get('iteration', 0), game_id), worker_id=worker_id, game_id=game_id,
            checkpoint_file=checkpoint_file,
        )
        examples.extend(game_examples)
        records.append(record)

    for record in records:
        save_game_record(record, args, args.checkpoint)
    result = {
        'worker_id': worker_id,
        'examples': examples,
    }
    if getattr(args, 'debugValueDiagnostics', False):
        result['value_stats'] = example_value_stats(examples)
    return result


def episode_seed(base_seed, iteration, game_id):
    return int(np.random.SeedSequence([int(base_seed), int(iteration), int(game_id)]).generate_state(1)[0])


def save_game_record(record, args, checkpoint_folder):
    if not getattr(args, 'saveSelfplayJsonl', False):
        return
    out_dir = getattr(args, 'selfplayOutputDir', None) or os.path.join(checkpoint_folder, 'selfplay_jsonl')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'worker_{}.jsonl'.format(record['worker_id']))
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, separators=(',', ':')) + '\n')


def _split_work(total_games, num_workers, games_per_worker):
    if games_per_worker is None or games_per_worker <= 0:
        games_per_worker = max(1, int(np.ceil(float(total_games) / max(1, num_workers))))

    chunks = []
    remaining = total_games
    while remaining > 0:
        chunk = min(games_per_worker, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def run_parallel_self_play(game, nnet, args, checkpoint_file='_parallel_selfplay.pth.tar'):
    # Workers still load independently; the inference-only handoff is temporary.
    with tempfile.TemporaryDirectory(prefix='hex-selfplay-') as folder:
        nnet.save_checkpoint(folder=folder, filename=checkpoint_file, include_training=False)
        return _run_workers(game, args, folder, checkpoint_file)


def _run_workers(game, args, checkpoint_folder, checkpoint_file):
    num_workers = max(1, int(getattr(args, 'numWorkers', 1)))
    games_per_worker = getattr(args, 'gamesPerWorker', None)
    chunks = _split_work(args.numEps, num_workers, games_per_worker)
    if not chunks:
        raise ValueError('numEps must be positive')

    log.info(
        'Parallel self-play: episodes=%s workers=%s tasks=%s checkpoint=%s',
        args.numEps,
        num_workers,
        len(chunks),
        checkpoint_file,
    )

    args_dict = dict(args)
    args_dict['parallelSelfPlay'] = False
    tasks = [
        (
            game.n,
            checkpoint_folder,
            checkpoint_file,
            args_dict,
            chunk,
            episode_seed(args_dict.get('seed', 0), args_dict.get('iteration', 0), task_id),
            task_id,
        )
        for task_id, chunk in enumerate(chunks)
    ]

    examples = deque([], maxlen=args.maxlenOfQueue)
    ctx = mp.get_context('spawn')
    pending = {}
    next_task = 0
    with ctx.Pool(processes=min(num_workers, len(tasks))) as pool:
        for result in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc='Parallel Self Play'):
            if isinstance(result, dict):
                if 'value_stats' in result:
                    log.info(
                        'Parallel self-play worker %s value targets: %s',
                        result['worker_id'],
                        format_value_stats(result['value_stats']),
                    )
                pending[result['worker_id']] = result['examples']
                # Completion stays asynchronous; replay ordering stays reproducible.
                while next_task in pending:
                    examples.extend(pending.pop(next_task))
                    next_task += 1
            else:
                raise TypeError('Unexpected self-play worker result')

    return examples
