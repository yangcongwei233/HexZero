#!/usr/bin/env python3
"""采集原版 Azalea 自博弈，生成软策略分布与终局结果训练目标。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from hex.teacher_data import canonical_target, file_digest, save_dataset

REFERENCE = None
SETTINGS = None
FORBIDDEN = None


def opening_key(actions, n):
    pair = tuple(actions[:2])
    return min(pair, tuple(n * n - 1 - action for action in pair))


def initialize(settings, forbidden):
    global REFERENCE, SETTINGS, FORBIDDEN
    import torch
    from hex.local_players import AzaleaReference
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    SETTINGS, FORBIDDEN = settings, set(map(tuple, forbidden))
    REFERENCE = AzaleaReference(settings['reference_source'], settings['reference_checkpoint'],
                                settings['board_size'], settings['device'])


def collect(game_id):
    from hex.HexLogic import has_won_bfs
    n, settings = SETTINGS['board_size'], SETTINGS
    for attempt in range(1000):
        seed = int(np.random.SeedSequence([settings['seed'], game_id, attempt]).generate_state(1)[0])
        engine = REFERENCE.player(settings['simulations'], seed)
        policy, env = engine.policy, engine.environment
        policy.settings = {'move_sampling': True, 'move_exploration': True}
        policy.exploration_depth = settings['temperature_plies']
        policy.exploration_temperature = 1.0
        policy.exploration_noise_alpha = settings['noise_alpha']
        policy.exploration_noise_scale = settings['noise_fraction']
        boards, policies, players, moves = [], [], [], []
        rejected = False
        while env.winner is None:
            board, player = env.board.copy(), env.player
            action, _ = engine.choose(board, player)
            canonical, pi = canonical_target(env.game, board, player, env.state.legal_moves,
                                               policy.tree.root.move_stats.num_visits.copy())
            boards.append(canonical); policies.append(pi); players.append(player); moves.append(action)
            engine.observe(board, player, action)
            independent = 1 if has_won_bfs(env.board, 1) else -1 if has_won_bfs(env.board, -1) else None
            if env.winner != independent:
                raise RuntimeError('Teacher terminal check disagrees with independent BFS')
            if len(moves) == 2 and opening_key(moves, n) in FORBIDDEN:
                rejected = True
                break
        if rejected:
            del engine, policy, env
            continue
        count = len(moves)
        arrays = dict(boards=np.asarray(boards, dtype=np.int8), policies=np.asarray(policies, dtype=np.float32),
                      players=np.asarray(players, dtype=np.int8),
                      values=np.asarray(players, dtype=np.int8) * env.winner,
                      game_ids=np.full(count, game_id, dtype=np.int32), plies=np.arange(1, count + 1, dtype=np.int16))
        record = dict(game_id=game_id, seed=seed, rejected_openings=attempt, winner=int(env.winner),
                      moves=moves, samples=count,
                      search=dict(simulations=settings['simulations'], batch_size=policy.search_batch_size,
                                  cpuct=policy.exploration_coef))
        return arrays, record
    raise RuntimeError('Could not generate a non-held-out opening')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, default=ROOT.parent / 'azalea')
    parser.add_argument('--reference-checkpoint', type=Path, default=ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth')
    parser.add_argument('--exclude-openings', type=Path, required=True)
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=800)
    parser.add_argument('--board-size', type=int, default=11)
    parser.add_argument('--temperature-plies', type=int, default=15)
    parser.add_argument('--noise-alpha', type=float, default=.03)
    parser.add_argument('--noise-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=2026101502)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if (min(args.games, args.workers, args.simulations, args.board_size, args.temperature_plies) < 1
            or args.seed < 0 or args.noise_alpha <= 0 or not 0 <= args.noise_fraction <= 1):
        parser.error('Invalid sampling parameters')
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    heldout = json.loads(args.exclude_openings.read_text())
    if any(len(item['actions']) != 2 or len(set(item['actions'])) != 2
           or any(type(a) is not int or not 0 <= a < args.board_size ** 2 for a in item['actions'])
           for item in heldout):
        parser.error('Held-out openings must be legal two-move prefixes')
    forbidden = sorted({opening_key(item['actions'], args.board_size) for item in heldout})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), board_size=args.board_size,
                    settings=settings, teacher_sha256=file_digest(args.reference_checkpoint),
                    excluded_openings_sha256=file_digest(args.exclude_openings), excluded_openings=forbidden,
                    rule='blue_top_bottom_red_left_right_no_swap',
                    note='Original Azalea self-play; both winners and losers; no forced opening labels')
    write_json(output / 'run.json', metadata)
    results, records, started = [], [], time.perf_counter()
    try:
        with (output / 'games.jsonl').open('w') as log, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize, initargs=(settings, forbidden)) as pool:
            futures = [pool.submit(collect, i) for i in range(args.games)]
            for future in as_completed(futures):
                arrays, record = future.result()
                results.append(arrays); records.append(record)
                log.write(json.dumps(record) + '\n'); log.flush()
                if len(records) % 8 == 0 or len(records) == args.games:
                    progress = dict(status='running', completed_games=len(records), target_games=args.games,
                                    raw_samples=sum(r['samples'] for r in records),
                                    elapsed_seconds=time.perf_counter() - started)
                    write_json(output / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
        # 按对局编号排序，保证数据字节和顺序不受任务完成顺序影响。
        order = sorted(range(len(records)), key=lambda i: records[i]['game_id'])
        arrays = {key: np.concatenate([results[i][key] for i in order]) for key in results[0]}
        hashes = {tuple(r['moves']) for r in records}
        metadata.update(games=len(records), unique_games=len(hashes), raw_samples=len(arrays['boards']),
                        wins_by_color=dict(Counter(r['winner'] for r in records)),
                        unique_openings=len({opening_key(r['moves'], args.board_size) for r in records}),
                        samples_by_color={str(p): int(np.sum(arrays['players'] == p)) for p in (1, -1)})
        if len(hashes) != len(records):
            raise RuntimeError('Duplicate full games detected; choose more diverse collection settings')
        if file_digest(args.reference_checkpoint) != metadata['teacher_sha256']:
            raise RuntimeError('Reference checkpoint changed during collection')
        path = output / 'teacher.npz'
        digest = save_dataset(path, arrays, metadata)
        final = dict(status='complete', **metadata, sha256=digest, bytes=path.stat().st_size,
                     elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', final)
        write_json(output / 'progress.json', final)
        print(json.dumps({key: final[key] for key in ('status', 'games', 'raw_samples', 'bytes', 'sha256', 'elapsed_seconds')}), flush=True)
    except BaseException as exc:
        write_json(output / 'progress.json', dict(status='failed', completed_games=len(records), error=str(exc)))
        raise


if __name__ == '__main__':
    main()
