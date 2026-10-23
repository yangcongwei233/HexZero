#!/usr/bin/env python3
"""Recheck selected disagreements more deeply, then let Azalea play both sides.

Counterfactual rollouts demonstrate what this engine can convert, not perfect
play or what the original trained model would achieve after a suggested move.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.review_azalea_losses as R


def rollout(board, side, forced, simulations, seed):
    import numpy as np
    from hex.HexLogic import has_won_bfs
    engine = R.ENGINE
    engine.policy.tree.reset()
    engine.policy.ply = int(np.count_nonzero(board))
    engine.policy.simulations = simulations
    engine.policy.rng.seed(seed)
    engine.environment.board = board.copy()
    engine.environment.player = side
    engine.environment.winner = None
    engine.environment._snapshot = None
    board = board.copy()
    mover, moves = side, []
    action = forced
    while True:
        assert not has_won_bfs(board, mover) and not has_won_bfs(board, -mover)
        engine.observe(board, mover, action)
        moves.append(dict(player=mover, action=int(action)))
        board, mover = R.GAME.getNextState(board, mover, action)
        winner = 1 if has_won_bfs(board, 1) else -1 if has_won_bfs(board, -1) else None
        assert engine.environment.winner == winner
        if winner:
            return dict(winner=winner, trained_side_won=winner == side,
                        continuation_plies=len(moves), moves=moves, final_board=board.tolist())
        assert len(moves) <= board.size
        action, _ = engine.choose(board, mover)


def verify(task):
    import numpy as np
    started = time.perf_counter()
    row = task['position']
    board, side = np.asarray(row['board'], dtype=np.int8), row['player']
    branches = []
    for name, action in (('actual', row['actual']), ('alternative', row['deep']['best_action'])):
        value = R.after_move(board, side, action, task['simulations'], row['seed'])
        played = rollout(board, side, action, task['rollout_simulations'], row['seed'])
        branches.append(dict(kind=name, **value, rollout=played))
    return dict(game=task['game'], ply=row['ply'], board=row['board'],
                player=side, trained_live_value=row['trained_live_value'],
                branches=branches, deep_gap=branches[1]['value'] - branches[0]['value'],
                seconds=round(time.perf_counter() - started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-per-seat', type=int, default=3)
    parser.add_argument('--simulations', type=int, default=12800)
    parser.add_argument('--rollout-simulations', type=int, default=800)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if min(args.top_per_seat, args.simulations, args.rollout_simulations, args.workers) < 1:
        parser.error('Counts must be positive')
    data = json.loads(args.review.read_text())
    assert data['status'] == 'complete'
    cases = []
    for game in data['games']:
        rows = [r for r in game['positions'] if 'deep' in r and r['deep']['best_action'] != r['actual']]
        if rows:
            row = max(rows, key=lambda r: r['deep']['gap'])
            cases.append(dict(game=game['game'], position=row,
                              simulations=args.simulations, rollout_simulations=args.rollout_simulations))
    tasks = []
    for side in (1, -1):
        tasks.extend(sorted([c for c in cases if c['position']['player'] == side],
                            key=lambda c: -c['position']['deep']['gap'])[:args.top_per_seat])
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=R.initialize,
                             initargs=(Path(data['settings']['reference_source']),
                                       Path(data['settings']['reference_checkpoint']), 11)) as pool:
        for future in as_completed([pool.submit(verify, t) for t in tasks]):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(opening=result['game']['opening'], side=result['player'],
                                  ply=result['ply'], gap=result['deep_gap'],
                                  branches=[dict(kind=b['kind'], action=b['action'], value=b['value'],
                                                 trained_side_won=b['rollout']['trained_side_won'])
                                            for b in result['branches']]), ensure_ascii=False), flush=True)
    results.sort(key=lambda r: -r['deep_gap'])
    R.save_json(args.output, dict(status='complete', source_review=str(args.review.resolve()),
                                 source_review_sha256=R.digest(args.review),
                                 simulations=args.simulations, rollout_simulations=args.rollout_simulations,
                                 source_reference_sha256=data['settings']['reference_sha256'],
                                 selection='Highest 3200 gap per game, top three games per seat (default).',
                                 rollout_note='Original Azalea controls BOTH sides, fresh initial tree, reused within each continuation. One seed per branch. Not a forced-win proof or trained-model win rate.',
                                 cases=results))


if __name__ == '__main__':
    main()
