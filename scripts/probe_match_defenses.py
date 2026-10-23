#!/usr/bin/env python3
"""Replay selected prefixes and test counterfactual defenses with local models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import evaluate_local_models as evaluation


def probe(task):
    from hex.HexLogic import has_won_bfs
    from hex_play_server import HexSession
    game = json.loads(Path(task['record']).read_text())
    app = evaluation.APP
    session = HexSession(app, mode='azalea', mine=game['myPlayer'],
                         simulations=800, seed=game['seed'])
    red_sims = task.get('red_simulations', 800)
    red = session.engines[-1]
    if session.kind(-1) == 'trained':
        red.policy.args.numMCTSSims = red_sims
    else:
        red.policy.simulations = red_sims
    started = time.perf_counter()
    for move in game['moves'][:task['prefix_plies']]:
        session.apply(move['action'], {'source': 'replayed_prefix'})
    if task.get('defense') and not task.get('defense_ply'):
        cell = task['defense']
        action = (int(cell[1:]) - 1) * app.game.n + ord(cell[0]) - 65
        assert session.player == -1
        session.apply(action, {'source': 'counterfactual_defense'})
    while session.winner is None:
        if len(session.moves) + 1 == task.get('defense_ply'):
            cell = task['defense']
            action = (int(cell[1:]) - 1) * app.game.n + ord(cell[0]) - 65
            assert session.player == -1
            session.apply(action, {'source': 'counterfactual_defense'})
        else:
            session.step(session.revision)
        expected = 1 if has_won_bfs(session.board, 1) else -1 if has_won_bfs(session.board, -1) else None
        assert session.winner == expected
    # The web record has a common per-side budget. Explicitly annotate unequal
    # budgets here so these diagnostic games cannot be mistaken for fair matches.
    record = session.record()
    record['diagnostic'] = dict(task, blue_simulations=800, red_simulations=red_sims,
                               replay_search_state='Fresh trees; prefix moves observed without searches'
                               if task['prefix_plies'] else 'Normal search from the empty board')
    evaluation.save_json(Path(session.saved_path), record)
    return dict(task=task, winner=session.kind(session.winner), winner_side=session.winner,
                moves=len(session.moves), seconds=round(time.perf_counter() - started, 3),
                record=session.saved_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(args.tasks.read_text())
    root = evaluation.ROOT
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                             initializer=evaluation.initialize,
                             initargs=(str(root / 'models/local_play/archive/iteration_350/trained.pth.tar'),
                                       str(root.parent / 'azalea'),
                                       str(root.parent / 'azalea/models/hex11-20180712-3362.policy.pth'),
                                       str(args.output), 800)) as pool:
        for future in as_completed([pool.submit(probe, task) for task in tasks]):
            result = future.result()
            results.append(result)
            evaluation.save_json(args.output / 'results.json', dict(status='running', games=results))
            print(json.dumps(result), flush=True)
    evaluation.save_json(args.output / 'results.json', dict(status='complete', games=results))


if __name__ == '__main__':
    main()
