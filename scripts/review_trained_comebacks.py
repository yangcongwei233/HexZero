#!/usr/bin/env python3
"""Use the immutable trained checkpoint to review Azalea's losses locally.

All saved Q values use the trained side's perspective. Thus Azalea's best
defense MINIMIZES Q. Original live scores, fresh searches and exact rules are
kept separate. Evaluation records never become training data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_azalea_losses import digest, save_json

GAME = NET = TRAINED = REFERENCE = None


def initialize(checkpoint, source, reference_checkpoint):
    global GAME, NET, TRAINED, REFERENCE
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HEX_FORCE_CPU'] = '1'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from hex.HexGame import HexGame
    from hex.pytorch.NNet import NNetWrapper
    from hex.local_players import AzaleaReference, TrainedPlayer
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['board_size'][0] == state['board_size'][1] == 11
    GAME = HexGame(11)
    NET = NNetWrapper(GAME, config={'cuda': False})
    NET.load_training_state({k: v for k, v in state.items() if k not in ('optimizer_state_dict', 'rng_state')})
    NET.nnet.eval()
    TRAINED = TrainedPlayer(GAME, NET, 800, 0)
    REFERENCE = AzaleaReference(source, reference_checkpoint, 11, 'cpu').player(800, 0)


def reset_engine(kind, board, player, simulations, seed):
    import numpy as np
    engine = TRAINED if kind == 'trained' else REFERENCE
    policy = engine.policy
    if kind == 'trained':
        policy.startGame()
        policy.args.numMCTSSims = simulations
        policy.mcts.rng = np.random.default_rng(seed)
    else:
        policy.tree.reset()
        policy.ply = int(np.count_nonzero(board))
        policy.simulations = simulations
        policy.rng.seed(seed)
        engine.environment.board = board.copy()
        engine.environment.player = player
        engine.environment.winner = None
        engine.environment._snapshot = None
    return engine


def search(board, player, trained_side, simulations, seed, kind='trained'):
    """Fresh tree, fixed perspective, original engine-specific batch/cpuct."""
    import numpy as np
    from hex.HexLogic import has_won_bfs
    from hex.local_analysis import tree_candidates
    assert not has_won_bfs(board, 1) and not has_won_bfs(board, -1)
    engine = reset_engine(kind, board, player, simulations, seed)
    action, info = engine.choose(board, player)
    candidates = tree_candidates(engine, board, player, GAME, kind, 3, 8)
    sign = player * trained_side
    for candidate in candidates:
        if candidate['value'] is not None:
            candidate['value'] *= sign
        for node in candidate['pv']:
            if node['value'] is not None:
                node['value'] *= sign
    if kind == 'trained':
        visits = TRAINED.policy.mcts.tree.root.N
        assert visits == simulations - 1  # One evaluation is spent on the root.
    else:
        visits = int(REFERENCE.policy.tree.root.num_visits)
        assert np.array_equal(REFERENCE.environment.board, board)
        assert REFERENCE.environment.player == player
    return dict(action=int(action), value=info['value'] * sign,
                visits=visits, candidates=candidates)


def after_move(board, player, action, trained_side, simulations, seed, kind='trained'):
    from hex.HexLogic import has_won_bfs
    next_board, next_player = GAME.getNextState(board, player, int(action))
    if has_won_bfs(next_board, player):
        return dict(action=int(action), value=float(player * trained_side),
                    visits=0, exact_terminal=True, pv=[int(action)])
    evaluated = search(next_board, next_player, trained_side, simulations, seed, kind)
    return dict(action=int(action), value=evaluated['value'], visits=evaluated['visits'],
                exact_terminal=False,
                pv=[int(action)] + [m['action'] for m in evaluated['candidates'][0]['pv']])


def positions_from_record(record, game, checkpoint_sha):
    import numpy as np
    from hex.HexLogic import has_won_bfs
    assert record['models']['trained']['metadata']['sha256'] == checkpoint_sha
    assert record['models']['trained']['metadata']['iteration'] == game['model_iteration']
    board = np.zeros((11, 11), dtype=np.int8)
    positions = []
    for i, move in enumerate(record['moves']):
        assert move['move'] == i + 1 and move['player'] == (1 if i % 2 == 0 else -1)
        assert move['actor'] == ('trained' if move['player'] == game['trained_side'] else 'azalea')
        assert divmod(move['action'], 11) == (move['row'], move['col'])
        assert not has_won_bfs(board, 1) and not has_won_bfs(board, -1)
        positions.append(board.copy())
        assert board.ravel()[move['action']] == 0
        board.ravel()[move['action']] = move['player']
    assert np.array_equal(board, record['board'])
    assert record['winnerKind'] == game['winner'] == 'trained'
    assert record['winner'] == game['winner_side'] == game['trained_side']
    assert has_won_bfs(board, game['trained_side'])
    actions = [m['action'] for m in record['moves']]
    assert hashlib.sha256(json.dumps(actions).encode()).hexdigest() == game['trajectory_sha256']
    prefix = game['opening']['actions']
    assert actions[:len(prefix)] == prefix
    return positions


def review(task):
    from scripts.analyze_match_tactics import immediate_wins
    started = time.perf_counter()
    game, settings = task['game'], task['settings']
    record = json.loads(Path(game['record']).read_text())
    positions = positions_from_record(record, game, settings['checkpoint_sha256'])
    side, rows = game['trained_side'], []
    moves = record['moves']
    prefix = len(game['opening']['actions'])
    opening = search(positions[prefix], moves[prefix]['player'], side, settings['screen_simulations'], 0)
    for i, move in enumerate(moves):
        if move['actor'] != 'azalea' or i < prefix or move.get('source') in ('preset_opening', 'manual_opening'):
            continue
        board, mover, actual = positions[i], move['player'], move['action']
        seed = (int(game['trajectory_sha256'][:8], 16) + i * 1009) % (2 ** 32)
        before = search(board, mover, side, settings['screen_simulations'], seed)
        after = after_move(board, mover, actual, side, settings['screen_simulations'], seed)
        own_wins = immediate_wins(board.copy(), mover, verify=True)
        other_wins = immediate_wins(board.copy(), -mover, verify=True)
        rows.append(dict(ply=i + 1, player=mover, trained_side=side, board=board.tolist(),
                         actual=actual, suggested=before['action'], before_q=before['value'],
                         after_q=after['value'], screen_gain=after['value'] - before['value'],
                         before_visits=before['visits'], after_visits=after['visits'],
                         azalea_live_q=-float(move['value']),
                         trained_reply=moves[i + 1]['action'], trained_live_reply_q=moves[i + 1]['value'],
                         candidates=before['candidates'], own_winning_moves=own_wins,
                         opponent_winning_moves=other_wins,
                         missed_immediate_win=bool(own_wins and actual not in own_wins),
                         missed_unique_block=bool(not own_wins and len(other_wins) == 1 and actual != other_wins[0]),
                         already_lost_double_threat=bool(not own_wins and len(other_wins) >= 2), seed=seed))
    different = [r for r in rows if r['actual'] != r['suggested']]
    selected = sorted(different, key=lambda r: (-r['screen_gain'], r['ply']))[:settings['deep_positions_per_game']]
    for row in selected:
        board, mover, seed = positions[row['ply'] - 1], row['player'], row['seed']
        deep = search(board, mover, side, settings['deep_simulations'], seed)
        actions = {row['actual'], row['suggested'], deep['action']}
        actions.update(c['action'] for c in deep['candidates'])
        branches = [after_move(board, mover, a, side, settings['deep_simulations'], seed) for a in sorted(actions)]
        branches.sort(key=lambda b: (b['value'], b['action']))  # Opponent minimizes OUR score.
        actual = next(b for b in branches if b['action'] == row['actual'])
        reply_board = positions[row['ply']]
        reply = search(reply_board, side, side, settings['deep_simulations'], seed)
        live_reply = after_move(reply_board, side, row['trained_reply'], side, settings['deep_simulations'], seed)
        row['deep'] = dict(before_q=deep['value'], suggested=deep['action'], candidates=deep['candidates'],
                           branches=branches, defense=branches[0]['action'], defense_q=branches[0]['value'],
                           actual_q=actual['value'], gain=actual['value'] - branches[0]['value'],
                           trained_reply_suggestion=reply['action'], trained_reply_value=live_reply['value'])
    # Confirm possible midgame disadvantage on identical boards with both
    # engines. Select at most two low positions, one from replay and one from
    # live scores; these are samples, not exhaustive deep searches of all plies.
    timeline = [(r['before_q'], r['ply']) for r in rows if r['ply'] >= 11]
    timeline += [(r['after_q'], r['ply'] + 1) for r in rows if r['ply'] + 1 >= 11]
    own_live = [m for m in moves if m['actor'] == 'trained' and 'value' in m]
    own_midgame = [m for m in own_live if m['move'] >= 11]
    min_live = min(own_live, key=lambda m: m['value'])
    floor_plies = set()
    if timeline:
        q, ply = min(timeline)
        if q <= -.35:
            floor_plies.add(ply)
    if own_midgame:
        m = min(own_midgame, key=lambda m: m['value'])
        if m['value'] <= -.35:
            floor_plies.add(m['move'])
    floors = []
    for ply in sorted(floor_plies):
        board, mover = positions[ply - 1], moves[ply - 1]['player']
        seed = (int(game['trajectory_sha256'][:8], 16) + ply * 1013) % (2 ** 32)
        trained = search(board, mover, side, settings['deep_simulations'], seed)
        azalea = search(board, mover, side, settings['deep_simulations'], seed, 'azalea')
        floors.append(dict(ply=ply, player=mover, board=board.tolist(), seed=seed,
                           trained_q=trained['value'], azalea_q=azalea['value'],
                           trained_suggestion=trained['action'], azalea_suggestion=azalea['action']))
    confirmed = any(f['trained_q'] <= -.5 and f['azalea_q'] <= -.5 for f in floors)
    counts = {key: sum(r[key] for r in rows) for key in
              ('missed_immediate_win', 'missed_unique_block', 'already_lost_double_threat')}
    return dict(game=game, source_results=task['source_results'],
                record_sha256=digest(game['record']), settings_signature=task['signature'],
                positions=rows, floors=floors, confirmed_midgame_comeback=confirmed,
                live_min=dict(ply=min_live['move'], value=min_live['value']),
                opening_q=opening['value'], counts=counts,
                moves=[dict(m, value_ours=m.get('value', 0) * m['player'] * side)
                       for m in moves], seconds=round(time.perf_counter() - started, 2))


def summarize(games):
    rows = [r for g in games for r in g['positions']]
    return dict(games=len(games), azalea_decisions=len(rows),
                deep_positions=sum('deep' in r for r in rows),
                live_pessimistic_wins=sum(g['live_min']['value'] <= -.5 for g in games),
                confirmed_midgame_comebacks=sum(g['confirmed_midgame_comeback'] for g in games),
                by_seat={name: dict(wins=sum(g['game']['trained_side'] == side for g in games),
                                   confirmed_comebacks=sum(g['game']['trained_side'] == side and
                                                          g['confirmed_midgame_comeback'] for g in games))
                         for side, name in ((1, 'first_blue'), (-1, 'second_red'))},
                games_with_defense_gap_03=sum(any(r.get('deep', {}).get('gain', 0) >= .3 for r in g['positions']) for g in games),
                exact_tactics={key: sum(g['counts'][key] for g in games) for key in
                              ('missed_immediate_win', 'missed_unique_block', 'already_lost_double_threat')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/local_play/archive/iteration_750/trained.pth.tar')
    parser.add_argument('--reference-source', type=Path, default=ROOT.parent / 'azalea')
    parser.add_argument('--reference-checkpoint', type=Path, default=ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--screen-simulations', type=int, default=800)
    parser.add_argument('--deep-simulations', type=int, default=3200)
    parser.add_argument('--deep-positions-per-game', type=int, default=2)
    parser.add_argument('--limit-games', type=int)
    args = parser.parse_args()
    if min(args.screen_simulations, args.deep_simulations) < 2 or min(args.workers, args.deep_positions_per_game) < 1:
        parser.error('Search counts must be >=2; other counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'games').mkdir(exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ('output', 'workers', 'limit_games')}
    settings = json.loads(json.dumps(settings, default=str))
    settings.update(checkpoint_sha256=digest(args.checkpoint), reference_sha256=digest(args.reference_checkpoint),
                    script_sha256=digest(__file__), input_sha256={str(p): digest(p) for p in args.results},
                    device='cpu', primary_reviewer='trained', q_perspective='trained_side',
                    batch_size=32, cpuct=.5, noise=False, temperature=0, fresh_tree=True)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    tasks, seen = [], set()
    for source in args.results:
        data = json.loads(source.read_text())
        assert data['status'] == 'complete'
        for game in data['games']:
            if game['winner'] != 'trained' or game['trajectory_sha256'] in seen:
                continue
            seen.add(game['trajectory_sha256'])
            tasks.append(dict(game=game, source_results=str(source.resolve()), settings=settings, signature=signature))
    if args.limit_games:
        tasks = tasks[:args.limit_games]
    if not tasks:
        parser.error('No trained-model wins found')
    save_json(args.output / 'run.json', dict(settings=settings, signature=signature,
                                            created_at=datetime.now(timezone.utc).isoformat(),
                                            games=len(tasks), workers=args.workers, writes_training_data=False))
    games, pending = [], []
    for task in tasks:
        path = args.output / 'games' / (task['game']['trajectory_sha256'] + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['settings_signature'] == signature
            assert saved['record_sha256'] == digest(task['game']['record'])
            games.append(saved)
        else:
            pending.append(task)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,
                             initargs=(args.checkpoint, args.reference_source, args.reference_checkpoint)) as pool:
        for future in as_completed([pool.submit(review, task) for task in pending]):
            game = future.result()
            save_json(args.output / 'games' / (game['game']['trajectory_sha256'] + '.json'), game)
            games.append(game)
            progress = dict(completed=len(games), total=len(tasks), elapsed_seconds=round(time.perf_counter() - started, 1),
                            confirmed_comebacks=sum(g['confirmed_midgame_comeback'] for g in games),
                            last_opening=game['game']['opening']['id'], last_side=game['game']['trained_side'])
            save_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    games.sort(key=lambda g: (not g['confirmed_midgame_comeback'], g['live_min']['value'], g['game']['opening']['id']))
    output = dict(status='complete', settings=settings, summary=summarize(games), games=games,
                  comeback_definition='A recorded win with a sampled position before ply >=11 where both models independently return Q <= -0.5 at 3200, from the trained side perspective.',
                  limitations=['Only wins are reviewed; counts are not an overall comeback probability.',
                               'Q is not calibrated win probability. Disadvantage is engine judgement, not a forced-loss proof.',
                               'Deep floors and alternative actions are sampled; absence of confirmation does not prove no comeback.',
                               'The trained model is the primary reviewer; Azalea only cross-checks sampled disadvantage positions.',
                               'All preset moves are excluded. No reviewed games are added to training.'])
    save_json(args.output / 'results.json', output)
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
