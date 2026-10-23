#!/usr/bin/env python3
"""使用原版 Azalea 在本地复盘独立评测集中的败局。
不修改训练数据和模型，价值统一采用训练模型一方的视角。
初筛仅作启发，候选后继均以相同预算的新树复查；估值不等于严格解。"""
import argparse
from collections import Counter
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
ENGINE = GAME = None


def digest(path):
    """计算评测模型文件哈希，用于确认复盘使用的权重身份。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    """以 UTF-8 将复盘数据写入临时文件后原子替换。"""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def initialize(source, checkpoint, size):
    """在本地复盘进程中限制推理线程，加载原 Azalea 参考引擎和对应规则环境。"""
    global ENGINE, GAME
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from hex.HexGame import HexGame
    from hex.local_players import AzaleaReference
    GAME = HexGame(size)
    ENGINE = AzaleaReference(source, checkpoint, size, 'cpu').player(800, 0)


def label(action, n=11):
    """将动作转为旧分析报告坐标，采用从上向下的行编号。"""
    row, col = divmod(int(action), n)
    return chr(65 + col) + str(row + 1)


def search(board, player, simulations, seed):
    """每个复盘根从空树开始，不继承历史访问或对手搜索树。"""
    import numpy as np
    from hex.HexLogic import has_won_bfs
    from hex.local_analysis import tree_candidates
    if has_won_bfs(board, player) or has_won_bfs(board, -player):
        raise ValueError('Search requires a nonterminal position')
    policy, env = ENGINE.policy, ENGINE.environment
    policy.tree.reset()  # 复用已分配的存储，但清空统计，避免反复分配大数组。
    policy.ply = int(np.count_nonzero(board))
    policy.simulations = simulations
    policy.rng.seed(seed)
    env.board, env.player, env.winner = board.copy(), int(player), None
    env._snapshot = None
    action, info = ENGINE.choose(board, player)
    assert np.array_equal(env.board, board) and env.player == player
    root = policy.tree.root
    legal = np.flatnonzero(board.ravel() == 0)
    stats = root.move_stats
    edges = {int(a): dict(visits=int(stats.num_visits[i]),
                         value=float(stats.total_value[i] / stats.num_visits[i])
                         if stats.num_visits[i] else None,
                         prior=float(stats.prior_prob[i]))
             for i, a in enumerate(legal)}
    return dict(action=action, value=info['value'], root_visits=int(root.num_visits),
                candidates=tree_candidates(ENGINE, board, player, GAME, 'azalea', 3, 8),
                edges=edges)


def after_move(board, player, action, simulations, seed):
    """评估指定动作后的局面；立即获胜直接返回规则结果，否则以新树搜索对手局面并将价值取反一次。"""
    from hex.HexLogic import has_won_bfs
    next_board, next_player = GAME.getNextState(board, player, action)
    if has_won_bfs(next_board, player):
        return dict(action=int(action), value=1.0, exact_terminal=True,
                    root_visits=0, pv=[int(action)])
    result = search(next_board, next_player, simulations, seed)
    # 后继局面轮到对手行动，价值只取反一次。
    pv = result['candidates'][0]['pv'] if result['candidates'] else []
    return dict(action=int(action), value=-result['value'], exact_terminal=False,
                root_visits=result['root_visits'],
                pv=[int(action)] + [p['action'] for p in pv])


def validated_positions(record, result):
    """完整重放棋谱，核对模型与开局记录、合法落子和实际胜负，返回各步搜索前局面。"""
    import numpy as np
    from hex.HexLogic import has_won_bfs
    board = np.zeros((record['boardSize'], record['boardSize']), dtype=np.int8)
    positions = []
    for i, move in enumerate(record['moves']):
        assert move['move'] == i + 1 and move['player'] == (1 if i % 2 == 0 else -1)
        assert move['actor'] == ('trained' if move['player'] == result['trained_side'] else 'azalea')
        assert divmod(move['action'], record['boardSize']) == (move['row'], move['col'])
        assert not has_won_bfs(board, 1) and not has_won_bfs(board, -1)
        positions.append(board.copy())
        assert board.ravel()[move['action']] == 0
        board.ravel()[move['action']] = move['player']
    assert np.array_equal(board, record['board'])
    assert record['winnerKind'] == result['winner'] == 'azalea'
    assert record['winner'] == result['winner_side'] == -result['trained_side']
    assert has_won_bfs(board, record['winner']) and not has_won_bfs(board, -record['winner'])
    actions = [m['action'] for m in record['moves']]
    assert hashlib.sha256(json.dumps(actions).encode()).hexdigest() == result['trajectory_sha256']
    prefix = result.get('opening', {}).get('actions', [])
    assert actions[:len(prefix)] == prefix
    return positions


def review(task):
    """用参考引擎筛查败局分歧及短步战术，再以等预算新树深查选定候选，记录证据而不修改训练数据。"""
    from scripts.analyze_match_tactics import immediate_wins
    started = time.perf_counter()
    result, settings = task['game'], task['settings']
    record = json.loads(Path(result['record']).read_text())
    positions = validated_positions(record, result)
    rows = []
    prefix_len = len(result.get('opening', {}).get('actions', []))
    prefix_player = 1 if prefix_len % 2 == 0 else -1
    opening = search(positions[prefix_len], prefix_player, settings['screen_simulations'], 0)
    opening_value = opening['value'] * prefix_player * result['trained_side']
    for i, move in enumerate(record['moves']):
        if move['actor'] != 'trained' or i < prefix_len or move.get('source') in ('preset_opening', 'manual_opening'):
            continue
        board, side, action = positions[i], move['player'], move['action']
        seed = (int(result['trajectory_sha256'][:8], 16) + i * 1009) % (2 ** 32)
        before = search(board, side, settings['screen_simulations'], seed)
        after = after_move(board, side, action, settings['screen_simulations'], seed)
        own = immediate_wins(board.copy(), side, verify=True)
        other = immediate_wins(board.copy(), -side, verify=True)
        rows.append(dict(
            ply=i + 1, player=side, board=board.tolist(), actual=action,
            trained_live_value=move.get('value'), azalea_action=before['action'],
            azalea_root_value=before['value'], actual_after_value=after['value'],
            screen_drop=before['value'] - after['value'],
            actual_edge=before['edges'][action], candidates=before['candidates'],
            root_visits=before['root_visits'], actual_after_root_visits=after['root_visits'],
            own_winning_moves=own, opponent_winning_moves=other,
            missed_immediate_win=bool(own and action not in own),
            missed_unique_block=bool(not own and len(other) == 1 and action != other[0]),
            already_lost_double_threat=bool(not own and len(other) >= 2), seed=seed))
    # 每局最多深查 N 个最可疑分歧，并非穷举复盘。
    # 微小差异和败势中的走法保留记录，不直接判为错误。
    disagreements = [r for r in rows if r['actual'] != r['azalea_action']]
    selected = sorted(disagreements, key=lambda r: (
        -int(r['missed_immediate_win'] or r['missed_unique_block']),
        -r['screen_drop'], r['ply']))[:settings['deep_positions_per_game']]
    for row in selected:
        board, side, seed = positions[row['ply'] - 1], row['player'], row['seed']
        deep = search(board, side, settings['deep_simulations'], seed)
        actions = {row['actual'], row['azalea_action'], deep['action']}
        actions.update(c['action'] for c in deep['candidates'])
        branches = [after_move(board, side, a, settings['deep_simulations'], seed)
                    for a in sorted(actions)]
        branches.sort(key=lambda b: (-b['value'], b['action']))
        actual = next(b for b in branches if b['action'] == row['actual'])
        row['deep'] = dict(root_value=deep['value'], root_action=deep['action'],
                           root_visits=deep['root_visits'], candidates=deep['candidates'],
                           branches=branches, best_action=branches[0]['action'],
                           best_value=branches[0]['value'], actual_value=actual['value'],
                           gap=branches[0]['value'] - actual['value'])
    counts = {key: sum(r[key] for r in rows) for key in
              ('missed_immediate_win', 'missed_unique_block', 'already_lost_double_threat')}
    return dict(id=record['id'], source_results=task['source_results'], game=result,
                record_sha256=digest(result['record']), opening_value=opening_value,
                counts=counts, positions=rows, settings_signature=task['signature'],
                seconds=round(time.perf_counter() - started, 3))


def summarize(games):
    """合并逐局复盘的战术事件、深查差异和按先后手划分的统计。"""
    rows = [r for g in games for r in g['positions']]
    deep = [r for r in rows if 'deep' in r]
    def seat_summary(subset):
        """汇总一个棋色子集的开局估值和深查改善案例，保留未命中案例以免选择性统计。"""
        return dict(games=len(subset),
                    opening_value_mean=sum(g['opening_value'] for g in subset) / len(subset) if subset else None,
                    games_opening_value_below_minus_05=sum(g['opening_value'] < -.5 for g in subset),
                    games_with_deep_gap_at_least_03=sum(any(r.get('deep', {}).get('gap', 0) >= .3
                                                           for r in g['positions']) for g in subset),
                    games_with_positive_alternative_negative_actual=sum(any(
                        r.get('deep', {}).get('best_value', -1) >= .2 and
                        r.get('deep', {}).get('actual_value', 1) <= -.2
                        for r in g['positions']) for g in subset))
    return dict(games=len(games), trained_decisions=len(rows), deep_positions=len(deep),
                agreement_count=sum(r['actual'] == r['azalea_action'] for r in rows),
                exact_tactics={key: dict(positions=sum(r[key] for r in rows),
                                        games=sum(any(r[key] for r in g['positions']) for g in games))
                              for key in ('missed_immediate_win', 'missed_unique_block', 'already_lost_double_threat')},
                deep_gap_at_least_03=sum(r['deep']['gap'] >= .3 for r in deep),
                deep_gap_at_least_05=sum(r['deep']['gap'] >= .5 for r in deep),
                by_seat={name: seat_summary([g for g in games if g['game']['trained_side'] == side])
                         for side, name in ((1, 'first_blue'), (-1, 'second_red'))})


def main():
    """选择指定评测中的败局，校验来源后并行复盘，原子保存单局记录及总报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, default=ROOT.parent / 'azalea')
    parser.add_argument('--reference-checkpoint', type=Path,
                        default=ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--screen-simulations', type=int, default=800)
    parser.add_argument('--deep-simulations', type=int, default=3200)
    parser.add_argument('--deep-positions-per-game', type=int, default=2)
    parser.add_argument('--limit-games', type=int)
    args = parser.parse_args()
    if min(args.workers, args.screen_simulations, args.deep_simulations, args.deep_positions_per_game) < 1:
        parser.error('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'games').mkdir(exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ('output', 'limit_games', 'workers')}
    settings = json.loads(json.dumps(settings, default=str))
    settings.update(device='cpu', threads_per_worker=1, fresh_tree_each_search=True,
                    noise=False, temperature=0,
                    reference_sha256=digest(args.reference_checkpoint),
                    script_sha256=digest(__file__),
                    inputs_sha256={str(p.resolve()): digest(p) for p in args.results})
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    tasks, seen = [], set()
    for path in args.results:
        data = json.loads(path.read_text())
        assert data['status'] == 'complete'
        for game in data['games']:
            if game['winner'] != 'azalea' or game['trajectory_sha256'] in seen:
                continue
            seen.add(game['trajectory_sha256'])
            tasks.append(dict(game=game, source_results=str(path.resolve()), settings=settings, signature=signature))
    if args.limit_games:
        tasks = tasks[:args.limit_games]
    if not tasks:
        parser.error('No losses found')
    sizes = {json.loads(Path(t['game']['record']).read_text())['boardSize'] for t in tasks}
    if len(sizes) != 1:
        parser.error('All records must have the same board size')
    save_json(args.output / 'run.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
                                            settings=settings, signature=signature,
                                            games=len(tasks), workers=args.workers,
                                            heldout_only=True, writes_training_data=False))
    games, pending = [], []
    for task in tasks:
        dest = args.output / 'games' / (task['game']['trajectory_sha256'] + '.json')
        if dest.exists():
            previous = json.loads(dest.read_text())
            if previous['settings_signature'] != signature or previous['record_sha256'] != digest(task['game']['record']):
                raise RuntimeError('Existing output belongs to different inputs/settings; use a new output directory')
            games.append(previous)
        else:
            pending.append(task)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,
                             initargs=(args.reference_source, args.reference_checkpoint, sizes.pop())) as pool:
        futures = [pool.submit(review, t) for t in pending]
        for future in as_completed(futures):
            game = future.result()
            save_json(args.output / 'games' / (game['game']['trajectory_sha256'] + '.json'), game)
            games.append(game)
            progress = dict(completed=len(games), total=len(tasks),
                            decisions=sum(len(g['positions']) for g in games),
                            elapsed_seconds=round(time.perf_counter() - started, 1),
                            last_game=game['game']['opening'], last_counts=game['counts'])
            save_json(args.output / 'progress.json', progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    games.sort(key=lambda g: (g['source_results'], g['game']['game']))
    output = dict(status='complete', settings=settings, summary=summarize(games), games=games,
                  limitations=[
                      'Only losses are reviewed; agreement or error rates are not overall playing statistics.',
                      'Q is Azalea search value in [-1,1], not calibrated win probability or game-theoretic proof.',
                      'Screen drop compares distinct searches and only selects cases for deeper review.',
                      'Deep comparisons use identical fresh successor budgets but only a candidate subset, not all legal moves.',
                      'Opponent reference is also the reviewer; its preferred style is not an independent ground truth.',
                      'Preset opening moves are excluded from model decisions; no evaluated data enter training.'])
    save_json(args.output / 'results.json', output)
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
