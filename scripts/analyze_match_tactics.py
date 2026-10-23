#!/usr/bin/env python3
"""检查棋谱中的一步获胜和双威胁必胜局面。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hex.HexLogic import has_won_bfs, has_won_fast


def label(action, size):
    """将动作编号转换为旧分析报告的字母列、从上往下编号的行坐标，区别于比赛 TXT 坐标。"""
    row, col = divmod(int(action), size)
    return chr(65 + col) + str(row + 1)


def immediate_wins(board, side, verify=False):
    """逐一试放合法落点寻找立即连通的动作，检查后恢复空位；可额外用独立 BFS 复核。"""
    found = []
    for action in np.flatnonzero(board.ravel() == 0):
        board.ravel()[action] = side
        won = bool(has_won_fast(board, side))
        if verify:
            assert won == bool(has_won_bfs(board, side))
        board.ravel()[action] = 0
        if won:
            found.append(int(action))
    return found


def winning_forks(board, side, opponent_wins):
    # 对手一步无法同时占据两个不同的制胜点。
    # 若对手不能立即获胜，则可证明我方下一步必胜。
    # 对手仅有一个立即获胜点时，我方必须先占据它。
    """枚举能制造两个立即获胜点的落子，并排除对手抢先获胜的情况，返回局部必胜证明。"""
    if len(opponent_wins) >= 2:
        return []
    candidates = opponent_wins or np.flatnonzero(board.ravel() == 0)
    found = []
    for action in candidates:
        board.ravel()[action] = side
        threats = immediate_wins(board, side)
        if len(threats) >= 2:
            assert not immediate_wins(board, -side)
            # 使用独立 BFS 判胜实现，验证每个证明点。
            assert threats == immediate_wins(board, side, verify=True)
            found.append(dict(action=int(action), winning_cells=threats))
        board.ravel()[action] = 0
    return found


def analyze(record_path, scan_forks):
    """重放一份棋谱，统计漏胜、漏唯一挡点和可选双威胁，返回逐手事件及汇总。"""
    game = json.loads(record_path.read_text())
    n = game['boardSize']
    board = np.zeros((n, n), dtype=np.int8)
    events = []
    for move in game['moves']:
        side, actual = move['player'], move['action']
        own = immediate_wins(board, side, verify=True)
        opponent = immediate_wins(board, -side, verify=True)
        forks = winning_forks(board, side, opponent) if scan_forks and not own else []
        if own or opponent or forks:
            events.append(dict(
                move=move['move'], actor=move['actor'], side=side,
                actual=label(actual, n), own_winning_moves=[label(a, n) for a in own],
                opponent_winning_moves=[label(a, n) for a in opponent],
                missed_immediate_win=bool(own and actual not in own),
                missed_unique_block=bool(not own and len(opponent) == 1 and actual != opponent[0]),
                already_lost_to_double_threat=bool(not own and len(opponent) >= 2),
                proven_win_in_three_plies=[dict(move=label(f['action'], n),
                    threats=[label(a, n) for a in f['winning_cells']]) for f in forks],
                played_proven_fork=any(f['action'] == actual for f in forks)))
        assert board.ravel()[actual] == 0
        board.ravel()[actual] = side
        if move is not game['moves'][-1]:
            assert not has_won_bfs(board, side)
    assert np.array_equal(board, game['board'])
    assert has_won_bfs(board, game['winner'])
    counts = {}
    for actor in ('trained', 'azalea'):
        subset = [e for e in events if e['actor'] == actor]
        counts[actor] = dict(missed_immediate_wins=sum(e['missed_immediate_win'] for e in subset),
                            missed_unique_blocks=sum(e['missed_unique_block'] for e in subset),
                            lost_double_threat_positions=sum(e['already_lost_to_double_threat'] for e in subset))
    return dict(record=str(record_path), winner=game['winnerKind'], winner_side=game['winner'],
                total_moves=len(game['moves']), counts=counts, events=events)


def main():
    """读取评测结果引用的棋谱，批量执行精确短步战术检查并保存报告，不进行网络训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan-forks', action='store_true')
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    distinct = {}
    for result in results['games']:
        distinct.setdefault(result['trajectory_sha256'], result)
    reports = []
    for result in distinct.values():
        report = analyze(Path(result['record']), args.scan_forks)
        report['example_game'] = result['game']
        report['trajectory_sha256'] = result['trajectory_sha256']
        reports.append(report)
        print(json.dumps({k: report[k] for k in ('example_game', 'winner', 'total_moves', 'counts')}), flush=True)
    output = dict(unique_games=len(reports), games=reports,
                  scope='Exact one-ply checks and optional three-ply double-threat certificates. Not a full-game solution.')
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
