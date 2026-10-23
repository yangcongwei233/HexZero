"""读取前三候选与已搜索主变化；Q 为根节点当前方视角，不补造未搜索分支。"""
import numpy as np


def _edges(node, board, player, game, kind):
    if node is None:
        return []
    if kind == 'trained':
        if node.P is None or node.terminal:
            return []
        if not np.array_equal(node.board, game.getCanonicalForm(board, player)):
            raise RuntimeError('训练模型分析树与棋盘不同步')
        edges = []
        for move in np.flatnonzero(node.valids):
            move = int(move)
            visits = node.Nsa.get(move, 0)
            edges.append(dict(action=game.actionFromCanonical(move, player),
                              visits=int(visits), value=float(node.Qsa[move]) if visits else None,
                              prior=float(node.P[move]), child=node.child(move)))
        return edges
    if not node.is_evaluated() or node.is_terminal():
        return []
    # Azalea 子节点编号遵循原始环境的合法动作顺序，
    # 即使后手的网络输入经过规范化，也不能改用网络坐标。
    legal = np.flatnonzero(board.ravel() == 0)
    if node.num_children != len(legal):
        raise RuntimeError('Azalea 分析树与合法落点不同步')
    stats = node.move_stats  # 此接口已将子节点价值取反为落子方视角。
    return [dict(action=int(action), visits=int(stats.num_visits[i]),
                 value=float(stats.total_value[i] / stats.num_visits[i]) if stats.num_visits[i] else None,
                 prior=float(stats.prior_prob[i]), child=node.child(i))
            for i, action in enumerate(legal)]


def _rank(edges):
    return sorted(edges, key=lambda e: (-e['visits'],
                                       -(e['value'] if e['value'] is not None else -2),
                                       -e['prior'], e['action']))


def tree_candidates(engine, board, player, game, kind, limit=3, depth=6):
    """搜索后提取候选，不推进正在对弈的引擎。"""
    root = engine.policy.mcts.tree.root if kind == 'trained' else engine.policy.tree.root
    candidates = []
    for first in _rank(_edges(root, board, player, game, kind))[:limit]:
        current_board, mover, edge = board.copy(), player, first
        pv, winner, stop = [], None, 'depth_limit'
        for _ in range(depth):
            action = edge['action']
            row, col = divmod(action, game.n)
            pv.append(dict(action=action, row=row, col=col, player=mover,
                           visits=edge['visits'],
                           value=edge['value'] * mover * player if edge['value'] is not None else None))
            current_board, mover = game.getNextState(current_board, mover, action)
            result = game.getGameEnded(current_board, mover)
            if result:
                winner, stop = int(mover * result), 'terminal'
                break
            visited = [e for e in _edges(edge['child'], current_board, mover, game, kind) if e['visits'] > 0]
            if not visited:
                stop = 'unsearched'
                break
            edge = _rank(visited)[0]
        row, col = divmod(first['action'], game.n)
        candidates.append(dict(rank=len(candidates) + 1, action=first['action'], row=row, col=col,
                               visits=first['visits'], value=first['value'], prior=first['prior'],
                               pv=pv, stopReason=stop, winner=winner))
    return candidates
