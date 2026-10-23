import numpy as np

from .HexLogic import action_to_coord
from .hex_tactics import count_bridges, HEX_NEIGHBORS, in_bounds


def _edge_distance_score(row, col, player, n):
    """按候选落点到己方目标边的最近距离评分，越靠近边界分数越高。"""
    if player == 1:
        return 1.0 / (1.0 + min(row, n - 1 - row))
    return 1.0 / (1.0 + min(col, n - 1 - col))


def _connection_span_score(board, row, col, player):
    """估计新落子对己方目标方向跨度的增量；己方尚无棋子时使用中央距离评分。"""
    stones = np.argwhere(board == player)
    n = board.shape[0]
    if stones.size == 0:
        center = (n - 1) / 2.0
        dist = abs(row - center) + abs(col - center)
        return 1.0 / (1.0 + dist)

    coords = stones[:, 0] if player == 1 else stones[:, 1]
    coord = row if player == 1 else col
    before = coords.max() - coords.min()
    after = max(coords.max(), coord) - min(coords.min(), coord)
    return float(after - before) / max(1, n - 1)


def _opponent_span_block_score(board, row, col, player):
    """按落点接近对方连接跨度中心的程度给出阻断启发分，不构成必防证明。"""
    opponent = -player
    stones = np.argwhere(board == opponent)
    if stones.size == 0:
        return 0.0
    coords = stones[:, 0] if opponent == 1 else stones[:, 1]
    coord = row if opponent == 1 else col
    center = 0.5 * (coords.min() + coords.max())
    return 1.0 / (1.0 + abs(coord - center))


def _neighbor_counts(board, row, col, player):
    """统计六个合法邻格中的己方、对方和空位数量，忽略棋盘外位置。"""
    n = board.shape[0]
    friendly = 0
    opponent = 0
    empty = 0
    for dr, dc in HEX_NEIGHBORS:
        nr, nc = row + dr, col + dc
        if not in_bounds(nr, nc, n):
            continue
        if board[nr, nc] == player:
            friendly += 1
        elif board[nr, nc] == -player:
            opponent += 1
        else:
            empty += 1
    return friendly, opponent, empty


def _bridge_score(board, row, col, player):
    """合并己方建桥和对方潜在棋桥数量，给出落点的局部连接启发分。"""
    return 1.2 * count_bridges(board, row, col, player) + 0.8 * count_bridges(board, row, col, -player)


def _boundary_shape_score(board, row, col, player):
    """结合目标边距离和己方邻子，给出边界形状奖励。"""
    n = board.shape[0]
    if player == 1:
        near_goal = row in (0, n - 1)
        near_support = row in (1, n - 2)
    else:
        near_goal = col in (0, n - 1)
        near_support = col in (1, n - 2)

    friendly, _, _ = _neighbor_counts(board, row, col, player)
    score = 0.0
    if near_goal:
        score += 1.0 + 0.4 * friendly
    if near_support:
        score += 0.5 + 0.25 * friendly
    return score


def heuristic_policy(board, player, game):
    """综合中央、邻子、跨度、边界和棋桥评分生成合法动作分布；总分为零时回退均匀合法分布。"""
    board = np.asarray(board)
    valids = game.getValidMoves(board, player).astype(np.float64)
    scores = np.zeros(game.getActionSize(), dtype=np.float64)
    n = game.n
    center = (n - 1) / 2.0

    for action in range(game.getActionSize()):
        if valids[action] == 0:
            continue

        row, col = action_to_coord(action, n)
        friendly, opponent, empty = _neighbor_counts(board, row, col, player)
        center_dist = abs(row - center) + abs(col - center)

        score = 0.05
        score += 1.2 / (1.0 + center_dist)
        score += 0.55 * friendly
        score += 0.35 * opponent
        score += 1.0 * _connection_span_score(board, row, col, player)
        score += 0.55 * _opponent_span_block_score(board, row, col, player)
        score += 0.75 * _edge_distance_score(row, col, player, n)
        score += _boundary_shape_score(board, row, col, player)
        score += _bridge_score(board, row, col, player)

        if friendly == 0 and opponent == 0 and empty >= 4 and center_dist > n / 2.0:
            score *= 0.45

        scores[action] = max(0.0, score)

    scores *= valids
    total = float(scores.sum())
    if total > 0:
        return scores / total

    valid_count = float(valids.sum())
    if valid_count > 0:
        return valids / valid_count
    return valids
