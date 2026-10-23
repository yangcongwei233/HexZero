import numpy as np

from .HexLogic import action_to_coord
from .hex_tactics import count_bridges, HEX_NEIGHBORS, in_bounds


def _edge_distance_score(row, col, player, n):
    if player == 1:
        return 1.0 / (1.0 + min(row, n - 1 - row))
    return 1.0 / (1.0 + min(col, n - 1 - col))


def _connection_span_score(board, row, col, player):
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
    opponent = -player
    stones = np.argwhere(board == opponent)
    if stones.size == 0:
        return 0.0
    coords = stones[:, 0] if opponent == 1 else stones[:, 1]
    coord = row if opponent == 1 else col
    center = 0.5 * (coords.min() + coords.max())
    return 1.0 / (1.0 + abs(coord - center))


def _neighbor_counts(board, row, col, player):
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
    return 1.2 * count_bridges(board, row, col, player) + 0.8 * count_bridges(board, row, col, -player)


def _boundary_shape_score(board, row, col, player):
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
