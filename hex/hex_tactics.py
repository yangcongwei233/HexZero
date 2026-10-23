import numpy as np

from .HexLogic import action_to_coord, coord_to_action, has_won_fast


HEX_NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
# 方向必须按环形排列；相邻方向对应棋桥的两个连接点。
CIRCULAR_NEIGHBORS = ((-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1))
BRIDGE_PAIRS = tuple((CIRCULAR_NEIGHBORS[d], CIRCULAR_NEIGHBORS[(d + 1) % 6]) for d in range(6))


def in_bounds(row, col, n):
    """判断行列是否位于边长为 n 的棋盘内。"""
    return 0 <= row < n and 0 <= col < n


def legal_actions(board, game):
    """返回空格的动作编号，供一步战术枚举使用。"""
    return np.flatnonzero(game.getValidMoves(board, 1))


def move_wins(board, player, game, action):
    """在棋盘副本上尝试指定落子并判胜，不改变原局面。"""
    next_board, _ = game.getNextState(board, player, int(action))
    return has_won_fast(next_board, player)


def find_immediate_win(board, player, game):
    """依次检查合法动作，返回首个本方立即连通获胜点；不存在时返回空值。"""
    for action in legal_actions(board, game):
        if move_wins(board, player, game, action):
            return int(action)
    return None


def find_immediate_block(board, player, game):
    """返回对方首个一步获胜点，作为阻挡候选；多个威胁时单点阻挡不保证安全。"""
    for action in legal_actions(board, game):
        if move_wins(board, -player, game, action):
            return int(action)
    return None


def _bridge_partner_candidates(board, row, col, player, game):
    """围绕对方刚落的棋子寻找被侵入棋桥的另一连接点，返回尚为空位的应手。"""
    candidates = []
    for d in range(6):
        a = CIRCULAR_NEIGHBORS[d]
        b = CIRCULAR_NEIGHBORS[(d + 2) % 6]
        gap = CIRCULAR_NEIGHBORS[(d + 1) % 6]
        ar, ac = row + a[0], col + a[1]
        br, bc = row + b[0], col + b[1]
        gr, gc = row + gap[0], col + gap[1]
        if all(in_bounds(r, c, game.n) for r, c in ((ar, ac), (br, bc), (gr, gc))):
            if board[ar, ac] == board[br, bc] == player and board[gr, gc] == 0:
                candidates.append(coord_to_action(gr, gc, game.n))
    return candidates


def find_bridge_reply(board, player, game, last_move=None):
    """确认上一手为对方落子后，寻找维持己方棋桥的直接应手；缺少有效上一手则不处理。"""
    if last_move is None:
        return None

    if isinstance(last_move, (int, np.integer)):
        row, col = action_to_coord(last_move, game.n)
    else:
        row, col = last_move

    if not in_bounds(row, col, game.n) or board[row, col] != -player:
        return None

    for action in _bridge_partner_candidates(board, row, col, player, game):
        r, c = action_to_coord(action, game.n)
        if board[r, c] == 0:
            return int(action)
    return None


def _axis_edge_bonus(row, col, player, n):
    """按本方连接方向奖励目标边及邻近边的落点。"""
    if player == 1:
        return int(row in (0, n - 1)) * 2 + int(row in (1, n - 2))
    return int(col in (0, n - 1)) * 2 + int(col in (1, n - 2))


def _friendly_neighbors(board, row, col, player):
    """统计给定位置六邻格中的己方棋子数。"""
    n = board.shape[0]
    count = 0
    for dr, dc in HEX_NEIGHBORS:
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc, n) and board[nr, nc] == player:
            count += 1
    return count


def count_bridges(board, row, col, player):
    """枚举假设落子后形成的棋桥，要求两个共同邻点均为空。"""
    n = board.shape[0]
    count = 0
    for a, b in BRIDGE_PAIRS:
        ar, ac = row + a[0], col + a[1]
        br, bc = row + b[0], col + b[1]
        er, ec = row + a[0] + b[0], col + a[1] + b[1]
        if all(in_bounds(r, c, n) for r, c in ((ar, ac), (br, bc), (er, ec))):
            count += board[er, ec] == player and board[ar, ac] == board[br, bc] == 0
    return int(count)


def _forms_bridge(board, row, col, player):
    """判断假设落子能否形成至少一个两连接点均为空的棋桥。"""
    return count_bridges(board, row, col, player) > 0


def find_boundary_pattern_move(board, player, game):
    """按边界、邻子和棋桥启发分选出超过阈值的落点；此函数不提供必胜保证。"""
    best_action = None
    best_score = 0
    for action in legal_actions(board, game):
        row, col = action_to_coord(action, game.n)
        score = _axis_edge_bonus(row, col, player, game.n)
        score += 2 * _friendly_neighbors(board, row, col, player)
        if _forms_bridge(board, row, col, player):
            score += 3
        if score > best_score:
            best_score = score
            best_action = int(action)
    if best_score >= 3:
        return best_action
    return None


def tactical_action(board, player, game, last_move=None):
    """按立即获胜、阻挡对手立即获胜、棋桥应手的顺序选择；均未命中时返回空值继续常规搜索。"""
    for finder in (
        lambda: find_immediate_win(board, player, game),
        lambda: find_immediate_block(board, player, game),
        lambda: find_bridge_reply(board, player, game, last_move=last_move),
    ):
        action = finder()
        if action is not None:
            return int(action)
    return None
