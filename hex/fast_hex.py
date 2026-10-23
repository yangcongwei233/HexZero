import numpy as np

from .HexLogic import (
    LEFT_RIGHT,
    TOP_BOTTOM,
    _has_connected_numba,
    has_connected_bfs,
)


def has_won_numba(board, player):
    """使用编译后的六邻接遍历检查标准方向的胜负；不可编译时由模块提供兼容实现。"""
    if _has_connected_numba is None:
        return has_won_bfs(board, player)
    board = np.asarray(board, dtype=np.int8)
    if player == 1:
        return bool(_has_connected_numba(board, player, 0))
    if player == -1:
        return bool(_has_connected_numba(board, player, 1))
    raise ValueError("player must be 1 or -1")


def has_won_bfs(board, player):
    """以纯 Python BFS 检查指定玩家的目标边连通性，作为可移植回退。"""
    if player == 1:
        return has_connected_bfs(board, player, TOP_BOTTOM)
    if player == -1:
        return has_connected_bfs(board, player, LEFT_RIGHT)
    raise ValueError("player must be 1 or -1")


def get_valid_moves_fast(board):
    """将空位检测转换成展平的合法动作掩码，避免逐格 Python 判断。"""
    return (np.asarray(board).ravel() == 0).astype(np.int8)


def has_won_mode(board, player, mode="numba"):
    """按指定模式分派到加速或基准判胜实现，并校验模式名称。"""
    if mode == "bfs":
        return has_won_bfs(board, player)
    if mode in ("fast", "numba"):
        return has_won_numba(board, player)
    raise ValueError("win check mode must be bfs, fast, or numba")
