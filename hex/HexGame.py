from __future__ import print_function

import sys

import numpy as np

sys.path.append('..')
from Game import Game
from .HexLogic import (
    LEFT_RIGHT,
    TOP_BOTTOM,
    action_to_coord,
    coord_to_action,
    has_connected_bfs,
    has_connected_fast,
    has_won_bfs,
    has_won_fast,
)
from .fast_hex import get_valid_moves_fast


class HexGame(Game):
    """适配 alpha-zero-general 的 Hex 规则。
    空格为 0，上下连接方为 1，左右连接方为 −1；动作为行号乘边长再加列号。"""

    def __init__(self, n=5, use_fast_win_check=True, win_check_mode=None):
        """设置棋盘边长和判胜后端；将兼容名称 fast 映射到 numba，拒绝未知模式。"""
        self.n = n
        if win_check_mode is None:
            win_check_mode = "numba" if use_fast_win_check else "bfs"
        if win_check_mode == "fast":
            win_check_mode = "numba"
        if win_check_mode not in ("bfs", "numba"):
            raise ValueError("win_check_mode must be bfs, fast, or numba")
        self.win_check_mode = win_check_mode
        self.use_fast_win_check = win_check_mode != "bfs"

    def getInitBoard(self):
        """返回新的 int8 空棋盘；调用方可独立修改，不共享上一局状态。"""
        return np.zeros((self.n, self.n), dtype=np.int8)

    def getBoardSize(self):
        """返回棋盘的行列尺寸，供网络和动作转换使用。"""
        return self.n, self.n

    def getActionSize(self):
        """返回全部格点动作数；Hex 不包含停着或交换动作。"""
        return self.n * self.n

    def getNextState(self, board, player, action):
        """校验动作范围和空位，在棋盘副本上落子，返回新棋盘及对手编号；不修改输入棋盘。"""
        if action < 0 or action >= self.getActionSize():
            raise ValueError("Action out of range")
        row, col = action_to_coord(action, self.n)
        if board[row, col] != 0:
            raise ValueError("Illegal move: position is already occupied")

        next_board = np.copy(board)
        next_board[row, col] = player
        return next_board, -player

    def getValidMoves(self, board, player):
        """返回按行展平的空位掩码；双方合法位置相同，player 仅用于统一规则接口。"""
        return get_valid_moves_fast(board)

    def getGameEnded(self, board, player):
        """检查双方目标边连通性，以传入 player 的视角返回胜 1、负 −1 或未结束 0。"""
        player_axis = self._axis_for_player_to_move(board, player)
        opponent_axis = LEFT_RIGHT if player_axis == TOP_BOTTOM else TOP_BOTTOM

        if self._has_connected(board, player, player_axis):
            return 1
        if self._has_connected(board, -player, opponent_axis):
            return -1
        return 0

    def _has_connected(self, board, stone, axis):
        """按配置调用基准 BFS 或加速连通性检查，返回指定棋色在指定轴上的连通结果。"""
        if self.win_check_mode == "bfs":
            return has_connected_bfs(board, stone, axis)
        return has_connected_fast(board, stone, axis)

    def getCanonicalForm(self, board, player):
        """将当前方统一成值为 1、连接上下的棋盘；后手需同时转置和换色，返回独立数组。"""
        board = np.asarray(board, dtype=np.int8)
        if player == 1:
            return np.copy(board)
        if player == -1:
            return np.ascontiguousarray(-board.T)
        raise ValueError("player must be 1 or -1")

    def actionToCanonical(self, action, player):
        """将原棋盘动作转成当前方规范坐标；后手交换行列，先手动作保持不变。"""
        if player == 1:
            return int(action)
        if player == -1:
            row, col = action_to_coord(action, self.n)
            return coord_to_action(col, row, self.n)
        raise ValueError("player must be 1 or -1")

    def actionFromCanonical(self, action, player):
        """将规范动作还原到实际棋盘；转置自身可逆，因此复用正向变换。"""
        return self.actionToCanonical(action, player)

    def policyToCanonical(self, pi, player):
        """将整个策略向量转换到当前方视角，后手策略随棋盘一起转置，输出独立单精度数组。"""
        pi_board = np.asarray(pi).reshape(self.n, self.n)
        if player == 1:
            return np.asarray(pi, dtype=np.float32).copy()
        if player == -1:
            return np.ascontiguousarray(pi_board.T).ravel().astype(np.float32)
        raise ValueError("player must be 1 or -1")

    def policyFromCanonical(self, pi, player):
        """将规范策略还原到原棋盘坐标，保持动作概率与实际落点对应。"""
        return self.policyToCanonical(pi, player)

    def _axis_for_player_to_move(self, board, player):
        """返回棋色的固定目标方向：1 连接上下，−1 连接左右；board 保留用于接口兼容。"""
        if player == -1:
            return LEFT_RIGHT
        return TOP_BOTTOM

    def getSymmetries(self, board, pi):
        """返回原局面与旋转 180° 的棋盘/策略对，用于训练增强。单轴镜像会改变六邻接拓扑，不能使用。"""
        assert len(pi) == self.getActionSize()
        pi_board = np.reshape(pi, (self.n, self.n))
        symmetries = []

        for new_board, new_pi in (
            (board, pi_board),
            # 单轴镜像会改变 Hex 的六邻接关系，不能用于对称增强。
            (np.rot90(board, 2), np.rot90(pi_board, 2)),
        ):
            symmetries.append((np.copy(new_board), list(new_pi.ravel())))

        return symmetries

    def stringRepresentation(self, board):
        """将棋盘转换为紧凑 int8 字节串，用作搜索状态缓存键。"""
        return np.asarray(board, dtype=np.int8).tobytes()

    @staticmethod
    def display(board):
        """在终端打印带缩进的六角棋盘，以 X、O 和点表示双方及空位。"""
        n = board.shape[0]
        print("   " + " ".join(str(i) for i in range(n)))
        for row in range(n):
            print(" " * row + str(row) + " ", end="")
            for col in range(n):
                piece = board[row, col]
                if piece == 1:
                    print("X", end=" ")
                elif piece == -1:
                    print("O", end=" ")
                else:
                    print(".", end=" ")
            print("")


__all__ = [
    "HexGame",
    "action_to_coord",
    "coord_to_action",
    "has_won_bfs",
    "has_won_fast",
]
