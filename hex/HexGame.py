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
        return np.zeros((self.n, self.n), dtype=np.int8)

    def getBoardSize(self):
        return self.n, self.n

    def getActionSize(self):
        return self.n * self.n

    def getNextState(self, board, player, action):
        if action < 0 or action >= self.getActionSize():
            raise ValueError("Action out of range")
        row, col = action_to_coord(action, self.n)
        if board[row, col] != 0:
            raise ValueError("Illegal move: position is already occupied")

        next_board = np.copy(board)
        next_board[row, col] = player
        return next_board, -player

    def getValidMoves(self, board, player):
        return get_valid_moves_fast(board)

    def getGameEnded(self, board, player):
        player_axis = self._axis_for_player_to_move(board, player)
        opponent_axis = LEFT_RIGHT if player_axis == TOP_BOTTOM else TOP_BOTTOM

        if self._has_connected(board, player, player_axis):
            return 1
        if self._has_connected(board, -player, opponent_axis):
            return -1
        return 0

    def _has_connected(self, board, stone, axis):
        if self.win_check_mode == "bfs":
            return has_connected_bfs(board, stone, axis)
        return has_connected_fast(board, stone, axis)

    def getCanonicalForm(self, board, player):
        board = np.asarray(board, dtype=np.int8)
        if player == 1:
            return np.copy(board)
        if player == -1:
            return np.ascontiguousarray(-board.T)
        raise ValueError("player must be 1 or -1")

    def actionToCanonical(self, action, player):
        if player == 1:
            return int(action)
        if player == -1:
            row, col = action_to_coord(action, self.n)
            return coord_to_action(col, row, self.n)
        raise ValueError("player must be 1 or -1")

    def actionFromCanonical(self, action, player):
        return self.actionToCanonical(action, player)

    def policyToCanonical(self, pi, player):
        pi_board = np.asarray(pi).reshape(self.n, self.n)
        if player == 1:
            return np.asarray(pi, dtype=np.float32).copy()
        if player == -1:
            return np.ascontiguousarray(pi_board.T).ravel().astype(np.float32)
        raise ValueError("player must be 1 or -1")

    def policyFromCanonical(self, pi, player):
        return self.policyToCanonical(pi, player)

    def _axis_for_player_to_move(self, board, player):
        if player == -1:
            return LEFT_RIGHT
        return TOP_BOTTOM

    def getSymmetries(self, board, pi):
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
        return np.asarray(board, dtype=np.int8).tobytes()

    @staticmethod
    def display(board):
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
