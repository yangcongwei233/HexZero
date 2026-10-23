from collections import deque

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


HEX_DIRECTIONS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
TOP_BOTTOM = "top_bottom"
LEFT_RIGHT = "left_right"


def action_to_coord(action, board_size):
    row = int(action) // board_size
    col = int(action) % board_size
    return row, col


def coord_to_action(row, col, board_size):
    return int(row) * board_size + int(col)


def has_connected_bfs(board, stone, axis):
    """Simple Hex connectivity check used as the reference implementation."""
    board = np.asarray(board)
    n = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    queue = deque()

    if axis == TOP_BOTTOM:
        for col in range(n):
            if board[0, col] == stone:
                queue.append((0, col))
                seen[0, col] = True
        target = lambda row, col: row == n - 1
    elif axis == LEFT_RIGHT:
        for row in range(n):
            if board[row, 0] == stone:
                queue.append((row, 0))
                seen[row, 0] = True
        target = lambda row, col: col == n - 1
    else:
        raise ValueError("axis must be TOP_BOTTOM or LEFT_RIGHT")

    while queue:
        row, col = queue.popleft()
        if target(row, col):
            return True
        for dr, dc in HEX_DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < n and 0 <= nc < n and not seen[nr, nc] and board[nr, nc] == stone:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return False


def has_won_bfs(board, player):
    """Hex win check with the normal game axes."""
    if player == 1:
        return has_connected_bfs(board, player, TOP_BOTTOM)
    if player == -1:
        return has_connected_bfs(board, player, LEFT_RIGHT)
    raise ValueError("player must be 1 or -1")


def build_regions(board, stone, axis):
    """
    Build padded connected-component labels for one player.

    Label 1 is the player's first virtual edge, label 2 is the opposite virtual
    edge. Stones merge adjacent labels using the six Hex neighbors. This is a
    stateless variant of the minihex regions/flood-fill idea, so MCTS can copy
    and hash boards without carrying mutable region caches.
    """
    board = np.asarray(board)
    n = board.shape[0]
    regions = np.zeros((n + 2, n + 2), dtype=np.int32)

    if axis == TOP_BOTTOM:
        regions[0, :] = 1
        regions[-1, :] = 2
    elif axis == LEFT_RIGHT:
        regions[:, 0] = 1
        regions[:, -1] = 2
    else:
        raise ValueError("axis must be TOP_BOTTOM or LEFT_RIGHT")

    next_label = 3
    for row in range(n):
        for col in range(n):
            if board[row, col] != stone:
                continue

            y, x = row + 1, col + 1
            neighborhood = regions[y - 1:y + 2, x - 1:x + 2].copy()
            neighborhood[0, 0] = 0
            neighborhood[2, 2] = 0
            labels = sorted(label for label in set(neighborhood.ravel()) if label)

            if not labels:
                regions[y, x] = next_label
                next_label += 1
                continue

            keep = labels[0]
            regions[y, x] = keep
            for label in labels[1:]:
                regions[regions == label] = keep

    return regions


if njit is not None:
    @njit(cache=True)
    def _has_connected_numba(board, stone, axis_id):
        n = board.shape[0]
        seen = np.zeros((n, n), dtype=np.uint8)
        queue_rows = np.empty(n * n, dtype=np.int32)
        queue_cols = np.empty(n * n, dtype=np.int32)
        head = 0
        tail = 0

        if axis_id == 0:
            for col in range(n):
                if board[0, col] == stone:
                    queue_rows[tail] = 0
                    queue_cols[tail] = col
                    tail += 1
                    seen[0, col] = 1
        else:
            for row in range(n):
                if board[row, 0] == stone:
                    queue_rows[tail] = row
                    queue_cols[tail] = 0
                    tail += 1
                    seen[row, 0] = 1

        while head < tail:
            row = queue_rows[head]
            col = queue_cols[head]
            head += 1

            if axis_id == 0:
                if row == n - 1:
                    return True
            else:
                if col == n - 1:
                    return True

            for k in range(6):
                if k == 0:
                    nr, nc = row - 1, col
                elif k == 1:
                    nr, nc = row - 1, col + 1
                elif k == 2:
                    nr, nc = row, col - 1
                elif k == 3:
                    nr, nc = row, col + 1
                elif k == 4:
                    nr, nc = row + 1, col - 1
                else:
                    nr, nc = row + 1, col

                if 0 <= nr < n and 0 <= nc < n and seen[nr, nc] == 0 and board[nr, nc] == stone:
                    seen[nr, nc] = 1
                    queue_rows[tail] = nr
                    queue_cols[tail] = nc
                    tail += 1

        return False
else:
    _has_connected_numba = None


def has_connected_fast(board, stone, axis):
    if _has_connected_numba is not None:
        axis_id = 0 if axis == TOP_BOTTOM else 1
        return bool(_has_connected_numba(np.asarray(board), stone, axis_id))
    return has_connected_bfs(board, stone, axis)


def has_won_fast(board, player):
    if player == 1:
        return has_connected_fast(board, player, TOP_BOTTOM)
    if player == -1:
        return has_connected_fast(board, player, LEFT_RIGHT)
    raise ValueError("player must be 1 or -1")


class Board:
    def __init__(self, n):
        self.n = n
        self.pieces = np.zeros((n, n), dtype=np.int8)

    def execute_move(self, move, player):
        row, col = move
        if self.pieces[row, col] != 0:
            raise ValueError("Illegal move: position is already occupied")
        self.pieces[row, col] = player

    def get_legal_moves(self):
        return np.argwhere(self.pieces.ravel() == 0).ravel()
