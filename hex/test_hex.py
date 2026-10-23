import random
import unittest

import numpy as np

from hex.HexGame import HexGame
from hex.HexLogic import action_to_coord, coord_to_action, has_won_bfs, has_won_fast


class TestHex(unittest.TestCase):
    def test_action_coordinate_conversion(self):
        n = 5
        for action in range(n * n):
            row, col = action_to_coord(action, n)
            self.assertEqual(action, coord_to_action(row, col, n))

    def test_invalid_moves_are_masked(self):
        game = HexGame(3)
        board = game.getInitBoard()
        board[1, 2] = 1
        valids = game.getValidMoves(board, 1)
        self.assertEqual(valids[coord_to_action(1, 2, 3)], 0)
        self.assertEqual(int(valids.sum()), 8)

    def test_player_one_top_bottom_win(self):
        board = np.array([
            [1, 0, 0],
            [1, -1, 0],
            [1, 0, -1],
        ], dtype=np.int8)
        game = HexGame(3)
        self.assertEqual(game.getGameEnded(board, 1), 1)
        self.assertEqual(game.getGameEnded(board, -1), -1)

    def test_player_minus_one_left_right_win(self):
        board = np.array([
            [-1, -1, -1],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=np.int8)
        game = HexGame(3)
        self.assertEqual(game.getGameEnded(board, -1), 1)
        self.assertEqual(game.getGameEnded(board, 1), -1)

    def test_canonical_form(self):
        game = HexGame(3)
        board = np.array([[1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=np.int8)
        np.testing.assert_array_equal(game.getCanonicalForm(board, 1), board)
        np.testing.assert_array_equal(game.getCanonicalForm(board, -1), -board.T)

    def test_canonical_minus_player_unifies_direction(self):
        game = HexGame(3)
        raw = np.array([
            [1, 1, 0],
            [-1, -1, 0],
            [0, 0, 1],
        ], dtype=np.int8)
        canonical = game.getCanonicalForm(raw, -1)
        self.assertEqual(game.getGameEnded(canonical, 1), 0)

        raw[1, 2] = -1
        canonical = game.getCanonicalForm(raw, -1)
        self.assertEqual(game.getGameEnded(canonical, 1), 1)

    def test_canonical_action_and_policy_mapping(self):
        game = HexGame(3)
        raw_action = coord_to_action(0, 2, 3)
        canonical_action = coord_to_action(2, 0, 3)
        self.assertEqual(game.actionToCanonical(raw_action, -1), canonical_action)
        self.assertEqual(game.actionFromCanonical(canonical_action, -1), raw_action)

        pi = np.zeros(game.getActionSize(), dtype=np.float32)
        pi[raw_action] = 1.0
        canonical_pi = game.policyToCanonical(pi, -1)
        self.assertEqual(int(np.argmax(canonical_pi)), canonical_action)

    def test_handwritten_win_checks_match(self):
        boards = [
            np.array([[1, 0, 0], [1, -1, 0], [1, 0, -1]], dtype=np.int8),
            np.array([[-1, -1, -1], [1, 1, 0], [0, 0, 1]], dtype=np.int8),
            np.array([[1, -1, 1], [-1, 1, -1], [0, 0, 0]], dtype=np.int8),
        ]
        for board in boards:
            for player in (1, -1):
                self.assertEqual(has_won_bfs(board, player), has_won_fast(board, player))

    def test_random_legal_boards_have_matching_win_checks(self):
        rng = random.Random(7)
        for n in (3, 5, 7):
            actions = list(range(n * n))
            for _ in range(200):
                board = np.zeros((n, n), dtype=np.int8)
                player = 1
                sample = actions[:]
                rng.shuffle(sample)
                for action in sample[:rng.randrange(n * n + 1)]:
                    row, col = action_to_coord(action, n)
                    board[row, col] = player
                    player = -player
                for p in (1, -1):
                    self.assertEqual(has_won_bfs(board, p), has_won_fast(board, p))

    def test_random_self_play_finishes(self):
        rng = random.Random(11)
        game = HexGame(5)
        board = game.getInitBoard()
        player = 1
        for _ in range(game.getActionSize()):
            valids = np.flatnonzero(game.getValidMoves(board, player))
            action = int(rng.choice(valids))
            board, player = game.getNextState(board, player, action)
            if game.getGameEnded(board, player) != 0:
                return
        self.assertNotEqual(game.getGameEnded(board, player), 0)

    def test_string_representation(self):
        game = HexGame(3)
        a = game.getInitBoard()
        b = game.getInitBoard()
        c = game.getInitBoard()
        c[0, 0] = 1
        self.assertEqual(game.stringRepresentation(a), game.stringRepresentation(b))
        self.assertNotEqual(game.stringRepresentation(a), game.stringRepresentation(c))


if __name__ == '__main__':
    unittest.main()
