import unittest

import numpy as np

from MCTS import MCTS
from hex.HexGame import HexGame
from hex.HexLogic import coord_to_action
from hex.hex_action_selector import select_action_with_priors
from hex.hex_heuristics import heuristic_policy
from hex.hex_tactics import find_immediate_block, find_immediate_win, tactical_action
from hex.opening_book import opening_book_action
from utils import dotdict


class DummyNNet:
    def __init__(self, game):
        self.game = game

    def predict(self, board):
        pi = np.ones(self.game.getActionSize(), dtype=np.float64)
        pi *= self.game.getValidMoves(board, 1)
        pi /= pi.sum()
        return pi, np.array([0.0])


class TestHexPriors(unittest.TestCase):
    def test_immediate_win_is_found(self):
        game = HexGame(3)
        board = np.array([
            [1, -1, 0],
            [1, -1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        self.assertEqual(find_immediate_win(board, 1, game), coord_to_action(2, 0, 3))

    def test_immediate_block_is_found(self):
        game = HexGame(3)
        board = np.array([
            [-1, -1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ], dtype=np.int8)
        self.assertEqual(find_immediate_block(board, 1, game), coord_to_action(0, 2, 3))

    def test_heuristic_policy_masks_invalids_and_normalizes(self):
        game = HexGame(5)
        board = game.getInitBoard()
        board[2, 2] = 1
        policy = heuristic_policy(board, -1, game)
        self.assertEqual(len(policy), game.getActionSize())
        self.assertAlmostEqual(float(policy.sum()), 1.0)
        self.assertEqual(policy[coord_to_action(2, 2, 5)], 0.0)
        self.assertTrue(np.all(policy >= 0))

    def test_opening_book_hits_empty_board(self):
        game = HexGame(5)
        board = game.getInitBoard()
        self.assertEqual(opening_book_action(board, 1, game), coord_to_action(2, 2, 5))

    def test_zero_heuristic_prior_matches_original_mcts_policy(self):
        game = HexGame(3)
        board = game.getInitBoard()
        nnet = DummyNNet(game)
        args_plain = dotdict({"numMCTSSims": 8, "cpuct": 1.0})
        args_zero = dotdict({"numMCTSSims": 8, "cpuct": 1.0, "heuristicPriorWeight": 0.0})

        np.random.seed(5)
        plain = MCTS(game, nnet, args_plain).getActionProb(board, temp=1)
        np.random.seed(5)
        zero = MCTS(game, nnet, args_zero).getActionProb(board, temp=1)
        np.testing.assert_allclose(plain, zero)

    def test_heuristic_prior_does_not_create_illegal_actions(self):
        game = HexGame(3)
        board = game.getInitBoard()
        board[1, 1] = 1
        nnet = DummyNNet(game)
        args = dotdict({"numMCTSSims": 10, "cpuct": 1.0, "heuristicPriorWeight": 0.5})
        probs = MCTS(game, nnet, args).getActionProb(board, temp=1)
        self.assertEqual(probs[coord_to_action(1, 1, 3)], 0)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_tactical_miss_falls_back_to_mcts(self):
        game = HexGame(5)
        board = game.getInitBoard()
        self.assertIsNone(tactical_action(board, 1, game))

        nnet = DummyNNet(game)
        args = dotdict({"numMCTSSims": 4, "cpuct": 1.0, "heuristicPriorWeight": 0.0})
        action, source, _ = select_action_with_priors(
            board,
            1,
            game,
            nnet,
            args,
            use_opening_book=False,
            use_tactical_check=True,
        )
        self.assertEqual(source, "mcts")
        self.assertEqual(game.getValidMoves(board, 1)[action], 1)


if __name__ == "__main__":
    unittest.main()
