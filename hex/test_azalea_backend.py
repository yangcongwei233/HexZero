import json
import os
import tempfile
import unittest

import numpy as np

from hex.HexGame import HexGame
from hex.HexLogic import coord_to_action, has_won_bfs
from hex.azalea_mcts import AzaleaMCTS, AzaleaStylePolicy
from hex.fast_hex import has_won_numba
from hex.replay_buffer import ReplayBuffer
from utils import dotdict


class DummyBatchNNet:
    def __init__(self, game, win_action=None):
        self.game = game
        self.win_action = win_action

    def predict(self, board):
        policies, values = self.predict_batch([board])
        return policies[0], values[0]

    def predict_batch(self, boards):
        policies = []
        values = []
        for board in boards:
            valids = self.game.getValidMoves(board, 1).astype(np.float64)
            if self.win_action is not None and valids[self.win_action]:
                pi = np.ones(self.game.getActionSize(), dtype=np.float64) * 0.01
                pi[self.win_action] = 1.0
                pi *= valids
            else:
                pi = valids
            pi /= pi.sum()
            policies.append(pi)
            values.append(0.0)
        return np.asarray(policies), np.asarray(values, dtype=np.float32)


class TestAzaleaBackend(unittest.TestCase):
    def args(self, **overrides):
        data = {
            "numMCTSSims": 24,
            "mctsBatchSize": 4,
            "cpuct": 1.0,
            "heuristicPriorWeight": 0.0,
            "reuseSearchTree": False,
            "rootDirichletNoise": False,
        }
        data.update(overrides)
        return dotdict(data)

    def test_action_probs_are_legal_and_sized_for_batch_sizes(self):
        game = HexGame(3)
        board = game.getInitBoard()
        board[1, 1] = 1
        for batch_size in (1, 8, 16):
            mcts = AzaleaMCTS(game, DummyBatchNNet(game), self.args(mctsBatchSize=batch_size))
            probs = np.asarray(mcts.getActionProb(board, temp=1))
            self.assertEqual(len(probs), game.getActionSize())
            self.assertAlmostEqual(float(probs.sum()), 1.0)
            self.assertEqual(probs[coord_to_action(1, 1, 3)], 0.0)
            self.assertTrue(np.all(probs[game.getValidMoves(board, 1) == 0] == 0))

    def test_heuristic_prior_changes_azalea_policy(self):
        game = HexGame(5)
        board = game.getInitBoard()
        plain = AzaleaMCTS(game, DummyBatchNNet(game), self.args(heuristicPriorWeight=0.0))
        heuristic = AzaleaMCTS(game, DummyBatchNNet(game), self.args(heuristicPriorWeight=0.7))
        p_plain = np.asarray(plain.getActionProb(board, temp=1))
        p_heuristic = np.asarray(heuristic.getActionProb(board, temp=1))
        self.assertGreater(np.abs(p_plain - p_heuristic).sum(), 0.0)

    def test_tree_reuse_updates_root(self):
        game = HexGame(3)
        args = self.args(reuseSearchTree=True)
        mcts = AzaleaMCTS(game, DummyBatchNNet(game), args)
        board = game.getInitBoard()
        probs = mcts.getActionProb(board, temp=0)
        action = int(np.argmax(probs))
        self.assertTrue(mcts.update_with_move(action))
        next_board, next_player = game.getNextState(board, 1, action)
        canonical = game.getCanonicalForm(next_board, next_player)
        np.testing.assert_array_equal(mcts.tree.root.board, canonical)

    def test_immediate_win_gets_positive_root_value_and_top_visit(self):
        game = HexGame(3)
        board = np.array([
            [1, -1, 0],
            [1, -1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        win_action = coord_to_action(2, 0, 3)
        mcts = AzaleaMCTS(game, DummyBatchNNet(game, win_action=win_action), self.args(numMCTSSims=40))
        probs = np.asarray(mcts.getActionProb(board, temp=1))
        self.assertEqual(int(np.argmax(probs)), win_action)
        self.assertGreater(mcts.last_value, 0.0)

    def test_canonical_minus_player_immediate_win_direction(self):
        game = HexGame(3)
        raw = np.array([
            [-1, -1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=np.int8)
        canonical = game.getCanonicalForm(raw, -1)
        win_action = coord_to_action(2, 0, 3)
        mcts = AzaleaMCTS(game, DummyBatchNNet(game, win_action=win_action), self.args(numMCTSSims=40))
        probs = np.asarray(mcts.getActionProb(canonical, temp=1))
        self.assertEqual(int(np.argmax(probs)), win_action)
        self.assertGreater(mcts.last_value, 0.0)

    def test_5x5_player_one_immediate_win_top_visit_and_positive_value(self):
        game = HexGame(5)
        board = np.array([
            [1, -1, -1, 0, 0],
            [1, 0, 0, -1, 0],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int8)
        win_action = coord_to_action(4, 0, 5)
        mcts = AzaleaMCTS(game, DummyBatchNNet(game, win_action=win_action), self.args(numMCTSSims=80))
        mcts.getActionProb(board, temp=1)
        visits = mcts.tree.root.Nsa
        self.assertEqual(max(visits, key=visits.get), win_action)
        self.assertGreater(mcts.last_value, 0.0)

    def test_5x5_player_minus_one_canonical_immediate_win_direction(self):
        game = HexGame(5)
        raw = np.array([
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ], dtype=np.int8)
        canonical = game.getCanonicalForm(raw, -1)
        win_action = coord_to_action(4, 2, 5)
        mcts = AzaleaMCTS(game, DummyBatchNNet(game, win_action=win_action), self.args(numMCTSSims=80))
        mcts.getActionProb(canonical, temp=1)
        visits = mcts.tree.root.Nsa
        self.assertEqual(max(visits, key=visits.get), win_action)
        self.assertGreater(mcts.last_value, 0.0)

    def test_bfs_and_numba_match_random_boards(self):
        rng = np.random.default_rng(13)
        for n in (3, 5):
            for _ in range(100):
                board = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, n))
                for player in (1, -1):
                    self.assertEqual(has_won_bfs(board, player), has_won_numba(board, player))

    def test_policy_choose_action_shape(self):
        game = HexGame(3)
        policy = AzaleaStylePolicy(game, DummyBatchNNet(game), self.args())
        action, pi, value, visit_topk, search_time = policy.choose_action(game.getInitBoard(), temperature=1)
        self.assertIn(action, range(game.getActionSize()))
        self.assertEqual(len(pi), game.getActionSize())
        self.assertIsInstance(value, float)
        self.assertIsInstance(visit_topk, list)
        self.assertGreaterEqual(search_time, 0.0)

    def test_replay_buffer_save_load_and_jsonl(self):
        game = HexGame(3)
        example = (game.getInitBoard(), np.ones(game.getActionSize()) / game.getActionSize(), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "replay.pkl")
            buf = ReplayBuffer(max_size=2)
            buf.extend([example, example, example])
            self.assertEqual(len(buf), 2)
            self.assertEqual(len(buf.sample(1)), 1)
            buf.save(path)
            loaded = ReplayBuffer.load(path)
            self.assertEqual(len(loaded), 2)

            jsonl = os.path.join(tmp, "games.jsonl")
            record = {"board_size": 3, "worker_id": 0, "moves": [0], "winner": 1}
            with open(jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            with open(jsonl, "r", encoding="utf-8") as f:
                self.assertEqual(json.loads(f.readline())["winner"], 1)


if __name__ == "__main__":
    unittest.main()
