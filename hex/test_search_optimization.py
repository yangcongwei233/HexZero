"""Optimizations must preserve complete search statistics, not just a move."""
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from hex.HexGame import HexGame
from hex.azalea_mcts import AzaleaMCTS, EdgeStats
from hex.azalea_cpu.search_tree import SearchTree, SearchTreeFull
from hex.test_regressions import search_args
from hex.test_azalea_backend import DummyBatchNNet
from hex.local_players import ReferenceEnvironment


class SearchOptimizationTests(unittest.TestCase):
    def test_reference_state_cache_handles_moves_restore_and_direct_board_edits(self):
        env = ReferenceEnvironment(3)
        initial = env.state
        self.assertIs(initial, env.state)
        env.snapshot()
        env.step(5)
        self.assertEqual(env.state.color, 1)
        self.assertEqual(env.state.board[1, 1], 1)
        self.assertNotIn(5, env.state.legal_moves)
        env.restore()
        np.testing.assert_array_equal(env.state.board, initial.board)
        np.testing.assert_array_equal(env.state.legal_moves, initial.legal_moves)
        env.board[0, 0] = -1
        self.assertEqual(env.state.board[0, 0], 2)
        self.assertNotIn(1, env.state.legal_moves)

    def test_native_selection_matches_python_through_reuse_and_noise(self):
        game = HexGame(5)
        for batch in [1, 8, 32]:
            args = search_args(numMCTSSims=151, mctsBatchSize=batch, rootDirichletNoise=True, seed=315)
            old = AzaleaMCTS(game, DummyBatchNNet(game), args)
            new = AzaleaMCTS(game, DummyBatchNNet(game), args)
            board, player = game.getInitBoard(), 1
            for _ in range(5):
                canonical = game.getCanonicalForm(board, player)
                with patch('hex.azalea_mcts.select_edge', None):
                    a = old.getActionProb(canonical, temp=1)
                b = new.getActionProb(canonical, temp=1)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(old.last_value, new.last_value)
                pairs = [(old.tree.root, new.tree.root)]
                while pairs:
                    left, right = pairs.pop()
                    self.assertEqual(left.N, right.N)
                    for field in ['Nsa', 'Wsa', 'Qsa', 'virtual_visits']:
                        self.assertEqual(getattr(left, field), getattr(right, field))
                    self.assertEqual(left.children.keys(), right.children.keys())
                    pairs.extend((left.children[k], right.children[k]) for k in left.children)
                action = int(np.argmax(a))
                old.update_with_move(action)
                new.update_with_move(action)
                board, player = game.getNextState(board, player, game.actionFromCanonical(action, player))
                if game.getGameEnded(board, player):
                    break

    def test_mutable_statistics_and_serialization_keep_native_view_current(self):
        stats = EdgeStats(9, np.float64)
        stats[2] = 3.5
        stats.update({4: -1})
        stats |= {8: 7}
        stats.setdefault(2, 99)
        stats.setdefault(0, 1)
        for clone in [stats, copy.deepcopy(stats), pickle.loads(pickle.dumps(stats))]:
            for action in range(9):
                self.assertEqual(clone.dense[action], clone.get(action, 0))
            clone.pop(2)
            clone.popitem()
            del clone[4]
            for action in range(9):
                self.assertEqual(clone.dense[action], clone.get(action, 0))
            clone.clear()
            self.assertFalse(clone)
            self.assertFalse(clone.dense.any())

    def test_reference_tree_grows_without_losing_statistics_or_hard_limit(self):
        with patch('hex.azalea_cpu.search_tree.INITIAL_NODES', 2), patch('hex.azalea_cpu.search_tree.MAX_NODES', 9):
            tree = SearchTree()
            tree.root.create_child_nodes(3, np.array([.2, .3, .5], np.float32))
            node = tree.root.child(0)
            node.num_visits, node.total_value = 7, .125
            before = {name: getattr(tree, name)[:tree.num_nodes].copy()
                      for name in ['parent', 'num_visits', 'total_value', 'prior_prob']}
            node.create_child_nodes(5, np.full(5, .2, np.float32))
            for name, expected in before.items():
                np.testing.assert_array_equal(getattr(tree, name)[:len(expected)], expected)
            self.assertEqual(tree.num_nodes, 9)
            self.assertLessEqual(len(tree.parent), 9)
            with self.assertRaises(SearchTreeFull):
                node.child(0).create_child_nodes(1, np.ones(1, np.float32))
            self.assertEqual(tree.num_nodes, 9)
            tree.reset()
            self.assertEqual(tree.num_nodes, 1)
            self.assertEqual(tree.root.num_visits, 0)


if __name__ == '__main__':
    unittest.main()
