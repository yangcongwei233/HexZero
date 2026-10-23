"""Differential tests against the established Python/Numba search."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from hex.HexGame import HexGame
from hex.azalea_mcts import AzaleaMCTS
from hex.native_mcts import NativeMCTS, local_mcts
from utils import dotdict


def args(**overrides):
    config = dict(numMCTSSims=200, mctsBatchSize=32, cpuct=.5, seed=315,
                  reuseSearchTree=True, rootDirichletNoise=False,
                  dirichletAlpha=.3, dirichletFraction=.25, heuristicPriorWeight=0.)
    config.update(overrides)
    return dotdict(config)


class RecordingNet:
    def __init__(self):
        self.calls = []

    def predict_batch(self, boards):
        boards = np.asarray(boards)
        self.calls.append(boards.copy())
        a = boards.shape[1] ** 2
        flat = boards.reshape(len(boards), a).astype(np.float64)
        # Deterministic nonuniform predictions, depending on the entire board.
        phase = (flat * np.arange(1, a+1)).sum(1)
        priors = np.exp(np.sin(phase[:, None] * .03 + np.arange(a)[None, :] * .7))
        priors /= priors.sum(1)[:, None]
        return priors.astype(np.float32), np.tanh(phase * .01).astype(np.float32)


class NativeSearchTests(unittest.TestCase):
    def assert_trees(self, old, new):
        pairs, count = [(old.tree.root, new.tree.root)], 0
        while pairs:
            left, right = pairs.pop()
            for field in ('board', 'P', 'valids'):
                np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
            for field in ('N', 'terminal', 'Nsa', 'Wsa', 'Qsa', 'prior', 'virtual_visits'):
                self.assertEqual(getattr(left, field), getattr(right, field), field)
            self.assertEqual(list(left.Nsa), list(right.Nsa))
            self.assertEqual(left.children.keys(), right.children.keys())
            pairs.extend((left.children[a], right.child(a)) for a in left.children)
            count += 1
        self.assertEqual(count, new.tree.call('hx_size'))

    def compare(self, game, config, moves=8, board=None, player=1):
        old_net, new_net = RecordingNet(), RecordingNet()
        old, new = AzaleaMCTS(game, old_net, config), NativeMCTS(game, new_net, config)
        board = game.getInitBoard() if board is None else board
        for move in range(moves):
            canonical = game.getCanonicalForm(board, player)
            for temp in (1, 0):
                a = old.getActionProb(canonical, temp=temp)
                b = new.getActionProb(canonical, temp=temp)
                np.testing.assert_array_equal(a, b)
                self.assertEqual(old.last_value, new.last_value)
                self.assertEqual(old.last_topk, new.last_topk)
                self.assertEqual(old.rng.bit_generator.state, new.rng.bit_generator.state)
                self.assert_trees(old, new)
                self.assertEqual(len(old_net.calls), len(new_net.calls))
                for l, r in zip(old_net.calls, new_net.calls):
                    np.testing.assert_array_equal(l, r)
                old_net.calls.clear(); new_net.calls.clear()
            action = int(np.argmax(a))
            self.assertEqual(old.update_with_move(action), new.update_with_move(action))
            if old.tree.root is not None:
                self.assert_trees(old, new)
            board, player = game.getNextState(board, player, game.actionFromCanonical(action, player))
            if game.getGameEnded(board, player):
                break

    def test_complete_statistics_inference_order_noise_reuse_and_temperatures(self):
        for n in (3, 5, 11):
            for batch in (1, 8, 32):
                for noise in (False, True):
                    with self.subTest(n=n, batch=batch, noise=noise):
                        self.compare(HexGame(n), args(mctsBatchSize=batch, rootDirichletNoise=noise), moves=5)

    def test_budget_edges_and_no_reuse(self):
        for budget in (1, 2, 32, 33, 65):
            self.compare(HexGame(3), args(numMCTSSims=budget, reuseSearchTree=False), moves=3)

    def test_tactical_terminal_paths_and_duplicate_leaves(self):
        board = np.array([[1, -1, 0], [1, -1, 0], [0, 0, 0]], np.int8)
        self.compare(HexGame(3), args(numMCTSSims=400, mctsBatchSize=32), moves=5, board=board)
        # Only one legal move; one inference, but many terminal backups.
        board = np.array([[1, -1, 1], [1, -1, -1], [0, 1, -1]], np.int8)
        self.compare(HexGame(3), args(numMCTSSims=100), moves=1, board=board)

    def test_native_rules_match_independent_bfs_on_random_boards(self):
        rng = np.random.default_rng(417)
        for n in (1, 3, 5, 11):
            game = HexGame(n, win_check_mode='bfs')
            mcts = NativeMCTS(game, RecordingNet(), args(numMCTSSims=1))
            for _ in range(150):
                board = rng.integers(-1, 2, (n, n), dtype=np.int8)
                expected = game.getGameEnded(board, 1)
                if expected:
                    with self.assertRaisesRegex(ValueError, 'terminal'):
                        mcts.getActionProb(board)
                else:
                    mcts.getActionProb(board)
                self.assertEqual(mcts.tree.root.terminal, expected)

    def test_masking_heuristics_and_failed_prediction_recovery(self):
        self.compare(HexGame(3), args(heuristicPriorWeight=.7), moves=4)
        game = HexGame(3)
        net = RecordingNet()
        native = NativeMCTS(game, net, args())
        original = net.predict_batch
        with patch.object(net, 'predict_batch', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                native.getActionProb(game.getInitBoard())
        net.predict_batch = original
        native.getActionProb(game.getInitBoard())
        self.assertEqual(native.tree.root.N, 199)
        self.assertFalse(native.tree.root.virtual_visits)

    def test_invalid_outputs_and_input_do_not_crash_native_process(self):
        game = HexGame(3)
        net = RecordingNet()
        native = NativeMCTS(game, net, args())
        with self.assertRaises(ValueError):
            native.getActionProb(np.zeros((5, 5), np.int8))
        with patch.object(net, 'predict_batch', return_value=(np.ones((1, 9)), np.array([np.nan]))):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                native.getActionProb(game.getInitBoard())
        native.getActionProb(game.getInitBoard())
        with self.assertRaisesRegex(RuntimeError, 'invalid move'):
            native.update_with_move(99)

    def test_backend_fallback(self):
        game = HexGame(3)
        with patch.dict(os.environ, HEX_SEARCH_BACKEND='python'):
            self.assertIs(type(local_mcts(game, RecordingNet(), args())), AzaleaMCTS)
        with patch.dict(os.environ, HEX_SEARCH_BACKEND='auto'), patch('hex.native_mcts.library', side_effect=OSError('missing')):
            with self.assertLogs('hex.native_mcts', 'WARNING'):
                self.assertIs(type(local_mcts(game, RecordingNet(), args())), AzaleaMCTS)

    def test_failed_leaf_inference_retries_without_virtual_visit_or_backup_leaks(self):
        game = HexGame(5)
        net = RecordingNet()
        original = net.predict_batch
        def fail_second(boards):
            if len(net.calls) == 1:
                raise RuntimeError('failed leaf batch')
            return original(boards)
        native = NativeMCTS(game, net, args())
        with patch.object(net, 'predict_batch', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'failed leaf batch'):
                native.getActionProb(game.getInitBoard())
        self.assertEqual(native.tree.root.N, 0)
        self.assertFalse(native.tree.root.virtual_visits)
        result = native.getActionProb(game.getInitBoard(), temp=1)
        reference = AzaleaMCTS(game, RecordingNet(), args(numMCTSSims=201))
        np.testing.assert_array_equal(result, reference.getActionProb(game.getInitBoard(), temp=1))
        self.assert_trees(reference, native)

    def test_native_allocation_lifetime_and_stale_views(self):
        import weakref
        game = HexGame(3)
        native = NativeMCTS(game, RecordingNet(), args())
        action = int(np.argmax(native.getActionProb(game.getInitBoard(), temp=0)))
        old_view = native.tree.root
        native.update_with_move(action)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            _ = old_view.board
        reference = weakref.ref(native.tree)
        finalizer = native.tree._finalizer
        del native
        self.assertIsNone(reference())
        self.assertFalse(finalizer.alive)

    def test_request_threads_and_errors_stay_with_each_tree(self):
        from concurrent.futures import ThreadPoolExecutor
        game = HexGame(3)
        engines = [NativeMCTS(game, RecordingNet(), args(numMCTSSims=65)) for _ in range(2)]
        # A tree survives HTTP thread teardown and is used in another request.
        def request(fn, *args):
            with ThreadPoolExecutor(1) as worker:
                return worker.submit(fn, *args).result()
        for _ in range(8):
            for engine in engines:
                request(engine.getActionProb, game.getInitBoard(), 0)
        with self.assertRaisesRegex(RuntimeError, 'invalid move'):
            request(engines[0].update_with_move, 99)
        request(engines[1].tree.call, 'hx_size')
        self.assertIn(b'invalid move', engines[0].tree.lib.hx_error(engines[0].tree.handle))
        self.assertEqual(b'', engines[1].tree.lib.hx_error(engines[1].tree.handle))


if __name__ == '__main__':
    unittest.main()
