"""Clock deadlines, high search budgets, and reference memory reclamation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hex.match_clock import MatchClock
from hex.HexGame import HexGame
from hex.test_native_mcts import RecordingNet, args
from hex.azalea_mcts import AzaleaMCTS
from hex.native_mcts import NativeMCTS
from hex.azalea_cpu.search_tree import SearchTree
from hex_play_server import HexApp


class ClockTests(unittest.TestCase):
    def test_wall_time_pause_switch_and_reserve(self):
        now = [100.]
        clock = MatchClock(30, lambda: now[0])
        clock.start(1)
        now[0] += 12
        clock.start(-1)
        now[0] += 7
        clock.pause()
        now[0] += 100
        self.assertEqual(clock.state()['redMs'], 1788000)
        self.assertEqual(clock.state()['blueMs'], 1793000)
        self.assertAlmostEqual(clock.budget(1, 121), (1788-30)/61)
        self.assertEqual(clock.budget(1, 121, 2), 2)
        clock.start(1)
        now[0] += 1788
        self.assertEqual(clock.expired(), 1)

    def app(self, folder):
        app = object.__new__(HexApp)
        app.game, app.net, app.reference = HexGame(3), RecordingNet(), None
        app.default_simulations, app.sessions, app.save_dir = 20, {}, Path(folder)
        app.reference_error = 'missing reference'
        app.config = lambda: dict(models={}, search={}, inference={})
        return app

    def test_session_clock_timeout_undo_and_manual_prefix(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'HEX_SEARCH_BACKEND': 'python'}):
            app = self.app(folder)
            st = app.new_session(mine=1, clock_minutes=30, manual_opening=True)
            s = app.sessions[st['id']]
            now = [0.]
            s.clock.now = lambda: now[0]
            s.human_move(1, 1, s.revision)
            self.assertFalse(s.clock.started)
            s.finish_opening(s.revision)
            s.control_clock(True, s.revision)
            now[0] += 3
            s.step(s.revision)
            previous = s.moves[-1]['clockAfter'].copy()
            now[0] += 6
            s.human_move(2, 2, s.revision)
            s.undo(s.revision)
            self.assertEqual(s.clock.state()['blueMs'], previous['blueMs'])
            self.assertIsNone(s.clock.active)
            with self.assertRaises(ValueError):
                s.human_move(2, 2, s.revision)
            s.clock.now = lambda: now[0]
            s.control_clock(True, s.revision)
            now[0] += 1801
            result = s.state()
            self.assertEqual(result['winnerReason'], 'timeout')
            self.assertEqual(result['winner'], -1)
            self.assertTrue(Path(s.saved_path).exists())
            with self.assertRaises(ValueError):
                s.human_move(2, 2, s.revision)

    def test_timeout_during_inference_does_not_apply_returned_move(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            st = app.new_session(mine=-1, clock_minutes=1)
            s = app.sessions[st['id']]
            now = [0.]
            s.clock.now = lambda: now[0]
            def slow_move(*a, **kw):
                now[0] += 61
                return 0, {}
            s.engines[1].choose = slow_move
            s.step(s.revision)
            self.assertFalse(s.moves)
            self.assertEqual(s.winner, -1)
            self.assertEqual(s.winner_reason, 'timeout')

    def test_budget_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            for sims in (3201, 12800, 100000):
                self.assertEqual(app.new_session(simulations=sims)['mctsSims'], sims)
            for params in ({'simulations': 100001}, {'clock_minutes': True}, {'clock_minutes': -1},
                           {'move_time_ms': 99}, {'move_time_ms': 120001}):
                with self.assertRaises(ValueError):
                    app.new_session(**params)

    def test_timeout_during_tree_advance_does_not_commit_move(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.app(folder)
            st = app.new_session(mine=1, clock_minutes=1)
            s = app.sessions[st['id']]
            now = [0.]
            s.clock.now = lambda: now[0]
            def slow_observe(*a):
                now[0] += 61
            s.engines[-1].observe = slow_observe
            s.human_move(0, 0, s.revision)
            self.assertFalse(s.moves)
            self.assertEqual(s.winner, -1)
            self.assertEqual(s.winner_reason, 'timeout')

    def test_time_stop_preserves_valid_tree_and_legal_move(self):
        for cls in (AzaleaMCTS, NativeMCTS):
            mcts = cls(HexGame(3), RecordingNet(), args(numMCTSSims=100000, searchTimeLimit=.00001))
            probs = mcts.getActionProb(np.zeros((3, 3), np.int8), temp=0)
            self.assertAlmostEqual(sum(probs), 1)
            self.assertEqual(mcts.last_stop_reason, 'time_budget')
            self.assertLess(mcts.last_simulations, 100000)
            mcts.args.searchTimeLimit, mcts.args.numMCTSSims = 0, 64
            mcts.getActionProb(np.zeros((3, 3), np.int8))
            self.assertEqual(mcts.last_simulations, 64)

    def test_python_memory_limit_returns_move_and_no_virtual_loss(self):
        mcts = AzaleaMCTS(HexGame(5), RecordingNet(), args(numMCTSSims=100000, maxSearchNodes=48))
        probs = mcts.getActionProb(np.zeros((5, 5), np.int8))
        self.assertEqual(mcts.last_stop_reason, 'node_limit')
        self.assertAlmostEqual(sum(probs), 1)
        stack = [mcts.tree.root]
        total = 0
        while stack:
            node = stack.pop()
            total += 1
            self.assertFalse(node.virtual_visits)
            stack.extend(node.children.values())
        self.assertLessEqual(total, 48)

    def test_reference_compaction_preserves_paths_and_stats(self):
        t = SearchTree()
        t.root.create_child_nodes(3, [.2, .3, .5])
        for i in range(3):
            t.root.child(i).create_child_nodes(2, [.4, .6])
            t.root.child(i).total_value = i + .5
            t.root.child(i).num_visits = i + 10
        target = t.root.child(1)
        target.child(0).create_child_nodes(0, [])
        self.assertEqual(t.first_child[target.child(0).id], t.num_nodes)
        before = (target.total_value, target.num_visits, target.move_stats.prior_prob.copy())
        t.root_id = target.id
        t.compact()
        self.assertEqual(t.num_nodes, 3)
        self.assertEqual(t.root.total_value, before[0])
        self.assertEqual(t.root.num_visits, before[1])
        np.testing.assert_array_equal(t.root.move_stats.prior_prob, before[2])
        self.assertEqual(t.root.child(0).parent.id, 0)
        self.assertTrue(t.root.child(0).is_terminal())
        self.assertEqual(t.first_child[t.root.child(0).id], -1)


if __name__ == '__main__':
    unittest.main()
