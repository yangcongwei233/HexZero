"""Regressions for Hex geometry, batched search, evaluation and training state."""
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Arena import Arena
from MCTS import MCTS
from hex.HexGame import HexGame
from hex.azalea_mcts import AzaleaMCTS, AzaleaStylePolicy
from hex.hex_tactics import CIRCULAR_NEIGHBORS, count_bridges, find_bridge_reply, tactical_action
from hex.parallel_self_play import _execute_episode, episode_seed
from hex.policies import ClassicPolicy
from hex.test_azalea_backend import DummyBatchNNet
from utils import dotdict


class CountingNet(DummyBatchNNet):
    def __init__(self, game):
        super().__init__(game)
        self.batch_sizes = []

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        return super().predict_batch(boards)


def search_args(**overrides):
    config = dict(numMCTSSims=80, mctsBatchSize=8, cpuct=1., reuseSearchTree=True,
                  rootDirichletNoise=False, dirichletAlpha=.3, dirichletFraction=.25,
                  heuristicPriorWeight=0., mctsBackend='azalea', seed=7,
                  useOpeningBook=False, useTacticalCheck=False, tempThreshold=3)
    config.update(overrides)
    return dotdict(config)


class TestSearchRegressions(unittest.TestCase):
    def setUp(self):
        self.game = HexGame(3, win_check_mode='bfs')

    def test_augmentation_preserves_winning_action(self):
        board = np.array([[0, -1, 0], [-1, 1, 0], [1, 0, 0]], dtype=np.int8)
        policy = np.zeros(9)
        policy[2] = 1
        augmented = self.game.getSymmetries(board, policy)
        self.assertEqual(len(augmented), 2)
        for b, p in augmented:
            self.assertEqual(self.game.getGameEnded(b, 1), 0)
            after, _ = self.game.getNextState(b, 1, int(np.argmax(p)))
            self.assertEqual(self.game.getGameEnded(after, 1), 1)

    def test_symmetry_preserves_results_on_random_boards(self):
        rng = np.random.default_rng(10)
        for _ in range(100):
            board = rng.integers(-1, 2, (3, 3), dtype=np.int8)
            for b, _ in self.game.getSymmetries(board, np.ones(9) / 9):
                self.assertEqual(self.game.getGameEnded(board, 1), self.game.getGameEnded(b, 1))

    def test_batch_diversity_and_simulation_budget(self):
        for size in (1, 4, 32):
            net = CountingNet(self.game)
            mcts = AzaleaMCTS(self.game, net, search_args(numMCTSSims=400, mctsBatchSize=size))
            mcts.getActionProb(self.game.getInitBoard())
            self.assertEqual(mcts.tree.root.N, 399)
            self.assertEqual(sum(mcts.tree.root.Nsa.values()), 399)
            if size > 1:
                self.assertGreater(max(net.batch_sizes), 1)
            stack = [mcts.tree.root]
            while stack:
                node = stack.pop()
                self.assertFalse(node.virtual_visits)
                stack.extend(node.children.values())

    def test_repeated_leaf_inference_is_deduplicated_but_backups_are_not(self):
        net = CountingNet(self.game)
        mcts = AzaleaMCTS(self.game, net, search_args(numMCTSSims=1))
        mcts.getActionProb(self.game.getInitBoard())
        leaf, path = mcts._select_batch(1)[0]
        before = mcts.tree.root.N
        mcts._evaluate_and_backup([(leaf, path)] * 8)
        self.assertEqual(net.batch_sizes[-1], 1)
        self.assertEqual(mcts.tree.root.N - before, 8)

    def test_noise_drawn_once_without_mutating_network_prior(self):
        mcts = AzaleaMCTS(self.game, CountingNet(self.game), search_args(rootDirichletNoise=True))
        inner = np.random.default_rng(2)
        class Spy:
            calls = 0
            def dirichlet(self, alpha):
                self.calls += 1
                return inner.dirichlet(alpha)
            def choice(self, *a, **kw):
                return inner.choice(*a, **kw)
        spy = Spy()
        mcts.rng = spy
        mcts.getActionProb(self.game.getInitBoard())
        self.assertEqual(spy.calls, 1)
        np.testing.assert_allclose(mcts.tree.root.P, np.ones(9) / 9)
        mcts.getActionProb(self.game.getInitBoard())
        self.assertEqual(spy.calls, 2)
        mcts.getActionProb(self.game.getInitBoard(), add_root_noise=False)
        self.assertEqual(spy.calls, 2)

    def test_one_simulation_is_legal_in_both_backends(self):
        board = self.game.getInitBoard()
        board[0, 0] = 1
        for cls in (MCTS, AzaleaMCTS):
            for temp in (0, 1):
                search = cls(self.game, DummyBatchNNet(self.game), search_args(numMCTSSims=1))
                policy = search.getActionProb(board, temp=temp)
                self.assertAlmostEqual(sum(policy), 1)
                self.assertEqual(policy[0], 0)
                if temp == 0:
                    self.assertEqual(np.count_nonzero(policy), 1)

    def test_arena_policy_reuses_both_players_moves_without_noise(self):
        policy = AzaleaStylePolicy(self.game, CountingNet(self.game), search_args(rootDirichletNoise=True))
        board = self.game.getInitBoard()
        with patch.object(policy.mcts, 'getActionProb', wraps=policy.mcts.getActionProb) as search:
            move = policy.select_action(board, 1)
            self.assertFalse(search.call_args.kwargs['add_root_noise'])
        board, player = self.game.getNextState(board, 1, move)
        root = policy.mcts.tree.root
        self.assertIsNotNone(root)
        np.testing.assert_array_equal(root.board, self.game.getCanonicalForm(board, player))
        # Ensure a child is available for the opponent notification.
        policy.mcts.getActionProb(root.board, add_root_noise=False)
        opponent_action, child = next(iter(root.children.items()))
        raw_action = self.game.actionFromCanonical(opponent_action, player)
        policy.notify(board, opponent_action)
        self.assertIs(policy.mcts.tree.root, child)
        self.assertEqual(policy.last_move, raw_action)
        board, player = self.game.getNextState(board, player, raw_action)
        np.testing.assert_array_equal(child.board, self.game.getCanonicalForm(board, player))

    def test_seed_covers_search_and_episode_sampling(self):
        args = search_args(savePolicyInJsonl=True)
        net = DummyBatchNNet(self.game)
        _, first = _execute_episode(self.game, net, args, 123)
        _, again = _execute_episode(self.game, net, args, 123)
        self.assertEqual(first['moves'], again['moves'])
        self.assertEqual(first['policies'], again['policies'])
        self.assertEqual(len(first['moves']), len(first['policies']))
        self.assertNotEqual(episode_seed(0, 1, 0), episode_seed(0, 2, 0))

    def test_arena_balances_and_restores_players(self):
        first, second = object(), object()
        arena = Arena(first, second, self.game)
        with patch.object(arena, 'playGame', return_value=1):
            self.assertEqual(arena.playGames(4), (2, 2, 0))
            self.assertIs(arena.player1, first)
            self.assertEqual(arena.playGames(4), (2, 2, 0))
        with self.assertRaises(ValueError):
            arena.playGames(3)

    def test_raw_prior_hooks_work_for_both_policies(self):
        board = self.game.getInitBoard()
        board[0, 0] = 1
        for cls in (ClassicPolicy, AzaleaStylePolicy):
            policy = cls(self.game, DummyBatchNNet(self.game), search_args(useOpeningBook=True))
            with patch('hex.opening_book.opening_book_action', return_value=2) as book:
                self.assertEqual(policy.select_action(board, -1), 6)
                self.assertEqual(book.call_args.args[1], -1)


class TestBridgeRegressions(unittest.TestCase):
    def test_bridge_response_in_all_six_orientations(self):
        game = HexGame(7, win_check_mode='bfs')
        for d in range(6):
            board = game.getInitBoard()
            board[3, 3] = -1
            for offset in (CIRCULAR_NEIGHBORS[d], CIRCULAR_NEIGHBORS[(d + 2) % 6]):
                board[3 + offset[0], 3 + offset[1]] = 1
            gap = CIRCULAR_NEIGHBORS[(d + 1) % 6]
            expected = (3 + gap[0]) * 7 + 3 + gap[1]
            self.assertEqual(find_bridge_reply(board, 1, game, 24), expected)
            offset = CIRCULAR_NEIGHBORS[d]
            board[3 + offset[0], 3 + offset[1]] = 0
            self.assertIsNone(find_bridge_reply(board, 1, game, np.int64(24)))

    def test_soft_bridge_requires_two_empty_carriers(self):
        board = np.zeros((5, 5), dtype=np.int8)
        board[1, 1] = 1
        self.assertEqual(count_bridges(board, 2, 2, 1), 1)
        board[1, 2] = -1
        self.assertEqual(count_bridges(board, 2, 2, 1), 0)

    def test_boundary_heuristic_does_not_override_search(self):
        game = HexGame(5, win_check_mode='bfs')
        board = game.getInitBoard()
        board[2, 2] = 1
        self.assertIsNone(tactical_action(board, -1, game))


try:
    import torch
    from hex.pytorch.NNet import NNetWrapper
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required for training regression tests')
class TestTrainingRegressions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = HexGame(3, win_check_mode='bfs')
        self.config = dict(cuda=False, num_channels=4, num_res_blocks=1, policy_channels=2,
                           value_channels=2, value_hidden=8, disable_value_bn=True,
                           epochs=1, batch_size=2, tensorboard_interval=1)
        self.net = NNetWrapper(self.game, self.config)
        self.example = (self.game.getInitBoard(), np.ones(9) / 9, 1.)

    def coach(self, directory, **kwargs):
        from Coach import Coach
        args = search_args(startIter=5, numIters=1, numEps=1, maxlenOfQueue=100,
                           replayBufferSize=100, numItersForTrainExamplesHistory=3,
                           checkpoint=directory, arenaCompare=2, updateThreshold=.55,
                           checkpointInterval=10, keepCheckpoints=2)
        args.update(kwargs)
        return Coach(self.game, self.net, args)

    def test_state_roundtrip_and_independent_network_configs(self):
        self.net.train([self.example] * 2)
        original_optimizer = self.net.optimizer
        self.net.train([self.example] * 2)
        self.assertIs(self.net.optimizer, original_optimizer)
        with tempfile.TemporaryDirectory() as directory:
            self.net.save_checkpoint(directory, 'test.pth.tar')
            expected = self.net.predict(self.example[0])
            restored = NNetWrapper(self.game, {'cuda': False})
            restored.load_checkpoint(directory, 'test.pth.tar', restore_rng=True)
            actual = restored.predict(self.example[0])
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])
            self.assertEqual(restored.args.value_channels, 2)
            self.assertTrue(restored.args.disable_value_bn)
            self.assertEqual(restored.train_step, 2)
            self.assertTrue(restored.optimizer.state)
            default = NNetWrapper(self.game, {'cuda': False})
            self.assertEqual(default.args.value_channels, 8)

    def test_checkpoint_restores_random_generators(self):
        state = self.net.get_training_state()
        expected = (random.random(), np.random.random(), torch.rand(1))
        self.net.load_training_state(state, restore_rng=True)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2])

    def test_worker_loads_nondefault_architecture(self):
        from hex.parallel_self_play import _worker
        with tempfile.TemporaryDirectory() as directory:
            self.net.save_checkpoint(directory, 'worker.pth.tar', include_training=False)
            args = search_args(checkpoint=directory, winCheckMode='bfs', selfplayDevice='cpu', numMCTSSims=4)
            task = (3, directory, 'worker.pth.tar', dict(args), 1, 123, 0)
            result = _worker(task)
            self.assertGreater(len(result['examples']), 0)

    def test_resume_skip_is_relative_to_start_and_replay_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            coach = self.coach(directory)
            coach.trainExamplesHistory = [[self.example] * 2]
            coach.skipFirstSelfPlay = True
            with patch.object(coach, 'executeEpisode') as episode, patch.object(coach.nnet, 'train', return_value={}) as train, patch('Coach.Arena.playGames', return_value=(1, 1, 0)):
                coach.learn()
                episode.assert_not_called()
                self.assertEqual(len(train.call_args.args[0]), 2)
            self.assertEqual(len(coach.replayBuffer), 2)
            with patch.object(coach, 'executeEpisode', return_value=[self.example]), patch.object(coach.nnet, 'train', return_value={}) as train, patch('Coach.Arena.playGames', return_value=(1, 1, 0)):
                coach.args.startIter = 6
                coach.learn()
                self.assertEqual(len(train.call_args.args[0]), 3)

    def test_snapshot_retention_and_matching_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            coach = self.coach(directory)
            coach.trainExamplesHistory = [[self.example]]
            unrelated = Path(directory) / 'checkpoint_999.pth.tar'
            unrelated.write_bytes(b'old experiment')
            for iteration in (10, 20, 30):
                coach._save_snapshot(iteration)
            manifest = json.loads((Path(directory) / 'latest.json').read_text())
            self.assertEqual(manifest['iteration'], 30)
            self.assertEqual(len(manifest['managed_snapshots']), 2)
            self.assertFalse((Path(directory) / 'checkpoint_10.pth.tar.examples').exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(os.stat(Path(directory) / 'latest.pth.tar').st_ino,
                             os.stat(Path(directory) / manifest['model']).st_ino)
            restored = self.coach(directory)
            restored.loadTrainExamples(Path(directory) / manifest['examples'])
            self.assertEqual(len(restored.replayBuffer), 1)

    def test_legacy_replay_is_not_silently_reused(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.examples'
            with path.open('wb') as f:
                pickle.dump([[self.example]], f)
            with self.assertRaisesRegex(ValueError, 'fresh-replay'):
                self.coach(directory).loadTrainExamples(path)

    def test_tensorboard_writes_losses_and_iteration_metrics(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            coach = self.coach(directory, tensorboard=True, tensorboardDir=directory + '/tb', startIter=1)
            with patch.object(coach, 'executeEpisode', return_value=[self.example] * 2), patch('Coach.Arena.playGames', return_value=(1, 1, 0)):
                coach.learn()
            reader = EventAccumulator(directory + '/tb').Reload()
            tags = reader.Tags()['scalars']
            for tag in ('train/policy_loss', 'train/value_loss', 'selfplay/games_per_second', 'arena/new_winrate', 'replay/samples'):
                self.assertIn(tag, tags)

    def test_eval_counts_wins_by_model_after_swapping(self):
        import hex_eval
        seating = []
        class FakeArena:
            def __init__(self, first, second, game):
                seating.append((first, second))
            def playGame(self, verbose=False):
                return 1
        with patch.object(hex_eval, 'Arena', FakeArena), patch.object(hex_eval, 'make_player', side_effect=lambda spec, *a: spec.name):
            result = hex_eval.play_pair(self.game, hex_eval.ModelSpec('A'), hex_eval.ModelSpec('B'), None, 4, 0)
        self.assertEqual(seating, [('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'A')])
        self.assertEqual(result, [2, 0, 2])


if __name__ == '__main__':
    unittest.main()
