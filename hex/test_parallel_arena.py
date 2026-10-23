"""Regression tests for seat accounting, spawn workers and inference handoff cleanup."""
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hex.HexGame import HexGame
from hex.parallel_arena import collect_results, play_arena_game, policy_args, run_parallel_arena
from hex.pytorch.NNet import NNetWrapper
from hex.test_regressions import search_args


class TestParallelArena(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.game = HexGame(3, win_check_mode='bfs')
        config = dict(cuda=False, num_channels=4, num_res_blocks=1,
                      policy_channels=2, value_channels=2, value_hidden=8,
                      disable_value_bn=True, epochs=1, batch_size=2)
        self.previous = NNetWrapper(self.game, config)
        self.candidate = NNetWrapper(self.game, {**config, 'value_channels': 4, 'num_res_blocks': 2})
        self.args = search_args(arenaCompare=6, arenaWorkers=2, arenaDevice='cpu',
                                arenaThreads=1, numMCTSSims=8, winCheckMode='bfs',
                                rootDirichletNoise=True, iteration=7)

    def test_seats_are_balanced_and_wins_follow_models(self):
        with patch('hex.parallel_arena.Arena.playGame', return_value=1):
            results = [play_arena_game(self.game, self.previous, self.candidate, self.args, i)
                       for i in range(6)]
        self.assertEqual(results, [(0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1)])
        self.assertEqual(collect_results(reversed(results), 6), (3, 3, 0))

    def test_seeds_are_independent_of_workers_and_noise_is_disabled(self):
        seeds = {policy_args(self.args, game_id, model_id).seed
                 for game_id in range(6) for model_id in range(2)}
        self.assertEqual(len(seeds), 12)
        config = policy_args(self.args, 2, 1)
        self.assertFalse(config.rootDirichletNoise)
        self.assertTrue(self.args.rootDirichletNoise)
        self.args.arenaWorkers = 12
        self.assertEqual(config.seed, policy_args(self.args, 2, 1).seed)
        self.args.iteration += 1
        self.assertNotEqual(config.seed, policy_args(self.args, 2, 1).seed)

    def test_real_spawn_matches_serial_for_both_backends_and_architectures(self):
        real_temporary_directory = tempfile.TemporaryDirectory
        with real_temporary_directory() as parent:
            for backend, workers in [('azalea', 2), ('classic', 3)]:
                self.args.mctsBackend, self.args.arenaWorkers = backend, workers
                expected = collect_results([
                    play_arena_game(self.game, self.previous, self.candidate, self.args, i)
                    for i in range(self.args.arenaCompare)], self.args.arenaCompare)
                with patch('hex.parallel_arena.tempfile.TemporaryDirectory',
                           side_effect=lambda **kw: real_temporary_directory(dir=parent, **kw)):
                    self.assertEqual(run_parallel_arena(
                        self.game, self.previous, self.candidate, self.args), expected)
                self.assertEqual(list(Path(parent).iterdir()), [])

    def test_handoff_is_inference_only_and_removed_after_failure(self):
        folders = []
        def fail(game, folder, args, device):
            folders.append(Path(folder))
            self.assertEqual(len(list(Path(folder).iterdir())), 2)
            for name in ('previous.pth.tar', 'candidate.pth.tar'):
                state = torch.load(Path(folder) / name, map_location='cpu', weights_only=True)
                self.assertIn('network_config', state)
                self.assertNotIn('optimizer_state_dict', state)
            raise RuntimeError('simulated worker failure')
        with patch('hex.parallel_arena._run_pool', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'simulated worker failure'):
                run_parallel_arena(self.game, self.previous, self.candidate, self.args)
        self.assertFalse(folders[0].exists())

    def test_initializer_failure_propagates_instead_of_respawning(self):
        state = self.previous.get_inference_state()
        state['board_size'] = [5, 5]
        with patch.object(self.previous, 'get_inference_state', return_value=state):
            with self.assertRaises(BrokenProcessPool):
                run_parallel_arena(self.game, self.previous, self.candidate, self.args)

    def test_missing_duplicate_and_invalid_results_are_rejected(self):
        for results, error in [([(0, 1)], RuntimeError),
                               ([(0, 1), (0, -1)], ValueError),
                               ([(0, 1), (2, -1)], ValueError),
                               ([(0, 1), (1, 2)], ValueError)]:
            with self.assertRaises(error):
                collect_results(results, 2)
        self.args.arenaCompare = 3
        with self.assertRaisesRegex(ValueError, 'even'):
            run_parallel_arena(self.game, self.previous, self.candidate, self.args)

    def test_coach_dispatches_to_parallel_arena(self):
        from Coach import Coach
        with tempfile.TemporaryDirectory() as folder:
            args = search_args(arenaWorkers=2, arenaCompare=2, numIters=1, startIter=1,
                               numEps=1, checkpoint=folder, maxlenOfQueue=10,
                               numItersForTrainExamplesHistory=1, updateThreshold=.55)
            coach = Coach(self.game, self.candidate, args)
            example = (self.game.getInitBoard(), np.ones(9) / 9, 1.)
            with patch.object(coach, 'executeEpisode', return_value=[example] * 2), \
                    patch.object(coach.nnet, 'train', return_value={}), \
                    patch('hex.parallel_arena.run_parallel_arena', return_value=(0, 2, 0)) as parallel, \
                    patch('Coach.Arena.playGames') as serial:
                coach.learn()
            parallel.assert_called_once_with(self.game, coach.pnet, coach.nnet, args)
            serial.assert_not_called()
            self.assertEqual(coach.nnet.checkpoint_metadata['iteration'], 1)


if __name__ == '__main__':
    unittest.main()
