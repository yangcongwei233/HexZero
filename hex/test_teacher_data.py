"""Teacher coordinate, validation, sampling, mixing and resume regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Coach import Coach
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper
from hex.teacher_data import TeacherDataset, canonical_target, save_dataset
from hex.test_regressions import search_args
from scripts.collect_azalea_teacher import opening_key


def fixture():
    game = HexGame(3)
    board, player = game.getInitBoard(), 1
    arrays = {key: [] for key in ('boards', 'policies', 'players', 'values', 'plies', 'game_ids')}
    for ply, action in enumerate((0, 1, 3, 2, 6), 1):
        legal = np.flatnonzero(board.ravel() == 0) + 1
        canonical, pi = canonical_target(game, board, player, legal, np.arange(1, len(legal) + 1))
        for key, value in dict(boards=canonical, policies=pi, players=player, values=player,
                               plies=ply, game_ids=0).items():
            arrays[key].append(value)
        board, player = game.getNextState(board, player, action)
    return {key: np.asarray(value) for key, value in arrays.items()}


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'teacher.npz'
        self.arrays = fixture()
        self.sha = save_dataset(self.path, self.arrays, {'board_size': 3})
        self.teacher = TeacherDataset(self.path, 3, self.sha)
        self.game = HexGame(3)

    def tearDown(self):
        self.temp.cleanup()

    def net(self):
        return NNetWrapper(self.game, dict(cuda=False, num_channels=4, num_res_blocks=1,
                           policy_channels=2, value_channels=2, value_hidden=8,
                           disable_value_bn=True, epochs=1, batch_size=4, tensorboard_interval=1))

    def test_red_uses_raw_visit_order_then_transposes_policy_and_board(self):
        board = np.zeros((3, 3), dtype=np.int8); board[0, 0] = 1
        canonical, pi = canonical_target(self.game, board, -1, np.arange(2, 10), np.arange(1, 9))
        np.testing.assert_array_equal(canonical, -board.T)
        self.assertAlmostEqual(pi[3], 1/36)  # Raw B1 -> canonical A2.
        self.assertAlmostEqual(pi[1], 3/36)  # Raw A2 -> canonical B1.
        self.assertEqual(pi[0], 0)
        with self.assertRaises(ValueError): canonical_target(self.game, board, -1, np.arange(2, 10)[::-1], np.ones(8))
        with self.assertRaises(ValueError): canonical_target(self.game, board, -1, np.arange(2, 10), np.zeros(8))

    def test_validation_rejects_illegal_targets_format_and_hash(self):
        for key, bad in [('values', np.zeros(5)), ('players', -self.arrays['players']),
                         ('plies', self.arrays['plies'] + 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                save_dataset(self.folder/'bad.npz', {**self.arrays, key: bad}, {'board_size': 3})
        bad = self.arrays['policies'].copy(); bad[1] = 0; bad[1, 0] = 1
        with self.assertRaises(ValueError): save_dataset(self.folder/'bad.npz', {**self.arrays, 'policies': bad}, {'board_size': 3})
        with self.assertRaises(ValueError): TeacherDataset(self.path, 5)
        with self.assertRaises(ValueError): TeacherDataset(self.path, 3, 'bad-hash')

    def test_sampling_balances_colors_and_only_uses_valid_symmetry(self):
        rng = np.random.RandomState(41)
        examples = self.teacher.sample(100, rng)
        # Blue canonical boards have equal stone counts; red has one fewer own stone.
        colors = [1 if (b == 1).sum() == (b == -1).sum() else -1 for b, _, _ in examples]
        self.assertEqual(colors.count(1), 50)
        originals = {(b.tobytes(), p.tobytes(), float(v)) for b, p, v in zip(
            self.teacher.boards, self.teacher.policies, self.teacher.values)}
        for board, pi, value in examples:
            self.assertTrue((board.tobytes(), pi.tobytes(), value) in originals or
                            (np.rot90(board, 2).tobytes(), pi[::-1].tobytes(), value) in originals)
            self.assertTrue(np.all(pi[board.ravel() != 0] == 0))

    def test_mixing_keeps_step_budget_and_reports_actual_ten_percent(self):
        net = self.net()
        example = (self.game.getInitBoard(), np.ones(9)/9, 1.)
        before = copy.deepcopy(net.get_training_state())
        with patch.object(self.teacher, 'sample', wraps=self.teacher.sample) as sample:
            metrics = net.train([example] * 40, teacher=self.teacher, teacher_fraction=.1)
        self.assertEqual(net.train_step, 10)
        self.assertEqual(sum(call.args[0] for call in sample.call_args_list), 4)
        self.assertEqual(metrics['teacher_fraction'], .1)
        self.assertEqual(metrics['teacher_samples'], 4)
        for source in ('teacher', 'selfplay'):
            self.assertTrue(np.isfinite(metrics[source + '_policy_loss']))
            self.assertTrue(np.isfinite(metrics[source + '_value_loss']))
        net.load_training_state(before)
        net.train([example] * 40)
        self.assertEqual(net.train_step, 10)

    def test_coach_keeps_teacher_out_of_fifo_and_saves_resume_provenance(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        net = self.net()
        args = search_args(startIter=5, numIters=1, numEps=1, maxlenOfQueue=40,
                           replayBufferSize=40, numItersForTrainExamplesHistory=3,
                           checkpoint=str(self.folder/'checkpoint'), arenaCompare=2, updateThreshold=.55,
                           checkpointInterval=10, keepCheckpoints=2, teacherData=str(self.path),
                           teacherFraction=.1, tensorboard=True, tensorboardDir=str(self.folder/'tb'))
        coach = Coach(self.game, net, args)
        example = (self.game.getInitBoard(), np.ones(9)/9, 1.)
        optimizer = net.optimizer
        with patch.object(coach, 'executeEpisode', return_value=[example]*40), patch('Coach.Arena.playGames', return_value=(0, 2, 0)):
            coach.learn()
        self.assertIs(net.optimizer, optimizer)
        self.assertEqual(len(coach.replayBuffer), 40)
        self.assertTrue(all(np.count_nonzero(e[0]) == 0 for e in coach.replayBuffer.examples))
        self.assertEqual(net.checkpoint_metadata['run_config']['teacherDataSha256'], self.sha)
        self.assertEqual(net.checkpoint_metadata['run_config']['teacherFraction'], .1)
        restored = self.net()
        restored.load_checkpoint(args.checkpoint, 'checkpoint_5.pth.tar', restore_rng=True)
        self.assertEqual(restored.train_step, 10)
        self.assertTrue(restored.optimizer.state)
        restored_coach = Coach(self.game, restored, args)
        restored_coach.loadTrainExamples(Path(args.checkpoint)/'checkpoint_5.pth.tar.examples')
        self.assertEqual(len(restored_coach.replayBuffer), 40)
        self.assertEqual(restored_coach.teacher.sha256, self.sha)
        tags = EventAccumulator(str(self.folder/'tb')).Reload().Tags()['scalars']
        for tag in ('iteration/teacher_fraction', 'iteration/teacher_policy_loss',
                    'iteration/selfplay_value_loss', 'teacher/dataset_samples'):
            self.assertIn(tag, tags)

    def test_holdout_rotation_and_argument_resume_overrides(self):
        self.assertEqual(opening_key([1, 7], 3), opening_key([7, 1], 3))
        self.assertNotEqual(opening_key([1, 2], 3), opening_key([2, 1], 3))
        import main_hex
        with patch('sys.argv', ['main_hex.py', '--teacher-data', str(self.path), '--teacher-fraction', '.1']):
            cli = main_hex.parse_args()
            args = main_hex.build_args(cli)
        self.assertEqual(args.teacherFraction, .1)
        self.assertEqual(args.teacherData, str(self.path))
        self.assertIn('--teacher-data', main_hex.RESUME_OPTIONS['teacherDataSha256'])
        bad_args = search_args(teacherFraction=.1, checkpoint=str(self.folder))
        with self.assertRaisesRegex(ValueError, 'requires'): Coach(self.game, self.net(), bad_args)


if __name__ == '__main__':
    unittest.main()
