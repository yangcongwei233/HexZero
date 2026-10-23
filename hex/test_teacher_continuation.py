"""Exercise the same checkpoint-boundary handoff used on the training server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from Coach import Coach
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper
from hex.teacher_data import save_dataset
from hex.test_teacher_data import fixture
from hex.test_regressions import search_args
from scripts.continue_hex_run import resume_command, teacher_options

ROOT = Path(__file__).resolve().parents[1]


class TeacherContinuationTests(unittest.TestCase):
    def test_resume_command_preserves_acceleration_and_uses_snapshot_state(self):
        original = ['python', 'main_hex.py', '--seed', '9', '--epochs=99', '--learning-rate', '.9',
                    '--num-iters', '800', '--parallel-self-play', '--num-workers', '12',
                    '--arena-workers', '12', '--mcts-batch-size', '32', '--reuse-search-tree',
                    '--checkpoint-interval', '10', '--fresh-replay', '--load-model']
        command = resume_command(original, Path('/checkpoints'),
                                 {'iteration': 800, 'model': 'checkpoint_800.pth.tar', 'examples': 'checkpoint_800.pth.tar.examples'}, 1500)
        for forbidden in ('--seed', '--epochs=99', '--learning-rate', '--fresh-replay'):
            self.assertNotIn(forbidden, command)
        for preserved in ('--parallel-self-play', '--reuse-search-tree', '--num-workers', '--arena-workers'):
            self.assertIn(preserved, command)
        self.assertEqual(command[command.index('--num-iters') + 1], '700')

    def test_real_supervisor_waits_validates_preserves_and_resumes_with_teacher(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            data = run/'data'; data.mkdir()
            sha = save_dataset(data/'teacher.npz', fixture(), {'board_size': 3})
            manifest = data/'manifest.json'
            manifest.write_text(json.dumps({'status': 'complete', 'sha256': sha, 'board_size': 3, 'games': 1}))
            with self.assertRaises(ValueError):
                bad = data/'bad.json';bad.write_text(json.dumps({'status':'running'}))
                teacher_options(bad, ROOT, .1)
            game = HexGame(3)
            net = NNetWrapper(game, dict(cuda=False, num_channels=4, num_res_blocks=1,
                              policy_channels=2, value_channels=2, value_hidden=8,
                              disable_value_bn=True, epochs=1, batch_size=4))
            args = search_args(checkpoint=str(run/'checkpoints'), checkpointInterval=10, keepCheckpoints=3)
            coach = Coach(game, net, args)
            coach.trainExamplesHistory = [[(game.getInitBoard(), np.ones(9)/9, 1.)] * 20]
            net.checkpoint_metadata.update(iteration=5, seed=17)
            coach._save_snapshot(5)
            # Force CPU inside the test's trainer entrypoint even though the real
            # production supervisor intentionally enables its CUDA environment.
            bootstrap = "import os,runpy;os.environ['HEX_FORCE_CPU']='1';runpy.run_path('main_hex.py',run_name='__main__')"
            original = [sys.executable, '-c', bootstrap, '--board-size', '3', '--num-eps', '1',
                        '--num-mcts-sims', '4', '--arena-compare', '2', '--mcts-backend', 'azalea',
                        '--mcts-batch-size', '4', '--replay-buffer-size', '100', '--checkpoint-interval', '10']
            (run/'run.json').write_text(json.dumps({'command':original,'source_dir':str(ROOT)}))
            previous = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(2)'])
            try:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/continue_hex_run.py'),
                    '--run-dir', str(run), '--trainer-pid', str(previous.pid), '--segment-end', '5',
                    '--target-iteration', '6', '--poll-seconds', '.1', '--resume-source-dir', str(ROOT),
                    '--teacher-manifest', str(manifest), '--teacher-fraction', '.1'],
                    capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stderr + (run/'train.log').read_text())
            finally:
                previous.wait(timeout=5)
            state = json.loads((run/'continuation.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['final_iteration'], 6)
            self.assertEqual(state['teacher']['sha256'], sha)
            self.assertTrue(os.path.samefile(run/'checkpoints/checkpoint_5.pth.tar',
                                            run/'milestones/iteration_5_before_teacher/checkpoint_5.pth.tar'))
            checkpoint = torch.load(run/'checkpoints/checkpoint_6.pth.tar', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['metadata']['run_config']['teacherFraction'], .1)
            self.assertEqual(checkpoint['metadata']['run_config']['teacherDataSha256'], sha)
            self.assertIn('Teacher mixture:', (run/'train.log').read_text())

            # A second extension has no new teacher manifest. It must retain
            # the active command, rather than reverting to the initial launch.
            previous = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(2)'])
            try:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/continue_hex_run.py'),
                    '--run-dir', str(run), '--trainer-pid', str(previous.pid), '--segment-end', '6',
                    '--target-iteration', '7', '--poll-seconds', '.1'],
                    capture_output=True, text=True, timeout=40)
                self.assertEqual(result.returncode, 0, result.stderr + (run/'train.log').read_text())
            finally:
                previous.wait(timeout=5)
            state = json.loads((run/'continuation.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['final_iteration'], 7)
            command = state['command']
            self.assertEqual(command[command.index('--teacher-data') + 1], str(data/'teacher.npz'))
            self.assertEqual(command[command.index('--teacher-fraction') + 1], '0.1')
            self.assertEqual(command[command.index('--num-iters') + 1], '1')
            self.assertEqual(command[command.index('--checkpoint-file') + 1], 'checkpoint_6.pth.tar')
            self.assertEqual(command[command.index('--examples-file') + 1], 'checkpoint_6.pth.tar.examples')
            self.assertEqual(state['teacher_fraction'], .1)
            checkpoint = torch.load(run/'checkpoints/checkpoint_7.pth.tar', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['metadata']['run_config']['teacherFraction'], .1)
            self.assertEqual(checkpoint['metadata']['run_config']['teacherDataSha256'], sha)


if __name__ == '__main__':
    unittest.main()
