"""Regression checks for portable inference, reference semantics and live threads."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from hex.HexGame import HexGame
from hex.local_players import AzaleaReference, ReferenceEnvironment
from hex.onnx_inference import OnnxNetwork, OnnxReference
from hex_play_server import HexApp

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models/local_play/onnx'


class OnnxPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def app(self):
        return HexApp(MODELS / 'iteration_800/trained.onnx', self.temp.name,
                      backend='onnx', threads=4, simulations=40,
                      reference_checkpoint=MODELS / 'azalea/reference.onnx')

    def test_live_thread_change_preserves_games_and_engine_references(self):
        app = self.app()
        state = app.new_session(mode='azalea', manual_opening=True)
        session = app.sessions[state['id']]
        session.human_move(5, 5, session.revision)
        session.human_move(5, 6, session.revision)
        original = session.state()
        trained = session.engines[-1].policy.nnet
        reference = session.engines[1].policy.net
        saved = []
        app.on_threads_changed = saved.append
        for threads in (16, 7, 4):
            result = app.set_threads(threads)
            self.assertEqual(result['inference']['threads'], threads)
            self.assertEqual(session.state(), original)
            self.assertIs(trained, app.net)
            self.assertIs(reference, app.reference.prototype.net)
            for net in (trained, reference):
                self.assertEqual(net.session.get_session_options().intra_op_num_threads, threads)
            session.analyze(session.revision, 'trained')
            self.assertEqual(len(session.analysis['candidates']), 3)
            session.analysis = None
        self.assertEqual(saved, [16, 7, 4])

    def test_thread_validation_and_failed_save_are_atomic(self):
        app = self.app()
        previous = app.net.session, app.reference.prototype.net.session
        for invalid in (None, True, 3, 17, '8', 8.5):
            with self.assertRaises(ValueError):
                app.set_threads(invalid)
        def fail(_):
            raise OSError('Cannot persist settings')
        app.on_threads_changed = fail
        with self.assertRaises(OSError):
            app.set_threads(8)
        self.assertEqual(app.threads, 4)
        self.assertIs(app.net.session, previous[0])
        self.assertIs(app.reference.prototype.net.session, previous[1])

    def test_hash_failure_is_detected_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model.onnx'
            shutil.copy2(MODELS / 'iteration_800/trained.onnx', model)
            shutil.copy2(MODELS / 'iteration_800/trained.onnx.json', str(model)+'.json')
            with model.open('ab') as f:
                f.write(b'corruption')
            with self.assertRaisesRegex(ValueError, '校验失败'):
                OnnxNetwork(model, 'trained', 4)

    def test_original_reference_search_semantics_are_preserved(self):
        # Identical float32 outputs isolate tree/normalization behavior from
        # harmless numerical differences between neural inference engines.
        class NumpyNet:
            def eval(self):
                return self
            def run(self, batch):
                moves = batch['legal_moves']
                logits = np.where(moves > 0, -(moves % 7).astype(np.float32) / 8, -99)
                logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
                value = ((batch['board'] == 1).sum((1, 2)) % 5 - 2).astype(np.float32) / 4
                return dict(moves_logprob=logp.astype(np.float32), value=value)
        class TorchNet:
            device = torch.device('cpu')
            def eval(self):
                return self
            def run(self, batch):
                output = NumpyNet().run({k: v.numpy() for k, v in batch.items()})
                return {k: torch.from_numpy(v) for k, v in output.items()}
        original = AzaleaReference(ROOT.parent / 'azalea',
                                   ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth', 11, 'cpu')
        updated = OnnxReference(MODELS / 'azalea/reference.onnx', 11, 4)
        original.prototype.net, updated.prototype.net = TorchNet(), NumpyNet()
        old, new = original.player(40, 7), updated.player(40, 7)
        game = HexGame(11)
        board, player = game.getInitBoard(), 1
        for _ in range(6):
            a, i = old.choose(board, player)
            b, j = new.choose(board, player)
            self.assertEqual(a, b)
            self.assertEqual(i['value'], j['value'])
            t0, t1 = old.policy.tree, new.policy.tree
            self.assertEqual(t0.num_nodes, t1.num_nodes)
            for field in ('num_visits', 'total_value', 'prior_prob', 'parent', 'first_child', 'num_children'):
                np.testing.assert_array_equal(getattr(t0, field)[:t0.num_nodes], getattr(t1, field)[:t1.num_nodes])
            for engine in (old, new):
                engine.observe(board, player, a)
            board, player = game.getNextState(board, player, a)

    def test_reference_network_both_colors_variable_padded_moves(self):
        original = AzaleaReference(ROOT.parent / 'azalea',
                                   ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth', 11, 'cpu')
        net = OnnxNetwork(MODELS / 'azalea/reference.onnx', 'azalea', 4)
        env = ReferenceEnvironment(11)
        for action in (61, 62, 50):
            env.step(action)
            s = env.state
            b, moves = s.board[None], s.legal_moves[None]
            if s.color:
                b, moves = env.flip_player_board_moves(b, moves)
            moves = np.pad(moves, ((0, 0), (0, 1)))
            with torch.inference_mode():
                expected = original.prototype.net.run({'board': torch.from_numpy(b), 'legal_moves': torch.from_numpy(moves)})
            actual = net.run({'board': b, 'legal_moves': moves})
            np.testing.assert_allclose(np.exp(expected['moves_logprob'].numpy()), np.exp(actual['moves_logprob']), atol=2e-5, rtol=3e-4)
            np.testing.assert_allclose(expected['value'].numpy(), actual['value'], atol=2e-5, rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
