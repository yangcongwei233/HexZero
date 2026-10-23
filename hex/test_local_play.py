import json
from pathlib import Path
import tempfile
import unittest
from http.server import ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import torch

from hex.HexGame import HexGame
from hex.local_players import ReferenceEnvironment
from hex.local_analysis import tree_candidates
from hex.azalea_mcts import SearchNode
from hex.pytorch.NNet import NNetWrapper
from hex_play_server import HexApp, make_handler, winning_path


class LocalPlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        net = NNetWrapper(HexGame(3), {'cuda': False, 'num_res_blocks': 1, 'num_channels': 8})
        net.save_checkpoint(str(cls.folder), 'model.pth.tar', include_training=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def app(self):
        return HexApp(self.folder / 'model.pth.tar', self.folder / 'games', simulations=8)

    def session(self, mine=1, manual_opening=False):
        app = self.app()
        state = app.new_session(mine=mine, manual_opening=manual_opening)
        return app.sessions[state['id']]

    def test_model_architecture_and_size_are_restored(self):
        app = self.app()
        self.assertEqual(app.game.n, 3)
        self.assertEqual(app.net.args.num_res_blocks, 1)
        self.assertEqual(app.net.args.num_channels, 8)
        with self.assertRaises(ValueError):
            HexApp(self.folder / 'model.pth.tar', self.folder, board_size=5)

    def test_coordinates_and_revision_cannot_alias_a_valid_cell(self):
        session = self.session()
        for row, col in [(0,3), (-1,3), (3,0), (1,-1), (0,1.5), (False,0)]:
            with self.assertRaises(ValueError):
                session.human_move(row, col, session.revision)
            self.assertFalse(session.moves)
        session.human_move(0,0,0)
        with self.assertRaises(ValueError):
            session.step(0)
        self.assertEqual(len(session.moves),1)

    def test_updated_ui_replaces_old_conditional_cache(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.app()))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/app.js'
            request = Request(url, headers={'If-Modified-Since': 'Wed, 21 Oct 2037 07:28:00 GMT'})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertIn(b'newManualGame', response.read())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_human_turn_and_undo_for_both_seats(self):
        for mine in (1,-1):
            session=self.session(mine)
            if mine == -1:
                with self.assertRaises(ValueError): session.human_move(0,0,0)
                session.step(session.revision)
                self.assertFalse(session.state()['canUndo'])
            before=session.board.copy()
            row,col=np.argwhere(before==0)[0]
            session.human_move(int(row),int(col),session.revision)
            session.step(session.revision)
            revision=session.revision
            session.undo(revision)
            np.testing.assert_array_equal(before,session.board)
            self.assertEqual(session.player,mine)
            self.assertGreater(session.revision,revision)
            self.assertFalse(session.state()['canUndo'])

    def test_terminal_game_saved_and_undo_reopens(self):
        session=self.session()
        for action in (0,1,3,2,6): session.apply(action)
        self.assertEqual(session.winner,1)
        self.assertEqual(winning_path(session.board,1),[[0,0],[1,0],[2,0]])
        record=json.loads(Path(session.saved_path).read_text())
        self.assertEqual(record['winnerKind'],'human')
        self.assertEqual(len(record['moves']),5)
        with self.assertRaises(ValueError): session.step(session.revision)
        session.undo(session.revision)
        self.assertIsNone(session.winner)
        self.assertEqual(session.player,1)
        session.human_move(2,0,session.revision)
        self.assertEqual(session.winner,1)

    def test_completed_matches_in_both_seats_have_legal_moves(self):
        for mine in (1,-1):
            session=self.session(mine)
            while session.winner is None:
                if session.player==mine:
                    row,col=np.argwhere(session.board==0)[0]
                    session.human_move(int(row),int(col),session.revision)
                else: session.step(session.revision)
            self.assertLessEqual(len(session.moves),9)
            self.assertTrue(winning_path(session.board,session.winner))
            self.assertEqual(len(set(m['action'] for m in session.moves)),len(session.moves))

    def test_invalid_configuration_and_missing_reference(self):
        app=self.app()
        for args in ({'mode':'random'},{'mode':'azalea'},{'mine':0},{'mine':True},
                     {'simulations':0},{'simulations':False},{'simulations':100001}, {'seed':-1},
                     {'manual_opening':'true'}, {'manual_opening':1}):
            with self.assertRaises(ValueError): app.new_session(**args)

    def test_manual_opening_both_colors_then_handoff_for_both_seats(self):
        for mine in (1, -1):
            session = self.session(mine, manual_opening=True)
            for action in (4, 1, 7):
                with self.assertRaises(ValueError): session.step(session.revision)
                session.human_move(*divmod(action, 3), session.revision)
            self.assertEqual(session.opening_length, 3)
            self.assertTrue(all(m.get('source') == 'manual_opening' for m in session.moves))
            before = session.board.copy()
            with self.assertRaises(ValueError): session.human_move(1, 1, session.revision)
            np.testing.assert_array_equal(session.board, before)
            revision = session.revision
            session.finish_opening(revision)
            self.assertFalse(session.opening_setup)
            with self.assertRaises(ValueError): session.step(revision)
            with self.assertRaises(ValueError): session.finish_opening(session.revision)
            if mine == 1:
                with self.assertRaises(ValueError): session.human_move(0, 0, session.revision)
                session.step(session.revision)
            else:
                session.human_move(0, 0, session.revision)
            self.assertEqual(len(session.moves), 4)
            self.assertNotIn('source', session.moves[-1])

    def test_manual_undo_reopens_prefix_and_preserves_tags(self):
        session = self.session(manual_opening=True)
        for action in (4, 1, 7): session.human_move(*divmod(action, 3), session.revision)
        session.undo(session.revision)  # One ply, even though this is human mode.
        self.assertEqual([m['action'] for m in session.moves], [4, 1])
        self.assertEqual(session.opening_length, 2)
        session.finish_opening(session.revision)
        session.human_move(2, 1, session.revision)
        session.step(session.revision)
        session.undo(session.revision)  # Return to the manually placed prefix.
        self.assertTrue(session.opening_setup)
        self.assertEqual(session.opening_length, 2)
        self.assertTrue(all(m['source'] == 'manual_opening' for m in session.moves))
        session.undo(session.revision)
        session.undo(session.revision)
        self.assertFalse(session.moves)
        self.assertFalse(session.state()['canUndo'])
        self.assertTrue(session.opening_setup)
        self.assertEqual(session.opening_length, 0)
        session.finish_opening(session.revision)  # Empty setup is allowed too.
        self.assertFalse(session.opening_setup)

    def test_manual_terminal_is_saved_with_complete_prefix(self):
        session = self.session(manual_opening=True)
        for action in (0, 1, 3, 2, 6): session.human_move(*divmod(action, 3), session.revision)
        record = json.loads(Path(session.saved_path).read_text())
        self.assertEqual(record['openingLength'], 5)
        self.assertEqual(record['winner'], 1)
        self.assertTrue(all(m['source'] == 'manual_opening' for m in record['moves']))
        with self.assertRaises(ValueError): session.finish_opening(session.revision)
        session.undo(session.revision)
        self.assertIsNone(session.winner)
        self.assertTrue(session.opening_setup)
        self.assertEqual(session.opening_length, 4)

    def test_analysis_is_separate_cached_and_invalidated(self):
        session = self.session(mine=-1, manual_opening=True)
        control = self.session(mine=-1, manual_opening=True)
        for game_session in (session, control):
            for action in (4, 1): game_session.human_move(*divmod(action, 3), game_session.revision)
        engine = session.engines[1]
        root = engine.policy.mcts.tree.root
        rng = json.dumps(engine.policy.mcts.rng.bit_generator.state)
        before = session.state()
        session.analyze(session.revision)
        analysis = session.analysis
        self.assertEqual(analysis['engine'], 'trained')
        self.assertEqual(analysis['player'], 1)
        self.assertEqual(len(analysis['candidates']), 3)
        self.assertIs(root, engine.policy.mcts.tree.root)
        self.assertEqual(rng, json.dumps(engine.policy.mcts.rng.bit_generator.state))
        for key in ('board', 'moves', 'revision', 'currentPlayer', 'openingSetup'):
            self.assertEqual(before[key], session.state()[key])
        for candidate in analysis['candidates']:
            board, player = session.board.copy(), session.player
            for move in candidate['pv']:
                self.assertEqual(player, move['player'])
                self.assertEqual(board[move['row'], move['col']], 0)
                board, player = session.game.getNextState(board, player, move['action'])
        session.analyze(session.revision)
        self.assertIs(analysis, session.analysis)
        with self.assertRaises(ValueError): session.analyze(session.revision - 1)
        with self.assertRaises(ValueError): session.analyze(session.revision, 'random')
        with self.assertRaises(ValueError): session.analyze(session.revision, 'azalea')
        for game_session in (session, control):
            game_session.finish_opening(game_session.revision)
            self.assertIsNone(game_session.analysis)
            game_session.step(game_session.revision)
        self.assertEqual(session.moves[-1]['action'], control.moves[-1]['action'])
        session.analyze(session.revision)  # Human turn uses the trained helper.
        self.assertEqual(session.analysis['player'], -1)
        session.undo(session.revision)
        self.assertIsNone(session.analysis)

    def test_trained_pv_coordinates_values_ranking_and_terminal_both_colors(self):
        game = HexGame(3)
        for first_player, actions in ((1, [0, 1, 3, 2, 6]), (-1, [2, 6, 1, 7, 0])):
            board = game.getInitBoard()
            if first_player == -1: board[2, 2] = 1
            root = SearchNode(game.getCanonicalForm(board, first_player))
            node, current, player = root, board.copy(), first_player
            for i, action in enumerate(actions):
                node.terminal = 0
                node.valids = game.getValidMoves(node.board, 1)
                node.P = node.valids / node.valids.sum()
                canonical = game.actionToCanonical(action, player)
                node.Nsa[canonical] = 90
                node.Qsa[canonical] = 0.6 if i % 2 == 0 else -0.3
                current, player = game.getNextState(current, player, action)
                child = SearchNode(game.getCanonicalForm(current, player))
                node.children[canonical] = child
                node = child
            node.terminal = -1
            alternative = next(a for a in np.flatnonzero(root.valids) if a not in root.Nsa)
            root.Nsa[int(alternative)], root.Qsa[int(alternative)] = 30, 0.9
            engine = SimpleNamespace(policy=SimpleNamespace(mcts=SimpleNamespace(tree=SimpleNamespace(root=root))))
            candidates = tree_candidates(engine, board, first_player, game, 'trained')
            self.assertEqual([m['action'] for m in candidates[0]['pv']], actions)
            self.assertEqual(candidates[0]['pv'][1]['value'], 0.3)
            self.assertEqual(candidates[0]['value'], 0.6)
            self.assertEqual(candidates[1]['value'], 0.9)  # Visits, not max noisy Q, decide rank.
            self.assertEqual(candidates[0]['winner'], first_player)
            self.assertEqual(candidates[0]['stopReason'], 'terminal')
            self.assertIsNone(candidates[2]['value'])
            self.assertEqual(len(candidates[2]['pv']), 1)
            self.assertEqual(candidates[2]['stopReason'], 'unsearched')

    def test_reference_environment_matches_hex_and_snapshot(self):
        rng=np.random.default_rng(91)
        for _ in range(20):
            env=ReferenceEnvironment(3)
            game=HexGame(3)
            board=game.getInitBoard();player=1
            while env.winner is None:
                state=env.state
                np.testing.assert_array_equal(state.board,np.where(board==-1,2,board))
                self.assertEqual(state.color,0 if player==1 else 1)
                action=int(rng.choice(np.flatnonzero(board.ravel()==0)))
                env.snapshot();env.step(action+1);env.restore()
                np.testing.assert_array_equal(env.board,board)
                env.step(action+1)
                board,player=game.getNextState(board,player,action)
                result=game.getGameEnded(board,player)
                self.assertEqual(env.winner,player*result if result else None)
            self.assertEqual(env.state.result,3 if env.winner==1 else 1)
            self.assertEqual(len(env.state.legal_moves),0)

    def test_reference_second_player_board_and_moves_transform_together(self):
        board=np.array([[[1,2,0],[0,1,0],[2,0,0]]],dtype=np.int32)
        moves=np.array([[3,4,6,8,9,0]],dtype=np.int32)
        flipped,actions=ReferenceEnvironment.flip_player_board_moves(board,moves)
        expected=np.array([[[0,0,0],[0,2,1],[1,0,2]]],dtype=np.int32)
        np.testing.assert_array_equal(flipped,expected)
        np.testing.assert_array_equal(actions,[[3,8,2,4,1,0]])
        for a in actions[0][actions[0]>0]: self.assertEqual(flipped.ravel()[a-1],0)
        restored,original=ReferenceEnvironment.flip_player_board_moves(flipped,actions)
        np.testing.assert_array_equal(restored,board)
        np.testing.assert_array_equal(original,moves)


if __name__=='__main__': unittest.main()
