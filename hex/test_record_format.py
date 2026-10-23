import base64
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper
from hex.record_format import parse_txt, parse_json_record, export_txt, mark_from_value
from hex_play_server import HexApp

SAMPLE = '{[HEX][先手参赛队 R][后手参赛队 B][先手胜][2017.07.29 14:00 重庆][2017 CCGC];R(E,7)MARK[1];B(E,6);R(F,7);B(G,7);R(D,6);B(F,6);R(C,6);B(G,6)}'


class RecordFormatTests(unittest.TestCase):
    def test_official_example_exact_gb2312_roundtrip(self):
        record = parse_txt(SAMPLE.encode('gb2312'))
        self.assertEqual(record['moves'][0]['action'], 4*11+4)
        self.assertEqual(record['moves'][0]['player'], 1)
        self.assertEqual(record['moves'][1]['player'], -1)
        self.assertEqual(record['moves'][0]['mark'], 1)
        self.assertTrue(record['warnings'])  # A declared winner does not prove connectivity.
        name, data = export_txt(record['moves'], record['metadata'])
        self.assertEqual(data, SAMPLE.encode('gb2312'))
        self.assertTrue(name.startswith('HEX-先手参赛队 R vs 后手参赛队 B-先手胜-'))
        self.assertNotIn(':', name)

    def test_lower_left_coordinates_and_terminal_validation(self):
        header = SAMPLE.split(';')[0]
        tokens = []
        for row in range(1, 12):
            tokens.append(f'R(A,{row})')
            if row < 11:
                tokens.append(f'B(K,{row})')
        text = header+';'+';'.join(tokens)+'}'
        record = parse_txt(text.encode('gb2312'))
        self.assertEqual(record['moves'][0]['action'], 110)
        self.assertEqual(record['moves'][-1]['action'], 0)
        self.assertEqual(record['winner'], 1)
        with self.assertRaisesRegex(ValueError, '矛盾'):
            parse_txt(text.replace('先手胜', '后手胜').encode('gb2312'))
        with self.assertRaisesRegex(ValueError, '之前已连通'):
            parse_txt((text[:-1]+';B(J,11)}').encode('gb2312'))

    def test_invalid_encoding_moves_marks_and_delimiters(self):
        for text in [SAMPLE.replace('R(E,7)', 'B(E,7)', 1), SAMPLE.replace('B(E,6)', 'B(E,7)'),
                     SAMPLE.replace('MARK[1]', 'MARK[3]'), SAMPLE.replace('E,7', 'E,12'),
                     SAMPLE.replace('E,7', 'L,7'), SAMPLE.replace('E,7', 'E，7'),
                     SAMPLE.replace('[HEX]', '[GO]'), SAMPLE.replace('[先手胜]', '[平局]'),
                     SAMPLE[:-1]+';}', SAMPLE+'garbage']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_txt(text.encode('gb2312'))
        with self.assertRaisesRegex(ValueError, 'GB2312'):
            parse_txt(SAMPLE.encode('utf-8'))

    def test_marks_and_detailed_json_preservation(self):
        record = parse_txt(SAMPLE.encode('gb2312'))
        moves = record['moves']
        for move, mark in zip(moves, [-2,-1,0,1,2]):
            move.update(mark=mark, comment='详细连接判断与后续变化。')
        j = dict(format='hexzero-local-match-v3', boardSize=11, recordMetadata=record['metadata'], recordMoves=moves)
        restored = parse_json_record(json.dumps(j, ensure_ascii=False).encode('utf-8'))
        self.assertEqual(restored['moves'][0]['comment'], '详细连接判断与后续变化。')
        _, data = export_txt(restored['moves'], restored['metadata'])
        self.assertNotIn('详细', data.decode('gb2312'))
        for mark in [-2,-1,0,1,2]:
            self.assertIn(f'MARK[{mark}]', data.decode('gb2312'))
        self.assertEqual([mark_from_value(x) for x in [-1,-.6,-.2,0,.2,.6,1]],[-2,-1,0,0,0,1,2])


class RecordSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        net = NNetWrapper(HexGame(11), dict(cuda=False,num_channels=8,num_res_blocks=1))
        net.save_checkpoint(str(cls.path), 'model.pth.tar', include_training=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def app(self):
        return HexApp(self.path/'model.pth.tar',self.path/'games',simulations=8)

    def imported(self):
        app = self.app()
        state = app.import_record(base64.b64encode(SAMPLE.encode('gb2312')).decode(), 'txt')
        return app,app.sessions[state['id']]

    def test_seek_is_read_only_export_retains_whole_record(self):
        app,s = self.imported()
        self.assertEqual(s.state()['replayTotal'],8)
        self.assertEqual(s.moves,[])
        s.seek(3,s.revision)
        self.assertEqual(s.board[4,4],1)
        self.assertEqual(s.board[5,4],-1)
        self.assertEqual(s.player,-1)
        with self.assertRaises(ValueError):s.step(s.revision)
        with self.assertRaises(ValueError):s.human_move(0,0,s.revision)
        s.analyze(s.revision,'trained')
        self.assertEqual(len(s.analysis['candidates']),3)
        self.assertEqual(s.competition_record()[1],SAMPLE.encode('gb2312'))
        s.seek(0,s.revision)
        self.assertEqual(np.count_nonzero(s.board),0)
        s.seek(8,s.revision)
        self.assertEqual(len(s.moves),8)

    def test_annotation_perspective_and_continue_without_overwriting_original(self):
        app,s = self.imported()
        s.seek(1,s.revision)
        s.analyze(s.revision,'trained')
        s.analysis['rootValue']=.8  # Blue to move likes the position: red's last move is disadvantaged.
        s.annotate(s.revision,generated=True)
        self.assertEqual(s.moves[-1]['mark'],-2)
        self.assertIn('红方视角 Q=-0.8000',s.moves[-1]['comment'])
        s.seek(3,s.revision)
        s.seek(1,s.revision)
        self.assertEqual(s.moves[-1]['mark'],-2)
        state=app.continue_record(s,s.revision)
        branch=app.sessions[state['id']]
        self.assertNotEqual(branch.id,s.id)
        self.assertTrue(branch.opening_setup)
        self.assertFalse(branch.replay_mode)
        self.assertEqual(branch.record_metadata['result'],'')
        self.assertEqual(branch.moves[0]['mark'],-2)
        self.assertEqual(len(s.record_moves),8)
        branch.human_move(0,0,branch.revision)
        self.assertEqual(len(s.record_moves),8)
        with self.assertRaisesRegex(ValueError,'比赛 TXT 必须填写'):
            branch.competition_record()

    def test_rejected_seek_does_not_mutate_live_game(self):
        app=self.app();state=app.new_session(manual_opening=True);s=app.sessions[state['id']]
        before=s.state()
        with self.assertRaises(ValueError):s.seek(1,s.revision)
        self.assertEqual(before,s.state())

    def test_detailed_annotations_larger_than_txt_limit_can_be_reimported(self):
        record = parse_txt(SAMPLE.encode('gb2312'))
        for move in record['moves']:
            move['comment'] = '连接判断' * 1000
        data = json.dumps(dict(format='hexzero-local-match-v3', boardSize=11,
                               recordMetadata=record['metadata'],
                               moves=record['moves'], recordMoves=record['moves'])).encode()
        self.assertGreater(len(data), 262144)
        app = self.app()
        state = app.import_record(base64.b64encode(data).decode(), 'json')
        session = app.sessions[state['id']]
        session.seek(8, session.revision)
        self.assertEqual(session.moves[-1]['comment'], '连接判断' * 1000)
        with self.assertRaisesRegex(ValueError, '过大'):
            app.import_record(base64.b64encode(data).decode(), 'txt')


if __name__=='__main__':unittest.main()
