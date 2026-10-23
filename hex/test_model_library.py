"""Runtime model discovery, safe switching, and portable selection persistence."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hex.model_library import ModelLibrary
from hex_play_server import HexApp

ROOT=Path(__file__).resolve().parents[1]

class ModelLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.models=self.root/'models';self.models.mkdir()
        self.add_model(800)

    def add_model(self,iteration,subfolder=None):
        source=ROOT/f'models/local_play/archive/iteration_{iteration}'
        if not (source/'trained.onnx').is_file():source=ROOT/f'models/local_play/onnx/iteration_{iteration}'
        folder=self.models/(subfolder or f'iteration_{iteration}');folder.mkdir(parents=True)
        for name in ['trained.onnx','trained.onnx.json']:shutil.copy2(source/name,folder/name)
        return folder

    def app(self):
        app=HexApp(self.models/'iteration_800/trained.onnx',self.root/'games',backend='onnx',threads=4,simulations=8,
                   reference_checkpoint=ROOT/'models/local_play/onnx/azalea/reference.onnx')
        app.enable_model_library(self.models)
        return app

    def test_copy_complete_remove_and_errors_are_discovered(self):
        lib=ModelLibrary(self.models);self.assertEqual(len(lib.entries),1)
        folder=self.add_model(880,'新的模型/880')
        manifest=folder/'trained.onnx.json';data=manifest.read_bytes();manifest.unlink()
        lib.refresh();entry=next(e for e in lib.entries if e['id'].startswith('新的'))
        self.assertFalse(entry['available']);self.assertIn('清单',entry['error'])
        manifest.write_bytes(data);lib.refresh();self.assertEqual(lib.choose(),'新的模型/880/trained.onnx')
        self.assertEqual(lib.entries[0]['teacherFraction'],.1)
        (folder/'trained.onnx').unlink();lib.refresh();self.assertFalse(next(e for e in lib.entries if e['id'].startswith('新的'))['available'])
        self.assertEqual(lib.choose(),'iteration_800/trained.onnx')
        extra=self.models/'raw';extra.mkdir();(extra/'new.pth.tar').write_bytes(b'checkpoint')
        lib.refresh();self.assertIn('先导出 ONNX',next(e for e in lib.entries if e['id'].endswith('.pth.tar'))['error'])
        with self.assertRaises(ValueError):lib.resolve('../outside.onnx')

    def test_corrupt_and_incompatible_model_do_not_hide_good_models(self):
        folder=self.add_model(880);model=folder/'trained.onnx'
        with model.open('ab') as f:f.write(b'corrupt')
        lib=ModelLibrary(self.models);self.assertEqual(lib.choose('800'),'iteration_800/trained.onnx')
        with self.assertRaisesRegex(ValueError,'校验失败'):lib.resolve('iteration_880/trained.onnx')
        bad=json.loads((self.models/'iteration_800/trained.onnx.json').read_text());bad['canonical_version']=2
        (self.models/'iteration_800/trained.onnx.json').write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'没有可用模型'):lib.choose()

    def test_switch_saves_original_model_and_discards_search_tree(self):
        self.add_model(880);app=self.app();state=app.new_session(manual_opening=True)
        old=app.sessions[state['id']];old.human_move(5,5,old.revision);old.analyze(old.revision,'trained')
        saved=[];app.on_model_changed=saved.append
        result=app.select_model('iteration_880/trained.onnx',mode='azalea',mine=-1,simulations=8)
        self.assertEqual(result['config']['models']['trained']['metadata']['iteration'],880)
        self.assertEqual(result['session']['moves'],[]);self.assertIsNone(result['session']['analysis'])
        self.assertNotEqual(result['session']['id'],old.id);self.assertEqual(saved,['iteration_880/trained.onnx'])
        record=json.loads(Path(old.saved_path).read_text())
        self.assertEqual(record['models']['trained']['metadata']['iteration'],800)
        self.assertEqual(len(record['moves']),1)
        new=app.sessions[result['session']['id']];new.analyze(new.revision,'trained');new.step(new.revision)
        self.assertEqual(len(new.moves),1)
        self.assertIs(new.engines[1].policy.nnet,app.net)

    def test_failed_selection_or_save_keeps_current_game_and_model(self):
        self.add_model(880);app=self.app();state=app.new_session(manual_opening=True)
        old=app.sessions[state['id']];old.human_move(5,5,old.revision);net=app.net
        for kwargs in [dict(ident='missing.onnx'),dict(ident='iteration_880/trained.onnx',mode='bad')]:
            with self.assertRaises(ValueError):app.select_model(**kwargs)
        def fail(_):raise OSError('settings unavailable')
        app.on_model_changed=fail
        with self.assertRaises(OSError):app.select_model('iteration_880/trained.onnx')
        self.assertIs(app.net,net);self.assertIs(app.sessions[old.id],old)
        self.assertEqual(app.metadata['iteration'],800);self.assertEqual(len(old.moves),1)
        with patch('hex.onnx_inference.validate_trained_network',side_effect=ValueError('bad outputs')):
            with self.assertRaisesRegex(ValueError,'未切换'):app.select_model('iteration_880/trained.onnx')
        self.assertIs(app.net,net)

    def test_overwritten_active_file_requires_explicit_model_reload(self):
        folder=self.add_model(880);app=self.app()
        for name in ['trained.onnx','trained.onnx.json']:
            shutil.copy2(folder/name,self.models/'iteration_800'/name)
        with self.assertRaisesRegex(ValueError,'已被替换'):app.set_threads(8)
        self.assertEqual(app.threads,4);self.assertEqual(app.metadata['iteration'],800)
        app.select_model('iteration_800/trained.onnx')
        self.assertEqual(app.metadata['iteration'],880)

    def test_settings_preserve_model_and_threads_and_fallback_when_removed(self):
        folder=self.add_model(880)
        spec=importlib.util.spec_from_file_location('portable_library_test',ROOT/'portable/launcher.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        with patch.object(launcher,'ROOT',self.root):
            self.assertEqual(launcher.choose_model(),'iteration_880/trained.onnx')
            launcher.save_model('iteration_800/trained.onnx');launcher.save_threads(12)
            self.assertEqual(launcher.choose_model(),'iteration_800/trained.onnx')
            self.assertEqual(launcher.saved_threads(),12)
            launcher.save_model('iteration_880/trained.onnx')
            self.assertEqual(launcher.saved_threads(),12)
            shutil.rmtree(folder)
            self.assertEqual(launcher.choose_model(),'iteration_800/trained.onnx')

if __name__=='__main__':unittest.main()
