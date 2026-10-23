"""桌面串行服务：工作线程独占模型和棋局，向 UI 返回深拷贝快照。"""
import base64
import copy
import json
from pathlib import Path
import platform
import struct
import time

from hex_play_core import HexApp, integer, MAX_RECORD_BYTES, MAX_JSON_RECORD_BYTES
from .local_players import TrainedPlayer
from .model_library import ModelLibrary
from .competition_profile import MODEL_ID, validate_competition_model

SEARCH_PRESETS = (800, 1200, 2000, 3200, 6400, 12800, 51200)


class DesktopService:
    """管理模型、会话和偏好；ONNX 与清单成对加载并校验 SHA-256。"""

    def __init__(self, models, data, model=None, threads=None, competition=False):
        self.models, self.data = Path(models).resolve(), Path(data).resolve()
        self.competition = bool(competition)
        self.model_error = None
        self.search_backend = None
        self.data.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.data / 'settings.json'
        self.settings = self._read_settings()
        report = dict(python=platform.python_version(), system=platform.platform(),
                      pointer_bits=struct.calcsize('P') * 8, status='checking')
        try:
            if report['pointer_bits'] != 64:
                raise ValueError('ONNX 推理需要 64 位 Python')
            threads = threads if threads is not None else self.settings.get('threads', 4)
            integer(threads, 'CPU 线程数', 4, 16)
            library = ModelLibrary(self.models)
            # 公开版没有权重时仍提供完整人人对弈；显式指定模型或比赛版则严格失败。
            ident = None
            if self.competition:
                if model is not None and str(model) not in ('3250', MODEL_ID):
                    raise ValueError('比赛版已锁定 3250，不能用启动参数指定其他模型')
                ident = library.choose(MODEL_ID)
            elif model is not None:
                ident = library.choose(model)
            else:
                try:
                    ident = library.choose(self.settings.get('model'))
                except (OSError, ValueError):
                    try:
                        ident = library.choose()
                    except (OSError, ValueError) as exc:
                        self.model_error = str(exc)
            sims = self.settings.get('simulations', 12800)
            if type(sims) is not int or sims not in SEARCH_PRESETS:
                sims = 12800
            def create_app(checkpoint):
                # 桌面不传 reference_checkpoint；复制 Azalea 权重也不会启用对战入口。
                return HexApp(checkpoint, self.data / 'games', device='cpu',
                              backend='onnx', threads=threads, simulations=sims)
            try:
                self.app = create_app(self.models / ident if ident else None)
                if self.app.net is not None:
                    self._warmup()
            except Exception as exc:
                if self.competition or model is not None:
                    raise
                self.model_error = f'模型未加载：{exc}'
                self.search_backend = None
                self.app = create_app(None)
            if not self.competition:
                self.app.enable_model_library(self.models)
            mode = 'human' if self.app.net is not None else 'pvp'
            self.session_id = self.app.new_session(mode=mode, simulations=sims)['id']
            import numpy
            report.update(status='passed', numpy=numpy.__version__,
                          search_backend=self.search_backend, threads=threads,
                          competition=self.competition, model_available=self.app.net is not None,
                          model_error=self.model_error)
            if self.app.net is not None:
                import onnxruntime
                report.update(onnxruntime=onnxruntime.__version__, precision='FP32',
                              model=ident, model_sha256=self.app.net.manifest['onnx_sha256'])
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            (self.data / 'environment-check.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _warmup(self):
        """用独立小树验证真实推理，不污染实战访问量；比赛版同时验证固定权重。"""
        from .onnx_inference import validate_trained_network
        if self.competition:
            validate_competition_model(self.app.net)
        validate_trained_network(self.app.net)
        probe = TrainedPlayer(self.app.game, self.app.net, 65, 0)
        probe.choose(self.app.game.getInitBoard(), 1)
        self.search_backend = getattr(probe.policy.mcts, 'backend', 'python')

    def _read_settings(self):
        """损坏的偏好文件不应阻止启动；模型清单损坏则由 ModelLibrary 严格拒绝。"""
        try:
            result = json.loads(self.settings_path.read_text(encoding='utf-8'))
            return result if isinstance(result, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_settings(self, **changes):
        """采用同目录临时文件和 replace，防止意外退出写出半份 JSON。"""
        self.settings.update(changes)
        temporary = self.settings_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.settings_path)

    @property
    def session(self):
        return self.app.sessions[self.session_id]

    def snapshot(self):
        """棋钟用单调时间；UI 只根据采样时刻显示倒计时，判负仍由核心负责。"""
        config = self.app.config()
        config.update(competition=self.competition, modelLocked=self.competition,
                      modelError=self.model_error, playModes=['pvp', 'human'] if self.app.net is not None else ['pvp'])
        return copy.deepcopy(dict(config=config, session=self.session.state(),
                                  sampled_at=time.monotonic(), search_backend=self.search_backend))

    def perform(self, operation, **params):
        """串行分派操作；revision 校验局面，棋谱仅本地保存，不加入训练池。"""
        session = self.session
        revision = params.pop('revision', session.revision)
        # 边界验证也在服务层执行，不能靠隐藏 UI 阻止旧入口继续发起 Azalea 对局。
        if operation in ('new', 'import'):
            if params.get('mode', 'pvp') not in ('human', 'pvp'):
                raise ValueError('桌面版只支持人人对弈和人机对弈')
            params.setdefault('mode', 'pvp' if self.app.net is None else 'human')
        if operation == 'new':
            if session.moves:
                session.save()
            state = self.app.new_session(**params)
            self.session_id = state['id']
            self._save_settings(simulations=state['mctsSims'])
        elif operation == 'move':
            session.human_move(params['row'], params['col'], revision)
        elif operation == 'step':
            session.step(revision)
        elif operation == 'start':
            session.finish_opening(revision)
        elif operation == 'undo':
            session.undo(revision)
        elif operation == 'analyze':
            if params.get('kind', 'current') not in ('trained', 'current'):
                raise ValueError('桌面版只使用 HexZero 模型分析')
            session.analyze(revision, params.get('kind', 'current'))
        elif operation == 'seek':
            session.seek(params['ply'], revision)
        elif operation == 'continue':
            self.session_id = self.app.continue_record(session, revision)['id']
        elif operation == 'annotate':
            session.annotate(revision, **params)
        elif operation == 'metadata':
            session.update_metadata(params['metadata'], revision)
        elif operation == 'clock':
            session.control_clock(params['running'], revision)
        elif operation == 'threads':
            self.app.set_threads(params['threads'])
            self._save_settings(threads=self.app.threads)
        elif operation == 'model':
            if self.competition:
                raise ValueError('比赛版已锁定 3250，不支持切换模型')
            result = self.app.select_model(params['ident'], mode=session.mode, mine=session.mine,
                simulations=session.simulations, seed=session.seed, move_time_ms=session.move_time_ms,
                clock_minutes=session.clock.minutes)
            self.session_id = result['session']['id']
            self.model_error = None
            self._warmup()
            self._save_settings(model=params['ident'])
        elif operation == 'refresh':
            self.app.model_catalog(refresh=True)
        elif operation == 'import':
            path = Path(params.pop('path'))
            kind = path.suffix.lower().lstrip('.')
            limit = MAX_RECORD_BYTES if kind == 'txt' else MAX_JSON_RECORD_BYTES
            # 限长读取防止误把模型或安装包当棋谱打开而占用大量内存。
            with path.open('rb') as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError('棋谱文件过大')
            if session.moves:
                session.save()
            self.session_id = self.app.import_record(
                base64.b64encode(data).decode('ascii'), kind, **params)['id']
        elif operation == 'export':
            path = Path(params['path'])
            if path.suffix.lower() == '.txt':
                _, data = session.competition_record()
            else:
                data = (json.dumps(session.record(), ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            path.write_bytes(data)
        elif operation == 'save':
            session.save()
        elif operation == 'close':
            for current in self.app.sessions.values():
                current.clock.pause()
                if current.moves or current.replay_mode:
                    current.save()
            self.app.sessions.clear()
            return None
        elif operation != 'state':
            raise ValueError(f'未知桌面操作：{operation}')
        return self.snapshot()
