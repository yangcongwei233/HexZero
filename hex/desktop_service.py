"""桌面对弈的串行服务层：将 GUI 操作映射到 HexApp/HexSession。

此模块不导入 Tk，不创建 HTTP 服务。DesktopWindow 的唯一工作线程拥有这里的
所有模型与搜索树；UI 只收到 deepcopy 后的快照，不能访问正在搜索的 NumPy 数组。
这种边界避免 GUI 刷新、棋钟和模型切换同时操作 C++ 树，也便于离线验证功能。
"""
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
from .onnx_inference import validate_trained_network

SEARCH_PRESETS = (800, 1200, 2000, 3200, 6400, 12800, 51200)


class DesktopService:
    """一个窗口对应一个服务，当前会话 ID 始终由业务核心生成。

settings.json 仅保存用户偏好，不保存模型本身；新模型由 ModelLibrary 扫描。
ONNX 和配套 JSON 必须一起复制，加载时验证 SHA-256，防止拿错模型或复制未完成。
"""

    def __init__(self, models, data, model=None, threads=None):
        self.models, self.data = Path(models).resolve(), Path(data).resolve()
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
            # 优先显式参数，其次上次选择；本项目已确定 3250 为默认比赛模型。
            preferred = model or self.settings.get('model')
            if preferred is None and (self.models / 'iteration_3250/trained.onnx').is_file():
                preferred = 'iteration_3250/trained.onnx'
            if model is not None:
                ident = library.choose(model)  # 明确要求的模型不可用时不可悄悄换模型。
            else:
                try:
                    ident = library.choose(preferred)
                except (OSError, ValueError):
                    ident = library.choose()
            sims = self.settings.get('simulations', 12800)
            if type(sims) is not int or sims not in SEARCH_PRESETS:
                sims = 12800
            self.app = HexApp(self.models / ident, self.data / 'games', device='cpu',
                              backend='onnx', threads=threads, simulations=sims,
                              reference_checkpoint=self.models / 'azalea/reference.onnx')
            self.app.enable_model_library(self.models)
            validate_trained_network(self.app.net)
            # 真实运行小批量搜索以验证环境并预热；树随即释放，不送入实战或经验池。
            probe = TrainedPlayer(self.app.game, self.app.net, 65, 0)
            probe.choose(self.app.game.getInitBoard(), 1)
            self.search_backend = getattr(probe.policy.mcts, 'backend', 'python')
            del probe
            if self.app.reference is not None:
                probe = self.app.reference.player(65, 1)
                probe.choose(self.app.game.getInitBoard(), 1)
                del probe
            self.session_id = self.app.new_session(simulations=sims)['id']
            import numpy, onnxruntime
            report.update(status='passed', numpy=numpy.__version__, onnxruntime=onnxruntime.__version__,
                          search_backend=self.search_backend, precision='FP32', threads=threads,
                          model=ident, model_sha256=self.app.net.manifest['onnx_sha256'],
                          azalea_available=self.app.reference is not None,
                          azalea_error=self.app.reference_error)
        except Exception as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            (self.data / 'environment-check.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

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
        return copy.deepcopy(dict(config=self.app.config(), session=self.session.state(),
                                  sampled_at=time.monotonic(), search_backend=self.search_backend))

    def perform(self, operation, **params):
        """所有操作均由同一工作线程执行，revision 来自 UI 最后显示的局面。

导入/导出保留原 GB2312 TXT 与 UTF-8 JSON 规范。JSON 带完整分析和详细评语；
TXT 只含标准 MARK[-2..2]。此处绝不把对弈棋谱混入训练数据。
"""
        session = self.session
        revision = params.pop('revision', session.revision)
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
            result = self.app.select_model(params['ident'])
            self.session_id = result['session']['id']
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
