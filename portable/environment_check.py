"""在独立子进程内检查环境，以便诊断原生动态库崩溃。"""
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time


def check_environment(root, threads, model=None):
    """在隔离进程中验证解释器、原生库、模型和真实搜索，包括短生命周期线程路径；结果仅写入本地环境报告。"""
    root = Path(root)
    report = dict(status='checking', checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  system=platform.system(), release=platform.release(), machine=platform.machine(),
                  python=platform.python_version(), pointer_bits=struct.calcsize('P') * 8,
                  logical_cpus=os.cpu_count(), requested_threads=threads, checks=[], models=[])
    stage = 'system'
    try:
        if report['pointer_bits'] != 64 or report['machine'].lower() not in ('amd64', 'x86_64'):
            raise RuntimeError('本包需要 Intel/AMD 64 位系统；当前架构不匹配')
        if sys.version_info < (3, 11):
            raise RuntimeError('需要包内附带的 Python 3.11 或更新版本')
        report['checks'].append(stage)
        stage = 'writable_directories'
        for name in ('logs', 'games', 'cache'):
            folder = root / name
            folder.mkdir(exist_ok=True)
            with tempfile.TemporaryFile(dir=folder) as f:
                f.write(b'hex')
        report['checks'].append(stage)
        stage = 'native_libraries'
        import numpy as np
        import numba
        import onnxruntime as ort
        report.update(numpy=np.__version__, numba=numba.__version__, onnxruntime=ort.__version__,
                      providers=ort.get_available_providers())
        if 'CPUExecutionProvider' not in report['providers']:
            raise RuntimeError('缺少 ONNX CPU 推理内核')
        report['checks'].append(stage)
        stage = 'hex_rules_native_compilation'
        from hex.HexGame import HexGame
        game = HexGame(11)
        board = game.getInitBoard()
        board[:, 0] = 1
        if game.getGameEnded(board, 1) != 1:
            raise RuntimeError('棋盘规则检查失败')
        report['checks'].append(stage)
        stage = 'search_native_compilation'
        from hex.search_kernels import select_edge, virtual_path, backup_paths
        if any(kernel is None for kernel in (select_edge, virtual_path, backup_paths)):
            raise RuntimeError('缺少搜索计算内核')
        legal = np.ones(2, np.int8)
        priors = np.array([.25, .75], np.float64)
        counts = np.zeros(2, np.int64)
        if select_edge(legal, priors, counts, np.zeros(2), counts, 1, .5) != 1:
            raise RuntimeError('搜索选点内核检查失败')
        parent = np.array([-1, 0], np.int32)
        visits, totals = np.zeros(2, np.float32), np.zeros(2, np.float32)
        virtual_path(parent, visits, totals, 1, 0, 1)
        virtual_path(parent, visits, totals, 1, 0, -1)
        backup_paths(parent, visits, totals, np.array([1], np.int64), 0, np.array([.5], np.float32))
        if not (np.array_equal(visits, [1, 1]) and np.array_equal(totals, [-.5, .5])):
            raise RuntimeError('搜索回传内核检查失败')
        report['checks'].append(stage)
        stage = 'cpp_search'
        from hex.native_mcts import local_mcts
        from utils import dotdict
        class CheckNet:
            def predict_batch(self, boards):
                """为搜索库环境验证返回固定均匀策略和零价值，隔离模型计算对兼容检查的影响。"""
                count = len(boards)
                return np.full((count, 121), 1 / 121, np.float32), np.zeros((count, 1), np.float32)
        search = local_mcts(game, CheckNet(), dotdict(numMCTSSims=65, mctsBatchSize=32,
                                                    cpuct=.5, seed=7, reuseSearchTree=True))
        probs = search.getActionProb(game.getInitBoard(), temp=0)
        if not (search.tree.root.N == 64 and sum(probs) == 1 and len(probs) == 121):
            raise RuntimeError('搜索引擎试运行失败')
        report['search_backend'] = getattr(search, 'backend', 'python')
        report['checks'].append(stage)
        stage = 'model_inference'
        from hex.onnx_inference import OnnxNetwork, validate_trained_network
        from hex.model_library import ModelLibrary
        library = ModelLibrary(root / 'models')
        selected = library.choose(model)
        report['model_catalog'] = library.snapshot()
        for kind, relative in [('trained', selected), ('azalea', 'azalea/reference.onnx')]:
            net = OnnxNetwork(root / 'models' / relative, kind, threads)
            if kind == 'trained':
                validate_trained_network(net)
                policy, value = net.predict_batch(np.zeros((2, 11, 11), np.float32))
            else:
                output = net.run({'board': np.zeros((2, 11, 11), np.int32),
                                  'legal_moves': np.tile(np.arange(1, 122, dtype=np.int32), (2, 1))})
                policy, value = np.exp(output['moves_logprob']), output['value']
            if not (np.isfinite(policy).all() and np.isfinite(value).all()
                    and np.allclose(policy.sum(1), 1, atol=1e-5) and np.all(np.abs(value) <= 1)):
                raise RuntimeError(f'模型输出异常：{relative}')
            report['models'].append(dict(model=relative, sha256=net.manifest['onnx_sha256'],
                                         cpu_inference='passed', threads=net.threads))
            del net
        report['checks'].append(stage)
        stage = 'local_match_workflow'
        # 网页服务会在短生命周期的请求线程中创建和释放引擎，
        # 因此除主线程推理外，也要检查该线程路径。
        from concurrent.futures import ThreadPoolExecutor
        from hex_play_server import HexApp
        with tempfile.TemporaryDirectory(prefix='hex-startup-check-') as temporary:
            app = HexApp(root / 'models' / selected, temporary, device='cpu', simulations=65,
                         reference_checkpoint=root / 'models/azalea/reference.onnx',
                         backend='onnx', threads=threads)
            if app.reference is None:
                raise RuntimeError(app.reference_error)
            def request(fn, *args, **kwargs):
                """在线程池中执行一次调用并等待退出，验证原生库在线程创建和销毁场景下是否安全。"""
                with ThreadPoolExecutor(1) as worker:
                    return worker.submit(fn, *args, **kwargs).result()
            request(app.new_session, mode='human', mine=1, simulations=65)
            for mine in (1, -1):
                state = request(app.new_session, mode='azalea', mine=mine, simulations=65)
                session = app.sessions[state['id']]
                for _ in range(2):
                    request(session.step, session.revision)
                if len(session.moves) != 2:
                    raise RuntimeError('双方搜索与模式切换试运行失败')
            request(app.sessions.clear)
            del session, app
        report['checks'].append(stage)
        if 'torch' in sys.modules:
            raise RuntimeError('优化版不应依赖 PyTorch')
        report.update(status='passed', pytorch_required=False,
                      native_windows_execution=sys.platform == 'win32')
    except Exception as exc:
        report.update(status='failed', failed_stage=stage, error=f'{type(exc).__name__}: {exc}')
    folder = root / 'logs'
    folder.mkdir(exist_ok=True)
    (folder / 'environment-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report
