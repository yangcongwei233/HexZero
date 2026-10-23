"""Offline launcher for the portable Windows Hex test bench."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / 'app'
sys.path[:0] = [str(APP_ROOT), str(ROOT)]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HEX_FORCE_CPU'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'cache' / 'numba')


def read_settings():
    path = ROOT / 'settings.json'
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict): raise ValueError('settings must be an object')
        return data
    except (OSError, ValueError) as exc:
        logging.warning('Invalid settings.json; using defaults: %s', exc)
        return {}


def saved_threads():
    try:
        value = read_settings().get('threads')
        if value is None: return None
        if isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 16:
            raise ValueError('threads must be 4 through 16')
        return value
    except (ValueError, KeyError, TypeError) as exc:
        logging.warning('Invalid settings.json; using default threads: %s', exc)
        return None


def save_setting(key, value):
    settings = read_settings()
    settings[key] = value
    temporary = ROOT / 'settings.json.tmp'
    temporary.write_text(json.dumps(settings, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(ROOT / 'settings.json')


def save_threads(threads):
    save_setting('threads', threads)


def save_model(ident):
    save_setting('model', ident)


def choose_model(requested=None):
    from hex.model_library import ModelLibrary
    library = ModelLibrary(ROOT / 'models')
    if requested is not None:
        return library.choose(requested)
    saved = read_settings().get('model')
    if saved:
        try:
            return library.choose(saved)
        except (OSError, ValueError) as exc:
            print(f'上次选择的模型不可用，改用可用模型：{exc}', flush=True)
    return library.choose()


def startup_check(threads, model=None):
    command = [sys.executable, '-X', 'utf8', '-X', 'faulthandler', str(Path(__file__).resolve()), '--environment-worker']
    report_path = ROOT / 'logs/environment-check.json'
    report_path.unlink(missing_ok=True)
    if threads is not None:
        command.extend(['--threads', str(threads)])
    if model is not None:
        command.extend(['--model', model])
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=90)
    except subprocess.TimeoutExpired as exc:
        report_path.write_text(json.dumps(dict(status='failed', failed_stage='timeout',
                                               error='Environment check exceeded 90 seconds')),
                               encoding='utf-8')
        raise RuntimeError('环境检查超过 90 秒；请查看 logs，或运行 CHECK_ENV.bat') from exc
    (ROOT / 'logs/environment-process.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
    if result.returncode:
        code = result.returncode & 0xffffffff
        hints = {0xc000001d: 'CPU 不支持运行库使用的指令',
                 0xc0000005: '原生库访问无效内存',
                 0xc0000135: '缺少运行库 DLL，请完整解压软件包',
                 0xc000007b: '运行库架构不匹配，请使用完整 x64 软件包'}
        hint = hints.get(code, 'Python 或 C++ 推理环境检查失败')
        if result.returncode == -4:
            hint = 'CPU 不支持运行库使用的指令'
        # Preserve a native crash even if the child cannot write its JSON report.
        if result.returncode != 1 or not report_path.is_file():
            report_path.write_text(json.dumps(
                dict(status='failed', failed_stage='native_process', returncode=result.returncode,
                     error=hint), ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            try:
                details = json.loads(report_path.read_text(encoding='utf-8'))
                hint += '：' + details.get('error', details.get('failed_stage', ''))
            except (OSError, ValueError):
                pass
        raise RuntimeError(f'{hint} (exit {result.returncode})。详见 logs/environment-check.json 和 environment-process.log')
    print('Environment check passed: Python, native CPU libraries, selected model and Azalea.', flush=True)


def supervise_server(args, model, threads):
    """Keep native process exits visible even when no Python exception is raised."""
    command = [sys.executable, '-u', '-X', 'utf8', '-X', 'faulthandler', str(Path(__file__).resolve()),
               '--server-worker', '--model', model, '--port', str(args.port)]
    if threads is not None:
        command += ['--threads', str(threads)]
    if args.no_browser:
        command.append('--no-browser')
    path = ROOT / 'logs' / ('server-' + time.strftime('%Y%m%d-%H%M%S') + '.log')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    interrupted = False
    with path.open('w', encoding='utf-8') as logfile:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', env=env) as process:
            try:
                for line in process.stdout:
                    logfile.write(line)
                    logfile.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    code = process.wait()
    if not interrupted and code:
        message = (f'Hex server exited unexpectedly: {code} (0x{code & 0xffffffff:08X}).\n'
                   f'Details: {path}\n'
                   'You can try START_HEX_PYTHON.bat to use the compatible search backend.')
        print(message, file=sys.stderr, flush=True)
        (ROOT / 'logs/server-exit.json').write_text(json.dumps(
            dict(returncode=code, hex_code=f'0x{code & 0xffffffff:08X}', log=str(path)), indent=2), encoding='utf-8')
        raise SystemExit(1)
    print('Hex has stopped.', flush=True)


def load_app(model, save_dir, simulations=None, threads=None):
    from hex_play_server import HexApp
    if simulations is None:
        simulations = read_settings().get('simulations', 3200)
        if (isinstance(simulations, bool) or not isinstance(simulations, int)
                or simulations not in (800, 1200, 2000, 3200, 6400, 12800, 51200)):
            logging.warning('Invalid saved search preset; using 3200')
            simulations = 3200
    ident = choose_model(model)
    app = HexApp(ROOT / 'models' / ident,
                  save_dir, device='cpu', simulations=simulations,
                  reference_source=APP_ROOT / 'vendor' / 'azalea',
                  reference_checkpoint=ROOT / 'models' / 'azalea' / 'reference.onnx',
                  backend='onnx', threads=threads)
    app.enable_model_library(ROOT / 'models', save_model)
    app.on_threads_changed = save_threads
    # Compile/load native signatures and warm the real search batch before the
    # first timed move. This tree is discarded and never enters a game.
    from hex.local_players import TrainedPlayer
    TrainedPlayer(app.game, app.net, 65, 0).choose(app.game.getInitBoard(), 1)
    return app


def self_test(threads=None):
    import numpy as np
    import numba
    import onnxruntime
    from hex.HexLogic import has_won_bfs
    report = dict(python=sys.version, platform=sys.platform, onnxruntime=onnxruntime.__version__,
                  numpy=np.__version__, numba=numba.__version__, models=[])
    with tempfile.TemporaryDirectory(prefix='hex-selftest-') as folder:
        from hex.model_library import ModelLibrary
        entries = [e for e in ModelLibrary(ROOT / 'models').entries if e['available']]
        for entry in entries:
            app = load_app(entry['id'], folder, simulations=80, threads=threads)
            if app.reference is None:
                raise RuntimeError(app.reference_error)
            assert app.metadata.get('iteration') == entry['iteration']
            for mine in (1, -1):
                state = app.new_session(mode='azalea', mine=mine, simulations=80, manual_opening=True)
                session = app.sessions[state['id']]
                for action in (60, 61):
                    session.human_move(*divmod(action, 11), session.revision)
                for engine in ('trained', 'azalea'):
                    before = session.board.copy()
                    revision = session.revision
                    session.analyze(revision, engine)
                    assert len(session.analysis['candidates']) == 3
                    assert session.revision == revision and np.array_equal(session.board, before)
                session.finish_opening(session.revision)
                session.step(session.revision)
                session.step(session.revision)
                assert len(session.moves) == 4
                assert not has_won_bfs(session.board, 1) and not has_won_bfs(session.board, -1)
            report['models'].append(dict(id=entry['id'], iteration=entry['iteration'], threads=app.threads, manual_opening=True,
                                         both_analysis_engines=True, both_seats=True, handoff=True))
    if 'torch' in sys.modules:
        raise RuntimeError('Portable inference unexpectedly imported PyTorch')
    report.update(status='passed', pytorch_imported=False)
    (ROOT / 'logs' / 'self-test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Iteration number or ONNX path relative to models; default remembers last selection')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--check-environment', action='store_true')
    parser.add_argument('--environment-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--server-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--threads', type=int, default=None, help='CPU inference threads; default up to 4')
    args = parser.parse_args()
    if not 1 <= args.port <= 65525:
        parser.error('Port must be between 1 and 65525')
    if args.threads is not None and not 4 <= args.threads <= 16:
        parser.error('Threads must be between 4 and 16')
    (ROOT / 'logs').mkdir(exist_ok=True)
    log_path = ROOT / 'logs' / ('hex-' + time.strftime('%Y%m%d-%H%M%S') + '.log')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_path, encoding='utf-8')])
    try:
        threads = args.threads if args.threads is not None else saved_threads()
        model = choose_model(args.model)
        if args.environment_worker:
            from environment_check import check_environment
            result = check_environment(ROOT, threads, model)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if result['status'] != 'passed':
                raise SystemExit(1)
            return
        if not args.server_worker:
            print('Checking Python and native CPU inference environment...', flush=True)
            startup_check(threads, model)
        if args.check_environment:
            print((ROOT / 'logs/environment-check.json').read_text(encoding='utf-8'), flush=True)
            return
        if args.self_test:
            return self_test(threads)
        if not args.server_worker:
            return supervise_server(args, model, threads)
        print('Loading optimized CPU models (ONNX Runtime native engine)...', flush=True)
        from http.server import ThreadingHTTPServer
        from hex_play_server import make_handler
        app = load_app(model, ROOT / 'games', threads=threads)
        if app.reference is None:
            raise RuntimeError(app.reference_error)
        save_model(model)
        for port in range(args.port, args.port + 11):
            try:
                server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(app))
                break
            except OSError as exc:
                if exc.errno not in (98, 48, 10048) and getattr(exc, 'winerror', None) != 10048:
                    raise
        else:
            raise RuntimeError('No free local port. Close another Hex window or use --port.')
        url = f'http://127.0.0.1:{server.server_port}/'
        from hex.local_players import TrainedPlayer
        search_backend = getattr(TrainedPlayer(app.game, app.net, 1, 0).policy.mcts, 'backend', 'python')
        print(f'Build: 2026-10-20-search-presets-4\nSearch backend: {search_backend}\n'
              f'Model: {model}\nCPU threads: {app.threads}\nOpen: {url}\n'
              'Keep this window open while playing. Ctrl+C stops Hex.', flush=True)
        logging.info('Model %s; ONNX Runtime, %s threads; URL %s', model, app.threads, url)
        if not args.no_browser:
            timer = threading.Timer(0.5, webbrowser.open, args=(url,))
            timer.daemon = True
            timer.start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    except Exception as exc:
        logging.exception('Hex startup failed')
        print(f'Could not start Hex: {exc}\nDetails: {log_path}', file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
