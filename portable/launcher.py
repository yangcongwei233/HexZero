"""Windows 便携式 Hex 对弈程序的离线启动器。"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
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
    """读取便携目录偏好 JSON，缺失或损坏时返回默认空设置并记录原因。"""
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
    """读取保存的线程数并校验范围，非法设置退回自动默认值。"""
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
    """更新单个偏好项并通过临时文件原子发布，保留其他已存设置。"""
    settings = read_settings()
    settings[key] = value
    temporary = ROOT / 'settings.json.tmp'
    temporary.write_text(json.dumps(settings, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(ROOT / 'settings.json')


def save_threads(threads):
    """保存用户选择的推理线程数，供下次启动沿用。"""
    save_setting('threads', threads)


def save_model(ident):
    """保存当前模型的相对标识，便于下次启动恢复选择。"""
    save_setting('model', ident)


def choose_model(requested=None):
    """优先使用显式模型，其次尝试保存的选择，最后从模型库选择可用默认项。"""
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




def supervise_server(args, model, threads):
    """即使未抛出 Python 异常，也显示原生进程退出信息。"""
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
    """加载便携模型和配置，绑定模型/线程偏好回调，并用独立小树预热推理后返回应用。"""
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
    return app




def main():
    """解析便携启动选项并启动服务，记录进程退出信息和错误日志。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Iteration number or ONNX path relative to models; default remembers last selection')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
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
