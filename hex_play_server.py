"""旧网页兼容入口；对弈状态与模型管理已迁入 hex_play_core。

新桌面程序直接调用同一个核心，无需启动此 HTTP 服务。保留原导入名称，
使已有本地测试、运行脚本及旧版便携启动器仍能使用。
"""
import argparse
import json
import logging
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse, quote

from hex_play_core import (HexApp, HexSession, integer, latest_checkpoint, winning_path,
                           MAX_RECORD_BYTES, MAX_JSON_RECORD_BYTES)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / 'hex_web'
log = logging.getLogger(__name__)


def make_handler(app):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

        def end_headers(self):
            # UI assets and the Python backend must always belong to the same version.
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def send_json(self, status, payload, download=False):
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            if download:
                self.send_header('Content-Disposition', 'attachment; filename="hex-match.json"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                with app.lock:
                    if url.path == '/api/config':
                        return self.send_json(200, app.config())
                    if url.path == '/api/models':
                        return self.send_json(200, app.model_catalog(refresh=True))
                    if url.path == '/api/export-txt':
                        session = app.sessions[parse_qs(url.query)['id'][0]]
                        filename, data = session.competition_record()
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/plain; charset=gb2312')
                        self.send_header('Content-Length', str(len(data)))
                        self.send_header('Content-Disposition', "attachment; filename=hex-match.txt; filename*=UTF-8''" + quote(filename, safe=''))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    if url.path in ('/api/session', '/api/export'):
                        session = app.sessions[parse_qs(url.query)['id'][0]]
                        return self.send_json(200, session.record() if url.path == '/api/export' else session.state(),
                                              download=url.path == '/api/export')
                if url.path not in ('/', '/index.html', '/app.js', '/styles.css'):
                    return self.send_error(404)
                # Do not revalidate an old asset cached before no-store was added.
                if 'If-Modified-Since' in self.headers:
                    del self.headers['If-Modified-Since']
                return super().do_GET()
            except ValueError as exc:
                return self.send_json(400, {'error': str(exc)})
            except KeyError as exc:
                return self.send_json(404, {'error': '棋局不存在或已经释放，请新开一局'})

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    return self.send_json(403, {'error': '不允许跨站请求'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 12 * 1024 * 1024:
                    raise ValueError('请求大小无效')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('请求必须是 JSON 对象')
                with app.lock:
                    if self.path == '/api/threads':
                        state = app.set_threads(body.get('threads'))
                    elif self.path == '/api/select-model':
                        state = app.select_model(body.get('modelId'), body.get('mode', 'human'),
                                                 body.get('myPlayer', 1), body.get('mctsSims'), body.get('seed', 0), body.get('moveTimeMs', 0), body.get('clockMinutes', 0))
                    elif self.path == '/api/import':
                        state = app.import_record(body.get('data'), body.get('format'), body.get('mode', 'human'),
                                                  body.get('myPlayer', 1), body.get('mctsSims'), body.get('seed', 0), body.get('moveTimeMs', 0), body.get('clockMinutes', 0))
                    elif self.path == '/api/new':
                        state = app.new_session(body.get('mode', 'human'), body.get('myPlayer', 1),
                                                body.get('mctsSims'), body.get('seed', 0),
                                                body.get('manualOpening', False), body.get('moveTimeMs', 0), body.get('clockMinutes', 0))
                    elif self.path in ('/api/move', '/api/step', '/api/undo', '/api/save', '/api/start', '/api/analyze',
                                       '/api/seek', '/api/continue-record', '/api/record-metadata', '/api/annotate', '/api/clock'):
                        session = app.sessions[body['sessionId']]
                        if self.path == '/api/move':
                            session.human_move(body['row'], body['col'], body['revision'])
                        elif self.path == '/api/step':
                            session.step(body['revision'])
                        elif self.path == '/api/clock':
                            session.control_clock(body.get('running'), body['revision'])
                        elif self.path == '/api/undo':
                            session.undo(body['revision'])
                        elif self.path == '/api/start':
                            session.finish_opening(body['revision'])
                        elif self.path == '/api/analyze':
                            session.analyze(body['revision'], body.get('engine', 'current'))
                        elif self.path == '/api/seek':
                            session.seek(body.get('ply'), body['revision'])
                        elif self.path == '/api/continue-record':
                            state = app.continue_record(session, body['revision'])
                            return self.send_json(200, state)
                        elif self.path == '/api/record-metadata':
                            session.update_metadata(body.get('metadata'), body['revision'])
                        elif self.path == '/api/annotate':
                            session.annotate(body['revision'], body.get('mark'), body.get('comment', ''),
                                             body.get('generated', False))
                        else:
                            session.save()
                        state = session.state()
                    else:
                        return self.send_json(404, {'error': '接口不存在'})
                self.send_json(200, state)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': str(exc)})
            except Exception:
                log.exception('Local match request failed')
                self.send_json(500, {'error': '对弈失败，请查看本地服务日志并重新开局'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-dir', default=str(ROOT / 'models/local_play'))
    parser.add_argument('--board-size', type=int)
    parser.add_argument('--save-dir', default=str(ROOT / 'runs/local_play/games'))
    parser.add_argument('--reference-source', default=str(ROOT.parent / 'azalea'))
    parser.add_argument('--reference-checkpoint', default=str(ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth'))
    parser.add_argument('--mcts-sims', type=int, default=3200)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--backend', choices=('torch', 'onnx'), default='torch')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    integer(args.threads, '线程数', 1, 32)
    integer(args.mcts_sims, '搜索次数', 1, 100000)
    if args.backend == 'onnx':
        if args.device == 'cuda':
            parser.error('ONNX 本地对弈后端只支持 CPU')
        device = 'cpu'
    else:
        import torch
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        if device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA 不可用，请使用 --device cpu')
        torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO)
    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    app = HexApp(checkpoint, args.save_dir, device, args.mcts_sims,
                 args.reference_source, args.reference_checkpoint, args.board_size,
                 backend=args.backend, threads=args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f'训练模型：{app.checkpoint}\n设备：{device}\n本地对弈：http://{args.host}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
