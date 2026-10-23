#!/usr/bin/env python3
"""以平衡先后手和可复现种子评估本地模型。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
APP = None


def digest(path):
    """计算权重文件 SHA-256，作为本批评测的模型身份记录。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    """通过临时文件原子保存评测清单与进度。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def initialize(checkpoint, reference_source, reference_checkpoint, output, simulations, backend, threads):
    """为评测进程设置计算线程并加载训练模型和 Azalea 参考，复用网络执行多局任务。"""
    global APP
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HEX_FORCE_CPU'] = '1'
    sys.path.insert(0, str(ROOT))
    if backend == 'torch':
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    from hex_play_server import HexApp
    APP = HexApp(checkpoint, Path(output) / 'games', device='cpu', simulations=simulations,
                 backend=backend, threads=threads,
                 reference_source=reference_source, reference_checkpoint=reference_checkpoint)
    if APP.reference is None:
        raise RuntimeError(APP.reference_error)


def play(task):
    """在指定开局和棋色分配下完成一局，独立验证胜负及落子唯一性，保存棋谱和双方搜索条件。"""
    from hex.HexLogic import has_won_bfs
    from hex_play_server import HexSession
    game_number, seed, trained_side, opening = task
    session = HexSession(APP, mode='azalea', mine=-trained_side,
                         simulations=APP.default_simulations, seed=seed)
    started = time.perf_counter()
    for action in opening['actions']:
        if session.winner is not None:
            raise ValueError('Opening continues after a terminal position')
        session.apply(action, {'source': 'preset_opening', 'opening_id': opening['id']})
    if session.winner is not None:
        raise ValueError('Opening must leave an unfinished game')
    while session.winner is None:
        session.step(session.revision)
        first_won = bool(has_won_bfs(session.board, 1))
        second_won = bool(has_won_bfs(session.board, -1))
        assert not (first_won and second_won), 'Both sides cannot win'
        independent_winner = 1 if first_won else -1 if second_won else None
        assert session.winner == independent_winner, 'Winner check disagrees with BFS'
        assert len(session.moves) <= APP.game.n ** 2
    actions = [m['action'] for m in session.moves]
    assert len(set(actions)) == len(actions), 'Repeated occupied move'
    reference = APP.reference.prototype
    return dict(game=game_number, seed=seed, trained_side=trained_side,
                opening=opening,
                winner=session.kind(session.winner), winner_side=session.winner,
                moves=len(actions), seconds=round(time.perf_counter() - started, 3),
                record=session.saved_path,
                trajectory_sha256=hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
                model_iteration=APP.metadata.get('iteration'),
                inference_backend=APP.backend,
                reference_search=dict(simulations=APP.default_simulations,
                                      batch_size=reference.search_batch_size,
                                      cpuct=reference.exploration_coef))


def summarize(results):
    """汇总训练模型对参考模型的总胜率及分先后手成绩。"""
    wins = Counter(r['winner'] for r in results)
    seats = {}
    for side, name in ((1, 'trained_first_red'), (-1, 'trained_second_blue')):
        subset = [r for r in results if r['trained_side'] == side]
        won = sum(r['winner'] == 'trained' for r in subset)
        seats[name] = dict(games=len(subset), trained_wins=won,
                           azalea_wins=len(subset) - won,
                           trained_win_rate=won / len(subset) if subset else None,
                           unique_trajectories=len({r['trajectory_sha256'] for r in subset}))
    return dict(completed=len(results), trained_wins=wins['trained'],
                azalea_wins=wins['azalea'],
                trained_win_rate=wins['trained'] / len(results) if results else None,
                by_seat=seats)


def main():
    """加载固定模型及开局列表，按交换先后手任务执行并行评测，保存可复现设置和完整结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, default=ROOT.parent / 'azalea')
    parser.add_argument('--reference-checkpoint', type=Path,
                        help='Reference weights; default follows the selected inference backend')
    parser.add_argument('--backend', choices=('auto', 'torch', 'onnx'), default='auto',
                        help='Auto uses ONNX Runtime for .onnx models, PyTorch otherwise')
    parser.add_argument('--threads', type=int, default=1, help='Inference threads per worker (1-16)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games-per-seat', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=800)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--openings', type=Path,
                        help='JSON list of {id, actions}: one preset position per seat pair')
    args = parser.parse_args()
    if args.backend == 'auto':
        args.backend = 'onnx' if args.checkpoint.suffix.lower() == '.onnx' else 'torch'
    if (args.checkpoint.suffix.lower() == '.onnx') != (args.backend == 'onnx'):
        parser.error('The model file must match --backend; export .onnx models before ONNX evaluation')
    if not 1 <= args.threads <= 16:
        parser.error('Threads must be between 1 and 16')
    if args.reference_checkpoint is None:
        args.reference_checkpoint = (ROOT / 'models/local_play/onnx/azalea/reference.onnx' if args.backend == 'onnx'
                                     else ROOT.parent / 'azalea/models/hex11-20180712-3362.policy.pth')
    if (args.reference_checkpoint.suffix.lower() == '.onnx') != (args.backend == 'onnx'):
        parser.error('Both models must use the selected inference backend')
    if min(args.games_per_seat, args.simulations, args.workers) < 1:
        parser.error('Counts must be positive')
    if not 0 <= args.seed <= 2 ** 32 - args.games_per_seat:
        parser.error('Seeds must fit in uint32')
    if args.openings:
        args.openings = args.openings.resolve()
        openings = json.loads(args.openings.read_text())
        if len(openings) != args.games_per_seat:
            parser.error('Opening count must equal games-per-seat')
        if len({opening['id'] for opening in openings}) != len(openings):
            parser.error('Opening IDs must be unique')
        for opening in openings:
            actions = opening['actions']
            if not isinstance(actions, list) or any(type(a) is not int or a < 0 for a in actions):
                parser.error('Opening actions must be nonnegative integers')
            if len(set(actions)) != len(actions):
                parser.error('Opening actions must be distinct')
    else:
        openings = [dict(id='empty', actions=[]) for _ in range(args.games_per_seat)]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoint', 'reference_source', 'reference_checkpoint'):
        setattr(args, name, getattr(args, name).resolve())
    model_hashes = {name: digest(getattr(args, name))
                    for name in ('checkpoint', 'reference_checkpoint')}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), settings=settings,
                    sha256=model_hashes, device='cpu', backend=args.backend, threads_per_worker=args.threads,
                    openings=openings, move_selection='greedy, no root noise',
                    trained_search=dict(batch_size=32, cpuct=0.5, reuse_tree=True,
                                        tactical_check=False, heuristic_prior_weight=0),
                    seed_pairing='Same base seed for both seats; Azalea uses base seed + 1',
                    rules='Red first connects top-bottom; blue second connects left-right; no swap rule',
                    opening_coordinates='Actions are authoritative internal row-major indices; legacy labels may use a top-left origin',
                    caution='Greedy games may repeat across seeds. Search implementations differ.')
    save_json(args.output / 'run.json', manifest)
    tasks = [(2 * i + seat + 1, args.seed + i, side, openings[i])
             for i in range(args.games_per_seat) for seat, side in enumerate((1, -1))]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,
                             initargs=(str(args.checkpoint), str(args.reference_source),
                                       str(args.reference_checkpoint), str(args.output),
                                       args.simulations, args.backend, args.threads)) as pool:
        futures = [pool.submit(play, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda r: r['game'])
            progress = dict(status='running', summary=summarize(results), games=results)
            save_json(args.output / 'results.json', progress)
            print(json.dumps(dict(completed=len(results), **result), ensure_ascii=False), flush=True)
    for name, expected in model_hashes.items():
        assert digest(getattr(args, name)) == expected, f'{name} changed during evaluation'
    final = dict(status='complete', summary=summarize(results), games=results,
                 elapsed_seconds=round(time.perf_counter() - started, 3),
                 validation='Every move checked for terminal state by independent BFS')
    save_json(args.output / 'results.json', final)
    print(json.dumps(final['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
