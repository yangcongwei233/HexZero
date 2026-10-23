#!/usr/bin/env python3
"""使用相同带颜色开局并交换先后手，比较两个 HexZero 断点。"""
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
APPS = None
SESSION = None
SEARCH = dict(batch_size=32, cpuct=0.5, reuse_tree=True, temperature=0,
              root_noise=False, tactical_check=False, heuristic_prior_weight=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def initialize(checkpoints, hashes, output, simulations, backend, threads):
    global APPS, SESSION
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HEX_FORCE_CPU'] = '1'
    sys.path.insert(0, str(ROOT))
    if backend == 'torch':
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    from hex_play_server import HexApp, HexSession
    from hex.local_players import TrainedPlayer

    for key, path in checkpoints.items():
        assert digest(path) == hashes[key], 'Checkpoint changed before worker initialization'
    APPS = {key: HexApp(path, Path(output) / 'games', device='cpu', simulations=simulations,
                      backend=backend, threads=threads)
            for key, path in checkpoints.items()}
    assert APPS['model_a'].game.n == APPS['model_b'].game.n, 'Board sizes differ'

    class CheckpointSession(HexSession):
        def make_engines(self):
            return {
                -self.mine: TrainedPlayer(self.game, APPS['model_a'].net, self.simulations, self.seed),
                self.mine: TrainedPlayer(self.game, APPS['model_b'].net, self.simulations,
                                         (self.seed + 1) % (2 ** 32)),
            }

        def kind(self, player):
            return 'model_b' if player == self.mine else 'model_a'

        def record(self):
            models = {key: dict(path=str(app.checkpoint), metadata=app.metadata, sha256=hashes[key])
                      for key, app in APPS.items()}
            return dict(format='hexzero-local-match-v3', matchType='checkpoints', createdAt=self.created_at,
                        models=models, search=SEARCH, inferenceBackend=backend, **self.state())

    SESSION = CheckpointSession


def play(task):
    from hex.HexLogic import has_won_bfs
    number, seed, model_a_side, opening = task
    session = SESSION(APPS['model_a'], mode='checkpoints', mine=-model_a_side,
                      simulations=APPS['model_a'].default_simulations, seed=seed)
    session.record_metadata.update(
        redTeam=f"HexZero {APPS[session.kind(1)].metadata.get('iteration')}",
        blueTeam=f"HexZero {APPS[session.kind(-1)].metadata.get('iteration')}",
        event='模型两两评测')
    started = time.perf_counter()

    def validate():
        red = bool(has_won_bfs(session.board, 1))
        blue = bool(has_won_bfs(session.board, -1))
        assert not (red and blue), 'Both colors cannot win'
        expected = 1 if red else -1 if blue else None
        assert session.winner == expected, 'Winner disagrees with independent BFS'
        assert len(session.moves) <= session.game.n ** 2

    for action in opening['actions']:
        if session.winner is not None:
            raise ValueError('Opening continues after terminal position')
        session.apply(action, dict(source='preset_opening', opening_id=opening['id']))
        validate()
    if session.winner is not None:
        raise ValueError('Opening must leave an unfinished game')
    while session.winner is None:
        session.step(session.revision)
        validate()
    actions = [move['action'] for move in session.moves]
    assert len(set(actions)) == len(actions), 'Repeated occupied move'
    winner = 'model_a' if session.winner == model_a_side else 'model_b'
    assert winner == session.kind(session.winner)
    return dict(game=number, seed=seed, model_a_side=model_a_side, opening=opening,
                winner=winner, winner_side=session.winner, moves=len(actions),
                seconds=round(time.perf_counter() - started, 3), record=session.saved_path,
                trajectory_sha256=hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
                iterations={key: app.metadata.get('iteration') for key, app in APPS.items()},
                inference_backend=APPS['model_a'].backend)


def summarize(games):
    def score(subset):
        a_wins = sum(g['winner'] == 'model_a' for g in subset)
        return dict(games=len(subset), model_a_wins=a_wins, model_b_wins=len(subset)-a_wins,
                    model_a_win_rate=a_wins/len(subset) if subset else None)
    seats = {name: score([g for g in games if g['model_a_side'] == side])
             for side, name in ((1, 'model_a_first_red'), (-1, 'model_a_second_blue'))}
    return dict(**score(games), by_seat=seats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-a', type=Path, required=True)
    parser.add_argument('--model-b', type=Path, required=True)
    parser.add_argument('--openings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulations', type=int, default=800)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backend', choices=('auto', 'torch', 'onnx'), default='auto')
    parser.add_argument('--threads', type=int, default=1, help='Inference threads per worker (1-16)')
    args = parser.parse_args()
    if args.backend == 'auto':
        args.backend = 'onnx' if args.model_a.suffix.lower() == '.onnx' else 'torch'
    if any((p.suffix.lower() == '.onnx') != (args.backend == 'onnx') for p in (args.model_a, args.model_b)):
        parser.error('Both model files must match the selected inference backend')
    if not 1 <= args.threads <= 16:
        parser.error('Threads must be between 1 and 16')
    if min(args.simulations, args.workers) < 1:
        parser.error('Search budget and worker count must be positive')
    for key in ('model_a', 'model_b', 'openings', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    openings = json.loads(args.openings.read_text())
    if not openings or len({o['id'] for o in openings}) != len(openings):
        parser.error('Provide at least one opening with unique IDs')
    if not 0 <= args.seed <= 2 ** 32 - len(openings) - 1:
        parser.error('Seeds, including the model B offset, must fit uint32')
    for opening in openings:
        actions = opening['actions']
        if (not isinstance(actions, list) or any(type(a) is not int or a < 0 for a in actions)
                or len(set(actions)) != len(actions)):
            parser.error('Opening actions must be distinct nonnegative integers')
    checkpoints = dict(model_a=str(args.model_a), model_b=str(args.model_b))
    hashes = {key: digest(path) for key, path in checkpoints.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), settings=settings,
                    sha256=hashes, openings=openings, search_both_models=SEARCH,
                    device='cpu', backend=args.backend, threads_per_worker=args.threads,
                    seed_pairing='Same base seed for both seats; model A uses base seed, B uses base seed + 1',
                    rules='Red first connects top-bottom; blue second connects left-right; no swap rule',
                    caution='Fixed opening suite; openings are not proven balanced.')
    save_json(args.output / 'run.json', manifest)
    tasks = [(2*i+seat+1, args.seed+i, side, opening)
             for i, opening in enumerate(openings) for seat, side in enumerate((1, -1))]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,
                             initargs=(checkpoints, hashes, str(args.output), args.simulations,
                                       args.backend, args.threads)) as pool:
        futures = [pool.submit(play, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda g: g['game'])
            save_json(args.output / 'results.json',
                      dict(status='running', summary=summarize(results), games=results))
            print(json.dumps(dict(completed=len(results), **result), ensure_ascii=False), flush=True)
    for key, path in checkpoints.items():
        assert digest(path) == hashes[key], 'Checkpoint changed during evaluation'
    paired_outcomes = Counter()
    for opening in openings:
        pair = [g for g in results if g['opening']['id'] == opening['id']]
        assert len(pair) == 2 and {g['model_a_side'] for g in pair} == {1, -1}
        assert pair[0]['opening'] == pair[1]['opening'] and pair[0]['seed'] == pair[1]['seed']
        wins = sum(g['winner'] == 'model_a' for g in pair)
        paired_outcomes[('model_b_wins_both', 'split', 'model_a_wins_both')[wins]] += 1
    final = dict(status='complete', summary=summarize(results), games=results,
                 elapsed_seconds=round(time.perf_counter()-started, 3),
                 paired_opening_outcomes=dict(paired_outcomes),
                 unique_trajectories=len({g['trajectory_sha256'] for g in results}),
                 validation='Every move checked by independent BFS; paired openings/seeds and immutable weights verified')
    save_json(args.output / 'results.json', final)
    print(json.dumps(final['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
