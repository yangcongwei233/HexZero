#!/usr/bin/env python3
"""Finish an active training segment, then resume its complete snapshot to a target iteration."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == 'Z' else f'{pid}:{fields[19]}'


def resume_command(original, folder, snapshot, target):
    # In particular, omit --seed: main_hex must restore the checkpoint RNG state.
    pairs = {'--num-iters', '--start-iter', '--checkpoint-file', '--examples-file',
             '--checkpoint-dir', '--seed', '--preset', '--epochs', '--batch-size', '--learning-rate'}
    flags = {'--load-model', '--fresh-replay', '--skip-first-self-play', '--allow-partial-checkpoint'}
    result = []
    index = 0
    while index < len(original):
        token = original[index]
        name = token.split('=', 1)[0]
        if name in pairs:
            index += 1 if '=' in token else 2
        elif name in flags:
            index += 1
        else:
            result.append(token)
            index += 1
    completed = int(snapshot['iteration'])
    if target <= completed:
        raise ValueError('Target must be greater than the completed iteration')
    result.extend(['--load-model', '--checkpoint-dir', str(folder),
                   '--checkpoint-file', snapshot['model'], '--examples-file', snapshot['examples'],
                   '--num-iters', str(target - completed)])
    return result


def teacher_options(manifest_path, source_dir, fraction):
    """Accept only an atomically published, verified complete teacher dataset."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or not manifest.get('sha256'):
        raise ValueError('Teacher collection is not complete')
    sys.path.insert(0, str(source_dir))
    from hex.teacher_data import TeacherDataset
    dataset = TeacherDataset(manifest_path.parent / 'teacher.npz', manifest['board_size'], manifest['sha256'])
    return ['--teacher-data', dataset.path, '--teacher-fraction', str(fraction)], dict(
        path=dataset.path, sha256=dataset.sha256, samples=len(dataset), fraction=fraction,
        games=manifest['games'])


def preserve_snapshot(run, folder, snapshot):
    destination = run / 'milestones' / f'iteration_{snapshot["iteration"]}_before_teacher'
    destination.mkdir(parents=True, exist_ok=True)
    for key in ('model', 'examples'):
        source, target = folder / snapshot[key], destination / snapshot[key]
        if target.exists():
            if not os.path.samefile(source, target):
                raise RuntimeError('A different pre-teacher milestone already exists')
        else:
            os.link(source, target)
    atomic_json(destination / 'manifest.json', {**snapshot, 'preserved_at_utc': now(),
                                               'reason': 'Last complete pure self-play checkpoint before teacher mixing'})
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--segment-end', type=int, required=True)
    parser.add_argument('--target-iteration', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--resume-source-dir', type=Path, help='Verified source tree for the next segment only.')
    parser.add_argument('--teacher-manifest', type=Path, help='Wait for this completed teacher collection before resuming.')
    parser.add_argument('--teacher-fraction', type=float, default=.1)
    args = parser.parse_args()
    if args.target_iteration <= args.segment_end or not 0 < args.poll_seconds <= 60:
        parser.error('Target must exceed segment end; polling must be in (0, 60] seconds')
    if not 0 < args.teacher_fraction < 1:
        parser.error('Teacher fraction must be in (0, 1)')
    run = args.run_dir.resolve()
    status_path = run / 'continuation.json'
    with (run / '.continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = json.loads((run / 'run.json').read_text())
        source_dir = (args.resume_source_dir or Path(initial['source_dir'])).resolve()
        if not (source_dir / 'main_hex.py').is_file():
            raise ValueError('Resume source directory is missing main_hex.py')
        identity = process_identity(args.trainer_pid)
        state = dict(status='waiting', supervisor_pid=os.getpid(), created_at_utc=now(),
                     original_trainer_pid=args.trainer_pid, original_process_identity=identity,
                     current_segment_end=args.segment_end, target_iteration=args.target_iteration,
                     next_start_iteration=args.segment_end + 1,
                     remaining_iterations=args.target_iteration - args.segment_end,
                     training_log=str(run / 'train.log'), resume_source_dir=str(source_dir),
                     teacher_manifest=str(args.teacher_manifest.resolve()) if args.teacher_manifest else None,
                     teacher_fraction=args.teacher_fraction if args.teacher_manifest else 0.)
        if not args.teacher_manifest and initial.get('teacher_guidance'):
            state.update(teacher=initial['teacher_guidance'],
                         teacher_fraction=initial['teacher_guidance']['fraction'])
        if identity is None:
            raise RuntimeError('Original trainer is no longer running; inspect its final checkpoint first')
        atomic_json(status_path, state)
        try:
            while process_identity(args.trainer_pid) == identity:
                time.sleep(args.poll_seconds)
            folder = run / 'checkpoints'
            snapshot = json.loads((folder / 'latest.json').read_text())
            if snapshot['iteration'] != args.segment_end:
                raise RuntimeError(f'Original trainer ended at saved iteration {snapshot["iteration"]}, '
                                   f'expected {args.segment_end}; refusing an unverified restart')
            for key in ('model', 'examples'):
                if Path(snapshot[key]).name != snapshot[key] or not (folder / snapshot[key]).is_file():
                    raise RuntimeError('Complete model/replay snapshot pair is missing')
            # Later extensions must inherit the current segment, including any
            # teacher mixture added after the original launch command.
            command = resume_command(initial.get('active_command', initial['command']),
                                     folder, snapshot, args.target_iteration)
            if args.teacher_manifest:
                state.update(status='waiting_teacher_data', segment_completed_at_utc=now())
                atomic_json(status_path, state)
                while not args.teacher_manifest.is_file():
                    progress_path = args.teacher_manifest.parent / 'progress.json'
                    if progress_path.is_file() and json.loads(progress_path.read_text()).get('status') == 'failed':
                        raise RuntimeError('Teacher collection failed; refusing to resume with unverified data')
                    time.sleep(args.poll_seconds)
                extra, teacher = teacher_options(args.teacher_manifest, source_dir, args.teacher_fraction)
                command.extend(extra)
                state.update(teacher=teacher, pre_teacher_milestone=preserve_snapshot(run, folder, snapshot))
            env = os.environ.copy()
            env.pop('HEX_FORCE_CPU', None)
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
            with (run / 'train.log').open('ab', buffering=0) as log:
                log.write(f'\nINFO:continuation:Resuming after iteration {args.segment_end} '
                          f'to target {args.target_iteration}\n'.encode())
                trainer = subprocess.Popen(command, cwd=source_dir, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            state.update(status='running', resumed_at_utc=now(), trainer_pid=trainer.pid,
                         resume_checkpoint=snapshot['model'], command=command)
            atomic_json(status_path, state)
            current = json.loads((run / 'run.json').read_text())
            current.update(trainer_pid=trainer.pid, active_command=command,
                           target_iteration=args.target_iteration, continuation_state=str(status_path))
            current.setdefault('original_source_dir', initial['source_dir'])
            current['source_dir'] = str(source_dir)
            source_manifest = source_dir / 'source_sha256.json'
            if source_manifest.is_file():
                current.setdefault('original_source_sha256', current.get('source_sha256', {}))
                current['source_sha256'] = json.loads(source_manifest.read_text())
            if args.teacher_manifest:
                current.update(teacher_guidance=state['teacher'],
                               pre_teacher_milestone=state['pre_teacher_milestone'])
            atomic_json(run / 'run.json', current)
            code = trainer.wait()
            final = json.loads((folder / 'latest.json').read_text())
            if code != 0 or final['iteration'] != args.target_iteration:
                raise RuntimeError(f'Resumed trainer exit={code}, latest saved iteration={final["iteration"]}')
            state.update(status='complete', completed_at_utc=now(), final_iteration=final['iteration'])
            atomic_json(status_path, state)
        except BaseException as exc:
            state.update(status='failed', failed_at_utc=now(), error=f'{type(exc).__name__}: {exc}')
            atomic_json(status_path, state)
            raise


if __name__ == '__main__':
    main()
