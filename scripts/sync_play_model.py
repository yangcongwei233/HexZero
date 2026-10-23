#!/usr/bin/env python3
"""Download only inference weights from the latest completed server checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Machine-specific SSH settings stay outside the public source repository.
    config_path = ROOT / 'local_config/server.json'
    try:
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        if not isinstance(config, dict):
            raise ValueError('expected a JSON object')
    except (OSError, ValueError) as exc:
        parser.error(f'Invalid local server config: {exc}')
    parser.add_argument('--server', default=config.get('server'),
                        required=not config.get('server'))
    parser.add_argument('--port', type=int, default=config.get('port', 22))
    parser.add_argument('--run-dir', default=config.get('run_dir'),
                        required=not config.get('run_dir'))
    parser.add_argument('--remote-python', default=config.get('remote_python', 'python3'))
    parser.add_argument('--output', type=Path, default=ROOT / 'models/local_play/trained.pth.tar')
    parser.add_argument('--require-teacher', action='store_true',
                        help='Refuse to export a checkpoint without configured teacher mixing')
    args = parser.parse_args()
    # Read the completed snapshot selected by the trainer's atomic manifest.
    # Transfer through stdout: no extra checkpoint or replay written remotely.
    script = '''import hashlib,io,json,sys,torch
from pathlib import Path
folder=Path(sys.argv[1])/'checkpoints'
manifest=json.loads((folder/'latest.json').read_text())
data=(folder/manifest['model']).read_bytes()
state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
if int(sys.argv[2]) and not state.get('metadata',{}).get('run_config',{}).get('teacherFraction',0)>0:
    raise ValueError('Latest checkpoint does not contain teacher mixing metadata')
keep={k:state[k] for k in ('format_version','board_size','canonical_version','network_config','state_dict') if k in state}
keep['metadata']=dict(state.get('metadata',{}), iteration=manifest['iteration'],
    source_checkpoint=manifest['model'],source_run=sys.argv[1],
    source_checkpoint_sha256=hashlib.sha256(data).hexdigest(),train_step=state.get('train_step'))
output=io.BytesIO();torch.save(keep,output);sys.stdout.buffer.write(output.getvalue())
'''
    command = ' '.join(shlex.quote(s) for s in ['env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                                                'CUDA_VISIBLE_DEVICES=', args.remote_python, '-', args.run_dir,
                                                str(int(args.require_teacher))])
    proc = subprocess.run(['ssh', '-T', '-p', str(args.port), '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=10', args.server, command], input=script.encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors='replace'))
    import torch
    payload = torch.load(io.BytesIO(proc.stdout), map_location='cpu', weights_only=True)
    if 'state_dict' not in payload or payload.get('canonical_version') != 1:
        raise ValueError('Downloaded checkpoint format is invalid')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_bytes(proc.stdout)
    temporary.replace(args.output)
    manifest = dict(payload['metadata'], downloaded_at=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(proc.stdout).hexdigest(), bytes=len(proc.stdout))
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"已导出第 {manifest['iteration']} 轮权重：{args.output} ({len(proc.stdout)/1024**2:.2f} MiB)")
    print('重启本地对弈服务以加载新权重。')


if __name__ == '__main__':
    main()
