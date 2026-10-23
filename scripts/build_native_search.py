"""Build portable x86-64 search; no fast-math, AVX requirement or runtime compiler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', action='store_true')
    parser.add_argument('--compiler')
    parser.add_argument('--output', type=Path, default=ROOT / 'hex/native')
    args = parser.parse_args()
    compiler = args.compiler or ('x86_64-w64-mingw32-g++-win32' if args.windows else 'g++')
    source = ROOT / 'hex/native/search.cpp'
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / ('hex_search.dll' if args.windows else 'libhex_search.so')
    command = [compiler, '-std=c++17', '-O3', '-DNDEBUG', '-fno-fast-math', '-ffp-contract=off',
               '-march=x86-64', '-mtune=generic', '-shared', '-Wall', '-Wextra', '-Werror']
    command += ['-static', '-static-libgcc', '-static-libstdc++'] if args.windows else ['-fPIC', '-fvisibility=hidden']
    command += [str(source), '-o', str(target)]
    subprocess.run(command, check=True)
    report = dict(command=command, compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size,
                  abi=2, precision='FP64 tree statistics; FP32 model inference unchanged',
                  required_isa='x86-64 baseline (SSE2)', windows=args.windows)
    target.with_suffix(target.suffix+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
