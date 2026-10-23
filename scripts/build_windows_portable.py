#!/usr/bin/env python3
"""用下载的 CPython 和 Windows 依赖构建离线 x64 软件包。"""
import argparse
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    """计算文件 SHA-256，供离线包校验清单使用。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(source, target):
    """复制运行所需目录，排除解释器/编译缓存及开发测试文件。"""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi', 'test_*.py'))


def main():
    """组装嵌入式 Python、Windows 依赖、源码、模型及启动器，生成离线 ZIP 和完整性清单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'dist/windows-build-cache')
    parser.add_argument('--wheels', type=Path, default=ROOT / 'dist/windows-onnx-wheels')
    parser.add_argument('--models', type=Path, default=ROOT / 'models/local_play/onnx')
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/HexLocal-Windows-x64')
    args = parser.parse_args()
    cache, output = args.cache.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Build output already exists: {output}')
    wheels = sorted(args.wheels.glob('*.whl'))
    metadata = {}
    for wheel in wheels:
        if not (wheel.name.endswith('win_amd64.whl') or wheel.name.endswith('none-any.whl')):
            raise ValueError(f'Not a Windows x64/universal wheel: {wheel}')
        with zipfile.ZipFile(wheel) as archive:
            text = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        info = Parser().parsestr(text)
        key = canonicalize_name(info['Name'])
        if key in metadata:
            raise ValueError(f'Duplicate wheel: {key}')
        metadata[key] = dict(name=info['Name'], version=info['Version'], wheel=wheel.name,
                             sha256=sha(wheel), requires=info.get_all('Requires-Dist', []))
    environment = dict(default_environment(), os_name='nt', sys_platform='win32',
                       platform_system='Windows', platform_machine='AMD64',
                       python_version='3.11', python_full_version='3.11.9', extra='')
    for info in metadata.values():
        for spec in info['requires']:
            req = Requirement(spec)
            if req.marker and not req.marker.evaluate(environment):
                continue
            installed = metadata.get(canonicalize_name(req.name))
            if installed is None or Version(installed['version']) not in req.specifier:
                raise ValueError(f'Missing/incompatible Windows dependency: {info["name"]} -> {req}')
    for name in ('onnxruntime', 'numpy', 'numba', 'typing-extensions'):
        assert name in metadata
    if 'torch' in metadata:
        raise ValueError('The optimized portable bundle must not include PyTorch')

    app, runtime = output / 'app', output / 'runtime'
    app.mkdir(parents=True)
    runtime.mkdir()
    embedded = cache / 'python-3.11.9-embed-amd64.zip'
    with zipfile.ZipFile(embedded) as archive:
        archive.extractall(runtime)
    msvc = cache / 'msvc-runtime'
    msvc_metadata = json.loads((msvc / 'manifest.json').read_text())
    for name in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'):
        assert (msvc / name).is_file(), f'Missing Microsoft runtime: {name}'
    for path in msvc.iterdir():
        if path.suffix in ('.dll', '.rtf'):
            shutil.copy2(path, runtime / path.name)
    # 保留标准库源码，供运行时需要检查源码的依赖使用。
    source = cache / 'Python-3.11.9.tar.xz'
    with tarfile.open(source) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith('Python-3.11.9/Lib/'):
                continue
            relative = PurePosixPath(member.name).relative_to('Python-3.11.9/Lib')
            if '..' in relative.parts or any(p in ('test', 'tests', 'idlelib', 'tkinter', 'ensurepip', '__pycache__') for p in relative.parts):
                continue
            target = runtime / 'Lib' / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    site = runtime / 'Lib/site-packages'
    site.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if name.endswith('/'):
                    continue
                # 可选的 TBB 亲和性插件另需 hwloc，
                # 本程序未使用这些插件或并行 Numba。
                if wheel.name.startswith('tbb-') and path.name.startswith('tbbbind') and path.suffix == '.dll':
                    continue
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe wheel member')
                target_root, relative = site, path
                if path.parts[0].endswith('.data'):
                    scheme = path.parts[1]
                    relative = PurePosixPath(*path.parts[2:])
                    if scheme in ('purelib', 'platlib'):
                        target_root = site
                    elif scheme == 'data':
                        target_root = runtime
                    elif scheme in ('scripts', 'headers'):
                        target_root = runtime / ('Scripts' if scheme == 'scripts' else 'Include')
                    else:
                        raise ValueError(f'Unsupported wheel layout: {name}')
                target = target_root / Path(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    # Intel 的 Windows 运行库将动态库放在 Library/bin，
    # 显式加入查找路径，无需修改系统 PATH。
    for path in (runtime / 'Library/bin').glob('*.dll'):
        shutil.copy2(path, runtime / path.name)
    (runtime / 'python311._pth').write_text('Lib\npython311.zip\n.\nLib/site-packages\n../app\nimport site\n')

    for name in ('hex_play_server.py', 'hex_play_core.py', 'Game.py', 'NeuralNet.py', 'MCTS.py', 'utils.py', 'LICENSE'):
        shutil.copy2(ROOT / name, app / name)
    copy_tree(ROOT / 'hex', app / 'hex')
    if not (app / 'hex/native/hex_search.dll').is_file():
        raise FileNotFoundError('Build the Windows search DLL with scripts/build_native_search.py --windows first')
    copy_tree(ROOT / 'hex_web', app / 'hex_web')
    reference = ROOT.parent / 'azalea'
    copy_tree(reference / 'azalea', app / 'vendor/azalea/azalea')
    shutil.copy2(reference / 'LICENSE', app / 'vendor/azalea/LICENSE')
    for iteration in (750, 800, 880):
        origin = ROOT / 'models/local_play/archive' / f'iteration_{iteration}'
        target = output / 'models' / f'iteration_{iteration}'
        target.mkdir(parents=True)
        for filename in ('trained.pth.tar', 'trained.pth.tar.json'):
            shutil.copy2(origin / filename, target / filename)
        for filename in ('trained.onnx', 'trained.onnx.json'):
            source_model = args.models / f'iteration_{iteration}' / filename
            shutil.copy2(source_model if source_model.is_file() else origin / filename, target / filename)
    target = output / 'models/azalea'
    target.mkdir()
    shutil.copy2(reference / 'models/hex11-20180712-3362.policy.pth', target)
    for filename in ('reference.onnx', 'reference.onnx.json'):
        shutil.copy2(args.models / 'azalea' / filename, target / filename)
    shutil.copy2(ROOT / 'portable/launcher.py', output / 'launcher.py')
    shutil.copy2(ROOT / 'portable/environment_check.py', output / 'environment_check.py')
    shutil.copy2(ROOT / 'portable/README-Windows.txt', output / 'README-Windows.txt')
    shutil.copy2(ROOT / 'portable/SAMPLE_HEX.txt', output / 'SAMPLE_HEX.txt')
    shutil.copy2(ROOT / 'portable/README-models.txt', output / 'models/README.txt')
    for filename, options in (('START_HEX.bat', ''), ('START_750.bat', '--model 750'),
                              ('START_800.bat', '--model 800'), ('START_880.bat', '--model 880'),
                              ('SELF_TEST.bat', '--self-test'), ('CHECK_ENV.bat', '--check-environment')):
        command = ('@echo off\nchcp 65001 >nul\ncd /d "%~dp0"\nset "PYTHONUTF8=1"\n'
                   'if not exist "%~dp0runtime\\python.exe" (\n'
                   '  echo 未找到运行环境，请将整个 ZIP 解压后再运行。\n  pause\n  exit /b 1\n)\n'
                   '"%~dp0runtime\\python.exe" "%~dp0launcher.py" ' + options +
                   '\nif errorlevel 1 echo 启动失败，请查看上方信息和 logs 文件夹；若无日志，请检查是否完整解压及系统是否为 Windows x64。\necho.\npause\n')
        (output / filename).write_bytes(command.replace('\n', '\r\n').encode('utf-8'))
    fallback = ('@echo off\nsetlocal\nset "HEX_SEARCH_BACKEND=python"\n'
                'call "%~dp0START_HEX.bat"\nendlocal\n')
    (output / 'START_HEX_PYTHON.bat').write_bytes(fallback.replace('\n', '\r\n').encode('utf-8'))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), platform='Windows x64',
                    python='3.11.9', default_model=880, available_models=[750, 800, 880],
                    model_library='models', recursive_model_discovery=True, persisted_model_selection=True,
                    reference_model='hex11-20180712-3362.policy.pth', device='CPU', offline=True,
                    inference='ONNX Runtime FP32 CPU, C++ HexZero batched search, original Azalea search, Python UI', precision='FP32',
                    default_threads='min(4, logical CPU count)', selectable_threads=list(range(4, 17)),
                    automatic_environment_check=True, pytorch_required=False,
                    notation=dict(encoding='GB2312', origin='bottom-left', first='red',
                                  redConnects='top-bottom', blueConnects='left-right', marks=[-2,-1,0,1,2]),
                    sources={'python': 'https://www.python.org/ftp/python/3.11.9/',
                             'onnxruntime': 'https://onnxruntime.ai/docs/install/',
                             'other_wheels': 'https://pypi.org/simple/'},
                    dependencies=list(metadata.values()),
                    microsoft_runtime=msvc_metadata,
                    omitted_optional_components=['TBB hardware-affinity plugins (tbbbind*.dll; require hwloc; unused by Hex)'],
                    python_archive_sha256=sha(embedded), stdlib_source_sha256=sha(source),
                    validation='Windows wheel dependency markers checked; see validation report for execution coverage.')
    (output / 'build-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if args.validation:
        shutil.copy2(args.validation, output / 'validation-report.json')
    # 生成校验清单，便于验证跨机器复制是否完整。
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    zip_path = output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in output.rglob('*') if p.is_file()):
            archive.write(path, path.relative_to(output.parent))
    zip_path.with_suffix('.zip.sha256').write_text(f'{sha(zip_path)}  {zip_path.name}\n')
    print(json.dumps(dict(directory=str(output), archive=str(zip_path), bytes=zip_path.stat().st_size,
                          sha256=sha(zip_path), dependencies=len(metadata)), indent=2))


if __name__ == '__main__':
    main()
