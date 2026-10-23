"""构建 Windows 桌面包：public 无模型，competition 固定 3250；输出目录须不存在。
输入为已有 Windows 运行库包和 Python 3.11.9 官方 tcltk.msi 解压目录。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """计算文件 SHA-256，供软件包完整性验证使用。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_sources(source, target):
    """源码保留原许可文件；编译缓存、自动测试和临时文件不进入产品包。"""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        '__pycache__', 'test', 'tests', 'test_*.py', '*.pyc', '*.nbc', '*.nbi', '*.tmp', '*.bak'))


def main():
    """从基础离线包和 Tk 运行库组装桌面包，按公开或比赛模式处理模型及配置，生成 ZIP 和校验文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tk-runtime', type=Path, required=True)
    parser.add_argument('--edition', choices=('public', 'competition'), default='public')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, tk_runtime = args.output.resolve(), args.tk_runtime.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    for name in ('DLLs/_tkinter.pyd', 'DLLs/tcl86t.dll', 'DLLs/tk86t.dll',
                 'Lib/tkinter/__init__.py', 'tcl/tcl8.6/init.tcl', 'tcl/tk8.6/tk.tcl'):
        if not (tk_runtime / name).is_file():
            raise FileNotFoundError(f'Incomplete Tcl/Tk runtime: {name}')
    output.mkdir(parents=True)
    with zipfile.ZipFile(args.base) as base:
        for info in base.infolist():
            relative = PurePosixPath(*PurePosixPath(info.filename).parts[1:])
            if info.is_dir() or relative.is_absolute() or '..' in relative.parts:
                continue
            name = relative.as_posix()
            # ORT 自带的 iris/sigmoid 示例数据不用于真实推理，公开包也无需携带。
            if name.startswith('runtime/Lib/site-packages/onnxruntime/datasets/'):
                continue
            if not (name.startswith('runtime/') or
                    (args.edition == 'competition' and name.startswith('models/iteration_3250/'))):
                continue
            target = output / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base.read(info))
    runtime, app = output / 'runtime', output / 'app'
    # Windows 嵌入式 Python 的 ._pth 是显式路径白名单，须加入 DLLs 以找到 _tkinter。
    for source, target in ((tk_runtime/'DLLs', runtime/'DLLs'),
                           (tk_runtime/'Lib/tkinter', runtime/'Lib/tkinter'),
                           (tk_runtime/'tcl', runtime/'tcl')):
        copy_sources(source, target)
    (runtime/'python311._pth').write_text('DLLs\nLib\npython311.zip\n.\nLib/site-packages\n../app\nimport site\n')
    app.mkdir()
    for name in ('hex_desktop.py', 'hex_play_core.py', 'Game.py', 'NeuralNet.py', 'MCTS.py', 'utils.py', 'LICENSE'):
        shutil.copy2(ROOT / name, app / name)
    copy_sources(ROOT/'hex', app/'hex')
    if not (app/'hex/native/hex_search.dll').is_file():
        raise FileNotFoundError('Build the Windows C++ search DLL before packaging')
    # Windows 包不附带 Linux 动态库，但完整保存 Azalea 的许可和修改说明。
    for path in (app/'hex/native').glob('*.so*'):
        path.unlink()
    launcher = (ROOT/'portable/desktop_launcher.py').read_text(encoding='utf-8')
    if args.edition == 'competition':
        launcher = launcher.replace('COMPETITION = False', 'COMPETITION = True')
    (output/'launcher.py').write_text(launcher, encoding='utf-8')
    (output/'models').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'portable/README-Desktop.txt', output/'README-Desktop.txt')
    shutil.copy2(ROOT/'portable/README-models.txt', output/'models/README.txt')
    for name, options in (('START_HEX.bat', ''), ('CHECK_ENV.bat', '--check-environment')):
        content = ('@echo off\nsetlocal\nchcp 65001 >nul\ncd /d "%~dp0"\n'
                   'set "PYTHONUTF8=1"\n'
                   'if not exist "%~dp0runtime\\python.exe" (\n'
                   '  echo Please extract the complete package before starting.\n  pause\n  exit /b 1\n)\n'
                   '"%~dp0runtime\\python.exe" -X utf8 "%~dp0launcher.py" '+options+'\n'
                   'if errorlevel 1 (\n  echo See userdata\\logs for details.\n  pause\n)\nendlocal\n')
        (output/name).write_bytes(content.replace('\n','\r\n').encode('utf-8'))
    (output/'START_HEX_PYTHON.bat').write_bytes(
        b'@echo off\r\nsetlocal\r\nset "HEX_SEARCH_BACKEND=python"\r\ncall "%~dp0START_HEX.bat"\r\nendlocal\r\n')
    if args.edition == 'competition':
        # 不能把任意名为 iteration_3250 的目录当成选定模型；检查实际内容及轮次。
        import sys
        sys.path.insert(0, str(ROOT))
        from hex.competition_profile import MODEL_ID, MODEL_SHA256
        model = output/'models'/MODEL_ID
        header = json.loads(Path(str(model)+'.json').read_text(encoding='utf-8'))
        if digest(model) != MODEL_SHA256 or header.get('onnx_sha256') != MODEL_SHA256 or header.get('metadata', {}).get('iteration') != 3250:
            raise ValueError('Competition model does not match the fixed 3250 weights')
    manifest = dict(release='HexZero native desktop', edition=args.edition, release_date='2026-10-23',
                    ui='Tk native desktop; no browser or HTTP server',
                    platform='Windows 10/11 x64', python='3.11.9', precision='FP32',
                    bundled_models=[3250] if args.edition == 'competition' else [],
                    model_locked=args.edition == 'competition', azalea_bundled=False,
                    default_simulations=12800, default_threads=4,
                    selectable_threads=list(range(4,17)), simulations=[800,1200,2000,3200,6400,12800,51200],
                    search_dll_sha256=digest(app/'hex/native/hex_search.dll'),
                    base_archive_sha256=digest(args.base),
                    tcltk_source='https://www.python.org/ftp/python/3.11.9/amd64/tcltk.msi',
                    tcltk_files={name:digest(tk_runtime/name) for name in ('DLLs/_tkinter.pyd','DLLs/tcl86t.dll','DLLs/tk86t.dll')},
                    validation='See local release validation report; Windows target execution not yet performed')
    (output/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipout:
        for path in sorted(p for p in output.rglob('*') if p.is_file()):
            zipout.write(path,path.relative_to(output.parent))
    archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive)),indent=2))


if __name__ == '__main__':
    main()
