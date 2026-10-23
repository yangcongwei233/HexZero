"""Windows 桌面启动器：配置内置 Tcl/Tk，保存子进程日志和异常退出码。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
# 打包脚本只为比赛包将此值写为 True；命令行和旧 settings.json 不能解除模型锁定。
COMPETITION = False
RUNTIME = ROOT / 'runtime'
sys.path.insert(0, str(ROOT / 'app'))
os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HEX_FORCE_CPU='1', CUDA_VISIBLE_DEVICES='')
os.environ['TCL_LIBRARY'] = str(RUNTIME / 'tcl/tcl8.6')
os.environ['TK_LIBRARY'] = str(RUNTIME / 'tcl/tk8.6')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'userdata/cache/numba')
# add_dll_directory 返回的句柄必须保留，垃圾回收会撤销该搜索目录。
_dll_directories = []
if sys.platform == 'win32':
    for folder in (RUNTIME, RUNTIME / 'DLLs'):
        _dll_directories.append(os.add_dll_directory(str(folder)))


def main():
    """设置便携运行目录和本地日志，识别公开版或比赛版配置后启动桌面程序，异常时保留可诊断信息。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--check-environment', action='store_true')
    args, options = parser.parse_known_args()
    logs = ROOT / 'userdata/logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.check_environment:
            import tkinter as tk
            from hex.desktop_service import DesktopService
            window = tk.Tk()
            window.withdraw()
            try:
                service = DesktopService(ROOT / 'models', ROOT / 'userdata', competition=COMPETITION)
                service.perform('close')
                print('环境检查通过：Tk 窗口与对弈功能；已加载模型时包含真实推理检查。', flush=True)
            finally:
                window.destroy()
        else:
            from hex_desktop import main as desktop_main
            desktop_main(['--models', str(ROOT / 'models'), '--data-dir', str(ROOT / 'userdata'),
                          *options, *(['--competition'] if COMPETITION else [])])
        return
    command = [sys.executable, '-u', '-X', 'utf8', '-X', 'faulthandler', str(Path(__file__).resolve()), '--worker']
    if args.check_environment:
        command.append('--check-environment')
    command.extend(options)
    path = logs / ('desktop-' + time.strftime('%Y%m%d-%H%M%S') + '.log')
    print('正在打开 HexZero 桌面窗口；首次启动会检查模型并预热搜索。', flush=True)
    with path.open('w', encoding='utf-8') as output:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        report = dict(returncode=code, hex_code=f'0x{code & 0xffffffff:08X}', log=str(path))
        (logs / 'exit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'启动或推理失败：{report["hex_code"]}。日志：{path}\n可运行 START_HEX_PYTHON.bat 使用兼容搜索。', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
