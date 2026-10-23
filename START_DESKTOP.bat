@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
py -3 hex_desktop.py %*
if errorlevel 1 (
  echo.
  echo Please install Python 3.11+ with Tcl/Tk, then run:
  echo py -3 -m pip install -r requirements-desktop.txt
  pause
)
endlocal
