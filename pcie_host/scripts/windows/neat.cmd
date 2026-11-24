@echo off
setlocal EnableExtensions

rem "neat status" on a Windows PCIe host: as on Linux, collect the host and card
rem inventory with the installed helper, then render it. No goto: cmd.exe label
rem search is unreliable in LF-only batch files.
rem Take this script's directory before any shift: shift also moves the script
rem name, which then resolves against the current directory.
set "NEAT_BIN=%~dp0"
if /i "%~1"=="status" shift
set "NEAT_JSON="
if "%~1"=="--json" (
  set "NEAT_JSON=1"
  shift
)
if not "%~1"=="" (
  1>&2 echo Usage: neat [status] [--json]
  exit /b 64
)

set "NEAT_HELPER=%NEAT_BIN%..\share\sima-pcie-host\neat_pcie_status.py"
if not exist "%NEAT_HELPER%" (
  1>&2 echo neat: PCIe status helper is missing; reinstall sima-pcie-host
  exit /b 1
)
set "NEAT_PYTHON=%USERPROFILE%\pyneatpcie\Scripts\python.exe"
rem Find py.exe and python.exe on PATH only: cmd.exe otherwise searches the current
rem directory first. Local to this script (setlocal).
set "NoDefaultCurrentDirectoryInExePath=1"
set "NEAT_PYTHON_ARGS="
if not exist "%NEAT_PYTHON%" (
  set "NEAT_PYTHON=python.exe"
  py.exe -3.14 -V >nul 2>nul && set "NEAT_PYTHON=py.exe" && set "NEAT_PYTHON_ARGS=-3.14"
)

if defined NEAT_JSON (
  "%NEAT_PYTHON%" %NEAT_PYTHON_ARGS% -B "%NEAT_HELPER%"
  exit /b
)
set "NEAT_STATUS_FILE=%TEMP%\neat-pcie-status-%RANDOM%%RANDOM%.json"
"%NEAT_PYTHON%" %NEAT_PYTHON_ARGS% -B "%NEAT_HELPER%" > "%NEAT_STATUS_FILE%"
if "%errorlevel%"=="0" "%NEAT_PYTHON%" %NEAT_PYTHON_ARGS% -B "%NEAT_HELPER%" --render < "%NEAT_STATUS_FILE%"
set "NEAT_STATUS=%errorlevel%"
del /q "%NEAT_STATUS_FILE%" >nul 2>nul
exit /b %NEAT_STATUS%
