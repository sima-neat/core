@echo off
setlocal EnableExtensions

call "%~dp0sima-pcie-host-env.cmd"
if not "%errorlevel%"=="0" exit /b %errorlevel%
if "%~1"=="" (
  1>&2 echo Usage: sima-pcie-host-run command [arguments...]
  exit /b 64
)

%*
exit /b %errorlevel%
