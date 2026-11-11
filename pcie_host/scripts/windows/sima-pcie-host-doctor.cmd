@echo off
setlocal EnableExtensions

call "%~dp0sima-pcie-host-env.cmd"
if errorlevel 1 exit /b %errorlevel%

where.exe ssh.exe >nul 2>nul
if errorlevel 1 (
  1>&2 echo Neat PCIe host: Windows OpenSSH client was not found.
  exit /b 10
)

sc.exe query SiMaMlaDaemon | findstr.exe /C:"RUNNING" >nul
if errorlevel 1 (
  1>&2 echo Neat PCIe host: SiMaMlaDaemon is not running.
  exit /b 11
)

"%GSTREAMER_ROOT_X86_64%\bin\gst-inspect-1.0.exe" "%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0\gstneatpciehost.dll" >nul
if errorlevel 1 (
  1>&2 echo Neat PCIe host: the installed neatpciehost GStreamer plug-in did not load.
  exit /b 12
)

echo Neat PCIe host prerequisites: PASS
echo Install root: %SIMA_PCIE_HOST_ROOT%
echo GStreamer root: %GSTREAMER_ROOT_X86_64%
echo Platform runtime: %SIMA_WINDOWS_PLATFORM_RUNTIME_DIR%
exit /b 0
