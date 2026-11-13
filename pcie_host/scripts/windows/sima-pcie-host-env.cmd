@echo off
rem Configure the current cmd.exe process for the installed Neat PCIe host SDK.
rem This script does not modify machine or user environment variables.

for %%I in ("%~dp0..") do set "SIMA_PCIE_HOST_ROOT=%%~fI"

if not defined GSTREAMER_ROOT_X86_64 (
  set "GSTREAMER_ROOT_X86_64=%ProgramFiles%\gstreamer\1.0\msvc_x86_64"
)
if not exist "%GSTREAMER_ROOT_X86_64%\bin\gstreamer-1.0-0.dll" (
  1>&2 echo Neat PCIe host: GStreamer MSVC x64 was not found at "%GSTREAMER_ROOT_X86_64%".
  1>&2 echo Install the supported GStreamer package or set GSTREAMER_ROOT_X86_64.
  exit /b 2
)

if not defined SIMA_WINDOWS_PLATFORM_RUNTIME_DIR (
  rem Do not depend on sc.exe's indentation; it differs between Windows builds.
  rem Expand for-loop variables directly; percent variables are expanded before this block runs.
  for /f "tokens=3" %%I in ('sc.exe qc SiMaMlaDaemon 2^>nul ^| findstr.exe /C:"BINARY_PATH_NAME"') do for %%J in ("%%~I") do set "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR=%%~dpJ"
)
if not exist "%SIMA_WINDOWS_PLATFORM_RUNTIME_DIR%\SiMaPCIe.dll" (
  1>&2 echo Neat PCIe host: the SiMa Platform runtime was not found at "%SIMA_WINDOWS_PLATFORM_RUNTIME_DIR%".
  1>&2 echo Install the supported Platform package or set SIMA_WINDOWS_PLATFORM_RUNTIME_DIR to its runtime directory.
  exit /b 3
)

if not exist "%SIMA_PCIE_HOST_ROOT%\bin\simaai_scheduler_pcie_host.dll" (
  1>&2 echo Neat PCIe host: the Scheduler runtime is missing from "%SIMA_PCIE_HOST_ROOT%\bin".
  exit /b 4
)
if not exist "%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0\gstneatpciehost.dll" (
  1>&2 echo Neat PCIe host: the Scheduler GStreamer plug-in is missing.
  exit /b 5
)

if /I not "%SIMA_PCIE_HOST_ENVIRONMENT%"=="%SIMA_PCIE_HOST_ROOT%" (
  set "PATH=%SIMA_PCIE_HOST_ROOT%\bin;%SIMA_WINDOWS_PLATFORM_RUNTIME_DIR%;%GSTREAMER_ROOT_X86_64%\bin;%PATH%"
  if defined CMAKE_PREFIX_PATH (set "CMAKE_PREFIX_PATH=%SIMA_PCIE_HOST_ROOT%;%GSTREAMER_ROOT_X86_64%;%CMAKE_PREFIX_PATH%") else set "CMAKE_PREFIX_PATH=%SIMA_PCIE_HOST_ROOT%;%GSTREAMER_ROOT_X86_64%"
  if defined GST_PLUGIN_PATH (set "GST_PLUGIN_PATH=%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0;%GST_PLUGIN_PATH%") else set "GST_PLUGIN_PATH=%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0"
  if defined GST_PLUGIN_PATH_1_0 (set "GST_PLUGIN_PATH_1_0=%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0;%GST_PLUGIN_PATH_1_0%") else set "GST_PLUGIN_PATH_1_0=%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0"
  set "SIMA_PCIE_HOST_LIBRARY_PATH=%SIMA_PCIE_HOST_ROOT%\bin"
  set "SIMA_PCIE_HOST_ENVIRONMENT=%SIMA_PCIE_HOST_ROOT%"
)

exit /b 0
