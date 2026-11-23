@echo off
rem Configure only this cmd.exe process for the installed Neat PCIe host SDK.

for %%I in ("%~dp0..") do set "SIMA_PCIE_HOST_ROOT=%%~fI"

rem The GStreamer installer sets GSTREAMER_1_0_ROOT_MSVC_X86_64; search it first.
set "_SIMA_PCIE_HOST_GSTREAMER="
for %%R in ("%GSTREAMER_1_0_ROOT_MSVC_X86_64%" "%GSTREAMER_ROOT_X86_64%" "%ProgramFiles%\gstreamer\1.0\msvc_x86_64") do (
  if not defined _SIMA_PCIE_HOST_GSTREAMER if not "%%~R"=="" if exist "%%~R\bin\gstreamer-1.0-0.dll" set "_SIMA_PCIE_HOST_GSTREAMER=%%~R"
)
if not defined _SIMA_PCIE_HOST_GSTREAMER (
  1>&2 echo Neat PCIe host: GStreamer MSVC x64 was not found under GSTREAMER_1_0_ROOT_MSVC_X86_64, GSTREAMER_ROOT_X86_64 or "%ProgramFiles%\gstreamer\1.0\msvc_x86_64".
  1>&2 echo Install the supported GStreamer package or set GSTREAMER_ROOT_X86_64 to its installation root.
  exit /b 2
)
set "GSTREAMER_ROOT_X86_64=%_SIMA_PCIE_HOST_GSTREAMER%"
set "_SIMA_PCIE_HOST_GSTREAMER="

set "_SIMA_PCIE_HOST_SERVICE="
if not defined SIMA_WINDOWS_PLATFORM_RUNTIME_DIR (
  rem reg.exe prints "ImagePath <type> <command line>" in every display language; the
  rem service-control labels are translated. "call set" expands %%VAR%% in REG_EXPAND_SZ.
  for /f "tokens=2,*" %%I in ('reg.exe query "HKLM\SYSTEM\CurrentControlSet\Services\SiMaMlaDaemon" /v ImagePath 2^>nul ^| findstr.exe /C:"ImagePath"') do call set "_SIMA_PCIE_HOST_SERVICE=%%J"
)
rem The executable is the first item of the command line; it is quoted when it has spaces.
if defined _SIMA_PCIE_HOST_SERVICE for %%K in (%_SIMA_PCIE_HOST_SERVICE%) do if not defined SIMA_WINDOWS_PLATFORM_RUNTIME_DIR set "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR=%%~dpK"
set "_SIMA_PCIE_HOST_SERVICE="
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
