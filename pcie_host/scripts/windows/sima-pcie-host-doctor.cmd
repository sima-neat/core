@echo off
setlocal EnableExtensions

call "%~dp0sima-pcie-host-env.cmd"
if not "%errorlevel%"=="0" exit /b %errorlevel%

rem System tools by full path: cmd.exe searches the current directory first.
"%SystemRoot%\System32\where.exe" ssh.exe >nul 2>nul
if not "%errorlevel%"=="0" (
  1>&2 echo Neat PCIe host: Windows OpenSSH client was not found.
  exit /b 10
)

rem Informational only: the test-signed Platform driver loads only with Secure Boot off and
rem test-signing on. SystemStartOptions is readable without elevation; Confirm-SecureBootUEFI
rem may not be.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -Command "if (Get-CimInstance Win32_PnPEntity -Filter 'Name=''SiMa.ai MLA Device'' AND ConfigManagerErrorCode=0' -ErrorAction SilentlyContinue) { exit 0 }; try { $secureBoot = if (Confirm-SecureBootUEFI -ErrorAction Stop) { 'on' } else { 'off' } } catch { $secureBoot = 'unknown, run Confirm-SecureBootUEFI from an elevated PowerShell to check' }; try { $testSigning = if ((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control' -ErrorAction Stop).SystemStartOptions -match 'TESTSIGNING') { 'on' } else { 'off' } } catch { $testSigning = 'unknown, run bcdedit from an elevated PowerShell to check' }; [Console]::Error.WriteLine('Neat PCIe host: no working SiMa.ai MLA Device. The test-signed Platform driver loads only with Secure Boot off and test-signing on.'); [Console]::Error.WriteLine('Neat PCIe host: Secure Boot: ' + $secureBoot + '; test-signing: ' + $testSigning + '.')"

rem Service status names are not translated; service-control text is.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -Command "if ((Get-Service -Name SiMaMlaDaemon -ErrorAction Stop).Status -ne 'Running') { exit 1 }" >nul 2>nul
if not "%errorlevel%"=="0" (
  1>&2 echo Neat PCIe host: SiMaMlaDaemon is not running.
  exit /b 11
)

"%GSTREAMER_ROOT_X86_64%\bin\gst-inspect-1.0.exe" "%SIMA_PCIE_HOST_ROOT%\bin\gstreamer-1.0\gstneatpciehost.dll" >nul
if not "%errorlevel%"=="0" (
  1>&2 echo Neat PCIe host: the installed neatpciehost GStreamer plug-in did not load.
  exit /b 12
)

echo Neat PCIe host prerequisites: PASS
echo Install root: %SIMA_PCIE_HOST_ROOT%
echo GStreamer root: %GSTREAMER_ROOT_X86_64%
echo Platform runtime: %SIMA_WINDOWS_PLATFORM_RUNTIME_DIR%
rem The installers add no PATH entry; print the full PowerShell command.
echo To show the host and card software, run in PowerShell:
echo ^& "%SIMA_PCIE_HOST_ROOT%\bin\neat.cmd"
exit /b 0
