<#
.SYNOPSIS
Run the native Windows Core producer in an x64 Visual Studio environment.

.DESCRIPTION
Reads the producer inputs that provision_windows_build.ps1 exported and the
dependency cohort that acquire_windows_dependencies.py resolved, then runs
build_windows.py below the run-owned SIMA_WINDOWS_SHORT_WORK_ROOT.
#>
param(
  [Parameter(Mandatory = $true)][string] $Dependencies,
  [Parameter(Mandatory = $true)][string] $Output
)
$ErrorActionPreference = "Stop"
$selection = Get-Content -Raw "$Dependencies\selection.json" | ConvertFrom-Json
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw "Visual Studio x64 C++ tools are unavailable" }
Import-Module (Join-Path $installation "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
# Hosted images and actions/setup-python export these; the producer takes its prefixes explicitly.
foreach ($name in @("CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR_PLATFORM", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH")) {
  Remove-Item -LiteralPath "env:$name" -ErrorAction SilentlyContinue
}
foreach ($name in @(
  "GSTREAMER_ROOT_X86_64",
  "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH",
  "SIMA_WINDOWS_PYTHON_WHEELHOUSE"
)) {
  if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue).Value) {
    throw "Required producer variable is unset: $name"
  }
}
if (-not $env:SIMA_WINDOWS_SHORT_WORK_ROOT.StartsWith("C:\nr\c-")) {
  throw "Windows producer work root must be a run-owned directory below C:\nr"
}
if (Test-Path -LiteralPath $env:SIMA_WINDOWS_SHORT_WORK_ROOT) {
  Remove-Item -LiteralPath $env:SIMA_WINDOWS_SHORT_WORK_ROOT -Recurse -Force
}
$arguments = @(
  "scripts/ci/pciehost/build_windows.py",
  "--branch", $env:SOURCE_BRANCH,
  "--expected-scheduler-commit", $selection.scheduler.commit,
  "--gstreamer-root", $env:GSTREAMER_ROOT_X86_64,
  "--python-wheelhouse", $env:SIMA_WINDOWS_PYTHON_WHEELHOUSE,
  "--scheduler-candidate", "$Dependencies\scheduler",
  "--work", $env:SIMA_WINDOWS_SHORT_WORK_ROOT,
  "--output", $Output
)
if ($env:SIMA_WINDOWS_CMAKE_PREFIX_PATH) {
  foreach ($prefix in $env:SIMA_WINDOWS_CMAKE_PREFIX_PATH.Split(";")) {
    if ($prefix) { $arguments += @("--dependency-prefix", $prefix) }
  }
}
foreach ($directory in $env:SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH.Split(";")) {
  if ($directory) { $arguments += @("--dependency-runtime-dir", $directory) }
}
if ("$env:REQUIRE_SIGNATURES" -eq "true") {
  if (-not $env:SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1) {
    throw "Signed packages require SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1"
  }
  $arguments += @(
    "--require-signatures",
    "--signing-certificate-sha1", $env:SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1
  )
  if ($env:SIMA_WINDOWS_SIGNING_TIMESTAMP_URL) {
    $arguments += @("--signing-timestamp-url", $env:SIMA_WINDOWS_SIGNING_TIMESTAMP_URL)
  }
  if ($env:SIMA_WINDOWS_SIGNING_MACHINE_STORE -eq "true") {
    $arguments += "--signing-machine-store"
  }
}
& python @arguments
if ($LASTEXITCODE -ne 0) { throw "Windows Core producer failed: $LASTEXITCODE" }
