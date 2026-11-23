<#
.SYNOPSIS
Provision the pinned inputs of the GitHub-hosted Windows PCIe-host build.

.DESCRIPTION
Downloads every input from its pinned URL in windows_build_inputs.json and
checks it with Get-FileHash before use. Platform files are not inputs: as on
Linux, the build takes the Platform service headers from the Scheduler package.
The "build" component provisions the producer inputs of build_windows.py; the
"sima-cli" component provisions the bootstrap CLI that resolves the published
Scheduler and sima-cli candidates. Results are exported through GITHUB_ENV.
#>
param(
  [Parameter(Mandatory = $true)][ValidateSet("build", "sima-cli")][string[]] $Component,
  [Parameter(Mandatory = $true)][string] $Root
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-PinnedFile([string] $Url, [string] $Sha256, [string] $Path) {
  if ($Url -notmatch '^https://' -or $Sha256 -notmatch '^[0-9a-f]{64}$') {
    throw "Windows build input is not pinned by HTTPS URL and SHA-256: $Url"
  }
  # One transient network failure must not fail the release build; the pin is never retried.
  for ($attempt = 1; ; $attempt++) {
    try { Invoke-WebRequest -Uri $Url -OutFile $Path; break }
    catch {
      if ($attempt -ge 4) { throw }
      Write-Warning "Download attempt $attempt of 4 failed for ${Url}: $($_.Exception.Message)"
      Start-Sleep -Seconds (10 * $attempt)
    }
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
  if ($actual -ne $Sha256) { throw "SHA-256 mismatch for ${Url}: expected $Sha256, got $actual" }
}

function Invoke-Checked([string] $FilePath, [string[]] $Arguments) {
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$FilePath failed: $LASTEXITCODE" }
}

function Assert-Files([string] $Directory, [string[]] $Names) {
  foreach ($name in $Names) {
    if (-not (Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) {
      throw "Provisioned input lacks $name below $Directory"
    }
  }
}

$manifest = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot "windows_build_inputs.json") |
  ConvertFrom-Json
$uses = @()
# The wheelhouse also serves the card lane's customer flow, so it carries sima-cli's closure.
if ($Component -contains "build") {
  $uses += @("gstreamer", "opencv", "nlohmann-json", "wix", "wheelhouse", "sima-cli")
}
if ($Component -contains "sima-cli") { $uses += "sima-cli" }
if (Test-Path -LiteralPath $Root) { throw "Provisioning root must be fresh: $Root" }
$downloads = New-Item -ItemType Directory -Path (Join-Path $Root "downloads")
$inputs = @{}
foreach ($entry in $manifest.files) {
  if ($uses -notcontains $entry.use) { continue }
  if ($entry.name -notmatch '^[A-Za-z0-9][A-Za-z0-9._+-]*$') { throw "Unsafe input name: $($entry.name)" }
  $path = Join-Path $downloads.FullName $entry.name
  Get-PinnedFile $entry.url $entry.sha256 $path
  if (-not $inputs.ContainsKey($entry.use)) {
    $inputs[$entry.use] = [System.Collections.Generic.List[string]]::new()
  }
  $inputs[$entry.use].Add($path)
}
$environment = @()

if ($Component -contains "build") {
  # The single vendor installer carries runtime and development files.
  $gstreamer = Join-Path $env:ProgramFiles "gstreamer\1.0\msvc_x86_64"
  $setup = Start-Process -FilePath $inputs["gstreamer"][0] -Wait -PassThru -ArgumentList @(
    "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/ALLUSERS", "/TYPE=devel",
    "/DIR=`"$gstreamer`"", "/LOG=`"$(Join-Path $Root 'gstreamer-setup.log')`"")
  if ($setup.ExitCode -ne 0) { throw "GStreamer installation failed: $($setup.ExitCode)" }
  Assert-Files $gstreamer @(
    "bin\gst-inspect-1.0.exe", "bin\pkg-config.exe", "lib\pkgconfig\gstreamer-1.0.pc")

  # Extract the vendor self-extractor instead of executing it.
  $sevenZip = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
  if (-not $sevenZip) { $sevenZip = Join-Path $env:ProgramFiles "7-Zip\7z.exe" }
  Invoke-Checked $sevenZip @("x", $inputs["opencv"][0], "-o$(Join-Path $Root 'opencv')", "-y", "-bso0", "-bsp0")
  $opencv = Join-Path $Root "opencv\opencv\build"
  $opencvRuntime = Join-Path $opencv "x64\vc16\bin"
  Assert-Files $opencv @("OpenCVConfig.cmake", "x64\vc16\bin\opencv_world500.dll")

  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  $installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $installation) { throw "Visual Studio x64 C++ tools are unavailable" }
  Import-Module (Join-Path $installation "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
  Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
  $jsonSource = New-Item -ItemType Directory -Path (Join-Path $Root "nlohmann-json-source")
  Invoke-Checked tar.exe @("-xf", $inputs["nlohmann-json"][0], "-C", $jsonSource.FullName)
  $json = Join-Path $Root "nlohmann-json"
  Invoke-Checked cmake.exe @(
    "-S", (Join-Path $jsonSource.FullName "json"), "-B", (Join-Path $Root "nlohmann-json-build"),
    "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DJSON_BuildTests=OFF", "-DJSON_Install=ON",
    "-DCMAKE_INSTALL_PREFIX=$json")
  Invoke-Checked cmake.exe @("--install", (Join-Path $Root "nlohmann-json-build"))
  Assert-Files $json @("include\nlohmann\json.hpp")

  $wix = Join-Path $Root "wix"
  Expand-Archive -LiteralPath $inputs["wix"][0] -DestinationPath $wix
  Assert-Files $wix @("candle.exe", "light.exe")

  $wheelhouse = New-Item -ItemType Directory -Path (Join-Path $Root "wheelhouse")
  foreach ($wheel in @($inputs["wheelhouse"]) + @($inputs["sima-cli"])) {
    Copy-Item -LiteralPath $wheel -Destination $wheelhouse.FullName
  }

  $environment += @(
    "GSTREAMER_ROOT_X86_64=$gstreamer",
    "SIMA_WINDOWS_CMAKE_PREFIX_PATH=$opencv;$json",
    "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH=$opencvRuntime",
    "SIMA_WINDOWS_PYTHON_WHEELHOUSE=$($wheelhouse.FullName)",
    "WIX=$wix"
  )
}

if ($Component -contains "sima-cli") {
  $bootstrap = New-Item -ItemType Directory -Path (Join-Path $Root "sima-cli-wheels")
  foreach ($wheel in $inputs["sima-cli"]) { Copy-Item -LiteralPath $wheel -Destination $bootstrap.FullName }
  $venv = Join-Path $Root "sima-cli-env"
  Invoke-Checked python.exe @("-m", "venv", $venv)
  Invoke-Checked (Join-Path $venv "Scripts\python.exe") @(
    "-m", "pip", "install", "--no-index", "--find-links", $bootstrap.FullName,
    "sima-cli==$($manifest.sima_cli_version)")
  $environment += "SIMA_CLI=$(Join-Path $venv 'Scripts\sima-cli.exe')"
}

Add-Content -LiteralPath $env:GITHUB_ENV -Value $environment
