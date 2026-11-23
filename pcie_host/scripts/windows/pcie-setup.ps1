#Requires -Version 5.1

[CmdletBinding()]
param(
  [string]$User = "sima",
  [string]$Key = (Join-Path $HOME ".ssh\sima_neat_pcie_ed25519"),
  [string[]]$Hosts = @(),
  [string]$Range,
  [switch]$NoDiscover,
  [switch]$NonInteractive,
  [switch]$Strict,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Port = 22

function Add-UniqueTarget {
  param(
    [System.Collections.Generic.List[string]]$Targets,
    [string]$Candidate
  )

  if ([string]::IsNullOrWhiteSpace($Candidate)) {
    return
  }
  $Candidate = $Candidate.Trim()
  if ($Candidate -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]*$') {
    throw "Invalid card host '$Candidate'."
  }
  if (-not $Targets.Contains($Candidate)) {
    $Targets.Add($Candidate)
  }
}

function Get-DiscoveredTargets {
  $discovered = New-Object 'System.Collections.Generic.List[string]'
  try {
    $interfaces = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()
    foreach ($interface in $interfaces) {
      if ($interface.OperationalStatus -ne [System.Net.NetworkInformation.OperationalStatus]::Up) {
        continue
      }
      foreach ($address in $interface.GetIPProperties().UnicastAddresses) {
        $text = $address.Address.ToString()
        if ($text -match '^10\.0\.(\d+)\.1$') {
          Add-UniqueTarget -Targets $discovered -Candidate "10.0.$($Matches[1]).2"
        }
      }
    }
  }
  catch {
    Write-Verbose "PCIe management address discovery failed: $($_.Exception.Message)"
  }
  return $discovered
}

function ConvertTo-ProcessArgument {
  param([string]$Value)

  if ($Value.Contains('"')) {
    throw "OpenSSH argument contains an unsupported quote."
  }
  if ($Value -match '\s') {
    return '"' + $Value + '"'
  }
  return $Value
}

function Invoke-BoundedCommand {
  param(
    [string]$Executable,
    [string[]]$Arguments,
    [int]$TimeoutMilliseconds = 10000
  )

  $process = $null
  $timedOut = $false
  try {
    $processArguments = @($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ })
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = $processArguments -join " "
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
      throw "Failed to start $Executable."
    }
    $standardOutputDrain = $process.StandardOutput.ReadToEndAsync()
    $standardErrorDrain = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutMilliseconds)) {
      $timedOut = $true
      $process.Kill()
      $process.WaitForExit()
    }
    [System.Threading.Tasks.Task]::WaitAll(
        [System.Threading.Tasks.Task[]]@($standardOutputDrain, $standardErrorDrain))
    return [pscustomobject]@{
      ExitCode = if ($timedOut) { $null } else { $process.ExitCode }
      TimedOut = $timedOut
    }
  }
  finally {
    if ($null -ne $process) {
      $process.Dispose()
    }
  }
}

function Test-KeyAccess {
  param(
    [string]$Ssh,
    [string]$PrivateKey,
    [string]$KnownHostsOption,
    [string]$Target
  )

  $result = Invoke-BoundedCommand -Executable $Ssh -Arguments @(
    "-i", $PrivateKey, "-o", "IdentitiesOnly=yes", "-p", "$Port", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes",
    "-o", $KnownHostsOption, $Target, "true"
  )
  return -not $result.TimedOut -and $result.ExitCode -eq 0
}

function Write-ManualEnrollmentHelp {
  param(
    [string]$Target,
    [string]$PublicKeyPath
  )

  $publicKey = (Get-Content -LiteralPath $PublicKeyPath -Raw).Trim()
  Write-Host "If the card does not accept password login, add the public key in"
  Write-Host "$PublicKeyPath to ~/.ssh/authorized_keys for $Target yourself, then run"
  Write-Host "pcie-setup again. From a shell on the card signed in as that user (the serial"
  Write-Host "console, or SSH from a machine that already has access), run:"
  Write-Host "  mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$publicKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
}

if ($User -notmatch '^[A-Za-z0-9._-]+$') {
  throw "Invalid remote user '$User'."
}

$requiredCommands = @(
  "ssh.exe",
  "scp.exe",
  "ssh-keygen.exe"
)
$commands = @{}
foreach ($name in $requiredCommands) {
  # PATH can hold more than one OpenSSH; use the first match, as the shell would.
  $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($null -eq $command) {
    throw "$name was not found. Install the Windows OpenSSH Client optional feature."
  }
  $commands[$name] = $command.Source
}

$targets = New-Object 'System.Collections.Generic.List[string]'
foreach ($value in $Hosts) {
  foreach ($candidate in ($value -split '[,\s]+')) {
    Add-UniqueTarget -Targets $targets -Candidate $candidate
  }
}

if (-not [string]::IsNullOrWhiteSpace($Range)) {
  if ($Range -notmatch '^(\d+)-(\d+)$') {
    throw "Invalid range '$Range'; expected FIRST-LAST."
  }
  $first = [int]$Matches[1]
  $last = [int]$Matches[2]
  if ($first -gt $last -or $first -lt 0 -or $last -gt 255) {
    throw "Invalid range '$Range'; card indices must be between 0 and 255."
  }
  foreach ($index in $first..$last) {
    Add-UniqueTarget -Targets $targets -Candidate "10.0.$index.2"
  }
}

if ($targets.Count -eq 0 -and -not $NoDiscover) {
  foreach ($candidate in (Get-DiscoveredTargets)) {
    Add-UniqueTarget -Targets $targets -Candidate $candidate
  }
}
if ($targets.Count -eq 0) {
  throw "No PCIe card management hosts found. Connect a card or pass -Hosts or -Range."
}

$Key = [System.IO.Path]::GetFullPath($Key)
$sshDirectory = Split-Path -Parent $Key
$publicKeyPath = "$Key.pub"
# Runtime ssh.exe reads the profile's known_hosts whatever key it uses, as on Linux.
$knownHostsPath = Join-Path $env:USERPROFILE ".ssh\known_hosts"
# OpenSSH splits -o values at spaces before expanding ${VAR}, so pass the path that way.
$env:SIMA_NEAT_PCIE_KNOWN_HOSTS = $knownHostsPath
$knownHostsOption = 'UserKnownHostsFile=${SIMA_NEAT_PCIE_KNOWN_HOSTS}'

Write-Host "SiMa Neat PCIe SSH provisioning"
Write-Host "User: $User"
Write-Host "Port: $Port"
Write-Host "Key:  $Key"
Write-Host "Known hosts: $knownHostsPath"
Write-Host "Hosts: $($targets -join ' ')"

if ($DryRun) {
  Write-Host "[dry-run] Would create or reuse the key and known_hosts."
}
else {
  New-Item -ItemType Directory -Path $sshDirectory -Force | Out-Null
  New-Item -ItemType Directory -Path (Split-Path -Parent $knownHostsPath) -Force | Out-Null
  if (-not (Test-Path -LiteralPath $Key -PathType Leaf)) {
    & $commands["ssh-keygen.exe"] -t ed25519 -N '""' -f $Key `
        -C "sima-neat-pcie@$env:COMPUTERNAME" | Out-Null
    if ($LASTEXITCODE -ne 0) {
      throw "ssh-keygen failed to create $Key."
    }
  }
  if (-not (Test-Path -LiteralPath $publicKeyPath -PathType Leaf)) {
    $publicKey = & $commands["ssh-keygen.exe"] -y -f $Key
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($publicKey)) {
      throw "ssh-keygen failed to derive $publicKeyPath."
    }
    [System.IO.File]::WriteAllText(
        $publicKeyPath,
        ($publicKey.Trim() + [Environment]::NewLine),
        (New-Object System.Text.UTF8Encoding($false)))
  }
  if (-not (Test-Path -LiteralPath $knownHostsPath -PathType Leaf)) {
    New-Item -ItemType File -Path $knownHostsPath -Force | Out-Null
  }
}

$success = 0
$failed = 0
$skipped = 0
$ssh = $commands["ssh.exe"]
$sshKeygen = $commands["ssh-keygen.exe"]

foreach ($cardHost in $targets) {
  $target = "${User}@${cardHost}"
  $keyAccessChecked = $false
  $keyAccessVerified = $false
  Write-Host ""
  Write-Host "==> ${target}:$Port"

  if ($DryRun) {
    Write-Host "[dry-run] Would record the host key, enroll the public key if needed, and verify SSH."
    $success += 1
    continue
  }

  $savedErrorPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    & $sshKeygen -F $cardHost -f $knownHostsPath *> $null
    $findExitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $savedErrorPreference
  }
  if ($findExitCode -ne 0) {
    # ssh-keyscan can hang here; a bounded accept-new login records the first-seen key.
    $bootstrapResult = Invoke-BoundedCommand -Executable $ssh -Arguments @(
      "-i", $Key, "-o", "IdentitiesOnly=yes", "-p", "$Port", "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
      "-o", $knownHostsOption, $target, "true"
    )
    $keyAccessChecked = $true
    $keyAccessVerified = (
      -not $bootstrapResult.TimedOut -and $bootstrapResult.ExitCode -eq 0)
    $savedErrorPreference = $ErrorActionPreference
    try {
      $ErrorActionPreference = "Continue"
      & $sshKeygen -F $cardHost -f $knownHostsPath *> $null
      $findExitCode = $LASTEXITCODE
    }
    finally {
      $ErrorActionPreference = $savedErrorPreference
    }
    if ($findExitCode -ne 0) {
      Write-Warning "Could not retrieve the SSH host key for $cardHost."
      $failed += 1
      continue
    }
  }

  if ($keyAccessVerified -or (
      -not $keyAccessChecked -and
      (Test-KeyAccess -Ssh $ssh -PrivateKey $Key -KnownHostsOption $knownHostsOption `
          -Target $target))) {
    Write-Host "Already provisioned."
    $success += 1
    continue
  }

  if ($NonInteractive) {
    Write-Warning "Skipped: the key is not enrolled and non-interactive mode cannot prompt."
    Write-ManualEnrollmentHelp -Target $target -PublicKeyPath $publicKeyPath
    $skipped += 1
    continue
  }

  $publicKey = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
  # Only the key (maybe CRLF-terminated) travels on stdin. Windows PowerShell does not
  # escape double quotes in native arguments, so the remote script must not use them.
  $remoteScript = @'
umask 077
mkdir -p ~/.ssh && cd ~/.ssh && touch authorized_keys || exit 1
head -n1 | tr -d '\r' > .neat-pcie-key || exit 1
if grep -Eqx '(ssh-(ed25519|rsa)|ecdsa-sha2-nistp(256|384|521)|sk-(ssh-ed25519|ecdsa-sha2-nistp256)@openssh[.]com) [A-Za-z0-9+/]+={0,3}( .*)?' .neat-pcie-key; then
  grep -qxFf .neat-pcie-key authorized_keys || cat .neat-pcie-key >> authorized_keys
  status=$?
else
  echo 'The enrolment input is not one OpenSSH public key line.' >&2
  status=1
fi
rm -f .neat-pcie-key
chmod 700 . && chmod 600 authorized_keys || status=1
exit $status
'@
  # A CRLF checkout of this file would otherwise end every remote line with CR.
  $remoteScript = $remoteScript.Replace("`r", "")
  Write-Host "Enter the card password when OpenSSH prompts. The password is not stored."
  $savedErrorPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $publicKey | & $ssh -p $Port -o ConnectTimeout=10 -o StrictHostKeyChecking=yes `
        -o $knownHostsOption $target $remoteScript
    $enrollExitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $savedErrorPreference
  }
  if ($enrollExitCode -ne 0) {
    Write-Warning "Public-key enrollment failed for $target."
    Write-ManualEnrollmentHelp -Target $target -PublicKeyPath $publicKeyPath
    $failed += 1
    continue
  }

  if (Test-KeyAccess -Ssh $ssh -PrivateKey $Key -KnownHostsOption $knownHostsOption `
      -Target $target) {
    Write-Host "Provisioned and verified."
    $success += 1
  }
  else {
    Write-Warning "The key was sent, but passwordless SSH verification failed for $target."
    $failed += 1
  }
}

Write-Host ""
Write-Host "Provisioning summary: success=$success failed=$failed skipped=$skipped"
if (($Strict -and ($failed -gt 0 -or $skipped -gt 0)) -or $success -eq 0) {
  exit 1
}
exit 0
