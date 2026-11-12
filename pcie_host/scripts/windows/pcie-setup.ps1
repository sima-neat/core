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
    if (-not $process.WaitForExit($TimeoutMilliseconds)) {
      $timedOut = $true
      $process.Kill()
      $process.WaitForExit()
    }
    $null = $process.StandardOutput.ReadToEnd()
    $null = $process.StandardError.ReadToEnd()
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
    [string]$KnownHosts,
    [string]$Target
  )

  $result = Invoke-BoundedCommand -Executable $Ssh -Arguments @(
    "-i", $PrivateKey, "-p", "$Port", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=$KnownHosts", $Target, "true"
  )
  return -not $result.TimedOut -and $result.ExitCode -eq 0
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
  $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
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
$knownHostsPath = Join-Path $sshDirectory "known_hosts"

Write-Host "SiMa Neat PCIe SSH provisioning"
Write-Host "User: $User"
Write-Host "Port: $Port"
Write-Host "Key:  $Key"
Write-Host "Hosts: $($targets -join ' ')"

if ($DryRun) {
  Write-Host "[dry-run] Would create or reuse the key and known_hosts under $sshDirectory."
}
else {
  New-Item -ItemType Directory -Path $sshDirectory -Force | Out-Null
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
    # Windows ssh-keyscan can hang against the card SSH server. Use OpenSSH's
    # bounded trust-on-first-use path instead. This may fail authentication;
    # its purpose is to record a first-seen host key while rejecting changes.
    $bootstrapResult = Invoke-BoundedCommand -Executable $ssh -Arguments @(
      "-i", $Key, "-p", "$Port", "-o", "BatchMode=yes",
      "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
      "-o", "UserKnownHostsFile=$knownHostsPath", $target, "true"
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
      (Test-KeyAccess -Ssh $ssh -PrivateKey $Key -KnownHosts $knownHostsPath -Target $target))) {
    Write-Host "Already provisioned."
    $success += 1
    continue
  }

  if ($NonInteractive) {
    Write-Warning "Skipped: the key is not enrolled and non-interactive mode cannot prompt."
    $skipped += 1
    continue
  }

  $publicKey = (Get-Content -LiteralPath $publicKeyPath -Raw).Trim()
  $remoteScript = @'
IFS= read -r key
umask 077
mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
if ! grep -qxF "$key" "$HOME/.ssh/authorized_keys"; then
  printf '%s\n' "$key" >> "$HOME/.ssh/authorized_keys"
fi
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/authorized_keys"
'@
  $payload = $publicKey + "`n" + $remoteScript
  Write-Host "Enter the card password when OpenSSH prompts. The password is not stored."
  $savedErrorPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $payload | & $ssh -p $Port -o ConnectTimeout=10 -o StrictHostKeyChecking=yes `
        -o "UserKnownHostsFile=$knownHostsPath" $target "sh -s"
    $enrollExitCode = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $savedErrorPreference
  }
  if ($enrollExitCode -ne 0) {
    Write-Warning "Public-key enrollment failed for $target."
    $failed += 1
    continue
  }

  if (Test-KeyAccess -Ssh $ssh -PrivateKey $Key -KnownHosts $knownHostsPath -Target $target) {
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
