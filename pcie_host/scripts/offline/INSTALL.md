# Install the offline Neat PCIe bundle

This installs Neat on an Ubuntu 24.04 amd64 host and every attached compatible
Platform 3.0 card. It does not convert firmware or install the host driver.

## Prepare the system

Complete Platform 3.0 conversion and install its compatible host driver first.
The corrected Scheduler producer supports both the official `sima-cli` host
installer and Debian provisioning. It checks installed Platform header hashes
and library architecture/exports without requiring Platform DEB registration.
Older Scheduler publications with hard Platform DEB dependencies must be rebuilt
with that producer correction, not modified after publication.
Every card needs a working discovery responder on its PCIe interface, a unique
PCIe IPv4 address, Python 3, and SSH access. The host needs Python 3.11 or later,
`ip`, OpenSSH, `sudo`, `timeout`, and `unshare`; cards need `sudo`, `flock`,
`timeout`, and `unshare`.
The SSH account must have sudo access. Use passwordless sudo or pass
`--ask-card-sudo` to securely prompt once for the shared card sudo password.
Passwords are not stored in the bundle, reports, files, or command arguments. Provision trusted card
SSH host keys through your system setup procedure. The installer never accepts
unknown keys or scans physical Ethernet interfaces.

Check the archive SHA-256 against the value obtained from your trusted download
channel before extracting it. The embedded manifest detects corruption but is
not an independent authentication source. Extract into a path without spaces.
Docker and Internet access are not needed on the customer system.

## Install

Run on the host, using the SSH key provisioned during system setup:

```sh
tar -xzf neat-pcie-offline-installer.tar.gz
cd neat-offline
sudo python3 -B install.py --preflight-only
sudo python3 -B install.py
```

Defaults: SSH user `sima`, the invoking user's
`~/.ssh/sima_neat_pcie_ed25519` key and `~/.ssh/known_hosts`. Override credentials
with `--ssh-user`, `--ssh-key`, and `--known-hosts`; do not supply card IPs.

The installer verifies the bundle, inventories all attached SiMa PCI cards,
discovers through each PCI-owned interface, checks direct routes and trusted SSH,
and stages the card repository. All targets must pass compatibility, package
simulation, producer admission, Python ABI, and free-space checks before any
package installation. Preflight writes temporary repository indexes and staging
files but does not install packages or change networking.

It installs and verifies the host once, then each card sequentially. Package
commands run in network namespaces without network connectivity; SSH transfers
remain outside those namespaces on the PCIe links. APT uses only the bundled
repository and pip uses only bundled wheels. Existing install scripts retain
Platform admission and runtime lifecycle handling. Optional Sentinel telemetry
is retained if installed; its online updater is never called.

Python environments are installed at `/opt/sima/neat/pyneatpcie` on the host and
`/opt/sima/neat/pyneat` on each card. Post-install checks cover exact native
package versions, package health, Python dependencies/imports, the host GStreamer
plugin, and the existing card installer's runtime checks. They do not replace an
application inference acceptance test.

## Results and retries

During host and card installation, terminals show an updating panel with the last
10 lines of command output and elapsed time. The panel shrinks on short terminals
and truncates long lines to avoid wrapping. Preflight retains its quieter activity
indicator. Redirected output uses a status line every 15 seconds without terminal
control sequences. Full output remains in the diagnostic log. Elapsed time is not
a completion percentage or estimated finish time.

The default output shows progress and a summary with the number of cards
installed and verified, each card's PCI address, virtual IP, and physical Ethernet
interface/MAC read over authenticated SSH. If multiple physical ports exist,
the installer lists them without guessing which is primary. An unavailable MAC
is reported explicitly; the virtual discovery MAC is never substituted.
`--preflight-only` reports checks passed and explicitly says no packages were
installed. Colors are enabled only on a capable terminal; set `NO_COLOR=1` to
disable them. Use `--json` for the final machine-readable report (progress goes
to stderr).

Detailed package output is retained in a root-only `/var/tmp/neat-offline-*.log`
file whose path is printed. Successful APT index retries and expected package
ownership alias probes stay in that log. A failed command retains its nonzero
status, prints recent diagnostics, and is reflected in the summary.

Card sudo authentication tries passwordless sudo, then the default `edgeai`
password, then prompts securely if needed. `--ask-card-sudo` lets you supply a
shared password explicitly. Passwords are not saved or passed as command arguments.

The final report identifies the host and every card by PCI address. A card failure
returns a nonzero exit status but remaining cards are attempted if host checks
and topology remain valid. Host failure or changed topology stops installation.
SSH transport loss, a local timeout, or Ctrl-C during card installation reports
an unknown remote outcome and stops further installs; check that card before retrying.
Its staging directory is retained and reported so a potentially running installer
does not lose its input files. Confirm that the remote transaction has stopped
before removing retained staging or retrying.
Each installation command has a 25-minute deadline and a 30-second kill grace.
No automatic rollback is attempted. Fix the reported cause and rerun the same
command; there is no persistent resume state. Temporary transfers are cleaned
up; a cleanup failure reports its card so it can be addressed.

If discovery, SSH, firmware compatibility, or driver checks fail, repair the
system setup first. This installer does not change firmware, drivers, discovery
services, SSH provisioning, or PCIe addressing.

## Debugging and support logs

Use `sudo python3 -B install.py --verbose` (or `-v`) to stream all command output
instead of the rolling panel, including diagnostic warnings normally kept in the
log. The complete output is still saved to the printed log path. This also works
with `--preflight-only`. With `--json --verbose`, logs and progress go to stderr
and the final JSON report goes to stdout. Authentication passwords are not logged.

If Ctrl-C interrupts host deployment, the installer stops the complete host
command group before releasing its lock. It allows up to 30 seconds for graceful
termination, then kills and reaps remaining children. Further Ctrl-C presses
cannot bypass cleanup. An interrupted package transaction may still require
repair; check `sudo dpkg --audit` and the diagnostic log before retrying.

## Disk space

The native installer reserves three times the expanded Debian dependency and
Python wheel contents, plus 128 MiB for package metadata and the Python
environment. Packages without an `Installed-Size` field are measured from their
archive contents. Already-staged bundle files are not counted again, including
when they are stored on NVMe or tmpfs. A space failure reports available,
required, and additional root space in MiB.
