# Install the offline Neat PCIe bundle

This installs Neat on an Ubuntu 24.04 amd64 host and every attached compatible
Platform 3.0 card. It does not convert firmware or install the host driver.

## Prepare the system

Complete Platform 3.0 conversion and install its compatible host driver first.
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
tar -xzf neat-offline.tar.gz
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

The final JSON reports the host and every card by PCI address. A card failure
returns a nonzero exit status but remaining cards are attempted if host checks
and topology remain valid. Host failure or changed topology stops installation.
SSH transport loss or a local timeout during installation reports an unknown
remote outcome and stops further installs; check that card before retrying.
Its staging directory is retained and reported so a potentially running installer
does not lose its input files.
Each installation command has a 25-minute deadline and a 30-second kill grace.
No automatic rollback is attempted. Fix the reported cause and rerun the same
command; there is no persistent resume state. Temporary transfers are cleaned
up; a cleanup failure reports its card so it can be addressed.

If discovery, SSH, firmware compatibility, or driver checks fail, repair the
system setup first. This installer does not change firmware, drivers, discovery
services, SSH provisioning, or PCIe addressing.
