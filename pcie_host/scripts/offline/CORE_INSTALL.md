# Offline standalone Neat Core installation

Use this bundle inside the Neat eLxr SDK for Platform 3.0 or on an ARM64 Modalix
DevKit already provisioned with Platform 3.0 and eLxr 26.04. The installer checks
the canonical `/etc/sdk-release` or `/etc/buildinfo` against the bundled package
manifest. It rejects other environments and incompatible Platform versions
before installation. Publication `metadata.json` is not needed on the target.
Platform firmware and OS provisioning remain prerequisites.
The environment must provide Python 3, `unshare`, and `timeout`. Stop applications
using Neat before upgrading. Allow root access and sufficient free disk space.

Copy `neat-core-offline-installer.tar.gz` to the DevKit, verify its SHA-256
against `offline/metadata.json` or `offline/metadata-minimal.json`, then run:

```sh
tar -xzf neat-core-offline-installer.tar.gz
cd neat-core-offline
./install.sh --check
./install.sh
./install.sh --verify
~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__)'
```

The archive includes the Core and matching Scheduler Debian packages, installer
lifecycle scripts, local APT dependency repository, Python wheels and their
dependencies, compatibility metadata, and a checksum inventory. DevKit
installation uses an isolated network namespace and only the bundled APT
repository and wheelhouse. It does not need sima-cli, a PCIe host, SSH, or
Internet access.
`--check` verifies integrity and compatibility and simulates the package
transaction without installing. `--verify` checks installed package versions,
APT health, Python dependencies, and the PyNeat import. These checks do not prove
model inference; validate your application separately on the target.

Inside the SDK, Core is extracted into `/opt/toolchain/aarch64/modalix`, using
the published Core installer and the bundled dependency repository. ARM64
Python wheels are retained for DevKit deployment, not installed into SDK host
Python. SDK `--check` validates environment, Platform, package architecture and
package composition; `--verify` compares installed sysroot files and symlinks
against the bundled DEBs.
The SDK path works in the standard unprivileged SDK container: it does not use
`unshare` or require additional container capabilities. Its private APT
configuration enables only the `file`, `copy`, and `store` methods. HTTP,
HTTPS, and other network transports are unavailable; missing dependencies fail
without an online fallback. Only the selected package archives are extracted
into the sysroot; their native package installation scripts are not executed.

If the SDK has `DEVKIT_SYNC_DEVKIT_IP` configured by DevKit Sync, the installer
transfers the same offline bundle over SSH and runs it on the paired DevKit.
Both SDK and DevKit preflight must pass before installation starts. SSH access
must already be provisioned. The transfer uses the SDK user's SSH identity;
`DEVKIT_DEPLOY_USER` defaults to `sima`. DevKit sudo authentication follows the
online installer's `SUDO_PASSWORD` / `DEVKIT_PASSWORD` convention. Set
`NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON` to install only into the SDK sysroot.
Package installation uses only bundled files on both targets; SSH is used only
to transfer the bundle and invoke its installer. An explicitly configured but
unreachable or incompatible paired DevKit fails installation.
Run the launcher as your regular user so it preserves the SDK pairing before
requesting sudo privileges. If invoking `sudo` yourself, preserve them explicitly:

```sh
sudo --preserve-env=DEVKIT_SYNC_DEVKIT_IP,DEVKIT_DEPLOY_USER,SUDO_PASSWORD,DEVKIT_PASSWORD,NEAT_INSTALLER_SKIP_DEVKIT_SYNC ./install.sh
```

The installer retains the published Core platform admission, package ownership,
and firmware lifecycle checks. It does not update optional Sentinel telemetry.
Offline installation uses the same Python paths as online installation. The PCIe host uses `~/pyneatpcie`. A DevKit uses `/media/nvme/pyneat` when `/media/nvme` is writable by the installing user; otherwise, it uses `~/pyneat`. For an NVMe installation, `~/pyneat` links to that environment. If a different environment already exists at `~/pyneat`, the installer keeps it in a `pyneat-before-offline-*` directory in your home directory before creating the link.
Activate the DevKit environment with
`source ~/pyneat/bin/activate`.

## Download profiles

```sh
sima-cli neat install core/offline@develop --download-only
sima-cli neat install core/offline@develop -t minimal --download-only
```

The previous `core@develop -t offline` command selects a different metadata
filename and is not the standalone Core package path. Keep all downloaded
archives together when transferring the full profile. Extract the separate
extras archive to a directory of your choice for tutorials and tests.

## Building and publication

The PCIe Offline Bundles workflow builds this standalone archive alongside the
four PCIe variants from the same immutable Core publication. All five builds
must pass before any new offline metadata is exposed. The standalone artifact
is published in Core's `offline/` subdirectory. `metadata.json` includes the
runtime bundle and the separate tutorials/tests extras archive;
`metadata-minimal.json` includes only the runtime bundle. Both profiles use the
same checksummed runtime archive. The PCIe variants retain their existing paths.
Existing completed publications are verified and never replaced.

For a local SDK-generated `dist/` publication, create a specification with
`scope: "core"` and one `targets.card` record (publication directory, metadata
SHA-256, Core revision, pinned baseline image ID, `arm64`, `elxr`, `26.04`). Run
`python3 -B assemble.py assemble --spec spec.json --output /path/neat-core-offline`.
The assembler collects against the pristine provisioned baseline, then verifies
native package installation and Python imports with Docker `--network=none`
before writing the archive. Use the provisioned baseline in `baselines.json`,
not the dependency-enriched SDK image, to avoid omitting target dependencies.

## Disk space

The native installer uses the offline APT transaction plan to reserve three times
the expanded contents of Debian packages selected for installation or
reinstallation and all Python wheels, plus 128 MiB for package metadata and the
Python environment. Dependencies left untouched by APT do not consume this
installation budget. Packages without an `Installed-Size` field are measured from their
archive contents. Already-staged bundle files are not counted again, including
when they are stored on NVMe or tmpfs. A space failure reports available,
required, and additional root space in MiB.

Paired DevKit transfers use a private directory under `/tmp`, which is RAM-backed on the standard DevKit. The staging directory is removed after the operation finishes, including reported installation failures and incomplete transfers. If SSH disconnects or times out during installation, the directory is retained and its path is printed because the remote installer may still be running.

Run `./install.sh` as your regular SDK or DevKit user. The launcher requests `sudo` privileges automatically and preserves only the explicit pairing, DevKit authentication, sync preference, and SSH-agent settings. Paired deployment continues to use the invoking SDK user’s SSH identity.

## Progress and logs

Standalone Core and PCIe installation use the same live terminal renderer. Each standalone step keeps its numbered phase and elapsed time, with up to 10 recent command-output lines refreshed every second and clipped to the terminal size. Redirected output uses plain 15-second activity updates. Detailed reports remain in a private `/tmp/neat-core-offline-*.log` file, including on failure. Use `NO_COLOR=1 ./install.sh` to disable colors or `./install.sh --verbose` to stream the full log. Repeated compatibility checks protect the package transaction; they are separate from installation.
