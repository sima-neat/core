# Offline standalone Neat Core installation

Use this bundle on an ARM64 Modalix DevKit already provisioned with Platform 3.0
and eLxr 26.04. Platform firmware and OS provisioning remain prerequisites.
The DevKit must provide Python 3, `unshare`, and `timeout`. Stop applications
using Neat before upgrading. Allow root access and sufficient free disk space.

Copy `neat-core-offline-installer.tar.gz` to the DevKit, verify its SHA-256
against `metadata-offline.json`, then run:

```sh
tar -xzf neat-core-offline-installer.tar.gz
cd neat-core-offline
sudo ./install.sh --check
sudo ./install.sh
sudo ./install.sh --verify
/opt/sima/neat/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__)'
```

The archive includes the Core and matching Scheduler Debian packages, installer
lifecycle scripts, local APT dependency repository, Python wheels and their
dependencies, compatibility metadata, and a checksum inventory. Installation
uses an isolated network namespace and only the bundled APT repository and
wheelhouse. It does not need sima-cli, a PCIe host, SSH, or Internet access.
`--check` verifies integrity and compatibility and simulates the package
transaction without installing. `--verify` checks installed package versions,
APT health, Python dependencies, and the PyNeat import. These checks do not prove
model inference; validate your application separately on the target.

The installer retains the published Core platform admission, package ownership,
and firmware lifecycle checks. It does not update optional Sentinel telemetry.
The isolated Python environment is `/opt/sima/neat/pyneat`; activate it with
`source /opt/sima/neat/pyneat/bin/activate`.

## Building and publication

The PCIe Offline Bundles workflow builds this standalone archive alongside the
four PCIe variants from the same immutable Core publication. All five builds
must pass before any new offline metadata is exposed. The standalone artifact
is published as `metadata-offline.json` beside Core's normal `metadata.json`;
the PCIe variants remain under their existing architecture-specific paths.
Existing completed publications are verified and never replaced.

For a local SDK-generated `dist/` publication, create a specification with
`scope: "core"` and one `targets.card` record (publication directory, metadata
SHA-256, Core revision, pinned baseline image ID, `arm64`, `elxr`, `26.04`). Run
`python3 -B assemble.py assemble --spec spec.json --output /path/neat-core-offline`.
The assembler collects against the pristine provisioned baseline, then verifies
native package installation and Python imports with Docker `--network=none`
before writing the archive. Use the provisioned baseline in `baselines.json`,
not the dependency-enriched SDK image, to avoid omitting target dependencies.
