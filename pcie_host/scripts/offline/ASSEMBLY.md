# Offline package assembly (draft)

## Delivery scope

There are two independent offline deliverables:

| Deliverable | Contents and responsibility |
| --- | --- |
| System package | Host Linux driver and compatible card firmware/update payload, with installation of both sides. Owned by Platform provisioning. |
| Neat package | Host Neat libraries, card Neat libraries, their native/Python runtime dependencies, and one discovery-driven installer. This is the scope of PR #1135. |

Do not combine the deliverables. Firmware payload format and the system-package
installer are outside this PR. A compatible system package must already be
installed, with PCIe networking, discovery responders, and SSH access provisioned.
The Neat installer checks those prerequisites and reports actionable errors; it
does not flash firmware, install drivers, or repair Platform networking.

The intended customer workflow is download, transfer if needed, extract, and run
one installer on the host. Customers do not need Docker or Internet access. The
distributable archive contains both architectures, local dependency repositories,
checksums, compatibility metadata, and the installer; no package downloads occur
during installation.

The installer must perform these steps:

1. Verify bundle integrity and host/system compatibility.
2. Inventory every attached SiMa PCI card and discover endpoints only through
   their sysfs-associated interfaces. No IP arguments or card-selection flags.
3. Preflight every card, including unambiguous endpoint mapping, SSH access,
   compatible system packages, disk space, and offline dependency resolution,
   before changing the host or any card.
4. Install and verify host Neat once using the existing install lifecycle.
5. Transfer the card payload, install, and verify each card sequentially. Continue
   after an individual card failure while host/topology remain valid; report every
   card's result and return nonzero if any card failed.

Keep reruns state-based; no persistent workflow engine, parallel updates, or
cross-package coordinator. Installation evidence must include the real paired
Neat payload with Internet access unavailable while local PCIe SSH remains usable.
A clean Platform baseline is a build/qualification input, not an additional
customer deliverable or a reason to implement firmware provisioning here.

**Current status:** the assembler includes `install.py`, the discovery-driven
all-card coordinator, and `deploy.py`/`lifecycle.sh` adapters for the published
installers. See [customer installation instructions](INSTALL.md). Physical
all-card qualification with actual Neat payloads remains pending; fixture results
are not a substitute for that acceptance test.

`assemble.py` builds **two separate dependency repositories**, Ubuntu 24.04 amd64
for the host and ARM64 for the card. It accepts already authenticated Core
publication directories and immutable clean OCI baseline images. No sima-cli is
needed during collection or validation. Docker and Python 3.11+ are build tools.

The bundled installer deploys Neat through the existing host/card installer
lifecycles. It does not configure discovery, flash system firmware, install host
drivers, or bypass producer Platform admission. Installation subprocesses run
inside isolated network namespaces; PCIe discovery and SSH run outside them.

## Inputs

Prepare a JSON spec with exactly `host` and `card` targets. Each target requires:

| Field | Meaning |
| --- | --- |
| `publication` | Local directory containing `metadata.json` and every declared resource |
| `metadata_sha256` | SHA-256 of the metadata from an authenticated publication transport |
| `core_revision` | Exact Core revision, 12–40 hex characters; must match both publications |
| `image` | Clean baseline OCI reference pinned as `repository@sha256:...` |
| `architecture` | `amd64` for host, `arm64` for card |
| `os_id`, `os_version` | Expected `/etc/os-release` ID and VERSION_ID |

Initial host support is exactly `ubuntu`, `24.04`, `amd64`. Choose a clean,
provisioned Platform 3.0 baseline image for the actual card; record its firmware
provenance separately. Plain Debian ARM64 is useful for tooling integration tests
but is **not** a substitute for Platform 3.0 qualification. Baselines must include
the separately provisioned Platform prerequisites and their repository trust
configuration, without an existing Neat installation. Never export an arbitrary
CI runner or a card with a preinstalled runtime as the clean baseline.

The metadata digest is a trust anchor supplied by the caller, not authentication
by itself. The assembler checks every declared payload against the metadata before
starting containers. Native Core/Scheduler producer helpers and receipts remain
in the bundle. Where present, the host producer's `pcie_delivery.py verify` also
checks the native composition before dependency collection.

## Build

```sh
python3 pcie_host/scripts/offline/assemble.py assemble \
  --spec /absolute/path/to/spec.json \
  --output /absolute/path/to/new-bundle
```

The output must be a new directory on a filesystem Docker can bind-mount.
Collection is online; installation validation is offline. Failed builds retain
an explicitly incomplete working directory for diagnosis and produce no success
archive. Do not distribute that directory as an installable bundle.

For each architecture, a fresh container records the baseline dpkg status **before**
installing collection tools. APT resolves all publication DEBs plus Python/venv,
pip, SSH, iproute2, GStreamer inspection, and certificate prerequisites against
that original status. It downloads the missing transitive dependencies into an
empty cache. This avoids treating collection tools or incidental CI packages as
preinstalled target dependencies. Dependencies already provided by the exact
baseline remain explicit in `baseline.status`; the bundle is not a general OS
replacement or driver installer.

Pip resolves the publication wheels and transitive dependencies with binary wheels
only, using the target's native Python ABI. No source builds are allowed. Exact
resolved wheel hashes, DEB versions/hashes, Python ABI, APT index hashes, baseline
image digest, and baseline dpkg status are recorded. Each target gets its own local
APT repository and wheelhouse. Resolution is frozen in these outputs; collection
against mutable upstream repositories on another day can select newer versions.
The archive itself is reproducible from identical locked output bytes.

## Offline validation

Both targets must pass before `<output>.tar.gz` is created:

1. Verify the complete bundle file inventory and hashes.
2. Start a fresh container of the same baseline with `--network=none` and a
   read-only bundle mount. Check that its pristine dpkg status matches collection.
3. Bootstrap Python from the local repository only. A private APT configuration
   uses empty lists/caches and excludes the container's configured sources,
   preferences, and APT configuration hooks.
4. Simulate and install the native package transaction with `--no-remove`, then
   run `apt-get check` and `dpkg --audit`.
5. Create a fresh venv, install the target wheel and dependencies using
   `--no-index`, the bundled wheelhouse, and no download cache; run `pip check`.
   Real `pyneatpcie`/`pyneat` payloads also require an import smoke test. Full
   runtime lifecycle verification runs on the prepared physical system through
   `install.py`; containers do not provide its hardware or running services.

Package maintainer scripts run inside the network-disabled container too, so
implicit downloads cannot succeed. No host network/firewall changes are required.
This is package-installation evidence, not firmware activation, systemd lifecycle,
physical PCIe inference, or multi-card qualification. An image without necessary
Platform packages/libraries should fail rather than generate a misleading success.
Package DEBs retain their redistribution notices; release owners must still review
third-party redistribution and maintainer-script behavior before publication.

Recheck an existing bundle with the repository's tool:

```sh
python3 pcie_host/scripts/offline/assemble.py validate /absolute/path/to/bundle
```

The customer entry point is `sudo python3 -B install.py`, with optional
`--preflight-only`. It authenticates SSH against provisioned host keys and probes
only PCI-owned interfaces. Complete preflight precedes package changes, and card
failures are collected while remaining cards proceed if host/topology remain valid.
Unknown SSH keys and duplicate card IPs require correction in system provisioning.

## Tests and publication prerequisite

```sh
python3 -m pytest -q tests/tools/test_offline_pcie_*.py
python3 tests/tools/run_offline_pcie_assembly_integration.py /absolute/path/to/new-test-work
```

The Docker integration test uses synthetic native/Python payloads with real
Ubuntu/Debian dependency packages. It exercises both architectures and both offline
installation paths. It is not a test of actual Neat binaries or Platform hardware.

Core host publication now enumerates checksum-verified `metadata.json` resources,
not extension globs. This includes producer-owned `platform-source.tar.gz` and
prevents marking an incomplete native publication ready. Metadata still uses the
existing `.pending` staging and aggregate verification process.

## Initial validation evidence and remaining prerequisites

- The fixture integration built both repositories and passed native and Python
  installation on Ubuntu 24.04 amd64 and Debian 13 arm64 with Docker networking
  disabled. This validates repository/closure mechanics only.
- Published Core `faf932e09bb8` could not resolve its host dependency on
  `simaai-mla-drv-dkms=2026.10.12.00413+6e585a3` on plain Ubuntu. The older
  publication still contains the Platform host package. Runtime assembly must
  not absorb driver provisioning; the collector now rejects such host payloads.
- Candidate Core `e1c667f05dea` passed the owning native closure verifier after
  recovering `platform-source.tar.gz` from Scheduler `26c265e6c9f0` and checking
  the exact declared hash. Core's own publication lacked that object; the
  publication-list regression fix above addresses this omission for future runs.
- That newer candidate requires preinstalled Platform host package
  `simaai-pcie-host=2026.10.12.00413+6e585a3`; plain Ubuntu is not a valid
  post-provisioning baseline and resolution correctly failed.
- Actual `e1c667f05dea` ARM64 packages failed resolution on plain Debian 13 for
  Platform dependencies `simaai-log`, `simaai-trace`, `simaai-mlart-modalix`,
  `simaai-heap`, and `tvm`. These must come from a qualified clean Platform
  baseline/repository, not copied from an already modified card.

No actual Neat release archive or full hardware-offline qualification is claimed.
The installed sima-cli on the qualification host now supports signed eLxr 3.0
SWUpdate bundles and a separate Linux host update path. `sima-cli -i update` on
a Linux host selects a checksum-verified `sima_pcie_host_pkg.sh` from the public
daily-build catalog. Do not run it as part of Neat installation: it applies
Platform drivers/libraries and may require a reboot.

The host installer downloaded for SDK build `3.0.0_daily_develop_B2308` was
verified against the catalog and inspected without running its payload. It
installs files and DKMS directly; it does not install the corresponding Platform
DEBs. This does not satisfy the current Scheduler producer's exact Debian
package dependency/admission contract. A host prepared with that script alone
must not be assumed compatible with this Neat bundle. Platform provisioning and
the Scheduler producer need an agreed, verifiable package contract; do not fake
dpkg registrations or weaken the dependency checks in the offline installer.

The locally built host payload at Core `e8d2ba7cd23f` still requires
`simaai-pcie-host=2026.10.12.00413+6e585a3`. The inspected public catalog had no
matching host package; B2308 supplies `2026.10.21.00415+a1e551a` instead. The
required older package is not interchangeable with a newer daily build merely
because both are Platform 3.0. Actual host preflight with networking disabled
failed on the missing Platform package before mutation. PCI-scoped discovery
also received no reply from the supplied card; its responder configuration names
an absent interface.

Use a qualified full eLxr firmware/rootfs image as the clean baseline source;
an in-place package update does not establish that Neat is absent. A PCIe-capable
image and its host driver compatibility still need verification before flashing.
No firmware update or card reconfiguration was performed for these checks.
