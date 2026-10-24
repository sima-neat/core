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

**Current status:** dependency assembly and container validation exist. The
customer installation entry point and physical all-card validation are not yet
implemented. The remaining sections describe the available assembler.

`assemble.py` builds **two separate dependency repositories**, Ubuntu 24.04 amd64
for the host and ARM64 for the card. It accepts already authenticated Core
publication directories and immutable clean OCI baseline images. No sima-cli is
needed during collection or validation. Docker and Python 3.11+ are build tools.

This is the package-assembly portion of #1132, not the automatic all-card installer.
It does not change hardware, configure discovery, or bypass the existing installers'
Platform admission and runtime lifecycle rules. Use the existing installers for
runtime deployment until their strict offline integration is complete.

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
   Real `pyneatpcie`/`pyneat` payloads also require an import smoke test.

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

All-card discovery and sequential install/verify orchestration remain a subsequent
integration step; there is deliberately no user-facing `install.sh` in this draft.

## Tests and publication prerequisite

```sh
python3 -m pytest -q tests/tools/test_offline_pcie_assembly.py tests/tools/test_offline_pcie_discovery.py
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
Inspection of sima-cli at `6d17551f8ff74a4db7f16f32733c6e0586fd6400`
confirms that its SWUpdate path is for Yocto. For eLxr, `update_firmware`
dispatches to local `simaai-ota`, rejects remote updates, and requires Internet
connectivity before refreshing APT metadata. The supplied Platform 3.0 card is
eLxr, so invoking that command is not an offline SWU provisioning solution.
Use a qualified full eLxr firmware/rootfs image as the clean baseline source;
an in-place package update does not establish that Neat is absent. A PCIe-capable
image and its host driver compatibility still need verification before flashing.
No firmware update or card reconfiguration was performed for these checks.

Next, provide digest-pinned clean images representing the completed offline
Platform provisioning step, then rerun assembly with complete matching Core
publications. Do not add dummy packages, disable dependency checks, or weaken
producer admission to make these tests pass.
