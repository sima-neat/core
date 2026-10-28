# Offline package assembly

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
installation with actual Neat payloads passed on a one-card system. Multi-card
hardware qualification remains pending; fixture results cover sequential ordering
and failure handling.

`assemble.py` builds **two separate dependency repositories**, Ubuntu 22.04 or
24.04 on AMD64 or ARM64 for the host, and ARM64 for the card. It accepts already authenticated Core
publication directories and immutable clean OCI baseline images. No sima-cli is
needed during collection or validation. Docker and Python 3.10+ are build tools.

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
| `image` | Clean baseline pinned as `repository@sha256:...` or an immutable local Docker image ID `sha256:...` |
| `architecture` | `amd64` or `arm64` for host, `arm64` for card |
| `os_id`, `os_version` | Expected `/etc/os-release` ID and VERSION_ID |

Host support is Ubuntu 22.04/24.04 on AMD64/ARM64. Choose a clean,
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

For local qualification before a corrected Scheduler publication is available,
build its native producer on Ubuntu 24.04 amd64, then select that trusted local
output with `./build.sh --all --scheduler-candidate /absolute/path/to/scheduler-native-dist`
(run from `pcie_host`). The input must be the complete producer output, including
`metadata.json`. Every declared resource is checksum-verified before loading the
producer helper; its verifier checks the distro, architecture, revision, package
ownership, and dependencies. Local metadata checks integrity, not publisher
authentication: use this option only for your own trusted producer build. Normal
builds continue to use authenticated publications. Do not edit published DEBs or
receipts to remove a dependency.

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
The archive itself is reproducible from identical locked output bytes. Identical
files use tar hard links to an earlier member, storing their contents once while
preserving all publication, repository, and wheelhouse paths and checksums.
Extraction with `tar -xzf` needs no additional preparation; card-only staging
remains self-contained.

## Offline validation

Both targets must pass before `neat-pcie-offline-installer.tar.gz` is created:

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

## Qualification inputs and producer prerequisite

A host prepared with `sima-cli -i update` may have Platform files and DKMS
installed directly, without corresponding Platform DEB registrations. Scheduler
PR #18 (`85b05060c221`) supports this layout by checking installed header hashes,
ELF architecture, and required exports. The offline release requires that corrected
producer; older publications with a hard Platform DEB dependency remain
incompatible. Do not fake dpkg registrations or alter producer receipts.

Local qualification used paired Core publications at `0c4b8df5ef48`, with the
corrected Scheduler host producer built from its source revision. The builder was
Ubuntu 24.04 with SDK image
`ghcr.io/sima-neat/sdk-develop@sha256:da219fe71f98982d08d8addc5deb3c7f74dc98f607c8c5843c270a8ab3600b63`.
Host CTests passed 14/14; the Scheduler producer passed nine native CTests,
65 Python tests, and installed/relocated consumer and plugin-load checks. Card
publication verification passed for all 20 artifacts.

Both real payloads passed native package installation, package audits, Python
installation, dependency checks, and import checks with Docker `--network=none`:

- Host: clean Ubuntu 24.04, with only the producer-built pinned Platform userspace
  library and public headers added. No driver package or invented dpkg registration.
- Card: the supplied Platform 3.0 B2121 card's read-only factory lower root,
  mounted at `/oldroot`, verified to contain no Neat packages or runtime tree.
  This excludes the mutable overlay containing the existing Neat installation.
  Archive SHA-256:
  `da453001abbb8cafe77ba06438022221e732fb7f38196497c2c7fba7cf4addc7`.
  The container omits host keys and user data, disables service startup, and removes
  the unavailable internal Platform repository. Signed Debian/eLxr repositories
  supply ordinary dependencies; Platform stays in the baseline.

The immutable baseline digests, original dpkg status, repository index hashes,
and selected package/wheel hashes are recorded in the bundle. Baselines and SDK
images are qualification/build inputs and are not part of the customer archive.

On an already installed card, preflight delegates legacy Neat replacement and
Scheduler transaction checks to the published installer. Installed system
prerequisites are retained; only missing prerequisites are requested. The host
still disallows package removal. Noninteractive host commands use `/dev/null`
for standard input so APT terminal restoration cannot stop the timeout process
group during an interactive installation. Host and card preflight allow installation of
the exact bundle versions even when Debian considers them downgrades, matching
the published installer lifecycles; producer Platform admission remains active.
Unsupported replacements fail before any target
is changed. After prerequisites are installed, the adapter regenerates its APT
plan before final admission, excluding prerequisites already installed. Card
reruns plan the complete native cohort with `--reinstall`, matching the published
installer. It retains the isolated Python environment built
from the wheelhouse instead of exposing unrelated system Python packages.

Physical multi-card and application-inference qualification remain separate from
container package/import checks. Firmware conversion and driver provisioning must
be completed before running the Neat installer.

## Physical offline installation result

The qualification archive `neat-offline-0c4b8df5ef48-v9.tar.gz` contains the paired
Core binaries above and installer tooling through `dffa5a65`. Its SHA-256 is
`17f01106c97825cb71a6f766d731b59ae032d112f495ae228a37919ecfa5a8d2`.
The native repositories, wheelhouses, baseline inputs, and container validator
remain byte-identical to the successful clean-container validation; subsequent
changes fix the physical deployment adapters.

On the prepared Ubuntu 24.04 host and one Platform 3.0 B2121 card, the bundled
installer exited zero with `installed_verified` for the host and card and an
empty error list. It discovered the card through its PCI-owned interface without
an IP argument. Package operations ran inside host/card network namespaces with
only loopback; SSH remained outside them. Exact installed versions, package
audits, Python dependency checks/imports, the host plugin, producer library
admission, and Neat-owned EV74 runtime activation passed. The run also exercised
reinstallation after the legacy Neat package migration.

Before testing, the separately provisioned card responder was repaired to use its
actual PCIe interface instead of an absent interface, with its original config
backed up. This repair is not part of the Neat installer. The host Platform
library hash and loaded driver version were unchanged; no Platform system image
was flashed. This is one-card installation acceptance, not multi-card hardware,
DMA/inference, or numerical model qualification. Release publication still needs
the corrected Scheduler producer from PR #18.

The deduplicated v9 archive is 523,275,601 bytes (499.03 MiB), compared with
609,225,029 bytes for v8: 85,949,428 bytes smaller (14.11%). Its 21 tar hard-link
entries preserve all file paths. All payload and deployment files are unchanged;
only `assemble.py` and its manifest hash differ. Extraction and full bundle
verification passed on the physical host, followed by another automatic
host/card offline installation with exit zero and both targets
`installed_verified`. Card-only staging is independently regression-tested,
including duplicates shared between the host and card trees.

## Offline publication layout

Publish `neat-pcie-offline-installer.tar.gz` and `metadata-offline.json` beside
the online `metadata.json` under the matching immutable
`core/{ref}/{revision}/pciehost/{distro}/{architecture}/` folder. Each bundle targets
one Ubuntu release (22.04 or 24.04) and one host architecture (amd64 or arm64),
and includes the ARM64 card payload. The offline metadata references the archive
with its SHA-256 and an empty `installation.script`, so downloading does not
start installation.

For example:

```sh
sima-cli neat install core/pciehost/ubuntu22/amd64@v0.4.0 -t offline
```

Replace `ubuntu22` with `ubuntu24`, `amd64` with `arm64`, and the ref with a
release or branch that publishes the offline variant. After downloading, transfer
the bundle to the prepared PCIe host, extract it, and run the bundled installer.

The isolated deployment and generated bootstrap APT configurations prefer the
uncompressed `Packages` index supplied by the flat bundle repository. This avoids
compressed-index retry errors without changing system APT configuration or
suppressing APT failures. Missing optional `InRelease`/`Release` metadata may
still produce `Ign` lines; bundle file checksums are verified before deployment.

## CI assembly and publication

Develop push runs call `.github/workflows/pcie-offline.yml` directly from **Vulcan
CI**, after the standard validation jobs and artifact promotion succeed. The
caller supplies its own run ID and admits ready same-branch publications while
that parent run remains in progress. This does not require a workflow listener on
`main`. Docs-only, failed, cancelled, PR, manual validation, and fuzz runs do not
enter this automatic publication path.

For `main`, `release-*`, and matching tags, the separate `workflow_run` listener
still requires the workflow on the default branch and a completed successful
source run. It excludes `develop` to avoid duplicate publication. Both host and
card publication jobs must have succeeded in the selected run attempt. Manual
dispatch retains its existing source-run checks.

Four native GitHub runner jobs cover Ubuntu 22.04/24.04 × AMD64/ARM64. They reuse
the authenticated Vulcan publications at the exact successful source SHA; they
do not rebuild Neat. Each job resolves dependencies in clean containers rather
than relying on packages preinstalled on the runner. AMD64 runners use QEMU only
for the ARM64 card container. A single emulated runner could assemble all four,
but would serialize the work and add emulation to host dependency validation.

`baselines.json` pins the Ubuntu multiarchitecture image manifests and the clean
Platform card image archive, including provenance and checksums. The card archive
and Platform source archive are build-only, content-addressed files directly under
`core/` in the configured artifact bucket. These are not branch directories: Vulcan
branch cleanup treats nested `core/<ref>/...` paths as Git refs and removes missing
branches. Keep baseline objects outside that branch layout. Production is seeded with this input; another Vulcan environment
must receive the same checksum-verified object before enabling this workflow.
No baseline image is shipped in the customer bundle. To update a baseline, qualify
and upload a new content-addressed object and change the manifest in review; do
not replace an existing input. The host baseline adds only the Platform userspace
library/headers built from the separate pinned Platform source archive using
the published producer's helper and policy. Every source file is verified against
that policy; a new source policy requires a reviewed matching build input. It does not install the driver. Older producers requiring Platform
DEBs are rejected, preserving the separate system-provisioning contract.

Each job validates both targets in fresh containers with `--network=none` before
uploading its archive and `metadata-offline.json.pending`. An aggregate job checks
all four source metadata digests and uploaded archive checksums before exposing
`metadata-offline.json` in each matching online package directory and invalidating
the metadata cache. Metadata has an empty install script. Failed validation never
publishes ready metadata. Existing completed publications are verified and reused;
changed source metadata or archive bytes fail instead of silently replacing a
release. Partial final publication can be retried safely; completed tuples are
retained and missing final metadata is created conditionally.

The bundle records the binary `core_revision` separately from `installer_revision`
(the trusted workflow checkout), plus source run and metadata hashes in publication
metadata. CI retains assembly logs, spec, manifest, and metadata for three days.
Container installation/import checks do not replace physical multi-card testing.

### Pre-merge matrix validation

Before the new workflow reaches the default branch, dispatch the existing
**Vulcan CI** workflow on the PR branch with `offline_source_run_id` set to a
same-branch Vulcan CI push run with successful host/card publication jobs. That
entry point skips normal CI and
calls the same four-job offline workflow in validation-only mode. The source must
still belong to this repository. Validation may start while other source CI tests
are running, but a failed or cancelled source run is rejected. Release publication
continues to require the entire source CI run to succeed.

Validation mode always rebuilds and validates all four bundles, never reuses an
existing offline release, never uploads release objects, and skips final metadata
publication. Download the resulting `offline-bundle-*` Actions artifacts within
seven days; evidence logs are retained for three days. This tests the actual matrix
before merging without exposing an unqualified release through sima-cli.


### Publish a pre-merge branch for end-to-end testing

Dispatch **Vulcan CI** on the PR branch with `offline_source_run_id` and explicitly
set `offline_publish=true`. This uses the same four native assembly jobs and clean
network-disabled validation, then uploads each archive and pending metadata beside
the source run's online packages. Only after all four jobs succeed does the final
job expose `metadata-offline.json` and invalidate its CloudFront cache. Online
`metadata.json` and branch `latest.tag` are not changed by offline publication.

```sh
gh workflow run vulcan-ci.yml --ref codex/offline-pcie-discovery \
  -f offline_source_run_id=40079421237 -f offline_publish=true
```

This explicit pre-merge mode requires successful host/card publications from the
same repository and branch. Other source CI jobs may still be queued/running;
a failed or cancelled source run is rejected. Automatic release-branch publication
continues to require complete successful CI. Without `offline_publish=true`, the
manual entry point remains validation-only. Completed offline publications are
verified and reused on rerun rather than replaced.
