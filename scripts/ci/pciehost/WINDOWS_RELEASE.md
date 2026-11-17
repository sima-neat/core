# Windows PCIe host release runbook

This runbook covers the Windows PCIe-host workflows in Scheduler, Core, and
sima-cli. It does not replace the Platform Windows release process. Use full
commits and immutable workflow run IDs throughout; do not substitute files from
a developer machine after a workflow starts.

## Runner setup

`pcie-windows-build` is the shared native producer. It needs MSVC x64, CMake,
Ninja, WiX v3, CPython 3.14 x64, GStreamer, the Platform development/runtime
package, and an offline Python wheelhouse. Configure these repository or
environment variables:

- Scheduler: `GSTREAMER_ROOT_X86_64`, `SIMA_WINDOWS_PLATFORM_INCLUDE_DIR`,
  `SIMA_WINDOWS_PLATFORM_LIBRARY`, `SIMA_WINDOWS_PLATFORM_RUNTIME_DIR`, and
  `SIMA_WINDOWS_PLATFORM_VERSION`;
- Core: `GSTREAMER_ROOT_X86_64`, `SIMA_WINDOWS_CMAKE_PREFIX_PATH`,
  `SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH`, `SIMA_WINDOWS_PLATFORM_INCLUDE_DIR`,
  `SIMA_WINDOWS_PLATFORM_RUNTIME_DIR`, and `SIMA_WINDOWS_PYTHON_WHEELHOUSE`;
- signed runs: `SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1`,
  `SIMA_WINDOWS_SIGNING_MACHINE_STORE`, and
  `SIMA_WINDOWS_SIGNING_TIMESTAMP_URL`.

`pcie-windows-hardware` must be a Windows 11 x64 client workstation with the
supported Platform package and a directly attached Modalix card. It uses the
same Core dependency variables plus the `SIMA_WINDOWS_PCIE_*` card address,
SSH-key, model, media, and GenAI asset variables named in
`.github/workflows/pcie-host-windows.yml`. Store the SSH private key on the
runner and pass only its path. Store the card sudo credential as the protected
`DEVKIT_PASSWORD` environment secret; it is sent only on standard input and is
never placed in a command line or retained artifact. The production environment
also supplies the read-only GitHub App credentials used only when an unpublished
Scheduler or sima-cli Actions artifact is selected as a bootstrap override.
Normal runs resolve publication manifests over the approved artifact endpoint.
For optional custom-root coverage, configure
`SIMA_WINDOWS_PCIE_GENAI_MEDIA_DIRECTORY` and
`SIMA_WINDOWS_PCIE_GENAI_MEDIA_SERVE_ROOT` together from the installed Platform
provider. Supplying only one fails qualification. Platform 3.0 defaults to
`%ProgramData%\SiMa.ai\data` and the `data` root. Core resolves `%ProgramData%`
through the Windows CommonApplicationData known folder, not from an environment
variable. Release qualification always removes both overrides in a disposable
non-administrator process, binds the installed default to the daemon's `[serve]`
`data` mapping, proves create/write/remove access, and runs the candidate C++ ASR
hardware test. Custom-root coverage cannot replace this default-root admission.
Do not substitute Core's Linux `/srv/simaai/data` default.
`SIMA_WINDOWS_CMAKE_PREFIX_PATH` must include the native OpenCV CMake package,
and `SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH` must include the matching directory
that contains its runtime DLLs. The current qualified input is the official
OpenCV 5.0.0 x64 package (SHA-256
`9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63`).
Scheduler, Core, and hardware must resolve the same GStreamer identity, exact
Platform runtime DLL hashes, and exact additional runtime DLL hashes; the
workflows reject a mixed dependency cohort. The card does not need a preinstalled
tensor worker. Core CI separately builds a private ARM64 qualification-dependency
bundle from the same immutable SDK image; that bundle is not part of
`neat-core-dist` and is never published as a customer package. The Windows
workflow verifies the SDK-image build receipt and checksum-binds the bundle to
`neat-core-dist` from the admitted Linux run, extracts the combined Debian
payload into an owned private directory, and
temporarily selects its matching firmware. It restores the original firmware
and removes the private card files before releasing the card lock.

## Normal CI path

Scheduler's main CI builds Linux and Windows native profiles from one Scheduler
revision. It publishes the Windows profile at
`scheduler/<branch>/<commit>/pcie-host-windows-amd64` and advances Scheduler's
branch `latest` tag only after the DevKit and every native profile pass.

Core's main `Vulcan CI` calls the Windows workflow after the same Core revision
has passed its Linux build, DevKit, and PCIe-card gates. The Windows lane:

1. follows Core's Scheduler snap policy and resolves the immutable Windows
   profile through the runner's existing bootstrap `sima-cli`;
2. resolves an immutable published sima-cli candidate for the customer-flow
   test and independently verifies both publication manifests and resources;
3. builds the ordered Scheduler/Core/wheel cohort;
4. checksum-binds the same run's Linux payload to its private SDK dependency
   bundle, then installs that candidate only in the card qualification sandbox;
5. runs the customer CLI, C++, Python, lifecycle, fault, performance, and GenAI
   acceptance; and
6. publishes the tested Windows cohort before Core can advance its branch
   `latest` tag.

This is the normal release path. It does not require maintainers to copy workflow
run IDs between repositories. Scheduler follows Core's branch/snap policy;
sima-cli resolves from its supported `main` channel.

Release qualification requires a real older Windows cohort. It installs that
cohort, upgrades to the candidate, removes the candidate, reinstalls and verifies
the older native/Python cohort, then upgrades and verifies the candidate again.
Synthetic version metadata is not accepted as upgrade or rollback evidence.

Native CTest and Python hardware/GenAI runs emit JUnit evidence. The candidate is
rejected if the observed case set differs from the required set or contains any
skip, failure, or error; a successful process exit alone is not release evidence.
The Core producer applies the same check to the selected Scheduler profile before
it installs or repackages that dependency.

The native producer also builds a CI-only MSI that owns one file and one registry
value. A deferred custom action runs after `WriteRegistryValues` (and therefore
after `InstallFiles`), records that both are present, then returns a failure
through Windows Installer. The gate requires exit code 1603, removal of
the fixture directory, registry key/value, and product registration, and an unchanged
Scheduler/Core file inventory and product state. It reruns the installed C++
consumer and Scheduler plug-in discovery afterward. The fixture MSI is kept only
in the workflow's temporary build directory; it is not part of the customer
candidate. `msi-transaction-rollback.json` and its checksum-bound verbose log are
the release evidence for this gate.

Before the card lock or any card-side change, the Windows qualifier binds the
selected `10.0.N.2` address to its unique local `10.0.N.1` address and one up
SiMa.ai virtual Ethernet adapter. The receipt records the adapter GUID and PnP
device identity and proves that the same route remains selected after cleanup.

Windows hardware qualification applies the existing Linux PCIe performance gate:
100 warmup iterations followed by 3,000 measured iterations, with mean latency no
higher than 12 ms and throughput no lower than 440 FPS for the qualified profile.

## Validation and bootstrap overrides

Run a non-publishing validation of the entire Core cohort with:

```text
gh workflow run vulcan-ci.yml --repo sima-neat/core \
  --ref <core-ref> -f validation_only=true
```

The main workflow still runs Windows hardware qualification, but skips Linux and
Windows publication and branch promotion. A direct Windows dispatch is useful
when debugging against a completed Linux run:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f linux_run_id=<same-commit-core-vulcan-run-id> \
  -f include_genai=false \
  -f validation_only=true
```

By default that dispatch resolves published Scheduler and sima-cli revisions.
During initial bootstrap, or when intentionally testing unpublished producer
changes, supply `scheduler_run_id` with `scheduler_commit`, or `sima_cli_run_id`
with `sima_cli_commit`. Each pair is admitted against its source workflow before
the artifact is used. The Scheduler selector names the successful root
`.github/workflows/ci.yml` run: that run invokes the reusable native-Windows job
and owns its retained artifact under the same run ID. This avoids depending on a
standalone workflow that cannot be dispatched from a feature ref until the
workflow already exists on Scheduler's default branch. Do not upload replacement
files or mix a run ID with a different commit.

Bootstrap with an exact-source `workflow_dispatch` of Scheduler's root workflow
on the feature branch without entering Scheduler publication lanes:

```text
gh workflow run ci.yml --repo sima-neat/scheduler \
  --ref <scheduler-feature-ref> -f validation_only=true
```

Core admits that run only when its referenced Windows
workflow, contracts job, native Windows job, final aggregate, and retained
artifact all match the selected full Scheduler commit and run attempt.

Validation runs do not publish. `require_signatures` defaults to `false` in both
Windows workflows, while checksums, immutable manifests, exact producer identity,
and hardware cleanup remain mandatory. This does not change Windows security
policy or the separately delivered Platform driver. A customer policy that
rejects unsigned user-space code needs a signed package; the installer never
disables that policy.

Direct Windows dispatch always requires an admitted completed Linux run. The
same-run `linux_current_run` input belongs only to the reusable workflow call
from Vulcan CI after its Linux build and hardware dependencies succeed; it is
not a manual-dispatch override.

## Publishing and formal release checks

Normal branch and tag runs enter the Windows lane from Core's main CI with
`include_genai=true` and `validation_only=false`. Publication is rejected unless
the Windows 11 customer-install, direct-card, complete GenAI, and cleanup evidence
passes. Core promotes only the bytes that passed; it does not rebuild them.

Cross-version upgrade is a formal release check rather than a cost paid by every
branch commit. To run it against an earlier qualified Windows cohort, dispatch
the Windows workflow with:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f linux_run_id=<same-commit-core-vulcan-run-id> \
  -f require_upgrade=true \
  -f upgrade_base_run_id=<earlier-core-windows-run-id> \
  -f upgrade_base_commit=<full-earlier-core-commit> \
  -f include_genai=true \
  -f validation_only=false
```

The baseline must have a lower Core package version and different package bytes.
The qualifier installs it through the same customer command, upgrades to the
candidate, repeats repair, plug-in discovery, API, fault-recovery, and cleanup,
and retains evidence for both cohorts.

Set `require_signatures=true` in Scheduler and Core only when delivery policy
requires Authenticode. The optional gate requires the certificate, timestamp
service, complete Scheduler/Core signature receipts, and accepted Platform DLL
signatures. Signing is additional provenance; it does not replace tests.

Wheel installation failure deliberately preserves successful native MSI work so
the customer can retry the Python step. This is not a transaction that rolls all
three packages back. MSI repair, repeat installation, failure recovery, and clean
removal are exercised on every publishing run; the older-version upgrade is the
separate formal gate described above.

## Evidence and recovery

Retain these artifacts with the release record:

- `scheduler-native-windows-amd64`;
- `core-pciehost-windows-dependencies`;
- `core-pciehost-windows-amd64`;
- `core-pciehost-windows-hardware-<run>-<attempt>`;
- `core-pciehost-linux-admission-<run>-<attempt>`.

The hardware workflow holds both a runner-local lock and an atomic card-side
lock, then stops card workers, restores the original firmware, removes the
private Linux card candidate, uninstalls the wheel and both MSI cohorts, removes
its disposable downloads, compares the final card state with the baseline,
including the exact Platform 3.0 `/etc/buildinfo` hash, and then releases both
locks. Emergency cleanup only touches a retained card lock when its owner token
matches the retained local record. Without an owned lock, it does not stop workers,
uninstall MSIs or remove a work directory. Tutorial paths are temporary links to
the private candidate, and cleanup only removes links pointing at that candidate.
Pre-existing worker files, directories and dangling links cause staging to fail
without deleting them. If cleanup fails, the lock
at `%USERPROFILE%\.sima\locks\neat-pcie-card0.lock` and its matching
`/tmp/neat-pcie-card0.lock` card directory are deliberately retained. Do not
delete either until the evidence logs have been reviewed and the runner,
installation root, workers, firmware, and queues have been restored manually.
