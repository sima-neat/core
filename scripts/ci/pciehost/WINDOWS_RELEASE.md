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
  `SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH`, `SIMA_WINDOWS_PLATFORM_RUNTIME_DIR`,
  and `SIMA_WINDOWS_PYTHON_WHEELHOUSE`;
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
`SIMA_WINDOWS_CMAKE_PREFIX_PATH` must include the native OpenCV CMake package,
and `SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH` must include the matching directory
that contains its runtime DLLs. The current qualified input is the official
OpenCV 5.0.0 x64 package (SHA-256
`9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63`).
Scheduler, Core, and hardware must resolve the same GStreamer identity, exact
Platform runtime DLL hashes, and exact additional runtime DLL hashes; the
workflows reject a mixed dependency cohort. The card does not need a preinstalled
tensor worker. The job downloads `neat-core-dist` from the admitted Linux run,
verifies its resource inventory, extracts the Debian payload into an owned
private directory, and temporarily selects its matching firmware. It restores
the original firmware and removes the private card files before releasing the
card lock.

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
4. installs and qualifies the same run's exact Linux card payload on the card;
5. runs the customer CLI, C++, Python, lifecycle, fault, performance, and GenAI
   acceptance; and
6. publishes the tested Windows cohort before Core can advance its branch
   `latest` tag.

This is the normal release path. It does not require maintainers to copy workflow
run IDs between repositories.

Native CTest and Python hardware/GenAI runs emit JUnit evidence. The candidate is
rejected if the observed case set differs from the required set or contains any
skip, failure, or error; a successful process exit alone is not release evidence.
The Core producer applies the same check to the selected Scheduler profile before
it installs or repackages that dependency.

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
the artifact is used. Do not upload replacement files or mix a run ID with a
different commit.

Validation runs do not publish. `require_signatures` defaults to `false` in both
Windows workflows, while checksums, immutable manifests, exact producer identity,
and hardware cleanup remain mandatory. This does not change Windows security
policy or the separately delivered Platform driver. A customer policy that
rejects unsigned user-space code needs a signed package; the installer never
disables that policy.

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
