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
also supplies the read-only GitHub App credentials used to fetch Scheduler and
sima-cli artifacts.
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

## Validation run

Run the four required workflows from the exact branches under review:

```text
gh workflow run vulcan-ci.yml --repo sima-neat/core \
  --ref <core-ref> -f validation_only=true

gh workflow run pcie-host-windows.yml --repo sima-neat/scheduler \
  --ref <scheduler-ref> -f validation_only=true

gh workflow run vulcan-ci.yml --repo sima-neat/sima-cli \
  --ref <sima-cli-ref>

gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f linux_run_id=<core-linux-run-id> \
  -f scheduler_run_id=<scheduler-run-id> \
  -f scheduler_commit=<full-scheduler-commit> \
  -f sima_cli_run_id=<sima-cli-run-id> \
  -f sima_cli_commit=<full-sima-cli-commit> \
  -f include_genai=false \
  -f validation_only=true
```

The Core run builds the ordered Scheduler/Core/wheel cohort and runs the
direct-card vision matrix. It must consume the named producer artifacts,
including the same-commit Linux card payload; do not upload replacements.
Validation artifacts may be unsigned and are not eligible for customer
publication.

## Release run

Before the signed run:

1. Run Core `Vulcan CI` for the candidate commit and retain its successful run
   ID as the Linux regression admission.
2. Run Scheduler's Windows workflow with `validation_only=false`. Use that run
   ID and its full Scheduler commit.
3. Retain a successful earlier Core Windows run whose `package-version` is
   lower than the candidate. Its `core-pciehost-windows-amd64` artifact must
   still be inside the seven-day retention window.
4. Confirm the selected sima-cli run produced `sima-cli-dist` for the exact
   sima-cli commit.
5. Confirm the signing certificate and timestamp service are available on the
   producer, and the Platform runtime DLL signatures are valid.
6. Set `include_genai=true` only after the released Windows Platform provider,
   daemon, models, and acceptance assets are installed. The lane selects the
   GenAI worker from the same admitted Linux candidate as the tensor worker.

The two release prerequisites that differ from the validation commands are:

```text
gh workflow run vulcan-ci.yml --repo sima-neat/core \
  --ref <core-ref> -f validation_only=true

gh workflow run pcie-host-windows.yml --repo sima-neat/scheduler \
  --ref <scheduler-ref> -f validation_only=false
```

Dispatch the candidate:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f scheduler_run_id=<signed-scheduler-run-id> \
  -f scheduler_commit=<full-scheduler-commit> \
  -f sima_cli_run_id=<sima-cli-run-id> \
  -f sima_cli_commit=<full-sima-cli-commit> \
  -f linux_run_id=<core-linux-run-id> \
  -f upgrade_base_run_id=<earlier-core-windows-run-id> \
  -f upgrade_base_commit=<full-earlier-core-commit> \
  -f include_genai=true \
  -f validation_only=false
```

The release workflow refuses to publish unless the exact Linux, Scheduler,
sima-cli, upgrade, signature, Windows 11, customer-install, direct-card, GenAI,
and cleanup evidence passes. Publication promotes the tested artifact; it does
not rebuild it.

## Evidence and recovery

Retain these artifacts with the release record:

- `scheduler-native-windows-amd64`;
- `core-pciehost-windows-amd64`;
- `core-pciehost-windows-hardware-<run>-<attempt>`;
- `core-pciehost-linux-admission-<run>-<attempt>`.

The hardware workflow holds both a runner-local lock and an atomic card-side
lock, then stops card workers, restores the original firmware, removes the
private Linux card candidate, uninstalls the wheel and both MSI cohorts, removes
its disposable downloads, compares the final card state with the baseline,
including the exact Platform 3.0 `/etc/buildinfo` hash, and then releases both
locks. Emergency cleanup only touches a retained card lock when its owner token
matches the retained local record. If cleanup fails, the lock
at `%USERPROFILE%\.sima\locks\neat-pcie-card0.lock` and its matching
`/tmp/neat-pcie-card0.lock` card directory are deliberately retained. Do not
delete either until the evidence logs have been reviewed and the runner,
installation root, workers, firmware, and queues have been restored manually.
