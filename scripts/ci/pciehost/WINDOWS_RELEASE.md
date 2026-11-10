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
  `SIMA_WINDOWS_PLATFORM_RUNTIME_DIR`, and `SIMA_WINDOWS_PYTHON_WHEELHOUSE`;
- signed runs: `SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1`,
  `SIMA_WINDOWS_SIGNING_MACHINE_STORE`, and
  `SIMA_WINDOWS_SIGNING_TIMESTAMP_URL`.

`pcie-windows-hardware` must be a Windows 11 x64 client workstation with the
supported Platform package and a directly attached Modalix card. It uses the
same Core dependency variables plus the `SIMA_WINDOWS_PCIE_*` card address,
SSH-key, model, media, worker-checksum, and GenAI asset variables named in
`.github/workflows/pcie-host-windows.yml`. Store the SSH private key on the
runner and pass only its path. The production environment also supplies the
read-only GitHub App credentials used to fetch Scheduler and sima-cli artifacts.
Scheduler, Core, and hardware must resolve the same GStreamer identity and exact
Platform runtime DLL hashes; the workflows reject a mixed dependency cohort.

## Validation run

Run the three producers from the exact branches under review:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/scheduler \
  --ref <scheduler-ref> -f validation_only=true

gh workflow run vulcan-ci.yml --repo sima-neat/sima-cli \
  --ref <sima-cli-ref>

gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f scheduler_run_id=<scheduler-run-id> \
  -f scheduler_commit=<full-scheduler-commit> \
  -f sima_cli_run_id=<sima-cli-run-id> \
  -f sima_cli_commit=<full-sima-cli-commit> \
  -f include_genai=false \
  -f validation_only=true
```

The Core run builds the ordered Scheduler/Core/wheel cohort and runs the
direct-card vision matrix. It must consume the named producer artifacts; do not
upload replacements. Validation artifacts may be unsigned and are not eligible
for customer publication.

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
   daemon, card worker, models, and acceptance assets are installed.

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

The hardware workflow always stops card workers, uninstalls the wheel and both
MSI cohorts, removes its disposable downloads, compares the final card state
with the baseline, and then releases the local lock. If cleanup fails, the lock
at `%USERPROFILE%\.sima\locks\neat-pcie-card0.lock` is deliberately retained.
Do not delete it until the evidence logs have been reviewed and the runner,
installation root, workers, firmware, and queues have been restored manually.
