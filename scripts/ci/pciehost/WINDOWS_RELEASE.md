# Windows PCIe host release runbook

This runbook covers the Windows PCIe-host workflows in Scheduler, Core, and
sima-cli. It does not replace the Platform Windows release process. Use full
commits and immutable workflow run IDs throughout; do not substitute files from
a developer machine after a workflow starts.

## Runner setup

The Windows build runs on GitHub-hosted `windows-2025` runners (Windows Server
2025 x64; GitHub offers no x64 Windows 11 image): the `Build PCIe Host windows
amd64` entry of Vulcan CI's PCIe host package matrix, and the `dependencies`
and `native` jobs of `pcie-host-windows.yml`. The image supplies MSVC x64 and
CMake/Ninja. `provision_windows_build.ps1` downloads every other input from the
HTTPS URL pinned in `windows_build_inputs.json` and checks its SHA-256 with
`Get-FileHash` before use: GStreamer 1.28.6 MSVC (runtime, development files,
`pkg-config` and zlib), OpenCV 5.0.0, nlohmann/json 3.11.3, WiX 3.14.1, the
Python wheelhouse, and the bootstrap sima-cli with its dependencies. CPython
3.14.7 comes from `actions/setup-python`, pinned by commit.

As on Linux, the bootstrap sima-cli comes from a sima-cli publication on the
approved artifact endpoint. Released sima-cli cannot read the MSI-style
`installation` metadata of Scheduler's Windows profile ("Missing 'script'"), so
the wheel is pinned to the first build that can (`sima_cli_publication` in
`windows_build_inputs.json`, sima-cli PR 396). After that change merges, re-pin
the wheel URL and SHA-256 to the merged publication, and later to the sima-cli
release that contains it.

As on Linux, the build downloads no Platform files and needs no Platform
installation. Core takes the Platform service headers (`simaai_svc.h`,
`simaai_svc_proto.h`) from the installed Scheduler package's
`pcie_service_headers` CMake component. Core loads `SiMaPCIe.dll` only at run
time, and its CTest suite uses a test-only fake. Checks that load the Scheduler
plug-in or need the Platform service (`gst-inspect-1.0` of `gstneatpciehost`,
`sima-pcie-host-doctor.cmd`) run in hardware qualification, on the card PC.
The build records only the Platform version from Scheduler's receipt, and
carries Scheduler's installed Platform record to the hardware job.
The Windows build needs Scheduler's published Windows profile for the selected
branch, or `develop` when that branch is unpublished. Without it, dependency
resolution fails.

Signed runs also need `SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1`,
`SIMA_WINDOWS_SIGNING_MACHINE_STORE`, and `SIMA_WINDOWS_SIGNING_TIMESTAMP_URL`,
and a signing certificate the runner can use.

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
Scheduler, Core, and hardware must resolve the same GStreamer identity and
exact additional runtime DLL hashes; the workflows reject a mixed dependency
cohort. Every Windows Platform build reports the same version, so the hardware
job also requires the card PC's installed Platform runtime to hold exactly the
DLLs that Scheduler's installed Platform record
(`share/SimaScheduler/host/platform-windows.json`, carried in the candidate as
`scheduler-platform-windows.json`) names, byte for byte. The card does not need a preinstalled
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

Vulcan CI builds, tests, and retains the Windows cohort on every standard
build, in the `windows amd64` entry of its PCIe host package matrix, as the
artifacts `core-pciehost-windows-amd64` and `core-pciehost-windows-dependencies`.
For a branch push, `publish-pciehost-package` publishes that cohort beside the
branch's Ubuntu packages, at `core/<branch>/<commit12>/pciehost/windows/amd64/`
with `metadata.json` last, so after promotion
`sima-cli neat install core/pciehost/windows/amd64@<branch>` resolves as it does
for Ubuntu. The publish runs on Ubuntu, from the artifact, because the hosted
Windows runner has no Vulcan publisher role. A tag's Windows release is
published only by the qualified release dispatch below, so the release path
holds only bytes that passed the card. The build receipt records the Visual
Studio, MSVC, Windows SDK, and runner image versions, which a hosted runner
cannot pin as Linux pins its image digest. A failed Windows build fails the
matrix, which blocks package metadata, branch `latest` promotion, and
validation-only runs.

Vulcan CI runs the self-hosted hardware lane only when the repository variable
`ENABLE_WINDOWS_PCIE_CI` is `true` and Vulcan CI is the top-level workflow
(not when another workflow calls it); otherwise it skips the lane, and branch
`latest` promotion and validation-only runs accept that skip.

Core's main `Vulcan CI` calls the Windows workflow after the same Core revision
has passed its Linux build, DevKit, and PCIe-card gates. The Windows lane:

1. follows Core's Scheduler snap policy and resolves the immutable Windows
   profile through the pinned bootstrap `sima-cli`;
2. resolves an immutable published sima-cli candidate for the customer-flow
   test and independently verifies both publication manifests and resources;
3. reuses the PCIe host matrix's cohort and dependency selection from the same
   run (a direct dispatch builds its own);
4. checksum-binds the same run's Linux payload to its private SDK dependency
   bundle, requires the card's `/etc/buildinfo` `DISTRO_VERSION` to equal the
   `platform-version` in `deps/manifest.json` (the Linux card lane's
   `check_card_compatibility.sh` rule), then installs that candidate only in the
   card qualification sandbox;
5. runs the customer CLI, C++, Python, lifecycle, fault, performance, and GenAI
   acceptance; and
6. publishes nothing: Vulcan CI calls it with `validation_only=true`, and while
   the lane is enabled its success is required before Core can advance its
   branch `latest` tag.

The producer, `scripts/ci/pciehost/build_windows.py`, verifies every Scheduler
resource against its published metadata and requires the Scheduler MSI's
`ProductVersion` to equal its metadata `package-version`. It builds Core and the
Python extension, runs the non-hardware CTest suite, verifies that every
generated PE file is x64, tests a relocated CMake consumer, installs and repeats
the ordered Scheduler-to-Core MSI cohort, imports the wheel from a fresh virtual
environment using only the pinned wheelhouse, and fails if cleanup leaves the
shared installation root behind.

The hardware qualifier serves the candidate from disposable loopback registries
and runs the documented `sima-cli neat install` command for a native-only
install, a missing-wheel download failure, a checksum-valid but invalid wheel
after native installation, managed-environment recovery with the real wheel,
and a repeated install. It records the Windows version, elevation, Smart App
Control, effective AppLocker visibility and driver-signing policy (Secure Boot,
boot test-signing and integrity-check options, WDAC and HVCI enforcement)
without changing policy, and removes Python, Core and Scheduler in reverse order.

This is the normal validation path. It does not require maintainers to copy workflow
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
consumer afterward. The fixture MSI is kept only
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

The main workflow still runs Windows hardware qualification when the lane is
enabled, and skips Linux publication and branch promotion; Vulcan CI never
publishes Windows packages. A direct Windows dispatch is useful
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
the artifact is used. A selected run can come from any branch, so both Vulcan CI
and the Windows workflow reject a selector unless `validation_only=true`;
publishing runs use only published dependencies.
The Scheduler selector names the successful root
`.github/workflows/ci.yml` run. That one run owns the retained Scheduler client
candidate, all four Linux native profiles, and the Windows profile. Core uses
those run-local candidates for code hygiene, the matching Linux/ARM package
build, and Windows qualification. The selector does not reach the SDK build:
Vulcan CI's `Build Core on Vulcan` job runs `build.sh`, which still resolves
Scheduler from the published channel under Core's snap policy. A bootstrap
therefore still needs Scheduler's same-named branch to have a promoted `latest`
tag: push the Scheduler branch first and wait for its CI to promote it before
running Core. If Scheduler's branch index lists the branch without a promoted
`latest`, the SDK build fails and every later job, including the Windows lane,
is skipped; `build.sh` falls back to `develop:latest` only when the index
confirms the branch is absent. Each consumer admits the exact root workflow,
reusable workflow, job, artifact, run attempt, repository, and full commit
before downloaded code can execute. This also avoids depending on a standalone
workflow that cannot be dispatched from a feature ref until the workflow already
exists on Scheduler's default branch. Do not upload replacement files or mix a
run ID with a different commit.

Bootstrap with an exact-source `workflow_dispatch` of Scheduler's root workflow
on the feature branch without entering Scheduler publication lanes:

```text
gh workflow run ci.yml --repo sima-neat/scheduler \
  --ref <scheduler-feature-ref> -f validation_only=true
```

Core admits that run only when the required reusable workflow, producing job,
final aggregate, and retained architecture-specific artifact all match the
selected full Scheduler commit and run attempt.

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
`include_genai=true` and `validation_only=true` when the lane is enabled. They
qualify the Windows cohort and gate branch `latest` promotion on that
qualification. The card lane never publishes. Branch pushes publish their
matrix build to the branch channel, as described above, but a push cannot
supply the upgrade baseline or first-release exemption that release
publication requires. Windows release packages are published only by a direct
`pcie-host-windows.yml` dispatch from the Neat release tag `v<package-version>`
(the `package-version` in `deps/manifest.json`) with `validation_only=false`
and either `require_upgrade=true` (below) or, for the first Windows release
only, `first_release_upgrade_exemption=true` with a non-empty
`first_release_upgrade_exemption_reason`. A publishing dispatch from any other
ref fails in the admission job, before anything is built. Every publishing
dispatch also needs `linux_run_id=<same-commit-core-vulcan-run-id>` and
`include_genai=true`; without `include_genai=true` the run fails only at the
hardware step. Publication is rejected unless
the Windows 11 customer-install, direct-card, complete GenAI, and cleanup evidence
passes. Core publishes only the bytes that passed; it does not rebuild them.

The publish job uploads `metadata.json`, its resources and their checksum files
to `core/<tag>/<commit12>/pciehost/windows/amd64/`, the layout of the Ubuntu
host packages, with the same credentials and KMS encryption.
`scripts/ci/pciehost/windows_release.py` owns that layout; the qualifier's local
registry and the upgrade baseline use the same helper. The job requires
`core/<tag>/latest.tag` to name this commit already. Vulcan CI's
`promote-latest-tag` writes it when the tag's Linux release passes, so run the
Windows dispatch after that run. `metadata.json` is uploaded last, after every
uploaded file is read back and compared with the qualified candidate. Customers
then install with:

```text
sima-cli neat install core/pciehost/windows/amd64@v<package-version>
```

Before the first production publication, rehearse against staging. The Vulcan
environment comes from the `VULCAN_ENV` variable, as for the Ubuntu packages, so
the rehearsal is a release-tag dispatch while that variable selects `staging`,
after the same tag's Linux release has promoted `latest.tag` in staging. Then,
on a clean Windows 11 host, run the customer command against staging and
confirm that both MSIs and the wheel download and install:

```text
sima-cli neat install core/pciehost/windows/amd64@v<package-version> --env staging
```

Cross-version upgrade is a formal release check rather than a cost paid by every
branch commit. To run it against the previous published Windows release,
dispatch the Windows workflow with:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref v<package-version> \
  -f linux_run_id=<same-commit-core-vulcan-run-id> \
  -f require_upgrade=true \
  -f upgrade_base_tag=v<previous-windows-release> \
  -f include_genai=true \
  -f validation_only=false
```

The hardware job resolves the baseline tag's commit from the repository and
downloads that release's `metadata.json` and resources from its published path,
checking `latest.tag` and every SHA-256. The baseline therefore does not expire
with an Actions artifact. A published release carries no CI receipts; its own
release gate checked them before publication. The baseline must have a lower
Core package version and different package bytes.
The qualifier installs it through the same customer command, upgrades to the
candidate, repeats repair, plug-in discovery, API, fault-recovery, and cleanup,
and retains evidence for both cohorts.

Set `require_signatures=true` in Scheduler and Core only when delivery policy
requires Authenticode. The optional gate requires the certificate, RFC 3161
timestamp service, and complete Scheduler/Core signature receipts. Platform DLLs
are the Platform installer's and are not part of this gate. Signing is
additional provenance; it does not replace tests.

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
