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
`neat.cmd --json` and `neat.cmd status` need neither: the build runs them
from a directory other than `bin` after installing the MSIs and requires both
installed products at their cohort versions and no card (`neat-status.log`).
Hardware qualification runs them again as the customer after installation and
requires the card at its address with the candidate's card packages
(`customer-neat-status.log`). The build's CTest suite runs the PCIe host
Python contract tests on Windows, including the tests that skip on Linux and
run the shipped scripts under `cmd.exe` from another directory.
The Windows build needs Scheduler's published Windows profile for the selected
branch, or `develop` when that branch is unpublished. Without it, dependency
resolution fails.

Authenticode signing happens only in a run with `require_signatures=true`, in
the `native` job of Scheduler's and of Core's `pcie-host-windows.yml`. Both run
`signtool sign /sha1 <thumbprint>` against the runner's Windows certificate
store and read three Actions variables, which must be set in both repositories.
Core's `native` job has no environment, so set Core's at repository or
organization level; Scheduler's job runs in the `production` environment.

- `SIMA_WINDOWS_SIGNING_CERTIFICATE_SHA1`: the certificate thumbprint, 40 hex
  digits. Scheduler and Core must sign with the same certificate; Core's
  release admission rejects a Scheduler package from another publisher.
- `SIMA_WINDOWS_SIGNING_MACHINE_STORE`: `true` selects the certificate from
  `LocalMachine\My` (`signtool /sm`); otherwise signtool uses `CurrentUser\My`
  of the runner account.
- `SIMA_WINDOWS_SIGNING_TIMESTAMP_URL`: an HTTPS RFC 3161 timestamp service.
  Unset, it is `https://timestamp.digicert.com`.

The certificate, with a private key the runner account can use without a
prompt, must already be in that store when the job starts. Neither workflow
installs one, and GitHub-hosted `windows-2025` runners start from a clean
image, so a signed run fails at `signtool` there. Before the first signed
release, either run both `native` jobs on a self-hosted Windows signing runner
that holds the certificate, or add a step to both jobs that makes it available
from the organization's signing service. Publicly trusted code-signing keys
must stay in hardware or a cloud HSM, so a PFX file stored as a secret is not an
option.

The card runner must be a Windows 11 x64 client workstation with the
Platform package that `sima-cli install drivers/windows` installs and a
directly attached Modalix card. Like the
Linux card job, the Windows card job selects its runner by labels only:
`self-hosted`, `Windows`, `X64`, and `pcie`.

Provision the host once before its first run:

1. Install the Platform package, which provides the drivers and the
   `SiMaMlaDaemon` service, with `sima-cli install drivers/windows` from
   PowerShell started as administrator after `sima-cli login`. The workflows
   never download or install it. Attach the card, and install GStreamer 1.28.6 MSVC
   and CPython 3.14 x64 for all users, as for a customer host. The card itself
   must pass the offline installer's preflight: eLxr 26.04 with `unshare`,
   `flock`, `timeout`, `python3`, `tar`, and `sha256sum`; a clean `dpkg --audit`;
   and free space on `/` and `/var/tmp` for the bundle. Its `sima` sudo password
   is the `DEVKIT_PASSWORD` secret, as on Linux.
2. Register the runner with the organization's `create-gha-runner.bat`. Keep
   its default labels and enter `pcie` at its labels prompt.
3. Log the runner service on as a local administrator account instead of the
   script's NETWORK SERVICE default. Grant that account "Log on as a service"
   (Local Security Policy, User Rights Assignment), then restart the service:

   ```text
   sc.exe config <runner service> obj= .\<administrator> password= <password>
   sc.exe stop <runner service>
   sc.exe start <runner service>
   ```

   The card job installs per-machine MSIs and creates a temporary standard
   user, so it fails at its first step on an unelevated runner. Linux PCIe hosts
   likewise need sudo for their runner user.

Every self-hosted Windows runner labelled `pcie` must have a card attached:
the job matches any of them.

The card job needs no repository variables, as on Linux. The card address is
`10.0.0.2`. The job provisions `%USERPROFILE%\.ssh\sima_neat_pcie_ed25519` with
`pcie-setup.ps1`. If the card does not accept the key yet, it enrolls it with
the `DEVKIT_PASSWORD` secret, as `pcie-setup.sh` does on Linux. Windows
OpenSSH cannot take a password non-interactively, so
`enroll_windows_card_key.py` does that one step with paramiko from the build's
wheelhouse. `prepare_windows_card_inputs.py` then resolves the rest in-job,
from the sources that Linux's `resolve_assets.sh` and
`prepare_pcie_genai_models.sh` use:

- the YOLOv8s (`yolov8s` `w8a8`) and MLA-only INT8 and BF16 (`yolo26n` `w8a8` and
  `w16a16`) models from the Model Browser `main` catalog, through
  `prepare_model_browser_fixture.py` and `sima-cli models download`;
- the test image from its download URL;
- the box-decode image and GenAI test fixtures from the checkout;
- the GenAI models from Hugging Face, cached under
  `%ProgramData%\sima-neat-ci\models_genai`.

Downloads are cached under `%ProgramData%\sima-neat-ci\pciehost-assets`, except the
Model Browser packages: like Linux, every run downloads the current builds again, into
its `model-browser` directory. Both caches
are machine-wide, where Linux keeps the models in the runner's home, because the job's
temporary standard user must read the GenAI models and fixtures and cannot read the
runner's profile.

It writes the hardware asset manifest from the files it resolved. The
wheelhouse and OpenCV runtime DLLs come from the build's
`core-pciehost-windows-card-inputs` artifact. GStreamer comes from its
installed location. The Platform runtime comes from the `SiMaMlaDaemon`
service path, as the installed `sima-pcie-host-env.cmd` finds it. The only
optional variable is `SIMAPCIE_PCIEHOST_CARD_GST_DEBUG`, shared with Linux.

The card sudo credential is the protected `DEVKIT_PASSWORD` secret. As on
Linux it is step environment (`SIMAPCIE_CARD_SUDO_PASSWORD`), and is never
placed in a command line or retained artifact. The production environment also supplies the read-only
GitHub App credentials, used only when an unpublished Scheduler or sima-cli
Actions artifact is selected as a bootstrap override. Normal runs resolve
publication manifests over the approved artifact endpoint.

The qualifier still accepts `SIMAPCIE_GENAI_MEDIA_DIRECTORY` and
`SIMAPCIE_GENAI_MEDIA_SERVE_ROOT`, together, for a custom media root; CI sets
neither, as on Linux. Platform 3.0 defaults to `%ProgramData%\SiMa.ai\data`
and the `data` root. Core resolves `%ProgramData%` through the Windows
CommonApplicationData known folder, not from an environment variable. Release
qualification always removes both overrides in a disposable non-administrator
process, binds the installed default to the daemon's `[serve]` `data` mapping,
proves create/write/remove access, and runs the candidate C++ ASR hardware
test. Do not substitute Core's Linux `/srv/simaai/data` default.
The current qualified OpenCV input is the official OpenCV 5.0.0 x64 package
(SHA-256 `9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63`).
Scheduler, Core, and hardware must resolve the same GStreamer identity and
exact additional runtime DLL hashes; the workflows reject a mixed dependency
cohort. Every Windows Platform build reports the same version, so the hardware
job also requires the card PC's installed Platform runtime to hold exactly the
DLLs that Scheduler's installed Platform record
(`share/SimaScheduler/host/platform-windows.json`, carried in the candidate as
`scheduler-platform-windows.json`) names, byte for byte. That record names the
DLLs of the package `sima-cli install drivers/windows` installs, so a card PC
with another Platform build fails the hardware job.

The card has no internet access, so the lane installs the candidate the way the
Linux offline installer installs a card without internet. Vulcan CI's
`build-pcie-card-offline-bundle` assembles the standalone Core offline bundle
from this run's `neat-core-dist` with `pcie_host/scripts/offline/ci.py`. The
bundle holds the candidate's card packages and their apt closure against the
qualified clean card image, validated with `--network=none`. `card_candidate`
checks that the bundle was assembled from the admitted Linux candidate. On the
card, the qualifier stages it under `/var/tmp/neat-offline-*` and runs the
bundle's own `deploy.py card preflight`, `install`, and `verify`, the same
installer lifecycle as `install.py` on Linux. It then checks the runtime as
`verify_card_runtime.sh` does. The card keeps the installed candidate after
the run, as a Linux card does. A direct release dispatch takes the bundle from
the admitted Linux run.

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

Vulcan CI runs the self-hosted hardware lane automatically when Vulcan CI is
the top-level workflow (not when another workflow calls it). The Linux card lane
runs for every non-docs push on two runners; the Windows lane has one card runner
and takes about 23 minutes, so it runs for every push to `develop`, every release
tag, every manual run, and a feature branch push only when the branch changes
the Windows host's inputs relative to `develop`: `pcie_host/`,
`scripts/ci/pciehost/`, `.github/workflows/pcie-host-windows.yml` or
`deps/manifest.json` (the `windows_pcie` output of `detect-changes`). A run that
includes the lane requires it to pass before branch `latest` promotion and in
validation-only runs; a run that skips it accepts the skip. A run waits for the
runner while it is busy; if the runner is offline, GitHub fails the job after
24 hours in the queue.
Without a `sima_cli_run_id` selector, the lane's customer-flow sima-cli is the
publication that the bootstrap pins (`sima_cli_publication` in
`windows_build_inputs.json`), because no released sima-cli reads the MSI-style
metadata yet. A direct release dispatch still resolves sima-cli `main`.

### Run the Windows card lane by hand

A manual run always includes the card lane, so use it to test a branch whose
push skipped the lane:

1. In GitHub, open sima-neat/core, then **Actions**, then **Vulcan CI**.
2. Click **Run workflow** and select the branch to test. Leave the inputs at
   their defaults.
3. Click **Run workflow**.

From a shell:

```text
gh workflow run vulcan-ci.yml --repo sima-neat/core --ref <branch>
```

The run is a normal Vulcan CI run of that branch with the card lane added. It
builds the Windows cohort once in the PCIe host matrix, then the card lane tests
those bytes on the self-hosted runner, beside the Linux DevKit and card tests. It publishes
and promotes as a push would, and promotion fails if the card lane fails.
Add `-f validation_only=true` to test without publishing or promoting. A new
push to the same branch cancels the run, as it does any branch run.

Core's main `Vulcan CI` calls the Windows workflow once the same Core revision's
Linux builds succeed, beside its DevKit and PCIe-card tests. The Windows lane:

1. follows Core's Scheduler snap policy and resolves the immutable Windows
   profile through the pinned bootstrap `sima-cli`;
2. resolves an immutable published sima-cli candidate for the customer-flow
   test and independently verifies both publication manifests and resources;
3. reuses the PCIe host matrix's cohort and dependency selection from the same
   run (a direct dispatch builds its own);
4. binds the same run's offline card bundle to its Linux payload, and requires
   the card's `/etc/buildinfo` `DISTRO_VERSION` to equal the `platform-version`
   in `deps/manifest.json` (the Linux card lane's `check_card_compatibility.sh`
   rule). It then installs that candidate on the card from the offline bundle;
5. runs the customer CLI install, standard-user activation, installed tutorials,
   and Linux's C++ CTest and pyneatpcie cases with their order, arguments and
   timeouts, printing one result line per test, continuing after a failure, and
   adding a results table to the job summary; and
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
and runs the documented `sima-cli neat install` command once, as Linux installs
its host package once; the card installs over SSH meanwhile. Release
qualification (`validation_only=false` or `release_qualification=true`,
`--release-qualification`) adds the
Windows-only checks that have no Linux equivalent: a native-only install, a
missing-wheel download failure, a checksum-valid but invalid wheel after native
installation, managed-environment recovery with the real wheel, a repeated
install, MSI repair, C++ worker-loss and client-death recovery (with a Platform
device remove and rescan), and GenAI worker-loss and service-loss recovery.
Release admission (`--require-release-qualification`) requires their evidence.
It records the Windows version, elevation, Smart App
Control, effective AppLocker visibility and driver-signing policy (Secure Boot,
boot test-signing and integrity-check options, WDAC and HVCI enforcement)
without changing policy, and removes Python, Core and Scheduler in reverse order.

This is the normal validation path. It does not require maintainers to copy workflow
run IDs between repositories. Scheduler follows Core's branch/snap policy;
sima-cli resolves from its supported `main` channel.

Once a previous Windows release exists, release qualification installs that
published release through `sima-cli`, upgrades to the candidate through
`sima-cli`, and verifies that the candidate's Scheduler MSI, Core MSI and
`pyneatpcie` wheel work together. Synthetic version metadata is not accepted as
upgrade evidence. After a successful upgrade, the qualifier does not remove the
candidate and reinstall the previous release; Windows Installer's
failed-transaction rollback is the separate CI fixture gate below.

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

Windows hardware qualification runs the Linux PCIe performance tests (C++ and
Python): 100 warmup iterations followed by 3,000 measured iterations. Until the
Windows runner moves to
a host comparable to the Linux ones, its gate is a floor for the current runner
(PCIe Gen3 x4, single-channel memory: about 320 FPS and 14.2 ms): mean latency no
higher than 16 ms and throughput no lower than 290 FPS. Linux keeps 12 ms and
440 FPS; a Gen5 Windows host measures about 680 FPS and 9 ms. Restore the Linux
gate when the runner is upgraded.

The same runner runs the parallel-queue tests on queues 0 and 1 instead of Linux's
0-5 (`SIMAPCIE_STRESS_QUEUES`). The Platform driver allocates each data queue as
physically contiguous memory, about 300 MB per queue at this test's tensor size,
and an 8 GB host has about 1 GB of free physical memory, so a fourth queue fails
with Error 1450. Restore all six queues when the runner has 16 GB or more.

## Validation and bootstrap overrides

Run a non-publishing validation of the entire Core cohort with:

```text
gh workflow run vulcan-ci.yml --repo sima-neat/core \
  --ref <core-ref> -f validation_only=true
```

The main workflow still runs Windows hardware qualification when the lane is
enabled, and skips Linux and Windows publication and branch promotion. A direct
Windows dispatch is useful
when debugging against a completed Linux run:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref <core-ref> \
  -f linux_run_id=<same-commit-core-vulcan-run-id> \
  -f validation_only=true
```

The admitted Linux run must contain `core-pcie-card-offline-bundle`: a release
tag (`v*`) run, or a branch run with the Windows card lane enabled. To rehearse release
qualification without publishing, add `-f release_qualification=true`. The
hardware job then runs the Windows-only release checks (`--release-qualification`)
and admission requires their evidence, exactly as a release does, while
`validation_only=true` keeps the publish job skipped.

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
from Vulcan CI after its Linux builds succeed; it is not a manual-dispatch
override.

## Publishing and formal release checks

Normal branch and release tag runs enter the Windows lane from Core's main CI
with `validation_only=true` when the lane is enabled. Other tags skip the lane
and the card bundle, whose metadata accepts only release tag versions. Like the Linux card job, every
run prepares the GenAI models and runs the GenAI tests. They
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
dispatch also needs `linux_run_id=<same-commit-core-vulcan-run-id>`.
The dispatch downloads that run's `neat-core-dist` and
`core-pcie-card-offline-bundle` artifacts. Release tag runs build the card
bundle even when the Windows card lane is off, but Vulcan CI keeps
`neat-core-dist` for only 3 days (`artifact_retention_days` of its `build` job),
so dispatch the Windows release within 3 days of the tag's Vulcan CI run. The
tag run's `publish-offline-develop` waits for the card bundle job: if that job
fails, the release's offline bundles are not published until it is re-run.
Publication is rejected unless
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

Before the first production publication, rehearse the customer install from
the `develop` channel. Staging holds no Core or Scheduler publications to
install from. A `develop` push publishes its Windows cohort with the same
`publish_windows.sh` and layout as a release, at
`core/develop/<commit12>/pciehost/windows/amd64/`, and promotes `develop`
`latest` once its lanes pass. On a clean Windows 11 host, run the customer
command and confirm that both MSIs and the wheel download and install:

```text
sima-cli neat install core/pciehost/windows/amd64@develop
```

Before the first Windows release publication, also confirm that the publisher
role can tell a missing object from a forbidden one. A release publication
reads `metadata.json` under the release prefix first, so that it never
overwrites a published release, and only HTTP 404 means "not published yet".
S3 answers a missing key with 404 only when the role has `s3:ListBucket` on the
bucket; otherwise it answers 403. Branch publications skip this read, so the
`develop` rehearsal does not cover it. With the publisher role's credentials,
read a key that does not exist under the release prefix:

```text
aws s3 cp s3://<artifact-bucket>/core/<release-tag>/000000000000/pciehost/windows/amd64/metadata.json -
```

It must fail with `An error occurred (404)`. If it reports `(403)`, the first
release publication fails with
`cannot read s3://<artifact-bucket>/core/<release-tag>/<commit12>/pciehost/windows/amd64/metadata.json: ... (403) ...`;
grant the role `s3:ListBucket` on the bucket before the release dispatch.

Cross-version upgrade is a formal release check rather than a cost paid by every
branch commit. To run it against the previous published Windows release,
dispatch the Windows workflow with:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/core \
  --ref v<package-version> \
  -f linux_run_id=<same-commit-core-vulcan-run-id> \
  -f require_upgrade=true \
  -f upgrade_base_tag=v<previous-windows-release> \
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
requires Authenticode. Neither main CI passes it: Vulcan CI calls the Core
Windows lane with `require_signatures=false`, and Scheduler's `ci.yml` leaves it
at its default. A signed cohort therefore needs a direct Scheduler dispatch that
publishes a signed profile, followed by the Core release dispatch above with
`-f require_signatures=true`:

```text
gh workflow run pcie-host-windows.yml --repo sima-neat/scheduler \
  --ref <scheduler-ref> -f validation_only=false -f require_signatures=true
```

The optional gate requires the certificate, RFC 3161
timestamp service, and complete Scheduler/Core signature receipts. Platform DLLs
are the Platform installer's and are not part of this gate. Signing is
additional provenance; it does not replace tests.

Wheel installation failure deliberately preserves successful native MSI work so
the customer can retry the Python step. This is not a transaction that rolls all
three packages back. MSI repair, repeat installation, and failure recovery
are exercised on every publishing run and `release_qualification` rehearsal, and clean removal on every run; the older-version upgrade is the
separate formal gate described above.

## Evidence and recovery

Retain these artifacts with the release record:

- `scheduler-native-windows-amd64`;
- `core-pciehost-windows-dependencies`;
- `core-pciehost-windows-amd64`;
- `core-pciehost-windows-hardware-<run>-<attempt>`;
- `core-pciehost-linux-admission-<run>-<attempt>`.

Actions deletes them after 7 to 14 days (`retention-days`), so download the Core
artifacts, including the hardware evidence, as soon as the release dispatch
finishes:

```text
gh run download <release-dispatch-run-id> --repo sima-neat/core --dir <release-record>
```

The hardware workflow holds both a runner-local lock and an atomic card-side
lock. At the end it stops card workers, removes its `/var/tmp/neat-offline-*`
staging directory, uninstalls the wheel and both MSI cohorts, and removes its
disposable downloads. It compares the final card state with the baseline,
including the exact Platform 3.0 `/etc/buildinfo` hash, then releases both
locks. The card keeps the installed candidate, as on Linux. If the install
phase began and the card installer still holds `/run/neat-offline-install.lock`,
the outcome is unknown, so the stage and both locks are retained. As on Linux,
a failed install is not rolled back: rerun the bundle or reflash the card.
Emergency cleanup only touches a retained card lock when its owner token
matches the retained local record. It finds the stage through the card lock's
`offline_stage` record. It also removes the temporary standard user, profile
and Python that a cancelled or restarted run left. Without an owned lock, it
does not stop workers, remove users, uninstall MSIs or remove a work directory.
If cleanup fails, the lock at
`%USERPROFILE%\.sima\locks\neat-pcie-card0.lock` and its matching
`/tmp/neat-pcie-card0.lock` card directory are deliberately retained. Do not
delete either until the evidence logs have been reviewed and the runner,
installation root, workers, staging directory, and queues have been restored manually.
