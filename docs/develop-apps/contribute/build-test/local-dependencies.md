# Build Core with its Scheduler and LLiMa dependencies

Core builds against two dependencies: Scheduler and LLiMa. Run `build.sh` from
your Core checkout in the Neat SDK:

```bash
./build.sh --all --no-doc
```

By default, `build.sh`:

1. Downloads the published Scheduler set and the LLiMa packages from Vulcan.
2. Installs both into the SDK sysroot.
3. Builds Core.

## Use local Scheduler or LLiMa packages

To build Core against your own Scheduler or LLiMa build, pass its DEB folder:

```bash
./build.sh --scheduler ../scheduler-debs --llima ../llima/dist/debs
```

Each option works alone; `build.sh` still downloads the other dependency. The
given packages replace the published ones in the sysroot.

The Scheduler folder holds the complete set that Scheduler was built with:

- `simaai-scheduler`
- the `simaai-runner` and `simaai-kernels-dev` packages it was built against
- the matching `simaai-kernels-firmware` package for EV-enabled builds
- the `neat-codecs` and `neat-codecs-dev` pair when Scheduler uses Codecs
- optionally `simaai-runtime` and, for test builds, `simaai-scheduler-client-test-dev`

Core does not select Kernels, Runner or Codecs. To test a change in one of them,
build Scheduler against it using Scheduler's build guide, put the resulting
packages in one folder, and pass that folder with `--scheduler`. Scheduler's
coordinator checks that the packages belong together and stops the build if they
do not.

`build.sh` stops with an error when a `NEAT_KERNELS_*`, `NEAT_RUNNER_*`,
`NEAT_SCHEDULER_*` or `NEAT_CODECS_*` source or package variable is set. Use
`--no-llima` to build without LLiMa.

## Platform dependencies

Build Kernels, Runner, Scheduler and Core with the same Platform 3.0 SDK/toolchain. Set
`CMAKE_TOOLCHAIN_FILE` when selecting a nondefault toolchain. Kernels and Runner
receipts must agree on target/compiler, kernel contracts and consumed build
configuration. The canonical package workflow also requires the exact consumed
Kernels payload identity. Rebuild the Runner package against the selected SDK
after a Kernels package change, even when Runner source does not change. Core
then relinks against the matching Scheduler client implementation; Scheduler
consumes the lower numerical packages.
Core's source-first DEBs pin the selected Scheduler and numerical SDK versions.
Runner pins Kernels; Scheduler pins Runner, Kernels and its enabled codec backend.
Test-enabled Core builds also require Scheduler's separate
`simaai-scheduler-client-test-dev` package from that exact producer, in the
same Scheduler folder. The coordinator checks its production receipt, source, payload and archive before
exposing the test-only CMake component. Runtime-only builds do not require it.
The build composes metadata-only `simaai-runtime` compatibility with exact Runner
and Scheduler dependencies. Canonical packages own their files once; Runner alone
owns the existing loader configuration. Always
install the complete matching distribution rather than replacing one package.

### Migrating GStreamer header ownership

The unreleased Core `0.6.0` moves ownership to Scheduler for
`/usr/include/gst/SimaaiGstErrors.h`. The include path remains unchanged. Core
no longer carries its own implementation or installs that file; consumers use
the installed Scheduler headers and `SimaScheduler::Headers` target.

Upgrade Core and Scheduler together using the complete matching distribution.
Scheduler declares the header transfer for the unreleased 0.6.0 package lineage
without marking corrected Core 0.6.0 packages incompatible. Before APT changes
files, the installer uses Scheduler's authenticated ownership check: the former
owner must be replaced in the same package set with that file removed, or be
covered by an explicitly declared and approved retirement. Duplicate incoming
owners and unrelated ownership conflicts are rejected.

On Platform 3.0, the complete split cohort also retires the monolithic
`neat-runtime`, `neat-gst-plugins`, and `neat-internals-dev` 0.4, 0.5, and
0.6.0 package lines, including development builds. Retirement requires matching
bounded `Breaks` and `Replaces` declarations from the incoming owners; versions
at or above `0.6.1~` are not covered. This does not authorize unrelated package
removal or claim compatibility with the retired APIs. Scheduler declares the
runtime/plugin file transfers; the complete runtime still requires the matching
Runner execution dependency. Installing Runner alone does not retire the
monolithic 0.6.0 packages.

EV-enabled Modalix distributions require the matching `simaai-kernels-firmware`
package selected by Scheduler. Missing or mismatched firmware fails admission.
Replacing an older `neat-ev74-firmware` package requires that exact selected
firmware package to declare version-matched `Replaces` and `Breaks` or `Conflicts`;
the normal transaction and removal checks still apply.
Firmware activation uses only the unique incoming package that delivers the
activation helper. The installed package identity, helper ownership, executable
mode, and bytes must match that selected package. A selected but mismatched helper
fails installation. For distributions that do not require EV74 firmware, a missing
helper leaves existing firmware unchanged. The installer never uses a helper
left by an earlier installation.

If installation on the DevKit was interrupted, rerun the complete verified distribution. The
installer can recover incomplete packages only when they belong to that incoming
package set and match its architectures. Recovery must install the exact selected
versions without changing unrelated packages. It can remove only configured,
unheld predecessor packages whose retirement is explicitly declared by the
incoming cohort and approved by the existing removal checks. Recovery first
completes the admitted transition, checks package state, then reinstalls every
selected package with removals disabled. Installation succeeds only after this
final reinstall. An unrelated broken package database still requires separate
repair; the installer does not
perform a general APT repair or guarantee rollback.

Do not install the Scheduler DEB alone over an existing Core development package,
use `--force-overwrite`, or delete dpkg-owned files. A standalone Scheduler test
image does not need Core installed; an existing Core installation needs the
coordinated upgrade.

## Incremental builds and installation

Each build refreshes the selected Scheduler set in `deps/scheduler/` and stages
its packages in `deps/backend/`. CMake builds Core incrementally; `--clean` also
removes Core's build.

Before linking, Scheduler's dependency coordinator verifies installed kernel
headers against their receipt and checks Runner's consumed contract and
Scheduler's exact Runner input. Before staging deployment
packages, it verifies each Debian package's `/usr` payload matches the selected
installed tree. Old Internals packages are not selected or downloaded.

Core consumes the installed `SimaScheduler` client component, not sibling source
includes or lower development interfaces. Lower runtime identities remain in the
validated dependency closure.
The generated dependency selection is recorded in `build/resolved_manifest.json`
and `build/buildinfo.json`. Only a complete build reaches final installation;
this is not a transactional rollback guarantee for SDK or device installation.

Package build info is provisional before CMake configuration. After compilation,
`build.sh` refreshes it from that build's CMake cache, configured compiler, target
Python headers and built extension, before install checks or packaging. Target
Python ABI flags must agree with the extension suffix; the host Python version
is not used. Python-disabled builds record that profile without a Python ABI.

`compatibility.host_sdk` describes the build host's observed SDK release;
`compatibility.target_sdk` describes the selected sysroot and compiler. A missing
target SDK release remains unavailable, even when the host has an SDK release.
The manifest's platform version is a separate compatibility requirement, not a
replacement for an unobserved SDK version. `ELXR_SDK_RELEASE_FILE` selects the
host release file. `NEAT_TARGET_SDK_RELEASE_FILE` can explicitly supply the target
release file, but must agree with a release file in the selected sysroot.
`NEAT_BUILDINFO_TOOLCHAIN` preserves an explicit display label alongside, not in
place of, the configured compiler identity.

The metadata check rejects conflicting compile sysroots, headers and extension
ABI tags. Separate compile/link sysroots and additional CMake compiler
argument/target/toolchain contexts are not supported by this metadata path;
it fails rather than reporting an unverified target identity. Marker-qualified
NumPy requirements are retained verbatim from `pyproject.toml`, not reduced to
one unconditional minimum.

The source record uses the actual checkout commit and dirty status. It records
tracked changes and nonignored untracked file/symlink identities separately;
ignored build products are not a source snapshot. The metadata output itself is
excluded from the untracked-content digest to avoid self-reference. A dirty local
candidate is not identified as a clean published artifact.

For header-only dependency preparation, `--install-deps-only` uses the
`--scheduler` folder or downloads the published set. It copies headers
without building or executing A65/EV code.
For native clang-tidy analysis, use `scripts/run_cpp_tidy.sh` with the selected
Scheduler and matching test-support package prefixes. `TIDY_INSTALL_DEPS=1` installs
native host analysis prerequisites. The script uses their public CMake exports
directly, without activating or modifying an SDK sysroot. LLiMa-backed
GenAI implementation files remain outside this host-only analysis profile.
The script does not substitute an ambient SDK LLiMa package for the selected
build's dependency. Documentation-only builds do not prepare native dependencies.

EV device builds remain a separate licensed workflow. None of these commands
implicitly invokes MetaWare, replaces firmware, or qualifies a firmware change.

## Published CI dependencies {#published-ci-dependencies}

The normal Vulcan build resolves the Scheduler policy in `deps/manifest.json`
through `sima-cli`, then downloads the selected immutable publication. As with
LLiMa, branch `snap` first tries the current Core branch. If resolution fails and
Scheduler's published branch index confirms that branch is absent, it reports
the fallback and retries `develop:latest`. Other lookup failures remain errors,
including an unavailable or invalid branch index. Tag snaps and explicit
branch/commit selectors never fall back. Device and PCIe-host acquisition share
this policy; PCIe `--scheduler-spec <commit>` also remains strict. Downloads,
verification, bootstrap and staging failures after resolution remain errors.
Receipts retain the branch and immutable commit actually selected.
`tools/ci/acquire_scheduler.py` verifies publisher hashes before
running Scheduler's bootstrap. The installed Scheduler coordinator partitions and
validates the bundled Runner, Kernels and Codecs archives. Core does not resolve
those repositories independently.

The selected Scheduler package installs its exact SDK policy. In the SDK,
`build.sh` runs `scripts/ci/prepare_scheduler_sdk.sh`, which uses the official
sysroot setup command for that revision and retains downloaded platform archives. The
coordinator checks the real GCC14 version/target and target SDK identities, then
selects the exact Runner-consumed runtime provider archives through
`NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`. Native ARM64 and cross-compilers need not
have identical executable bytes. Their target ABI and selected platform must
match. No device installation or firmware qualification happens in this step.

A `--scheduler` folder is never mixed with or replaced by a download. The existing
`tools/ci/materialize_dependencies.py --manifest ... --output-root ...` command
remains an optional transport for a separately reviewed immutable manifest. It is
not the default workflow and there is no implicit `ci/dependencies.json`.

Code hygiene uses `tools/ci/acquire_scheduler.py` to acquire a published Scheduler
independently under the same dependency policy, without waiting for the Core build.
Scheduler's bootstrap validates the package cohort. Host analysis consumes its
frontend and matching test-support CMake exports without SDK activation or target
execution.

See [Hardware test ownership](./hardware-test-safety.md) for the target-side
supervisor required by the guarded performance lane.

## Verify the finished distribution

`build.sh --all` checks the completed distribution after generating package
metadata. `scripts/ci/verify_core_dist.sh` runs the same read-only check in CI,
using the selected `sima-cli` Python environment. Set `SIMA_CLI_PYTHON` explicitly
if the interpreter is not beside the selected CLI executable.

The check validates dependency package bytes, compatibility fields, the installer
allowlist, all resource checksums, public-wheel isolation and installed CTest
trees. It consumes existing producer receipts; it does not resolve or repair
dependencies. Keep these documents distinct:

- `dist/manifest.json`: platform, Model Zoo and ABI compatibility.
- `dist/resolved-deps-manifest.json`: the selected backend receipt, copied exactly.
- `build/resolved_manifest.json`: the fuller Core build resolution.

Do not replace files or recompute selected checksums after this check. Rebuild
metadata and verify the complete distribution again if an artifact changes.


## Explicit platform provider packages

Core's production box/mask publication uses int32 coordinates. Select a Scheduler
built with canonical Kernels rendering and the matching Runner execution APIs.
`SIMANEAT_REQUIRE_INT32_RENDERER` defaults to the runtime-artifact requirement and
fails configuration if the installed Scheduler lacks this capability. Set it to
`OFF` only for a Core profile that does not use Core box/mask rendering.

Rendering uses the selected Kernels/Runner packages, not a separate platform
renderer archive. Rebuild the cohort together for request API revision 2.

In the SDK, `build.sh` sets the following through
`scripts/ci/prepare_scheduler_sdk.sh`. Outside the SDK, provide:

- `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`: a JSON array of absolute paths to the exact
  runtime provider DEBs selected by Runner. Include its required SimaHeap runtime
  archive and the MLA and TVM runtime archives when those backends are enabled.
  Do not include development archives in this array. Runner verifies actual loader
  names and library bytes. Scheduler retains
  exact archive identities and payload inventories as build evidence. Providers
  declared Platform-owned by Runner stay outside Core's installable DEB set;
  undeclared providers retain exact archive composition.

This selection stays explicit with a `--scheduler` folder too. The immutable four-component materializer does not fetch platform
provider archives. Acquire them from their supported producer before invoking
Core; missing or unconsumed providers fail rather than selecting a sysroot
fallback. Use matching development packages for compilation. On a DevKit,
Scheduler's authenticated admission tool checks the installed Platform providers
against the incoming consumers before package mutation. The installer rejects
APT plans that replace or remove those providers and verifies their identity and
installed consumer closure afterward. `FORCE_INSTALL` does not waive these checks.
Do not rebuild a DEB merely to label a private DSO as a released provider.

Package composition checks disjoint payload ownership, exact dependency
versions and unchanged archives. These build-time checks do not qualify numerical
execution, dynamic masks or hardware retirement. Those require the corresponding
integration tests on the selected target cohort.
