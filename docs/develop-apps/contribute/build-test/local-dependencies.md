# Build Core with local Kernels, Runner and Scheduler

Run the usual command from your Core checkout in the Neat development environment:

```bash
./build.sh --all --no-doc --install
```

The build prepares **Kernels → Runner → Scheduler → Core**, then performs the existing final
installation/deployment. Child builds never install into the SDK, deploy to a
DevKit, execute target tests, compile EV firmware, or activate firmware.

Place the checkouts beside each other for automatic source discovery:

```text
development/
├── core/
├── kernels/
├── runner/
└── scheduler/
```

Runner contains framework-neutral execution, preparation, memory and native
transport. Scheduler contains the GStreamer integration/plugins and compatibility
adapters. Each has its own source root and installed package. Checkouts need not
contain a `.git` directory; worktrees and source snapshots are supported.

## Select sources or existing local packages

For a checkout elsewhere, set `NEAT_KERNELS_SOURCE`, `NEAT_RUNNER_SOURCE` or `NEAT_SCHEDULER_SOURCE`:

```bash
NEAT_RUNNER_SOURCE=/path/to/simaai-runtime ./build.sh --all --no-doc
```

To reuse a locally built package instead, set `NEAT_KERNELS_PACKAGE_DIR` or
`NEAT_RUNNER_PACKAGE_DIR` or `NEAT_SCHEDULER_PACKAGE_DIR`. Each package directory must contain its installed
`usr/` tree, build receipt, and matching Debian package produced by the component's canonical package workflow.
Kernels uses standard CMake configure/build/install and its package producer;
Runner and Scheduler use their respective root `build.sh` entry points:

```bash
NEAT_RUNNER_PACKAGE_DIR=/path/to/runner/dist/package \
  ./build.sh --all --no-doc
```

Selection is explicit override, then valid sibling source, then sibling
`dist/package`. Explicit package selection overrides sibling source. Setting
both source and package overrides for the same component is an error. A failed
local build stops the build; it never falls back to an older package.

Without local overrides or a complete sibling stack, `build.sh` acquires one published
Scheduler dependency set, including matching Runner, Kernels and Codecs, using the manifest policy.
Explicit local selections remain authoritative; failed local builds never trigger downloads.
See [Published CI dependencies](#published-ci-dependencies) for acquisition details.
Use `--no-llima` when the optional LLiMa backend is not needed.

## Platform dependencies

Use the same Platform 3.0 SDK/toolchain for all four repositories. Set
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
`simaai-scheduler-client-test-dev` package from that exact producer. Source builds
request it with `--client-test-support`; package builds select the adjacent
`client-test-support/` directory or `NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR`.
The coordinator checks its production receipt, source, payload and archive before
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

### Backend providers

For local source builds, Scheduler requires separately installed platform backends.
These are not copied into Scheduler or fetched by the local source build:

- Set `NEAT_CODECS_PACKAGE_DIR` to the separately built Codecs package directory.
  When Scheduler enables codecs, this directory must contain the exact installed
  `usr/` prefix it consumed and the matching `neat-codecs` and `neat-codecs-dev`
  Debian packages. Scheduler's dependency coordinator verifies their payloads,
  ownership and dependency controls; Core retains the validated closure and
  includes both archives in the final distribution. It does not build Codecs
  or substitute another version.
- Use the SDK's matching MLA Runtime, rendering and tracing/logging libraries.
- When SDK paths need explicit selection, pass CMake definitions through
  `NEAT_SCHEDULER_CMAKE_ARGS`, for example
  `-DSIMA_RUNTIME_MLART_LIBRARY=/path/to/libmlart.so` and its matching include
  directory. Use the option names documented in Scheduler's build guide.

Arguments in `NEAT_RUNNER_CMAKE_ARGS` and `NEAT_SCHEDULER_CMAKE_ARGS` are parsed as shell-style words, not
executed as shell code. They cannot override the selected dependency packages
or install layout.

## Incremental builds and installation

Core-owned dependency builds and packages live under `deps/backend/`, separate
from each repository's standalone build directory. CMake builds incrementally;
`--clean` cleans the selected local dependency builds and Core's build.

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

For header-only dependency preparation, `--install-deps-only` uses the selected
packages, acquiring them when no local inputs are selected. It copies headers
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

The selected Scheduler package installs its exact SDK policy. In the disposable
CI SDK container, `scripts/ci/prepare_scheduler_sdk.sh` uses the official sysroot
setup command for that revision and retains downloaded platform archives. The
coordinator checks the real GCC14 version/target and target SDK identities, then
selects the exact Runner-consumed runtime provider archives through
`NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`. Native ARM64 and cross-compilers need not
have identical executable bytes. Their target ABI and selected platform must
match. No device installation or firmware qualification happens in this step.

Local source/package overrides remain authoritative. CI acquisition is explicit;
a failed local source build never downloads a substitute. The existing
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

Before running Core's source-first `build.sh`, provide:

- `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`: a JSON array of absolute paths to the exact
  runtime provider DEBs selected by Runner. Include its required SimaHeap runtime
  archive and the MLA and TVM runtime archives when those backends are enabled.
  Do not include development archives in this array. Runner verifies actual loader
  names and library bytes. Scheduler retains
  exact archive identities and payload inventories as build evidence. Providers
  declared Platform-owned by Runner stay outside Core's installable DEB set;
  undeclared providers retain exact archive composition.

These selections remain explicit even with prebuilt Runner/Scheduler package
overrides. The immutable four-component materializer does not fetch platform
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
