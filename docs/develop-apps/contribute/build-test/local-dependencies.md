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

**TODO: downloading native dependency artifacts.** Until implemented, provide local
sources or packages for all three dependencies. Missing inputs fail before dependency
installation. Existing LLiMa dependency handling is unchanged; use `--no-llima`
when that optional backend is not needed.

## Platform dependencies

Use the same Platform 3.0 SDK/toolchain for all four repositories. Set
`CMAKE_TOOLCHAIN_FILE` when selecting a nondefault toolchain. Kernels and Runner
receipts must agree on target/compiler, kernel contracts and consumed build
configuration. The canonical package workflow also requires the exact consumed
Kernels payload identity. Rebuild the Runner package against the selected SDK
after a Kernels package change, even when Runner source does not change. Core
then relinks against the matching Kernels archives.
Core's source-first DEBs pin the selected Scheduler and numerical SDK versions.
Runner pins Kernels; Scheduler pins Runner, Kernels and its enabled codec backend.
The build composes metadata-only `simaai-runtime` compatibility with exact Runner
and Scheduler dependencies. Canonical packages own their files once; Runner alone
owns the existing loader configuration. Always
install the complete matching distribution rather than replacing one package.

### Migrating GStreamer header ownership

Starting with Core package version `0.6.1`, Scheduler is the sole owner of
`/usr/include/gst/SimaaiGstErrors.h`. The include path remains unchanged. Core
no longer carries its own implementation or installs that file; consumers use
the installed Scheduler headers and `SimaScheduler::Headers` target.

Upgrade Core and Scheduler together using the matching distribution. Scheduler's
bounded `Breaks`/`Replaces` relationship prevents an older Core development
package from silently losing the header. Do not use `--force-overwrite` or delete
dpkg-owned files. A standalone Scheduler test image does not need Core installed;
an existing Core installation needs the coordinated upgrade.

### Backend providers

Scheduler requires separately installed platform backends. These are not
copied into Scheduler or fetched by this workflow:

- Set `NEAT_CODECS_PACKAGE_DIR` to the separately built Codecs package directory.
  When Scheduler enables codecs, this directory must contain the exact installed
  `usr/` prefix it consumed and the matching `neat-codecs` and `neat-codecs-dev`
  Debian packages. Core verifies their payloads, ownership and dependency controls,
  then includes both archives in the final distribution. It does not build Codecs
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

Before linking, the build verifies installed kernel headers against their
receipt and checks Runner's consumed contract and Scheduler's exact Runner input. Before staging deployment
packages, it verifies each Debian package's `/usr` payload matches the selected
installed tree. Old Internals packages are not selected or downloaded.

Core consumes explicit installed CMake packages, not sibling source includes.
The generated dependency selection is recorded in `build/resolved_manifest.json`
and `build/buildinfo.json`. Only a complete build reaches final installation;
this is not a transactional rollback guarantee for SDK or device installation.

For header-only analysis, `--install-deps-only` requires local package overrides.
It copies headers without building or executing A65/EV code. Documentation-only
builds do not prepare native dependencies.

EV device builds remain a separate licensed workflow. None of these commands
implicitly invokes MetaWare, replaces firmware, or qualifies a firmware change.

## Immutable CI materialization

`tools/ci/materialize_dependencies.py` transports a reviewed repository-relative
`NEAT_CI_DEPENDENCY_MANIFEST` into a fresh workspace and emits explicit producer
selection variables. The Vulcan build defaults the path to `ci/dependencies.json`
and fails closed if that reviewed manifest has not been provided. No public
artifact location or unpromoted private commit is invented by the build.

Manifest schema1 requires exactly `kernels`, `runner`, `scheduler` and `codecs` under `inputs`:

- Kernels/Runner/Scheduler: `kind: "git"`, credential-free HTTPS `repository`, immutable
  full `commit`; or `kind: "package-bundle"`, HTTPS `url`, SHA256 `sha256`.
- Codecs: an independent `package-bundle` with the exact installed prefix and
  matching runtime/development DEBs. It is not another source-build dependency.

The materializer performs no device installation and no fallback/latest
selection. Core's existing dependency adapter still verifies installed receipts,
payloads, architecture, exact dependencies and package composition. Immutable
transport is not qualification: publish/promote only after the selected cohort
passes the required hardware gates.

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
  optional backend runtime DEBs selected by Runner. For example, supply the
  checked-completion MLA runtime archive and the matching TVM runtime archive
  when both backends are enabled. Do not include development archives in this
  array. Runner verifies actual loader names/library bytes; Core requires the
  same archive controls and hashes, then composes those unchanged archives.

These selections remain explicit even with prebuilt Runner/Scheduler package
overrides. The immutable four-component materializer does not fetch platform
provider archives. Acquire them from their supported producer before invoking
Core; missing or unconsumed providers fail rather than selecting a sysroot
fallback. Use each producer's matching development package for compilation and
its actual runtime package for deployment. Do not rebuild a DEB merely to label
an existing private DSO as a released provider.

Package composition checks disjoint payload ownership, exact dependency
versions and unchanged archives. These build-time checks do not qualify numerical
execution, dynamic masks or hardware retirement. Those require the corresponding
integration tests on the selected target cohort.
