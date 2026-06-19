# CLAUDE.md

Guidance for Claude (and other agents) working in this repository.

This file complements `AGENTS.md`, `CONTRIBUTING.md`, and the authoritative
contributor docs under `docs/develop-apps/contribute/` — read those for full rationale.
This file extracts the operational essentials.

## What this repo is

SiMa Neat — a C++20 library for building, validating, running, and debugging
GStreamer pipelines with a typed, composable API. Ships C++ libraries plus
Python bindings (`pyneat`) built with nanobind + scikit-build-core.

- C++ namespace: `simaai::neat`
- CMake project: `SimaNeat`
- Core runtime types: `Model`, `Session`, `Run`
- Public API surface: headers under `include/` (treated as **stable**)
- Canonical production pipeline test (source of truth):
  `tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`

## Build commands

The single entrypoint is `build.sh`. Output lands under `build/`.

```bash
./build.sh                       # dev-only: core lib + headers (default)
./build.sh --all                 # lib + tests + tutorials + docs + pyneat wheel
./build.sh --all --clean         # clean full build
./build.sh --python              # add pyneat bindings to selected targets
./build.sh --doc                 # docs only
./build.sh --fuzz                # fuzz-enabled package artifacts
./build.sh --asan-ubsan          # ASan+UBSan instrumented build
./build.sh --tsan                # TSan instrumented build
./build.sh --install-deps-only   # bootstrap system deps, exit
./build.sh -h                    # full help
```

Under the hood: CMake 3.16+, `CMAKE_CXX_STANDARD 20`, `Release` by default.
Key options surfaced as `-DSIMANEAT_BUILD_*` toggles in `CMakeLists.txt`.

### Python bindings (`pyneat`)

```bash
python -m pip install -e .[dev]   # editable install with dev extras
pytest -q                          # runs python/tests and tests/tutorials
```

`pyproject.toml` declares the build (scikit-build-core + nanobind ≥ 2.4).
Extension module is `pyneat._pyneat_core`. NumPy/PyTorch interop is via DLPack.

## Test commands

Match the **tier** to the change (from `CONTRIBUTING.md`):

| Change type | Required check(s) |
| --- | --- |
| Runtime/library (`src/`, `include/`) | `ctest --test-dir build/tests --output-on-failure` + `bash scripts/ci/run_crash_correctness_gate.sh` |
| Examples / tutorials wiring | `ctest --test-dir build/tutorials --output-on-failure` |
| Docs or naming | `bash scripts/ci/check_naming_and_conflicts.sh` + `DOCS_STRICT_LINKS=1 npm --prefix website run build` |
| Repo hygiene / release policy | `bash scripts/ci/check_artifacts.sh` + `bash scripts/ci/check_release_policy.sh` |
| MPK or security | `bash scripts/ci/run_mpk_security_gate.sh` |
| Packaging / install | `bash scripts/ci/run_install_smoke.sh` |
| Perf or long-run | `bash scripts/ci/run_perf_regression_gate.sh` |
| Python bindings | `pytest -q python/tests` |

Test directory layout under `tests/`: `unit_testing/`, `e2e_pipelines/`,
`e2e_testing/`, `fuzz/`, `perf/`, `security/`, `stress/`, `install_smoke/`,
`metrics/`, `tools/`, `tutorials/`. `tests/CMakeLists.txt` defines a
**strict** lane (default; cannot use `skip_test(...)`) and a **long** lane
(may `return 77` to skip; runs in weekly CI).

Skip policy is strict — only tests explicitly labeled `long` may skip. Don't
introduce new skips in strict paths.

## Style and formatting

Enforced by pre-commit / pre-push hooks. Install them once per clone:

```bash
bash scripts/dev/install_hooks.sh
```

Before pushing, the equivalent manual run is:

```bash
bash scripts/check_format.sh --changed-only
bash scripts/check_cmake_format.sh --changed-only
bash scripts/check_duplicate_includes.sh --changed-only
```

C++ rules (`.clang-format`, `.editorconfig`):

- C++20, LLVM base, 2-space indent, 100-col limit, LF endings, final newline.
- `PointerAlignment: Left`, `BreakBeforeBraces: Attach`.
- `SortIncludes: false` and `IncludeBlocks: Preserve` — do **not** reorder
  include groups; the existing order is load-bearing.
- No duplicate includes in C/C++ sources.

CMake rules: enforced by `scripts/check_cmake_style.py` (invoked by
`check_cmake_format.sh`). 2-space indent.

Run optional analysis with `bash scripts/run_cpp_tidy.sh`.

## SiMa-specific coding guidelines

These come from `docs/develop-apps/contribute/start-here/coding_standard.md`,
`docs/develop-apps/contribute/start-here/architecture.md`, and
`docs/develop-apps/contribute/start-here/naming.md`. Treat
them as load-bearing.

### Naming contract (`docs/develop-apps/contribute/start-here/naming.md`)

- Use canonical names in new code and docs: `Model`, `Session`, `Run`,
  `Input`, `Output`.
- Legacy aliases (`PipelineSession`, `PipelineRun`, `NeatModel`,
  `InputAppSrc`, `OutputAppSink`) are compatibility-only — don't introduce
  them in new public docs/examples.
- CI enforces this via `scripts/ci/check_naming_and_conflicts.sh`.

### Module dependency rules (non-negotiable)

- `builder/` — STL-only; must **not** depend on GStreamer or `pipeline/`.
- `gst/` — must **not** depend on `pipeline/`.
- `nodes/` — must **not** depend on `pipeline/` (Nodes are build-time
  descriptions, not runtime orchestrators).
- `pipeline/` — orchestrator; may depend on `gst/`, `builder/`, `nodes/`,
  `contracts/`, `policy/`, `mpk/`.

### Public API stability

- Headers under `include/*` are installed and treated as stable API.
- Prefer **non-breaking** additions (new overloads, optional fields, new APIs).
- A breaking signature change requires:
  1. A `Breaking API Change` section in the PR using
     `.github/pull_request_template.md`.
  2. Impact analysis, migration steps, and versioning intent.
  3. Updated docs and examples in the same change set.
  4. Explicit maintainer approval.
- Prefer a deprecation period (old symbol kept + new path added) over
  immediate removal.
- Internal headers (`src/**/internal`) are not installed; examples and
  tutorials must use only public API.

### Determinism is first-class

- Node output must be deterministic for equal inputs/config.
- Element names, generated pipeline strings, serialized pipeline JSON, and
  report fields must be reproducible.
- Each Node owns:
  - `backend_fragment(index)` — deterministic GStreamer fragment.
  - `element_names(index)` — exact names produced by the fragment.

### Diagnostics and error handling

- Prefer structured failures with actionable context — don't swallow
  plugin/caps/runtime errors.
- Map terminal failures to stable `SessionReport.error_code` families:
  `misconfig.pipeline_shape`, `misconfig.caps`, `misconfig.input_shape`,
  `build.parse_launch`, `runtime.pull`, `io.parse`, `io.open`.
- `SessionReport.repro_note` is the human summary; include offending value
  and node/element context.
- Diagnostics updated from streaming threads must be **lock-free** (use the
  `BoundaryFlowCounters` / `ElementTimingCounters` / `ElementFlowCounters`
  atomic patterns under `src/pipeline/internal/`).

### Crash and correctness policy (hard rule)

- Crash / segfault / use-after-free / data race = **immediate fix on
  current branch**.
- No push if local crash/correctness gate fails.
- No merge if CI crash/correctness gate fails.
- Do not defer crash fixes behind waivers, skip lists, or release exceptions.

### Concurrency and lifecycle

- Never block indefinitely in teardown — the runtime prefers to leak rather
  than hang the host process / CI.
- Treat state transitions defensively (`EOS`, `NULL`, timeout-safe paths).
- Keep streaming-thread logic lightweight and side-effect controlled.

### MPK / manifest contract

- `mpk.json` / `*_mpk.json` is the **source of truth** for routing, dtype,
  shape, quantization, and stage decisions.
- Per-stage JSON files are plugin-private — the framework must **not**
  rewrite or validate per-stage JSON during pipeline build.
- Wiring source of truth (in order):
  1. Deterministic GStreamer element names from node fragments.
  2. `stage-id` on SIMA model-path elements
     (`simaaiprocesscvu`, `simaaiprocessmla`, `simaaiboxdecode`).
  3. `sima.model.manifest.v1` `GstContext` for static stage/tensor lookup.
- Repo boundary: this repo must **not** add build-time dependencies on
  plugin/dispatcher repos. Integration is interface-only (runtime
  `GstContext`, properties, caps/meta, C-ABI).

### Test obligations on change

From `docs/develop-apps/contribute/build-test/test_requirements.md`:

- **New node / node-group:** deterministic fragment unit test, caps/parse
  coverage, at least one integration chain.
- **Runtime/pipeline orchestration:** success-path + failure-path
  (timeout / missing plugin / invalid graph) + teardown safety.
- **Public API signature change:** compile-time coverage exercising the
  changed headers; non-breaking extensions must keep existing usage compiling.
- **Python bindings:** `pytest` coverage under `python/tests` + DLPack
  interop + smoke test against installed wheel.
- **Diagnostics / observability:** assertions on new report fields +
  concurrency-safe update tests.
- Bug fixes **must** include a regression test.

### Documentation drift guard

When behavior or public API changes:

- Update `docs/develop-apps/contribute/start-here/architecture.md` if architecture/contracts changed.
- Update user-facing docs for workflow/configuration changes.
- Update examples if API usage changed.
- If you add a new environment knob (`SIMA_GST_*` etc.), add it to the
  "Environment / configuration knobs" section of
  `docs/develop-apps/contribute/start-here/architecture.md`.
- If you change `tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`,
  update `README.md` and `docs/develop-apps/contribute/start-here/architecture.md`.

## Commit / PR conventions

- Imperative subject lines (e.g., `Add release gate policy checks`).
- Keep commits focused and scoped.
- Every PR must cover: change type, risk assessment, test evidence, docs
  impact, migration impact (see `.github/pull_request_template.md`).
- If public behavior changes, update docs in the same PR.

### PR template

Every PR description must follow this template:

````markdown
## Summary

Describe what changed and why.

## Change Type

- [ ] Bug fix
- [ ] Refactor
- [ ] Feature
- [ ] Docs
- [ ] Test-only
- [ ] Build/CI
- [ ] Breaking change

## Risk

- [ ] Low
- [ ] Medium
- [ ] High

Risk notes (required for Medium/High):

## Test Evidence

Commands run and outcomes:

```text
paste command output summary here
```

## Docs Impact

- [ ] No docs update needed
- [ ] Docs updated in this PR
- [ ] Docs follow-up required (linked issue)

Docs notes:

## Migration Impact

- [ ] No migration impact
- [ ] Migration required and documented

Migration notes (required if impact exists):

## Breaking API Change

- [ ] Not applicable
- [ ] This PR changes public API signatures under `include/*`

If applicable, provide:

- Affected headers/symbols:
- Downstream breakage risk:
- Migration steps:
- Planned release/version for the break:
- Explicit maintainer approval reference:

## API / Deprecation Checklist

- [ ] Public API unchanged
- [ ] Public API changed with deprecation path documented under `docs/develop-apps/`
- [ ] Naming contract respected (`docs/develop-apps/contribute/start-here/naming.md`)

## Release Hygiene Checklist

- [ ] `scripts/ci/check_naming_and_conflicts.sh` passed
- [ ] `scripts/ci/check_artifacts.sh` passed
- [ ] `scripts/ci/check_release_policy.sh` passed
````

## Runtime debugging knobs (cheat sheet)

Set in the environment (no recompile needed):

- `SIMA_GST_DOT_DIR` — directory to write DOT graphs on failures.
- `SIMA_GST_BOUNDARY_PROBES=1` — boundary flow counters for stall localization.
- `SIMA_GST_ELEMENT_TIMINGS=1` — per-element compute timings.
- `SIMA_GST_FLOW_DEBUG=1` — per-element flow / byte / caps counters.
- `SIMA_GST_ENFORCE_NAMES=1` — strict naming contract enforcement.
- `SIMA_GST_RUN_INPUT_TIMEOUT_MS` / `SIMA_GST_VALIDATE_TIMEOUT_MS` — timeouts.
- `SIMA_GST_TEARDOWN_TIMEOUT_MS` / `_REAPER_MS` / `_ASYNC` — teardown tuning.

Debugging order: capture `Run::report()` / `SessionError::report()` →
reproduce with `Session::describe_backend()` (`gst-launch-1.0`-ready
string) → enable the targeted probes above → inspect DOT graphs.

## Where to look first

- High-level overview: `README.md`
- Agent / contributor expectations: `AGENTS.md`, `CONTRIBUTING.md`
- Architecture deep-dive: `docs/develop-apps/contribute/start-here/architecture.md`
- Coding rules: `docs/develop-apps/contribute/start-here/coding_standard.md`
- Test rules: `docs/develop-apps/contribute/build-test/test_requirements.md`
- Naming contract: `docs/develop-apps/contribute/start-here/naming.md`
- MPK contract: `docs/develop-apps/contribute/contracts-internals/mpk_contract.md`
- Public headers: `include/`
- Canonical production e2e: `tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`
