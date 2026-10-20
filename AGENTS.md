# Agent Guidance

This file defines baseline quality expectations for automated and human contributors.

## Required standards

- Follow the coding rules in `docs/develop-apps/contribute/start-here/coding_standard.md`.
- Follow the testing rules in `docs/develop-apps/contribute/build-test/test_requirements.md`.
- Treat public headers under `include/*` as stable API.

## API compatibility

- Prefer non-breaking API changes.
- Any breaking public API signature change must follow the documented process in `docs/develop-apps/contribute/start-here/coding_standard.md`.
- PRs with breaking API changes must complete the `Breaking API Change` section in `.github/pull_request_template.md`.

## Documentation updates

When behavior or public API changes:

- Update `docs/develop-apps/contribute/start-here/architecture.md` if architecture/contracts changed.
- Update user-facing docs for workflow/configuration changes.
- Update examples if API usage changed.

For official documentation:

- Keep pages focused on the reader's task, with prerequisites, steps, and examples where useful.
- Update translations required by `sima-i18n*.json` in the same change; refresh source hashes after verifying content alignment.
- Pass localization checks and the relevant docs build, and report any validation still pending.

## Runtime install contract

When rebuilding Runner or Scheduler libraries/plugins consumed by Core:

- Keep `/usr/lib/aarch64-linux-gnu/neat/gst-plugins` and
  `/usr/lib/aarch64-linux-gnu/neat/runtime` as the canonical runtime roots.
- Prefer Core's source-first `build.sh` workflow; it stages dependency installs
  before configuring Core and deploys the completed package set once.
- For manual builds, build Runner then Scheduler with `CMAKE_INSTALL_PREFIX=/usr` and
  `CMAKE_INSTALL_LIBDIR=lib/aarch64-linux-gnu`, install into a clean staging root,
  then select that installed package when configuring Core.
- Never consume the retired Internals umbrella or its build-tree plugins as a
  fallback. Runner owns execution; Scheduler owns GStreamer and its installed contracts.
- For AArch64 test binaries, use `dk` or repository DevKit wrappers rather than
  executing them locally.

## Runtime transport ownership

- `SimaScheduler::elements` owns `neatcamerabridge`, `neatmultipartjpegdemux` and
  `neatlatestbystreammux`; Core's `src/gst/` files only preserve forwarding APIs.
- Keep graph policy, public Sample/Tensor objects and credit projection in Core.
  Do not add Core-private includes to Runner/Scheduler or recreate the Scheduler mux loan registry.
- Consume the installed Scheduler allocator/camera ABI headers. Do not mirror ABI
  structs in Core; preserve one allocation and metadata authority across DSOs.
- Keep implementation-only transport tests in Scheduler and public/graph/device
  integration tests in Core. Run both suites for cross-boundary changes.
- Install matching Runner and Scheduler packages before rebuilding Core. Preserve existing factories, properties,
  library names, zero-copy ownership and flushing behavior.

## Preproc and MPK contract invariants

When editing route planning, modelpack contract usage, or processcvu/preproc config projection:

- Treat `*_mpk.json` as the source of truth for inference-side contracts (MLA ingress/egress tensors, postprocess/inference transforms, tensor dtypes/shapes/routing).
- When inspecting a model `.tar.gz` or extracted model pack, read only the MPK manifest JSON (`mpk.json` / `*_mpk.json`).
- Ignore every other JSON inside the model archive or extracted model tree; do not derive contracts, routing, geometry, preprocess, or postprocess behavior from them.
- If the MPK manifest JSON is missing, treat that as a contract error instead of falling back to other JSON files.
- Do not treat MPK MLA ingress shape as a hard cap for preproc input image size.
- Preproc input is user/runtime-defined and may be larger than model ingress; preproc is responsible for dynamic resize.
- Preproc output must match model/inference ingress expectations (for example, resize to `640x640` when that is the model input contract).
- `quant`, `tess`, and `quanttess` are shape-preserving transforms; they do not perform spatial resize.

In short: MPK inference contract constrains preproc output, not preproc input.

## Definition of done

A change is ready when:

- Relevant tests were added/updated and pass.
- Docs are updated for user-visible changes.
- API compatibility impact was assessed and documented.

## Source-first local builds

- `build.sh` delegates dependency preparation and closure validation to Scheduler, then builds Core against the installed frontend. Scheduler coordinates Kernels, Runner and Codecs; retain exact receipts for the complete runtime stack.
- `NEAT_KERNELS_SOURCE` / `NEAT_RUNNER_SOURCE` / `NEAT_SCHEDULER_SOURCE` select source roots; the corresponding
  `_PACKAGE_DIR` overrides select local package directories instead.
- Do not silently fetch old Internals, replace failed source builds with artifacts,
  or deploy from child builds. Explicit `tools/ci/acquire_scheduler.py` acquisition
  resolves one same-branch immutable Scheduler publication and lets its installed
  coordinator validate the delivered cohort; `build.sh` never silently fetches it.
- Preserve the selected package receipts and contract checks. Rebuild consumers
  after ABI changes; numerical source-only changes do not require Runner/Scheduler source edits, but static consumers must relink to consume new Kernels artifacts.
- See `docs/develop-apps/contribute/build-test/local-dependencies.md`.

## Canonical numerical preparation

- Core owns numerical request construction, MPK/publication policy and dynamic image geometry.
- Use the installed `SimaScheduler::frontend` and `simaai::scheduler::client` APIs. Scheduler owns the compiled boundary; Runner delegates numerical admission to Kernels. Do not use the advanced lower-type alias facade from canonical Core code.
- Do not restore the removed ProcessCVU prepared ABI, adapter or per-operation Scheduler dispatch.
- Use generic `TensorStageRequest` publication/placement, immutable constants and the application-owned frame adapter. Constants are copied once at preparation, not uploaded per frame.
- Reusable Core recipes retain immutable checked policy and executable descriptors only. Instantiate their resolver caches and Preproc adapters once per cold preparation, after topology validation and before Scheduler admission; never reset or share an active geometry/session cache between independent preparations. Preserve current frame loans and bounded immutable geometry caching.
- Manifest contexts own metadata backing, not numerical recipes or sessions. Normalize canonical payloads to `NONE` before dropping recipe references; retain contexts while borrowing ABI pointers. Keep cold pipeline/context ownership under RAII until successful handoff.
- The retained shared prepared context is for MLA/other nonnumerical compatibility, not numerical kernel requests.

## Scheduler client boundary

- Canonical Core must not include, find, link or invoke Runner, Kernels or Codecs development APIs. Scheduler-owned historical `runtime::gstreamer` ABI names remain valid; ownership is determined by the defining target, not namespace spelling.
- Keep customer APIs, MPK interpretation, graph budgets, target/fusion policy and image geometry in Core. Forward generic checked numerical requests, not operation-specific Scheduler dispatch.
- Treat checked executable, prepared selection, allocation identity and frame loan as different identities. Copy immutable authoring recipes; instantiate mutable adapters/resolvers only during whole-cohort cold admission.
- Use Scheduler selected-memory access for the original authority and scope. Do not substitute broad Gst mapping, COW, raw FD reimport, copies or destruction-as-retirement. Borrowed access cookies must finish while their owner remains alive.
- Keep host publication callbacks allocation-free on warmed success. Buffer views are call-scoped; error views must refer to immutable payload-owned strings or literals.
- Core tests retain policy, numerical and publication assertions. Explicit `SimaScheduler::client_test_support` is test-only; it hides provider mechanics, not expected outputs or independent references.
- Verify installed frontend consumers with lower development discovery/headers denied, plus actual shared/static consumers. Preserve existing workflows and GCC14. Separate host, board correctness and matched performance evidence.
