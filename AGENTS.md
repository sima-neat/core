# Agent Guidance

This file defines baseline quality expectations for automated and human contributors.

## Required standards

- Follow the coding rules in `docs/contribute/coding_standard.md`.
- Follow the testing rules in `docs/contribute/test_requirements.md`.
- Treat public headers under `include/*` as stable API.

## API compatibility

- Prefer non-breaking API changes.
- Any breaking public API signature change must follow the documented process in `docs/contribute/coding_standard.md`.
- PRs with breaking API changes must complete the `Breaking API Change` section in `.github/pull_request_template.md`.

## Documentation updates

When behavior or public API changes:

- Update `docs/contribute/architecture.md` if architecture/contracts changed.
- Update user-facing docs for workflow/configuration changes.
- Update examples if API usage changed.

## Internals install contract

When changing or rebuilding anything under `internals/` that produces NEAT runtime/plugin `.so` files:

- Treat `/usr/lib/aarch64-linux-gnu/neat/gst-plugins` and `/usr/lib/aarch64-linux-gnu/neat/runtime` as the canonical runtime roots.
- Configure the `internals/` build for the canonical Debian multiarch install layout:
  `cmake -S internals -B <internals-build-dir> -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/aarch64-linux-gnu`.
- After `cmake --build <internals-build-dir>`, always run `cmake --install <internals-build-dir>` before rebuilding or testing `core/`.
- Do not rely on `internals/build/artifacts/gst-plugins` as the default runtime source for `core`; build-tree plugin use should be an explicit override for debugging only.
- For `aarch64` test binaries, use `dk` or the repo devkit wrappers instead of local execution.

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
