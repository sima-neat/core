# Driver consolidation checkpoint — 2026-09-09

## Scope

Consolidate the existing driver supersets. Do not rebase onto `develop`,
migrate the kernel, change customer API signatures, or deploy this checkpoint.
The review base requested for both PRs is `3.0.0-prep`. Target-branch adaptation
and board acceptance remain separate from this checkpoint.

| Source | Starting commit |
| --- | --- |
| Core `codex/dmabuf-driver-consolidated` | `cd2b33d5e28aede0cad8c8456132070c67024402` |
| Internals `codex/dmabuf-driver-consolidated` | `a4085ed51663c7af77e4e253325abc0ccf5161ae` |
| Matching kernel, unchanged | `bae3db458eb0cbfd02e935efc62ffa65df7f4d41` |

`decisions.csv` records path-level dispositions and reference blob IDs. No
foundation, adoption, codec, or older cleanup branch was merged wholesale.

## Preservation

Source snapshots are upstream under `backup/driver-consolidation-20260909-*`.
The `core-consolidation`, `core-local`, `internals-consolidation`, and
`internals-local` suffixes identify the four working source checkpoints.
Additional snapshots retain dirty comparison worktrees, including unresolved
merge trials; their original indexes were not resolved or overwritten.
The `core-index-state` and `internals-index-state` backup branches retain exact
indexes, staged/unstaged patches, untracked archives and worktree inventories.
Local full Git bundles preserve history independently of the remote.

Existing worktrees and the packaged `app48-demo:20260817a` payload were not
modified. The demo package is a preserved reference, not an artifact produced
or qualified by this consolidation.

## Decisions that matter

- Keep DMA-BUF physical planning and typed model contracts. Semantic model
  ownership is independent of whether executable plans have been cached.
- Keep prepared MLA execution, CVU reusable schemas/persistent imports,
  shared DMA-BUF ownership, and direct codec sessions from the supersets.
- Remove the memory-backend selector and production admission socket.
  Graph policy still chooses zero-copy boundaries; the decoder library owns
  reservations on kernel command FDs. Diagnostic and admission-policy knobs
  do not select a transport or bypass the kernel's capacity checks.
- Keep the internal injectable admission-policy seam for its existing unit
  coverage. It has no production socket client.
- Preserve the full compiler-authored input carrier while validating the
  MLA runtime's accessed prefix. Reject too-small carriers; do not inflate
  the device binding merely to match host allocation padding.
- Preserve SystemMemory input APIs: a graph-authored NV12 materialization
  boundary performs the required copy into CMA. Already device-visible input
  remains direct. Do not silently restore an encoder upload fallback.
- Keep ConfigManager, AppComplex/shared configuration and supported PCIe,
  camera, software-codec and multi-output graph behavior. An old directory
  name is not grounds for deleting useful code.
- Exclude wholesale CVU comparison-test deletions, raw-H265 API expansion,
  older PCIe deletions and separate EV74 algorithm changes. In particular,
  the old C3 tessellation edit is not a host-driver prerequisite: INT8 was
  already dispatched earlier, and the image kernel itself has element-size
  handling. Do not claim a new firmware correctness result from this build.

## Integration fixes found by the clean build

- Declare the exported CMake package's GStreamer/GLib dependency targets.
- Include the capability ABI and tensor-meta alias in the development component.
- Quarantine retired plugin objects in the private plugin directory too.
- Link the existing CVU syscall unit test to the existing DMA-BUF test-support
  copy. Linker `--wrap` cannot intercept calls inside the production shared
  library. Production still has exactly one shared DMA-BUF implementation;
  the cross-DSO ownership test remains unchanged.

## Validation boundary

Internals and Core are cross-built for ARM64. Core consumes the fresh installed
`NeatInternals` package, not build-tree libraries or stale SDK plugin headers.
The packaged component view contains only the intended private runtime/plugin
paths and the launch scripts from `debian/rules`.

The removed environment-parser test has no implementation left to test.
Existing tensor placement, eligibility and model tests remain. The decoder
boundary test checks direct command FDs instead of an obsolete socket lease;
its frame, rate, zero-copy and independent teardown assertions are retained.
The VideoSender test retains its API inputs and validates explicit NV12
materialization, including aligned plane storage.

No new CI gates are introduced for this checkpoint. Native driver checks are
supplementary, not substitutes for established CI/CD. The fixture-specific
QMLA golden test requires its exact ELF; a different ResNet package is not an
acceptable substitute. No golden assertions or skip policies were weakened.

Not claimed: complete CI/CD acceptance, firmware activation, saturated FPS,
48-channel accuracy/rate, fault recovery, or absence of board regressions.
The legacy dispatcher-comparison test sources are retained for review rather
than deleted to manufacture a passing count; they are not used by production
or enabled in this production build. Board validation must use the pinned
stack on the development board, never the stable demo device.
