# Pipeline and automatic caps negotiation {#pipeline_concepts}

## Building blocks (Node, reusable Graph fragment, Graph)

- **Node** (`include/builder/Node.h`) is the atomic unit.
- **Reusable Graph fragment** is a premade `Graph` that contains an ordered set of nodes.
- **Graph** (`include/pipeline/Graph.h`) is the public composition and build layer.

Graph composes these into deterministic GStreamer pipelines.

Within one Graph composition, one Node object is one logical vertex. Adding or importing the same
object twice fails atomically. Use the same object in multiple `connect()` calls to create fan-out,
or construct a separate Node object to create another vertex of the same type.

## Deterministic naming

Element names are derived from node order (`n<idx>_...`). This keeps:

- `describe_backend()` stable
- diagnostics/probes reproducible
- contract checks deterministic

`build()` validates names automatically; you do not need to call `validate()` first. Explicit
short names must be unique within each materialized GStreamer pipeline segment. Separate connected
segments may reuse a name because they are parsed independently. Name declarations and named-pad
references are transformed together, but collisions are never auto-renamed.

Complete segments fail during `build()`. A connected segment that depends on the first input may
materialize on the first `push()`, in which case its structured `GraphReport` is available through
the resulting runtime error or `PullError`.

For a connected Graph, `validate()` checks endpoint topology without inventing placeholder input
caps. The mandatory launch check still runs against each segment's real final string when that
segment materializes.

## Push pipeline negotiation model

For `Input` pipelines, runtime derives caps from real inputs and applies caps on
`appsrc` before streaming. Later inputs are validated against runtime policy:

- **Raw video (`video/x-raw`)**:
  format/size changes are handled automatically with a stability guard.
- **Tensor (`application/vnd.simaai.tensor`)**:
  tensor caps are strict at runtime (shape/dtype/layout changes are rejected).
- **Encoded**:
  encoded `caps_string` changes are rejected.

## Dynamic input behavior

Dynamic spatial dimensions are accepted by default.

- Width/height changes renegotiate caps automatically.
- `stability_frames` controls anti-flap behavior:
  - `1`: apply immediately
  - `>1`: require N consistent frames before switching

For tensor rank normalization:

- If input rank is `N+1` and leading dim is `1`, runtime treats it as implicit
  batch-1 and normalizes to `N`.
- If leading batch is `>1` and the pipeline is not batch-capable, runtime fails
  with an explicit error.

## Buffer sizing and hard-fail policy

Runtime enforces a strict input allocation policy:

- `RunAdvancedOptions::max_input_bytes` is the hard cap.
- If required bytes exceed allocated/capped bytes, runtime hard-fails with
  actionable error text.
- No one-time grow behavior is exposed.

## Fixed-caps mode

`InputOptions::caps_override` pins caps for push pipelines.

- When set, renegotiation is disabled.
- Use only for truly fixed inputs.

## Output-side normalization

`Graph::add_output_tensor(...)` inserts convert/scale/caps + output sink, so
downstream output contracts are explicit and stable.

## Source pipelines

For file/RTSP/image-group source pipelines:

- negotiation is handled by source/decode elements
- explicit caps constraints can still be inserted with caps nodes

## Runtime options mental model

Use `RunOptions` for intent-driven control:

- `preset`: `Realtime` / `Balanced` / `Reliable`
- `queue_depth`
- `overflow_policy`
- `output_memory`

Use `RunAdvancedOptions` only for expert tuning:

- `copy_input`
- `max_input_bytes`

## Practical guidance

- Prefer preset + small queue tuning first.
- Use `Balanced` for default operation; it may start zero-copy and fall back to
  copied output if reliability trips.
- Use `Reliable` for deterministic ownership and conservative behavior.
- Set `max_input_bytes` in long-running services to cap memory growth.
