# Recipes

## 1) Sync request/response pipeline

Use when you want one input to one output with straightforward control flow.

- Build session with input + processing + output nodes.
- Build `PipelineRun` with `PipelineRunMode::Sync`.
- Use `push_and_pull(...)`.

## 2) Async streaming pipeline

Use for continuous throughput and decoupled producer/consumer.

- Build session once.
- Create `PipelineRun` with `PipelineRunMode::Async`.
- `push(...)` in producer loop, `pull(...)` in consumer loop.
- Close input with `close_input()` when done.

## 3) Model pipeline via `Model`

Use when model pack stage composition is required.

- Construct `simaai::neat::Model`.
- Add `model.preprocess()`, `model.inference()`, `model.postprocess()` or `model.session()`.
- Build/run through `Session` or `Model::Runner`.

## 4) Source pipeline (no pushed input)

Use for RTSP/file source-driven flows.

- Build `Session` with source node/group and terminal output.
- Use `Session::build(const PipelineRunOptions&)`.
- Pull outputs from `PipelineRun`.

## 5) Validation-first flow

Use for pipeline generation/contract checks before execution.

- `Session::validate(...)`
- Inspect `PipelineReport` for parse/caps/plugin failures.

## 6) RTSP serving flow

Use when you need to expose a session output over RTSP.

- Build a source-driven session (image/video/rtsp input groups or equivalent).
- Call `Session::run_rtsp(const RtspServerOptions&)`.
- Keep `RtspServerHandle` alive for serving lifetime.

## 7) Input caps and renegotiation control

Use when input format or dimensions may vary.

- `InputAppSrcOptions` controls negotiated caps behavior.
- `caps_override` fixes caps and disables renegotiation behavior.
- For dynamic dimensions, set max bounds (`max_width`, `max_height`, `max_depth`) and handle failures explicitly.

## 8) Model stage composition flow

Use when you need explicit model stages rather than a single opaque pipeline.

- Compose with:
  - `model.preprocess()`
  - `model.inference()`
  - `model.postprocess()`
- Or use `model.session()` for full default model pipeline.
- For advanced wiring, combine model groups with explicit nodes (boxdecode, joins, sinks).

## 9) Hybrid graph flow (advanced)

Use when you need DAG orchestration across pipeline and stage-style nodes.

- Use `graph::Graph` + `graph::GraphSession`.
- Key nodes for multistream:
  - `StreamSchedulerNode`
  - `FanOutNode`
  - `JoinBundleNode`
  - `LambdaStageNode`
- Ensure stream metadata (`stream_id`, `frame_id`) is preserved for joins/schedulers.

## 10) PCIe flow

Use for host/SoC send-receive and model execution via PCIe plugins.

- Verify plugin availability (`element_exists(...)`) before run.
- Configure explicit buffer sizing and sink/source options.
- Treat missing plugins as skip/fallback condition, not silent success.
