# Advanced Patterns

## Plugin availability checks

Before using hardware/path-specific flows, check required elements:

- Example pattern: `simaai::neat::element_exists("...")`
- If unavailable, return a clear skip/fallback path.

Use this for:

- SimaAI processing plugins (`simaaiprocesscvu`, `simaaiprocessmla`, `simaaiboxdecode`)
- PCIe plugins (`simaaipciesrc`, `simaaipciesink`)

## `InputAppSrcOptions` contract-heavy setups

When generating appsrc-driven code:

- Explicitly set format and dimensions when known.
- Use `caps_override` only when truly fixed caps are desired.
- Set dynamic bounds (`max_width`, `max_height`, `max_depth`) for variable inputs.
- Keep `buffer_name` aligned with routing conventions when pushing `Sample` messages.

## Sync vs async selection

Prefer:

- Sync for deterministic single request/response paths and simplest correctness.
- Async for throughput and decoupled producer/consumer loops.

In async code, always handle:

- `PullStatus::Timeout`
- `PullStatus::Closed`
- `PullStatus::Error`

## Output policy tuning

For terminal behavior and backpressure control:

- Tune `OutputAppSinkOptions` (`max_buffers`, `drop`, `sync`)
- Use `OutputTensorOptions::sink` when using `add_output_tensor(...)`.

## Graph multistream orchestration

For multi-source or fairness-sensitive pipelines:

- Use `StreamSchedulerNode` for fair scheduling.
- Use `FanOutNode` to branch shared streams.
- Use `JoinBundleNode` for synchronized multi-branch merge.
- Preserve stream keys (`stream_id`, `frame_id`/`pts`) consistently.
