---
title: Runtime Tuning
description: Queue sizing, drop policy, and runtime knobs
sidebar_position: 1
---

# Runtime tuning & safety knobs

This page expands on the Doxygen runtime docs with practical guidance for
queue sizing, drop policies, caps changes, and output lifetime safety.

## Queue sizing (`queue_depth`)

Heuristics:
- Start with `queue_depth = 4–16` for low‑latency pipelines.
- Increase queues if your producer is bursty or if downstream elements have
  variable latency (decode/MLA/postproc).
- Keep queues small if you need **freshest** frames (e.g., live camera preview).

## Overflow policy (`RunOptions::overflow_policy`)

- `Block`: safest for correctness; producer waits when queue is full.
- `DropIncoming`: keep queued work, drop incoming samples when saturated.
- `KeepLatest`: prefer freshest frames, drop the oldest queued samples.

For live feeds, `KeepLatest` usually yields the lowest end-to-end latency.

## Presets and renegotiation

Use `RunOptions::preset` to control latency/safety tradeoffs:
- `Realtime`: lowest latency, aggressive freshness behavior.
- `Balanced`: starts zero-copy when possible, runs startup probe checks, and falls back to copy mode if reliability trips.
- `Reliable`: conservative behavior and stable output ownership.

Input shape renegotiation is automatic for dynamic inputs.

## Output lifetimes (`output_memory`)

- `output_memory = Owned`: returned `Tensor` owns its data.
- `output_memory = ZeroCopy`: tensor may reference runtime buffers reused after pull.
- `output_memory = Auto`: runtime chooses zero-copy first and falls back to owned
  where reliability requires it.

If you need to keep tensor data beyond the current step, call `clone()` or
`cpu().contiguous()`.

## Caps change decision flow

Use this guide for **push pipelines** (`Input`):

1) **Do you need fixed caps?**
   - Yes → set `InputOptions::caps_override`. Renegotiation is disabled.
   - No → leave `caps_override` empty.

2) **Do you expect size changes?**
   - Yes → no extra flag is required; dynamic dimensions are accepted by default.
   - No → set fixed dimensions in `InputOptions` and optionally use `caps_override`.

## Buffer pool safety

- `RunAdvancedOptions::max_input_bytes` sets a hard upper bound on input buffer allocation.
- If a larger buffer is required, runtime fails fast with an explicit error.

Use these to protect long‑running processes from unbounded allocations when
inputs change size.

## Related docs

- [Tutorials](/tutorials)
- [Data Formats](/reference/data_formats)
- [Architecture](/contribute/architecture)
