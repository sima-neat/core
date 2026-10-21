---
title: Diagnose SSD component latency
description: Retain performance evidence without changing the acceptance contract
---

# Diagnose SSD component latency

Use this procedure when `ssd_mobilenet_boxdecode` fails its component-latency
budget. A passing FPS result does not override a failing component p99.

## Keep the measurement boundaries explicit

The current graph prepares numerical BoxDecode and its wire-publication adapter
as separate A65 members. Core owns the output wire format; Kernels owns the
numerical operation. Scheduler submits the checked members through Runner.

| Component | Begins | Ends | Included work |
|---|---|---|---|
| `boxdecode_backend_wall_v2` | Numerical decode invocation | Dependent wire-publication invocation completes | Decode, intervening scheduling and access work, wire encoding |
| `boxdecode_stage_wall_v2` | Complete-frame admission, before output acquisition | Retirement, finish and input release complete | Pool acquisition, binding, all member execution and finalization |

Neither stage timing nor backend timing includes that frame's own downstream
push. The backend span excludes any preceding conversion members. The stage
span includes them.

These are wall-time envelopes, not CPU utilization or pure kernel time. They
are deliberately different identities from the old `boxdecode_backend` and
`boxdecode_plugin_exclusive` spans. Do not compare unlike spans as proof of a
numerical-kernel regression, or remove queue time to make a result pass.

The checked-in weekly baseline requires 1,000 samples and absolute p99 limits
of 0.7 ms for backend and 0.93 ms for stage. The general throughput tolerance does
not widen these component limits. Keep those limits, fixtures, warmups,
iterations and deadlines unchanged during diagnosis.

## Retain the existing evidence

The performance matrix now saves complete captured stdout and stderr for every
scenario attempt, including successful runs, malformed output, skips and
supervisor failures. Each result's `run_meta.artifacts` points to a unique
`scenario-logs/` directory containing:

- `stdout.log` and `stderr.log`: complete captured streams, not only their tails.
- `invocation.json`: command, working directory, iterations, deadline and the two
  trace-control environment values. It does not dump the whole environment.
- `outcome.json`: comparison result and available process/supervisor outcome.

Logs contain whatever the executable emits; review them before sharing. A
supervisor refusal may supply no process output. Empty logs are not proof that
a process ran, finished or released device resources. Preserve its failure
record and follow the normal recovery procedure.

For a separate diagnostic run, use the existing controls:

- `SIMA_NEAT_METRICS_RETAIN_TRACE=1` retains LTTng Common Trace Format (CTF) files.
- `SIMA_NEAT_METRICS_TRACE_DIR` selects their parent directory; the collector
  creates a unique child directory for each session.

Run the matrix **on the DevKit**, inside a fresh admitted hardware-test scope
with the correct installed providers and firmware. Configure these environment
values before admission; do not modify a live pinned qualification scope. This
guarded diagnostic procedure opts into the persistent supervisor through
`SIMA_HARDWARE_SUPERVISOR_SOCKET`; standard CI does not require it. The matrix
does not obtain board ownership or authorize firmware changes itself.

For example, from the target Core checkout, after selecting the installed test
directory and admitting the diagnostic environment:

```bash
: "${CORE_ROOT:?set the target Core checkout path}"
: "${CORE_TESTS:?set the installed target test directory}"
: "${SIMA_HARDWARE_SUPERVISOR_SOCKET:?use an admitted hardware-test scope}"
cd "$CORE_ROOT"
export SIMA_NEAT_METRICS_RETAIN_TRACE=1
export SIMA_NEAT_METRICS_TRACE_DIR="$CORE_ROOT/ssd-diagnostic-traces"
python3 tests/perf/tools/run_perf_matrix.py \
  --repo-root "$CORE_ROOT" \
  --prebuilt-tests-dir "$CORE_TESTS" \
  --include-long \
  --results-dir "$CORE_ROOT/ssd-diagnostic-results"
```

This runs the full weekly performance matrix, including SSD. It does not rebuild
providers. Use SDK cross-compilation and the DevKit launcher for target programs;
never execute AArch64 tests on an x86 host.

With trace retention, a nonempty trace-directory override, or an iteration
override, the result records `run_meta.execution_purpose: diagnostic`. Comparisons
and failure codes remain unchanged; the label is not permission to relax the
gate. Without those overrides it records `baseline_comparison`, which by itself
is not proof of a complete canonical CI run.

The requested trace parent is not the actual session identity. Read
`run_meta.measure_report.plugin_latency.trace_dir` and its collection status,
then archive that exact directory together with the result and scenario logs.
Preserve plugin instance, run, graph, frame and request identities from the CTF
records. Report missing events or trace loss rather than inventing correlations.
Reusing a results directory replaces top-level result JSON but preserves earlier
attempt-log directories; use fresh result and trace roots for separate campaigns.

## Attribute the cost before optimizing

Retained wall-span CTF can expose outliers and event identity, but cannot by
itself separate numerical time from the decode-to-publication scheduling gap.
Do not subtract independently calculated percentiles. Correlate the same frames
and collect these boundaries in a separately instrumented build if needed:

1. Output acquisition and binding/access preparation.
2. Member enqueue, worker start, execution begin and execution end.
3. Numerical decode end to publication execution begin.
4. Publication execution, access retirement, finish and input release.
5. Allocation, import, copied-byte and queue/wakeup counts in the warmed window.

Runner and Scheduler have private `SIMA_RUNTIME_TEST_TRACE` test seams, but
production builds disable them. Using those seams requires separately built,
matching providers and verified cross-thread/member correlation; enabling an
environment variable alone does not create those events. Keep instrumented
measurements separate from the final canonical performance result.

If dispatch dominates, first evaluate a generic bounded ready-work continuation
that preserves each admitted job and its current memory loans. Existing
`PreparedSequence` accepts persistent `BoundOperation` objects; it is not a
drop-in replacement for Scheduler's invocation-only DMA loans. Never pretend
those loans are persistent, run dependent work recursively inside completion
callbacks, or remove access epochs to save time.

If decode dominates, optimize the numerical implementation in Kernels with
output-parity checks. If acquisition or publication dominates, optimize that
measured path without adding hidden payload copies or weakening ownership.
Require fairness, bounded stack/capacity, failure propagation, rebinding,
retirement and accurate metric-boundary tests for any scheduling change.

Finally rerun the unchanged canonical matrix without diagnostic overrides.
Retain the original failed result; a diagnostic pass does not replace it.
