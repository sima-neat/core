# 011 Diagnose and Profile a Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Model | None |
| Labels | diagnostics, debugging, observability |

## Concept

Triage a pipeline with three checks — `graph.validate()`, one metrics-enabled `run.run()`, and `run.stats()` + `run.report()` — to answer whether it is wired correctly and how it is performing, before reaching for deep debugging.

## Walkthrough

When a pipeline misbehaves, the temptation is to jump straight into element-level debugging. This chapter teaches the cheaper first move: a repeatable triage pass that answers three questions in order — *Is the graph contract valid? Does one run succeed? What do the runtime diagnostics say?* It catches most misconfiguration in seconds, before it becomes a multi-hour session, and it works on the same minimal Input → Output graph you already know from chapter 003.

By the end you will have validated a graph's contract, run a single frame with metrics turned on, and printed the runtime counters, report size, and diagnostics summary that tell you whether the pipeline is healthy.

### Validate the contract {#step-validate-graph}

`validate()` is a contract-level check that runs *before* `build()`. It exercises the node order, caps, and backend parse path without streaming any data, and returns a report carrying a canonical `error_code`. An empty/`ok` code means the graph is structurally sound; anything else buckets the failure (see the error taxonomy below) so you know where to look. Running this first means you never waste time debugging runtime behavior on a graph that was never going to build.

### Run one frame with metrics {#step-run-with-metrics}

Next, build and run a single deterministic frame — but with `enable_metrics = true` on `RunOptions` so the runtime records latency and counter data that `stats()` can read back. `output_memory = Owned` asks for owned output buffers so the result stays valid after the call. One frame is enough: if it succeeds, the pipeline is live; if it throws, the exception carries a structured report you can bucket the same way as `validate()`.

### Read the runtime diagnostics {#step-read-diagnostics}

With one run on record, three accessors summarize the pipeline's health. `stats()` returns the flow counters (`inputs_enqueued`, `outputs_pulled`, latency) — the fastest confirmation that frames went in and came out. `report()` is the per-element structured report; its size tells you how many elements were instrumented. `diagnostics_summary()` is a single human-readable string fit for a log line or a support bundle. Together they are the baseline you capture before escalating to the probes and DOT graphs described in [In Practice](#in-practice).

## Run

Run it and you should see the validate code, run counters, and diagnostics summary printed to stdout. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

**Python:**
```bash
python3 share/sima-neat/tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_diagnose_a_pipeline
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_diagnose_a_pipeline
./build/tutorials-standalone/tutorial_011_diagnose_a_pipeline
```

Expected output (counter values and the summary string vary by run):

```text
validate.error_code=
stats.inputs_enqueued=1 outputs_pulled=1
report.size=2
diagnostics_summary=
[OK] 011_diagnose_a_pipeline
```

(The Python build prints `validate_error_code=`, `inputs_enqueued=... outputs_pulled=...`, `report_size=...`, and `diagnostics_summary=...`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

Structured diagnostics, the error taxonomy, debug knobs, and the plugin-failure workflow you reach for when `validate()` / `stats()` / `report()` point at a problem.

### GraphReport

`GraphReport` captures structured diagnostics:
- pipeline string (for reproduction)
- canonical `error_code` (machine triage)
- `repro_note` (human summary + hint)
- node reports and owned element names
- bus messages and error details
- optional flow/timing counters

When an error occurs, `NeatError` carries a `GraphReport` you can log or serialize.

### Error taxonomy

Framework errors use stable code families:

| Error code | Meaning | Typical fix |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Node order/shape contract violation | Ensure `Input()` first for push pipelines and `Output()` last for pull pipelines |
| `misconfig.caps` | Caps negotiation/override mismatch | Align `caps_override`, format, and downstream caps |
| `misconfig.input_shape` | Input tensor/frame/sample shape/layout mismatch | Validate width/height/depth, layout, dtype, storage |
| `build.parse_launch` | `gst_parse_launch` failed | Validate fragment syntax and plugin availability |
| `runtime.pull` | Runtime pull/timeout/closed-output failure | Check sink output production, queue pressure, and upstream errors |
| `io.parse` | Saved-graph JSON parse/schema failure | Validate JSON and required node fields |
| `io.open` | Graph save/load file open/read/write failure | Check path existence, permissions, and storage health |

`PullError.code` uses the same taxonomy (not only exception paths).

### Programmatic handling

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error &&
      perr.code == simaai::neat::error_codes::kRuntimePull) {
    // runtime pull triage path
  }
} catch (const simaai::neat::NeatError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kParseLaunch) {
    // build/parse-launch triage path
  }
}
```

### Debug knobs (environment)

Key environment variables (see [Architecture](/contribute/architecture) for detail):
- `SIMA_GST_DOT_DIR`: write DOT graphs for failures
- `SIMA_GST_BOUNDARY_PROBES`: boundary flow counters
- `SIMA_GST_ELEMENT_TIMINGS`: per-element timings
- `SIMA_GST_FLOW_DEBUG`: per-element flow counters
- `SIMA_GST_ENFORCE_NAMES`: enforce naming contract

### Debug workflow

1. Capture `GraphReport.error_code` and bucket the failure by taxonomy first.
2. Capture `GraphReport.repro_note` for concrete context and built-in hint.
3. Capture pipeline text: `Graph::describe_backend()` or `last_pipeline()`.
4. Capture structured diagnostics: `Run::report()` or `NeatError::report()`.
5. Inspect `GraphReport.bus` for first terminal `ERROR` source + detail.
6. If runtime stalls/timeouts, enable boundary/element probes to localize flow stop.

Recommended support bundle:
- `error_code`
- `repro_note`
- full `pipeline_string`
- first 3-5 terminal bus errors (`GraphReport.bus`)
- environment overrides used in run/validate

### Common failures → fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing ... plugin` | GStreamer plugin not found | Check `GST_PLUGIN_PATH`, run `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | Missing terminal `Output()` | Ensure `Output` is the last node in run/build pipelines |
| `caps_override is set; renegotiation disabled` | caps pinned | Remove `caps_override` or keep input caps fixed |
| `tensor caps change not supported` | Tensor shape/dtype change at runtime | Keep tensor shape/dtype stable (no renegotiation) |

### Debugging plugin failures

When a plugin fails, NEAT raises a `NeatError` whose message contains the GStreamer error and a structured debug string. Use the fields to locate the root cause quickly.

1. **Read the structured fields.** Look for the `debug` key/value fields in the error text:
   - `node`: the failing element name in the pipeline
   - `config_path`: JSON config file (if applicable)
   - `model_path`: model/pack path (if applicable)
   - `hint`: actionable fix guidance
   - `detail`: extra context such as missing keys or allocator state

   See the [Error Format Reference](/reference/error_format) for the full list.
2. **Confirm the pipeline context.** Use the pipeline string from `Graph::last_pipeline()` or from the error report:
   - Verify the `node` name appears in the pipeline.
   - Confirm the `config_path` exists and is readable.
   - For caps errors, check upstream elements that negotiate into the failing node.
3. **Apply common fixes.**
   - **Config errors**: verify JSON syntax, required keys, and any model paths.
   - **Caps errors**: add or fix parser elements (e.g., `h264parse`), ensure caps include required fields like `parsed=true`, `stream-format=byte-stream`, `alignment=au`.
   - **Allocator errors**: ensure upstream elements use the required allocator type (system vs. simaai memory/segment).
4. **Capture more diagnostics** with the debug knobs above (`SIMA_GST_DOT_DIR`, `SIMA_GST_FLOW_DEBUG`, `SIMA_GST_ELEMENT_TIMINGS`).

## Source Files
- C++: `tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.cpp`
- Python: `tutorials/011_diagnose_a_pipeline/diagnose_a_pipeline.py`
