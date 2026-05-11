# Diagnostics and debugging {#diagnostics}

## SessionReport

`SessionReport` captures structured diagnostics:
- pipeline string (for reproduction)
- canonical `error_code` (machine triage)
- `repro_note` (human summary + hint)
- node reports and owned element names
- bus messages and error details
- optional flow/timing counters

When an error occurs, `SessionError` carries a `SessionReport` you can log or
serialize.

## Error taxonomy

Framework errors use stable code families:

| Error code | Meaning | Typical fix |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Node order/shape contract violation | Ensure `Input()` first for push pipelines and `Output()` last for pull pipelines |
| `misconfig.caps` | Caps negotiation/override mismatch | Align `caps_override`, format, and downstream caps |
| `misconfig.input_shape` | Input tensor/frame/sample shape/layout mismatch | Validate width/height/depth, layout, dtype, storage |
| `build.parse_launch` | `gst_parse_launch` failed | Validate fragment syntax and plugin availability |
| `runtime.pull` | Runtime pull/timeout/closed-output failure | Check sink output production, queue pressure, and upstream errors |
| `io.parse` | Saved-session JSON parse/schema failure | Validate JSON and required node fields |
| `io.open` | Session save/load file open/read/write failure | Check path existence, permissions, and storage health |

`PullError.code` uses the same taxonomy (not only exception paths).

## Programmatic handling

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"

try {
  auto run = session.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error &&
      perr.code == simaai::neat::error_codes::kRuntimePull) {
    // runtime pull triage path
  }
} catch (const simaai::neat::SessionError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kParseLaunch) {
    // build/parse-launch triage path
  }
}
```

## Debug knobs (environment)

Key environment variables (see [Architecture](/contribute/architecture) for detail):
- `SIMA_GST_DOT_DIR`: write DOT graphs for failures
- `SIMA_GST_BOUNDARY_PROBES`: boundary flow counters
- `SIMA_GST_ELEMENT_TIMINGS`: per-element timings
- `SIMA_GST_FLOW_DEBUG`: per-element flow counters
- `SIMA_GST_ENFORCE_NAMES`: enforce naming contract

## Debug workflow

1) Capture `SessionReport.error_code` and bucket the failure by taxonomy first.
2) Capture `SessionReport.repro_note` for concrete context and built-in hint.
3) Capture pipeline text: `Session::describe_backend()` or `last_pipeline()`.
4) Capture structured diagnostics: `Run::report()` or `SessionError::report()`.
5) Inspect `SessionReport.bus` for first terminal `ERROR` source + detail.
6) If runtime stalls/timeouts, enable boundary/element probes to localize flow stop.

Recommended support bundle:
- `error_code`
- `repro_note`
- full `pipeline_string`
- first 3-5 terminal bus errors (`SessionReport.bus`)
- environment overrides used in run/validate

## Common failures → fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing ... plugin` | GStreamer plugin not found | Check `GST_PLUGIN_PATH`, run `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | Missing terminal `Output()` | Ensure `Output` is the last node in run/build pipelines |
| `caps_override is set; renegotiation disabled` | caps pinned | Remove `caps_override` or keep input caps fixed |
| `tensor caps change not supported` | Tensor shape/dtype change at runtime | Keep tensor shape/dtype stable (no renegotiation) |

For structured plugin errors and actionable hints, see
[How To Debug Plugin Failures](/how-to/plugin_failures).
