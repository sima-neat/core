---
title: Diagnostics and debugging
description: Collect GraphReport diagnostics, runtime error codes, and graph metrics artifacts
sidebar_position: 9
---

# Diagnostics and debugging

## GraphReport

`GraphReport` captures structured diagnostics:
- pipeline string (for reproduction)
- canonical `error_code` (machine triage)
- `repro_note` (human summary + hint)
- node reports and owned element names
- bus messages and error details
- optional flow/timing counters

When an error occurs, `NeatError` carries a `GraphReport` you can log or
serialize.

## Error taxonomy

Framework errors use stable code families:

| Error code | Meaning | Typical fix |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Node order/shape contract violation | Ensure `Input()` first for push pipelines and `Output()` last for pull pipelines |
| `misconfig.caps` | Framework caps-override or adjacent Node contract mismatch | Align `caps_override` and the declared Node contracts |
| `misconfig.input_shape` | Input tensor/frame/sample shape or data type does not match the model contract | Provide the expected shape and data type, or configure model preprocessing |
| `misconfig.runtime_abi_mismatch` | Neat and a runtime plugin use incompatible ABIs | Install a version-matched Neat Library and runtime |
| `misconfig.graph_element_name` | A custom element cannot be assigned a stable Node name | Give custom elements stable, unique names |
| `misconfig.input_capacity` | Source image exceeds preprocessing input capacity | Increase `input_max_width` / `input_max_height`, or scale before the model stage |
| `misconfig.media_caps` | Adjacent GStreamer stages require incompatible media caps | Align format, resolution, and frame rate or insert conversion |
| `misconfig.media_format` | A stage received an unsupported media format | Configure a supported format or insert format conversion |
| `misconfig.tensor_dtype_missing` | Tensor contract has no dtype/format | Declare a supported tensor dtype in the upstream contract |
| `misconfig.option_out_of_range` | A stage option is invalid for the current tensor | Choose a value in the range shown by the diagnostic |
| `build.parse_launch` | A `gst_parse_launch` failure has no more specific classification | Inspect the attached report for the parser context |
| `build.pipeline_syntax` | Custom GStreamer fragment syntax is invalid | Correct and validate the fragment with `gst-launch-1.0` |
| `build.plugin_missing` | A required GStreamer element or codec plugin is not installed | Install/replace it and check with `gst-inspect-1.0` |
| `build.property_invalid` | An element property is unknown or invalid | Check the property name and value with `gst-inspect-1.0` |
| `runtime.pull` | A pull failed without a more specific root cause | Inspect the attached report and first upstream error |
| `runtime.element_failed` | A stage failed without a more specific mapping | Correct the reported stage and its upstream input |
| `runtime.output_timeout` | No output arrived before the configured timeout | Verify source flow or increase an expected timeout |
| `runtime.unexpected_eos` | The pipeline reached EOS before a required output | Check for premature source EOS and supply enough input |
| `io.parse` | JSON or stage-configuration parse/schema failure | Validate configuration syntax and required fields |
| `io.open` | Graph save/load file open/read/write failure | Check path existence, permissions, and storage health |
| `io.file_not_found` | Input file does not exist | Correct the path and confirm the file exists on the DevKit |
| `io.permission_denied` | File or device is not readable | Correct ownership/permissions |
| `io.rtsp_connection_failed` | RTSP source cannot be contacted | Verify URL, reachability, server, and credentials |
| `io.camera_not_found` | Requested camera is unavailable | Select a reported camera or use the default |
| `io.model_not_found` | Requested model archive does not exist | Correct the model path and confirm it is installed |
| `io.source_ended` | Input source reached its normal end | Stop consuming it or provide more input |
| `codec.invalid_h264_stream` | Input has no valid H.264 frames | Supply a complete H.264 stream or correct the codec |
| `codec.decode_failed` | Decoder failed after accepting the stream | Verify the codec and input integrity |
| `codec.encode_failed` | Encoder could not encode the supplied frames | Verify input format, resolution, and encoder settings |
| `resource.memory_allocation_failed` | A required memory allocation failed | Reduce workload memory use and free memory used by other applications or pipelines |
| `resource.device_memory_exhausted` | Device DMA/CMA allocation failed | Reduce concurrent streams, resolution, or buffering |
| `resource.output_pool_exhausted` | All output buffers remain in use | Release zero-copy outputs or use owned copies |
| `resource.buffer_too_small` | A buffer is smaller than its declared payload | Correct dimensions/stride or allocate the required bytes |
| `resource.disk_full` | A write failed because storage is full | Free space or choose another destination |
| `infra.dispatcher_unavailable` | Accelerator runtime cannot be acquired | Stop competing workloads and verify DevKit compatibility |
| `infra.accelerator_execution_failed` | Accelerator could not execute a model stage | Restart the pipeline and reduce concurrent accelerator work |
| `DispatcherUnavailable` | Legacy spelling of `infra.dispatcher_unavailable` | Migrate handlers to the canonical infrastructure code |
| `internal.plugin_failure` | A plugin failed without a user-actionable classification | Capture the report and contact support |

`PullError.code` uses the same taxonomy (not only exception paths).
See the [Error code catalog](/reference/error-codes) for the C++ and Python constant names and
migration guidance for applications that matched the previous coarse codes.

Production messages intentionally omit GStreamer internals. Plugin debug
verbosity adds the raw GError domain/code, element factory, message, and
structured plugin details. Recognized credentials and URL secret parameters—including URI
userinfo, `auth`, `playback-token`, `hdnts`, `stream-key`, and `tkn`—are redacted before either
form is stored. Report-facing pipeline strings, Node fragments, reproducer commands, and
serialized JSON are redacted without changing the executable pipeline held internally.

## Programmatic handling

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error) {
    if (perr.code == simaai::neat::error_codes::kMediaCaps) {
      // Fix the incompatible upstream/downstream media contract.
    } else {
      // Handle another specific code, including future codes, or report it.
    }
  }
} catch (const simaai::neat::NeatError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kPluginMissing) {
    // Install or replace the missing GStreamer component.
  }
}
```

## Debug knobs (environment)

Key environment variables (see [Architecture](/develop-apps/contribute/architecture) for detail):
- `SIMA_GST_DOT_DIR`: write DOT graphs for failures
- `SIMA_GST_BOUNDARY_PROBES`: boundary flow counters
- `SIMA_GST_ELEMENT_TIMINGS`: per-element timings
- `SIMA_GST_FLOW_DEBUG`: per-element flow counters
- `SIMA_GST_ENFORCE_NAMES`: enforce naming contract

To append redacted raw GStreamer context to `NeatError::what()` and
`GraphReport.repro_note`, set both variables for the failing command:

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` is not a Neat Library setting. Keep verbose output disabled in normal
operation; it is intended for short diagnostic runs and may contain deployment-specific paths or
media addresses even though recognized credential fields are redacted.

## Debug workflow

1) Capture `GraphReport.error_code` and bucket the failure by taxonomy first.
2) Capture `GraphReport.repro_note` for concrete context and built-in hint.
3) Capture pipeline text: `Graph::describe_backend()` or `last_pipeline()`.
4) Capture structured diagnostics: `MeasureReport::to_text()` or `NeatError::report()`.
5) Inspect `GraphReport.bus` for first terminal `ERROR` source + detail.
6) If runtime stalls/timeouts, enable boundary/element probes to localize flow stop.

Recommended support bundle:
- `error_code`
- `repro_note`
- full `pipeline_string`
- first 3-5 terminal bus errors (`GraphReport.bus`)
- environment overrides used in run/validate

## Customer graph performance artifact

For throughput/latency/power reporting, prefer the graph-run JSON export:

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

The export keeps scopes explicit:

- `run.graph_metrics.throughput_fps` and `run.graph_metrics.power` are graph-level headlines.
- `run.node_metrics[]` contains node/plugin latency only; node/plugin power is intentionally absent.
- `latency_semantics` and `aggregation` tell you whether values are run-lifetime or measured-window deltas.
- `plugin_metrics_unattributed[]` preserves kernel/plugin rows that could not be mapped to exactly one node.

For a measured window, use `Run::start_measurement()` and pass the returned `MeasureReport` to
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)`. Measured-window node
`min_ms`/`max_ms` are marked unavailable because cumulative min/max counters cannot be subtracted
exactly without window-local counters.

Power note: the current DVT board can validate option plumbing and JSON shape, but its wattage
readings are not treated as numerically reliable. SOM hardware is the intended platform for
power-number validation.

## Common failures → fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing ... plugin` | GStreamer plugin not found | Check `GST_PLUGIN_PATH`, run `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | Missing terminal `Output()` | Ensure `Output` is the last node in run/build pipelines |
| `caps_override is set; renegotiation disabled` | caps pinned | Remove `caps_override` or keep input caps fixed |
| `tensor caps change not supported` | Tensor shape/dtype change at runtime | Keep tensor shape/dtype stable (no renegotiation) |

For structured plugin errors and actionable hints, see
[Troubleshooting](/reference/troubleshooting).

## B1157 direct-driver recovery

On the Modalix 3.0.0 B1157 runtime, an unavailable dispatcher is an error to
investigate, not a request to start legacy services. Automatic recovery and
`fix_devkit_runtime.sh` refuse to restart AppComplex, run legacy MLA memory
initialization, activate EV74 firmware, or cycle remote processors. The refusal
is recorded in `GraphReport.repro_note`. Hard-reset and unsafe-reset environment
options do not override this boundary.

If a driver reports unknown completion, keep its DMA buffers and original pool
loans retained. Closing a file descriptor, stopping the application, or restarting
a service does not prove that hardware stopped accessing memory. Collect the
failure report and use the platform-approved recovery procedure before retrying.

Legacy recovery remains available only on a positively identified Modalix 2.1.x
image with no installed direct-runtime receipt. Missing, conflicting, or
unreadable identity fails closed. Do not remove receipts to bypass this check.

### Keep Core and Internals paired

Build and install Core with the matching B1157 Internals packages. Core checks
the runtime profile, kernel source revision, and SDK sysroot receipt separately
from the public C++ ABI version. An older package with the same public ABI is
not a compatible substitute.

The installer checks the bundled `neat-runtime` profile and matching
`neat-gst-plugins` version before changing packages. The platform-check override
does not bypass runtime pairing. If a check fails, obtain a matching bundle;
do not replace its receipt or force an older runtime into the installation.

For a board installation, first establish an exclusive maintenance window and
DMA quiescence using the platform procedure. Only then set
`NEAT_INSTALLER_B1157_MAINTENANCE=confirmed` in the **DevKit's installer process**.
The installer still checks for active accelerator owners and active or enabled
legacy services; it never stops those owners for you. The attestation is not a
reset command or a claim that an empty owner scan proves hardware retirement.
It is not automatically forwarded by SDK-to-DevKit deployment.
