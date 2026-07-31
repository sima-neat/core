---
title: Error code catalog
description: Stable framework error codes, when they occur, and how to respond
sidebar_position: 7
---

# Error code catalog

Neat surfaces typed failures through `NeatError` and `PullError`. Each failure provides a stable
error code, a human-readable message, and—when available—a `GraphReport` with structured context.

Use the error code for programmatic triage. Show the message to the developer. The complete set of
public constants lives in
[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).

## Misconfiguration

| Code | When raised | What to do |
| --- | --- | --- |
| `misconfig.pipeline_shape` | The graph has an invalid topology or missing input/output boundary. | Correct the graph connections and required `Input` or `Output` Nodes. |
| `misconfig.caps` | A caps override or adjacent Node contract is incompatible. | Align the declared format, dimensions, rate, and downstream caps. |
| `misconfig.input_shape` | An input tensor does not match the expected shape or data type. | Provide the expected input, or configure model preprocessing through model options. |
| `misconfig.runtime_abi_mismatch` | Neat and an installed runtime plugin use incompatible ABIs. | Install matching Neat Library and runtime-plugin builds. |
| `misconfig.graph_element_name` | A custom fragment contains an element that cannot be assigned a stable Node name. | Give custom elements stable, unique names. |
| `misconfig.media_caps` | Connected GStreamer stages require incompatible media caps. | Align the stages or insert the required conversion, scaling, or rate-conversion Node. |
| `misconfig.media_format` | Connected stages require incompatible media formats. | Configure a common format or add an explicit format conversion. |
| `misconfig.input_capacity` | A source image exceeds the configured preprocessing input capacity. | Increase `input_max_width` and `input_max_height`, or scale the source before the model stage. |
| `misconfig.tensor_dtype_missing` | A tensor contract omits its data type or format. | Declare a supported data type in the upstream tensor contract. |
| `misconfig.option_out_of_range` | An option is invalid for the current input contract. | Set the option to a value in the range shown by the diagnostic. |

## Build failures

| Code | When raised | What to do |
| --- | --- | --- |
| `build.parse_launch` | GStreamer cannot build the generated pipeline. | Check the custom fragment, element properties, and plugin availability. |
| `build.pipeline_syntax` | A custom GStreamer fragment has invalid syntax. | Correct the fragment and validate it with `gst-launch-1.0`. |
| `build.plugin_missing` | A required GStreamer element or codec plugin is unavailable. | Install or replace the component, then verify it with `gst-inspect-1.0`. |
| `build.property_invalid` | An element property name or value is invalid. | Check the property with `gst-inspect-1.0 <element>`. |

## Runtime failures

| Code | When raised | What to do |
| --- | --- | --- |
| `runtime.pull` | A pull operation fails without a more specific code. | Inspect the attached report and the first upstream error. |
| `runtime.element_failed` | A pipeline stage stops without a more specific classification. | Correct the reported stage configuration and its upstream input. |
| `runtime.output_timeout` | No output arrives before the configured wait expires. | Verify source flow and back-pressure, or adjust the timeout when the wait is expected. |
| `runtime.unexpected_eos` | The pipeline reaches EOS before producing a required output. | Check the input for premature EOS and confirm that enough input was supplied. |

## I/O failures

| Code | When raised | What to do |
| --- | --- | --- |
| `io.parse` | Neat cannot parse JSON, a model contract, or stage configuration. | Validate the configuration syntax, schema, and required fields. |
| `io.open` | Neat cannot open a file, device, or remote resource. | Verify the path or address, permissions, and resource availability. |
| `io.file_not_found` | An input file does not exist. | Correct the path and confirm that the file exists on the DevKit. |
| `io.permission_denied` | A file or device cannot be opened with the required access. | Correct ownership or permissions for the reported resource. |
| `io.rtsp_connection_failed` | Neat cannot connect to an RTSP source. | Verify the URL, server, network reachability, and credentials. |
| `io.camera_not_found` | The requested camera is unavailable. | Select an available camera or use the default camera. |
| `io.model_not_found` | The requested model archive does not exist. | Correct the model path and confirm that the archive is installed. |
| `io.source_ended` | An input source reaches its normal end. | Stop consuming that source or provide additional input if the application expects more data. |

## Codec failures

| Code | When raised | What to do |
| --- | --- | --- |
| `codec.invalid_h264_stream` | The input contains no valid H.264 frames. | Supply a complete H.264 stream and confirm the configured codec. |
| `codec.decode_failed` | A decoder cannot decode the accepted stream. | Confirm the codec and check that the encoded input is complete and uncorrupted. |
| `codec.encode_failed` | An encoder cannot encode the supplied frames. | Verify the input format, resolution, and encoder settings. |

## Resource failures

| Code | When raised | What to do |
| --- | --- | --- |
| `resource.memory_allocation_failed` | A required memory allocation fails without a device-specific cause. | Reduce stream count, resolution, or buffering, and free memory used by other workloads. |
| `resource.device_memory_exhausted` | Contiguous device DMA/CMA memory is exhausted. | Reduce concurrent streams, input resolution, or buffer depth. |
| `resource.output_pool_exhausted` | All output buffers remain in use. | Release zero-copy outputs promptly or use owned copies. |
| `resource.buffer_too_small` | A buffer is smaller than its declared frame or tensor payload. | Correct upstream dimensions and stride, or allocate the required number of bytes. |
| `resource.disk_full` | A write fails because the destination has insufficient free space. | Free space or choose another destination. |

## Infrastructure failures

| Code | When raised | What to do |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat cannot acquire the accelerator runtime. | Confirm DevKit compatibility and stop workloads that exclusively own the accelerator. |
| `infra.accelerator_execution_failed` | The accelerator cannot execute a model stage. | Restart the pipeline and reduce concurrent accelerator workloads. |

## Internal failures

| Code | When raised | What to do |
| --- | --- | --- |
| `internal.plugin_failure` | A Neat plugin fails without a user-actionable classification. | Capture the attached `GraphReport` and report the failure to support. |

`DispatcherUnavailable` is a legacy spelling accepted for compatibility. New applications should
use `infra.dispatcher_unavailable` and the `error_codes::kDispatcherUnavailable` constant.

## Handle errors programmatically

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build();
  // Push and pull application data.
} catch (const simaai::neat::NeatError& error) {
  if (error.report().error_code == simaai::neat::error_codes::kInputShape) {
    handle_input_contract_error(error.report());
  } else {
    throw;
  }
}
```

`PullError.code` uses the same constants. Do not parse `what()` or match human-readable text.

## Further reading

- [Diagnostics and debugging](/reference/diagnostics) — production messages, debug details, and
  `GraphReport` collection.
- [Plugin error format](/reference/error_format) — the structured contract for GStreamer plugin
  errors.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — the typed exception.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — structured error context.
