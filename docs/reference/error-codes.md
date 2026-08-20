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

## Behavioral breaking change and migration

The diagnostic taxonomy now preserves specific GStreamer root causes. Public method signatures are
unchanged, but code that compares exact error strings may need to migrate:

| Previous match | More specific code now returned | Migration |
| --- | --- | --- |
| `misconfig.caps` for a runtime GStreamer negotiation error | `misconfig.media_caps`, or `misconfig.media_format` when only the format is incompatible | Handle the media code. Keep `misconfig.caps` only for framework validation of caps overrides and adjacent Node contracts. |
| `build.parse_launch` for every `gst_parse_launch` failure | `build.plugin_missing`, `build.property_invalid`, or `build.pipeline_syntax` | Handle the specific build codes. Keep `build.parse_launch` as the fallback for an unclassified parser failure. |
| `runtime.pull` for a propagated bus failure | The root-cause code, such as `misconfig.media_caps`, `io.rtsp_connection_failed`, or `resource.output_pool_exhausted` | Handle the root-cause codes and keep a default branch. `runtime.pull` remains the fallback for a local pull failure with no specific cause. |

Use the C++ or Python constants rather than repeating string literals. Always keep a default path
for codes introduced by a newer Neat Library build.

## Public constants

The same values are available in both language APIs:

| Error code | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.runtime_abi_mismatch` | `error_codes::kRuntimeAbiMismatch` | `pyneat.ERROR_RUNTIME_ABI_MISMATCH` |
| `misconfig.graph_element_name` | `error_codes::kGraphElementName` | `pyneat.ERROR_GRAPH_ELEMENT_NAME` |
| `misconfig.media_caps` | `error_codes::kMediaCaps` | `pyneat.ERROR_MEDIA_CAPS` |
| `misconfig.media_format` | `error_codes::kMediaFormat` | `pyneat.ERROR_MEDIA_FORMAT` |
| `misconfig.input_capacity` | `error_codes::kInputCapacity` | `pyneat.ERROR_INPUT_CAPACITY` |
| `misconfig.tensor_dtype_missing` | `error_codes::kTensorDtypeMissing` | `pyneat.ERROR_TENSOR_DTYPE_MISSING` |
| `misconfig.option_out_of_range` | `error_codes::kOptionOutOfRange` | `pyneat.ERROR_OPTION_OUT_OF_RANGE` |
| `build.parse_launch` | `error_codes::kParseLaunch` | `pyneat.ERROR_PARSE_LAUNCH` |
| `build.pipeline_syntax` | `error_codes::kPipelineSyntax` | `pyneat.ERROR_PIPELINE_SYNTAX` |
| `build.plugin_missing` | `error_codes::kPluginMissing` | `pyneat.ERROR_PLUGIN_MISSING` |
| `build.property_invalid` | `error_codes::kPropertyInvalid` | `pyneat.ERROR_PROPERTY_INVALID` |
| `runtime.pull` | `error_codes::kRuntimePull` | `pyneat.ERROR_RUNTIME_PULL` |
| `runtime.element_failed` | `error_codes::kRuntimeElementFailed` | `pyneat.ERROR_RUNTIME_ELEMENT_FAILED` |
| `runtime.output_timeout` | `error_codes::kOutputTimeout` | `pyneat.ERROR_OUTPUT_TIMEOUT` |
| `runtime.unexpected_eos` | `error_codes::kUnexpectedEos` | `pyneat.ERROR_UNEXPECTED_EOS` |
| `io.parse` | `error_codes::kIoParse` | `pyneat.ERROR_IO_PARSE` |
| `io.open` | `error_codes::kIoOpen` | `pyneat.ERROR_IO_OPEN` |
| `io.file_not_found` | `error_codes::kFileNotFound` | `pyneat.ERROR_FILE_NOT_FOUND` |
| `io.permission_denied` | `error_codes::kPermissionDenied` | `pyneat.ERROR_PERMISSION_DENIED` |
| `io.rtsp_connection_failed` | `error_codes::kRtspConnectionFailed` | `pyneat.ERROR_RTSP_CONNECTION_FAILED` |
| `io.camera_not_found` | `error_codes::kCameraNotFound` | `pyneat.ERROR_CAMERA_NOT_FOUND` |
| `io.model_not_found` | `error_codes::kModelNotFound` | `pyneat.ERROR_MODEL_NOT_FOUND` |
| `io.source_ended` | `error_codes::kSourceEnded` | `pyneat.ERROR_SOURCE_ENDED` |
| `codec.invalid_h264_stream` | `error_codes::kInvalidH264Stream` | `pyneat.ERROR_INVALID_H264_STREAM` |
| `codec.decode_failed` | `error_codes::kDecodeFailed` | `pyneat.ERROR_DECODE_FAILED` |
| `codec.encode_failed` | `error_codes::kEncodeFailed` | `pyneat.ERROR_ENCODE_FAILED` |
| `resource.memory_allocation_failed` | `error_codes::kMemoryAllocationFailed` | `pyneat.ERROR_MEMORY_ALLOCATION_FAILED` |
| `resource.device_memory_exhausted` | `error_codes::kDeviceMemoryExhausted` | `pyneat.ERROR_DEVICE_MEMORY_EXHAUSTED` |
| `resource.output_pool_exhausted` | `error_codes::kOutputPoolExhausted` | `pyneat.ERROR_OUTPUT_POOL_EXHAUSTED` |
| `resource.buffer_too_small` | `error_codes::kBufferTooSmall` | `pyneat.ERROR_BUFFER_TOO_SMALL` |
| `resource.disk_full` | `error_codes::kDiskFull` | `pyneat.ERROR_DISK_FULL` |
| `infra.dispatcher_unavailable` | `error_codes::kDispatcherUnavailable` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE` |
| `infra.accelerator_execution_failed` | `error_codes::kAcceleratorExecutionFailed` | `pyneat.ERROR_ACCELERATOR_EXECUTION_FAILED` |
| `DispatcherUnavailable` (legacy) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## Misconfiguration

| Code | When raised | What to do |
| --- | --- | --- |
| `misconfig.pipeline_shape` | The graph has an invalid topology or missing input/output boundary. | Correct the graph connections and required `Input` or `Output` Nodes. |
| `misconfig.caps` | A caps override or adjacent Node contract is incompatible during framework validation. | Align the declared format, dimensions, rate, and adjacent Node contract. |
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

## Pipeline materialization failures

| Code | When raised | What to do |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Pipeline topology is invalid, or final element names are duplicate, ambiguous, or missing after GStreamer construction. | Give every explicit element a unique short name within its materialized segment. Keep `name=` declarations and named-pad references synchronized. |
| `build.parse_launch` | GStreamer cannot parse or construct the final launch string because syntax, a plugin, or a property is invalid. | Inspect `GraphReport::pipeline_string`; verify the fragment with `gst-launch-1.0` and the plugin with `gst-inspect-1.0`. |

These checks are automatic during `Graph::build()`. For input-dependent connected segments, the
same code and `GraphReport` can surface when the first input materializes the segment.

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
