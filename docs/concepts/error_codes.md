---
title: Error code catalog
description: The framework's `error_codes::*` constants — when each is raised, what to do about it.
sidebar_position: 11
---

# Error code catalog

The framework raises `SessionError` for every recoverable failure. Each error carries:

- A **structured error code** (a string like `"io.parse"` or `"plan.unsupported"`).
- A human-readable message.
- An attached `SessionReport` with build-time context.

The error code is the triage hook. Switch on it programmatically; let the message be for the human.

## Code families

Codes are grouped by the layer that raises them. The full set of constants lives in [`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).

### `io.*` — I/O and parse failures

| Code | When raised | What to do |
|---|---|---|
| `io.not_found` | A required file (MPK, config, asset) is missing. | Check the path; check working directory at `Session::build()` time. |
| `io.parse` | A file was found but couldn't be parsed (MPK manifest, config JSON). | Validate the MPK with `MpKLoader::inspect()` first. |
| `io.size_limit` | An MPK or entry exceeds the configured `MpKLoaderOptions` cap. | Either the model is too large (raise the cap) or the archive is corrupt. |
| `io.path_traversal` | An archive entry's path would escape the extraction root. | Reject the pack — it's malicious or corrupt. |

### `mpk.*` — model-pack contract

| Code | When raised | What to do |
|---|---|---|
| `mpk.schema` | Manifest is structurally valid JSON but doesn't match the schema. | Re-export from the model-compiler with a current schema version. |
| `mpk.unsupported_version` | Manifest declares a version this loader doesn't understand. | Update the framework or downgrade the model. |
| `mpk.unsupported_kernel` | Manifest references a kernel binary not in the framework's allowlist. | The model uses an op the runtime can't load — recompile the model. |
| `mpk.missing_section` | A required manifest section (`pipeline_sequence`, model binary) is absent. | Re-export the model pack with the missing artifacts. |

### `plan.*` — planner / route-graph failures

| Code | When raised | What to do |
|---|---|---|
| `plan.unsupported` | The planner can't construct a route for the requested ops. | Check that your transforms aren't asking for something the framework doesn't support. |
| `plan.contract_mismatch` | An MPK contract conflicts with the requested `PreprocessOptions`. | Reconcile your options against the MPK manifest. |
| `plan.no_kernel` | No CVU/MLA kernel exists for a stage in the chosen route. | Likely a missing kernel binary in the build. |

### `caps.*` — caps negotiation

| Code | When raised | What to do |
|---|---|---|
| `caps.mismatch` | Two adjacent Nodes negotiate incompatible caps. | Inspect with `Session::describe()`; insert an explicit converter Node. |
| `caps.no_format` | A `Source` Node can't determine its output caps. | Specify caps explicitly on the source. |

### `runtime.*` — live runtime

| Code | When raised | What to do |
|---|---|---|
| `runtime.timeout` | A `pull()` / `push()` exceeded its timeout. | Check back-pressure with `RunDiagSnapshot::boundary_flow_stats`. |
| `runtime.gst_error` | GStreamer posted a `GST_MESSAGE_ERROR`. | Read the wrapped debug string ([Plugin error format](/reference/error_format)). |
| `runtime.eos` | Pipeline reached EOS before all expected outputs arrived. | Check the input source for premature EOS; verify there's enough input data. |

## How to consume

```cpp
try {
  sess.run();
} catch (const sima::SessionError& e) {
  if (e.error_code() == sima::error_codes::caps_mismatch) {
    // Switch on the code, not on `e.what()`.
    handle_caps_mismatch(e.report());
  } else {
    throw;
  }
}
```

The `SessionReport` attached to the error names the failing Node and the build phase, so you don't have to grep `e.what()` for context.

## Further reading

- [Plugin error format](/reference/error_format) — how GStreamer-side plugin errors are encoded into the debug string.
- [`SessionError`](/reference/cppapi/classes/simaai-neat-sessionerror) — the exception type.
- [`SessionReport`](/reference/cppapi/structs/simaai-neat-sessionreport) — the structured context attached to every error.
- "Validation, SessionReport" — §29 and §41 of the design deep dive.
