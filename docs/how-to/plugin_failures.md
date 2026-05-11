# How To Debug Plugin Failures

When a plugin fails, Neat raises a `SessionError` whose message contains the
GStreamer error and a structured debug string. Use the fields to locate the
root cause quickly.

## 1) Read the Structured Fields

Look for the `debug` key/value fields in the error text:

- `node`: the failing element name in the pipeline
- `config_path`: JSON config file (if applicable)
- `model_path`: model/pack path (if applicable)
- `hint`: actionable fix guidance
- `detail`: extra context such as missing keys or allocator state

See the [Error Format Reference](/reference/error_format) for the full list.

## 2) Confirm the Pipeline Context

Use the pipeline string from `Session::last_pipeline()` or from the error report:

- Verify the `node` name appears in the pipeline.
- Confirm the `config_path` exists and is readable.
- For caps errors, check upstream elements that negotiate into the failing node.

## 3) Common Fixes

- **Config errors**: verify JSON syntax, required keys, and any model paths.
- **Caps errors**: add or fix parser elements (e.g., `h264parse`), ensure caps
  include required fields like `parsed=true`, `stream-format=byte-stream`,
  `alignment=au`.
- **Allocator errors**: ensure upstream elements use the required allocator
  type (system vs. simaai memory/segment).

## 4) Capture More Diagnostics

Enable diagnostics to gather more context:

- `SIMA_GST_DOT_DIR` for DOT graph dumps
- `SIMA_GST_FLOW_DEBUG` for buffer flow stats
- `SIMA_GST_ELEMENT_TIMINGS` for per‑element timings

These tools are summarized in [Diagnostics and Debugging](/how-to/diagnostics).
