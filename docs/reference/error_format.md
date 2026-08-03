---
title: Plugin Error Format
description: Structured error fields from plugin failures
sidebar_position: 8
---

# Plugin Error Format

When a plugin encounters a fatal condition, it posts a `GST_MESSAGE_ERROR` on the
GStreamer bus. Neat promotes the error to `NeatError` and preserves supported structured details
for classification and rendering.

## Error Domains and Codes

These are the recommended domains/codes used across plugins:

- Config parsing/validation: `GST_RESOURCE_ERROR_SETTINGS`
- File missing: `GST_RESOURCE_ERROR_NOT_FOUND`
- Dispatcher unavailable: `GST_RESOURCE_ERROR_BUSY`; use
  `GST_RESOURCE_ERROR_NOT_FOUND` only with a dispatcher-specific diagnostic ID
  or structured dispatcher field
- Allocation failures: `GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- Caps/negotiation errors: `GST_STREAM_ERROR_FORMAT`
- Runtime processing failures: `GST_STREAM_ERROR_FAILED`

## Versioned structured details

New Neat plugin errors attach a `GstStructure` named `simaai-neat-error`. Version 1 contains the
unsigned integer field `neat-schema-version=1`. Core reads structured fields from version 1 and
falls back to the ordinary GStreamer domain, code, message, and debug string for an unknown or
missing version. This prevents a future schema from being interpreted with old assumptions.

Common fields:
- `neat-schema-version`
- `neat-diagnostic-id`
- `neat-reason`
- `plugin`
- `node`
- `stage`
- `graph-id`
- `frame-id`
- `stream-id`
- `input-caps`
- `output-caps`
- `allocator`
- `dispatcher-error`

Input-capacity errors also provide `actual-width`, `actual-height`, `actual-stride`,
`maximum-width`, `maximum-height`, `maximum-stride`, `resize-width`, `resize-height`,
`required-bytes`, `allocated-bytes`, and `input-format`.

Input-contract errors also provide `input-name`, `segment-name`, `required-bytes`, `actual-bytes`,
`expected-shape`, `expected-layout`, `expected-dtype`, `received-shape`, `received-layout`, and
`received-dtype`. The layout fields disambiguate shapes such as `[3, 224, 224]` (`CHW`) and
`[224, 224, 3]` (`HWC`).

Older plugins may place a space-separated `key='value'` list in the debug string. Core continues
to use those fields as a compatibility fallback.

## Example

```text
simaai-neat-error, neat-schema-version=(uint)1,
neat-diagnostic-id=(string)neatprocesscvu.input_contract_mismatch,
plugin=(string)neatprocesscvu, node=(string)model_0,
expected-shape=(string)"[3, 224, 224]", expected-layout=(string)CHW,
expected-dtype=(string)Float32, received-shape=(string)"[224, 224, 3]",
received-layout=(string)HWC, received-dtype=(string)UInt8;
```

## Notes

- By default, `NeatError::what()` contains the normalized error code, user-facing
  context, corrective actions, and diagnostic ID. It omits the raw GStreamer
  message and debug string.
- Set `SIMA_NEAT_VERBOSE_LEVEL=2` and `SIMA_NEAT_VERBOSE_TOPICS=gstreamer` for a short
  diagnostic run. This appends redacted technical details to
  `NeatError::what()` and `GraphReport.repro_note`. `NeatError::report()` remains
  the structured interface for diagnostics.
- `NEAT_LOG_LEVEL=debug` is not a Neat Library setting.
- URI userinfo and recognized credential fields—including `auth`, `playback-token`, `hdnts`,
  `stream-key`, and `tkn`—are redacted in report-facing pipeline strings, reproducer commands,
  structured details, and JSON. Review a support bundle for deployment-specific paths and media
  addresses before sharing it.
