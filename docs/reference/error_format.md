---
title: Plugin Error Format
description: Structured error fields from plugin failures
sidebar_position: 3
---

# Plugin Error Format

When a plugin encounters a fatal condition it posts a `GST_MESSAGE_ERROR` on the
GStreamer bus. NEAT promotes these to `SessionError` exceptions. The error
`debug` string embeds structured fields to make failures actionable.

## Error Domains and Codes

These are the recommended domains/codes used across plugins:

- Config parsing/validation: `GST_RESOURCE_ERROR_SETTINGS`
- File missing: `GST_RESOURCE_ERROR_NOT_FOUND`
- Dispatcher unavailable: `GST_RESOURCE_ERROR_BUSY` (or `GST_RESOURCE_ERROR_NOT_FOUND`)
- Allocation failures: `GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- Caps/negotiation errors: `GST_STREAM_ERROR_FORMAT`
- Runtime processing failures: `GST_STREAM_ERROR_FAILED`

## Structured Fields (Debug String)

The debug string is a space‑separated `key='value'` list. Not all fields are
present for every plugin.

Common fields:
- `plugin`
- `node`
- `config_path`
- `model_path`
- `graph_id`
- `frame_id`
- `stream_id`
- `input_caps`
- `output_caps`
- `input_dims`
- `output_dims`
- `allocator`
- `dispatcher_err`
- `hint`
- `detail`

## Example

```
GST ERROR: Error parsing config | plugin='simaaiboxdecode' node='box_fail'
config_path='/tmp/boxdecode.json' input_dims='640x480' allocator='2'
hint='Config parsing failed; check config_path and JSON syntax'
detail='caps_parse_failed=true'
```

## Notes

- `hint` is intended to be human‑actionable.
- `detail` captures machine‑oriented context for debugging.
- `SessionError::what()` includes both the GStreamer message and the debug string.
