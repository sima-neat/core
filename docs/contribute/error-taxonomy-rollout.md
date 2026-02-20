# Error Taxonomy Rollout

This checklist tracks framework-only rollout of canonical error semantics.

## Canonical codes

- `misconfig.pipeline_shape`
- `misconfig.caps`
- `misconfig.input_shape`
- `build.parse_launch`
- `runtime.pull`
- `io.parse`
- `io.open`

## Execution slices

1. Taxonomy scaffolding
2. Build/validate coding
3. Runtime pull coding
4. Session IO parser/open coding
5. Tests + docs

## Verification checklist

- `SessionError.report().error_code` is non-empty on terminal framework failures.
- `PullError.code` is populated on runtime pull errors.
- Session wrapper errors include code + context + hint (no generic fallback text).
- JSON parse failures include `offset=` and `near='...'`.
- Negative tests assert code + stable message fragments per taxonomy class.
- Diagnostics docs and architecture docs include triage flow:
  - read `error_code`
  - inspect `repro_note`
  - inspect bus diagnostics
  - replay with `repro_gst_launch`
