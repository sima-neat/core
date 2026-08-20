---
title: Error Taxonomy Rollout
description: Structured error-code migration and verification checklist
sidebar_position: 2
slug: /develop-apps/contribute/error-taxonomy-rollout
---

# Error Taxonomy Rollout

This checklist tracks rollout of canonical error semantics across Core and runtime plugins.

## Canonical codes

[`include/pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h) is the
source of truth. The [error code catalog](/reference/error-codes) must document every C++ constant,
every Python `ERROR_*` name, and the migration from coarse to specific codes.

## Execution slices

1. Taxonomy scaffolding
2. Build/validate coding
3. Runtime pull coding
4. Graph IO parser/open coding
5. Tests + docs

## Compatibility review

- Treat a change in the exact code returned by an existing failure as a behavioral breaking
  change, even when no C++ or Python signature changes.
- Document old-to-new matches in the public migration table.
- Keep fallback codes (`build.parse_launch`, `runtime.pull`, and
  `runtime.element_failed`) only for failures without a specific classification.
- Test the versioned `simaai-neat-error` wire keys at production builders and parse a real
  `GstMessage` through Core.

## Verification checklist

- `NeatError.report().error_code` is non-empty on terminal framework failures.
- `PullError.code` is populated on runtime pull errors.
- Graph wrapper errors include code + context + hint (no generic fallback text).
- JSON parse failures include `offset=` and `near='...'`.
- Negative tests assert code + stable message fragments per taxonomy class.
- Diagnostics docs and architecture docs include triage flow: read `error_code`,
  inspect `repro_note`, inspect bus diagnostics, then replay with
  `repro_gst_launch`.
