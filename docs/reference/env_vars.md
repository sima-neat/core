---
title: Environment Variables
description: Runtime and builder environment variables
sidebar_position: 1
---

# Environment variables

This page consolidates `SIMA_*` environment variables used by the runtime and
builder. Many are debug/diagnostic toggles; most users can ignore them unless
troubleshooting.

> Note: Some knobs are internal/test-only and may change. They are included here
> because they appear in code paths today.

## Core build/run

- `SIMA_PIPELINE_STRING_DEBUG=1` — print the final gst-launch string on build.
- `SIMA_PIPELINE_STATE_DEBUG=1` — extra state-change logs.
- `SIMA_PIPELINE_TEARDOWN_DEBUG=1` — logs pipeline teardown steps.
- `SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS=<ms>` — drain time before teardown (default 1500).
- `SIMA_PIPELINE_DRAIN_MIN_OUTPUTS=<n>` — minimum outputs to drain before teardown (default 1).
- `SIMA_PIPELINE_DOT_DIR=<dir>` — dump DOT graphs for internal errors (builder side).

## GStreamer init + suppression

- `SIMA_ALLOW_GST_INIT=1` — allow manual `gst_init` if already initialized.
- `SIMA_GST_SUPPRESS_JSON_WARNINGS=0/1` — silence JSON warnings (default true).
- `SIMA_GST_SUPPRESS_GOBJECT_ASSERTS=0/1` — silence GLib assert logs (default true).
- `SIMA_GST_SUPPRESS_DEVICE_LOGS=0/1` — silence device logs (default true).

## GStreamer timeouts

- `SIMA_STATE_CHANGE_TIMEOUT_MS=<ms>` — pipeline state change timeout (default 15000).
- `SIMA_GST_TEARDOWN_TIMEOUT_MS=<ms>` — teardown timeout (default 2000).
- `SIMA_GST_TEARDOWN_REAPER_MS=<ms>` — teardown watchdog (default 250).
- `SIMA_GST_TEARDOWN_ASYNC=1` — async teardown.
- `SIMA_GST_POLL_SLICE_MS=<ms>` — poll slice for appsink pulls (default 200).
- `SIMA_GST_VALIDATE_TIMEOUT_MS=<ms>` — validate() timeout (default 2000/10000).
- `SIMA_GST_RUN_INPUT_TIMEOUT_MS=<ms>` — run() input timeout (default 10000).

## Diagnostics + probes

- `SIMA_GST_DOT_DIR=<dir>` — dump DOT graphs for pipeline failures/debug.
- `SIMA_GST_BOUNDARY_PROBES=1` — attach boundary flow probes.
- `SIMA_GST_STAGE_TIMINGS=1` — stage timing probes.
- `SIMA_GST_ELEMENT_TIMINGS=1` — element timing probes.
- `SIMA_GST_FLOW_DEBUG=1` — element flow probes.
- `SIMA_GST_ENFORCE_NAMES=1` — enforce name contract on build.
- `SIMA_GST_OPTIONS_DEBUG=1` — log GStreamer options during build.
- `SIMA_GST_BUFFER_DEBUG_LIMIT=<n>` — cap buffer debug prints.
- `SIMA_GST_DETESS_INPUT_DEBUG=1` — detess input debug.
- `SIMA_GST_DETESS_OUTPUT_DEBUG=1` — detess output debug.
- `SIMA_GST_DETESS_POOL_DEBUG=1` — detess pool debug.
- `SIMA_GST_APPSINK_BUFFER_DEBUG=1` — appsink buffer debug.
- `SIMA_GST_ALL_BUFFER_DEBUG=1` — verbose buffer debug.
- `SIMA_GST_RUN_INSERT_BOUNDARIES=1` — insert boundaries during run().
- `SIMA_GST_VALIDATE_INSERT_BOUNDARIES=1` — insert boundaries during validate().

## Dispatcher / runtime

- `SIMA_DISPATCHER_TRACE=1` — trace dispatcher steps.
- `SIMA_DISPATCHER_AUTO_RECOVER=0/1` — auto-recover dispatcher (default true).
- `SIMA_ASYNC_TPUT_DIAG=1` — async throughput diagnostics.
- `SIMA_ASYNC_WARMUP=<n>` — async warmup frames.
- `SIMA_PULL_TIMEOUT_DIAG=0/1` — report on pull timeouts (default true).
- `SIMA_RUN_INPUT_TIMINGS=1` — enable input timing stats in run().
- `SIMA_STAGE_DEBUG=1` — StageRun debug logs.

## InputStream / Sample debugging

- `SIMA_INPUTSTREAM_DEBUG=1` — verbose InputStream logs.
- `SIMA_INPUTSTREAM_WARN=1` — warnings on InputStream events.
- `SIMA_INPUTSTREAM_POLL_MS=<ms>` — InputStream poll slice (default 50).
- `SIMA_INPUTSTREAM_DOT_ON_TIMEOUT=1` — dump DOT on timeout.
- `SIMA_INPUTSTREAM_META_DEBUG=1` — log GstSimaMeta details.
- `SIMA_INPUTSTREAM_ALLOC_DEBUG=1` — allocation debug.
- `SIMA_INPUTSTREAM_PUSH_TIMING=1` — push timing logs.
- `SIMA_INPUTSTREAM_PREFLIGHT_RUN=1` — preflight run for InputStream.
- `SIMA_SAMPLE_DEBUG=1` — log sample conversions.
- `SIMA_SAMPLE_BYTES=1` — log sample byte sizes.
- `SIMA_SAMPLE_FORCE_BUNDLE=1` — force bundle output for debugging.
- `SIMA_NEAT_CAPS_TRACE=1` — trace Tensor cap derivation.

## Preproc / Detess / wiring

- `SIMA_PREPROC_DEBUG_CONFIG=1` — dump preproc config wiring.
- `SIMA_DETESS_MULTI_BUFFER=1` — deprecated (warns if set).
- `SIMA_KEEP_DETESS_CONFIG=1` — keep detess config outputs.
- `SIMA_DETESS_ASSERT_ON_ZERO=1` — assert on zero detess output.
- `SIMA_MLA_CONFIG_DEBUG=1` — MLA plugin config debug.
- `SIMA_STRICT_CONFIG_WIRING=1` — deprecated no-op (legacy JSON wiring checks removed).
- `SIMA_WIRE_BY_ORDER_DEBUG=1` — deprecated no-op (legacy JSON wiring rewrite removed).
- `SIMA_BOXDECODE_WIRE_DEBUG=1` — deprecated no-op (legacy JSON wiring rewrite removed).
- `SIMA_CLAMP_DETESS_NUM_BUFFERS=1` — clamp detess num-buffers.
- `SIMA_FORCE_SYNC_NUMBUFFERS_ONE=1` — force num-buffers=1 in sync mode.
- `SIMA_DISABLE_SYNC_NUMBUFFERS_CVU_MLA=1` — disable sync num-buffers clamps.

## Model (legacy env var names retained)
- `SIMA_NEATMODEL_USE_MLA=1` — force the Model MLA path (legacy env var name).
- `SIMA_MLA_NEXT_CPU=<domain>` — override MLA next_cpu.

## RTSP / H264

- `SIMA_H264_SDP_DUMP=<path>` — dump H264 SDP to file.
- `SIMA_H264_SPS_FIXUP_STREAM=<path>` — fix up SPS in stream.

## OutputSpec

- `SIMA_DEBUG_OUTPUTSPEC_LOG=1` — log OutputSpec propagation details.

## Test / internal hooks

- `SIMA_GUARD_TEST_HOLD_MS=<ms>` — internal sleep hook during build.
- `SIMA_TENSOR_MAPFAIL_DEBUG=1` — log Tensor map failures.
