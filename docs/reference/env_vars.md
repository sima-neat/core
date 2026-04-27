---
title: Environment Variables
description: User-settable environment variables for build, runtime, examples, tests, and docs
sidebar_position: 1
---

# Environment variables

This page documents the environment variables that `core/` reads directly in
build scripts, runtime code, examples, tests, and docs helpers.

The list is split into:

- `Currently used` variables that are reasonable to set today.
- `Deprecated or legacy` variables that still appear in compatibility code
  paths, old debug flows, or transition behavior.

`/data/workspace/sima-neat/core/.env` is your local sourceable file. Keep that
file short and use this page as the detailed explanation of what each variable
actually does.

## Currently used

### Build, packaging, and SDK

- `CMAKE_BUILD_PARALLEL_LEVEL`
  Controls build parallelism for `build.sh` and repo scripts that call
  `cmake --build`. If you set `CMAKE_BUILD_PARALLEL_LEVEL=16`, those commands
  effectively build with `-j16`. If unset, `build.sh` falls back to CPU
  detection and then to `8`.

- `SIMANEAT_STRICT_WARNINGS`
  Passed through to CMake as `-DSIMANEAT_STRICT_WARNINGS=...`. Use `ON` when
  you want stricter local compiler-warning enforcement and `OFF` for normal
  development.

- `NEAT_INTERNALS_MANIFEST`
  Path to the manifest file used by `build.sh --install-deps`.
  Override it only when you want to test against a different manifest than the
  default `deps/manifest.json`.

- `NEAT_INTERNALS_BASE_URL`
  Base URL used to download `deps` artifacts and checksums. Override
  it for staging, mirrors, or private artifact servers.

- `NEAT_INTERNALS_DIR`
  Local directory where `build.sh` stages downloaded `deps` content.
  Change it only if you want those artifacts somewhere other than the default
  `deps/`.

- `NEAT_INTERNALS_BASIC_AUTH`
  Optional `user:password` credential pair used when `build.sh` fetches
  protected `deps` artifacts. Leave it unset for public endpoints.

- `ELXR_SDK_RELEASE_FILE`
  File that `build.sh` checks to detect whether it is running inside a Neat SDK
  environment. Override it only if your SDK metadata is not at `/etc/sdk-release`.

- `ELXR_INIT_SCRIPT`
  SDK environment activation script sourced by `build.sh` in eLxr builds.
  Override it only if your SDK install uses a different path.

- `ELXR_MACHINE`
  Machine profile passed to the eLxr init script. Keep `modalix` unless you are
  targeting a different SDK machine target.

- `ELXR_WHEEL_HOST_PLATFORM`
  Explicit wheel-platform override for Python packaging in eLxr builds. If
  unset, `build.sh` derives the value from SDK architecture hints.

- `DOXYGEN2DOCUSAURUS_CMD`
  Command used by `tools/generate_api_docs.sh` for the Doxygen-to-Docusaurus
  conversion step. The default pins both `@xpack/doxygen2docusaurus@2.0.0`
  and `fast-xml-parser@5.2.5` to keep docs builds reproducible across
  containers. Override it only if you need a custom wrapper or tool path.

- `DOCS_STRICT_LINKS`
  Contributor-facing flag for strict docs builds. Use `DOCS_STRICT_LINKS=1`
  when you want the website build to fail on broken links instead of tolerating
  them.

### GStreamer plugin discovery and startup

- `SIMA_GST_PLUGIN_DIR`
  Overrides the Neat plugin directory that runtime startup uses as its
  preferred third-party plugin root. Set this when you want `Session` and
  `gst_init_once()` to load plugins from a custom install instead of the
  default system path.

- `GST_PLUGIN_PATH_1_0` and `GST_PLUGIN_PATH`
  Standard GStreamer plugin search paths. Neat startup can set or prepend these
  automatically, but you can manage them yourself if you are controlling plugin
  discovery outside the framework.

- `GST_PLUGIN_SYSTEM_PATH_1_0`
  Standard GStreamer system plugin path. Neat treats a pre-set value as risky
  because it can bypass duplicate-plugin filtering; use it only when you
  intentionally want unrestricted system plugin discovery.

- `GST_PLUGIN_SCANNER`
  Standard GStreamer scanner executable override. Neat can wrap this scanner to
  force the right library path for plugin probing; set it only when you need to
  pin a specific scanner binary.

- `GST_REGISTRY_1_0`
  Standard GStreamer registry file path. If unset, Neat creates a per-process
  registry under `/tmp`. Set it when you want deterministic registry reuse or
  easier plugin-discovery debugging.

- `LD_LIBRARY_PATH`
  Standard dynamic library search path. Neat may prepend the third-party plugin
  directory so plugin-side shared libraries resolve correctly. You usually only
  need to set it yourself for custom install layouts.

- `SIMA_SET_GST_SYSTEM_PATH`
  Read by `scripts/use_neatdecoder.sh`. When set, that helper clears
  `GST_PLUGIN_SYSTEM_PATH_1_0` so the current shell sees only the intended Neat
  plugin location.

- `SIMA_ALLOW_GST_INIT`
  Allows the process to continue even if GStreamer was initialized before
  `simaai::neat::gst_init_once()`. Leave it unset unless you intentionally want
  to bypass the startup guard.

- `SIMA_GST_NEAT_ONLY`
  Controls whether startup forces a strict Neat-only plugin view. The effective
  default is `1`, which favors the Neat plugin directory and filtered system
  plugins.

- `SIMA_GST_ALLOW_SYSTEM_PLUGINS`
  Disables Neat's guard against unrestricted system plugin discovery. Use it
  only when you intentionally want the full system plugin set and accept the
  risk of duplicate registrations or conflicting factories.

- `SIMA_GST_ALLOW_SYSTEM_ALLOCATOR`
  Relaxes the allocator-origin check. By default the runtime verifies that the
  allocator symbol came from the expected third-party plugin directory; set
  this only when you deliberately want the system allocator instead.

- `SIMA_GST_WRAP_SCANNER`
  Controls whether Neat writes a wrapper around the plugin scanner so the
  scanner runs with the correct `LD_LIBRARY_PATH`. Leave it enabled unless you
  are debugging scanner behavior itself.

- `SIMA_GST_PLUGIN_PATH_DEBUG`
  Turns on verbose logging about which plugin directories were selected,
  filtered, skipped, or wrapped.

- `SIMA_GST_SUPPRESS_JSON_WARNINGS`
  Suppresses noisy JSON-GLib warnings that are usually not actionable for
  normal Neat runs. Set it to `0` when you need the raw warnings.

- `SIMA_GST_SUPPRESS_GOBJECT_ASSERTS`
  Suppresses known noisy `GLib-GObject` assert lines that often appear during
  scanner and plugin initialization. Set it to `0` when you want the raw log.

- `SIMA_GST_SUPPRESS_SEGMENT_WARNINGS`
  Suppresses common GStreamer segment warnings that are often repetitive and
  low-signal in normal runs. Set it to `0` when debugging segment/seek issues.

- `SIMA_GST_SUPPRESS_DEVICE_LOGS`
  Filters noisy device-side log lines, especially around rpmsg and model-load
  chatter. Set it to `0` if you need the raw device output.

### Pipeline lifecycle and general runtime behavior

- `SIMA_PIPELINE_STRING_DEBUG`
  Prints the final generated pipeline string before or during build. This is
  usually the fastest way to see what fragment composition actually produced.

- `SIMA_PIPELINE_STATE_DEBUG`
  Emits extra state-transition logging around build and runtime state changes.

- `SIMA_PIPELINE_DEBUG`
  Enables broader pipeline-level debug logging, especially around stop/close
  paths and zero-copy behavior.

- `SIMA_PIPELINE_TEARDOWN_DEBUG`
  Logs teardown ref-count and close-path decisions. This is narrower than
  `SIMA_PIPELINE_DEBUG` and most useful for leaked-handle or deferred-teardown
  investigations.

- `SIMA_STOP_TRACE`
  Emits explicit stop/close trace messages for both pipeline and graph
  runtimes.

- `SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS`
  Best-effort drain window before final teardown. The runtime uses it to allow
  a short flush of outputs before destroying the pipeline.

- `SIMA_PIPELINE_DRAIN_MIN_OUTPUTS`
  Minimum output count threshold used with the drain-before-teardown logic.

- `SIMA_PIPELINE_ABORT_ON_HUNG_STOP_THREADS`
  Changes hung-stop behavior from "log and detach" to "abort the process." This
  is a deliberate escalation knob for catching shutdown deadlocks.

- `SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS`
  First deadline for stopping the underlying stream during `Run::stop()`.

- `SIMA_PIPELINE_STREAM_STOP_TIMEOUT_MS_2`
  Optional second grace period after the runtime has already tried a forced
  stop path.

- `SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS`
  First deadline for the input thread to exit during `Run::stop()`.

- `SIMA_PIPELINE_INPUT_THREAD_STOP_TIMEOUT_MS_2`
  Optional second grace period for the input thread.

- `SIMA_PIPELINE_PUSH_RETURN_DEBUG`
  Adds extra logging around push return paths.

- `SIMA_PIPELINE_OUTPUT_DROP_ON_ZERO_COPY`
  Controls whether the runtime may drop outputs instead of retaining too many
  zero-copy buffers under pressure.

- `SIMA_STATE_CHANGE_TIMEOUT_MS`
  Main timeout for GStreamer state transitions during build/preroll.

- `SIMA_GST_RUN_INPUT_TIMEOUT_MS`
  Default timeout used by input-driven build/run paths when waiting for the
  first meaningful runtime signal.

- `SIMA_GST_VALIDATE_TIMEOUT_MS`
  Timeout used by `validate()`-style preroll checks.

- `SIMA_GST_TEARDOWN_TIMEOUT_MS`
  Timeout for waiting on GStreamer teardown to reach `NULL`.

- `SIMA_GST_TEARDOWN_REAPER_MS`
  Poll interval used by the teardown watchdog/reaper logic.

- `SIMA_GST_TEARDOWN_ASYNC`
  Lets teardown proceed asynchronously instead of waiting synchronously for the
  pipeline to reach `NULL`.

- `SIMA_GST_TEARDOWN_DEFER_NO_FLUSH`
  Controls whether async teardown uses the more conservative deferred-no-flush
  path.

- `SIMA_GST_DOT_DIR`
  Directory where the runtime writes DOT graphs for debug or failure snapshots.

### Diagnostics and probe wiring

- `SIMA_GST_BOUNDARY_PROBES`
  Enables boundary-flow probes for stall localization.

- `SIMA_GST_BOUNDARY_BUFFER_DEBUG`
  Adds verbose buffer logging to those boundary probes.

- `SIMA_GST_STAGE_TIMINGS`
  Attaches stage-level timing probes.

- `SIMA_GST_ELEMENT_TIMINGS`
  Attaches per-element timing probes.

- `SIMA_GST_FLOW_DEBUG`
  Attaches per-element flow probes and counters.

- `SIMA_GST_ENFORCE_NAMES`
  Turns on stricter element-name contract checks during build.

- `SIMA_GST_OPTIONS_DEBUG`
  Logs resolved GStreamer build options and selected element settings.

- `SIMA_GST_BUFFER_DEBUG_LIMIT`
  Caps how many buffer-debug lines a probe emits before it quiets down.

- `SIMA_GST_DETESS_INPUT_DEBUG`
  Adds focused debug output for Detess inputs.

- `SIMA_GST_DETESS_OUTPUT_DEBUG`
  Adds focused debug output for Detess outputs.

- `SIMA_GST_DETESS_POOL_DEBUG`
  Adds focused debug output for Detess-side pool behavior.

- `SIMA_GST_APPSINK_BUFFER_DEBUG`
  Adds appsink buffer logging.

- `SIMA_GST_ALL_BUFFER_DEBUG`
  Adds very broad buffer logging across the pipeline.

- `SIMA_GST_BOXDECODE_BUFFER_DEBUG`
  Adds focused debug output for boxdecode buffers.

- `SIMA_GST_PAD_LINK_DEBUG`
  Logs pad-link decisions and events.

- `SIMA_GST_BUFFER_MEMFLAGS_DEBUG`
  Adds buffer memory-flag detail to debug output.

- `SIMA_GST_RUN_INSERT_BOUNDARIES`
  Forces boundary insertion in run/build flows.

- `SIMA_GST_VALIDATE_INSERT_BOUNDARIES`
  Forces boundary insertion in validate flows.

- `SIMA_GST_DATA_ADAPTER_DEBUG`
  Debugs the internal data-adapter logic used in some conversion paths.

- `SIMA_GST_ZERO_COPY_WRITABLE_VIEW`
  Changes and debugs writable-view handling in zero-copy paths.

- `SIMA_APPSINK_CAPS_DEBUG`
  Adds appsink caps logging.

- `SIMA_APPSINK_PULL_DEBUG`
  Adds appsink pull-path logging.

- `SIMA_APPSINK_LAST_SAMPLE_DEBUG`
  Adds debug output around appsink last-sample access.

- `SIMA_APPSINK_CB_DEBUG`
  Adds appsink callback-path logging.

- `SIMA_APPSINK_DROP_LAST_DEBUG`
  Adds logging when appsink last-sample/drop behavior matters.

- `NEAT_MODEL_DEBUG`
  Convenience switch for `Model`-driven flows. When enabled, the runtime turns
  on several verbose diagnostics internally, including flow debug, element
  timings, boundary probes, pad-link debug, and related stage-config traces.

### Dispatcher, model-pack, and stage wiring

- `SIMA_DISPATCHER_TRACE`
  Logs dispatcher lifecycle steps, especially around build and startup.

- `SIMA_DISPATCHER_AUTO_RECOVER`
  Controls whether dispatcher-backed flows try to auto-recover from certain
  failure modes.

- `SIMA_DISPATCHER_WATCHDOG`
  Enables the dispatcher watchdog helper.

- `SIMA_DISPATCHER_WATCHDOG_PATH`
  Overrides the executable path used for the dispatcher watchdog helper.

- `SIMA_ASYNC_TPUT_DIAG`
  Enables async throughput diagnostics.

- `SIMA_ASYNC_WARMUP`
  Controls how many initial async frames are treated as warmup.

- `SIMA_PULL_TIMEOUT_DIAG`
  Adds diagnostics when pull operations time out.

- `SIMA_PULL_TIMEOUT_POOL_DIAG`
  Adds extra pool-side detail when pull operations time out.

- `SIMA_STAGE_DEBUG`
  Logs StageRun activity and runtime decisions.

- `SIMA_MPK_EXTRACT_ROOT`
  Overrides where `ModelPack` extracts `.mpk` contents.

- `SIMA_MLA_NEXT_CPU`
  Overrides the `next_cpu` value used for MLA stage wiring in model-backed
  flows.

- `SIMA_PREPROC_DEBUG_CONFIG`
  Dumps preprocessing config wiring and resolved options.

- `SIMA_KEEP_DETESS_CONFIG`
  Keeps Detess config outputs that would otherwise be treated as transient.

- `SIMA_DETESS_ASSERT_ON_ZERO`
  Turns suspicious all-zero Detess output into an explicit failure.

- `SIMA_DETESS_ZERO_COPY`
  Favors zero-copy behavior in certain Detess-related paths.

- `SIMA_DETESS_FORCE_CPU_OUT`
  Forces Detess outputs back to CPU memory for easier inspection or
  compatibility.

- `SIMA_MLA_CONFIG_DEBUG`
  Emits MLA plugin configuration detail.

- `SIMA_FORCE_SYNC_NUMBUFFERS_ONE`
  Forces sync-mode `num-buffers` decisions toward `1`.

- `SIMA_CLAMP_DETESS_NUM_BUFFERS`
  Enables explicit clamping of Detess `num-buffers`.

- `SIMA_DISABLE_SYNC_NUMBUFFERS_CVU_MLA`
  Disables the default sync-mode `num-buffers` clamp across CVU/MLA paths.

- `SIMA_FORCE_MODEL_NUM_BUFFERS`
  Overrides resolved model-stage `num-buffers`.

- `SIMA_FORCE_DECODER_NUM_BUFFERS`
  Overrides decoder `num-buffers`.

- `SIMA_FORCE_DECODER_POOL_BUFFERS`
  Overrides decoder pool-buffer sizing.

- `SIMA_MLA_NUM_BUFFERS_DEBUG`
  Logs MLA-side `num-buffers` resolution decisions.

- `SIMA_IMAGEFREEZE_MIN_BUFFERS`
  Controls the minimum `imagefreeze` buffer count used by image-input groups.

- `SIMA_ENABLE_ASYNC_QUEUE2`
  Forces queue2 insertion behavior in async build paths.

- `SIMA_SYNC_RUN_NUM_BUFFERS`
  Overrides the sync-mode `num-buffers` count used by build-input logic.

- `SIMA_DEBUG_OUTPUTSPEC_LOG`
  Logs `OutputSpec` propagation decisions across pipeline composition.

### InputStream, Sample, and Tensor handling

- `SIMA_BUILD_MODE_DEBUG`
  Logs how build-input logic resolved run mode, queue insertion, and similar
  decisions.

- `SIMA_INPUTSTREAM_DEBUG`
  Turns on general InputStream logging.

- `SIMA_INPUTSTREAM_WARN`
  Turns on InputStream warnings even when broader debug is off.

- `SIMA_INPUTSTREAM_POLL_MS`
  Changes the InputStream worker poll interval.

- `SIMA_INPUTSTREAM_DOT_ON_TIMEOUT`
  Requests DOT dumps when InputStream hits a timeout path.

- `SIMA_INPUTSTREAM_META_DEBUG`
  Logs GstSimaMeta-related behavior.

- `SIMA_INPUTSTREAM_ALLOC_DEBUG`
  Logs InputStream allocation behavior.

- `SIMA_INPUTSTREAM_PUSH_TIMING`
  Logs push timing detail.

- `SIMA_INPUTSTREAM_HOLDER_DEBUG`
  Logs holder lifetime and inflight behavior.

- `SIMA_INPUTSTREAM_PUSH_REF_DEBUG`
  Logs push-path ref behavior.

- `SIMA_INPUTSTREAM_UNREF_ON_PUSH_FAIL`
  Changes ref cleanup behavior on push failure.

- `SIMA_INPUTSTREAM_PUSH_FAIL_DEBUG`
  Adds InputStream push-failure logging.

- `SIMA_INPUTSTREAM_PUSH_FAIL_DETAIL`
  Adds more detail to InputStream push-failure logging.

- `SIMA_INPUTSTREAM_DROP_HOLDER_AFTER_PUSH`
  Controls whether holders are released immediately after push.

- `SIMA_INPUTSTREAM_EOS_DEBUG`
  Logs end-of-stream behavior in InputStream.

- `SIMA_INPUTSTREAM_POOL_DEBUG`
  Adds debug output for InputStream pool behavior.

- `SIMA_INPUTSTREAM_USE_APPSINK_CALLBACKS`
  Switches InputStream toward appsink callback mode.

- `SIMA_INPUTSTREAM_STOP_UNBLOCK`
  Controls whether InputStream stop actively unblocks pending work.

- `SIMA_INPUTSTREAM_STOP_FLUSH`
  Controls whether InputStream stop flushes pending work.

- `SIMA_INPUTSTREAM_HOLDER_MAX_INFLIGHT`
  Caps holder inflight count when that protection is enabled.

- `SIMA_INPUTSTREAM_CB_STOP_TIMEOUT_MS`
  Timeout for callback-mode stop cleanup.

- `SIMA_INPUTSTREAM_STOP_FLUSH_TIMEOUT_MS`
  Timeout for stop-flush behavior.

- `SIMA_INPUTSTREAM_STOP_TIMEOUT_MS`
  Overall InputStream stop timeout.

- `SIMA_INPUTSTREAM_ELASTIC_MAX_MB`
  Caps elastic input-pool growth in megabytes.

- `SIMA_INPUTSTREAM_POOL_WAIT_LOG_MS`
  Controls how often pool-wait logging is emitted.

- `SIMA_INPUTSTREAM_PREFLIGHT_RUN`
  Enables a preflight run for some InputStream paths.

- `SIMA_NEAT_CAPS_TRACE`
  Traces caps inference and tensor-cap derivation.

- `SIMA_SAMPLE_DEBUG`
  Logs sample conversion behavior.

- `SIMA_SAMPLE_BYTES`
  Logs sample byte sizes.

- `SIMA_SAMPLE_FORCE_BUNDLE`
  Forces bundle output for debugging.

- `SIMA_TENSOR_MAPFAIL_DEBUG`
  Logs more information when tensor mapping fails.

### Graph runtime

- `SIMA_GRAPH_DEBUG`
  Enables broad graph-runtime logging.

- `SIMA_GRAPH_PUSH_FAIL_DEBUG`
  Adds detail around graph push failures.

- `SIMA_GRAPH_SERIAL_PIPELINE_BUILD`
  Forces serialized graph pipeline build behavior.

- `SIMA_GRAPH_OUTPUT_RATE_MS`
  Enables periodic output-rate reporting at the chosen interval.

- `SIMA_GRAPH_SCHED_DEBUG`
  Turns on graph scheduler assignment logging.

- `SIMA_GRAPH_SCHED_LOG_EVERY`
  Controls how often scheduler count logs are emitted.

- `SIMA_GRAPH_SCHED_LOG_FIRST_STREAM`
  Controls whether the first assignment of a stream is logged.

- `SIMA_GRAPH_STOP_TIMEOUT_MS`
  Sets the graph stop deadline.

- `SIMA_GRAPH_DIAG_ON_STOP`
  Emits extra graph diagnostics during stop.

- `SIMA_GRAPH_IDENTITY_MAP_CAPACITY`
  Sets the identity-map capacity used by graph runtime bookkeeping.

- `SIMA_GRAPH_ZERO_COPY_BACKPRESSURE_CAP`
  Caps queue pressure before graph runtime clones zero-copy outputs.

- `SIMA_GRAPH_ZERO_COPY_MAX_INFLIGHT`
  Legacy-compatible alias used by some zero-copy backpressure logic.

- `SIMA_GRAPH_ZERO_COPY_DEBUG`
  Logs zero-copy backpressure decisions in graph runtime.

- `SIMA_GRAPH_BUILD_TIMEOUT_MS`
  Sets a graph build timeout.

- `SIMA_GRAPH_PIPELINE_DIAG_MS`
  Enables timed graph-pipeline diagnostics.

- `SIMA_GRAPH_PIPELINE_DIAG_SUMMARY`
  Enables a summary printout for graph-pipeline diagnostics.

- `SIMA_GRAPH_GDB_ON_PUSH_FAIL`
  Escalation knob that can drop into a debugger on graph push failure.

- `SIMA_GRAPH_EDGE_LOG`
  Adds edge-level logging in some graph examples and tests.

- `SIMA_GRAPH_SYNC_DEBUG`
  Adds sync-path graph debug output.

- `SIMA_GRAPH_EDGE_QUEUE`
  Sets edge queue sizing in some graph-heavy tests/examples.

- `SIMA_GRAPH_PUSH_TIMEOUT_MS`
  Controls graph push timeout behavior in some hybrid graph flows.

- `SIMA_GRAPH_OUTPUT_COPY_DEBUG`
  Adds logging around graph output copy decisions.

### RTSP and H264

- `SIMA_RTSP_ALLOW_BACKPRESSURE`
  Allows RTSP input build logic to tolerate more backpressure-oriented behavior.

- `SIMA_RTSP_DEBUG`
  Enables RTSP lifecycle logging.

- `SIMA_RTSP_STATS_DEBUG`
  Enables periodic RTSP stats logging.

- `SIMA_RTSP_STATS_POLL_MS`
  Poll interval used by RTSP stats collection.

- `SIMA_RTSP_STOP_TIMEOUT_MS`
  RTSP-specific stop timeout.

- `SIMA_H264_SDP_DUMP`
  Turns on H264 SDP dumping.

- `SIMA_H264_SPS_FIXUP_STREAM`
  Turns on SPS fix-up stream handling.

- `SIMA_H264ENC_BITRATE_KBPS`
  Overrides software H264 encoder bitrate, especially for `x264enc`.

- `SIMA_H264ENC_LOSSLESS`
  Forces lossless-style behavior for supported software H264 encoders.

- `SIMA_H264ENC_QP`
  Overrides software H264 encoder quantization parameter.

- `SIMA_NEATENCODER_DUMP_CNT`
  Passes dump-count configuration into the hardware `neatencoder` element.

- `SIMA_NEATENCODER_DUMP_PATH`
  Passes dump-path configuration into the hardware `neatencoder` element.

### Models, examples, and local assets

- `SIMA_RESNET50_TAR`
  Explicit path to the ResNet50 model artifact used by examples or tests.

- `SIMA_YOLO_TAR`
  Explicit path to the YOLO model artifact used by examples or tests.

- `SIMA_YOLO11_POSE_BF16_TAR`
  Explicit path to the YOLO11 pose bf16 artifact used by specific tests.

- `SIMA_MIDAS_TAR`
  Explicit path to the MiDaS model artifact used by examples.

- `SIMA_MIDAS_SAMPLE_VIDEO`
  Explicit path to a sample video used by the MiDaS example.

- `SIMA_COCO_URL`
  Alternate image URL used by asset helpers.

- `REPO_ROOT_SIMA`
  Used by some examples to find repo-relative data when the current working
  directory is not enough.

- `RTSP_HOST`, `HOST_IP`, `RTSP_PORT`, `UDP_HOST`, `OUTPUT_HOST`, `UDP_PORT`,
  `VIDEO_PORT`, `RTSP_FPS`, and `OCR_THROTTLE`
  Example-level knobs used mainly by the ANPR and streaming helpers to control
  output hosts, ports, FPS, and OCR throttling.

### Tutorials, tests, and perf helpers

- `SIMA_RUN_TUTORIALS_FULL`
  Enables heavier tutorial paths that are skipped by default.

- `SIMA_TUTORIAL_ENFORCE_RUNTIME`
  Tightens tutorial parity/runtime checking.

- `SIMA_TUTORIAL_TIMEOUT_SEC`
  Overrides tutorial timeout behavior in Python parity tests.

- `SIMA_TEST_IS_LONG`
  Marks a run as willing to execute longer tests.

- `SIMA_TEST_DUMP_ERRORS`
  Makes some tests print fuller failure detail.

- `SIMA_STRESS_ITERS`
  Controls stress-test iteration counts.

- `SIMA_STRESS_MODEL_TAR`
  Explicit model artifact path used by some stress tests.

- `SIMA_PERF_ITERS`
  Controls perf-test iteration counts.

- `SIMA_PERF_SCENARIO_TIMEOUT_SEC`
  Controls per-scenario timeout budgets in perf helpers.

- `SIMA_DECODER_MIN_FRAMES`
  Raises the minimum number of frames expected before a decoder test is
  considered valid.

- `SIMA_DEC_IPC_PROTOCOL`
  Forces a specific decoder IPC protocol path in relevant tests.

- `SIMA_DECODER_ELEMENT`
  Lets decoder tests choose which decoder element name to exercise.

- `SIMA_OS_DECODER_ELEMENT`
  Lets comparison tests choose an alternate OS-side decoder element.

- `SIMA_DECODER_COMPARE_IMAGE`
  Explicit image path used by decoder image-comparison tests.

- `SIMA_DECODER_COMPARE_MAE_MAX`
  Maximum allowed mean absolute error for decoder image-comparison tests.

- `SIMA_DECODER_COMPARE_PSNR_MIN`
  Minimum allowed PSNR for decoder image-comparison tests.

- `SIMA_DECODER_COMPARE_MAX_ABS`
  Maximum allowed absolute pixel delta for decoder image-comparison tests.

- `SIMA_DECODER_COMPARE_DUMP`
  Enables dump-on-failure behavior for decoder comparison tests.

- `SIMA_RTSP_PORT_BASE`, `SIMA_RTSP_PORT`, `SIMA_RTSP_PORT_RANGE`,
  `SIMA_RTSP_RTP_PORT_OFFSET`, `SIMA_RTSP_RTP_PORT_COUNT`,
  `SIMA_RTSP_RTP_PORT_STRIDE`, `SIMA_RTSP_WARMUP`, and `SIMA_RTSP_ATTEMPTS`
  Test and example helpers use these to allocate RTSP/RTP ports and to decide
  how much warmup or retry logic to apply.

- `SIMA_PREPROC_USE_POOL`, `SIMA_PREPROC_NUM_BUFFERS`,
  `SIMA_PREPROC_KEEP_CONFIG`, and `SIMA_INPUT_TIMEOUT_MS`
  Narrow knobs used by standalone preproc tests.

### Docs site and docs-index scripts

- `DOCS_PATH`, `DOCS_ORG`, `DOCS_PROJECT`, `DOCS_URL`, and `DOCS_BASE_URL`
  Drive Docusaurus docs resolution and public site URL generation.

- `DOCS_ALGOLIA_APP_ID`, `DOCS_ALGOLIA_API_KEY`, and `DOCS_ALGOLIA_INDEX_NAME`
  Feed Algolia search configuration into the website build.

- `DOCS_DIR`, `SITE_BASE_URL`, `BATCH_SIZE`, and `MAX_RECORD_BYTES`
  Configure `scripts/ci/sync_algolia_docs_index.sh`: which docs directory is
  indexed, which public base URL is baked into records, and how batching and
  truncation behave.

- `ALGOLIA_APP_ID`, `ALGOLIA_ADMIN_API_KEY`, and `ALGOLIA_INDEX_NAME`
  Used by the Algolia indexing script for authenticated record upload.

- `AWS_REGION`, `S3_BUCKET`, and `CLOUDFRONT_DISTRIBUTION_ID`
  Used by `website/deploy.sh` to choose the AWS region, S3 bucket, and
  optional CloudFront invalidation target for docs deployment.

## Deprecated or legacy

These variables still exist in compatibility code paths or old debug flows, but
they are not good defaults for new work.

- `SIMA_DETESS_MULTI_BUFFER`
  Deprecated compatibility knob. Setting it only preserves or triggers legacy
  Detess behavior and may emit warnings.

- `SIMA_STRICT_CONFIG_WIRING`
  Deprecated no-op from older JSON wiring validation flows.

- `SIMA_WIRE_BY_ORDER_DEBUG`
  Deprecated no-op from legacy config rewrite debugging.

- `SIMA_BOXDECODE_WIRE_DEBUG`
  Deprecated no-op from older boxdecode wiring rewrite debugging.

- `SIMA_NEATMODEL_USE_MLA`
  Legacy env-var name retained for older `Model` MLA forcing behavior. Prefer
  current public APIs and current runtime configuration instead.
