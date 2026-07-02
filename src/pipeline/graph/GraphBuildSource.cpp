/**
 * @file SessionBuildSource.cpp
 * @brief Graph source-mode build/run methods.
 */

#include "GraphDetail.h"
#include "internal/GraphBuildInternal.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"

#include "builder/OutputSpec.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace simaai::neat {

// Source-mode build product. Holds RAII handles for the materialised (and paused)
// GStreamer pipeline so that any throw from the caller's state-transition step
// tears down cleanly via stack unwinding — no manual `gst_object_unref` cleanup
// needed at call sites.
struct Graph::PreparedSource {
  pipeline_internal::GstPipelinePtr pipeline;       // PAUSED on return; non-null
  pipeline_internal::GstSinkPtr sink;               // empty if has_sink == false
  std::shared_ptr<pipeline_internal::DiagCtx> diag; // typed: needed by APIs below;
                                                    // converts implicitly to
                                                    // shared_ptr<void> for BuiltState
  RunOptions merged_opt;
  InputStreamOptions stream_opt;
  bool has_sink = false;
  NameTransform name_transform{};
};

namespace {

struct PreparedSourcePipeline {
  pipeline_internal::GstPipelinePtr pipeline; // PAUSED on return; non-null
  pipeline_internal::GstSinkPtr sink;         // empty if has_sink == false
  std::shared_ptr<pipeline_internal::DiagCtx> diag;
  RunOptions merged_opt;
  InputStreamOptions stream_opt;
  bool has_sink = false;
  NameTransform name_transform{};
};

void maybe_compile_source_contracts(BuildResult* build_result,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const GraphOptions& sess_opt, const char* where) {
  ContractCompileInput compile_input;
  compile_input.pipeline_label = where ? where : "Graph::build(source)";
  compile_input.processcvu_requested_run_target = sess_opt.processcvu_requested_run_target;
  compile_input.processcvu = sess_opt.processcvu;
  session_build_compile_contracts(build_result, nodes, compile_input, where, nullptr);
}

std::string source_meta_upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

bool source_meta_debug_enabled() {
  static const bool enabled =
      env_bool("SIMA_SOURCE_META_DEBUG", false) || env_bool("SIMA_INPUTSTREAM_META_DEBUG", false);
  return enabled;
}

int source_meta_channels_from_format(const std::string& format, int fallback = -1) {
  const std::string up = source_meta_upper_copy(format);
  if (up == "GRAY" || up == "GRAY8")
    return 1;
  if (up == "RGB" || up == "BGR" || up == "NV12" || up == "I420" || up == "YUYV")
    return 3;
  return fallback;
}

bool source_meta_dtype_is_quantized(const std::string& dtype) {
  const std::string up = source_meta_upper_copy(dtype);
  return up == "INT8" || up == "UINT8" || up == "U8" || up == "EVXX_INT8" ||
         up == "INT16" || up == "UINT16" || up == "EVXX_INT16";
}

std::optional<PreprocessMetaTemplate>
source_meta_template_from_preproc(const PreprocOptions& opt, const OutputSpec& source_spec) {
  PreprocessMetaTemplate meta;
  meta.enabled = true;
  meta.normalize = opt.normalize;
  meta.quantize = source_meta_dtype_is_quantized(opt.output_dtype);
  meta.tessellate = opt.tessellate;
  meta.target_width = opt.output_width() > 0 ? opt.output_width() : opt.scaled_width;
  meta.target_height = opt.output_height() > 0 ? opt.output_height() : opt.scaled_height;
  meta.scaled_width = opt.scaled_width > 0 ? opt.scaled_width : meta.target_width;
  meta.scaled_height = opt.scaled_height > 0 ? opt.scaled_height : meta.target_height;
  meta.resize_mode =
      (meta.target_width > 0 && meta.target_height > 0) ? (opt.aspect_ratio ? "letterbox"
                                                                            : "stretch")
                                                        : "none";
  meta.pad_value = opt.pad_value;
  meta.color_in = !opt.input_img_type.empty() ? opt.input_img_type : source_spec.format;
  meta.color_out = opt.output_img_type;
  return meta;
}

OutputSpec source_meta_output_spec_for_node(const std::shared_ptr<Node>& node) {
  if (!node)
    return {};
  const auto* provider = dynamic_cast<const OutputSpecProvider*>(node.get());
  if (!provider)
    return {};
  return provider->output_spec({});
}

std::optional<PreprocessMetaTemplate>
derive_source_preprocess_meta_template(const std::vector<std::shared_ptr<Node>>& nodes,
                                       std::size_t source_index, const OutputSpec& source_spec) {
  for (std::size_t i = source_index + 1; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    if (nodes[i]->input_role() != InputRole::None)
      break;
    if (const auto* preproc = dynamic_cast<const Preproc*>(nodes[i].get())) {
      return source_meta_template_from_preproc(preproc->options(), source_spec);
    }
    if (nodes[i]->memory_contract() == MemoryContract::PreferDeviceZeroCopy) {
      break;
    }
  }
  return std::nullopt;
}

InputOptions source_meta_input_options_for_node(const std::vector<std::shared_ptr<Node>>& nodes,
                                                std::size_t node_index,
                                                const std::string& buffer_name) {
  InputOptions opt;
  opt.buffer_name = buffer_name;
  opt.use_simaai_pool = true;
  opt.memory_policy = InputMemoryPolicy::Ev74;

  const OutputSpec spec = source_meta_output_spec_for_node(nodes[node_index]);
  if (spec.media_type == "video/x-raw" ||
      is_raw_video_format(format_tag_from_string(spec.format))) {
    opt.payload_type = PayloadType::Image;
  }
  if (!spec.format.empty()) {
    opt.format = spec.format;
  }
  opt.width = spec.width;
  opt.height = spec.height;
  opt.depth = spec.depth > 0 ? spec.depth : source_meta_channels_from_format(spec.format, -1);
  opt.fps_n = spec.fps_num;
  opt.fps_d = spec.fps_den > 0 ? spec.fps_den : 1;
  opt.max_width = spec.width;
  opt.max_height = spec.height;
  opt.max_depth = opt.depth;
  opt.preprocess_meta = derive_source_preprocess_meta_template(nodes, node_index, spec);
  return opt;
}

struct SourceSimaMetaProbeCtx {
  InputOptions opt;
  InputBufferPoolGuard guard;
  std::string element_name;
  std::string buffer_name;
  std::atomic<bool> logged{false};
};

struct RawVideoCapsInfo {
  int width = -1;
  int height = -1;
};

RawVideoCapsInfo source_meta_current_raw_video_caps(GstPad* pad) {
  RawVideoCapsInfo out;
  if (!pad)
    return out;
  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps) {
    caps = gst_pad_query_caps(pad, nullptr);
  }
  if (!caps)
    return out;
  if (!gst_caps_is_empty(caps)) {
    GstStructure* s = gst_caps_get_structure(caps, 0);
    if (s && g_strcmp0(gst_structure_get_name(s), "video/x-raw") == 0) {
      (void)gst_structure_get_int(s, "width", &out.width);
      (void)gst_structure_get_int(s, "height", &out.height);
    }
  }
  gst_caps_unref(caps);
  return out;
}

void source_meta_maybe_apply_preprocess_template(GstPad* pad, GstBuffer* buffer,
                                                 SourceSimaMetaProbeCtx* ctx) {
  if (!buffer || !ctx || !ctx->opt.preprocess_meta.has_value())
    return;
  if (has_simaai_preprocess_meta(buffer))
    return;

  int width = ctx->opt.width;
  int height = ctx->opt.height;
  if (width <= 0 || height <= 0) {
    const RawVideoCapsInfo caps = source_meta_current_raw_video_caps(pad);
    if (width <= 0)
      width = caps.width;
    if (height <= 0)
      height = caps.height;
  }
  if (width <= 0 || height <= 0)
    return;

  (void)apply_simaai_preprocess_meta_template(buffer, ctx->opt, width, height);
}

GstPadProbeReturn source_sima_meta_probe_cb(GstPad* pad, GstPadProbeInfo* info,
                                            gpointer user_data) {
  auto* ctx = static_cast<SourceSimaMetaProbeCtx*>(user_data);
  if (!ctx || !info || (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0)
    return GST_PAD_PROBE_OK;

  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer)
    return GST_PAD_PROBE_OK;

  const int64_t frame_id = next_input_frame_id();
  SampleTimingOverrides timing;
  timing.frame_id = frame_id;
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))) {
    timing.pts_ns = static_cast<uint64_t>(GST_BUFFER_PTS(buffer));
  }
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))) {
    timing.dts_ns = static_cast<uint64_t>(GST_BUFFER_DTS(buffer));
  }
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))) {
    timing.duration_ns = static_cast<uint64_t>(GST_BUFFER_DURATION(buffer));
  }

  // Source/decode metadata is stamped before this buffer crosses any graph edge.
  // Leave the stream id empty here: graph links and realtime routing own per-source
  // identity, so multi-source graphs do not collapse to a synthetic "0" stream.
  GstBuffer* writable = attach_simaai_meta_inplace(
      buffer, ctx->opt, ctx->guard, ctx->element_name.c_str(), std::optional<int64_t>{frame_id},
      StreamIdOverride{std::optional<std::string>{std::string{}}},
      BufferNameOverride{std::optional<std::string>{ctx->buffer_name}});
  if (!writable)
    return GST_PAD_PROBE_OK;

  const std::optional<int64_t> frame_opt{frame_id};
  const std::optional<std::string> stream_id{std::string{}};
  const std::optional<std::string> buffer_name{ctx->buffer_name};
  (void)update_simaai_meta_fields(writable, frame_opt, frame_opt, frame_opt, stream_id,
                                  buffer_name, timing.pts_ns, buffer_name,
                                  std::optional<int>{0});
  (void)write_sample_timing_to_gst_buffer(writable, timing);
  source_meta_maybe_apply_preprocess_template(pad, writable, ctx);

  GST_PAD_PROBE_INFO_DATA(info) = writable;

  if (source_meta_debug_enabled() && !ctx->logged.exchange(true)) {
    std::fprintf(stderr,
                 "[source-meta] attached GstSimaMeta at %s buffer-name=%s preprocess_meta=%d\n",
                 ctx->element_name.c_str(), ctx->buffer_name.c_str(),
                 ctx->opt.preprocess_meta.has_value() ? 1 : 0);
    dump_sima_meta_full(writable, ctx->element_name.c_str());
  }
  return GST_PAD_PROBE_OK;
}

void source_sima_meta_probe_destroy(gpointer data) {
  delete static_cast<SourceSimaMetaProbeCtx*>(data);
}

void attach_source_sima_meta_probes(GstElement* pipeline,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const NameTransform& name_transform) {
  if (!pipeline)
    return;

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node || node->memory_contract() != MemoryContract::PreferDeviceZeroCopy) {
      continue;
    }
    const std::string buffer_name = node->buffer_name_hint(static_cast<int>(i));
    if (buffer_name.empty())
      continue;

    std::vector<std::string> element_names = node->element_names(static_cast<int>(i));
    if (element_names.empty())
      continue;
    element_names = apply_name_transform(name_transform, element_names);
    const std::string source_tail = element_names.back();

    GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), source_tail.c_str());
    if (!element) {
      session_build_throw_session_error_simple(
          error_codes::kPipelineShape,
          std::string("source metadata probe could not find element '") + source_tail +
              "' for node " + node->kind(),
          "Ensure device-output node element_names() matches backend_fragment().");
    }
    GstPad* pad = gst_element_get_static_pad(element, "src");
    gst_object_unref(element);
    if (!pad) {
      session_build_throw_session_error_simple(
          error_codes::kPipelineShape,
          std::string("source metadata probe could not find static src pad on '") + source_tail +
              "' for node " + node->kind(),
          "Device-output nodes need a static src pad for metadata stamping.");
    }

    auto* ctx = new SourceSimaMetaProbeCtx();
    ctx->opt = source_meta_input_options_for_node(nodes, i, buffer_name);
    ctx->element_name = source_tail;
    ctx->buffer_name = buffer_name;
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, source_sima_meta_probe_cb, ctx,
                      source_sima_meta_probe_destroy);
    gst_object_unref(pad);

    if (source_meta_debug_enabled()) {
      std::fprintf(stderr,
                   "[source-meta] armed source metadata probe on %s buffer-name=%s "
                   "preprocess_meta=%d\n",
                   source_tail.c_str(), buffer_name.c_str(),
                   ctx->opt.preprocess_meta.has_value() ? 1 : 0);
    }
  }
}

PreparedSourcePipeline prepare_source_pipeline_from_nodes(
    const std::vector<std::shared_ptr<Node>>& nodes, const GraphOptions& sess_opt, RunMode mode,
    const RunOptions& opt, bool require_sink, bool public_output_contract,
    std::string& last_pipeline, const char* where) {
  const pipeline_internal::ScopedBuildTiming timing(
      "prepare_source_pipeline_from_nodes",
      std::string("where=") + (where ? where : "Graph::build(source)") + " nodes=" +
          std::to_string(nodes.size()) + " require_sink=" + std::to_string(require_sink ? 1 : 0));
  // --- 1. validation ----------------------------------------------------------------
  enforce_caps_behavior(nodes, where);
  enforce_source_run_mode(nodes, where);
  if (require_sink) {
    enforce_sink_last(nodes);
  } else {
    enforce_sink_last_if_present(nodes, where);
  }

  // --- 2. node materialization + sink-presence decision -----------------------------
  std::vector<std::shared_ptr<Node>> build_nodes =
      session_build_materialize_model_bound_nodes(nodes, false);
  session_build_apply_derived_input_contracts(&build_nodes);

  // require_sink means "this code path needs a sink"; verify it.
  // !require_sink means "build whatever the caller described".
  const bool has_sink = require_sink ? true : session_build_has_output_appsink(build_nodes);
  if (has_sink) {
    require_element("appsink", where);
    require_element("identity", where);
  }

  // --- 3. option resolution ---------------------------------------------------------
  const RunOptions requested_opt = session_build_resolve_build_opt(mode, opt);
  RunOptions merged_opt = session_build_apply_run_defaults(requested_opt, sess_opt);
  InputStreamOptions stream_opt = session_build_make_stream_options(merged_opt, mode);
  stream_opt.public_output_contract = public_output_contract;
  session_build_maybe_enable_rtsp_appsink_drop(stream_opt, build_nodes);
  const bool insert_queue2 = session_build_should_insert_async_queue2(mode, merged_opt);

  // --- 4. compile -------------------------------------------------------------------
  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);
  const NameTransform name_transform = make_name_transform(sess_opt);
  BuildResult br = build_pipeline_full(build_nodes, insert_boundaries, "mysink", insert_queue2,
                                       name_transform, &sess_opt);
  maybe_compile_source_contracts(&br, build_nodes, sess_opt, where);
  if (has_sink && stream_opt.public_output_contract && br.rendered_manifest.has_value()) {
    const auto endpoint = session_build_public_output_endpoint_selector(build_nodes);
    std::string error;
    auto override = build_public_terminal_output_override_with_fallback(*br.rendered_manifest,
                                                                        endpoint, &error);
    if (override.has_value()) {
      stream_opt.output_override = std::move(*override);
    } else if (pipeline_internal::env_bool("SIMA_DETESS_OVERRIDE_DEBUG", false)) {
      std::fprintf(stderr, "[output-override] %s terminal override unavailable: %s\n",
                   where ? where : "Graph::build(source)", error.c_str());
    }
  }
  session_build_finalize_public_zero_copy_holder_loan_credits(stream_opt);
  last_pipeline = br.pipeline_string;
  session_build_enforce_mla_num_buffers(last_pipeline, where);
  session_build_maybe_dump_pipeline_string(last_pipeline, where);

  // --- 5. parse + probes ------------------------------------------------------------
  // pipeline ownership is taken by the RAII handle immediately so any throw below
  // tears the bin down via stack unwinding (no manual cleanup blocks needed).
  pipeline_internal::GstPipelinePtr pipeline(session_build_parse_pipeline_or_throw(br, where));
  session_build_dump_pipeline_element_properties(pipeline.get());
  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline.get(), br);
  }
  attach_boundary_probes(pipeline.get(), br.diag);
  attach_stage_timing_probes(pipeline.get(), br.diag, stream_opt.enable_timings);
  attach_element_timing_probes(pipeline.get(), br.diag, stream_opt.enable_timings);
  attach_element_flow_probes(pipeline.get(), br.diag);
  attach_source_sima_meta_probes(pipeline.get(), build_nodes, name_transform);
  session_build_attach_debug_detess_input_probes(pipeline.get());
  session_build_attach_debug_appsink_probes(pipeline.get());
  session_build_attach_debug_all_buffer_probes(pipeline.get());
  session_build_attach_debug_element_buffer_probes(pipeline.get());
  session_build_attach_boxdecode_debug_probes(pipeline.get());
  session_build_attach_h264_caps_fixups(pipeline.get(), build_nodes, name_transform);
  session_build_attach_encoded_caps_fixups(pipeline.get(), build_nodes, name_transform);
  session_build_attach_rtsp_debug(pipeline.get(), build_nodes, name_transform);

  // --- 6. resolve and configure sink ------------------------------------------------
  pipeline_internal::GstSinkPtr sink;
  if (has_sink) {
    const std::string appsink_name = apply_name_transform(name_transform, "mysink");
    GstElement* raw_sink = gst_bin_get_by_name(GST_BIN(pipeline.get()), appsink_name.c_str());
    if (!raw_sink) {
      maybe_dump_dot(pipeline.get(), "build_missing_mysink");
      // pipeline RAII teardown handles the bin on the next line via stack unwinding.
      session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                               std::string(where) + ": appsink '" + appsink_name +
                                                   "' not found.\nPipeline:\n" + last_pipeline,
                                               "Add Output() as the last node.", last_pipeline);
    }
    sink.reset(raw_sink);
    session_build_configure_appsink_for_input_stream(sink.get(), stream_opt);
    session_build_configure_appsink_allocation_preference(sink.get(), build_nodes);
  }

  // --- 7. drive to PAUSED -----------------------------------------------------------
  // On failure both `pipeline` and `sink` RAII handles unwind, no manual cleanup.
  set_state_or_throw(pipeline.get(), GST_STATE_PAUSED, where, br.diag);

  return PreparedSourcePipeline{
      /*pipeline=*/std::move(pipeline),
      /*sink=*/std::move(sink),
      /*diag=*/br.diag,
      /*merged_opt=*/std::move(merged_opt),
      /*stream_opt=*/std::move(stream_opt),
      /*has_sink=*/has_sink,
      /*name_transform=*/name_transform,
  };
}

} // namespace

// Build a source-mode pipeline to PAUSED. Single source of truth for the
// (formerly duplicated) bodies of `build(RunOptions)` and `build_cached_source()`.
// Differences between the two callers are expressed via parameters, not as
// missing-or-skipped operations — that's how the two paths used to drift.
Graph::PreparedSource Graph::prepare_source_(RunMode mode, const RunOptions& opt,
                                             SinkRequirement sink_req, const char* where) {
  const auto nodes = linear_nodes_snapshot(where);
  PreparedSourcePipeline src = prepare_source_pipeline_from_nodes(
      nodes, opt_, mode, opt, sink_req == SinkRequirement::Required,
      /*public_output_contract=*/true, last_pipeline_, where);

  return PreparedSource{
      /*pipeline=*/std::move(src.pipeline),
      /*sink=*/std::move(src.sink),
      /*diag=*/std::move(src.diag),
      /*merged_opt=*/std::move(src.merged_opt),
      /*stream_opt=*/std::move(src.stream_opt),
      /*has_sink=*/src.has_sink,
      /*name_transform=*/std::move(src.name_transform),
  };
}

SourceStreamBuildContext session_build_source_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    std::string& last_pipeline, const GraphOptions& sess_opt, const RunOptions& opt, RunMode mode,
    bool require_sink, bool public_output_contract, const char* where) {
  gst_init_once();

  PreparedSourcePipeline src = prepare_source_pipeline_from_nodes(
      nodes, sess_opt, mode, opt, require_sink, public_output_contract, last_pipeline, where);

  set_state_or_throw(src.pipeline.get(), GST_STATE_PLAYING, where, src.diag);

  SampleSpec spec = session_build_make_placeholder_spec();
  InputOptions src_opt = session_build_resolve_appsrc_options(InputOptions{}, src.name_transform);
  GstElement* pipeline_raw = src.pipeline.release();
  GstElement* sink_raw = src.sink.release();
  InputStream stream = InputStream::create(pipeline_raw, nullptr, sink_raw, spec, src_opt,
                                           src.stream_opt, src.diag, guard);

  return SourceStreamBuildContext{
      /*stream=*/std::move(stream),
      /*merged_opt=*/std::move(src.merged_opt),
      /*stream_opt=*/std::move(src.stream_opt),
  };
}

void Graph::run() {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  gst_init_once();

  const auto nodes = linear_nodes_snapshot("Graph::run");
  enforce_caps_behavior(nodes, "Graph::run");
  enforce_source_run_mode(nodes, "Graph::run");

  enforce_sink_last(nodes);

  require_element("appsink", "Graph::run");
  require_element("identity", "Graph::run");

  if (!tensor_cb_) {
    session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                             "Graph::run: tensor callback not set");
  }

  RunOptions run_opt;
  auto runner = build(run_opt);
  const int timeout_ms = std::max(-1, opt_.callback_timeout_ms);

  while (true) {
    Sample out;
    PullError err;
    const PullStatus status = runner.pull(timeout_ms, out, &err);
    if (status == PullStatus::Ok) {
      if (!sample_has_tensor_list(out) || out.tensors.size() != 1U) {
        session_build_throw_session_error_simple(error_codes::kRuntimePull,
                                                 "Graph::run: expected tensor output");
      }
      if (!tensor_cb_(out.tensors.front()))
        break;
      continue;
    }
    if (status == PullStatus::Timeout) {
      continue;
    }
    if (status == PullStatus::Closed) {
      break;
    }
    if (err.report.has_value()) {
      GraphReport rep = std::move(*err.report);
      if (rep.error_code.empty()) {
        rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
      }
      if (rep.repro_note.empty()) {
        rep.repro_note = "Graph::run: runtime pull failed (status=Error)\n"
                         "Hint: inspect the attached GraphReport diagnostics.";
      }
      const std::string msg =
          err.message.empty()
              ? session_build_decorate_with_error_code(rep.error_code, rep.repro_note)
              : err.message;
      throw NeatError(msg, std::move(rep));
    }
    const std::string detail = err.message.empty() ? "Graph::run: pull failed" : err.message;
    GraphReport rep;
    rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
    rep.repro_note = detail + "\nHint: inspect the attached GraphReport diagnostics.";
    const std::string msg = session_build_decorate_with_error_code(rep.error_code, rep.repro_note);
    throw NeatError(msg, std::move(rep));
  }
}

Run Graph::build(const RunOptions& opt) {
  const pipeline_internal::ScopedBuildTiming timing("Graph::build", "kind=no_input");
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt_.verbose, 4);
  progress.step("Initializing runtime...");
  gst_init_once();

  progress.step("Building graph...");
  runtime::ExecutionGraphPlan plan = runtime::compile_public_graph(*this, opt, std::nullopt);
  if (plan.linear_compat) {
    progress.detail(std::string("mode=async nodes=") +
                    std::to_string(linear_nodes_snapshot("Graph::build").size()));
  } else {
    progress.detail(std::string("mode=async graph_segments=") +
                    std::to_string(plan.pipeline_segments.size() + plan.stage_nodes.size()));
  }

  progress.step("Starting pipeline...");
  runtime::RunCoreStartOptions start_opt;
  start_opt.run_options = opt;
  start_opt.mode = RunMode::Async;
  start_opt.graph_options = runtime::graph_runtime_options_from_run_options(opt, opt_.verbose);
  start_opt.guard = guard_;
  start_opt.input_route_processor = input_route_processor_;
  start_opt.last_pipeline = &last_pipeline_;
  start_opt.owner = this;
  start_opt.require_sink = false;
  auto core = runtime::RunCore::start(std::move(plan), std::move(start_opt));
  progress.done("Graph ready");
  return Run(std::move(core));
}

void Graph::build_cached_source() {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  gst_init_once();

  // Cheap version-keyed cache check: if the composition hasn't changed and the
  // previous prepared pipeline is still resident, we're done.
  const uint64_t version = nodes_version_.load(std::memory_order_relaxed);
  if (built_ && built_->pipeline && built_->sink && built_version_ == version) {
    return;
  }

  // Drop any stale built state via RAII (BuiltState destructor unrefs sink
  // and stop_and_unrefs the pipeline in the right order).
  invalidate_built_();

  // Build to PAUSED via the shared helper. SinkRequirement::Required matches the
  // historical `build_cached_source()` behavior of demanding a terminal appsink.
  PreparedSource src =
      prepare_source_(RunMode::Async, RunOptions{}, SinkRequirement::Required, "Graph::build");

  // Stash directly into BuiltState — both fields are the same RAII handle types,
  // so this is a unique_ptr move with no manual unref-on-failure paths.
  built_ = std::make_unique<BuiltState>();
  built_->pipeline = std::move(src.pipeline);
  built_->sink = std::move(src.sink);
  built_->diag = std::move(src.diag);
  built_version_ = version;
}

std::string Graph::describe_backend(bool insert_boundaries) const {
  if (composition_ && !composition_->is_linear()) {
    (void)insert_boundaries;
    const runtime::ExecutionGraphPlan plan =
        runtime::compile_public_graph(*this, RunOptions{}, std::nullopt);
    std::ostringstream oss;
    oss << "ExecutionGraphPlan {\n";
    oss << "  mode: connected\n";
    oss << "  pipeline_segments: " << plan.pipeline_segments.size() << "\n";
    for (const auto& segment : plan.pipeline_segments) {
      oss << "    segment " << segment.id << " nodes=[";
      for (std::size_t i = 0; i < segment.nodes.size(); ++i) {
        if (i != 0U) {
          oss << ", ";
        }
        const auto& node = segment.nodes[i];
        oss << (node ? node->kind() : std::string("<null>"));
      }
      oss << "] needs_input=" << (segment.boundary.needs_input ? "true" : "false")
          << " needs_output=" << (segment.boundary.needs_output ? "true" : "false")
          << " terminal_output=" << (segment.boundary.terminal_output ? "true" : "false") << "\n";
    }
    oss << "  stage_nodes: " << plan.stage_nodes.size() << "\n";
    for (const auto& stage : plan.stage_nodes) {
      oss << "    stage n" << stage.node_id;
      if (stage.node) {
        oss << " kind=" << stage.node->kind() << " label=" << stage.node->user_label();
      }
      oss << "\n";
    }
    oss << "  edges: " << plan.edges.size() << "\n";
    auto port_name = [&](graph::PortId port) -> std::string {
      if (port == graph::kInvalidPort) {
        return "auto";
      }
      return port < plan.port_names.size() ? plan.port_names[port]
                                           : ("port" + std::to_string(port));
    };
    for (std::size_t i = 0; i < plan.edges.size(); ++i) {
      const auto& edge = plan.edges[i];
      oss << "    e" << i << ": n" << edge.from << ":" << port_name(edge.from_port) << " -> n"
          << edge.to << ":" << port_name(edge.to_port) << "\n";
    }
    oss << "  named_inputs: [";
    bool first = true;
    for (const auto& [name, endpoint] : plan.named_inputs) {
      (void)endpoint;
      if (!first) {
        oss << ", ";
      }
      first = false;
      oss << name;
    }
    oss << "]\n";
    oss << "  named_outputs: [";
    first = true;
    for (const auto& [name, endpoint] : plan.named_outputs) {
      (void)endpoint;
      if (!first) {
        oss << ", ";
      }
      first = false;
      oss << name;
    }
    oss << "]\n";
    oss << "}\n";
    return oss.str();
  }

  const NameTransform name_transform = make_name_transform(opt_);
  const auto nodes = linear_nodes_snapshot("Graph::describe_backend");
  std::vector<std::shared_ptr<Node>> build_nodes =
      session_build_materialize_model_bound_nodes(nodes, false);
  session_build_apply_derived_input_contracts(&build_nodes);
  BuildResult br =
      build_pipeline_full(build_nodes, insert_boundaries, "mysink", false, name_transform, &opt_);
  return br.pipeline_string;
}

} // namespace simaai::neat
