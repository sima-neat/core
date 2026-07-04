/**
 * @file SessionBuildSource.cpp
 * @brief Graph source-mode build/run methods.
 */

#include "GraphDetail.h"
#include "internal/GraphBuildInternal.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "gst/GstLatestByStreamMux.h"

#include "builder/InputContractConfigurable.h"
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
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

InputContract input_contract_from_fused_ingress_spec(const OutputSpec& spec);

void maybe_compile_source_contracts(BuildResult* build_result,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const GraphOptions& sess_opt, const char* where,
                                    const OutputSpec* ingress_spec = nullptr) {
  ContractCompileInput compile_input;
  compile_input.pipeline_label = where ? where : "Graph::build(source)";
  compile_input.processcvu_requested_run_target = sess_opt.processcvu_requested_run_target;
  compile_input.processcvu = sess_opt.processcvu;
  if (ingress_spec) {
    compile_input.ingress.ingress_spec = *ingress_spec;
    compile_input.ingress.ingress_contract = input_contract_from_fused_ingress_spec(*ingress_spec);
  }
  session_build_compile_contracts(build_result, nodes, compile_input, where, nullptr);
}

InputContract input_contract_from_fused_ingress_spec(const OutputSpec& spec) {
  InputContract out;
  out.payload_type = payload_type_from_media_type(spec.media_type);
  out.media_type = spec.media_type;
  out.format = spec.format;
  out.dtype = spec.dtype;
  out.layout = spec.layout;
  out.width = spec.width;
  out.height = spec.height;
  out.depth = spec.depth;
  return out;
}

OutputSpec fused_ingress_spec_for_contracts(const runtime::FusedRealtimeIngress& ingress) {
  OutputSpec merged;
  for (const auto& branch : ingress.branches) {
    const OutputSpec& spec = branch.output_spec;
    if (merged.media_type.empty()) {
      merged.media_type = spec.media_type;
    }
    if (merged.format.empty()) {
      merged.format = spec.format;
    }
    if (merged.dtype.empty()) {
      merged.dtype = spec.dtype;
    }
    if (merged.layout.empty()) {
      merged.layout = spec.layout;
    }
    merged.width = std::max(merged.width, spec.width);
    merged.height = std::max(merged.height, spec.height);
    merged.depth = std::max(merged.depth, spec.depth);
    if (merged.fps_num <= 0 && spec.fps_num > 0) {
      merged.fps_num = spec.fps_num;
      merged.fps_den = spec.fps_den > 0 ? spec.fps_den : 1;
    }
    if (merged.memory.empty()) {
      merged.memory = spec.memory;
    }
    if (merged.payload_type == PayloadType::Auto && spec.payload_type != PayloadType::Auto) {
      merged.payload_type = spec.payload_type;
    }
  }
  if (merged.payload_type == PayloadType::Auto) {
    merged.payload_type = payload_type_from_media_type(merged.media_type);
  }
  return merged;
}

void apply_fused_ingress_contract_to_nodes(std::vector<std::shared_ptr<Node>>* nodes,
                                           const OutputSpec& ingress_spec, const char* where) {
  if (!nodes || nodes->empty()) {
    return;
  }

  InputContract current = input_contract_from_fused_ingress_spec(ingress_spec);
  for (std::size_t i = 0; i < nodes->size(); ++i) {
    const auto& node = (*nodes)[i];
    if (!node) {
      continue;
    }
    if (auto* configurable = dynamic_cast<InputContractConfigurable*>(node.get())) {
      std::string err;
      configurable->apply_input_contract(current, &err);
      if (!err.empty()) {
        session_build_throw_session_error_simple(
            error_codes::kPipelineShape,
            std::string(where ? where : "Graph::build(fused)") +
                ": failed to configure fused realtime consumer node '" + node->kind() +
                "' from upstream caps: " + err,
            "Check source resolution/format against the downstream node's declared capacity.");
      }
    }
    if (auto* provider = dynamic_cast<OutputSpecProvider*>(node.get())) {
      OutputSpec in_spec;
      in_spec.payload_type = current.payload_type;
      in_spec.media_type = current.media_type;
      in_spec.format = current.format;
      in_spec.dtype = current.dtype;
      in_spec.layout = current.layout;
      in_spec.width = current.width;
      in_spec.height = current.height;
      in_spec.depth = current.depth;
      current = input_contract_from_fused_ingress_spec(provider->output_spec(in_spec));
    }
  }
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
  return up == "INT8" || up == "UINT8" || up == "U8" || up == "EVXX_INT8" || up == "INT16" ||
         up == "UINT16" || up == "EVXX_INT16";
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
  meta.resize_mode = (meta.target_width > 0 && meta.target_height > 0)
                         ? (opt.aspect_ratio ? "letterbox" : "stretch")
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
      buffer, ctx->opt, ctx->guard, ctx->element_name.c_str(), timing.frame_id,
      StreamIdOverride{std::optional<std::string>{std::string{}}},
      BufferNameOverride{std::optional<std::string>{ctx->buffer_name}});
  if (!writable)
    return GST_PAD_PROBE_OK;

  const std::optional<int64_t> frame_opt = timing.frame_id;
  const std::optional<int64_t> input_seq = frame_opt;
  const std::optional<int64_t> orig_input_seq = input_seq;
  const std::optional<std::string> stream_id = std::string{};
  const std::optional<std::string> buffer_name{ctx->buffer_name};
  (void)update_simaai_meta_fields(writable, frame_opt, input_seq, orig_input_seq, stream_id,
                                  buffer_name, timing.pts_ns, buffer_name, std::optional<int>{0});
  (void)write_sample_timing_to_gst_buffer(writable, timing);
  source_meta_maybe_apply_preprocess_template(pad, writable, ctx);

  GST_PAD_PROBE_INFO_DATA(info) = writable;

  if (source_meta_debug_enabled() && !ctx->logged.exchange(true)) {
    std::fprintf(stderr,
                 "[source-meta] attached GstSimaMeta at %s buffer-name=%s preprocess_meta=%d "
                 "\n",
                 ctx->element_name.c_str(), ctx->buffer_name.c_str(),
                 ctx->opt.preprocess_meta.has_value() ? 1 : 0);
    dump_sima_meta_full(writable, ctx->element_name.c_str());
  }
  return GST_PAD_PROBE_OK;
}

void source_sima_meta_probe_destroy(gpointer data) {
  delete static_cast<SourceSimaMetaProbeCtx*>(data);
}

bool source_sima_meta_probe_required_for_node(const Node& node) {
  if (node.memory_contract() == MemoryContract::PreferDeviceZeroCopy) {
    return true;
  }
  const std::string kind = node.kind();
  return kind == "Preproc" || kind == "ModelFragment" || kind == "SimaBoxDecode";
}

bool attach_source_sima_meta_probe_for_node(GstElement* pipeline,
                                            const std::vector<std::shared_ptr<Node>>& logical_nodes,
                                            std::size_t logical_index, int actual_element_index,
                                            const NameTransform& name_transform) {
  if (!pipeline || logical_index >= logical_nodes.size()) {
    return false;
  }
  const auto& node = logical_nodes[logical_index];
  if (!node) {
    return false;
  }
  if (!source_sima_meta_probe_required_for_node(*node)) {
    return false;
  }

  std::vector<std::string> element_names = node->element_names(actual_element_index);
  if (element_names.empty()) {
    return false;
  }
  element_names = apply_name_transform(name_transform, element_names);
  const std::string source_tail = element_names.back();
  std::string buffer_name = node->buffer_name_hint(actual_element_index);
  if (buffer_name.empty()) {
    buffer_name = source_tail;
  }

  GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), source_tail.c_str());
  if (!element) {
    session_build_throw_session_error_simple(
        error_codes::kPipelineShape,
        std::string("source metadata probe could not find element '") + source_tail +
            "' for node " + node->kind(),
        "Ensure device-output node element_names() matches backend_fragment().");
  }
  GstPad* pad = gst_element_get_static_pad(element, "src");
  if (!pad) {
    gst_object_unref(element);
    session_build_throw_session_error_simple(
        error_codes::kPipelineShape,
        std::string("source metadata probe could not find static src pad on '") + source_tail +
            "' for node " + node->kind(),
        "Device-output nodes need a static src pad for metadata stamping.");
  }

  auto* ctx = new SourceSimaMetaProbeCtx();
  ctx->opt = source_meta_input_options_for_node(logical_nodes, logical_index, buffer_name);
  ctx->element_name = source_tail;
  ctx->buffer_name = buffer_name;
  gst_object_unref(element);
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
  return true;
}

void attach_source_sima_meta_probes(GstElement* pipeline,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const NameTransform& name_transform) {
  if (!pipeline)
    return;

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    attach_source_sima_meta_probe_for_node(pipeline, nodes, i, static_cast<int>(i), name_transform);
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
  // Source-mode pipelines own live/producers such as MIPI/libcamera, RTSP, and
  // other self-driven sources.  They must reach NULL before Run::close() returns;
  // otherwise deferred no-flush teardown can race process/plugin destruction
  // after the application has already observed successful outputs.
  stream_opt.prefer_synchronous_teardown = true;
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
  //
  // Public source pipelines keep the normal diagnostic state snapshot wait so
  // build() can surface early RTSP/caps/plugin failures before returning a
  // user-visible stream.  Graph-internal source segments are different: RunCore
  // starts many live producers before the downstream segment and application
  // pull loop are ready to drain them.  Waiting for each source's state snapshot
  // serializes startup and lets already-started sources fill the internal edge.
  // Use a non-blocking snapshot for graph-internal sources; set_state itself is
  // still bounded by the normal state-change timeout and bus errors are still
  // drained immediately after the call.
  const int snapshot_wait_override_ms = public_output_contract ? -1 : 0;
  set_state_or_throw(pipeline.get(), GST_STATE_PAUSED, where, br.diag, snapshot_wait_override_ms);

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

  const int snapshot_wait_override_ms = public_output_contract ? -1 : 0;
  set_state_or_throw(src.pipeline.get(), GST_STATE_PLAYING, where, src.diag,
                     snapshot_wait_override_ms);

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

std::vector<std::shared_ptr<Node>>
fused_materialize_nodes(const std::vector<std::shared_ptr<Node>>& nodes) {
  std::vector<std::shared_ptr<Node>> out =
      session_build_materialize_model_bound_nodes(nodes, false);
  session_build_apply_derived_input_contracts(&out);
  return out;
}

void append_fused_node_fragment(BuildResult* br, std::ostringstream* pipeline,
                                const std::shared_ptr<Node>& node, int actual_index,
                                const NameTransform& name_transform, bool prepend_link,
                                const char* extra_fragment_props = nullptr) {
  if (!br || !br->diag || !pipeline || !node) {
    return;
  }
  if (prepend_link) {
    (*pipeline) << " ! ";
  }
  NodeFragment frag = make_node_fragment(node, actual_index, name_transform);
  NodeReport nr;
  nr.index = actual_index;
  nr.kind = node->kind();
  nr.user_label = node->user_label();
  nr.backend_fragment = frag.fragment;
  if (extra_fragment_props && *extra_fragment_props) {
    nr.backend_fragment += extra_fragment_props;
  }
  nr.elements = frag.element_names;
  br->diag->node_reports.push_back(nr);
  (*pipeline) << nr.backend_fragment;
}

std::string gst_double_quote(std::string value) {
  std::string out;
  out.reserve(value.size() + 2U);
  out.push_back('"');
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::string fused_stream_ids_csv(const runtime::FusedRealtimeIngress& ingress) {
  std::string csv;
  for (std::size_t i = 0; i < ingress.branches.size(); ++i) {
    if (i) {
      csv += ',';
    }
    const std::string fallback = "stream" + std::to_string(i);
    csv += ingress.branches[i].stream_id.empty() ? fallback : ingress.branches[i].stream_id;
  }
  return csv;
}

NameTransform fused_branch_name_transform(const NameTransform& base, std::size_t branch_index) {
  NameTransform out = base;
  // Fused realtime ingress puts several copies of the same source/decode
  // fragment into one GStreamer pipeline.  Session-level name suffixing keeps
  // separate Runs apart, but it is not enough inside this single fused pipeline:
  // explicit node names such as "decoder" would collide.  Add a branch-local
  // suffix before the session suffix so element names and internal buffer names
  // stay unique while the public graph remains unchanged.
  out.suffix = "_b" + std::to_string(branch_index) + base.suffix;
  return out;
}

struct FusedRealtimePadProbeCtx {
  std::string element_name;
  std::string factory_name;
  std::string pad_name;
  std::string direction;
  int limit = 0;
  std::atomic<int> seen{0};
};

std::vector<std::string> fused_realtime_probe_patterns() {
  const std::string raw =
      env_str("SIMA_FUSED_REALTIME_PAD_PROBE_ELEMENTS",
              "latestmux,neatlatest,preproc,processcvu,processmla,mla,boxdecode,appsink");
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    std::string trimmed = trim_copy(cur);
    if (!trimmed.empty()) {
      out.push_back(lower_copy(trimmed));
    }
    cur.clear();
  };
  for (char c : raw) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      flush();
      continue;
    }
    cur.push_back(c);
  }
  flush();
  return out;
}

bool fused_realtime_probe_match(const std::string& haystack,
                                const std::vector<std::string>& patterns) {
  if (haystack.empty()) {
    return false;
  }
  const std::string lower = lower_copy(haystack);
  return std::any_of(patterns.begin(), patterns.end(), [&](const std::string& pattern) {
    return !pattern.empty() && lower.find(pattern) != std::string::npos;
  });
}

GstPadProbeReturn fused_realtime_pad_probe_cb(GstPad* pad, GstPadProbeInfo* info,
                                              gpointer user_data) {
  auto* ctx = reinterpret_cast<FusedRealtimePadProbeCtx*>(user_data);
  if (!ctx || ctx->limit <= 0) {
    return GST_PAD_PROBE_OK;
  }
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  const int count = ctx->seen.fetch_add(1, std::memory_order_relaxed) + 1;
  if (count > ctx->limit) {
    return GST_PAD_PROBE_OK;
  }
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) {
    return GST_PAD_PROBE_OK;
  }

  std::string stream_id;
  gint64 frame_id = -1;
  if (GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
    if (GstStructure* s = gst_custom_meta_get_structure(meta)) {
      if (const char* stream = gst_structure_get_string(s, "orig-stream-id")) {
        stream_id = stream;
      }
      if (stream_id.empty()) {
        if (const char* stream = gst_structure_get_string(s, "stream-id")) {
          stream_id = stream;
        }
      }
      gboolean sample_frame_valid = FALSE;
      if (gst_structure_get_boolean(s, "sample-frame-id-valid", &sample_frame_valid) == TRUE &&
          sample_frame_valid == TRUE) {
        (void)gst_structure_get_int64(s, "sample-frame-id", &frame_id);
      }
      if (frame_id < 0) {
        (void)gst_structure_get_int64(s, "frame-id", &frame_id);
      }
    }
  }

  GstCaps* caps = pad ? gst_pad_get_current_caps(pad) : nullptr;
  gchar* caps_str = caps ? gst_caps_to_string(caps) : nullptr;
  std::fprintf(stderr,
               "[fused-probe] element=%s factory=%s pad=%s dir=%s count=%d size=%" G_GUINT64_FORMAT,
               ctx->element_name.c_str(), ctx->factory_name.c_str(), ctx->pad_name.c_str(),
               ctx->direction.c_str(), count, static_cast<guint64>(gst_buffer_get_size(buffer)));
  if (GST_BUFFER_PTS_IS_VALID(buffer)) {
    std::fprintf(stderr, " pts=%" G_GUINT64_FORMAT, static_cast<guint64>(GST_BUFFER_PTS(buffer)));
  }
  if (!stream_id.empty()) {
    std::fprintf(stderr, " stream=%s", stream_id.c_str());
  }
  if (frame_id >= 0) {
    std::fprintf(stderr, " frame=%" G_GINT64_FORMAT, frame_id);
  }
  if (caps_str) {
    std::fprintf(stderr, " caps=%s", caps_str);
  }
  std::fprintf(stderr, "\n");
  if (caps_str) {
    g_free(caps_str);
  }
  if (caps) {
    gst_caps_unref(caps);
  }
  return GST_PAD_PROBE_OK;
}

void attach_fused_realtime_pad_probes(GstElement* pipeline) {
  if (!pipeline || !env_bool("SIMA_FUSED_REALTIME_PAD_PROBES", false)) {
    return;
  }
  const int limit = std::max(1, env_int("SIMA_FUSED_REALTIME_PAD_PROBE_LIMIT", 5));
  const auto patterns = fused_realtime_probe_patterns();
  if (patterns.empty()) {
    return;
  }

  int attached = 0;
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) {
    return;
  }
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const char* elem_name_c = GST_ELEMENT_NAME(elem);
    const std::string elem_name = elem_name_c ? elem_name_c : "";
    GstElementFactory* factory = gst_element_get_factory(elem);
    const char* factory_name_c =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
    const std::string factory_name = factory_name_c ? factory_name_c : "";
    if (!fused_realtime_probe_match(elem_name, patterns) &&
        !fused_realtime_probe_match(factory_name, patterns)) {
      g_value_reset(&item);
      continue;
    }

    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (!pad_it) {
      g_value_reset(&item);
      continue;
    }
    GValue pad_val = G_VALUE_INIT;
    while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
      GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
      if (pad) {
        const GstPadDirection dir = gst_pad_get_direction(pad);
        if (dir == GST_PAD_SRC || dir == GST_PAD_SINK) {
          const char* pad_name_c = GST_PAD_NAME(pad);
          auto* ctx = new FusedRealtimePadProbeCtx();
          ctx->element_name = elem_name.empty() ? "<unnamed>" : elem_name;
          ctx->factory_name = factory_name.empty() ? "<unknown>" : factory_name;
          ctx->pad_name = pad_name_c ? pad_name_c : "<unknown>";
          ctx->direction = (dir == GST_PAD_SRC) ? "src" : "sink";
          ctx->limit = limit;
          gst_pad_add_probe(
              pad, GST_PAD_PROBE_TYPE_BUFFER, fused_realtime_pad_probe_cb, ctx,
              +[](gpointer p) { delete reinterpret_cast<FusedRealtimePadProbeCtx*>(p); });
          ++attached;
          std::fprintf(stderr, "[fused-probe] armed element=%s factory=%s pad=%s dir=%s limit=%d\n",
                       ctx->element_name.c_str(), ctx->factory_name.c_str(), ctx->pad_name.c_str(),
                       ctx->direction.c_str(), limit);
        }
      }
      g_value_reset(&pad_val);
    }
    g_value_unset(&pad_val);
    gst_iterator_free(pad_it);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  std::fprintf(stderr, "[fused-probe] armed_total=%d\n", attached);
}

enum class FusedStageCounterKind : std::size_t {
  AppsinkSink,
  BoxDecodeSink,
  BoxDecodeSrc,
  MlaSink,
  MlaSrc,
  MuxSink,
  MuxSrc,
  PreprocSink,
  PreprocSrc,
  Count,
};

constexpr std::size_t kFusedStageCounterKindCount =
    static_cast<std::size_t>(FusedStageCounterKind::Count);

const char* fused_stage_counter_kind_name(FusedStageCounterKind kind) {
  switch (kind) {
  case FusedStageCounterKind::AppsinkSink:
    return "appsink_sink";
  case FusedStageCounterKind::BoxDecodeSink:
    return "boxdecode_sink";
  case FusedStageCounterKind::BoxDecodeSrc:
    return "boxdecode_src";
  case FusedStageCounterKind::MlaSink:
    return "mla_sink";
  case FusedStageCounterKind::MlaSrc:
    return "mla_src";
  case FusedStageCounterKind::MuxSink:
    return "mux_sink";
  case FusedStageCounterKind::MuxSrc:
    return "mux_src";
  case FusedStageCounterKind::PreprocSink:
    return "preproc_sink";
  case FusedStageCounterKind::PreprocSrc:
    return "preproc_src";
  case FusedStageCounterKind::Count:
    break;
  }
  return "unknown";
}

struct FusedStageCounterStageData {
  std::vector<std::atomic<std::uint64_t>> by_stream;
  std::atomic<std::uint64_t> total{0};
  std::atomic<std::uint64_t> missing_meta{0};
  std::atomic<std::uint64_t> unknown_stream{0};
};

struct FusedStageCounterState {
  std::array<FusedStageCounterStageData, kFusedStageCounterKindCount> stages;
  std::vector<std::string> expected_streams;
  std::map<std::string, std::size_t> stream_to_index;
  std::atomic<std::uint64_t> total{0};
  std::uint64_t interval = 4000;
  std::mutex print_mu;
};

struct FusedStageCounterProbeCtx {
  std::shared_ptr<FusedStageCounterState> state;
  FusedStageCounterKind stage = FusedStageCounterKind::Count;
  std::string element_name;
  std::string factory_name;
  std::string pad_name;
  std::optional<std::size_t> stream_index_override;
};

std::vector<std::string> fused_stage_counter_patterns() {
  const std::string raw =
      env_str("SIMA_FUSED_STAGE_COUNTER_ELEMENTS",
              "latestmux,neatlatest,processcvu,preproc,processmla,mla,boxdecode,objectdecode,"
              "appsink");
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&]() {
    std::string trimmed = trim_copy(cur);
    if (!trimmed.empty()) {
      out.push_back(lower_copy(trimmed));
    }
    cur.clear();
  };
  for (char c : raw) {
    if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c))) {
      flush();
      continue;
    }
    cur.push_back(c);
  }
  flush();
  return out;
}

std::string fused_stage_counter_stream_id(GstBuffer* buffer) {
  if (!buffer) {
    return {};
  }
  if (GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
    if (GstStructure* s = gst_custom_meta_get_structure(meta)) {
      if (const char* stream = gst_structure_get_string(s, "orig-stream-id")) {
        return stream;
      }
      if (const char* stream = gst_structure_get_string(s, "stream-id")) {
        return stream;
      }
    }
  }
  return {};
}

std::optional<FusedStageCounterKind>
fused_stage_counter_kind_for_pad(const std::string& elem_name, const std::string& factory_name,
                                 GstPadDirection direction, const std::string& pad_name) {
  const std::string elem = lower_copy(elem_name);
  const std::string factory = lower_copy(factory_name);
  const std::string pad = lower_copy(pad_name);
  const bool is_sink = direction == GST_PAD_SINK;
  const bool is_src = direction == GST_PAD_SRC;

  if (factory.find("neatlatestbystreammux") != std::string::npos ||
      elem.find("neat_live_mux") != std::string::npos ||
      elem.find("latestbystreammux") != std::string::npos) {
    if (is_sink && pad.rfind("sink", 0) == 0) {
      return FusedStageCounterKind::MuxSink;
    }
    if (is_src) {
      return FusedStageCounterKind::MuxSrc;
    }
  }

  if (factory.find("processcvu") != std::string::npos ||
      factory.find("preproc") != std::string::npos ||
      elem.find("processcvu") != std::string::npos || elem.find("preproc") != std::string::npos) {
    if (is_sink) {
      return FusedStageCounterKind::PreprocSink;
    }
    if (is_src) {
      return FusedStageCounterKind::PreprocSrc;
    }
  }

  if (factory.find("processmla") != std::string::npos ||
      elem.find("processmla") != std::string::npos) {
    if (is_sink) {
      return FusedStageCounterKind::MlaSink;
    }
    if (is_src) {
      return FusedStageCounterKind::MlaSrc;
    }
  }

  if (factory.find("boxdecode") != std::string::npos ||
      factory.find("objectdecode") != std::string::npos ||
      elem.find("boxdecode") != std::string::npos || elem.find("objectdecode") != std::string::npos) {
    if (is_sink) {
      return FusedStageCounterKind::BoxDecodeSink;
    }
    if (is_src) {
      return FusedStageCounterKind::BoxDecodeSrc;
    }
  }

  if (factory == "appsink" || elem == "mysink" || elem.find("appsink") != std::string::npos) {
    if (is_sink) {
      return FusedStageCounterKind::AppsinkSink;
    }
  }

  return std::nullopt;
}

std::optional<std::size_t> fused_stage_counter_mux_sink_index(const std::string& pad_name) {
  if (pad_name.rfind("sink_", 0) != 0) {
    return std::nullopt;
  }
  std::size_t pos = std::string("sink_").size();
  if (pos >= pad_name.size() || !std::isdigit(static_cast<unsigned char>(pad_name[pos]))) {
    return std::nullopt;
  }
  std::size_t value = 0;
  while (pos < pad_name.size() && std::isdigit(static_cast<unsigned char>(pad_name[pos]))) {
    value = value * 10U + static_cast<std::size_t>(pad_name[pos] - '0');
    ++pos;
  }
  return value;
}

void print_fused_stage_counter_snapshot(const std::shared_ptr<FusedStageCounterState>& state,
                                        const char* reason) {
  if (!state) {
    return;
  }
  std::lock_guard<std::mutex> guard(state->print_mu);
  const std::uint64_t total = state->total.load(std::memory_order_relaxed);
  std::fprintf(stderr,
               "[fused-stage-counters] reason=%s total=%llu stages=%zu expected_streams=%zu\n",
               reason ? reason : "unknown", static_cast<unsigned long long>(total),
               kFusedStageCounterKindCount, state->expected_streams.size());
  for (std::size_t stage_index = 0; stage_index < kFusedStageCounterKindCount; ++stage_index) {
    const auto kind = static_cast<FusedStageCounterKind>(stage_index);
    const FusedStageCounterStageData& data = state->stages[stage_index];
    const std::uint64_t stage_total = data.total.load(std::memory_order_relaxed);
    if (stage_total == 0) {
      continue;
    }
    std::vector<std::pair<std::string, std::uint64_t>> counts;
    counts.reserve(state->expected_streams.size());
    for (std::size_t i = 0; i < state->expected_streams.size(); ++i) {
      std::uint64_t count = 0;
      if (i < data.by_stream.size()) {
        count = data.by_stream[i].load(std::memory_order_relaxed);
      }
      counts.emplace_back(state->expected_streams[i], count);
    }
    std::sort(counts.begin(), counts.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.second != rhs.second) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    });
    std::uint64_t min_count = 0;
    std::uint64_t max_count = 0;
    if (!counts.empty()) {
      min_count = counts.front().second;
      max_count = counts.back().second;
    }
    const std::size_t edge_count = std::min<std::size_t>(8, counts.size());
    std::ostringstream lows;
    for (std::size_t i = 0; i < edge_count; ++i) {
      if (i) {
        lows << ',';
      }
      lows << counts[i].first << ':' << counts[i].second;
    }
    std::ostringstream highs;
    for (std::size_t i = 0; i < edge_count; ++i) {
      const std::size_t idx = counts.size() - 1U - i;
      if (i) {
        highs << ',';
      }
      highs << counts[idx].first << ':' << counts[idx].second;
    }
    std::fprintf(stderr,
                 "[fused-stage-counters] stage=%s total=%llu streams=%zu min=%llu max=%llu "
                 "missing_meta=%llu unknown_stream=%llu lows=%s highs=%s\n",
                 fused_stage_counter_kind_name(kind), static_cast<unsigned long long>(stage_total),
                 counts.size(), static_cast<unsigned long long>(min_count),
                 static_cast<unsigned long long>(max_count),
                 static_cast<unsigned long long>(data.missing_meta.load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(data.unknown_stream.load(std::memory_order_relaxed)),
                 lows.str().c_str(), highs.str().c_str());
  }
  std::fflush(stderr);
}

GstPadProbeReturn fused_stage_counter_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = reinterpret_cast<FusedStageCounterProbeCtx*>(user_data);
  if (!ctx || !ctx->state || ctx->stage == FusedStageCounterKind::Count ||
      (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) {
    return GST_PAD_PROBE_OK;
  }

  const std::size_t stage_index = static_cast<std::size_t>(ctx->stage);
  FusedStageCounterStageData& data = ctx->state->stages[stage_index];
  if (ctx->stream_index_override.has_value()) {
    const std::size_t stream_index = *ctx->stream_index_override;
    if (stream_index < data.by_stream.size()) {
      data.by_stream[stream_index].fetch_add(1, std::memory_order_relaxed);
    } else {
      data.unknown_stream.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    const std::string stream_id = fused_stage_counter_stream_id(buffer);
    if (stream_id.empty()) {
      data.missing_meta.fetch_add(1, std::memory_order_relaxed);
    } else {
      const auto stream_it = ctx->state->stream_to_index.find(stream_id);
      if (stream_it == ctx->state->stream_to_index.end()) {
        data.unknown_stream.fetch_add(1, std::memory_order_relaxed);
      } else if (stream_it->second < data.by_stream.size()) {
        data.by_stream[stream_it->second].fetch_add(1, std::memory_order_relaxed);
      } else {
        data.unknown_stream.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  data.total.fetch_add(1, std::memory_order_relaxed);
  const std::uint64_t total = ctx->state->total.fetch_add(1, std::memory_order_relaxed) + 1U;
  if (ctx->state->interval > 0 && total % ctx->state->interval == 0) {
    print_fused_stage_counter_snapshot(ctx->state, "periodic");
  }
  return GST_PAD_PROBE_OK;
}

void attach_fused_stage_counter_probe(GstPad* pad,
                                      const std::shared_ptr<FusedStageCounterState>& state,
                                      FusedStageCounterKind stage,
                                      const std::string& element_name,
                                      const std::string& factory_name,
                                      std::optional<std::size_t> stream_index_override = {}) {
  if (!pad || !state || stage == FusedStageCounterKind::Count) {
    return;
  }
  auto* ctx = new FusedStageCounterProbeCtx();
  ctx->state = state;
  ctx->stage = stage;
  ctx->element_name = element_name.empty() ? "<unnamed>" : element_name;
  ctx->factory_name = factory_name.empty() ? "<unknown>" : factory_name;
  const char* pad_name_c = GST_PAD_NAME(pad);
  ctx->pad_name = pad_name_c ? pad_name_c : "<unknown>";
  ctx->stream_index_override = stream_index_override;
  gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER, fused_stage_counter_probe_cb, ctx,
      +[](gpointer p) { delete reinterpret_cast<FusedStageCounterProbeCtx*>(p); });
  if (env_bool("SIMA_FUSED_STAGE_COUNTERS_DEBUG", false)) {
    const char* stream = "";
    std::string stream_storage;
    if (stream_index_override.has_value() && *stream_index_override < state->expected_streams.size()) {
      stream_storage = state->expected_streams[*stream_index_override];
      stream = stream_storage.c_str();
    }
    std::fprintf(stderr,
                 "[fused-stage-counters] armed stage=%s element=%s factory=%s pad=%s stream=%s\n",
                 fused_stage_counter_kind_name(stage), ctx->element_name.c_str(),
                 ctx->factory_name.c_str(), ctx->pad_name.c_str(), stream);
  }
}

void attach_fused_realtime_stage_counters(GstElement* pipeline,
                                          const runtime::FusedRealtimeIngress& ingress) {
  if (!pipeline || !env_bool("SIMA_FUSED_STAGE_COUNTERS", false)) {
    return;
  }
  const auto patterns = fused_stage_counter_patterns();
  if (patterns.empty()) {
    return;
  }
  auto state = std::make_shared<FusedStageCounterState>();
  state->interval =
      static_cast<std::uint64_t>(std::max(1, env_int("SIMA_FUSED_STAGE_COUNTERS_EVERY", 4000)));
  state->expected_streams.reserve(ingress.branches.size());
  for (std::size_t i = 0; i < ingress.branches.size(); ++i) {
    const std::string fallback = "stream" + std::to_string(i);
    state->expected_streams.push_back(
        ingress.branches[i].stream_id.empty() ? fallback : ingress.branches[i].stream_id);
  }
  for (std::size_t i = 0; i < state->expected_streams.size(); ++i) {
    state->stream_to_index[state->expected_streams[i]] = i;
  }
  for (auto& stage : state->stages) {
    stage.by_stream = std::vector<std::atomic<std::uint64_t>>(state->expected_streams.size());
  }
  if (env_bool("SIMA_FUSED_STAGE_COUNTERS_DEBUG", false)) {
    for (std::size_t i = 0; i < state->expected_streams.size(); ++i) {
      std::fprintf(stderr, "[fused-stage-counters] stream_map branch=%zu stream=%s\n", i,
                   state->expected_streams[i].c_str());
    }
  }

  int attached = 0;
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) {
    return;
  }
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const char* elem_name_c = GST_ELEMENT_NAME(elem);
    const std::string elem_name = elem_name_c ? elem_name_c : "";
    GstElementFactory* factory = gst_element_get_factory(elem);
    const char* factory_name_c =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
    const std::string factory_name = factory_name_c ? factory_name_c : "";
    if (!fused_realtime_probe_match(elem_name, patterns) &&
        !fused_realtime_probe_match(factory_name, patterns)) {
      g_value_reset(&item);
      continue;
    }

    GstIterator* pad_it = gst_element_iterate_pads(elem);
    if (!pad_it) {
      g_value_reset(&item);
      continue;
    }
    GValue pad_val = G_VALUE_INIT;
    while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
      GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
      if (pad) {
        const GstPadDirection dir = gst_pad_get_direction(pad);
        const char* pad_name_c = GST_PAD_NAME(pad);
        const std::string pad_name = pad_name_c ? pad_name_c : "";
        const auto stage = fused_stage_counter_kind_for_pad(elem_name, factory_name, dir, pad_name);
        if (stage.has_value()) {
          std::optional<std::size_t> stream_index_override;
          if (*stage == FusedStageCounterKind::MuxSink) {
            const auto mux_index = fused_stage_counter_mux_sink_index(pad_name);
            if (mux_index.has_value() && *mux_index < state->expected_streams.size()) {
              stream_index_override = *mux_index;
            }
          }
          attach_fused_stage_counter_probe(pad, state, *stage, elem_name, factory_name,
                                           stream_index_override);
          ++attached;
        }
      }
      g_value_reset(&pad_val);
    }
    g_value_unset(&pad_val);
    gst_iterator_free(pad_it);
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  std::fprintf(stderr, "[fused-stage-counters] armed_total=%d expected_streams=%zu interval=%llu\n",
               attached, state->expected_streams.size(),
               static_cast<unsigned long long>(state->interval));
  std::fflush(stderr);
}

enum class FusedBranchCounterStage {
  RtpCaps,
  H264Caps,
  Decoder,
  RawCaps,
  QueuePreSink,
  QueuePre,
  QueuePostSink,
  QueuePost,
  MuxSink,
};

const char* fused_branch_counter_stage_name(FusedBranchCounterStage stage) {
  switch (stage) {
  case FusedBranchCounterStage::RtpCaps:
    return "rtp";
  case FusedBranchCounterStage::H264Caps:
    return "h264";
  case FusedBranchCounterStage::Decoder:
    return "decoder";
  case FusedBranchCounterStage::RawCaps:
    return "raw";
  case FusedBranchCounterStage::QueuePreSink:
    return "queue_pre_sink";
  case FusedBranchCounterStage::QueuePre:
    return "queue_pre";
  case FusedBranchCounterStage::QueuePostSink:
    return "queue_post_sink";
  case FusedBranchCounterStage::QueuePost:
    return "queue_post";
  case FusedBranchCounterStage::MuxSink:
    return "mux";
  }
  return "unknown";
}

struct FusedBranchCounters {
  std::atomic<std::uint64_t> rtp_caps{0};
  std::atomic<std::uint64_t> h264_caps{0};
  std::atomic<std::uint64_t> decoder{0};
  std::atomic<std::uint64_t> raw_caps{0};
  std::atomic<std::uint64_t> queue_pre_sink{0};
  std::atomic<std::uint64_t> queue_pre{0};
  std::atomic<std::uint64_t> queue_post_sink{0};
  std::atomic<std::uint64_t> queue_post{0};
  std::atomic<std::uint64_t> mux_sink{0};
};

struct FusedBranchCounterState {
  std::vector<std::unique_ptr<FusedBranchCounters>> branches;
  std::atomic<std::uint64_t> total{0};
  std::uint64_t interval = 2000;
  std::mutex print_mu;
};

struct FusedBranchCounterProbeCtx {
  std::shared_ptr<FusedBranchCounterState> state;
  std::size_t branch = 0;
  FusedBranchCounterStage stage = FusedBranchCounterStage::RtpCaps;
};

std::optional<std::size_t> parse_unsigned_suffix_after(const std::string& text, std::size_t pos) {
  if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
    return std::nullopt;
  }
  std::size_t value = 0;
  while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
    value = value * 10U + static_cast<std::size_t>(text[pos] - '0');
    ++pos;
  }
  return value;
}

std::optional<std::size_t> parse_fused_branch_index_from_name(const std::string& name) {
  std::size_t search = 0;
  while (search < name.size()) {
    const std::size_t pos = name.find("_b", search);
    if (pos == std::string::npos) {
      return std::nullopt;
    }
    if (auto parsed = parse_unsigned_suffix_after(name, pos + 2U); parsed.has_value()) {
      return parsed;
    }
    search = pos + 2U;
  }
  if (name.rfind("decoder_b", 0) == 0) {
    return parse_unsigned_suffix_after(name, std::string("decoder_b").size());
  }
  return std::nullopt;
}

std::optional<std::size_t> parse_mux_sink_branch_index(const std::string& pad_name) {
  if (pad_name.rfind("sink_", 0) != 0) {
    return std::nullopt;
  }
  return parse_unsigned_suffix_after(pad_name, std::string("sink_").size());
}

std::optional<std::size_t> parse_leading_node_index(const std::string& name) {
  if (name.empty() || name.front() != 'n') {
    return std::nullopt;
  }
  return parse_unsigned_suffix_after(name, 1U);
}

std::atomic<std::uint64_t>* fused_branch_counter_for_stage(FusedBranchCounters* counters,
                                                           FusedBranchCounterStage stage) {
  if (!counters) {
    return nullptr;
  }
  switch (stage) {
  case FusedBranchCounterStage::RtpCaps:
    return &counters->rtp_caps;
  case FusedBranchCounterStage::H264Caps:
    return &counters->h264_caps;
  case FusedBranchCounterStage::Decoder:
    return &counters->decoder;
  case FusedBranchCounterStage::RawCaps:
    return &counters->raw_caps;
  case FusedBranchCounterStage::QueuePreSink:
    return &counters->queue_pre_sink;
  case FusedBranchCounterStage::QueuePre:
    return &counters->queue_pre;
  case FusedBranchCounterStage::QueuePostSink:
    return &counters->queue_post_sink;
  case FusedBranchCounterStage::QueuePost:
    return &counters->queue_post;
  case FusedBranchCounterStage::MuxSink:
    return &counters->mux_sink;
  }
  return nullptr;
}

void print_fused_branch_counter_snapshot(const std::shared_ptr<FusedBranchCounterState>& state,
                                         const char* reason) {
  if (!state) {
    return;
  }
  std::lock_guard<std::mutex> guard(state->print_mu);
  const std::uint64_t total = state->total.load(std::memory_order_relaxed);
  std::fprintf(stderr, "[fused-branch-counters] reason=%s branches=%zu total=%llu\n",
               reason ? reason : "unknown", state->branches.size(),
               static_cast<unsigned long long>(total));
  for (std::size_t i = 0; i < state->branches.size(); ++i) {
    const auto* c = state->branches[i].get();
    if (!c) {
      continue;
    }
    std::fprintf(
        stderr,
        "[fused-branch-counters] branch=%zu rtp=%llu h264=%llu decoder=%llu raw=%llu "
        "queue_pre_sink=%llu queue_pre=%llu queue_post_sink=%llu queue_post=%llu "
        "mux=%llu\n",
        i, static_cast<unsigned long long>(c->rtp_caps.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->h264_caps.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->decoder.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->raw_caps.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->queue_pre_sink.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->queue_pre.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->queue_post_sink.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->queue_post.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(c->mux_sink.load(std::memory_order_relaxed)));
  }
  std::fflush(stderr);
}

GstPadProbeReturn fused_branch_counter_probe_cb(GstPad*, GstPadProbeInfo* info,
                                                gpointer user_data) {
  auto* ctx = reinterpret_cast<FusedBranchCounterProbeCtx*>(user_data);
  if (!ctx || !ctx->state || (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  if (ctx->branch >= ctx->state->branches.size()) {
    return GST_PAD_PROBE_OK;
  }
  auto* counter =
      fused_branch_counter_for_stage(ctx->state->branches[ctx->branch].get(), ctx->stage);
  if (counter) {
    counter->fetch_add(1, std::memory_order_relaxed);
  }
  const std::uint64_t total = ctx->state->total.fetch_add(1, std::memory_order_relaxed) + 1U;
  if (ctx->state->interval > 0 && total % ctx->state->interval == 0) {
    print_fused_branch_counter_snapshot(ctx->state, "periodic");
  }
  return GST_PAD_PROBE_OK;
}

std::optional<FusedBranchCounterStage>
classify_fused_branch_counter_element(const std::string& elem_name,
                                      const std::string& factory_name) {
  if (elem_name.find("_rtp_caps_b") != std::string::npos) {
    return FusedBranchCounterStage::RtpCaps;
  }
  if (elem_name.find("_h264_caps_b") != std::string::npos) {
    return FusedBranchCounterStage::H264Caps;
  }
  if (elem_name.rfind("decoder_b", 0) == 0) {
    return FusedBranchCounterStage::Decoder;
  }
  if (elem_name.find("_caps_b") != std::string::npos) {
    return FusedBranchCounterStage::RawCaps;
  }
  (void)factory_name;
  return std::nullopt;
}

void attach_fused_branch_counter_probe(GstPad* pad,
                                       const std::shared_ptr<FusedBranchCounterState>& state,
                                       std::size_t branch, FusedBranchCounterStage stage,
                                       const std::string& element_name) {
  if (!pad || !state || branch >= state->branches.size()) {
    return;
  }
  auto* ctx = new FusedBranchCounterProbeCtx();
  ctx->state = state;
  ctx->branch = branch;
  ctx->stage = stage;
  gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER, fused_branch_counter_probe_cb, ctx,
      +[](gpointer p) { delete reinterpret_cast<FusedBranchCounterProbeCtx*>(p); });
  if (env_bool("SIMA_FUSED_BRANCH_COUNTERS_DEBUG", false)) {
    std::fprintf(stderr, "[fused-branch-counters] armed branch=%zu stage=%s element=%s pad=%s\n",
                 branch, fused_branch_counter_stage_name(stage),
                 element_name.empty() ? "<unknown>" : element_name.c_str(), GST_PAD_NAME(pad));
  }
}

void attach_fused_realtime_branch_counters(GstElement* pipeline, std::size_t branch_count) {
  if (!pipeline || branch_count == 0 || !env_bool("SIMA_FUSED_BRANCH_COUNTERS", false)) {
    return;
  }
  auto state = std::make_shared<FusedBranchCounterState>();
  state->interval =
      static_cast<std::uint64_t>(std::max(1, env_int("SIMA_FUSED_BRANCH_COUNTERS_EVERY", 2000)));
  state->branches.reserve(branch_count);
  for (std::size_t i = 0; i < branch_count; ++i) {
    state->branches.push_back(std::make_unique<FusedBranchCounters>());
  }

  std::vector<std::vector<std::size_t>> queue_node_indices(branch_count);
  {
    GstIterator* queue_it = gst_bin_iterate_elements(GST_BIN(pipeline));
    if (queue_it) {
      GValue item = G_VALUE_INIT;
      while (gst_iterator_next(queue_it, &item) == GST_ITERATOR_OK) {
        GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
        if (!elem) {
          g_value_reset(&item);
          continue;
        }
        GstElementFactory* factory = gst_element_get_factory(elem);
        const char* factory_name_c =
            factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
        const std::string factory_name = factory_name_c ? factory_name_c : "";
        const char* elem_name_c = GST_ELEMENT_NAME(elem);
        const std::string elem_name = elem_name_c ? elem_name_c : "";
        const auto branch = parse_fused_branch_index_from_name(elem_name);
        const auto node_index = parse_leading_node_index(elem_name);
        if (factory_name == "queue" && branch.has_value() && *branch < branch_count &&
            node_index.has_value()) {
          queue_node_indices[*branch].push_back(*node_index);
        }
        g_value_reset(&item);
      }
      g_value_unset(&item);
      gst_iterator_free(queue_it);
    }
    for (auto& indices : queue_node_indices) {
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }
  }

  int attached = 0;
  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) {
    return;
  }
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* elem = GST_ELEMENT(g_value_get_object(&item));
    if (!elem) {
      g_value_reset(&item);
      continue;
    }
    const char* elem_name_c = GST_ELEMENT_NAME(elem);
    const std::string elem_name = elem_name_c ? elem_name_c : "";
    GstElementFactory* factory = gst_element_get_factory(elem);
    const char* factory_name_c =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
    const std::string factory_name = factory_name_c ? factory_name_c : "";

    if (elem_name.find("neat_live_mux") != std::string::npos) {
      GstIterator* pad_it = gst_element_iterate_sink_pads(elem);
      if (pad_it) {
        GValue pad_val = G_VALUE_INIT;
        while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
          GstPad* pad = GST_PAD(g_value_get_object(&pad_val));
          const char* pad_name_c = pad ? GST_PAD_NAME(pad) : nullptr;
          const auto branch =
              parse_mux_sink_branch_index(pad_name_c ? std::string(pad_name_c) : "");
          if (pad && branch.has_value() && *branch < branch_count) {
            attach_fused_branch_counter_probe(pad, state, *branch, FusedBranchCounterStage::MuxSink,
                                              elem_name);
            ++attached;
          }
          g_value_reset(&pad_val);
        }
        g_value_unset(&pad_val);
        gst_iterator_free(pad_it);
      }
      g_value_reset(&item);
      continue;
    }

    const auto branch = parse_fused_branch_index_from_name(elem_name);
    std::optional<FusedBranchCounterStage> stage;
    std::optional<FusedBranchCounterStage> sink_stage;
    if (factory_name == "queue" && branch.has_value() && *branch < branch_count) {
      const auto node_index = parse_leading_node_index(elem_name);
      const auto& indices = queue_node_indices[*branch];
      if (node_index.has_value() && !indices.empty()) {
        const bool is_pre_queue = (*node_index == indices.front());
        stage =
            is_pre_queue ? FusedBranchCounterStage::QueuePre : FusedBranchCounterStage::QueuePost;
        sink_stage = is_pre_queue ? FusedBranchCounterStage::QueuePreSink
                                  : FusedBranchCounterStage::QueuePostSink;
      }
    } else {
      stage = classify_fused_branch_counter_element(elem_name, factory_name);
    }
    if (stage.has_value() && branch.has_value() && *branch < branch_count) {
      if (sink_stage.has_value()) {
        GstPad* sink = gst_element_get_static_pad(elem, "sink");
        if (sink) {
          attach_fused_branch_counter_probe(sink, state, *branch, *sink_stage, elem_name);
          ++attached;
          gst_object_unref(sink);
        }
      }
      GstPad* src = gst_element_get_static_pad(elem, "src");
      if (src) {
        attach_fused_branch_counter_probe(src, state, *branch, *stage, elem_name);
        ++attached;
        gst_object_unref(src);
      }
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  std::fprintf(stderr, "[fused-branch-counters] armed_total=%d branches=%zu interval=%llu\n",
               attached, branch_count, static_cast<unsigned long long>(state->interval));
  std::fflush(stderr);
}

GstPadProbeReturn fused_realtime_loan_release_probe_cb(GstPad*, GstPadProbeInfo* info, gpointer) {
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer) {
    (void)pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(buffer);
  }
  return GST_PAD_PROBE_OK;
}

void attach_fused_realtime_loan_release_probe(GstElement* pipeline,
                                              const std::string& appsink_name) {
  if (!pipeline || appsink_name.empty()) {
    return;
  }
  GstElement* raw_sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
  if (!raw_sink) {
    return;
  }
  GstPad* sink_pad = gst_element_get_static_pad(raw_sink, "sink");
  if (sink_pad) {
    gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, fused_realtime_loan_release_probe_cb,
                      nullptr, nullptr);
    if (env_bool("SIMA_LATEST_MUX_LOAN_DEBUG", false)) {
      std::fprintf(stderr, "[latestmux][loan] armed terminal release probe appsink=%s\n",
                   appsink_name.c_str());
    }
    gst_object_unref(sink_pad);
  }
  gst_object_unref(raw_sink);
}

BuildResult build_fused_realtime_source_pipeline(
    const runtime::FusedRealtimeIngress& ingress,
    const std::vector<std::shared_ptr<Node>>& consumer_nodes,
    const std::vector<std::vector<std::shared_ptr<Node>>>& branch_nodes,
    const NameTransform& name_transform, const std::string& appsink_name) {
  BuildResult br;
  br.diag = std::make_shared<DiagCtx>();
  br.appsink_name = apply_name_transform(name_transform, appsink_name);
  br.name_transform = name_transform;

  const std::string mux_name = apply_name_transform(name_transform, "neat_live_mux");
  br.diag->node_reports.reserve(consumer_nodes.size() + 1U);
  NodeReport mux_report;
  mux_report.index = -1;
  mux_report.kind = "RealtimeLatestMux";
  const std::string stream_ids = fused_stream_ids_csv(ingress);
  mux_report.backend_fragment =
      "neatlatestbystreammux name=" + mux_name + " stream-ids=" + gst_double_quote(stream_ids);
  mux_report.elements.push_back(mux_name);
  br.diag->node_reports.push_back(mux_report);

  std::ostringstream ss;
  ss << "neatlatestbystreammux name=" << mux_name << " stream-ids=" << gst_double_quote(stream_ids);
  int actual_index = 0;
  for (const auto& node : consumer_nodes) {
    append_fused_node_fragment(&br, &ss, node, actual_index++, name_transform,
                               /*prepend_link=*/true);
  }

  for (std::size_t branch_index = 0; branch_index < branch_nodes.size(); ++branch_index) {
    const auto& nodes = branch_nodes[branch_index];
    if (nodes.empty()) {
      continue;
    }
    const NameTransform branch_transform =
        fused_branch_name_transform(name_transform, branch_index);
    ss << ' ';
    bool first = true;
    bool saw_branch_queue = false;
    const bool queue_pre_leaky_downstream =
        env_bool("SIMA_FUSED_REALTIME_QUEUE_PRE_LEAKY_DOWNSTREAM", false);
    for (const auto& node : nodes) {
      const bool is_queue = node && node->kind() == "Queue";
      const bool force_pre_queue_leaky =
          queue_pre_leaky_downstream && is_queue && !saw_branch_queue;
      append_fused_node_fragment(&br, &ss, node, actual_index++, branch_transform,
                                 /*prepend_link=*/!first,
                                 force_pre_queue_leaky ? " leaky=downstream" : nullptr);
      if (is_queue) {
        saw_branch_queue = true;
      }
      first = false;
    }
    if (!env_bool("SIMA_FUSED_REALTIME_BYPASS_INGRESS_QUEUE", false)) {
      const int queue_depth = std::max(1, env_int("SIMA_FUSED_REALTIME_INGRESS_QUEUE_DEPTH", 1));
      ss << " ! queue max-size-buffers=" << queue_depth
         << " max-size-bytes=0 max-size-time=0 leaky=downstream";
    }
    ss << " ! " << mux_name << ".sink_" << branch_index;
  }

  br.diag->pipeline_string = ss.str();
  br.pipeline_string = br.diag->pipeline_string;
  return br;
}

SourceStreamBuildContext session_build_fused_realtime_source_stream_internal(
    const runtime::FusedRealtimeIngress& ingress,
    const std::vector<std::shared_ptr<Node>>& consumer_nodes, const std::shared_ptr<void>& guard,
    std::string& last_pipeline, const GraphOptions& sess_opt, const RunOptions& opt, RunMode mode,
    bool require_sink, bool public_output_contract, const char* where) {
  gst_init_once();
  const pipeline_internal::ScopedBuildTiming timing(
      "prepare_fused_realtime_source_pipeline",
      std::string("where=") + (where ? where : "Graph::build(fused)") +
          " branches=" + std::to_string(ingress.branches.size()) +
          " consumer_nodes=" + std::to_string(consumer_nodes.size()));

  require_element("neatlatestbystreammux", where);
  std::vector<std::shared_ptr<Node>> build_consumer_nodes = fused_materialize_nodes(consumer_nodes);
  if (require_sink) {
    enforce_sink_last(build_consumer_nodes);
  } else {
    enforce_sink_last_if_present(build_consumer_nodes, where);
  }
  const bool has_sink =
      require_sink ? true : session_build_has_output_appsink(build_consumer_nodes);
  if (has_sink) {
    require_element("appsink", where);
    require_element("identity", where);
  }

  std::vector<std::vector<std::shared_ptr<Node>>> build_branch_nodes;
  build_branch_nodes.reserve(ingress.branches.size());
  for (const auto& branch : ingress.branches) {
    std::vector<std::shared_ptr<Node>> nodes = fused_materialize_nodes(branch.nodes);
    enforce_caps_behavior(nodes, where);
    enforce_source_run_mode(nodes, where);
    enforce_sink_last_if_present(nodes, where);
    build_branch_nodes.push_back(std::move(nodes));
  }
  enforce_caps_behavior(build_consumer_nodes, where);
  const OutputSpec fused_ingress_spec = fused_ingress_spec_for_contracts(ingress);
  apply_fused_ingress_contract_to_nodes(&build_consumer_nodes, fused_ingress_spec, where);

  const RunOptions requested_opt = session_build_resolve_build_opt(mode, opt);
  RunOptions merged_opt = session_build_apply_run_defaults(requested_opt, sess_opt);
  InputStreamOptions stream_opt = session_build_make_stream_options(merged_opt, mode);
  stream_opt.public_output_contract = public_output_contract;
  stream_opt.prefer_synchronous_teardown = true;
  session_build_maybe_enable_rtsp_appsink_drop(stream_opt, build_consumer_nodes);

  const NameTransform name_transform = make_name_transform(sess_opt);
  std::vector<NameTransform> branch_name_transforms;
  branch_name_transforms.reserve(build_branch_nodes.size());
  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    branch_name_transforms.push_back(fused_branch_name_transform(name_transform, branch_index));
  }
  BuildResult br = build_fused_realtime_source_pipeline(
      ingress, build_consumer_nodes, build_branch_nodes, name_transform, "mysink");
  maybe_compile_source_contracts(&br, build_consumer_nodes, sess_opt, where, &fused_ingress_spec);
  if (has_sink && stream_opt.public_output_contract && br.rendered_manifest.has_value()) {
    const auto endpoint = session_build_public_output_endpoint_selector(build_consumer_nodes);
    std::string error;
    auto override = build_public_terminal_output_override_with_fallback(*br.rendered_manifest,
                                                                        endpoint, &error);
    if (override.has_value()) {
      stream_opt.output_override = std::move(*override);
    }
  }
  session_build_finalize_public_zero_copy_holder_loan_credits(stream_opt);
  last_pipeline = br.pipeline_string;
  session_build_enforce_mla_num_buffers(last_pipeline, where);
  session_build_maybe_dump_pipeline_string(last_pipeline, where);

  pipeline_internal::GstPipelinePtr pipeline(session_build_parse_pipeline_or_throw(br, where));
  session_build_dump_pipeline_element_properties(pipeline.get());
  if (env_bool("SIMA_GST_ENFORCE_NAMES", false)) {
    enforce_names_contract(pipeline.get(), br);
  }
  attach_boundary_probes(pipeline.get(), br.diag);
  attach_stage_timing_probes(pipeline.get(), br.diag, stream_opt.enable_timings);
  attach_element_timing_probes(pipeline.get(), br.diag, stream_opt.enable_timings);
  attach_element_flow_probes(pipeline.get(), br.diag);
  attach_fused_realtime_branch_counters(pipeline.get(), ingress.branches.size());
  attach_fused_realtime_stage_counters(pipeline.get(), ingress);

  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    std::vector<std::shared_ptr<Node>> logical = build_branch_nodes[branch_index];
    logical.insert(logical.end(), build_consumer_nodes.begin(), build_consumer_nodes.end());
    int actual_index = static_cast<int>(build_consumer_nodes.size());
    for (std::size_t prior = 0; prior < branch_index; ++prior) {
      actual_index += static_cast<int>(build_branch_nodes[prior].size());
    }
    for (std::size_t i = 0; i < build_branch_nodes[branch_index].size(); ++i) {
      attach_source_sima_meta_probe_for_node(pipeline.get(), logical, i, actual_index++,
                                             branch_name_transforms[branch_index]);
    }
  }
  for (std::size_t i = 0; i < build_consumer_nodes.size(); ++i) {
    attach_source_sima_meta_probe_for_node(pipeline.get(), build_consumer_nodes, i,
                                           static_cast<int>(i), name_transform);
  }
  session_build_attach_debug_detess_input_probes(pipeline.get());
  session_build_attach_debug_appsink_probes(pipeline.get());
  session_build_attach_debug_all_buffer_probes(pipeline.get());
  session_build_attach_debug_element_buffer_probes(pipeline.get());
  session_build_attach_boxdecode_debug_probes(pipeline.get());
  attach_fused_realtime_pad_probes(pipeline.get());
  if (has_sink) {
    attach_fused_realtime_loan_release_probe(pipeline.get(), br.appsink_name);
  }
  session_build_attach_h264_caps_fixups(pipeline.get(), build_consumer_nodes, name_transform);
  session_build_attach_rtsp_debug(pipeline.get(), build_consumer_nodes, name_transform);
  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    session_build_attach_h264_caps_fixups(pipeline.get(), build_branch_nodes[branch_index],
                                          branch_name_transforms[branch_index]);
    session_build_attach_rtsp_debug(pipeline.get(), build_branch_nodes[branch_index],
                                    branch_name_transforms[branch_index]);
  }

  pipeline_internal::GstSinkPtr sink;
  if (has_sink) {
    const std::string appsink_name = apply_name_transform(name_transform, "mysink");
    GstElement* raw_sink = gst_bin_get_by_name(GST_BIN(pipeline.get()), appsink_name.c_str());
    if (!raw_sink) {
      maybe_dump_dot(pipeline.get(), "build_missing_mysink_fused");
      session_build_throw_session_error_simple(
          error_codes::kPipelineShape,
          std::string(where ? where : "Graph::build(fused)") + ": appsink '" + appsink_name +
              "' not found.\nPipeline:\n" + last_pipeline,
          "Add Output() as the last consumer node.", last_pipeline);
    }
    sink.reset(raw_sink);
    session_build_configure_appsink_for_input_stream(sink.get(), stream_opt);
    session_build_configure_appsink_allocation_preference(sink.get(), build_consumer_nodes);
  }

  const int snapshot_wait_override_ms = public_output_contract ? -1 : 0;
  set_state_or_throw(pipeline.get(), GST_STATE_PAUSED, where, br.diag, snapshot_wait_override_ms);
  set_state_or_throw(pipeline.get(), GST_STATE_PLAYING, where, br.diag, snapshot_wait_override_ms);

  SampleSpec spec = session_build_make_placeholder_spec();
  InputOptions src_opt = session_build_resolve_appsrc_options(InputOptions{}, name_transform);
  GstElement* pipeline_raw = pipeline.release();
  GstElement* sink_raw = sink.release();
  InputStream stream = InputStream::create(pipeline_raw, nullptr, sink_raw, spec, src_opt,
                                           stream_opt, br.diag, guard);
  return SourceStreamBuildContext{
      /*stream=*/std::move(stream),
      /*merged_opt=*/std::move(merged_opt),
      /*stream_opt=*/std::move(stream_opt),
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
          << " terminal_output=" << (segment.boundary.terminal_output ? "true" : "false");
      if (segment.fused_realtime_ingress.has_value()) {
        oss << " fused_realtime_branches=" << segment.fused_realtime_ingress->branches.size();
      }
      if (segment.consumed_by_fused_realtime_ingress) {
        oss << " consumed_by_fused_realtime=true";
      }
      oss << "\n";
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
