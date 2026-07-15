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
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "pipeline/internal/BuildTiming.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/RealtimeLinkOptions.h"
#include "pipeline/internal/SimaaiGstCompat.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gst/video/video.h>

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
                                    const OutputSpec* ingress_spec = nullptr,
                                    const std::vector<int>* node_indices = nullptr) {
  ContractCompileInput compile_input;
  compile_input.pipeline_label = where ? where : "Graph::build(source)";
  compile_input.processcvu_requested_run_target = sess_opt.processcvu_requested_run_target;
  compile_input.processcvu = sess_opt.processcvu;
  if (node_indices) {
    compile_input.node_indices = *node_indices;
  }
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

const char* payload_type_label(PayloadType payload_type) {
  switch (payload_type) {
  case PayloadType::Image:
    return "Image";
  case PayloadType::Tensor:
    return "Tensor";
  case PayloadType::Encoded:
    return "Encoded";
  case PayloadType::Auto:
  default:
    return "Auto";
  }
}

bool known_memory_token(const std::string& value) {
  return !value.empty() && value != "Unknown";
}

[[noreturn]] void throw_fused_realtime_caps_mismatch(const char* field, std::size_t lhs_branch,
                                                     const std::string& lhs, std::size_t rhs_branch,
                                                     const std::string& rhs) {
  std::ostringstream ss;
  ss << "Graph::build(fused realtime ingress): branch output caps differ for " << field
     << " (branch " << lhs_branch << " has '" << lhs << "', branch " << rhs_branch << " has '"
     << rhs
     << "'). Realtime fused fan-in requires homogeneous media/format contracts; normalize each "
        "branch before the fan-in or use explicit non-fused routing.";
  throw std::runtime_error(ss.str());
}

void validate_fused_realtime_ingress_caps(const runtime::FusedRealtimeIngress& ingress) {
  struct StringField {
    const char* name = "";
    std::string value;
    std::size_t branch = 0;
    bool set = false;
    bool ignore_unknown_memory = false;
  };

  StringField media{.name = "media_type"};
  StringField format{.name = "format"};
  StringField dtype{.name = "dtype"};
  StringField layout{.name = "layout"};
  StringField memory{.name = "memory", .ignore_unknown_memory = true};
  PayloadType payload_type = PayloadType::Auto;
  std::size_t payload_branch = 0;

  struct IntField {
    const char* name = "";
    int value = -1;
    std::size_t branch = 0;
    bool set = false;
  };
  IntField width{.name = "width"};
  IntField height{.name = "height"};
  IntField depth{.name = "depth"};
  int fps_num = 0;
  int fps_den = 1;
  std::size_t fps_branch = 0;

  const auto update = [](StringField* field, std::size_t branch, const std::string& value) {
    if (!field || value.empty()) {
      return;
    }
    if (field->ignore_unknown_memory && !known_memory_token(value)) {
      return;
    }
    if (!field->set) {
      field->value = value;
      field->branch = branch;
      field->set = true;
      return;
    }
    if (field->value != value) {
      throw_fused_realtime_caps_mismatch(field->name, field->branch, field->value, branch, value);
    }
  };
  const auto update_int = [](IntField* field, std::size_t branch, int value) {
    if (!field || value <= 0) {
      return;
    }
    if (!field->set) {
      field->value = value;
      field->branch = branch;
      field->set = true;
      return;
    }
    if (field->value != value) {
      throw_fused_realtime_caps_mismatch(field->name, field->branch, std::to_string(field->value),
                                         branch, std::to_string(value));
    }
  };

  for (std::size_t branch = 0; branch < ingress.branches.size(); ++branch) {
    const OutputSpec& spec = ingress.branches[branch].output_spec;
    update(&media, branch, spec.media_type);
    update(&format, branch, spec.format);
    update(&dtype, branch, spec.dtype);
    update(&layout, branch, spec.layout);
    update(&memory, branch, spec.memory);
    update_int(&width, branch, spec.width);
    update_int(&height, branch, spec.height);
    update_int(&depth, branch, spec.depth);
    if (spec.fps_num > 0 && spec.fps_den > 0) {
      if (fps_num == 0) {
        fps_num = spec.fps_num;
        fps_den = spec.fps_den;
        fps_branch = branch;
      } else if (static_cast<std::int64_t>(fps_num) * spec.fps_den !=
                 static_cast<std::int64_t>(spec.fps_num) * fps_den) {
        throw_fused_realtime_caps_mismatch(
            "framerate", fps_branch, std::to_string(fps_num) + "/" + std::to_string(fps_den),
            branch, std::to_string(spec.fps_num) + "/" + std::to_string(spec.fps_den));
      }
    }
    if (spec.payload_type != PayloadType::Auto) {
      if (payload_type == PayloadType::Auto) {
        payload_type = spec.payload_type;
        payload_branch = branch;
      } else if (payload_type != spec.payload_type) {
        throw_fused_realtime_caps_mismatch("payload_type", payload_branch,
                                           payload_type_label(payload_type), branch,
                                           payload_type_label(spec.payload_type));
      }
    }
  }
}

OutputSpec fused_ingress_spec_for_contracts(const runtime::FusedRealtimeIngress& ingress) {
  validate_fused_realtime_ingress_caps(ingress);

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

RawVideoCapsInfo source_meta_buffer_raw_video_size(GstBuffer* buffer) {
  RawVideoCapsInfo out;
  GstVideoMeta* meta = buffer ? gst_buffer_get_video_meta(buffer) : nullptr;
  if (!meta)
    return out;
  out.width = static_cast<int>(meta->width);
  out.height = static_cast<int>(meta->height);
  return out;
}

void source_meta_maybe_apply_preprocess_template(GstPad* pad, GstBuffer* buffer,
                                                 SourceSimaMetaProbeCtx* ctx) {
  if (!buffer || !ctx || !ctx->opt.preprocess_meta.has_value())
    return;
  if (has_simaai_preprocess_meta(buffer))
    return;

  const RawVideoCapsInfo video_meta = source_meta_buffer_raw_video_size(buffer);
  int width = video_meta.width;
  int height = video_meta.height;
  if (width <= 0 || height <= 0) {
    const RawVideoCapsInfo caps = source_meta_current_raw_video_caps(pad);
    if (width <= 0)
      width = caps.width;
    if (height <= 0)
      height = caps.height;
  }
  if (width <= 0)
    width = ctx->opt.width;
  if (height <= 0)
    height = ctx->opt.height;
  if (width <= 0 || height <= 0)
    return;

  (void)apply_simaai_preprocess_meta_template(buffer, ctx->opt, width, height);
}

bool source_sima_meta_structure_mutable(GstStructure* s) {
  if (!s) {
    return false;
  }
#if defined(GST_STRUCTURE_IS_MUTABLE)
  return GST_STRUCTURE_IS_MUTABLE(s);
#elif defined(GST_STRUCTURE_IS_WRITABLE)
  return GST_STRUCTURE_IS_WRITABLE(s);
#else
  return false;
#endif
}

GstStructure* source_sima_meta_get_mutable_structure(GstBuffer* buffer) {
  if (!buffer) {
    return nullptr;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  if (!meta) {
    return nullptr;
  }
  GstStructure* s = gst_custom_meta_get_structure(meta);
  if (!s) {
    return nullptr;
  }
  if (source_sima_meta_structure_mutable(s)) {
    return s;
  }
  if (!gst_buffer_is_writable(buffer)) {
    return nullptr;
  }

  GstStructure* snapshot = gst_structure_copy(s);
  gst_buffer_remove_meta(buffer, &meta->meta);
  meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    if (snapshot) {
      gst_structure_free(snapshot);
    }
    return nullptr;
  }
  if (snapshot) {
    gst_structure_foreach(
        snapshot,
        +[](GQuark field_id, const GValue* value, gpointer user_data) -> gboolean {
          auto* dst = static_cast<GstStructure*>(user_data);
          gst_structure_set_value(dst, g_quark_to_string(field_id), value);
          return TRUE;
        },
        s);
    gst_structure_free(snapshot);
  }
  return s;
}

void source_sima_meta_set_int64_if_missing(GstStructure* s, const char* field, gint64 value) {
  gint64 existing = 0;
  if (!s || gst_structure_get_int64(s, field, &existing) == TRUE) {
    return;
  }
  gst_structure_set(s, field, G_TYPE_INT64, value, nullptr);
}

void source_sima_meta_set_uint64_if_missing(GstStructure* s, const char* field, guint64 value) {
  guint64 existing = 0;
  if (!s || gst_structure_get_uint64(s, field, &existing) == TRUE) {
    return;
  }
  gst_structure_set(s, field, G_TYPE_UINT64, value, nullptr);
}

void source_sima_meta_set_int_if_missing(GstStructure* s, const char* field, gint value) {
  gint existing = 0;
  if (!s || gst_structure_get_int(s, field, &existing) == TRUE) {
    return;
  }
  gst_structure_set(s, field, G_TYPE_INT, value, nullptr);
}

void source_sima_meta_set_string_if_missing(GstStructure* s, const char* field,
                                            const std::string& value) {
  if (!s || value.empty()) {
    return;
  }
  const gchar* existing = gst_structure_get_string(s, field);
  if (existing && *existing) {
    return;
  }
  gst_structure_set(s, field, G_TYPE_STRING, value.c_str(), nullptr);
}

GstBuffer* source_sima_meta_preserve_existing(GstBuffer* buffer, SourceSimaMetaProbeCtx* ctx,
                                              const SampleTimingOverrides& timing) {
  GstBuffer* writable = gst_buffer_make_writable(buffer);
  if (!writable) {
    return nullptr;
  }
  GstStructure* s = source_sima_meta_get_mutable_structure(writable);
  if (!s) {
    return writable;
  }

  gint64 existing_frame = 0;
  const bool has_frame = gst_structure_get_int64(s, "frame-id", &existing_frame) == TRUE;
  const gint64 frame_id =
      has_frame ? existing_frame : static_cast<gint64>(timing.frame_id.value_or(0));
  source_sima_meta_set_int64_if_missing(s, "frame-id", frame_id);
  source_sima_meta_set_int64_if_missing(s, "input-seq", frame_id);
  source_sima_meta_set_int64_if_missing(s, "orig-input-seq", frame_id);

  const gchar* stream_id = gst_structure_get_string(s, "stream-id");
  if (stream_id && *stream_id) {
    source_sima_meta_set_string_if_missing(s, "orig-stream-id", stream_id);
  }

  const std::string fallback_buffer_name = ctx ? ctx->buffer_name : std::string{};
  source_sima_meta_set_string_if_missing(s, "buffer-name", fallback_buffer_name);
  const gchar* buffer_name = gst_structure_get_string(s, "buffer-name");
  source_sima_meta_set_string_if_missing(s, "origin_stage_id",
                                         (buffer_name && *buffer_name) ? std::string(buffer_name)
                                                                       : fallback_buffer_name);
  source_sima_meta_set_int_if_missing(s, "origin_output_slot", 0);

  gint64 phys_addr = 0;
  if (gst_buffer_n_memory(writable) > 0) {
    phys_addr = static_cast<gint64>(
        gst_simaai_segment_memory_get_phys_addr(gst_buffer_peek_memory(writable, 0)));
  }
  source_sima_meta_set_int64_if_missing(s, "buffer-id", phys_addr);
  source_sima_meta_set_int64_if_missing(s, "buffer-offset", 0);
  source_sima_meta_set_uint64_if_missing(
      s, "timestamp", static_cast<guint64>(timing.pts_ns.value_or(static_cast<std::uint64_t>(0))));
  return writable;
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

  const bool had_sima_meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta") != nullptr;
  GstBuffer* writable = nullptr;
  if (had_sima_meta) {
    writable = source_sima_meta_preserve_existing(buffer, ctx, timing);
  } else {
    // Source/decode metadata is stamped before this buffer crosses any graph edge.
    // Leave the stream id empty here: graph links and realtime routing own per-source
    // identity, so multi-source graphs do not collapse to a synthetic "0" stream.
    writable = attach_simaai_meta_inplace(
        buffer, ctx->opt, ctx->guard, ctx->element_name.c_str(), timing.frame_id,
        StreamIdOverride{std::optional<std::string>{std::string{}}},
        BufferNameOverride{std::optional<std::string>{ctx->buffer_name}});
  }
  if (!writable)
    return GST_PAD_PROBE_OK;

  if (!had_sima_meta) {
    const std::optional<int64_t> frame_opt = timing.frame_id;
    const std::optional<int64_t> input_seq = frame_opt;
    const std::optional<int64_t> orig_input_seq = input_seq;
    const std::optional<std::string> stream_id = std::string{};
    const std::optional<std::string> buffer_name{ctx->buffer_name};
    (void)update_simaai_meta_fields(writable, frame_opt, input_seq, orig_input_seq, stream_id,
                                    buffer_name, timing.pts_ns, buffer_name, std::optional<int>{0});
  }
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
  graph_build_internal::apply_explicit_public_output_options(stream_opt, build_nodes);
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
                                const NameTransform& name_transform, const GraphOptions* sess_opt,
                                bool prepend_link, const char* extra_fragment_props = nullptr) {
  if (!br || !br->diag || !pipeline || !node) {
    return;
  }
  if (prepend_link) {
    (*pipeline) << " ! ";
  }
  NodeFragment frag = make_node_fragment(node, actual_index, name_transform);
  // GraphNaming intentionally preserves GStreamer's conventional RTSP-server
  // payloader names (pay0, pay1, ...).  A fused ingress is a single ordinary
  // gst_parse_launch() pipeline, however, and may contain one RTP packetizer
  // per source.  Those elements must have unique names inside that pipeline;
  // they are not exported as RTSP media factory payloaders.
  if (node->kind() == "H264Packetize" && name_transform_enabled(name_transform)) {
    const std::string unique_payloader_name = "neat_fused_pay_" + std::to_string(actual_index);
    frag.fragment = rewrite_fragment_names(frag.fragment, {{"pay0", unique_payloader_name}});
    for (auto& element_name : frag.element_names) {
      if (element_name == "pay0") {
        element_name = unique_payloader_name;
      }
    }
  }
  NodeReport nr;
  nr.index = actual_index;
  nr.kind = node->kind();
  nr.user_label = node->user_label();
  nr.backend_fragment = session_build_apply_fast_path_options_to_fragment(frag.fragment, sess_opt);
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

std::string fused_stream_inflight_limits_csv(const runtime::FusedRealtimeIngress& ingress,
                                             bool enable_terminal_loans) {
  std::string csv;
  for (std::size_t i = 0; i < ingress.branches.size(); ++i) {
    if (i) {
      csv += ',';
    }
    const int limit = enable_terminal_loans
                          ? pipeline_internal::resolved_realtime_max_inflight_per_stream(
                                ingress.branches[i].link_options)
                          : 0;
    csv += std::to_string(limit);
  }
  return csv;
}

int fused_max_inflight_total(const runtime::FusedRealtimeIngress& ingress,
                             bool enable_terminal_loans) {
  if (!enable_terminal_loans || ingress.branches.empty()) {
    return 0;
  }

  int total_capacity = 0;
  int explicit_limit = -1;
  for (const auto& branch : ingress.branches) {
    const int per_stream =
        pipeline_internal::resolved_realtime_max_inflight_per_stream(branch.link_options);
    if (total_capacity > std::numeric_limits<int>::max() - per_stream) {
      total_capacity = std::numeric_limits<int>::max();
    } else {
      total_capacity += per_stream;
    }
    if (branch.link_options.max_inflight_total > 0) {
      explicit_limit = explicit_limit > 0
                           ? std::min(explicit_limit, branch.link_options.max_inflight_total)
                           : branch.link_options.max_inflight_total;
    }
  }
  if (explicit_limit > 0) {
    return explicit_limit;
  }

  int env_limit = 0;
  if (pipeline_internal::env_int("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL", &env_limit)) {
    return std::max(0, env_limit);
  }
  return pipeline_internal::default_realtime_max_inflight_total(total_capacity);
}

GstBuffer* make_fused_terminal_probe_buffer_writable(GstPadProbeInfo* info) {
  if (!info || (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return nullptr;
  }
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer || gst_buffer_is_writable(buffer)) {
    return buffer;
  }

  // Transfer the probe data slot's streaming reference directly through
  // GStreamer's copy-on-write operation and replace the slot immediately.
  // gst_buffer_make_writable() consumes exactly that reference, so no
  // separate unref of the probe's original pointer is valid here.
  GstBuffer* writable = gst_buffer_make_writable(buffer);
  GST_PAD_PROBE_INFO_DATA(info) = writable;
  return writable;
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

std::string fused_encoded_output_tap_name(const NameTransform& base, std::size_t branch_index) {
  return apply_name_transform(fused_branch_name_transform(base, branch_index),
                              "neat_encoded_output_tap");
}

NameTransform fused_branch_local_name_transform(std::size_t branch_index) {
  NameTransform out;
  out.suffix = "_b" + std::to_string(branch_index);
  return out;
}

void apply_name_transform_to_fused_manifest_stage(pipeline_internal::sima::StageStaticSpec* stage,
                                                  const NameTransform& name_transform) {
  if (!stage || !name_transform_enabled(name_transform)) {
    return;
  }
  stage->element_name = apply_name_transform(name_transform, stage->element_name);
  stage->logical_stage_id = apply_name_transform(name_transform, stage->logical_stage_id);
  for (auto& binding : stage->input_bindings) {
    binding.src_stage_id = apply_name_transform(name_transform, binding.src_stage_id);
  }
}

void apply_name_transform_to_runtime_contract(CompiledRuntimeContract* contract,
                                              const NameTransform& name_transform) {
  if (!contract || !name_transform_enabled(name_transform)) {
    return;
  }
  for (auto& binding : contract->input_bindings) {
    binding.src_stage_id = apply_name_transform(name_transform, binding.src_stage_id);
  }
}

void apply_name_transform_to_compiled_contract_stage(CompiledNodeContract* stage,
                                                     const NameTransform& name_transform) {
  if (!stage || !name_transform_enabled(name_transform)) {
    return;
  }
  stage->element_name = apply_name_transform(name_transform, stage->element_name);
  stage->logical_stage_id = apply_name_transform(name_transform, stage->logical_stage_id);
  if (stage->processcvu) {
    apply_name_transform_to_runtime_contract(&stage->processcvu->runtime_contract, name_transform);
  }
  if (stage->processmla) {
    apply_name_transform_to_runtime_contract(&stage->processmla->runtime_contract, name_transform);
  }
  if (stage->boxdecode) {
    apply_name_transform_to_runtime_contract(&stage->boxdecode->runtime_contract, name_transform);
  }
  if (stage->dequant) {
    apply_name_transform_to_runtime_contract(&stage->dequant->runtime_contract, name_transform);
  }
  if (stage->transport) {
    apply_name_transform_to_runtime_contract(&stage->transport->runtime_contract, name_transform);
  }
  for (auto& child : stage->child_stages) {
    apply_name_transform_to_compiled_contract_stage(&child, name_transform);
  }
}

void append_compiled_contracts(BuildResult* dst, const BuildResult& src,
                               const NameTransform& local_name_transform) {
  if (!dst || !src.compiled_contracts) {
    return;
  }
  if (!dst->compiled_contracts) {
    dst->compiled_contracts = std::make_shared<CompiledPipelineContracts>();
    dst->compiled_contracts->fully_renderable = true;
  }
  dst->compiled_contracts->fully_renderable =
      dst->compiled_contracts->fully_renderable && src.compiled_contracts->fully_renderable;
  dst->compiled_contracts->stages.reserve(dst->compiled_contracts->stages.size() +
                                          src.compiled_contracts->stages.size());
  for (auto stage : src.compiled_contracts->stages) {
    apply_name_transform_to_compiled_contract_stage(&stage, local_name_transform);
    dst->compiled_contracts->stages.push_back(std::move(stage));
  }
}

void append_rendered_manifest(BuildResult* dst, const BuildResult& src,
                              const NameTransform& local_name_transform) {
  if (!dst || !src.rendered_manifest.has_value()) {
    return;
  }
  if (!dst->rendered_manifest.has_value()) {
    dst->rendered_manifest = pipeline_internal::sima::SimaPluginStaticManifest{};
  }
  for (auto stage : src.rendered_manifest->stages) {
    apply_name_transform_to_fused_manifest_stage(&stage, local_name_transform);
    dst->rendered_manifest->stages.push_back(std::move(stage));
  }
}

void append_manifest_diagnostics(BuildResult* dst, const BuildResult& src) {
  if (!dst) {
    return;
  }
  dst->manifest_diagnostics.errors.insert(dst->manifest_diagnostics.errors.end(),
                                          src.manifest_diagnostics.errors.begin(),
                                          src.manifest_diagnostics.errors.end());
  dst->manifest_diagnostics.warnings.insert(dst->manifest_diagnostics.warnings.end(),
                                            src.manifest_diagnostics.warnings.begin(),
                                            src.manifest_diagnostics.warnings.end());
}

void append_model_source_paths(BuildResult* dst, const BuildResult& src) {
  if (!dst) {
    return;
  }
  for (const auto& path : src.model_source_paths) {
    if (std::find(dst->model_source_paths.begin(), dst->model_source_paths.end(), path) ==
        dst->model_source_paths.end()) {
      dst->model_source_paths.push_back(path);
    }
  }
}

void merge_fused_contract_build_result(BuildResult* dst, const BuildResult& src,
                                       const NameTransform& local_name_transform) {
  append_compiled_contracts(dst, src, local_name_transform);
  append_rendered_manifest(dst, src, local_name_transform);
  append_manifest_diagnostics(dst, src);
  append_model_source_paths(dst, src);
}

void maybe_compile_fused_realtime_contracts(
    BuildResult* br, const std::vector<std::shared_ptr<Node>>& consumer_nodes,
    const std::vector<std::vector<std::shared_ptr<Node>>>& branch_nodes,
    const std::vector<std::vector<int>>& branch_actual_indices, const GraphOptions& sess_opt,
    const char* where, const OutputSpec& fused_ingress_spec) {
  if (!br) {
    return;
  }

  BuildResult consumer_result;
  maybe_compile_source_contracts(&consumer_result, consumer_nodes, sess_opt, where,
                                 &fused_ingress_spec);
  merge_fused_contract_build_result(br, consumer_result, NameTransform{});

  for (std::size_t branch_index = 0; branch_index < branch_nodes.size(); ++branch_index) {
    const auto& nodes = branch_nodes[branch_index];
    if (nodes.empty()) {
      continue;
    }
    const std::vector<int>* node_indices = nullptr;
    if (branch_index < branch_actual_indices.size() &&
        branch_actual_indices[branch_index].size() == nodes.size()) {
      node_indices = &branch_actual_indices[branch_index];
    }
    BuildResult branch_result;
    maybe_compile_source_contracts(&branch_result, nodes, sess_opt, where,
                                   /*ingress_spec=*/nullptr, node_indices);
    merge_fused_contract_build_result(br, branch_result,
                                      fused_branch_local_name_transform(branch_index));
  }
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
      elem.find("boxdecode") != std::string::npos ||
      elem.find("objectdecode") != std::string::npos) {
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
    std::fprintf(
        stderr,
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
                                      FusedStageCounterKind stage, const std::string& element_name,
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
    if (stream_index_override.has_value() &&
        *stream_index_override < state->expected_streams.size()) {
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

std::optional<std::size_t> parse_fused_branch_index_from_name(const std::string& name);

struct FusedEncodedOutputProbeContext {
  runtime::FusedRealtimeIngressBranch::EncodedOutput output;
  runtime::FusedEncodedOutputDispatch dispatch;
  bool copy_output = false;
};

Sample make_fused_encoded_output_sample(GstBuffer* buffer, GstCaps* caps,
                                        const std::string& stream_id, bool copy_output) {
  if (!buffer) {
    throw std::runtime_error("encoded H.264 buffer is null");
  }
  if (!caps) {
    throw std::runtime_error("encoded H.264 caps are unavailable");
  }

  if (copy_output) {
    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      throw std::runtime_error("failed to map encoded H.264 buffer");
    }
    std::vector<std::uint8_t> bytes;
    try {
      bytes.assign(map.data, map.data + map.size);
    } catch (...) {
      gst_buffer_unmap(buffer, &map);
      throw;
    }
    gst_buffer_unmap(buffer, &map);
    if (bytes.empty()) {
      throw std::runtime_error("encoded H.264 buffer is empty");
    }

    gchar* caps_text = gst_caps_to_string(caps);
    if (!caps_text || !*caps_text) {
      if (caps_text) {
        g_free(caps_text);
      }
      throw std::runtime_error("encoded H.264 caps are empty");
    }
    const std::string caps_string(caps_text);
    g_free(caps_text);
    const std::int64_t pts = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))
                                 ? static_cast<std::int64_t>(GST_BUFFER_PTS(buffer))
                                 : -1;
    const std::int64_t dts = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))
                                 ? static_cast<std::int64_t>(GST_BUFFER_DTS(buffer))
                                 : -1;
    const std::int64_t duration = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))
                                      ? static_cast<std::int64_t>(GST_BUFFER_DURATION(buffer))
                                      : -1;
    Sample sample = make_encoded_sample(std::move(bytes), caps_string, pts, dts, duration);
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    const char* media_type = structure ? gst_structure_get_name(structure) : nullptr;
    sample.media_type = media_type ? media_type : "";
    sample.stream_id = stream_id;
    return sample;
  }

  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  if (!gst_sample) {
    throw std::runtime_error("failed to create encoded H.264 sample");
  }
  try {
    // Encoded parser output is ordinary CPU-readable GstMemory. Avoid the
    // general raw/tensor envelope path here: it performs device-memory and
    // segment discovery that is useful for accelerator buffers but needlessly
    // expensive on every AU across 24/48 live streams. Keep the exact
    // GstSample reference so payload bytes and buffer flags remain zero-copy.
    auto holder = std::shared_ptr<void>(gst_sample_ref(gst_sample), [](void* value) {
      gst_sample_unref(static_cast<GstSample*>(value));
    });
    struct EncodedMapState {
      std::mutex mutex;
      GstMapInfo info{};
      bool mapped = false;
    };
    auto map_state = std::make_shared<EncodedMapState>();

    auto storage = std::make_shared<Storage>();
    storage->kind = StorageKind::GstSample;
    storage->device = {DeviceType::CPU, 0};
    storage->size_bytes = gst_buffer_get_size(buffer);
    storage->holder = holder;
    storage->map_fn = [holder, map_state](MapMode mode) {
      if (mode != MapMode::Read) {
        return Mapping{};
      }
      GstSample* retained = static_cast<GstSample*>(holder.get());
      GstBuffer* retained_buffer = retained ? gst_sample_get_buffer(retained) : nullptr;
      if (!retained_buffer) {
        return Mapping{};
      }
      std::lock_guard<std::mutex> lock(map_state->mutex);
      if (map_state->mapped || !gst_buffer_map(retained_buffer, &map_state->info, GST_MAP_READ)) {
        return Mapping{};
      }
      map_state->mapped = true;
      GstBuffer* buffer_ref = gst_buffer_ref(retained_buffer);
      Mapping mapping;
      mapping.data = map_state->info.data;
      mapping.size_bytes = map_state->info.size;
      mapping.keepalive = holder;
      mapping.unmap = [buffer_ref, map_state]() {
        std::lock_guard<std::mutex> unmap_lock(map_state->mutex);
        if (map_state->mapped) {
          gst_buffer_unmap(buffer_ref, &map_state->info);
          map_state->mapped = false;
        }
        gst_buffer_unref(buffer_ref);
      };
      return mapping;
    };

    gchar* caps_text = gst_caps_to_string(caps);
    if (!caps_text || !*caps_text) {
      if (caps_text) {
        g_free(caps_text);
      }
      throw std::runtime_error("encoded Output caps are empty");
    }
    const std::string caps_string(caps_text);
    g_free(caps_text);
    const EncodedSpec::Codec codec = caps_to_codec(caps_string);
    if (codec == EncodedSpec::Codec::UNKNOWN) {
      throw std::runtime_error("encoded Output caps do not identify a supported codec");
    }

    Tensor tensor;
    tensor.storage = std::move(storage);
    tensor.dtype = TensorDType::UInt8;
    tensor.device = {DeviceType::CPU, 0};
    tensor.read_only = true;
    tensor.layout = TensorLayout::Unknown;
    tensor.shape = {static_cast<std::int64_t>(gst_buffer_get_size(buffer))};
    tensor.strides_bytes = {1};
    tensor.semantic.encoded = EncodedSpec{};
    tensor.semantic.encoded->codec = codec;

    Sample sample;
    sample.kind = SampleKind::TensorSet;
    sample.owned = false;
    sample.tensors = TensorList{std::move(tensor)};
    sample.caps_string = caps_string;
    sample.payload_type = PayloadType::Encoded;
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    const char* media_type = structure ? gst_structure_get_name(structure) : nullptr;
    sample.media_type = media_type ? media_type : "";
    sample.payload_tag = codec == EncodedSpec::Codec::H264 ? "H264" : "H265";
    sample.pts_ns = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))
                        ? static_cast<std::int64_t>(GST_BUFFER_PTS(buffer))
                        : -1;
    sample.dts_ns = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))
                        ? static_cast<std::int64_t>(GST_BUFFER_DTS(buffer))
                        : -1;
    sample.duration_ns = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))
                             ? static_cast<std::int64_t>(GST_BUFFER_DURATION(buffer))
                             : -1;
    sample.stream_id = stream_id;
    gst_sample_unref(gst_sample);
    return sample;
  } catch (...) {
    gst_sample_unref(gst_sample);
    throw;
  }
}

GstPadProbeReturn fused_encoded_output_probe(GstPad* pad, GstPadProbeInfo* info,
                                             gpointer user_data) {
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  auto* context = static_cast<FusedEncodedOutputProbeContext*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  GstCaps* caps = pad ? gst_pad_get_current_caps(pad) : nullptr;
  std::string error;
  runtime::FusedEncodedOutputDispatchResult dispatch_result =
      runtime::FusedEncodedOutputDispatchResult::Failed;
  try {
    if (!context || !context->dispatch) {
      throw std::runtime_error("encoded output dispatcher is unavailable");
    }
    Sample sample = make_fused_encoded_output_sample(buffer, caps, context->output.stream_id,
                                                     context->copy_output);
    dispatch_result = context->dispatch(context->output, std::move(sample), &error);
  } catch (const std::exception& ex) {
    error = ex.what();
  } catch (...) {
    error = "unknown fused encoded output failure";
  }
  if (caps) {
    gst_caps_unref(caps);
  }
  if (dispatch_result == runtime::FusedEncodedOutputDispatchResult::Delivered) {
    return GST_PAD_PROBE_OK;
  }
  if (dispatch_result == runtime::FusedEncodedOutputDispatchResult::Stopping) {
    return GST_PAD_PROBE_DROP;
  }

  GstElement* parent = pad ? gst_pad_get_parent_element(pad) : nullptr;
  if (parent) {
    const char* id = context && !context->output.stream_id.empty()
                         ? context->output.stream_id.c_str()
                         : "unknown";
    GST_ELEMENT_ERROR(parent, RESOURCE, FAILED,
                      ("Encoded Output dispatch failed for stream '%s'", id),
                      ("%s", error.empty() ? "unknown encoded Output failure" : error.c_str()));
    gst_object_unref(parent);
  } else {
    std::fprintf(stderr, "[fused-encoded-output] fatal dispatch failure: %s\n", error.c_str());
  }
  return GST_PAD_PROBE_DROP;
}

std::size_t attach_fused_encoded_output_probes(GstElement* pipeline,
                                               const runtime::FusedRealtimeIngress& ingress,
                                               const runtime::FusedEncodedOutputDispatch& dispatch,
                                               bool copy_output,
                                               const NameTransform& name_transform) {
  if (!pipeline) {
    return 0U;
  }

  const std::size_t expected = static_cast<std::size_t>(
      std::count_if(ingress.branches.begin(), ingress.branches.end(),
                    [](const auto& branch) { return branch.encoded_output.has_value(); }));
  if (expected == 0U) {
    return 0U;
  }
  if (!dispatch) {
    throw std::runtime_error("fused encoded Output dispatcher is unavailable");
  }

  std::size_t attached = 0U;
  for (std::size_t branch_index = 0; branch_index < ingress.branches.size(); ++branch_index) {
    const auto& branch = ingress.branches[branch_index];
    if (!branch.encoded_output.has_value()) {
      continue;
    }

    // The renderer inserts exactly one private identity at the original
    // source-to-decoder split. Resolve that element by its deterministic name
    // rather than scanning every H.264 capsfilter in the branch: decoder-side
    // parsers are allowed to enforce their own caps and are not public taps.
    const std::string tap_name = fused_encoded_output_tap_name(name_transform, branch_index);
    GstElement* tap = gst_bin_get_by_name(GST_BIN(pipeline), tap_name.c_str());
    if (!tap) {
      throw std::runtime_error("fused encoded Output tap '" + tap_name + "' was not materialized");
    }
    GstPad* src = gst_element_get_static_pad(tap, "src");
    gst_object_unref(tap);
    if (!src) {
      throw std::runtime_error("fused encoded Output tap '" + tap_name + "' has no src pad");
    }

    auto* context =
        new FusedEncodedOutputProbeContext{*branch.encoded_output, dispatch, copy_output};
    const gulong probe = gst_pad_add_probe(
        src, GST_PAD_PROBE_TYPE_BUFFER, fused_encoded_output_probe, context,
        +[](gpointer data) { delete static_cast<FusedEncodedOutputProbeContext*>(data); });
    gst_object_unref(src);
    if (probe == 0) {
      delete context;
      throw std::runtime_error("failed to attach fused encoded Output probe to '" + tap_name + "'");
    }
    ++attached;
  }
  if (attached != expected) {
    throw std::runtime_error("failed to attach exactly one fused encoded Output probe per branch");
  }
  return attached;
}

struct FusedDecoderTimingBranch {
  std::mutex mutex;
  std::deque<SampleTimingOverrides> pending;
};

struct FusedDecoderTimingProbeCtx {
  std::shared_ptr<FusedDecoderTimingBranch> branch;
  bool decoder_output = false;
};

std::optional<std::size_t>
find_fused_decoder_timing_match(const std::deque<SampleTimingOverrides>& pending,
                                std::optional<std::uint64_t> output_pts,
                                std::optional<std::uint64_t> output_dts) {
  const auto find_timestamp = [&](const auto& select,
                                  std::uint64_t value) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < pending.size(); ++i) {
      const auto timestamp = select(pending[i]);
      if (timestamp.has_value() && *timestamp == value) {
        return i;
      }
    }
    return std::nullopt;
  };
  if (output_pts.has_value()) {
    if (const auto match = find_timestamp(
            [](const SampleTimingOverrides& timing) { return timing.pts_ns; }, *output_pts)) {
      return match;
    }
  }
  if (output_dts.has_value()) {
    return find_timestamp([](const SampleTimingOverrides& timing) { return timing.dts_ns; },
                          *output_dts);
  }
  return std::nullopt;
}

GstPadProbeReturn fused_decoder_timing_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
    return GST_PAD_PROBE_OK;
  }
  auto* ctx = static_cast<FusedDecoderTimingProbeCtx*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!ctx || !ctx->branch || !buffer) {
    return GST_PAD_PROBE_OK;
  }
  if (!ctx->decoder_output) {
    SampleTimingOverrides timing;
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))) {
      timing.pts_ns = static_cast<std::uint64_t>(GST_BUFFER_PTS(buffer));
    }
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))) {
      timing.dts_ns = static_cast<std::uint64_t>(GST_BUFFER_DTS(buffer));
    }
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DURATION(buffer))) {
      timing.duration_ns = static_cast<std::uint64_t>(GST_BUFFER_DURATION(buffer));
    }
    std::lock_guard<std::mutex> lock(ctx->branch->mutex);
    ctx->branch->pending.push_back(std::move(timing));
    if (ctx->branch->pending.size() > 64U) {
      ctx->branch->pending.pop_front();
    }
    return GST_PAD_PROBE_OK;
  }

  std::optional<std::uint64_t> output_pts;
  std::optional<std::uint64_t> output_dts;
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))) {
    output_pts = static_cast<std::uint64_t>(GST_BUFFER_PTS(buffer));
  }
  if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_DTS(buffer))) {
    output_dts = static_cast<std::uint64_t>(GST_BUFFER_DTS(buffer));
  }

  SampleTimingOverrides timing;
  {
    std::lock_guard<std::mutex> lock(ctx->branch->mutex);
    // GstVideoDecoder preserves the input AU presentation timestamp on the
    // corresponding decoded frame, including when frames are reordered. Use
    // that identity (or DTS when PTS is unavailable) instead of queue age: an
    // AU may be dropped without producing a raw frame, and FIFO consumption
    // would then shift every subsequent timing record onto the wrong output.
    const auto match =
        find_fused_decoder_timing_match(ctx->branch->pending, output_pts, output_dts);
    if (!match.has_value()) {
      return GST_PAD_PROBE_OK;
    }
    auto it = ctx->branch->pending.begin() + static_cast<std::ptrdiff_t>(*match);
    timing = std::move(*it);
    ctx->branch->pending.erase(it);
  }
  if (timing.pts_ns.has_value()) {
    GST_BUFFER_PTS(buffer) = static_cast<GstClockTime>(*timing.pts_ns);
  }
  if (timing.dts_ns.has_value()) {
    GST_BUFFER_DTS(buffer) = static_cast<GstClockTime>(*timing.dts_ns);
  }
  if (timing.duration_ns.has_value()) {
    GST_BUFFER_DURATION(buffer) = static_cast<GstClockTime>(*timing.duration_ns);
  }
  (void)write_sample_timing_to_gst_buffer(buffer, timing);
  return GST_PAD_PROBE_OK;
}

std::size_t attach_fused_decoder_timing_probes(GstElement* pipeline,
                                               const runtime::FusedRealtimeIngress& ingress) {
  if (!pipeline || ingress.branches.empty()) {
    return 0U;
  }
  std::vector<std::shared_ptr<FusedDecoderTimingBranch>> timing;
  timing.reserve(ingress.branches.size());
  for (std::size_t i = 0; i < ingress.branches.size(); ++i) {
    timing.push_back(std::make_shared<FusedDecoderTimingBranch>());
  }

  GstIterator* it = gst_bin_iterate_elements(GST_BIN(pipeline));
  if (!it) {
    return 0U;
  }
  std::size_t attached = 0U;
  GValue item = G_VALUE_INIT;
  while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    GstElement* element = GST_ELEMENT(g_value_get_object(&item));
    const char* name_c = element ? GST_ELEMENT_NAME(element) : nullptr;
    const std::string name = name_c ? name_c : "";
    GstElementFactory* factory = element ? gst_element_get_factory(element) : nullptr;
    const char* factory_c =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : nullptr;
    const std::string factory_name = factory_c ? factory_c : "";
    const auto branch_index = parse_fused_branch_index_from_name(name);
    if (!branch_index.has_value() || *branch_index >= timing.size() ||
        factory_name != "neatdecoder") {
      g_value_reset(&item);
      continue;
    }

    // Capture the AU timing at this decoder's input and restore it on this
    // decoder's output.  Name-matching every *_h264_caps element is unsafe in
    // an encoded fan-out: the source/decode prefix and VideoSender branch can
    // each contain an H264Parse capsfilter, which would enqueue the same AU
    // more than once and leave stale timing records behind.
    for (const auto& [pad_name, decoder_output] :
         std::array<std::pair<const char*, bool>, 2>{{{"sink", false}, {"src", true}}}) {
      GstPad* pad = gst_element_get_static_pad(element, pad_name);
      if (pad) {
        auto* ctx = new FusedDecoderTimingProbeCtx{timing[*branch_index], decoder_output};
        const gulong probe_id = gst_pad_add_probe(
            pad, GST_PAD_PROBE_TYPE_BUFFER, fused_decoder_timing_probe, ctx,
            +[](gpointer data) { delete static_cast<FusedDecoderTimingProbeCtx*>(data); });
        if (probe_id != 0U) {
          ++attached;
        } else {
          delete ctx;
        }
        gst_object_unref(pad);
      }
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  return attached;
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

std::vector<std::string> split_fused_consumer_segments(const std::string& fragment) {
  std::vector<std::string> segments;
  bool in_single_quote = false;
  bool in_double_quote = false;
  bool escaped = false;
  std::size_t start = 0;
  for (std::size_t i = 0; i < fragment.size(); ++i) {
    const char c = fragment[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\' && (in_single_quote || in_double_quote)) {
      escaped = true;
      continue;
    }
    if (c == '\'' && !in_double_quote) {
      in_single_quote = !in_single_quote;
      continue;
    }
    if (c == '"' && !in_single_quote) {
      in_double_quote = !in_double_quote;
      continue;
    }
    if (c != '!' || in_single_quote || in_double_quote) {
      continue;
    }
    const std::string segment = trim_copy(fragment.substr(start, i - start));
    if (!segment.empty()) {
      segments.push_back(segment);
    }
    start = i + 1U;
  }
  const std::string tail = trim_copy(fragment.substr(start));
  if (!tail.empty()) {
    segments.push_back(tail);
  }
  return segments;
}

std::string fused_consumer_segment_factory(const std::string& segment) {
  const std::size_t start = segment.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = segment.find_first_of(" \t\r\n", start);
  return lower_copy(
      segment.substr(start, end == std::string::npos ? std::string::npos : end - start));
}

bool is_fused_consumer_stage_factory(const std::string& factory) {
  return factory == "neatprocesscvu" || factory == "neatprocessmla" ||
         factory == "neatobjectdecode" || factory == "neatboxdecode";
}

bool fused_consumer_fragment_replaces_buffers(const std::string& fragment) {
  for (const auto& segment : split_fused_consumer_segments(fragment)) {
    if (is_fused_consumer_stage_factory(fused_consumer_segment_factory(segment))) {
      return true;
    }
  }
  return false;
}

std::optional<std::string> fused_consumer_segment_property_value(const std::string& segment,
                                                                 const std::string& property) {
  const std::string lower = lower_copy(segment);
  const auto is_space = [](char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
  };
  bool in_single_quote = false;
  bool in_double_quote = false;
  std::optional<std::string> effective_value;
  for (std::size_t i = 0; i < lower.size(); ++i) {
    const char c = lower[i];
    if (c == '\'' && !in_double_quote) {
      in_single_quote = !in_single_quote;
      continue;
    }
    if (c == '"' && !in_single_quote) {
      in_double_quote = !in_double_quote;
      continue;
    }
    if (in_single_quote || in_double_quote || (i != 0U && !is_space(lower[i - 1U])) ||
        lower.compare(i, property.size(), property) != 0) {
      continue;
    }
    std::size_t cursor = i + property.size();
    if (cursor < lower.size() && lower[cursor] != '=' && !is_space(lower[cursor])) {
      continue;
    }
    while (cursor < lower.size() && is_space(lower[cursor])) {
      ++cursor;
    }
    if (cursor >= lower.size() || lower[cursor] != '=') {
      continue;
    }
    ++cursor;
    while (cursor < lower.size() && is_space(lower[cursor])) {
      ++cursor;
    }
    // gst-launch accepts optional type annotations such as
    // leaky=(GstQueueLeaky)downstream (and whitespace after the annotation).
    if (cursor < lower.size() && lower[cursor] == '(') {
      const std::size_t close = lower.find(')', cursor + 1U);
      if (close == std::string::npos) {
        effective_value = std::string{};
        break;
      }
      cursor = close + 1U;
      while (cursor < lower.size() && is_space(lower[cursor])) {
        ++cursor;
      }
    }
    if (cursor >= lower.size()) {
      effective_value = std::string{};
      break;
    }
    if (lower[cursor] == '\'' || lower[cursor] == '"') {
      const char quote = lower[cursor++];
      const std::size_t close = lower.find(quote, cursor);
      effective_value =
          lower.substr(cursor, close == std::string::npos ? std::string::npos : close - cursor);
      if (close == std::string::npos) {
        break;
      }
      i = close;
      continue;
    }
    std::size_t end = cursor;
    while (end < lower.size() && !is_space(lower[end])) {
      ++end;
    }
    effective_value = lower.substr(cursor, end - cursor);
    i = end;
  }
  return effective_value;
}

bool fused_consumer_property_is_zero(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  char* end = nullptr;
  const double parsed = std::strtod(value.c_str(), &end);
  return end != value.c_str() && end && *end == '\0' && parsed == 0.0;
}

bool fused_consumer_fragment_may_drop(const std::string& fragment) {
  for (const auto& segment : split_fused_consumer_segments(fragment)) {
    const std::string factory = fused_consumer_segment_factory(segment);
    if (factory == "videorate") {
      return true;
    }
    if (factory == "queue") {
      if (const auto leaky = fused_consumer_segment_property_value(segment, "leaky"); leaky) {
        // GstQueueLeaky's non-dropping value is 0 / "no". Unknown values are
        // rejected conservatively because this check is the ownership proof
        // that permits disabling source-buffer lifetime guards.
        if (*leaky != "0" && *leaky != "no" && *leaky != "none") {
          return true;
        }
      }
    } else if (factory == "identity") {
      if (const auto probability =
              fused_consumer_segment_property_value(segment, "drop-probability");
          probability && !fused_consumer_property_is_zero(*probability)) {
        return true;
      }
      if (const auto flags = fused_consumer_segment_property_value(segment, "drop-buffer-flags");
          flags && *flags != "0" && *flags != "none") {
        return true;
      }
    } else if (factory == "valve") {
      if (const auto drop = fused_consumer_segment_property_value(segment, "drop");
          drop && *drop != "0" && *drop != "false" && *drop != "no") {
        return true;
      }
    }
  }
  return false;
}

std::string insert_fused_consumer_stage_queues(std::string fragment, int requested_depth) {
  if (requested_depth < 0 || requested_depth > 1024) {
    throw std::invalid_argument(
        "GraphOptions::async_queue_depth must be in [0, 1024] for fused realtime ingress");
  }
  if (requested_depth == 0 || fragment.empty()) {
    return fragment;
  }

  const auto segments = split_fused_consumer_segments(fragment);
  std::vector<std::string> rendered;
  rendered.reserve(segments.size() + 3U);
  const std::string queue = session_build_async_queue2_fragment(requested_depth);
  for (const auto& segment : segments) {
    const std::string factory = fused_consumer_segment_factory(segment);
    if (is_fused_consumer_stage_factory(factory)) {
      rendered.push_back(queue);
    }
    rendered.push_back(segment);
  }

  std::ostringstream out;
  for (std::size_t i = 0; i < rendered.size(); ++i) {
    if (i != 0U) {
      out << " ! ";
    }
    out << rendered[i];
  }
  return out.str();
}

BuildResult build_fused_realtime_source_pipeline(
    const runtime::FusedRealtimeIngress& ingress,
    const std::vector<std::shared_ptr<Node>>& consumer_nodes,
    const std::vector<std::vector<std::shared_ptr<Node>>>& branch_nodes,
    const std::vector<std::vector<int>>& branch_actual_indices, const NameTransform& name_transform,
    const std::string& appsink_name, const GraphOptions& sess_opt, bool enable_terminal_loans) {
  BuildResult br;
  br.diag = std::make_shared<DiagCtx>();
  br.appsink_name = apply_name_transform(name_transform, appsink_name);
  br.name_transform = name_transform;
  br.diag->queue2_enabled = sess_opt.async_queue_depth > 0;
  br.diag->queue2_depth =
      br.diag->queue2_enabled ? session_build_async_queue2_depth(sess_opt.async_queue_depth) : 0;

  const std::string mux_name = apply_name_transform(name_transform, "neat_live_mux");
  br.diag->node_reports.reserve(consumer_nodes.size() + 1U);
  NodeReport mux_report;
  mux_report.index = -1;
  const std::string stream_ids = fused_stream_ids_csv(ingress);
  const std::string stream_inflight_limits =
      fused_stream_inflight_limits_csv(ingress, enable_terminal_loans);
  const int max_inflight_total = fused_max_inflight_total(ingress, enable_terminal_loans);
  mux_report.kind = "RealtimeLatestMux";
  mux_report.backend_fragment =
      "neatlatestbystreammux name=" + mux_name + " stream-ids=" + gst_double_quote(stream_ids) +
      " stream-inflight-limits=" + gst_double_quote(stream_inflight_limits) +
      " max-inflight-total=" + std::to_string(max_inflight_total);
  mux_report.elements.push_back(mux_name);
  br.diag->node_reports.push_back(mux_report);

  std::ostringstream ss;
  ss << "neatlatestbystreammux name=" << mux_name << " stream-ids=" << gst_double_quote(stream_ids)
     << " stream-inflight-limits=" << gst_double_quote(stream_inflight_limits)
     << " max-inflight-total=" << max_inflight_total;
  std::ostringstream consumer_pipeline;
  int actual_index = 0;
  bool first_consumer = true;
  for (const auto& node : consumer_nodes) {
    append_fused_node_fragment(&br, &consumer_pipeline, node, actual_index++, name_transform,
                               &sess_opt, /*prepend_link=*/!first_consumer);
    first_consumer = false;
  }
  const std::string rendered_consumer =
      insert_fused_consumer_stage_queues(consumer_pipeline.str(), sess_opt.async_queue_depth);
  br.fused_consumer_replaces_buffers = fused_consumer_fragment_replaces_buffers(rendered_consumer);
  if (enable_terminal_loans && br.fused_consumer_replaces_buffers &&
      fused_consumer_fragment_may_drop(rendered_consumer)) {
    throw std::invalid_argument(
        "fused realtime terminal-loan consumers that replace buffers must not contain "
        "pre-terminal dropping elements (videorate, leaky queue, valve drop, or identity "
        "drop-probability); keep the chain ordered/non-dropping or use the non-fused graph path");
  }
  if (!rendered_consumer.empty()) {
    ss << " ! " << rendered_consumer;
  }

  int encoded_actual_index = static_cast<int>(consumer_nodes.size());
  for (const auto& nodes : branch_nodes) {
    encoded_actual_index += static_cast<int>(nodes.size());
  }
  for (std::size_t branch_index = 0; branch_index < branch_nodes.size(); ++branch_index) {
    const auto& nodes = branch_nodes[branch_index];
    if (nodes.empty()) {
      continue;
    }
    const NameTransform branch_transform =
        fused_branch_name_transform(name_transform, branch_index);
    const auto decoder = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) {
      return node && node->kind() == "SimaDecode";
    });
    const bool has_encoded_output = branch_index < ingress.branches.size() &&
                                    ingress.branches[branch_index].encoded_output.has_value();
    const bool has_encoded_sink = branch_index < ingress.branches.size() &&
                                  !ingress.branches[branch_index].encoded_sink_nodes.empty();
    const bool has_encoded_boundary = has_encoded_output || has_encoded_sink;
    std::size_t encoded_split = 0U;
    if (has_encoded_boundary) {
      encoded_split = ingress.branches[branch_index].encoded_split_node_index.value_or(
          decoder == nodes.end() ? nodes.size()
                                 : static_cast<std::size_t>(std::distance(nodes.begin(), decoder)));
    }
    const bool decoder_after_split =
        has_encoded_boundary && encoded_split <= nodes.size() &&
        std::any_of(nodes.begin() + static_cast<std::ptrdiff_t>(encoded_split), nodes.end(),
                    [](const auto& node) { return node && node->kind() == "SimaDecode"; });
    if (has_encoded_boundary && (encoded_split > nodes.size() || !decoder_after_split)) {
      throw std::invalid_argument(
          "fused encoded branch requires a valid source boundary before H.264 SimaDecode");
    }

    ss << ' ';
    bool first = true;
    const std::size_t prefix_end = has_encoded_boundary ? encoded_split : nodes.size();
    for (std::size_t node_index = 0; node_index < prefix_end; ++node_index) {
      const auto& node = nodes[node_index];
      append_fused_node_fragment(&br, &ss, node, branch_actual_indices[branch_index][node_index],
                                 branch_transform, &sess_opt, /*prepend_link=*/!first);
      first = false;
    }
    if (has_encoded_output) {
      // This private identity is the public source fan-out boundary made
      // concrete inside the fused pipeline. A single graph-owned probe on its
      // src pad retains the AU for Run::pull(); no capsfilter-name heuristic or
      // second encoded branch is needed.
      ss << " ! identity name=" << fused_encoded_output_tap_name(name_transform, branch_index)
         << " silent=true";
    }
    if (has_encoded_sink) {
      const std::string tee_name = apply_name_transform(branch_transform, "neat_encoded_tee");
      ss << " ! tee name=" << tee_name;

      // Preserve the public edge policy on the encoded branch. Default is
      // lossless and may backpressure at this one-AU compressed queue;
      // RealtimeLatestByStream keeps the producer non-blocking by replacing a
      // whole old AU. This queue contains no decoded EV memory.
      ss << ' ' << tee_name << ". ! queue max-size-buffers=1 max-size-bytes=0 "
         << "max-size-time=0";
      if (ingress.branches[branch_index].encoded_sink_link_options.policy ==
          GraphLinkPolicy::RealtimeLatestByStream) {
        ss << " leaky=downstream";
      }
      for (const auto& node : ingress.branches[branch_index].encoded_sink_nodes) {
        append_fused_node_fragment(&br, &ss, node, encoded_actual_index++, branch_transform,
                                   &sess_opt, /*prepend_link=*/true);
      }

      // The decoder branch is lossless. If the encoded prefix already ends in
      // the framework's Queue, its worker is the tee's decoder-side task
      // boundary while the egress branch has its own queue. Otherwise add one
      // compressed-AU queue so a decoder cannot run on the RTSP source thread.
      // Recognize the concrete Queue type rather than kind()=="Queue": custom
      // Nodes may use that label without providing the same scheduling contract.
      const bool prefix_ends_in_typed_queue =
          encoded_split > 0U &&
          dynamic_cast<const simaai::neat::Queue*>(nodes[encoded_split - 1U].get()) != nullptr;
      ss << ' ' << tee_name << ".";
      if (!prefix_ends_in_typed_queue) {
        ss << " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0";
      }
      for (std::size_t node_index = encoded_split; node_index < nodes.size(); ++node_index) {
        append_fused_node_fragment(&br, &ss, nodes[node_index],
                                   branch_actual_indices[branch_index][node_index],
                                   branch_transform, &sess_opt, /*prepend_link=*/true);
      }
    } else if (has_encoded_output) {
      for (std::size_t node_index = encoded_split; node_index < nodes.size(); ++node_index) {
        append_fused_node_fragment(&br, &ss, nodes[node_index],
                                   branch_actual_indices[branch_index][node_index],
                                   branch_transform, &sess_opt, /*prepend_link=*/true);
      }
    }
    ss << " ! " << mux_name << ".sink_" << branch_index;
  }

  br.diag->pipeline_string = ss.str();
  br.pipeline_string = br.diag->pipeline_string;
  return br;
}

namespace session_test {

Sample make_fused_encoded_output_sample_for_test(GstBuffer* buffer, GstCaps* caps,
                                                 const std::string& stream_id, bool copy_output) {
  return make_fused_encoded_output_sample(buffer, caps, stream_id, copy_output);
}

std::string fused_encoded_output_tap_name_for_test(const GraphOptions& options,
                                                   std::size_t branch_index) {
  return fused_encoded_output_tap_name(make_name_transform(options), branch_index);
}

std::size_t attach_fused_encoded_output_probe_for_test(
    GstElement* pipeline, const runtime::FusedRealtimeIngress& ingress, const GraphOptions& options,
    const std::function<void(const Sample&)>& observe, bool copy_output) {
  runtime::FusedEncodedOutputDispatch dispatch;
  if (observe) {
    dispatch = [observe](const runtime::FusedRealtimeIngressBranch::EncodedOutput&, Sample&& sample,
                         std::string*) {
      observe(sample);
      return runtime::FusedEncodedOutputDispatchResult::Delivered;
    };
  }
  return attach_fused_encoded_output_probes(pipeline, ingress, dispatch, copy_output,
                                            make_name_transform(options));
}

std::optional<std::size_t>
find_fused_decoder_timing_match_for_test(const std::vector<std::uint64_t>& pending_pts,
                                         std::optional<std::uint64_t> output_pts) {
  std::deque<SampleTimingOverrides> pending;
  for (const std::uint64_t pts : pending_pts) {
    SampleTimingOverrides timing;
    timing.pts_ns = pts;
    pending.push_back(std::move(timing));
  }
  return find_fused_decoder_timing_match(pending, output_pts, std::nullopt);
}

std::size_t
attach_fused_decoder_timing_probes_for_test(GstElement* pipeline,
                                            const runtime::FusedRealtimeIngress& ingress) {
  return attach_fused_decoder_timing_probes(pipeline, ingress);
}

GstBuffer* make_fused_terminal_probe_buffer_writable_for_test(GstPadProbeInfo* info) {
  return make_fused_terminal_probe_buffer_writable(info);
}

std::string render_fused_realtime_consumer_pipeline_for_test(
    const std::vector<std::shared_ptr<Node>>& consumer_nodes, const GraphOptions& options) {
  runtime::FusedRealtimeIngress ingress;
  runtime::FusedRealtimeIngressBranch branch;
  branch.stream_id = "stream0";
  ingress.branches.push_back(std::move(branch));
  const std::vector<std::vector<std::shared_ptr<Node>>> branch_nodes(1);
  const std::vector<std::vector<int>> branch_actual_indices(1);
  return build_fused_realtime_source_pipeline(ingress, consumer_nodes, branch_nodes,
                                              branch_actual_indices, NameTransform{}, "mysink",
                                              options, /*enable_terminal_loans=*/true)
      .pipeline_string;
}

std::string render_fused_realtime_consumer_pipeline_for_test(
    const std::vector<std::shared_ptr<Node>>& consumer_nodes, const GraphOptions& options,
    const std::vector<GraphLinkOptions>& link_options) {
  runtime::FusedRealtimeIngress ingress;
  for (std::size_t i = 0; i < link_options.size(); ++i) {
    runtime::FusedRealtimeIngressBranch branch;
    branch.stream_id = "stream" + std::to_string(i);
    branch.link_options = link_options[i];
    ingress.branches.push_back(std::move(branch));
  }
  const std::vector<std::vector<std::shared_ptr<Node>>> branch_nodes(link_options.size());
  const std::vector<std::vector<int>> branch_actual_indices(link_options.size());
  return build_fused_realtime_source_pipeline(ingress, consumer_nodes, branch_nodes,
                                              branch_actual_indices, NameTransform{},
                                              "neat_test_output", options,
                                              /*enable_terminal_loans=*/true)
      .pipeline_string;
}

std::string render_fused_realtime_consumer_pipeline_for_test(
    const std::vector<std::shared_ptr<Node>>& consumer_nodes, const GraphOptions& options,
    const std::vector<GraphLinkOptions>& link_options, bool enable_terminal_loans) {
  runtime::FusedRealtimeIngress ingress;
  for (std::size_t i = 0; i < link_options.size(); ++i) {
    runtime::FusedRealtimeIngressBranch branch;
    branch.stream_id = "stream" + std::to_string(i);
    branch.link_options = link_options[i];
    ingress.branches.push_back(std::move(branch));
  }
  const std::vector<std::vector<std::shared_ptr<Node>>> branch_nodes(link_options.size());
  const std::vector<std::vector<int>> branch_actual_indices(link_options.size());
  return build_fused_realtime_source_pipeline(ingress, consumer_nodes, branch_nodes,
                                              branch_actual_indices, NameTransform{},
                                              "neat_test_output", options, enable_terminal_loans)
      .pipeline_string;
}

std::string
render_fused_realtime_pipeline_for_test(const runtime::FusedRealtimeIngress& ingress,
                                        const std::vector<std::shared_ptr<Node>>& consumer_nodes,
                                        const GraphOptions& options) {
  std::vector<std::vector<std::shared_ptr<Node>>> branch_nodes;
  std::vector<std::vector<int>> branch_actual_indices;
  branch_nodes.reserve(ingress.branches.size());
  branch_actual_indices.reserve(ingress.branches.size());

  int next_actual_index = static_cast<int>(consumer_nodes.size());
  for (const auto& branch : ingress.branches) {
    branch_nodes.push_back(branch.nodes);
    std::vector<int> indices;
    indices.reserve(branch.nodes.size());
    for (std::size_t i = 0; i < branch.nodes.size(); ++i) {
      indices.push_back(next_actual_index++);
    }
    branch_actual_indices.push_back(std::move(indices));
  }

  return build_fused_realtime_source_pipeline(ingress, consumer_nodes, branch_nodes,
                                              branch_actual_indices, make_name_transform(options),
                                              "neat_test_output", options,
                                              /*enable_terminal_loans=*/true)
      .pipeline_string;
}

} // namespace session_test

SourceStreamBuildContext session_build_fused_realtime_source_stream_internal(
    const runtime::FusedRealtimeIngress& ingress,
    const std::vector<std::shared_ptr<Node>>& consumer_nodes, const std::shared_ptr<void>& guard,
    std::string& last_pipeline, const GraphOptions& sess_opt, const RunOptions& opt, RunMode mode,
    bool require_sink, bool public_output_contract, const char* where,
    const runtime::FusedEncodedOutputDispatch& encoded_output_dispatch) {
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
  graph_build_internal::apply_explicit_public_output_options(stream_opt, build_consumer_nodes);
  stream_opt.prefer_synchronous_teardown = true;
  session_build_maybe_enable_rtsp_appsink_drop(stream_opt, build_consumer_nodes,
                                               build_branch_nodes);

  const NameTransform name_transform = make_name_transform(sess_opt);
  std::vector<NameTransform> branch_name_transforms;
  branch_name_transforms.reserve(build_branch_nodes.size());
  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    branch_name_transforms.push_back(fused_branch_name_transform(name_transform, branch_index));
  }
  std::vector<std::vector<int>> branch_actual_indices;
  branch_actual_indices.reserve(build_branch_nodes.size());
  int next_branch_actual_index = static_cast<int>(build_consumer_nodes.size());
  for (const auto& branch_nodes : build_branch_nodes) {
    // Fused branches are rendered after the consumer nodes in one pipeline, so
    // any post-parse probes must use these global indices to find element names.
    std::vector<int> indices;
    indices.reserve(branch_nodes.size());
    for (std::size_t i = 0; i < branch_nodes.size(); ++i) {
      indices.push_back(next_branch_actual_index++);
    }
    branch_actual_indices.push_back(std::move(indices));
  }
  BuildResult br = build_fused_realtime_source_pipeline(ingress, build_consumer_nodes,
                                                        build_branch_nodes, branch_actual_indices,
                                                        name_transform, "mysink", sess_opt,
                                                        /*enable_terminal_loans=*/has_sink);
  maybe_compile_fused_realtime_contracts(&br, build_consumer_nodes, build_branch_nodes,
                                         branch_actual_indices, sess_opt, where,
                                         fused_ingress_spec);
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
  attach_fused_decoder_timing_probes(pipeline.get(), ingress);
  attach_fused_encoded_output_probes(pipeline.get(), ingress, encoded_output_dispatch,
                                     stream_opt.copy_output, name_transform);

  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    std::vector<std::shared_ptr<Node>> logical = build_branch_nodes[branch_index];
    logical.insert(logical.end(), build_consumer_nodes.begin(), build_consumer_nodes.end());
    for (std::size_t i = 0; i < build_branch_nodes[branch_index].size(); ++i) {
      attach_source_sima_meta_probe_for_node(pipeline.get(), logical, i,
                                             branch_actual_indices[branch_index][i],
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
  session_build_attach_h264_caps_fixups(pipeline.get(), build_consumer_nodes, name_transform);
  session_build_attach_encoded_caps_fixups(pipeline.get(), build_consumer_nodes, name_transform);
  session_build_attach_rtsp_debug(pipeline.get(), build_consumer_nodes, name_transform);
  for (std::size_t branch_index = 0; branch_index < build_branch_nodes.size(); ++branch_index) {
    session_build_attach_h264_caps_fixups(pipeline.get(), build_branch_nodes[branch_index],
                                          branch_name_transforms[branch_index],
                                          &branch_actual_indices[branch_index]);
    session_build_attach_encoded_caps_fixups(pipeline.get(), build_branch_nodes[branch_index],
                                             branch_name_transforms[branch_index],
                                             &branch_actual_indices[branch_index]);
    session_build_attach_rtsp_debug(pipeline.get(), build_branch_nodes[branch_index],
                                    branch_name_transforms[branch_index],
                                    &branch_actual_indices[branch_index]);
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
    const std::string mux_name = apply_name_transform(name_transform, "neat_live_mux");
    GstElement* raw_mux = gst_bin_get_by_name(GST_BIN(pipeline.get()), mux_name.c_str());
    const std::uint64_t mux_namespace = latest_by_stream_mux_namespace(raw_mux);
    // Fused CVU/MLA/decode stages allocate replacement GstBuffers and do not
    // invoke arbitrary GstMeta transform callbacks.  A lifetime guard on the
    // decoded input would therefore mistake successful stage consumption for a
    // downstream drop and finalize its admission/timing loan before the derived
    // terminal buffer arrives.  The ordered/non-dropping contract
    // checked while rendering makes the explicit appsink completion probe below
    // authoritative for such chains.  Identity-preserving chains retain the
    // guard, including its pre-terminal drop protection.
    const bool lifetime_guard_configured =
        raw_mux &&
        (!br.fused_consumer_replaces_buffers ||
         pipeline_internal::set_latest_by_stream_mux_lifetime_guard_enabled(raw_mux,
                                                                            /*enabled=*/false));
    if (raw_mux) {
      gst_object_unref(raw_mux);
    }
    if (mux_namespace == 0 || !lifetime_guard_configured) {
      session_build_throw_session_error_simple(
          error_codes::kPipelineShape,
          std::string(where ? where : "Graph::build(fused)") + ": latest-by-stream mux '" +
              mux_name + "' could not configure terminal loan ownership.\nPipeline:\n" +
              last_pipeline,
          "Keep the fused latest-by-stream mux in the source pipeline.", last_pipeline);
    }
    // A fused mux loan covers work from mux emission through the terminal
    // consumer pipeline. Release it as soon as the result reaches GstAppSink's
    // sink pad, before GstAppSink can retain it as preroll or discard it under
    // its own bounded-queue policy. Waiting for Run::pull() is insufficient:
    // samples consumed internally by GstAppSink never reach that path, and
    // decoder buffer pools can recycle GstBuffers without finalizing them.
    if (GstPad* terminal_pad = gst_element_get_static_pad(raw_sink, "sink")) {
      const gulong probe_id = gst_pad_add_probe(
          terminal_pad, GST_PAD_PROBE_TYPE_BUFFER,
          +[](GstPad*, GstPadProbeInfo* info, gpointer user_data) -> GstPadProbeReturn {
            GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
            const auto* mux_namespace = static_cast<const std::uint64_t*>(user_data);
            if (!buffer || !mux_namespace || *mux_namespace == 0) {
              return GST_PAD_PROBE_OK;
            }

            // The release helper restores the selected decoder timing on the
            // terminal result before GstAppSink sees it. Pad probes may be
            // handed a shared GstBuffer, so take GStreamer's copy-on-write
            // path only when needed and replace the probe data with the
            // writable buffer before mutating timestamps or custom metadata.
            buffer = make_fused_terminal_probe_buffer_writable(info);
            if (!buffer) {
              return GST_PAD_PROBE_OK;
            }
            (void)pipeline_internal::release_latest_by_stream_mux_loan_for_buffer(buffer,
                                                                                  *mux_namespace);
            return GST_PAD_PROBE_OK;
          },
          new std::uint64_t(mux_namespace),
          +[](gpointer data) { delete static_cast<std::uint64_t*>(data); });
      if (pipeline_internal::env_bool("SIMA_LATEST_MUX_LOAN_DEBUG", false)) {
        std::fprintf(stderr,
                     "[latestmux][loan] terminal appsink release probe attached id=%lu sink=%s "
                     "namespace=%llu\n",
                     static_cast<unsigned long>(probe_id), appsink_name.c_str(),
                     static_cast<unsigned long long>(mux_namespace));
      }
      gst_object_unref(terminal_pad);
    }
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
  return build_source_internal_(opt);
}

Run Graph::build_source_internal_(const RunOptions& opt) {
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
