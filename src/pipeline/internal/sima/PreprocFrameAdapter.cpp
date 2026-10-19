#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include <simaai/scheduler/client/diagnostics.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <simaai/scheduler/client/video_layout.h>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
struct PreprocFrameAdapter::CheckedPolicy {
  std::shared_ptr<const ProcessCvuStagePayload> prepared;
  KernelExecutable initial_executable;
  std::optional<KernelVariant> geometry_variant;
  TensorSpec output;
  std::vector<TensorSpec> outputs;
  std::array<std::size_t, 3> storage_rows{};
  std::shared_ptr<const gstreamer::TensorStageImagePublication> image_publication;
  bool identity_interpolation = false;
  Geometry envelope, initial_geometry;
  PaddingPlacement padding = PaddingPlacement::Center;
};
PreprocFrameAdapter::PreprocFrameAdapter(std::shared_ptr<const CheckedPolicy> policy)
    : policy_(std::move(policy)), default_geometry_(policy_->initial_geometry) {}
const KernelExecutable& PreprocFrameAdapter::initial_executable() const {
  return policy_->initial_executable;
}
const KernelExecutable& PreprocFrameAdapter::checked_executable(const CheckedPolicy& policy) {
  return policy.initial_executable;
}

std::shared_ptr<const gstreamer::TensorStageImagePublication>
PreprocFrameAdapter::checked_image_publication(const CheckedPolicy& policy) {
  return policy.image_publication;
}

using namespace ::simaai::scheduler::client;
namespace {

Error invalid(const char* field, std::string message) {
  return {ErrorCode::InvalidRequest, field, std::move(message)};
}
// Checked fact integers preserve their signedness in the canonical vocabulary.
// This is readback validation, never a second ROI filtering policy.
std::optional<std::int64_t> integer_fact(const ParameterScalar& scalar) {
  if (const auto* value = std::get_if<std::int64_t>(&scalar.value))
    return *value;
  if (const auto* value = std::get_if<std::uint64_t>(&scalar.value);
      value && *value <= std::uint64_t(INT64_MAX))
    return std::int64_t(*value);
  return std::nullopt;
}
std::string upper(std::string value) {
  for (auto& c : value)
    c = g_ascii_toupper(c);
  return value;
}
std::string format_name(GstVideoFormat format) {
  switch (format) {
  case GST_VIDEO_FORMAT_RGB:
    return "rgb";
  case GST_VIDEO_FORMAT_BGR:
    return "bgr";
  case GST_VIDEO_FORMAT_GRAY8:
    return "gray";
  case GST_VIDEO_FORMAT_NV12:
    return "nv12";
  case GST_VIDEO_FORMAT_NV16:
    return "nv16";
  case GST_VIDEO_FORMAT_I420:
    return "i420";
  default:
    return {};
  }
}
std::size_t planes(GstVideoFormat format) {
  return format == GST_VIDEO_FORMAT_I420
             ? 3
             : (format == GST_VIDEO_FORMAT_NV12 || format == GST_VIDEO_FORMAT_NV16 ? 2 : 1);
}
std::size_t channels(GstVideoFormat format) {
  return format == GST_VIDEO_FORMAT_RGB || format == GST_VIDEO_FORMAT_BGR ? 3 : 1;
}
std::int64_t axis(const sima_ev_tensor_desc& desc, std::uint8_t semantic) {
  for (std::size_t i = 0; i < desc.shape.rank; ++i)
    if (desc.shape.axis_semantics[i] == semantic)
      return desc.shape.sizes[i];
  return 0;
}
bool roi_metadata(const GstStructure* meta) {
  return meta && (gst_structure_has_field(meta, "preproc_roi_list_enable") ||
                  gst_structure_has_field(meta, "preproc_roi_count") ||
                  gst_structure_has_field(meta, "preproc_roi_rects") ||
                  gst_structure_has_field(meta, "preproc_roi_batch_indices") ||
                  gst_structure_has_field(meta, "preproc_roi_enable"));
}
bool roi_list_enabled(const GstStructure* meta) {
  if (!meta)
    return false;
  gboolean enabled = gst_structure_has_field(meta, "preproc_roi_count") ||
                     gst_structure_has_field(meta, "preproc_roi_rects") ||
                     gst_structure_has_field(meta, "preproc_roi_batch_indices");
  gst_structure_get_boolean(meta, "preproc_roi_list_enable", &enabled);
  return enabled;
}
using Rectangle = std::array<std::int64_t, 4>;
// Only numerical ROI inputs belong in the immutable-variant key. Never retain
// a frame/carrier, correlation fields, or a borrowed GValue. Exact types
// matter: accepting an INT64 list as equal to an INT list would bypass input
// validation.
enum class SignatureType { Integer, Boolean, Rectangles, Indices };
struct SignatureField {
  const char* name;
  SignatureType type = SignatureType::Integer;
};
// Check the coordinates/records first: moving ROIs reject a previous signature
// without repeatedly looking up all the unchanged mode flags.
constexpr std::array signature_fields{
    SignatureField{"preproc_roi_x"},
    SignatureField{"preproc_roi_y"},
    SignatureField{"preproc_roi_rects", SignatureType::Rectangles},
    SignatureField{"preproc_roi_batch_indices", SignatureType::Indices},
    SignatureField{"preproc_roi_width"},
    SignatureField{"preproc_roi_height"},
    SignatureField{"preproc_roi_pad_value"},
    SignatureField{"preproc_roi_enable", SignatureType::Boolean},
    SignatureField{"preproc_roi_list_enable", SignatureType::Boolean},
    SignatureField{"preproc_roi_count"}};
const GValue* field(const GstStructure* meta, const char* name) {
  return meta ? gst_structure_get_value(meta, name) : nullptr;
}
bool same_value(const GValue* a, const GValue* b) {
  if (!a || !b)
    return a == b;
  if (G_VALUE_TYPE(a) != G_VALUE_TYPE(b))
    return false;
  const bool array = GST_VALUE_HOLDS_ARRAY(a);
  if (array || GST_VALUE_HOLDS_LIST(a)) {
    const auto size = array ? gst_value_array_get_size(a) : gst_value_list_get_size(a);
    if (size != (array ? gst_value_array_get_size(b) : gst_value_list_get_size(b)))
      return false;
    for (guint i = 0; i < size; ++i) {
      const auto* left = array ? gst_value_array_get_value(a, i) : gst_value_list_get_value(a, i);
      const auto* right = array ? gst_value_array_get_value(b, i) : gst_value_list_get_value(b, i);
      if (!G_VALUE_HOLDS_INT(left) || !G_VALUE_HOLDS_INT(right) ||
          g_value_get_int(left) != g_value_get_int(right))
        return false;
    }
    return true;
  }
  return gst_value_compare(a, b) == GST_VALUE_EQUAL;
}
bool same_signature(const GstStructure* a, const GstStructure* b) {
  return std::all_of(signature_fields.begin(), signature_fields.end(), [&](const auto& entry) {
    return same_value(field(a, entry.name), field(b, entry.name));
  });
}
GstStructure* copy_signature(const GstStructure* source) {
  // Ignored malformed optional fields retain their existing behavior, but are
  // not cached. Thus the cache itself is strictly bounded even for foreign
  // meta.
  for (const auto& entry : signature_fields) {
    const auto* value = field(source, entry.name);
    if (!value)
      continue;
    if (entry.type == SignatureType::Boolean) {
      if (!G_VALUE_HOLDS_BOOLEAN(value))
        return nullptr;
    } else if (entry.type == SignatureType::Rectangles || entry.type == SignatureType::Indices) {
      const bool array = GST_VALUE_HOLDS_ARRAY(value);
      if (!array && !GST_VALUE_HOLDS_LIST(value))
        return nullptr;
      const auto count = array ? gst_value_array_get_size(value) : gst_value_list_get_size(value);
      if (count > (entry.type == SignatureType::Rectangles ? 16384U : 4096U))
        return nullptr;
      for (guint j = 0; j < count; ++j)
        if (!G_VALUE_HOLDS_INT(array ? gst_value_array_get_value(value, j)
                                     : gst_value_list_get_value(value, j)))
          return nullptr;
    } else if (!G_VALUE_HOLDS_INT(value))
      return nullptr;
  }
  auto* copy = gst_structure_new_empty("preproc-invocation");
  for (const auto& entry : signature_fields)
    if (const auto* value = field(source, entry.name))
      gst_structure_set_value(copy, entry.name, value);
  return copy;
}
// Publication names are fixed by the legacy wire contract. Intern them once
// at preparation, rather than hashing every scalar field on every frame.
// GLib owns the copied names even if the compatibility module is unloaded.
struct PublicationKeys {
  GQuark roi_batch_indices = g_quark_from_string("preproc_roi_batch_indices");
  GQuark roi_rects = g_quark_from_string("preproc_roi_rects");
  GQuark roi_affines = g_quark_from_string("preproc_roi_affines");
  GQuark original_width = g_quark_from_string("preproc_original_width");
  GQuark original_height = g_quark_from_string("preproc_original_height");
  GQuark resized_width = g_quark_from_string("preproc_resized_width");
  GQuark resized_height = g_quark_from_string("preproc_resized_height");
  GQuark scaled_width = g_quark_from_string("preproc_scaled_width");
  GQuark scaled_height = g_quark_from_string("preproc_scaled_height");
  GQuark pad_left = g_quark_from_string("preproc_pad_left");
  GQuark pad_top = g_quark_from_string("preproc_pad_top");
  GQuark pad_right = g_quark_from_string("preproc_pad_right");
  GQuark pad_bottom = g_quark_from_string("preproc_pad_bottom");
  GQuark resize_mode = g_quark_from_string("preproc_resize_mode");
  GQuark color_in = g_quark_from_string("preproc_color_in");
  GQuark color_out = g_quark_from_string("preproc_color_out");
  GQuark normalize = g_quark_from_string("preproc_normalize");
  GQuark quantize = g_quark_from_string("preproc_quantize");
  GQuark tessellate = g_quark_from_string("preproc_tessellate");
  GQuark affine_m00 = g_quark_from_string("preproc_affine_m00");
  GQuark affine_m01 = g_quark_from_string("preproc_affine_m01");
  GQuark affine_m02 = g_quark_from_string("preproc_affine_m02");
  GQuark affine_m10 = g_quark_from_string("preproc_affine_m10");
  GQuark affine_m11 = g_quark_from_string("preproc_affine_m11");
  GQuark affine_m12 = g_quark_from_string("preproc_affine_m12");
  GQuark affine_scale_x = g_quark_from_string("preproc_affine_scale_x");
  GQuark affine_scale_y = g_quark_from_string("preproc_affine_scale_y");
  GQuark affine_offset_x = g_quark_from_string("preproc_affine_offset_x");
  GQuark affine_offset_y = g_quark_from_string("preproc_affine_offset_y");
  GQuark roi_enable = g_quark_from_string("preproc_roi_enable");
  GQuark roi_x = g_quark_from_string("preproc_roi_x");
  GQuark roi_y = g_quark_from_string("preproc_roi_y");
  GQuark roi_width = g_quark_from_string("preproc_roi_width");
  GQuark roi_height = g_quark_from_string("preproc_roi_height");
  GQuark roi_source_width = g_quark_from_string("preproc_roi_source_width");
  GQuark roi_source_height = g_quark_from_string("preproc_roi_source_height");
  GQuark roi_batch_index = g_quark_from_string("preproc_roi_batch_index");
  GQuark roi_list_enable = g_quark_from_string("preproc_roi_list_enable");
  GQuark roi_count = g_quark_from_string("preproc_roi_count");
  GQuark valid_batch_count = g_quark_from_string("preproc_valid_batch_count");
  GQuark roi_capacity = g_quark_from_string("preproc_roi_capacity");
  GQuark roi_input_batch_size = g_quark_from_string("preproc_roi_input_batch_size");
  GQuark roi_input_count = g_quark_from_string("preproc_roi_input_count");
  GQuark roi_dropped_invalid = g_quark_from_string("preproc_roi_dropped_invalid");
  GQuark roi_dropped_overflow = g_quark_from_string("preproc_roi_dropped_overflow");
  GQuark roi_source_stride_bytes = g_quark_from_string("preproc_roi_source_stride_bytes");
  GQuark roi_pad_value = g_quark_from_string("preproc_roi_pad_value");
};
const PublicationKeys& publication_keys() {
  static const PublicationKeys keys;
  return keys;
}
const auto& owned_publication_keys() {
  static const auto owned = [] {
    const auto& keys = publication_keys();
    std::array result{
        keys.roi_batch_indices,
        keys.roi_rects,
        keys.roi_affines,
        keys.original_width,
        keys.original_height,
        keys.resized_width,
        keys.resized_height,
        keys.scaled_width,
        keys.scaled_height,
        keys.pad_left,
        keys.pad_top,
        keys.pad_right,
        keys.pad_bottom,
        keys.resize_mode,
        keys.color_in,
        keys.color_out,
        keys.normalize,
        keys.quantize,
        keys.tessellate,
        keys.affine_m00,
        keys.affine_m01,
        keys.affine_m02,
        keys.affine_m10,
        keys.affine_m11,
        keys.affine_m12,
        keys.affine_scale_x,
        keys.affine_scale_y,
        keys.affine_offset_x,
        keys.affine_offset_y,
        keys.roi_enable,
        keys.roi_x,
        keys.roi_y,
        keys.roi_width,
        keys.roi_height,
        keys.roi_source_width,
        keys.roi_source_height,
        keys.roi_batch_index,
        keys.roi_list_enable,
        keys.roi_count,
        keys.valid_batch_count,
        keys.roi_capacity,
        keys.roi_input_batch_size,
        keys.roi_input_count,
        keys.roi_dropped_invalid,
        keys.roi_dropped_overflow,
        keys.roi_source_stride_bytes,
        keys.roi_pad_value,
    };
    std::sort(result.begin(), result.end());
    return result;
  }();
  return owned;
}

} // namespace

// A validated borrowed list, used only during this bind. Variants retain their
// own immutable signatures; no frame-owned GValue or loan is cached.
struct PreprocFrameAdapter::IntegerList {
  const GValue* value = nullptr;
  bool array = false;
  std::size_t source_batch = 1;
  int operator[](std::size_t index) const {
    return value ? g_value_get_int(array ? gst_value_array_get_value(value, index)
                                         : gst_value_list_get_value(value, index))
                 : 0;
  }
};
Result<PreprocFrameAdapter::IntegerList> PreprocFrameAdapter::integer_list(const GstStructure* meta,
                                                                           const char* name,
                                                                           std::size_t count,
                                                                           bool optional) {
  const GValue* value = meta ? gst_structure_get_value(meta, name) : nullptr;
  if (!value) {
    if (optional || count == 0)
      return IntegerList{};
    return invalid(name, "required ROI list is missing");
  }
  const bool array = GST_VALUE_HOLDS_ARRAY(value);
  if (!array && !GST_VALUE_HOLDS_LIST(value))
    return invalid(name, "ROI field must be an integer list");
  const auto size = array ? gst_value_array_get_size(value) : gst_value_list_get_size(value);
  if (size != count)
    return invalid(name, "ROI list length differs from declared count");
  IntegerList result{value, array};
  for (std::size_t i = 0; i < count; ++i) {
    const auto* entry =
        array ? gst_value_array_get_value(value, i) : gst_value_list_get_value(value, i);
    if (!G_VALUE_HOLDS_INT(entry))
      return invalid(name, "ROI entries must be signed integers");
    const int index = g_value_get_int(entry);
    if (index >= 0)
      result.source_batch = std::max(result.source_batch, std::size_t(index) + 1);
  }
  return result;
}
bool PreprocFrameAdapter::owns_publication_field(GQuark field) noexcept {
  const auto& owned = owned_publication_keys();
  return std::binary_search(owned.begin(), owned.end(), field);
}

Result<Rectangle> PreprocFrameAdapter::destination(std::int64_t source_w, std::int64_t source_h,
                                                   std::int64_t w, std::int64_t h) const {
  const auto& cfg = *policy_->prepared;
  if (source_w <= 0 || source_h <= 0)
    return invalid("source_roi", "ROI dimensions must be positive");
  std::int64_t scaled_w = cfg.scaled_width > 0 ? cfg.scaled_width : w;
  std::int64_t scaled_h = cfg.scaled_height > 0 ? cfg.scaled_height : h;
  if (cfg.aspect_ratio == 1) {
    // Legacy contain geometry rounds the non-limiting axis upward in FP32.
    if (w * source_h < h * source_w) {
      scaled_w = w;
      scaled_h = std::int64_t(std::ceil(float(source_h * w) / float(source_w)));
    } else {
      scaled_h = h;
      scaled_w = std::int64_t(std::ceil(float(source_w * h) / float(source_h)));
    }
  }
  if (policy_->identity_interpolation) {
    // Legacy NO_SCALING copies the top-left source subset, never resizes it.
    scaled_w = std::min(source_w, w);
    scaled_h = std::min(source_h, h);
  }
  if (scaled_w <= 0 || scaled_h <= 0 || scaled_w > w || scaled_h > h)
    return invalid("destination", "scaled rectangle exceeds output canvas");
  const auto x = policy_->padding == PaddingPlacement::Center
                     ? (w - scaled_w) / 2
                     : (policy_->padding == PaddingPlacement::TopRight ||
                                policy_->padding == PaddingPlacement::BottomRight
                            ? w - scaled_w
                            : 0);
  const auto y = policy_->padding == PaddingPlacement::Center
                     ? (h - scaled_h) / 2
                     : (policy_->padding == PaddingPlacement::BottomLeft ||
                                policy_->padding == PaddingPlacement::BottomRight
                            ? h - scaled_h
                            : 0);
  return Rectangle{x, y, scaled_w, scaled_h};
}
Result<PreprocFrameAdapter::Geometry>
PreprocFrameAdapter::describe_geometry(Geometry geometry, std::size_t width, std::size_t height) {
  GstVideoInfo info;
  gst_video_info_init(&info);
  if (!gst_video_info_set_format(&info, geometry.format, width, height))
    return invalid("input.geometry", "invalid source format dimensions");
  geometry.width = width;
  geometry.height = height;
  geometry.frame_bytes = info.size;
  for (std::size_t i = 0; i < planes(geometry.format); ++i) {
    geometry.offsets[i] = info.offset[i];
    geometry.strides[i] = info.stride[i];
  }
  return geometry;
}

Result<std::shared_ptr<const PreprocFrameAdapter::CheckedPolicy>>
PreprocFrameAdapter::check_policy(std::shared_ptr<const ProcessCvuStagePayload> prepared) {
  if (!prepared)
    return invalid("stage.contract", "Preproc requires owned Core model policy");
  return check_policy(*prepared);
}

Result<std::shared_ptr<const PreprocFrameAdapter::CheckedPolicy>>
PreprocFrameAdapter::check_policy(const ProcessCvuStagePayload& prepared) {
  // Snapshot application values once; mutable caller aliases cannot change a
  // previously checked recipe or later execution instances.
  auto policy = std::make_shared<CheckedPolicy>();
  policy->prepared = std::make_shared<const ProcessCvuStagePayload>(prepared);
  const auto& cfg = *policy->prepared;
  if (cfg.graph_name != "preproc")
    return invalid("graph", "not the canonical prepared PreprocFrameAdapter contract");
  if (cfg.resolved_exec_backend == "A65")
    return Error{ErrorCode::UnsupportedTarget, "target",
                 "PreprocFrameAdapter has no A65 implementation"};
  if (cfg.resolved_exec_backend != "EV74" && cfg.resolved_exec_backend != "EVXX")
    return invalid("target", "Core must resolve the PreprocFrameAdapter target");
  if (cfg.input_tensors.size() != 1 || cfg.output_tensors.empty())
    return invalid("tensors",
                   "PreprocFrameAdapter requires one authored source envelope and selected output");
  // This cold builder only calls geometry/schema helpers. It never prepares a
  // session; the returned policy cannot retain this temporary adapter.
  auto result = std::shared_ptr<PreprocFrameAdapter>(new PreprocFrameAdapter(policy));
  auto format = upper(cfg.input_img_type);
  if (format == "GRAY")
    format = "GRAY8";
  if (format == "IYUV")
    format = "I420";
  policy->envelope.format = gst_video_format_from_string(format.c_str());
  const auto width = axis(cfg.input_tensors[0], SIMA_EV_AXIS_W);
  const auto height = axis(cfg.input_tensors[0], SIMA_EV_AXIS_H);
  if (format_name(policy->envelope.format).empty() || width <= 0 || height <= 0 ||
      width > G_MAXINT || height > G_MAXINT || cfg.batch_size <= 0)
    return invalid("input", "invalid image format, capacity envelope or batch");
  policy->envelope.width = width;
  policy->envelope.height = height;
  policy->envelope.batch = cfg.batch_size;
  std::size_t selected = 0;
  if (cfg.output_tensors.size() > 1) {
    const auto output = std::find(cfg.default_output_names.begin(), cfg.default_output_names.end(),
                                  policy->prepared->primary_output_name);
    if (output == cfg.default_output_names.end())
      return invalid("output",
                     "multiple PreprocFrameAdapter outputs require an explicit selected handoff");
    selected = std::size_t(output - cfg.default_output_names.begin());
  }
  if (selected >= cfg.output_tensors.size())
    return invalid("output", "selected descriptor is missing");
  auto output = tensor_spec_from_model_descriptor(cfg.output_tensors[selected]);
  if (auto* error = std::get_if<Error>(&output))
    return *error;
  policy->output = std::get<TensorSpec>(std::move(output));
  if (cfg.normalize != 1 && policy->output.dtype == DType::Int8) {
    // Legacy raw INT8 names byte-preserving [0,255] storage, not signed math.
    // Native UInt8 explicitly preserves those bytes in dense or tiled storage.
    policy->output.dtype = DType::UInt8;
  }
  const auto& output_descriptor = cfg.output_tensors[selected].descriptor();
  const auto output_batch = axis(output_descriptor, SIMA_EV_AXIS_N);
  if (output_batch != 0) {
    // The legacy contract may describe one output member even when its public
    // carrier holds several ROIs. Native admission derives capacity from the
    // tensor shape, so materialize that batch here, not once per invocation.
    // An already batched descriptor must agree with the declared capacity.
    if (output_descriptor.shape.axis_semantics[0] != SIMA_EV_AXIS_N ||
        (output_batch != 1 && output_batch != cfg.batch_size))
      return invalid("output.shape",
                     "output batch must be leading and match capacity or one member");
    auto shape = policy->output.shape;
    std::vector<std::int64_t> dimensions(shape.begin(), shape.end());
    dimensions[0] = cfg.batch_size;
    policy->output.shape = std::move(dimensions);
  } else {
    auto shape = policy->output.shape;
    if (shape.size() == std::size(output_descriptor.shape.sizes))
      return invalid("output.shape", "batch exceeds six axes");
    std::vector<std::int64_t> dimensions{cfg.batch_size};
    dimensions.insert(dimensions.end(), shape.begin(), shape.end());
    policy->output.shape = std::move(dimensions);
    if (auto* tile = std::get_if<::simaai::scheduler::client::MlaTiled>(&policy->output.layout)) {
      for (auto& value : tile->axes)
        if (value >= 0)
          ++value;
    } else if (auto* strides =
                   std::get_if<::simaai::scheduler::client::Strided>(&policy->output.layout)) {
      auto previous = strides->strides_bytes;
      for (std::size_t i = shape.size(); i > 0; --i)
        strides->strides_bytes[i] = previous[i - 1];
      strides->strides_bytes[0] = previous[0] * shape[0];
    }
  }
  if (cfg.image_transform) {
    if (cfg.batch_size != 1 || cfg.output_shapes.empty() || cfg.output_shapes[0].size() < 2)
      return invalid("output.shape",
                     "image frontends require one frame with explicit output geometry");
    const auto h = cfg.output_shapes[0][0], w = cfg.output_shapes[0][1];
    const auto output_format = upper(cfg.output_img_type);
    const auto format =
        gst_video_format_from_string(output_format == "GRAY" ? "GRAY8" : output_format.c_str());
    if (h <= 0 || w <= 0 || format_name(format).empty())
      return invalid("output.shape", "invalid image output geometry or format");
    const bool yuv = planes(format) > 1;
    if (yuv && ((w & 1) || (format != GST_VIDEO_FORMAT_NV16 && (h & 1))))
      return invalid("output.shape", "YUV output violates its chroma sampling grid");
    policy->output = {std::vector<std::int64_t>{1, h, w, std::int64_t(channels(format))},
                      cfg.image_transform->raw_output() ? DType::UInt8 : DType::BFloat16,
                      ::simaai::scheduler::client::Contiguous{}};
    if (cfg.image_transform->raw_output()) {
      auto publication = std::make_shared<gstreamer::TensorStageImagePublication>();
      publication->format = format;
      publication->width = w;
      publication->height = h;
      VideoLayout encoder;
      if (cfg.image_transform->surface == ImageSurfaceLayout::EncoderNv12) {
        if (format != GST_VIDEO_FORMAT_NV12)
          return invalid("output.surface", "encoder surface requires NV12");
        auto layout = encoder_nv12_layout(w, h);
        if (auto* error = std::get_if<Error>(&layout))
          return *error;
        encoder = std::get<VideoLayout>(std::move(layout));
      }
      for (std::size_t i = 0; i < planes(format); ++i) {
        const auto rows = i && format != GST_VIDEO_FORMAT_NV16 ? h / 2 : h;
        const auto columns = i ? w / 2 : w;
        const auto c = i ? (format == GST_VIDEO_FORMAT_I420 ? 1 : 2) : channels(format);
        const auto row_bytes = std::size_t(columns) * c;
        // Kernel DMA and same-owner range checks use explicit positive pitches.
        std::size_t pitch = yuv ? (row_bytes + 15) & ~std::size_t{15} : row_bytes;
        std::size_t storage_rows = rows;
        if (cfg.image_transform->surface == ImageSurfaceLayout::EncoderNv12) {
          pitch = encoder.planes[i].stride;
          storage_rows = i ? (encoder.allocation_size - encoder.planes[1].offset) / pitch
                           : encoder.planes[1].offset / pitch;
        }
        if (pitch > G_MAXINT || storage_rows > SIZE_MAX / pitch)
          return invalid("output.surface", "image surface extent overflows");
        policy->storage_rows[i] = storage_rows;
        Layout layout = ::simaai::scheduler::client::Contiguous{};
        if (yuv)
          layout = ::simaai::scheduler::client::Strided{
              {std::int64_t(pitch * storage_rows), std::int64_t(pitch), std::int64_t(c), 1}};
        policy->outputs.push_back(
            {std::vector<std::int64_t>{1, rows, columns, std::int64_t(c)}, DType::UInt8, layout});
        publication->planes.push_back({int(i), 0, gint(pitch), guint(rows), guint(row_bytes)});
      }
      policy->image_publication = std::move(publication);
    }
  }
  if (policy->outputs.empty())
    policy->outputs.push_back(policy->output);
  auto initial_geometry = describe_geometry(policy->envelope, width, height);
  if (auto* error = std::get_if<Error>(&initial_geometry))
    return *error;
  auto geometry = std::get<Geometry>(initial_geometry);
  const auto source_batch = axis(cfg.input_tensors[0], SIMA_EV_AXIS_N);
  geometry.batch = source_batch > 0 ? std::size_t(source_batch) : 1;
  policy->envelope.batch = std::max(policy->envelope.batch, geometry.batch);
  result->default_geometry_ = geometry;
  auto specs = result->input_specs(geometry);
  if (auto* error = std::get_if<Error>(&specs))
    return *error;
  auto constants = result->static_parameters();
  if (auto* error = std::get_if<Error>(&constants))
    return *error;
  policy->identity_interpolation = upper(cfg.scaling_type) == "NO_SCALING";
  const auto padding = upper(cfg.padding_type);
  if (padding == "CENTER")
    policy->padding = PaddingPlacement::Center;
  else if (padding == "TOPLEFT")
    policy->padding = PaddingPlacement::TopLeft;
  else if (padding == "TOPRIGHT")
    policy->padding = PaddingPlacement::TopRight;
  else if (padding == "BOTTOMLEFT")
    policy->padding = PaddingPlacement::BottomLeft;
  else if (padding == "BOTTOMRIGHT")
    policy->padding = PaddingPlacement::BottomRight;
  else
    return invalid("padding_type", "unknown or unsupported padding placement");
  (void)owned_publication_keys();
  auto params = result->parameters(geometry, nullptr);
  if (auto* error = std::get_if<Error>(&params))
    return *error;
  const bool roi_capacity = std::size_t(cfg.batch_size) != geometry.batch;
  if (roi_capacity) {
    auto& values = std::get<GeometryParameters>(params);
    values[SourceRoi].reset();
    values[SourceRois] = ParameterRecords{};
  }
  auto parameters = std::get<std::vector<Parameter>>(std::move(constants));
  constexpr std::array names{"source_roi", "source_rois", "destination", "padding"};
  auto& values = std::get<GeometryParameters>(params);
  for (std::size_t i = 0; i < values.size(); ++i)
    if (values[i])
      parameters.push_back({names[i], std::move(*values[i])});
  auto checked = ::simaai::scheduler::client::check_kernel(
      {"preproc", Hardware::EV74, std::get<std::vector<TensorSpec>>(specs), policy->outputs,
       parameters});
  if (auto* error = std::get_if<Error>(&checked))
    return *error;
  policy->initial_executable = std::get<KernelExecutable>(std::move(checked));
  if (policy->image_publication) {
    auto inspected = ::simaai::scheduler::client::inspect_kernel(policy->initial_executable);
    if (auto* error = std::get_if<Error>(&inspected))
      return *error;
    const auto requirements = std::get<KernelInspection>(inspected).output_requirements();
    auto publication =
        std::make_shared<gstreamer::TensorStageImagePublication>(*policy->image_publication);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < publication->planes.size(); ++i) {
      if (offset > SIZE_MAX - 63)
        return invalid("output.surface", "plane alignment overflows");
      offset = (offset + 63) & ~std::size_t{63};
      publication->planes[i].byte_offset = offset;
      const auto pitch = std::size_t(publication->planes[i].stride);
      const auto bytes = std::max(requirements[i].storage_bytes, pitch * policy->storage_rows[i]);
      if (bytes > SIZE_MAX - offset)
        return invalid("output.surface", "plane span overflows");
      offset += bytes;
    }
    policy->image_publication = std::move(publication);
  }
  auto bound =
      ::simaai::scheduler::client::bind_kernel_variant(policy->initial_executable, "geometry");
  if (auto* error = std::get_if<Error>(&bound))
    return *error;
  policy->geometry_variant = std::get<KernelVariant>(std::move(bound));
  if (policy->geometry_variant->contract_policy() != KernelVariantPolicy::FullRecheck)
    return invalid("preproc.geometry", "geometry capability must recheck the full contract");
  const auto fields = policy->geometry_variant->fields();
  if (fields.size() != names.size() || !std::equal(fields.begin(), fields.end(), names.begin()))
    return invalid("preproc.geometry",
                   "geometry capability fields differ from the prepared recipe");
  policy->initial_geometry = geometry;
  return std::shared_ptr<const CheckedPolicy>(std::move(policy));
}

Result<std::shared_ptr<PreprocFrameAdapter>>
PreprocFrameAdapter::instantiate(std::shared_ptr<const CheckedPolicy> policy) {
  if (!policy || !policy->initial_executable || !policy->geometry_variant)
    return invalid("preproc.policy", "checked immutable image policy is required");
  auto result = std::shared_ptr<PreprocFrameAdapter>(new PreprocFrameAdapter(std::move(policy)));
  result->variants_[0].geometry = result->policy_->initial_geometry;
  result->variants_[0].signature.reset(copy_signature(nullptr));
  return result;
}

Result<std::shared_ptr<PreprocFrameAdapter>>
PreprocFrameAdapter::check(std::shared_ptr<const ProcessCvuStagePayload> prepared) {
  auto checked = check_policy(std::move(prepared));
  if (const auto* failure = std::get_if<Error>(&checked))
    return *failure;
  return instantiate(std::get<std::shared_ptr<const CheckedPolicy>>(std::move(checked)));
}

Status PreprocFrameAdapter::admitted(const VariantPreparer& preparer) {
  if (preparation_started_.exchange(true, std::memory_order_acq_rel))
    return invalid("preproc.preparation", "image recipe must be prepared exactly once");
  if (!preparer || !preparer.base() || preparer.base().executable() != policy_->initial_executable)
    return invalid("preproc.preparation", "admission does not match the checked image recipe");
  preparer_ = preparer;
  auto& initial = variants_[0];
  const bool roi_capacity = std::size_t(policy_->prepared->batch_size) != initial.geometry.batch;
  if (!roi_capacity) {
    initial.operation = preparer_.base();
    auto publication = prepare_publication(initial.geometry, policy_->initial_executable, nullptr);
    if (auto* failure = std::get_if<Error>(&publication))
      return *failure;
    initial.publication = std::get<Publication>(std::move(publication));
  }
  return {};
}

Result<std::vector<TensorSpec>> PreprocFrameAdapter::input_specs(const Geometry& geometry) const {
  if (geometry.width == 0 || geometry.height == 0 || geometry.batch == 0 ||
      geometry.width > policy_->envelope.width || geometry.height > policy_->envelope.height ||
      geometry.format != policy_->envelope.format || geometry.batch > policy_->envelope.batch)
    return invalid("input.geometry", "image exceeds its prepared capacity/format envelope");
  std::vector<TensorSpec> result;
  for (std::size_t i = 0; i < planes(geometry.format); ++i) {
    const bool chroma = i != 0;
    const auto h =
        chroma && geometry.format != GST_VIDEO_FORMAT_NV16 ? geometry.height / 2 : geometry.height;
    const auto w = chroma ? geometry.width / 2 : geometry.width;
    const auto c = chroma && geometry.format != GST_VIDEO_FORMAT_I420
                       ? 2
                       : (chroma ? 1 : channels(geometry.format));
    if (geometry.strides[i] > std::size_t(INT64_MAX) ||
        geometry.frame_bytes > std::size_t(INT64_MAX))
      return invalid("input.stride", "image stride is not representable");
    result.push_back({std::vector<std::int64_t>{std::int64_t(geometry.batch), std::int64_t(h),
                                                std::int64_t(w), std::int64_t(c)},
                      DType::UInt8,
                      ::simaai::scheduler::client::Strided{{std::int64_t(geometry.frame_bytes),
                                                            std::int64_t(geometry.strides[i]),
                                                            std::int64_t(c), 1}}});
  }
  return result;
}

Result<std::vector<Parameter>> PreprocFrameAdapter::static_parameters() const {
  const auto& cfg = *policy_->prepared;
  const auto dims = policy_->output.shape;
  if (dims.size() < 3)
    return invalid("output.shape", "PreprocFrameAdapter output needs HWC axes");
  const auto h = dims[dims.size() - 3], w = dims[dims.size() - 2];
  if (h <= 0 || w <= 0 || h > INT32_MAX || w > INT32_MAX)
    return invalid("output.shape", "output rectangle cannot be represented");
  std::vector<Parameter> result;
  std::string_view interpolation_value, output_format_value;
  const auto interpolation = upper(cfg.scaling_type);
  if (interpolation == "BILINEAR")
    interpolation_value = "bilinear";
  else if (interpolation == "NEAREST" || interpolation == "NEAREST_NEIGHBOUR")
    interpolation_value = "nearest";
  else if (interpolation == "BICUBIC")
    interpolation_value = "bicubic";
  else if (interpolation == "INTERAREA")
    interpolation_value = "box_average";
  else if (interpolation == "NO_SCALING")
    interpolation_value = "identity";
  else
    return invalid("scaling_type", "unknown prepared interpolation");
  const auto output_format = upper(cfg.output_img_type);
  if (output_format == "RGB")
    output_format_value = "rgb";
  else if (output_format == "BGR")
    output_format_value = "bgr";
  else if (output_format == "GRAY" || output_format == "GRAY8")
    output_format_value = "gray";
  else if (cfg.image_transform && output_format == "NV12")
    output_format_value = "nv12";
  else if (cfg.image_transform && output_format == "I420")
    output_format_value = "i420";
  else if (cfg.image_transform && output_format == "NV16")
    output_format_value = "nv16";
  else
    return invalid("output_img_type", "unknown output color format");
  const auto input_format = format_name(policy_->envelope.format);
  if (input_format.empty())
    return invalid("input_format", "unsupported input color format");
  result.push_back({"interpolation", interpolation_value});
  result.push_back({"input_format", input_format});
  result.push_back({"output_format", output_format_value});
  if (cfg.image_transform) {
    const auto& p = *cfg.image_transform;
    result.push_back(
        {"color_matrix", p.color.matrix == ImageColorMatrix::Bt601 ? "bt601_full" : "bt709_full"});
    if (output_format_value == "nv12" || output_format_value == "i420" ||
        output_format_value == "nv16") {
      result.push_back({"yuv_range", p.color.range == ImageYuvRange::Full ? "full" : "limited"});
      const auto siting = [](ImageChromaSiting s) {
        return s == ImageChromaSiting::Centered ? "centered" : "cosited";
      };
      result.push_back({"input_chroma_x", siting(p.color.input_siting[0])});
      result.push_back({"input_chroma_y", siting(p.color.input_siting[1])});
      result.push_back({"output_chroma_x", siting(p.color.output_siting[0])});
      result.push_back({"output_chroma_y", siting(p.color.output_siting[1])});
      result.push_back({"output_padding_mode", p.surface == ImageSurfaceLayout::EncoderNv12
                                                   ? "initialize_storage"
                                                   : "preserve_holes"});
      ParameterArray rows;
      for (const auto value : policy_->storage_rows)
        rows.emplace_back(std::uint64_t(value));
      result.push_back({"output_storage_rows", std::move(rows)});
      if (p.yuv_padding) {
        ParameterArray values;
        for (const auto value : *p.yuv_padding)
          values.emplace_back(std::uint64_t(value));
        result.push_back({"yuv_padding", std::move(values)});
      }
    }
  }
  ParameterArray offsets, multipliers;
  if (policy_->output.dtype == DType::Int8 && (!cfg.has_q_scale || !cfg.has_q_zp))
    return invalid("quantization", "INT8 PreprocFrameAdapter requires multiplier and zero point");
  for (std::size_t i = 0; i < 3; ++i) {
    const float mean =
        cfg.normalize == 1 && (cfg.channel_mean.size() == 3) ? cfg.channel_mean[i] : 0;
    const float divisor =
        cfg.normalize == 1 && (cfg.channel_stddev.size() == 3) ? cfg.channel_stddev[i] : 1;
    if (!std::isfinite(mean) || !std::isfinite(divisor) || divisor == 0)
      return invalid("normalization",
                     "normalization coefficients must be finite with nonzero divisor");
    // Preserve the legacy coefficient construction (PIXEL_RANGE is a double
    // literal), including the single rounding of the combined INT8 scale.
    const float scale =
        policy_->output.dtype == DType::Int8 ? static_cast<float>(cfg.q_scale) : 1.0f;
    offsets.emplace_back(cfg.normalize == 1 ? float(mean * 255.0) : 0.0f);
    multipliers.emplace_back(cfg.normalize == 1 ? float(scale / (divisor * 255.0)) : 1.0f);
  }
  result.push_back({"affine_offset", std::move(offsets)});
  result.push_back({"affine_multiplier", std::move(multipliers)});
  if (policy_->output.dtype == DType::Int8) {
    result.push_back({"quantization_multiplier", 1.0f});
    result.push_back({"quantization_zero_point", cfg.q_zp});
  }
  return result;
}

Result<PreprocFrameAdapter::GeometryParameters>
PreprocFrameAdapter::parameters(const Geometry& geometry, const GstStructure* meta,
                                const IntegerList* checked_indices) const {
  const auto& cfg = *policy_->prepared;
  const auto dims = policy_->output.shape;
  const auto h = dims[dims.size() - 3], w = dims[dims.size() - 2];
  GeometryParameters result;
  const bool identity = policy_->identity_interpolation;
  std::int64_t source_w = geometry.width, source_h = geometry.height;
  gboolean scalar_roi = FALSE, list = FALSE;
  if (meta) {
    gst_structure_get_boolean(meta, "preproc_roi_enable", &scalar_roi);
    list = roi_list_enabled(meta);
  }
  if (list) {
    int count = -1;
    if (!gst_structure_get_int(meta, "preproc_roi_count", &count) || count < 0 || count > 4096)
      return invalid("preproc_roi_count", "ROI list must be explicitly bounded to 4096 entries");
    auto rects = integer_list(meta, "preproc_roi_rects", std::size_t(count) * 4);
    if (auto* error = std::get_if<Error>(&rects))
      return *error;
    IntegerList batches;
    if (checked_indices) {
      batches = *checked_indices;
    } else {
      auto parsed = integer_list(meta, "preproc_roi_batch_indices", count, true);
      if (auto* error = std::get_if<Error>(&parsed))
        return *error;
      batches = std::get<IntegerList>(parsed);
    }
    ParameterRecords records;
    records.reserve(count);
    for (int i = 0; i < count; ++i) {
      const auto& rect = std::get<IntegerList>(rects);
      records.push_back(
          {{"source_index", batches[i]},
           {"x", rect[4 * i]},
           {"y", rect[4 * i + 1]},
           {"width", identity ? std::min(rect[4 * i + 2], int(w)) : rect[4 * i + 2]},
           {"height", identity ? std::min(rect[4 * i + 3], int(h)) : rect[4 * i + 3]}});
      // Invalid rectangles still reach kernel-owned filtering; no division on
      // their dimensions and no duplicate selection policy in the adapter.
      if (rect[4 * i + 2] > 0 && rect[4 * i + 3] > 0) {
        auto placed = destination(rect[4 * i + 2], rect[4 * i + 3], w, h);
        if (auto* error = std::get_if<Error>(&placed))
          return *error;
        const auto& d = std::get<Rectangle>(placed);
        constexpr std::array names{"destination_x", "destination_y", "destination_width",
                                   "destination_height"};
        for (std::size_t j = 0; j < d.size(); ++j)
          records.back().push_back({names[j], std::int32_t(d[j])});
      }
    }
    result[SourceRois] = std::move(records);
  } else if (scalar_roi) {
    std::array<int, 4> roi{};
    constexpr std::array names{"preproc_roi_x", "preproc_roi_y", "preproc_roi_width",
                               "preproc_roi_height"};
    for (std::size_t i = 0; i < names.size(); ++i)
      if (!gst_structure_get_int(meta, names[i], &roi[i]))
        return invalid(names[i], "ROI field missing");
    if (identity) {
      roi[2] = std::min<std::int64_t>(roi[2], w);
      roi[3] = std::min<std::int64_t>(roi[3], h);
    }
    source_w = roi[2];
    source_h = roi[3];
    // The native list path also preserves virtual outside-image padding.
    result[SourceRois] = ParameterRecords{
        {{"source_index", 0}, {"x", roi[0]}, {"y", roi[1]}, {"width", roi[2]}, {"height", roi[3]}}};
  }
  if (identity && !list && !scalar_roi && (source_w > w || source_h > h)) {
    source_w = std::min(source_w, w);
    source_h = std::min(source_h, h);
    result[SourceRoi] = ParameterArray{0, 0, std::int32_t(source_w), std::int32_t(source_h)};
  }
  auto placed = destination(source_w, source_h, w, h);
  if (auto* error = std::get_if<Error>(&placed))
    return *error;
  const auto& d = std::get<Rectangle>(placed);
  result[Destination] = ParameterArray{std::int32_t(d[0]), std::int32_t(d[1]), std::int32_t(d[2]),
                                       std::int32_t(d[3])};
  const bool raw_yuv = cfg.image_transform && (upper(cfg.output_img_type) == "NV12" ||
                                               upper(cfg.output_img_type) == "NV16" ||
                                               upper(cfg.output_img_type) == "I420");
  int pad = raw_yuv ? 0 : cfg.pad_value;
  if (meta)
    gst_structure_get_int(meta, "preproc_roi_pad_value", &pad);
  if (pad < 0 || pad > 255)
    return invalid("padding", "padding must be a byte value");
  result[Padding] = ParameterArray{std::uint8_t(pad), std::uint8_t(pad), std::uint8_t(pad)};
  return result;
}

Result<GstVideoInfo> PreprocFrameAdapter::parse_caps(const GstCaps* caps) const {
  GstVideoInfo info;
  gst_video_info_init(&info);
  if (!caps || !gst_caps_is_fixed(caps) || !gst_video_info_from_caps(&info, caps))
    return invalid("input.caps", "fixed raw-video geometry is required");
  if (info.finfo->format != policy_->envelope.format || info.width <= 0 || info.height <= 0 ||
      std::size_t(info.width) > policy_->envelope.width ||
      std::size_t(info.height) > policy_->envelope.height)
    return invalid("input.caps", "video caps exceed the prepared format/capacity envelope");
  if (policy_->prepared->image_transform && !policy_->prepared->image_transform->dynamic_input() &&
      (std::size_t(info.width) != policy_->initial_geometry.width ||
       std::size_t(info.height) != policy_->initial_geometry.height))
    return invalid("input.caps",
                   "ColorConvert/Normalization dimensions are fixed per run; reprepare");
  if (policy_->prepared->image_transform) {
    const auto* structure = gst_caps_get_structure(caps, 0);
    if (const auto* mode = gst_structure_get_string(structure, "interlace-mode");
        mode && std::string_view(mode) != "progressive")
      return invalid("input.caps", "image frontends require progressive video");
    if (planes(info.finfo->format) > 1) {
      const auto& authored = policy_->prepared->image_transform->color;
      if (const auto* text = gst_structure_get_string(structure, "colorimetry")) {
        GstVideoColorimetry color{};
        if (!gst_video_colorimetry_from_string(&color, text))
          return invalid("input.colorimetry", "malformed explicit video colorimetry");
        const auto range = authored.range == ImageYuvRange::Full ? GST_VIDEO_COLOR_RANGE_0_255
                                                                 : GST_VIDEO_COLOR_RANGE_16_235;
        const auto matrix = authored.matrix == ImageColorMatrix::Bt601
                                ? GST_VIDEO_COLOR_MATRIX_BT601
                                : GST_VIDEO_COLOR_MATRIX_BT709;
        if ((color.range != GST_VIDEO_COLOR_RANGE_UNKNOWN && color.range != range) ||
            (color.matrix != GST_VIDEO_COLOR_MATRIX_UNKNOWN && color.matrix != matrix))
          return invalid("input.colorimetry",
                         "explicit source range/matrix contradicts image policy");
      }
      if (const auto* text = gst_structure_get_string(structure, "chroma-site")) {
        const auto site = gst_video_chroma_site_from_string(text);
        if (site == GST_VIDEO_CHROMA_SITE_UNKNOWN || (site & GST_VIDEO_CHROMA_SITE_ALT_LINE) ||
            bool(site & GST_VIDEO_CHROMA_SITE_H_COSITED) !=
                (authored.input_siting[0] == ImageChromaSiting::Cosited) ||
            bool(site & GST_VIDEO_CHROMA_SITE_V_COSITED) !=
                (authored.input_siting[1] == ImageChromaSiting::Cosited))
          return invalid("input.chroma-site",
                         "unsupported or contradictory explicit source chroma siting");
      }
    }
  }
  return info;
}

Status PreprocFrameAdapter::accepts_caps(const GstCaps* caps) const {
  auto parsed = parse_caps(caps);
  if (auto* failure = std::get_if<Error>(&parsed))
    return *failure;
  return {};
}

Status PreprocFrameAdapter::check_caps(const GstCaps* caps) const {
  auto parsed = parse_caps(caps);
  if (auto* failure = std::get_if<Error>(&parsed))
    return *failure;
  std::lock_guard lock(mutex_);
  const auto& info = std::get<GstVideoInfo>(parsed);
  auto geometry = describe_geometry(default_geometry_, info.width, info.height);
  if (auto* failure = std::get_if<Error>(&geometry))
    return *failure;
  default_geometry_ = std::get<Geometry>(geometry);
  return {};
}

Status
PreprocFrameAdapter::bind(GstBuffer* buffer, BindingWriter& binding, GstStructure* out,
                          bool fresh_metadata,
                          simaai::gst::TensorBufferPreprocessFactsOwner* compact_facts) const {
  if (!preparer_)
    return invalid("preproc.preparation", "image policy has not been admitted");
  SIMA_SCHEDULER_TRACE_CURRENT(PreprocCheckBegin);
  std::lock_guard lock(mutex_);
  Geometry geometry = default_geometry_;
  auto* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  const auto* source_meta = custom ? gst_custom_meta_get_structure(custom) : nullptr;
  // Legacy ROI fields on an upstream result are facts, not instructions to apply a second crop.
  const auto* meta = policy_->prepared->image_transform ? nullptr : source_meta;
  if (policy_->prepared->image_transform && source_meta) {
    if (roi_list_enabled(source_meta))
      return invalid("input.preprocess", "image frontends require one frame, not an ROI list");
    gboolean roi = FALSE;
    if (gst_structure_get_boolean(source_meta, "preproc_roi_enable", &roi) && roi &&
        !gst_structure_has_field(source_meta, "preproc_affine_m00"))
      return invalid("input.preprocess", "image frontends do not accept ROI crop requests; "
                                         "complete spatial provenance is required");
  }
  gboolean scalar_roi = FALSE;
  if (meta)
    gst_structure_get_boolean(meta, "preproc_roi_enable", &scalar_roi);
  const bool active_roi = scalar_roi || roi_list_enabled(meta);
  auto has_override = [&](const char* name, const char* alias) {
    return active_roi &&
           (gst_structure_has_field(meta, name) || gst_structure_has_field(meta, alias));
  };
  auto read_override = [&](const char* name, const char* alias, std::size_t& value) -> Status {
    if (!active_roi)
      return {};
    const char* field = gst_structure_has_field(meta, name) ? name : alias;
    if (!gst_structure_has_field(meta, field))
      return {};
    int parsed = 0;
    if (!gst_structure_get_int(meta, field, &parsed) || parsed <= 0)
      return invalid(field, "source geometry override must be a positive integer");
    value = std::size_t(parsed);
    return {};
  };
  auto width = geometry.width, height = geometry.height;
  if (auto e = read_override("preproc_roi_source_width", "roi_source_width", width))
    return e;
  if (auto e = read_override("preproc_roi_source_height", "roi_source_height", height))
    return e;
  if (width > policy_->envelope.width || height > policy_->envelope.height)
    return invalid("input.geometry", "source override exceeds prepared envelope");
  if (width != geometry.width || height != geometry.height) {
    auto described = describe_geometry(geometry, width, height);
    if (auto* failure = std::get_if<Error>(&described))
      return *failure;
    geometry = std::get<Geometry>(described);
  }
  auto row_stride = geometry.strides[0];
  if (auto e =
          read_override("preproc_roi_source_stride_bytes", "roi_source_stride_bytes", row_stride))
    return e;
  if (row_stride != geometry.strides[0]) {
    if (planes(geometry.format) != 1)
      return invalid("input.stride", "planar pitch override requires authoritative GstVideoMeta");
    if (row_stride < width * channels(geometry.format) || row_stride > SIZE_MAX / height)
      return invalid("input.stride", "source row span is invalid");
    geometry.strides[0] = row_stride;
    geometry.frame_bytes = row_stride * height;
  }
  if (auto* video = gst_buffer_get_video_meta(buffer)) {
    if (policy_->prepared->image_transform && (video->flags & GST_VIDEO_FRAME_FLAG_INTERLACED))
      return invalid("input.video-meta", "image frontends require progressive video");
    if (video->format != geometry.format || video->n_planes != planes(geometry.format))
      return invalid("input.video-meta", "plane format differs from negotiated caps");
    if ((has_override("preproc_roi_source_width", "roi_source_width") && width != video->width) ||
        (has_override("preproc_roi_source_height", "roi_source_height") &&
         height != video->height) ||
        (has_override("preproc_roi_source_stride_bytes", "roi_source_stride_bytes") &&
         row_stride != std::size_t(video->stride[0])))
      return invalid("input.video-meta",
                     "ROI source overrides disagree with authoritative video geometry");
    geometry.width = video->width;
    geometry.height = video->height;
    geometry.frame_bytes = 0;
    for (std::size_t i = 0; i < video->n_planes; ++i) {
      if (video->stride[i] <= 0)
        return invalid("input.video-meta", "nonpositive row stride");
      geometry.offsets[i] = video->offset[i];
      geometry.strides[i] = video->stride[i];
      const auto rows =
          i && geometry.format != GST_VIDEO_FORMAT_NV16 ? geometry.height / 2 : geometry.height;
      if (rows && geometry.strides[i] > (SIZE_MAX - geometry.offsets[i]) / rows)
        return invalid("input.video-meta", "plane span overflow");
      geometry.frame_bytes =
          std::max(geometry.frame_bytes, geometry.offsets[i] + geometry.strides[i] * rows);
    }
  }
  if (policy_->prepared->image_transform && !policy_->prepared->image_transform->dynamic_input() &&
      (geometry.width != policy_->envelope.width || geometry.height != policy_->envelope.height))
    return invalid("input.video-meta", "ColorConvert/Normalization geometry is fixed per run");
  std::optional<IntegerList> inferred_indices;
  if (roi_metadata(meta)) {
    int batch = 0;
    if (gst_structure_get_int(meta, "preproc_roi_input_batch_size", &batch)) {
      if (batch <= 0)
        return invalid("preproc_roi_input_batch_size", "source batch must be positive");
      geometry.batch = batch;
    } else {
      geometry.batch = 1;
      if (roi_list_enabled(meta)) {
        int count = -1;
        if (!gst_structure_get_int(meta, "preproc_roi_count", &count) || count < 0 || count > 4096)
          return invalid("preproc_roi_count", "ROI list must be bounded to 4096 entries");
        auto indices = integer_list(meta, "preproc_roi_batch_indices", count, true);
        if (auto* error = std::get_if<Error>(&indices))
          return *error;
        inferred_indices = std::get<IntegerList>(indices);
        geometry.batch = inferred_indices->source_batch;
      }
    }
  }
  if (!geometry.frame_bytes || geometry.batch > gst_buffer_get_size(buffer) / geometry.frame_bytes)
    return invalid("input.batch", "source batch exceeds the carrier's actual storage capacity");
  const auto matches = [&](std::size_t index) {
    const auto& variant = variants_[index];
    return variant.operation && variant.signature && geometry == variant.geometry &&
           same_signature(meta, variant.signature.get());
  };
  std::size_t selected = current_variant_;
  bool cached = matches(selected);
  if (!cached) {
    selected = 1 - selected;
    cached = matches(selected);
  }
  auto operation = cached ? variants_[selected].operation : PreparedSelection{};
  const bool prepared_now = !operation;
  if (cached)
    SIMA_SCHEDULER_TRACE_CURRENT(PreprocVariantHit);
  else
    SIMA_SCHEDULER_TRACE_CURRENT(PreprocVariantMiss);
  if (prepared_now) {
    auto params = parameters(geometry, meta, inferred_indices ? &*inferred_indices : nullptr);
    if (auto* error = std::get_if<Error>(&params))
      return *error;
    auto specs = input_specs(geometry);
    if (auto* error = std::get_if<Error>(&specs))
      return *error;
    SIMA_SCHEDULER_TRACE_CURRENT(PreprocPrepare);
    auto prepared = preparer_.prepare(*policy_->geometry_variant,
                                      {std::get<std::vector<TensorSpec>>(specs), policy_->outputs,
                                       std::get<GeometryParameters>(params)});
    if (auto* error = std::get_if<Error>(&prepared))
      return *error;
    operation = std::get<PreparedSelection>(std::move(prepared));
  }
  SIMA_SCHEDULER_TRACE_CURRENT(PreprocCheckEnd);
  for (std::size_t i = 0; i < operation.input_count(); ++i) {
    const auto bytes = operation.input_requirements(i).storage_bytes;
    guint memory_index = 0, memory_count = 0;
    gsize skip = 0;
    if (!gst_buffer_find_memory(buffer, geometry.offsets[i], bytes, &memory_index, &memory_count,
                                &skip) ||
        memory_count != 1)
      return invalid("input.plane", "a plane must have one authoritative contiguous memory span");
    auto* memory = gst_buffer_peek_memory(buffer, memory_index);
    if (auto error = binding.append_input(i, memory, skip, bytes))
      return error;
  }
  if (!out && !compact_facts)
    return invalid("output.meta", "missing preprocessing publication destination");
  SIMA_SCHEDULER_TRACE_CURRENT(PreprocPublishBegin);
  Publication publication;
  if (prepared_now) {
    auto facts = prepare_publication(geometry, operation.executable(), meta);
    if (auto* error = std::get_if<Error>(&facts))
      return *error;
    publication = std::get<Publication>(std::move(facts));
  }
  auto facts = prepared_now ? publication : variants_[selected].publication;
  if (policy_->prepared->image_transform && source_meta) {
    constexpr std::array names{"preproc_affine_m00", "preproc_affine_m01", "preproc_affine_m02",
                               "preproc_affine_m10", "preproc_affine_m11", "preproc_affine_m12"};
    const bool transformed = std::any_of(names.begin(), names.end(), [&](const char* name) {
      return gst_structure_has_field(source_meta, name);
    });
    if (transformed) {
      std::array<double, 6> a{};
      for (std::size_t i = 0; i < a.size(); ++i)
        if (!gst_structure_get_double(source_meta, names[i], &a[i]) || !std::isfinite(a[i]))
          return invalid("input.preprocess", "incomplete or nonfinite upstream spatial transform");
      int original_w = 0, original_h = 0;
      if (!gst_structure_get_int(source_meta, "preproc_original_width", &original_w) ||
          !gst_structure_get_int(source_meta, "preproc_original_height", &original_h) ||
          original_w <= 0 || original_h <= 0)
        return invalid("input.preprocess",
                       "upstream transform requires original source dimensions");
      gboolean upstream_roi = FALSE;
      gst_structure_get_boolean(source_meta, "preproc_roi_enable", &upstream_roi);
      std::array<int, 4> roi{};
      if (upstream_roi) {
        constexpr std::array roi_names{"preproc_roi_x", "preproc_roi_y", "preproc_roi_width",
                                       "preproc_roi_height"};
        for (std::size_t i = 0; i < roi_names.size(); ++i)
          if (!gst_structure_get_int(source_meta, roi_names[i], &roi[i]) || roi[i] < 0 ||
              (i >= 2 && roi[i] == 0))
            return invalid("input.preprocess", "malformed upstream ROI provenance");
      }
      const auto* color_value = gst_structure_get_string(source_meta, "preproc_color_in");
      const std::string_view color = color_value ? color_value : "";
      auto& cache = composed_[selected];
      if (cache.base == facts && cache.upstream == a && cache.original_width == original_w &&
          cache.original_height == original_h && cache.has_roi == bool(upstream_roi) &&
          cache.roi == roi && cache.color_in == color) {
        facts = cache.result;
      } else {
        auto composed = *simaai::gst::tensor_buffer_preprocess_facts(facts);
        const auto b = composed.affine;
        composed.affine = {a[0] * b[0] + a[1] * b[3],        a[0] * b[1] + a[1] * b[4],
                           a[0] * b[2] + a[1] * b[5] + a[2], a[3] * b[0] + a[4] * b[3],
                           a[3] * b[1] + a[4] * b[4],        a[3] * b[2] + a[4] * b[5] + a[5]};
        if (upstream_roi) {
          std::copy(roi.begin(), roi.end(), composed.primary.begin());
          composed.has_roi = true;
        }
        composed.original_width = original_w;
        composed.original_height = original_h;
        if (!color.empty())
          composed.color_in = color;
        std::string error;
        auto frozen =
            simaai::gst::tensor_buffer_freeze_preprocess_facts(std::move(composed), &error);
        if (!frozen)
          return invalid("output.preprocess", std::move(error));
        cache = {
            facts, frozen, a, roi, original_w, original_h, bool(upstream_roi), std::string(color)};
        facts = std::move(frozen);
      }
    }
  }
  if (compact_facts)
    *compact_facts = facts;
  if (out)
    simaai::gst::tensor_buffer_write_preprocess_facts(out, facts, fresh_metadata);
  SIMA_SCHEDULER_TRACE_CURRENT(PreprocPublishEnd);
  if (prepared_now) {
    // Commit only after both current storage and publication validate. Failed
    // frames must not evict a usable checked variant. An uncacheable foreign
    // optional field retains its existing admission behavior without storage.
    std::unique_ptr<GstStructure, StructureFree> signature(copy_signature(meta));
    if (signature) {
      variants_[selected] = {geometry, std::move(signature), operation, std::move(publication)};
      current_variant_ = selected;
    }
  } else {
    current_variant_ = selected;
  }
  return binding.select(operation);
}

Result<PreprocFrameAdapter::Publication>
PreprocFrameAdapter::prepare_publication(const Geometry& geometry,
                                         const KernelExecutable& executable,
                                         const GstStructure* source) const {
  const auto& cfg = *policy_->prepared;
  const auto dims = policy_->output.shape;
  const auto h = dims[dims.size() - 3], w = dims[dims.size() - 2];
  const bool list = roi_list_enabled(source);
  // Decode operation-owned facts only while freezing a new publication. A warm
  // frame uses the cached immutable publication and performs no fact lookup.
  auto inspected = ::simaai::scheduler::client::inspect_kernel(executable);
  if (const auto* error = std::get_if<Error>(&inspected))
    return *error;
  const auto& inspection = std::get<KernelInspection>(inspected);
  const auto facts = inspection.preparation_facts();
  const auto find_fact = [&](std::string_view name) -> const ParameterValue* {
    const auto found = std::find_if(facts.begin(), facts.end(),
                                    [&](const auto& fact) { return fact.name == name; });
    return found == facts.end() ? nullptr : &found->value;
  };
  const auto* roi_fact = find_fact("accepted_rois");
  const auto* accepted = roi_fact ? std::get_if<ParameterRecords>(&roi_fact->value) : nullptr;
  if (roi_fact && !accepted)
    return invalid("preproc.facts", "invalid accepted ROI records");
  const bool has_admitted_rois = accepted != nullptr;
  std::array<std::uint64_t, 4> counts{};
  if (has_admitted_rois) {
    constexpr std::array names{"requested_rois", "valid_rois", "dropped_invalid",
                               "dropped_overflow"};
    for (std::size_t i = 0; i < names.size(); ++i) {
      const auto* fact = find_fact(names[i]);
      const auto* scalar = fact ? std::get_if<ParameterScalar>(&fact->value) : nullptr;
      const auto value = scalar ? integer_fact(*scalar) : std::nullopt;
      if (!value || *value < 0 || *value > G_MAXINT)
        return invalid("preproc.facts", "invalid ROI count");
      counts[i] = *value;
    }
    if (counts[1] != accepted->size())
      return invalid("preproc.facts", "accepted ROI count differs from records");
  }
  Rectangle primary{0, 0, std::int64_t(geometry.width), std::int64_t(geometry.height)};
  auto normal = destination(geometry.width, geometry.height, w, h);
  if (auto* error = std::get_if<Error>(&normal))
    return *error;
  Rectangle placed = std::get<Rectangle>(normal);
  if (policy_->identity_interpolation) {
    primary[2] = placed[2];
    primary[3] = placed[3];
  }
  std::vector<std::int64_t> indices, rects;
  std::vector<double> affines;
  auto affine = [](const Rectangle& r, const Rectangle& d) {
    const double x = double(r[2]) / double(d[2]), y = double(r[3]) / double(d[3]);
    return std::array<double, 6>{x, 0, double(r[0]) - double(d[0]) * x,
                                 0, y, double(r[1]) - double(d[1]) * y};
  };
  if (has_admitted_rois) {
    // Admission already bounded records and validated their destinations.
    indices.reserve(accepted->size());
    rects.reserve(accepted->size() * 4);
    affines.reserve(accepted->size() * 6);
    for (const auto& selection : *accepted) {
      constexpr std::array names{"source_index",
                                 "x",
                                 "y",
                                 "width",
                                 "height",
                                 "destination_x",
                                 "destination_y",
                                 "destination_width",
                                 "destination_height"};
      std::array<std::int64_t, names.size()> values{};
      for (std::size_t i = 0; i < names.size(); ++i) {
        const auto found = std::find_if(selection.begin(), selection.end(),
                                        [&](const auto& field) { return field.name == names[i]; });
        const auto value = found == selection.end() ? std::nullopt : integer_fact(found->value);
        if (!value)
          return invalid("preproc.facts", "invalid accepted ROI field");
        values[i] = *value;
      }
      const Rectangle r{values[1], values[2], values[3], values[4]};
      const Rectangle d{values[5], values[6], values[7], values[8]};
      if (indices.empty()) {
        primary = r;
        placed = d;
      }
      indices.push_back(values[0]);
      rects.insert(rects.end(), r.begin(), r.end());
      const auto a = affine(r, d);
      affines.insert(affines.end(), a.begin(), a.end());
    }
  }
  simaai::gst::TensorBufferPreprocessFacts publication;
  publication.original_width = geometry.width;
  publication.original_height = geometry.height;
  publication.resized_width = w;
  publication.resized_height = h;
  publication.input_batch = geometry.batch;
  publication.capacity = cfg.batch_size;
  publication.source_stride = geometry.strides[0];
  publication.normalize = cfg.normalize == 1;
  publication.quantize = policy_->output.dtype == DType::Int8;
  publication.tessellate =
      std::holds_alternative<::simaai::scheduler::client::MlaTiled>(policy_->output.layout);
  publication.resize_mode = cfg.aspect_ratio == 1 ? "letterbox" : "stretch";
  publication.color_in = cfg.input_img_type;
  publication.color_out = cfg.output_img_type;
  publication.roi_list = list;
  publication.primary = primary;
  publication.placed = placed;
  publication.affine = affine(primary, placed);
  publication.counts = counts;
  publication.has_roi = !indices.empty();
  publication.source_index = indices.empty() ? 0 : indices.front();
  if (list) {
    if (!has_admitted_rois)
      return invalid("preproc.facts", "kernel did not publish ROI admission facts");
    publication.pad = cfg.pad_value;
    if (source)
      gst_structure_get_int(source, "preproc_roi_pad_value", &publication.pad);
    publication.roi_batch_indices = std::move(indices);
    publication.roi_rects = std::move(rects);
    publication.roi_affines = std::move(affines);
  }
  std::string error;
  auto frozen = simaai::gst::tensor_buffer_freeze_preprocess_facts(std::move(publication), &error);
  if (!frozen)
    return invalid("preproc.facts", std::move(error));
  return frozen;
}

} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
