#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/client/kernel.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace rt = ::simaai::scheduler::client;
namespace scheduler = ::simaai::scheduler::client;
using Transform = std::array<float, 4>;
struct Geometry {
  int width = 0, height = 0, model_width = 0, model_height = 0;
  std::string_view resize;
  std::optional<Transform> transform;
  // Canvas and the scaled image's {x, y, width, height} on it, normalized as
  // sam_boxdecode_geometry() admits them.
  std::int64_t canvas_width = 0, canvas_height = 0;
  std::array<std::int64_t, 4> placed{};
  bool roi_list = false;
};
rt::Error invalid(const char* message) {
  return {rt::ErrorCode::InvalidRequest, "boxdecode.geometry", message};
}
rt::Result<Transform> checked_transform(const std::array<double, 6>& matrix, bool uniform) {
  for (const auto value : matrix)
    if (!std::isfinite(value))
      return invalid("preprocess affine must be finite");
  if (matrix[1] != 0. || matrix[3] != 0. || matrix[0] <= 0. || matrix[4] <= 0.)
    return invalid("preprocess affine requires positive axis-aligned scales without shear");
  if (uniform && matrix[0] != matrix[4])
    return invalid("oriented boxes require equal authoritative affine scales");
  // PreprocFrameAdapter and the Scheduler facts serializer publish SOURCE from
  // DESTINATION coordinates: original = model * scale + offset. Do not invert.
  Transform result{static_cast<float>(matrix[0]), static_cast<float>(matrix[4]),
                   static_cast<float>(matrix[2]), static_cast<float>(matrix[5])};
  for (const auto value : result)
    if (!std::isfinite(value))
      return invalid("preprocess affine cannot be represented as FP32");
  if (result[0] <= 0.f || result[1] <= 0.f)
    return invalid("preprocess affine scales underflow FP32");
  return result;
}
rt::Result<Geometry> read_geometry(GstBuffer* buffer, bool complete, bool uniform) {
  if (!buffer)
    return invalid("input buffer is absent");
  std::string error;
  const auto* frame = ::simaai::gst::tensor_buffer_frame_metadata(buffer, &error);
  if (!error.empty())
    return invalid("compact frame metadata is malformed");
  if (frame) {
    const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(frame->preprocess);
    if (!facts)
      return Geometry{};
    if (facts->original_width < 1 || facts->original_width > 1048576 ||
        facts->original_height < 1 || facts->original_height > 1048576)
      return invalid("original image dimensions must be 1..1048576");
    Geometry geometry{static_cast<int>(facts->original_width),
                      static_cast<int>(facts->original_height),
                      static_cast<int>(facts->resized_width),
                      static_cast<int>(facts->resized_height),
                      facts->resize_mode,
                      {},
                      facts->resized_width,
                      facts->resized_height,
                      facts->placed,
                      facts->roi_list};
    if (complete) {
      if (facts->roi_list)
        return invalid("one BoxDecode frame requires one unambiguous preprocessing transform");
      auto transform = checked_transform(facts->affine, uniform);
      if (const auto* error = std::get_if<rt::Error>(&transform))
        return *error;
      geometry.transform = std::get<Transform>(transform);
    }
    return geometry;
  }
  auto* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  const auto* meta = custom ? gst_custom_meta_get_structure(custom) : nullptr;
  if (!meta)
    return Geometry{};
  Geometry geometry;
  gst_structure_get_int(meta, "preproc_original_width", &geometry.width);
  gst_structure_get_int(meta, "preproc_original_height", &geometry.height);
  if (const auto* mode = gst_structure_get_string(meta, "preproc_resize_mode"))
    geometry.resize = mode;
  {
    int resized_width = 0, resized_height = 0, scaled_width = 0, scaled_height = 0;
    int left = 0, right = 0, top = 0, bottom = 0;
    gst_structure_get_int(meta, "preproc_resized_width", &resized_width);
    gst_structure_get_int(meta, "preproc_resized_height", &resized_height);
    gst_structure_get_int(meta, "preproc_scaled_width", &scaled_width);
    gst_structure_get_int(meta, "preproc_scaled_height", &scaled_height);
    gst_structure_get_int(meta, "preproc_pad_left", &left);
    gst_structure_get_int(meta, "preproc_pad_right", &right);
    gst_structure_get_int(meta, "preproc_pad_top", &top);
    gst_structure_get_int(meta, "preproc_pad_bottom", &bottom);
    gboolean roi_list = FALSE;
    gst_structure_get_boolean(meta, "preproc_roi_list_enable", &roi_list);
    geometry.roi_list = roi_list;
    const std::int64_t width = scaled_width > 0 ? scaled_width : resized_width;
    const std::int64_t height = scaled_height > 0 ? scaled_height : resized_height;
    geometry.placed = {left, top, width, height};
    geometry.canvas_width = width + left + right;
    geometry.canvas_height = height + top + bottom;
  }
  if (complete) {
    gst_structure_get_int(meta, "preproc_resized_width", &geometry.model_width);
    gst_structure_get_int(meta, "preproc_resized_height", &geometry.model_height);
    gboolean roi_list = FALSE;
    gst_structure_get_boolean(meta, "preproc_roi_list_enable", &roi_list);
    if (roi_list)
      return invalid("one BoxDecode frame requires one unambiguous preprocessing transform");
    constexpr std::array matrix_names{"preproc_affine_m00", "preproc_affine_m01",
                                      "preproc_affine_m02", "preproc_affine_m10",
                                      "preproc_affine_m11", "preproc_affine_m12"};
    constexpr std::array alias_names{"preproc_affine_scale_x", "preproc_affine_scale_y",
                                     "preproc_affine_offset_x", "preproc_affine_offset_y"};
    std::array<double, 6> matrix{};
    bool matrix_present = false, alias_present = false;
    for (const auto* name : matrix_names)
      matrix_present |= gst_structure_has_field(meta, name);
    for (const auto* name : alias_names)
      alias_present |= gst_structure_has_field(meta, name);
    if (matrix_present) {
      for (std::size_t i = 0; i < matrix.size(); ++i)
        if (!gst_structure_get_double(meta, matrix_names[i], &matrix[i]))
          return invalid("preprocess affine matrix is partial or has invalid field types");
      auto transform = checked_transform(matrix, uniform);
      if (const auto* error = std::get_if<rt::Error>(&transform))
        return *error;
      geometry.transform = std::get<Transform>(transform);
    }
    if (alias_present) {
      std::array<double, 4> values{};
      for (std::size_t i = 0; i < values.size(); ++i)
        if (!gst_structure_get_double(meta, alias_names[i], &values[i]))
          return invalid("preprocess affine aliases are partial or have invalid field types");
      auto transform =
          checked_transform({values[0], 0., values[2], 0., values[1], values[3]}, uniform);
      if (const auto* error = std::get_if<rt::Error>(&transform))
        return *error;
      if (geometry.transform && *geometry.transform != std::get<Transform>(transform))
        return invalid("preprocess affine matrix and aliases disagree");
      geometry.transform = std::get<Transform>(transform);
    }
  }
  return geometry;
}
struct Cache {
  struct Entry {
    int width = 0, height = 0;
    std::optional<Transform> transform;
    rt::PreparedSelection operation;
  };
  // Per-admission bounded cache. Entries retain values, never metadata views.
  std::mutex mutex;
  rt::PreparedSelection base;
  std::array<Entry, 4> variants;
  std::size_t next = 0;
};
} // namespace
void attach_boxdecode_geometry_resolver(KernelStageRecipe& recipe, std::size_t member_index,
                                        BoxDecodeGeometryOptions options,
                                        bool original_size_from_metadata,
                                        bool require_resize_metadata) {
  if (!original_size_from_metadata && !require_resize_metadata && !options.complete_geometry)
    return;
  const auto& member = recipe.members.at(member_index);
  std::optional<rt::KernelVariant> variant;
  if (original_size_from_metadata || options.complete_geometry) {
    auto bound = scheduler::bind_kernel_variant(
        member.executable, options.complete_geometry ? "image_geometry" : "image_size");
    if (const auto* error = std::get_if<rt::Error>(&bound))
      throw std::invalid_argument("BoxDecode geometry: " + error->message);
    variant = std::get<rt::KernelVariant>(std::move(bound));
    const auto fields = variant->fields();
    if (variant->contract_policy() != rt::KernelVariantPolicy::PreserveContract ||
        fields.size() != (options.complete_geometry ? 2U : 1U) || fields.front() != "image_size" ||
        (options.complete_geometry && fields[1] != "model_to_original"))
      throw std::invalid_argument("BoxDecode geometry: incompatible image geometry capability");
  }
  recipe.resolver_factories.push_back(
      {member_index,
       [variant = std::move(variant), options, original_size_from_metadata,
        require_resize_metadata]() -> KernelStageRecipe::Resolver {
         return [cache = std::make_shared<Cache>(), variant, options, original_size_from_metadata,
                 require_resize_metadata](
                    const rt::VariantPreparer& preparer,
                    std::span<GstBuffer* const> buffers) -> rt::Result<rt::PreparedSelection> {
           const auto& base = preparer.base();
           if (!base || buffers.empty())
             return invalid("prepared operation and input buffers are required");
           Geometry selected;
           bool found = false, found_transform_authority = false;
           for (auto* buffer : buffers) {
             auto parsed = read_geometry(buffer, options.complete_geometry,
                                         options.uniform_transform_required);
             if (const auto* error = std::get_if<rt::Error>(&parsed))
               return *error;
             const auto geometry = std::get<Geometry>(parsed);
             if (!geometry.resize.empty()) {
               if (!selected.resize.empty() && selected.resize != geometry.resize)
                 return invalid("input head carriers disagree on resize mode");
               selected.resize = geometry.resize;
             }
             if (options.complete_geometry) {
               if ((geometry.model_width || geometry.model_height) &&
                   (geometry.model_width != options.model_width ||
                    geometry.model_height != options.model_height))
                 return invalid("preprocess destination dimensions differ from the checked model");
               if (geometry.transform || geometry.width || geometry.height) {
                 if (found_transform_authority && selected.transform != geometry.transform)
                   return invalid("input head carriers disagree on preprocessing transform");
                 selected.transform = geometry.transform;
                 found_transform_authority = true;
               }
               if (geometry.transform && !original_size_from_metadata &&
                   (geometry.width != options.original_width ||
                    geometry.height != options.original_height))
                 return invalid(
                     "preprocess affine source dimensions conflict with explicit image size");
             }
             // Explicit image dimensions retain their established precedence.
             if (!original_size_from_metadata || (geometry.width == 0 && geometry.height == 0))
               continue;
             if (geometry.width < 1 || geometry.width > 1048576 || geometry.height < 1 ||
                 geometry.height > 1048576)
               return invalid("original image dimensions are missing or invalid");
             if (found && (selected.width != geometry.width || selected.height != geometry.height))
               return invalid("input head carriers disagree on image geometry");
             selected.width = geometry.width;
             selected.height = geometry.height;
             found = true;
           }
           const std::string_view expected_resize =
               options.resize == ResizeMode::Stretch ? "stretch"
               : options.resize == ResizeMode::Crop  ? "crop"
                                                     : "letterbox";
           if (require_resize_metadata && selected.resize.empty())
             return invalid("required resize-mode metadata is missing");
           if (!selected.resize.empty() && selected.resize != expected_resize)
             return invalid("preprocess resize mode differs from the checked decode geometry");
           if (!original_size_from_metadata) {
             selected.width = options.original_width;
             selected.height = options.original_height;
           } else if (!found) {
             return invalid("original image dimensions require preprocess metadata");
           }
           if (!options.complete_geometry && !original_size_from_metadata)
             return base;
           if (options.complete_geometry) {
             if (options.require_frame_transform &&
                 (!found_transform_authority || !selected.transform))
               return invalid(
                   "source-driven complete geometry requires per-frame preprocess affine");
             if (!found_transform_authority)
               selected.transform = options.model_to_original;
             if (options.uniform_transform_required &&
                 (!selected.transform || (*selected.transform)[0] <= 0 ||
                  (*selected.transform)[0] != (*selected.transform)[1] ||
                  !std::all_of(selected.transform->begin(), selected.transform->end(),
                               [](float value) { return std::isfinite(value); })))
               return invalid("oriented boxes require a positive uniform explicit transform");
             if (options.resize == ResizeMode::Crop && !selected.transform)
               return invalid("crop decoding requires the actual model-to-original affine");
           }
           if (selected.width == options.original_width &&
               selected.height == options.original_height &&
               selected.transform == options.model_to_original)
             return base;
           std::lock_guard lock(cache->mutex);
           if (cache->base != base) {
             cache->base = base;
             cache->variants = {};
             cache->next = 0;
           }
           for (const auto& entry : cache->variants)
             if (entry.operation && entry.width == selected.width &&
                 entry.height == selected.height && entry.transform == selected.transform)
               return entry.operation;
           std::array<std::optional<::simaai::scheduler::client::ParameterValue>, 2> values{
               ::simaai::scheduler::client::ParameterValue{
                   ::simaai::scheduler::client::ParameterArray{selected.width, selected.height}},
               {}};
           if (selected.transform) {
             const auto& t = *selected.transform;
             values[1] = ::simaai::scheduler::client::ParameterArray{t[0], t[1], t[2], t[3]};
           }
           auto prepared = preparer.prepare(
               *variant, {{}, {}, {values.data(), options.complete_geometry ? 2U : 1U}});
           if (const auto* error = std::get_if<rt::Error>(&prepared))
             return *error;
           auto operation = std::get<rt::PreparedSelection>(std::move(prepared));
           cache->variants[cache->next] = {selected.width, selected.height, selected.transform,
                                           operation};
           cache->next = (cache->next + 1) % cache->variants.size();
           return operation;
         };
       }});
}
void attach_boxdecode_fixed_image_guard(KernelStageRecipe& recipe, std::size_t member_index,
                                        BoxDecodeFixedImage image) {
  if (member_index >= recipe.members.size() || image.original_width < 1 ||
      image.original_height < 1 || image.canvas_width < 1 || image.canvas_height < 1 ||
      image.resize_mode.empty())
    throw std::invalid_argument(
        "BoxDecode fixed image guard requires a member, the image size and its preprocessing");
  recipe.resolver_factories.push_back(
      {member_index, [image = std::move(image)]() -> KernelStageRecipe::Resolver {
         return [image](const rt::VariantPreparer& preparer,
                        std::span<GstBuffer* const> buffers) -> rt::Result<rt::PreparedSelection> {
           const auto& base = preparer.base();
           if (!base || buffers.empty())
             return invalid("prepared operation and input buffers are required");
           bool described = false;
           for (auto* buffer : buffers) {
             auto parsed = read_geometry(buffer, false, false);
             if (const auto* error = std::get_if<rt::Error>(&parsed))
               return *error;
             const auto& geometry = std::get<Geometry>(parsed);
             // An input without preprocess facts (a prompt) describes no image.
             if (!geometry.width && !geometry.height)
               continue;
             described = true;
             // As cold admission requires: one image, not an ROI list.
             if (geometry.roi_list)
               return invalid("Segment Anything decodes one image; ROI-list preprocessing is not "
                              "admitted");
             if (geometry.width != image.original_width || geometry.height != image.original_height)
               return invalid("the prepared output is sized for another original image; a new "
                              "frame size requires a new runner");
             // The kernel maps masks back with the admitted resize, canvas and placement.
             if (geometry.resize != image.resize_mode ||
                 geometry.canvas_width != image.canvas_width ||
                 geometry.canvas_height != image.canvas_height ||
                 !std::equal(image.placed.begin(), image.placed.end(), geometry.placed.begin()))
               return invalid("the frame was preprocessed with another resize mode, canvas or "
                              "placement than the prepared plan; other preprocessing requires a "
                              "new runner");
           }
           // Every run decodes for an image: without its facts the masks would silently use the
           // previous image's geometry.
           if (!described)
             return invalid("Segment Anything decoding needs the image's semantic.preprocess facts "
                            "on its inputs on every run; pass the encoder outputs, for example "
                            "through .cvu()");
           return base;
         };
       }});
}
} // namespace simaai::neat::pipeline_internal::sima
