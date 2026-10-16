#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <gstsimaaitensorbuffer.h>
#include <simaai/scheduler/kernel_preparation.h>
#include <array>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {
namespace {
namespace rt = ::simaai::runtime;
namespace scheduler = ::simaai::scheduler;
struct Geometry {
  int width = 0, height = 0;
  std::string_view resize;
};
rt::Error invalid(const char* message) {
  return {rt::ErrorCode::InvalidRequest, "boxdecode.geometry", message};
}
rt::Result<Geometry> read_geometry(GstBuffer* buffer) {
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
    return Geometry{static_cast<int>(facts->original_width),
                    static_cast<int>(facts->original_height), facts->resize_mode};
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
  return geometry;
}
struct Cache {
  struct Entry {
    int width = 0, height = 0;
    std::shared_ptr<const rt::PreparedOperation> operation;
  };
  // The same immutable recipe can be prepared by more than one graph. Never
  // reuse another executor's admitted operation just because geometry matches.
  std::mutex mutex;
  std::shared_ptr<const rt::PreparedOperation> base;
  std::array<Entry, 4> variants;
  std::size_t next = 0;
};
} // namespace
void attach_boxdecode_geometry_resolver(KernelStageRecipe& recipe, std::size_t member_index,
                                        BoxDecodeGeometryOptions options,
                                        bool original_size_from_metadata,
                                        bool require_resize_metadata) {
  if (!original_size_from_metadata && !require_resize_metadata)
    return;
  const auto& member = recipe.members.at(member_index);
  std::optional<rt::KernelVariant> variant;
  if (original_size_from_metadata) {
    auto bound = scheduler::bind_kernel_variant(member.executable, "image_size");
    if (const auto* error = std::get_if<rt::Error>(&bound))
      throw std::invalid_argument("BoxDecode geometry: " + error->message);
    variant = std::get<rt::KernelVariant>(std::move(bound));
    const auto fields = variant->fields();
    if (variant->contract_policy() != rt::KernelVariantPolicy::PreserveContract ||
        fields.size() != 1 || fields.front() != "image_size")
      throw std::invalid_argument("BoxDecode geometry: incompatible image_size capability");
  }
  recipe.resolver_factories.push_back(
      {member_index,
       [variant = std::move(variant), options, original_size_from_metadata,
        require_resize_metadata]() -> KernelStageRecipe::Resolver {
         // The factory is reusable; only this request instance owns its mutable cache.
         return [cache = std::make_shared<Cache>(), variant, options, original_size_from_metadata,
                 require_resize_metadata](const rt::Executor& executor,
                                          const std::shared_ptr<const rt::PreparedOperation>& base,
                                          std::span<GstBuffer* const> buffers)
                    -> rt::Result<std::shared_ptr<const rt::PreparedOperation>> {
           if (!base || buffers.empty())
             return invalid("prepared operation and input buffers are required");
           Geometry selected;
           bool found = false;
           for (auto* buffer : buffers) {
             auto parsed = read_geometry(buffer);
             if (auto* error = std::get_if<rt::Error>(&parsed))
               return *error;
             const auto geometry = std::get<Geometry>(parsed);
             // Explicit image dimensions retain their original API precedence. Only
             // metadata-derived dimensions need agreement across split-head carriers.
             if (!geometry.resize.empty()) {
               if (!selected.resize.empty() && selected.resize != geometry.resize)
                 return invalid("input head carriers disagree on resize mode");
               selected.resize = geometry.resize;
             }
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
               options.resize == ResizeMode::Stretch ? "stretch" : "letterbox";
           if (require_resize_metadata && selected.resize.empty())
             return invalid("required resize-mode metadata is missing");
           if (!selected.resize.empty() && selected.resize != expected_resize)
             return invalid("preprocess resize mode differs from the checked decode geometry");
           if (!original_size_from_metadata)
             return base;
           if (!found)
             return invalid("original image dimensions require preprocess metadata");
           if (selected.width == options.original_width &&
               selected.height == options.original_height)
             return base;
           std::lock_guard lock(cache->mutex);
           if (cache->base != base) {
             cache->base = base;
             cache->variants = {};
             cache->next = 0;
           }
           for (const auto& entry : cache->variants)
             if (entry.operation && entry.width == selected.width &&
                 entry.height == selected.height)
               return entry.operation;
           const std::array<std::optional<::simaai::kernels::ParameterValue>, 1> values{
               ::simaai::kernels::ParameterValue{
                   ::simaai::kernels::ParameterArray{selected.width, selected.height}}};
           auto checked = scheduler::apply_kernel_variant(*variant, {{}, {}, values});
           if (const auto* error = std::get_if<rt::Error>(&checked))
             return *error;
           auto prepared =
               executor.prepare(std::get<::simaai::kernels::Executable>(std::move(checked)), *base);
           if (const auto* error = std::get_if<rt::Error>(&prepared))
             return *error;
           auto operation = std::make_shared<const rt::PreparedOperation>(
               std::get<rt::PreparedOperation>(std::move(prepared)));
           cache->variants[cache->next] = {selected.width, selected.height, operation};
           cache->next = (cache->next + 1) % cache->variants.size();
           return operation;
         };
       }});
}
} // namespace simaai::neat::pipeline_internal::sima
