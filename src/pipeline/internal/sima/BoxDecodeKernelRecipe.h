#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/PreprocessPlan.h"
#include <array>
#include <optional>
#include "pipeline/internal/contract/PluginCompiledContracts.h"

namespace simaai::neat::pipeline_internal::sima {
const char* boxdecode_output_format(BoxDecodeType type, SuperPointOutputFormat feature_format);

// Nominal geometry belongs to the admitted decode recipe, even for identity
// model-shaped input with no resize operation. Explicit provenance still wins.
constexpr ResizeMode default_boxdecode_resize_mode(BoxDecodeType type) {
  return (type == BoxDecodeType::Ssd || type == BoxDecodeType::Detr ||
          type == BoxDecodeType::DetrFocal || type == BoxDecodeType::EdgeTamSeg ||
          type == BoxDecodeType::EdgeTamTrack || type == BoxDecodeType::EfficientSam ||
          type == BoxDecodeType::EfficientSam3)
             ? ResizeMode::Stretch
             : ResizeMode::Letterbox;
}

struct BoxDecodeGeometryOptions {
  int model_width = 0, model_height = 0;
  int original_width = 0, original_height = 0;
  ResizeMode resize = ResizeMode::Letterbox;
  // Checked producer transform: original = model * scale + offset (x, y).
  // Complete-geometry profiles retain this together with the original extent.
  std::optional<std::array<float, 4>> model_to_original;
  bool complete_geometry = false;
  bool uniform_transform_required = false;
  // A source-driven cold seed must never substitute for per-frame authority.
  bool require_frame_transform = false;
};
// Axis-aligned ordinary boxes only; proposals and auxiliary payloads are not
// implicitly admitted when a new decode type is added.
bool boxdecode_supports_detection_transform(BoxDecodeType);
// Segment Anything: the resize each model was trained on (MobileSAM a top-left
// letterbox, the others a stretch).
ResizeMode sam_required_resize(BoxDecodeType);
// One prompt-decoder run's plane choice for Kernels profile sam_mask: "fixed"
// (plane first), "best_iou" (best IoU of [first, first + count)) or
// "stability" (plane 0 when stable, else best IoU of [first, first + count)).
struct SamMaskSelection {
  const char* mode = "fixed";
  int first = 0;
  int count = 1;
};
// Model::Options::multimask_output for a model's decoder with C mask channels,
// as upstream predict(multimask_output=...) answers it. Throws
// std::invalid_argument, naming the model, for a combination it cannot answer.
SamMaskSelection resolve_sam_mask_selection(BoxDecodeType, int masks, bool multimask_output);
struct BoxDecodeKernelRecipeResult {
  CompiledKernelContract compiled;
  std::size_t decode_member = 0;
};
// Cold lowering only: immutable numerical admission, explicit format transforms
// and Core wire publication. No input bytes, device connection or per-frame math.
// Geometry must come from node/preprocess contracts, never guessed from heads.
BoxDecodeKernelRecipeResult prepare_boxdecode_kernel(const CompiledBoxDecodeContract&,
                                                     const BoxDecodeGeometryOptions&);
} // namespace simaai::neat::pipeline_internal::sima
