#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/MaskOptions.h"
#include <cmath>
#include <stdexcept>

namespace simaai::neat::pipeline_internal::sima {
// RF export profile v1 binds normalized cxcywh, independent sigmoid scores,
// and optional query-mask logits to the model's declared output slots.
struct RfDetrStaticContract {
  int boxes_input_index = -1;
  int scores_input_index = -1;
  int masks_input_index = -1;
  int candidate_limit = 300;
  MaskOptions masks;
};
inline void validate_rfdetr_controls(double score_threshold, double nms, int top_k,
                                     const MaskOptions& masks) {
  const auto probability = [](double x) { return std::isfinite(x) && x >= 0 && x <= 1; };
  if (!probability(score_threshold) || !probability(masks.threshold) || nms != 0 || top_k < 0)
    throw std::invalid_argument(
        "RF-DETR requires finite probability thresholds, nonnegative top_k and no NMS");
  if (masks.output != MaskOutput::Binary && masks.output != MaskOutput::Probabilities)
    throw std::invalid_argument("RF-DETR mask output is unsupported");
  if (masks.size != MaskSize::Native && masks.size != MaskSize::Source &&
      masks.size != MaskSize::Fixed)
    throw std::invalid_argument("RF-DETR mask size is unsupported");
  if ((masks.size == MaskSize::Fixed && (masks.width <= 0 || masks.height <= 0)) ||
      (masks.size != MaskSize::Fixed && (masks.width != 0 || masks.height != 0)) ||
      (masks.output == MaskOutput::Probabilities && masks.size != MaskSize::Native))
    throw std::invalid_argument(
        "RF-DETR fixed masks require positive dimensions; probability masks require native size");
}
} // namespace simaai::neat::pipeline_internal::sima
