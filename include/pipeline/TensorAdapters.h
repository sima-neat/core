/**
 * @file
 * @ingroup tensors
 * @brief Adapters for mapping GStreamer samples into Tensor.
 */
#pragma once

#include "pipeline/TensorCore.h"

#if defined(SIMA_WITH_OPENCV)
#include "pipeline/TensorOpenCV.h"
#endif

typedef struct _GstSample GstSample;

namespace simaai::neat {

Tensor from_gst_sample(GstSample* sample);

} // namespace simaai::neat
