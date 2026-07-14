/**
 * @file
 * @brief Umbrella include for SiMa NEAT's Graph and runtime tensor types.
 *
 * Pulls in the Graph/Run lifecycle types (Graph, Run, GraphOptions, NeatError,
 * GraphReport) along with the tensor surface area applications interact with at
 * runtime (Tensor, TensorCore, TensorSpec, TensorTypes, TensorAdapters,
 * TensorConversion, TensorOpenCV, TessellatedTensor).
 *
 * Include this instead of cherry-picking individual `pipeline` subheaders.
 */
#pragma once

#include "pipeline/GraphOptions.h"
#include "pipeline/LatestByStreamFrameTap.h"
#include "pipeline/Run.h"
#include "pipeline/Graph.h"
#include "pipeline/RunExport.h"
#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/TensorConversion.h"
#include "pipeline/TensorCore.h"
#include "pipeline/TensorOpenCV.h"
#include "pipeline/TensorSpec.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/TessellatedTensor.h"

#include <future>

namespace simaai::neat {

/**
 * @brief Initialize NEAT's runtime substrate early.
 *
 * This performs the same one-time GStreamer/plugin setup that the first
 * Graph::build() would otherwise perform. Calling it during application
 * startup/model loading can hide cold plugin-registry cost from the first
 * latency-sensitive build.
 */
void prewarm_runtime();

/**
 * @brief Start NEAT runtime prewarm on a background thread.
 *
 * The returned future rethrows any initialization error from prewarm_runtime().
 */
std::future<void> prewarm_runtime_async();

} // namespace simaai::neat
