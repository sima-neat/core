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
#include "pipeline/Run.h"
#include "pipeline/Graph.h"
#include "pipeline/RunExport.h"
#include "pipeline/TelemetryMonitor.h"
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
 * This performs the same one-time setup that the first Graph::build() on the
 * process's backend (set_backend) would otherwise perform: on GStreamer, its
 * plugin setup; on Dataflow, the device memory's (a process on Dataflow never
 * starts GStreamer). Calling it during application startup/model loading can
 * hide that cold cost from the first latency-sensitive build. It reads the
 * backend set so far and does not fix it.
 */
void prewarm_runtime();

/**
 * @brief Start NEAT runtime prewarm on a background thread.
 *
 * The returned future rethrows any initialization error from prewarm_runtime().
 */
std::future<void> prewarm_runtime_async();

/**
 * @brief Select the backend that runs this process's Models and Graphs.
 *
 * A process runs on one backend, `RunBackend::GStreamer` unless the program selects another. Call
 * this before the program constructs its first `Model` or builds its first `Graph`: every
 * `Model`, `Graph`, `Run`, `Model::Runner` and benchmark of the process then runs on `backend`.
 * This call is the only selector: no option of a Run and no environment variable chooses one.
 *
 * The first `Model` constructed or `Graph` built fixes the backend for the rest of the process,
 * whether or not that object still exists and whether or not its build succeeded. Until then the
 * call may be repeated, from any thread. After it, a call with the same value does nothing, and a
 * call with the other value throws `NeatError` (`misconfig.option_out_of_range`) and changes
 * nothing.
 *
 * `RunBackend::Dataflow` throws `NeatError` in a library built without the Dataflow backend; such
 * a library runs everything on GStreamer.
 */
void set_backend(RunBackend backend);

/**
 * @brief The backend this process's Models and Graphs run on (see `set_backend`).
 */
RunBackend backend();

} // namespace simaai::neat
