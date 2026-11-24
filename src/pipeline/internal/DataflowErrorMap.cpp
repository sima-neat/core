// The Dataflow backend's error table: a runtime Error (dataflow/diag/error.h) to Core's PullStatus
// and error_codes::* families, and to whether its action is to run the program on the GStreamer
// backend. Built only with SIMANEAT_DATAFLOW_SOURCE_DIR.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/internal/DataflowErrorMap.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

namespace {

// A model route loads its models and checks its kernels before any Net exists (soc::MlaModel::load
// at Phase::Prepare, soc::Kernel::make at Phase::Describe), so their errors name no operator; the
// operators and ports of a Plan name theirs (Plan::instantiate) and the Plan's own checks are
// Origin::Dataflow.
enum class Load { None, Model, Kernel };

Load load_of(const df::Error& e) noexcept {
  if (e.origin == df::Origin::Dataflow || e.op != df::kNoOperator) {
    return Load::None;
  }
  return e.phase == df::Phase::Prepare    ? Load::Model
         : e.phase == df::Phase::Describe ? Load::Kernel
                                          : Load::None;
}

// Rows in ErrorCode order. Every error a Run being built on Dataflow meets is Dataflow's own, and
// its action is the GStreamer backend, except two that GStreamer meets as well: a device the
// process lost (Unretired; nothing in the process may drive it again) and a model file the MLA
// does not load (GStreamer loads the same file).
//   InvalidArgument  Process: a pushed extent outside the input's bounds (Core checks it first, so
//                    this row is the backstop): misconfig.input_shape. The MLA's load: no such
//                    file, or not a compiled model: io.open. The Kernels package refused a kernel's
//                    request: misconfig.kernel_request. Otherwise the Net or Plan Core composed
//                    (compile, instantiate, start): misconfig.pipeline_shape.
//   ResourceExhausted  origin Runner: the device heap (CMA) has no memory for a device pool's slot,
//                    at Graph::build or at the input's first push:
//                    resource.device_memory_exhausted.
//   Timeout          Process: a pull's deadline: PullStatus::Timeout, not latched. Stop and Drain:
//                    logged by the caller only. Origin Runner: a device job that did not finish in
//                    the Runner's time: infra.accelerator_execution_failed, latched.
//   Unsupported      The MLA's load: the MLA's context cannot be opened in this process:
//                    infra.dispatcher_unavailable. The Kernels package has no implementation of a
//                    kernel for its target: misconfig.kernel_unsupported. Another origin Runner:
//                    the device memory's caches cannot be maintained in this process, a device
//                    without the Kernels firmware: infra.accelerator_execution_failed. Origin
//                    Dataflow: misconfig.pipeline_shape.
//   Unretired        a push whose device slot's write epoch could not end (the Run latches it:
//                    DataflowRun.cpp, settle), or a lost device.
//   Internal         origins Runner, Kernels, Codecs: infra.accelerator_execution_failed.
DataflowErrorMapping row(const df::Error& e) noexcept {
  const Load load = load_of(e);
  switch (e.code) {
  case df::ErrorCode::InvalidArgument:
    if (e.phase == df::Phase::Process) {
      return {PullStatus::Error, error_codes::kInputShape, true, true, ""};
    }
    if (load == Load::Model) {
      return {PullStatus::Error, error_codes::kIoOpen, true, false,
              "Check the model package: the MLA does not load its model file."};
    }
    return {PullStatus::Error,
            load == Load::Kernel ? error_codes::kKernelRequest : error_codes::kPipelineShape, true,
            true, ""};
  case df::ErrorCode::ResourceExhausted:
    return {PullStatus::Error,
            e.origin == df::Origin::Runner ? error_codes::kDeviceMemoryExhausted
                                           : error_codes::kMemoryAllocationFailed,
            true, true, ""};
  case df::ErrorCode::Backpressure:
    // A push outcome (try_push false); a pull never returns it.
    return {PullStatus::Error, error_codes::kRuntimePull, false, true, ""};
  case df::ErrorCode::Timeout:
    // Origin Runner: a job the device did not finish in the Runner's own time (MlaOp). The
    // operator failed and its Session with it: the Run's terminal error, not a pull's deadline.
    if (e.origin == df::Origin::Runner) {
      return {PullStatus::Error, error_codes::kAcceleratorExecutionFailed, true, true, ""};
    }
    return {e.phase == df::Phase::Process ? PullStatus::Timeout : PullStatus::Error,
            error_codes::kRuntimePull, false, true, ""};
  case df::ErrorCode::Closed:
    // Closed and drained, refused after stop, a closed pool or port: one Closed for Core.
    return {PullStatus::Closed, error_codes::kRuntimePull, false, true, ""};
  case df::ErrorCode::Unsupported:
    if (load == Load::Model) {
      return {PullStatus::Error, error_codes::kDispatcherUnavailable, true, true, ""};
    }
    if (load == Load::Kernel) {
      return {PullStatus::Error, error_codes::kKernelUnsupported, true, true, ""};
    }
    return {PullStatus::Error,
            e.origin == df::Origin::Runner ? error_codes::kAcceleratorExecutionFailed
                                           : error_codes::kPipelineShape,
            true, true, ""};
  case df::ErrorCode::Stale:
    return {PullStatus::Error, error_codes::kInternalPluginFailure, true, true, ""};
  case df::ErrorCode::Unretired:
    return {PullStatus::Error, error_codes::kAcceleratorExecutionFailed, true, false, ""};
  case df::ErrorCode::Internal:
    return {PullStatus::Error,
            e.origin == df::Origin::Dataflow || e.origin == df::Origin::Core ||
                    e.origin == df::Origin::Application
                ? error_codes::kInternalPluginFailure
                : error_codes::kAcceleratorExecutionFailed,
            true, true, ""};
  }
  return {PullStatus::Error, error_codes::kInternalPluginFailure, true, true, ""};
}

} // namespace

DataflowErrorMapping map_dataflow_error(const df::Error& error) noexcept {
  return row(error);
}

std::string dataflow_error_text(const df::Error& error, std::string_view where) {
  std::string text(where);
  text += ": ";
  text += (error.message && *error.message) ? error.message : "runtime error";
  text += " (Dataflow backend)";
  return text;
}

[[noreturn]] void throw_dataflow_error(const df::Error& error, std::string_view where,
                                       std::string_view detail, DataflowErrorSite site) {
  const DataflowErrorMapping mapping = map_dataflow_error(error);
  std::string summary = dataflow_error_text(error, where);
  if (!detail.empty()) {
    summary += "\n";
    summary += detail;
  }
  std::string action = mapping.hint;
  const auto add = [&action](const std::string& sentence) {
    action += action.empty() ? "" : " ";
    action += sentence;
  };
  if (site == DataflowErrorSite::FirstPush) {
    add("The input was bound by this push: declare the input's shape or build the Run with a "
        "seed, and this is found at Graph::build.");
  }
  if (mapping.backend_action &&
      (site != DataflowErrorSite::Running || error.code == df::ErrorCode::Unsupported)) {
    add(dataflow_backend_action(DataflowRemedy::Backend));
  }
  error_util::throw_session_error(mapping.code, summary, {}, action);
}

} // namespace simaai::neat::pipeline_internal

#endif // SIMANEAT_WITH_DATAFLOW
