// The Dataflow backend's error table (H0 plan 3.8): a runtime Error (dataflow/diag/error.h) to
// Core's PullStatus and error_codes::* families. Built only with SIMANEAT_DATAFLOW_SOURCE_DIR.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/internal/DataflowErrorMap.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

namespace {

// Rows in ErrorCode order. Phase and Origin refine three of them:
//   InvalidArgument  Describe/Prepare/Start: the Net or Plan (compile, instantiate, start):
//                    misconfig.pipeline_shape. Process: FR1, a pushed extent outside the input's
//                    bounds (Core checks it first, so this row is the backstop):
//                    misconfig.input_shape.
//   Timeout          Process: FR8, a pull's deadline: PullStatus::Timeout, not latched. Stop/Drain:
//                    SR2/SR3, logged by the caller only.
//   Internal         Runner/Kernels/Codecs origins (H0b on): infra.accelerator_execution_failed.
//   ResourceExhausted, Unsupported  origin Runner (H0b, a device pool at Graph::build or at the
//                    input's first push): resource.device_memory_exhausted (no CMA) and
//                    infra.accelerator_execution_failed (no EL0 cache maintenance).
//   Unretired        FR16 (H0b): a push whose device slot's write epoch could not end; the Run
//                    latches it (DataflowRun.cpp, settle).
DataflowErrorMapping row(const df::Error& e) noexcept {
  switch (e.code) {
  case df::ErrorCode::InvalidArgument:
    return {PullStatus::Error,
            e.phase == df::Phase::Process ? error_codes::kInputShape : error_codes::kPipelineShape,
            true};
  case df::ErrorCode::ResourceExhausted:
    // Origin Runner (H0b): the device heap (CMA) has no memory for a slot of a device pool.
    return {PullStatus::Error,
            e.origin == df::Origin::Runner ? error_codes::kDeviceMemoryExhausted
                                           : error_codes::kMemoryAllocationFailed,
            true};
  case df::ErrorCode::Backpressure:
    // FR2 is a push outcome (try_push false); a pull never returns it (WORKER.md 15.2).
    return {PullStatus::Error, error_codes::kRuntimePull, false};
  case df::ErrorCode::Timeout:
    return {e.phase == df::Phase::Process ? PullStatus::Timeout : PullStatus::Error,
            error_codes::kRuntimePull, false};
  case df::ErrorCode::Closed:
    // FR9 (closed and drained), SR1 (refused after stop), O9/O10: one Closed for Core.
    return {PullStatus::Closed, error_codes::kRuntimePull, false};
  case df::ErrorCode::Unsupported:
    // Origin Runner (H0b): this process cannot maintain the caches of device memory.
    return {PullStatus::Error,
            e.origin == df::Origin::Runner ? error_codes::kAcceleratorExecutionFailed
                                           : error_codes::kPipelineShape,
            true};
  case df::ErrorCode::Stale:
    return {PullStatus::Error, error_codes::kInternalPluginFailure, true};
  case df::ErrorCode::Unretired:
    return {PullStatus::Error, error_codes::kAcceleratorExecutionFailed, true};
  case df::ErrorCode::Internal:
    return {PullStatus::Error,
            e.origin == df::Origin::Dataflow || e.origin == df::Origin::Core ||
                    e.origin == df::Origin::Application
                ? error_codes::kInternalPluginFailure
                : error_codes::kAcceleratorExecutionFailed,
            true};
  }
  return {PullStatus::Error, error_codes::kInternalPluginFailure, true};
}

} // namespace

DataflowErrorMapping map_dataflow_error(const df::Error& error) noexcept {
  return row(error);
}

std::string dataflow_error_text(const df::Error& error, std::string_view where) {
  std::string text(where);
  text += ": ";
  text += (error.message && *error.message) ? error.message : "runtime error";
  if (error.code == df::ErrorCode::Unsupported && error.origin != df::Origin::Runner) {
    text += "; not supported before H1";
  }
  text += " (Dataflow backend)";
  return text;
}

[[noreturn]] void throw_dataflow_error(const df::Error& error, std::string_view where,
                                       std::string_view detail) {
  std::string summary = dataflow_error_text(error, where);
  if (!detail.empty()) {
    summary += "\n";
    summary += detail;
  }
  error_util::throw_session_error(map_dataflow_error(error).code, summary);
}

} // namespace simaai::neat::pipeline_internal

#endif // SIMANEAT_WITH_DATAFLOW
