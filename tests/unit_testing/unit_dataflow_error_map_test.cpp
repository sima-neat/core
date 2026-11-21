// The Dataflow backend's error table (src/pipeline/internal/DataflowErrorMap.cpp, compiled into this
// test: libsima_neat.so hides it): for each runtime Error Core can meet, the Core error code, the
// pull status, whether a pull latches it, and whether RunBackend::Auto builds a Run on GStreamer when
// Graph::build meets it on Dataflow. Among them the MLA's load errors as the runtime reports them (a
// missing file or one that is not a model: InvalidArgument; a context that cannot be opened:
// Unsupported; a lost device: Unretired) and the Kernels package's. No device is opened.

#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

using namespace simaai::neat;
namespace df = pipeline_internal::df;
using pipeline_internal::DataflowErrorMapping;
using pipeline_internal::DataflowErrorOnEveryBackend;

constexpr std::uint16_t kOperator = 1;

df::Error error(df::ErrorCode code, df::Origin origin, df::Phase phase,
                std::uint16_t op = df::kNoOperator, const char* message = "the runtime's text") {
  return df::Error{.code = code, .origin = origin, .phase = phase, .op = op, .port = df::kNoPort,
                   .message = message};
}

struct Row {
  const char* what;
  df::Error error;
  const char* code;
  PullStatus status;
  bool terminal;
  bool retry_on_gstreamer;
};

void table() {
  using C = df::ErrorCode;
  using O = df::Origin;
  using P = df::Phase;
  const Row rows[] = {
      // soc::MlaModel::load, before any Net: the MLA's own codes.
      {"MLA load: no such file, or not a compiled model", error(C::InvalidArgument, O::Runner, P::Prepare),
       error_codes::kIoOpen, PullStatus::Error, true, false},
      {"MLA load: the context cannot be opened", error(C::Unsupported, O::Runner, P::Prepare),
       error_codes::kDispatcherUnavailable, PullStatus::Error, true, true},
      {"MLA load: out of memory", error(C::ResourceExhausted, O::Runner, P::Prepare),
       error_codes::kDeviceMemoryExhausted, PullStatus::Error, true, true},
      {"MLA load: the device was lost", error(C::Unretired, O::Runner, P::Prepare),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, false},
      {"MLA load: the Runner did not load the model", error(C::Internal, O::Runner, P::Prepare),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, true},
      // soc::Kernel::make, before any Net.
      {"Kernels: the request is refused", error(C::InvalidArgument, O::Runner, P::Describe),
       error_codes::kKernelRequest, PullStatus::Error, true, true},
      {"Kernels: no implementation for the target", error(C::Unsupported, O::Runner, P::Describe),
       error_codes::kKernelUnsupported, PullStatus::Error, true, true},
      // Plan::instantiate: an operator or a port of the Plan names itself.
      {"MlaOp: a port's spec is not the model's", error(C::InvalidArgument, O::Runner, P::Prepare, kOperator),
       error_codes::kPipelineShape, PullStatus::Error, true, true},
      {"MlaOp: a model with more than one input", error(C::Unsupported, O::Runner, P::Prepare, kOperator),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, true},
      {"a device pool: no CMA", error(C::ResourceExhausted, O::Runner, P::Prepare, kOperator),
       error_codes::kDeviceMemoryExhausted, PullStatus::Error, true, true},
      {"a host pool: no memory", error(C::ResourceExhausted, O::Dataflow, P::Prepare, kOperator),
       error_codes::kMemoryAllocationFailed, PullStatus::Error, true, true},
      {"an SoC operator after a lost device", error(C::Unretired, O::Runner, P::Prepare, kOperator),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, false},
      // Plan::compile and the Net.
      {"Plan: a spec mismatch", error(C::InvalidArgument, O::Dataflow, P::Describe),
       error_codes::kPipelineShape, PullStatus::Error, true, true},
      {"Plan: unsupported", error(C::Unsupported, O::Dataflow, P::Describe),
       error_codes::kPipelineShape, PullStatus::Error, true, true},
      // A running Session.
      {"push: an extent outside the bounds", error(C::InvalidArgument, O::Dataflow, P::Process),
       error_codes::kInputShape, PullStatus::Error, true, true},
      {"pull: the deadline", error(C::Timeout, O::Dataflow, P::Process), error_codes::kRuntimePull,
       PullStatus::Timeout, false, true},
      {"MlaOp: the Runner's own deadline", error(C::Timeout, O::Runner, P::Retire, kOperator),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, true},
      {"pull: closed and drained", error(C::Closed, O::Dataflow, P::Process), error_codes::kRuntimePull,
       PullStatus::Closed, false, true},
      {"push: a device slot's write epoch did not end", error(C::Unretired, O::Runner, P::Process),
       error_codes::kAcceleratorExecutionFailed, PullStatus::Error, true, false},
      {"an operator's failure", error(C::Internal, O::Dataflow, P::Process, kOperator),
       error_codes::kInternalPluginFailure, PullStatus::Error, true, true},
  };
  for (const Row& r : rows) {
    const DataflowErrorMapping m = pipeline_internal::map_dataflow_error(r.error);
    require(std::string(m.code) == r.code,
            std::string(r.what) + ": code " + m.code + ", expected " + r.code);
    require(m.status == r.status, std::string(r.what) + ": pull status");
    require(m.terminal == r.terminal, std::string(r.what) + ": latched or not");
    require(m.retry_on_gstreamer == r.retry_on_gstreamer,
            std::string(r.what) + ": RunBackend::Auto's retry on GStreamer");
  }
}

// What throw_dataflow_error throws: the table's code, "<where>: <text> (Dataflow backend)", the
// detail line, the row's action and the caller's; the type RunBackend::Auto reads.
void thrown() {
  const df::Error file =
      error(df::ErrorCode::InvalidArgument, df::Origin::Runner, df::Phase::Prepare, df::kNoOperator,
            "MLA: the file is not a model the MLA loads (no such file, or not a compiled model)");
  try {
    pipeline_internal::throw_dataflow_error(file, "Graph::build", "MLA model /no/such/model.elf");
    require(false, "throw_dataflow_error returned");
  } catch (const DataflowErrorOnEveryBackend& e) {
    const std::string what = e.what();
    require(e.report().error_code == error_codes::kIoOpen, "the model file: io.open");
    require_contains(what, "Graph::build: MLA: the file is not a model", "the phase and the text");
    require_contains(what, "(Dataflow backend)\nMLA model /no/such/model.elf", "the detail line");
    require_contains(what, "Hint: Check the model package", "what to do");
  }
  const df::Error context = error(df::ErrorCode::Unsupported, df::Origin::Runner, df::Phase::Prepare,
                                  df::kNoOperator, "MLA: the MLA's context cannot be opened");
  bool plain = false;
  try {
    pipeline_internal::throw_dataflow_error(context, "Run::push", {}, "The caller's action.");
  } catch (const DataflowErrorOnEveryBackend&) {
    plain = false;
  } catch (const NeatError& e) {
    plain = true;
    const std::string what = e.what();
    require(e.report().error_code == error_codes::kDispatcherUnavailable,
            "the context: infra.dispatcher_unavailable");
    require_contains(what, "Run::push: MLA: the MLA's context cannot be opened (Dataflow backend)",
                     "the phase the caller names");
    require_contains(what, "Hint: Build this Run with RunBackend::GStreamer. The caller's action.",
                     "the row's action, then the caller's");
    require(what.find("before H1") == std::string::npos, "no milestone in the text");
  }
  require(plain, "an error GStreamer would not meet is a plain NeatError");
}

} // namespace

RUN_TEST("unit_dataflow_error_map_test", ([] {
           table();
           thrown();
         }));
