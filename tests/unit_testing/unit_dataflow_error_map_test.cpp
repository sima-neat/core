// The Dataflow backend's error table (src/pipeline/internal/DataflowErrorMap.cpp, compiled into
// this test: libsima_neat.so hides it): for each runtime Error Core can meet, the Core error code,
// the pull status, whether a pull latches it, and whether its action is to run the program on the
// GStreamer backend (false where GStreamer meets the same error). Among them the MLA's load errors
// as the runtime reports them (a missing file or one that is not a model: InvalidArgument; a
// context that cannot be opened: Unsupported; a lost device: Unretired) and the Kernels package's.
// No device is opened.

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
using pipeline_internal::DataflowErrorSite;

constexpr std::uint16_t kOperator = 1;

df::Error error(df::ErrorCode code, df::Origin origin, df::Phase phase,
                std::uint16_t op = df::kNoOperator, const char* message = "the runtime's text") {
  return df::Error{.code = code,
                   .origin = origin,
                   .phase = phase,
                   .op = op,
                   .port = df::kNoPort,
                   .message = message};
}

struct Row {
  const char* what;
  df::Error error;
  const char* code;
  PullStatus status;
  bool terminal;
  bool backend_action;
};

void table() {
  using C = df::ErrorCode;
  using O = df::Origin;
  using P = df::Phase;
  const Row rows[] = {
      // soc::MlaModel::load, before any Net: the MLA's own codes.
      {"MLA load: no such file, or not a compiled model",
       error(C::InvalidArgument, O::Runner, P::Prepare), error_codes::kIoOpen, PullStatus::Error,
       true, false},
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
      {"MlaOp: a port's spec is not the model's",
       error(C::InvalidArgument, O::Runner, P::Prepare, kOperator), error_codes::kPipelineShape,
       PullStatus::Error, true, true},
      {"MlaOp: a model with more than one input",
       error(C::Unsupported, O::Runner, P::Prepare, kOperator),
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
      {"pull: closed and drained", error(C::Closed, O::Dataflow, P::Process),
       error_codes::kRuntimePull, PullStatus::Closed, false, true},
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
    require(m.backend_action == r.backend_action,
            std::string(r.what) + ": whether the action is the GStreamer backend");
  }
}

// What throw_dataflow_error throws: the table's code, "<where>: <text> (Dataflow backend)", the
// detail line, and as its hint the row's own and the action its site gives the row: the GStreamer
// backend for an error that is the Dataflow backend's own, nothing for one GStreamer meets too.
void thrown() {
  const std::string backend =
      pipeline_internal::dataflow_backend_action(pipeline_internal::DataflowRemedy::Backend);
  const df::Error file =
      error(df::ErrorCode::InvalidArgument, df::Origin::Runner, df::Phase::Prepare, df::kNoOperator,
            "MLA: the file is not a model the MLA loads (no such file, or not a compiled model)");
  try {
    pipeline_internal::throw_dataflow_error(file, "Graph::build", "MLA model /no/such/model.elf",
                                            DataflowErrorSite::Build);
    require(false, "throw_dataflow_error returned");
  } catch (const NeatError& e) {
    const std::string what = e.what();
    require(e.report().error_code == error_codes::kIoOpen, "the model file: io.open");
    require_contains(what, "Graph::build: MLA: the file is not a model", "the phase and the text");
    require_contains(what, "(Dataflow backend)\nMLA model /no/such/model.elf", "the detail line");
    require_contains(what, "Hint: Check the model package", "what to do");
    require(what.find("GStreamer") == std::string::npos,
            "an error GStreamer meets too names no backend action: " + what);
  }
  const df::Error context =
      error(df::ErrorCode::Unsupported, df::Origin::Runner, df::Phase::Prepare, df::kNoOperator,
            "MLA: the MLA's context cannot be opened");
  const struct {
    DataflowErrorSite site;
    const char* where;
    std::string hint;
  } sites[] = {
      {DataflowErrorSite::Build, "Graph::build", "Hint: " + backend},
      {DataflowErrorSite::FirstPush, "Run::push",
       "Hint: The input was bound by this push: declare the input's shape or build the Run with "
       "a seed, and this is found at Graph::build. " +
           backend},
      {DataflowErrorSite::Running, "Run::push", "Hint: " + backend},
  };
  for (const auto& s : sites) {
    try {
      pipeline_internal::throw_dataflow_error(context, s.where, {}, s.site);
      require(false, "throw_dataflow_error returned");
    } catch (const NeatError& e) {
      const std::string what = e.what();
      require(e.report().error_code == error_codes::kDispatcherUnavailable,
              "the context: infra.dispatcher_unavailable");
      require_contains(what,
                       std::string(s.where) +
                           ": MLA: the MLA's context cannot be opened (Dataflow backend)",
                       "the call the caller names");
      require(what.size() >= s.hint.size() &&
                  what.compare(what.size() - s.hint.size(), s.hint.size(), s.hint) == 0,
              "the site's action ends the text: " + what);
    }
  }
  // On a running Session only what the backend does not implement names the backend action.
  const df::Error lost_job = error(df::ErrorCode::Internal, df::Origin::Runner, df::Phase::Process,
                                   kOperator, "the device did not finish the job");
  try {
    pipeline_internal::throw_dataflow_error(lost_job, "Run::push", {}, DataflowErrorSite::Running);
    require(false, "throw_dataflow_error returned");
  } catch (const NeatError& e) {
    require(std::string(e.what()).find("Hint:") == std::string::npos,
            std::string("a device failure on a running Session names no action: ") + e.what());
  }
}

} // namespace

RUN_TEST("unit_dataflow_error_map_test", ([] {
           table();
           thrown();
         }));
