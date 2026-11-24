#include "pipeline/internal/DataflowNotCarried.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

namespace {

// How the program selects the GStreamer backend: the one place that names it.
constexpr const char* kSelectGStreamer =
    "build this Run with RunOptions::backend = RunBackend::GStreamer";

} // namespace

std::string dataflow_not_carried_sentence(const DataflowNotCarried& what) {
  return what.subject + " is not carried by the Dataflow backend" +
         (what.remedy == DataflowRemedy::BackendOrFirmware ? " on this board: " : ": ") + what.fact;
}

std::string dataflow_backend_action(DataflowRemedy remedy) {
  std::string action = std::string("To run this program on the GStreamer backend, ") +
                       kSelectGStreamer + ".";
  switch (remedy) {
  case DataflowRemedy::Backend:
    break;
  case DataflowRemedy::BackendAndRebuild:
    action += " A Run keeps the backend it was built on.";
    break;
  case DataflowRemedy::BackendOrFirmware:
    action += " Or select the Kernels firmware on the board (package simaai-kernels-firmware) "
              "and start the program again.";
    break;
  }
  return action;
}

[[noreturn]] void throw_dataflow_not_carried(std::string_view where,
                                             const DataflowNotCarried& what) {
  error_util::throw_session_error(what.code ? what.code : error_codes::kPipelineShape,
                                  std::string(where) + ": " + dataflow_not_carried_sentence(what) +
                                      ".",
                                  {}, dataflow_backend_action(what.remedy));
}

} // namespace simaai::neat::pipeline_internal
