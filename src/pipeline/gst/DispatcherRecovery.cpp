#include "pipeline/internal/DispatcherRecovery.h"

namespace simaai::neat::pipeline_internal {

bool match_dispatcher_unavailable(const std::string& message) {
  return message.find("Unable to connect to the server from dispatcher") != std::string::npos;
}

bool is_dispatcher_unavailable(const GraphReport& report) {
  return simaai::neat::error_codes::is_dispatcher_unavailable(report.error_code);
}

} // namespace simaai::neat::pipeline_internal
