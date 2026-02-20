#pragma once

#include <string>

namespace simaai::neat::pipeline_internal {

std::string make_temp_json_path(const std::string& dir, const std::string& prefix,
                                const char* error_label);

} // namespace simaai::neat::pipeline_internal
