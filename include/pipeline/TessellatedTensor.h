#pragma once

#include <string>

namespace simaai::neat {

bool is_tessellated_int8_format(const std::string& fmt);
bool is_tessellated_bf16_format(const std::string& fmt);

} // namespace simaai::neat
