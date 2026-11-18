#pragma once

#include <string_view>

namespace simaai::neat::contracts {

[[noreturn]] void throw_invalid_option(std::string_view message);
[[noreturn]] void throw_invalid_caps(std::string_view message);

} // namespace simaai::neat::contracts
