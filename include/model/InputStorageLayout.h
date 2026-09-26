#pragma once

#include <map>
#include <string>

namespace simaai::neat {

/// Storage expected by an MLA input, independently of its logical channel count.
enum class InputStorageLayout { HWC, HWC16 };

/// Overrides keyed by public MPK input name; omitted inputs use MPK metadata.
using InputStorageLayouts = std::map<std::string, InputStorageLayout>;

} // namespace simaai::neat
