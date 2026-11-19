#pragma once

#include <string>

namespace simaai::neat {

struct NameTransform {
  std::string prefix;
  std::string suffix;
};

// Fragment-backed nodes can retain element names rendered by the graph that saved them.
struct SerializedElementNames {
  NameTransform applied_name_transform;
};

} // namespace simaai::neat
