#pragma once

#include <string>

namespace simaai::neat {

class NextCpuConfigurable {
public:
  virtual ~NextCpuConfigurable() = default;
  virtual bool set_next_cpu_if_auto(const std::string& next_cpu) = 0;
};

} // namespace simaai::neat
