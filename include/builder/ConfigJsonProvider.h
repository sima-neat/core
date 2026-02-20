#pragma once

#include <nlohmann/json.hpp>

namespace simaai::neat {

class ConfigJsonProvider {
public:
  virtual ~ConfigJsonProvider() = default;
  virtual const nlohmann::json* config_json() const = 0;
};

} // namespace simaai::neat
