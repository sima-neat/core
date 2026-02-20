#pragma once

#include <functional>

#include <nlohmann/json.hpp>

namespace simaai::neat {

class ConfigJsonOverride {
public:
  virtual ~ConfigJsonOverride() = default;
  virtual bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                                    const std::string& tag) = 0;
};

} // namespace simaai::neat
