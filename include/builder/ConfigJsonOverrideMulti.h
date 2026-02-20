#pragma once

#include <functional>
#include <cstddef>

#include <nlohmann/json.hpp>

namespace simaai::neat {

class ConfigJsonOverrideMulti {
public:
  virtual ~ConfigJsonOverrideMulti() = default;
  virtual bool override_config_json_multi(
      const std::function<void(nlohmann::json&, std::size_t, std::size_t)>& edit,
      const std::string& tag) = 0;
};

} // namespace simaai::neat
