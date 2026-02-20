#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace simaai::neat {

class ConfigJsonConsumer {
public:
  virtual ~ConfigJsonConsumer() = default;
  virtual void apply_upstream_config(const nlohmann::json& upstream,
                                     const std::string& upstream_kind) = 0;
};

} // namespace simaai::neat
