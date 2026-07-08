#pragma once

#include "simaai/neat/pcie/Model.h"

#include <optional>
#include <string>

namespace simaai::neat::pcie::internal {

struct ModelOptionsJson {
  std::optional<std::string> json;
};

ModelOptionsJson write_model_options_json(const simaai::neat::pcie::ModelOptions& options);

} // namespace simaai::neat::pcie::internal
