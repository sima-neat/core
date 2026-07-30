#pragma once

#include "simaai/neat/pcie/Model.h"

#include <cstdint>
#include <optional>

namespace simaai::neat::pcie::internal {

struct RuntimeInferenceResult {
  std::int32_t request_id = 0;
  TensorList outputs;
};

class RuntimeModelAccess {
public:
  static bool try_push(Model& model, std::int32_t request_id, const TensorList& tensors);
  static std::optional<RuntimeInferenceResult> pull_result(Model& model, int timeout_ms);
};

} // namespace simaai::neat::pcie::internal
