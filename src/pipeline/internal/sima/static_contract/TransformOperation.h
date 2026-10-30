#pragma once

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <array>
#include <optional>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// Core's graph rewrite policy, not a numerical implementation catalogue. Recognize
// mandatory pairs even when their implementation is absent from the installed SDK.
// Numerical admission and compiled target availability remain owned by Kernels.
struct TransformOperationRule {
  std::string_view operation;
  OpKind first;
  std::optional<OpKind> second;
  static constexpr std::uint32_t maximum_members = 32U;
  constexpr std::size_t semantic_pattern_length() const noexcept {
    return second ? 2U : 1U;
  }
  constexpr bool mandatory_when_matched() const noexcept {
    return second.has_value();
  }
};

// Typed rewrite rules do not depend on the installed implementation catalogue.
inline constexpr std::array transform_operation_rules{
    TransformOperationRule{"tess", OpKind::Tessellate, {}},
    TransformOperationRule{"detess", OpKind::Detessellate, {}},
    TransformOperationRule{"cast", OpKind::Cast, {}},
    TransformOperationRule{"quant", OpKind::Quantize, {}},
    TransformOperationRule{"dequant", OpKind::Dequantize, {}},
    TransformOperationRule{"casttess", OpKind::Cast, OpKind::Tessellate},
    TransformOperationRule{"detesscast", OpKind::Detessellate, OpKind::Cast},
    TransformOperationRule{"quanttess", OpKind::Quantize, OpKind::Tessellate},
    TransformOperationRule{"detessdequant", OpKind::Detessellate, OpKind::Dequantize}};

inline const TransformOperationRule* transform_operation(std::string_view name) noexcept {
  for (const auto& rule : transform_operation_rules)
    if (rule.operation == name)
      return &rule;
  return nullptr;
}
inline const TransformOperationRule*
transform_operation(OpKind first, std::optional<OpKind> second = {}) noexcept {
  for (const auto& rule : transform_operation_rules)
    if (rule.first == first && rule.second == second)
      return &rule;
  return nullptr;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
