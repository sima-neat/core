/**
 * @file
 * @ingroup pipeline
 * @brief Format-string predicates for MLA-tessellated tensor payloads.
 *
 * The Neat dtype contract keeps tessellated tensors out of the public Tensor
 * API; this header exposes a couple of small predicates used by advanced code
 * paths and diagnostics that need to recognise tessellated payload formats by
 * their caps-string token.
 */
#pragma once

#include <string>

namespace simaai::neat {

/// @brief True iff @p fmt names an INT8 tessellated tensor format string.
bool is_tessellated_int8_format(const std::string& fmt);
/// @brief True iff @p fmt names a BF16 tessellated tensor format string.
bool is_tessellated_bf16_format(const std::string& fmt);

} // namespace simaai::neat
