/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Helpers for working with `BoxDecodeType`
 *        enums and their token strings.
 *
 * Small set of converters that turn the public `BoxDecodeType` / `BoxDecodeTypeOption` enums
 * into stable string tokens (and back) used in MPK manifests, diagnostics, and route-planning
 * decisions. Used by the route planner and the box-decode static-contract extractor; not part
 * of the public Neat API.
 *
 * @see BoxDecodeType (public enum)
 * @see BoxDecodeStaticContractExtractor.h
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/BoxDecodeType.h"

#include <optional>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

/// Parse a token string (e.g., `"yolov5"`, `"ssd"`) into a `BoxDecodeType`; nullopt if unknown.
std::optional<BoxDecodeType> parse_box_decode_type_token(std::string_view token);

/// Parse a token string into a `BoxDecodeTypeOption`; nullopt if unknown.
std::optional<BoxDecodeTypeOption> parse_box_decode_type_option_token(std::string_view token);

/// True when `type` is anything other than `BoxDecodeType::Unspecified`.
bool is_box_decode_type_specified(BoxDecodeType type);

/// True when the option implies all class-score channels live in the same domain (e.g.,
/// post-sigmoid).
bool is_box_decode_type_option_requires_uniform_score_domain(BoxDecodeTypeOption option);

/// Stable string token for a `BoxDecodeType` (e.g., `"yolov5"`); empty for `Unspecified`.
std::string box_decode_type_token_string(BoxDecodeType type);

/// Stable string token for a `BoxDecodeTypeOption`.
std::string box_decode_type_option_token_string(BoxDecodeTypeOption option);

} // namespace simaai::neat::pipeline_internal::sima
