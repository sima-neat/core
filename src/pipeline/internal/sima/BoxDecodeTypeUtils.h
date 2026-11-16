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
#include "pipeline/SuperPointTypes.h"

#include <optional>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

/// Parse a token string (e.g., `"yolov5"`, `"ssd"`) into a `BoxDecodeType`; nullopt if unknown.
std::optional<BoxDecodeType> parse_box_decode_type_token(std::string_view token);

/// Parse a token string into a `BoxDecodeTypeOption`; nullopt if unknown.
std::optional<BoxDecodeTypeOption> parse_box_decode_type_option_token(std::string_view token);

/// Parse a stable SuperPoint numerical-profile token; nullopt if unknown.
std::optional<SuperPointProfile> parse_superpoint_profile_token(std::string_view token);

/// Parse a stable SuperPoint output-format token; nullopt if unknown.
std::optional<SuperPointOutputFormat> parse_superpoint_output_format_token(std::string_view token);

/// True when `type` is anything other than `BoxDecodeType::Unspecified`.
bool is_box_decode_type_specified(BoxDecodeType type);

/// True when each port owns its dtype/quantization rather than one route-wide dtype.
/// Numerical per-head compatibility remains a Kernels admission responsibility.
bool box_decode_type_uses_per_port_dtype(BoxDecodeType type);

/// True for the Segment Anything decode types (`MobileSam`, `EdgeTam`, `EfficientSam`,
/// `EfficientSam3`). Their masks have the original image size, so geometry is fixed per plan.
/// A prompt decoder uses the Kernels profile `sam_mask`; an EfficientSAM3 concept image model,
/// recognized by its detections and per-query logits, uses `sam_concept`.
bool box_decode_type_is_sam(BoxDecodeType type);

/// Model name for messages: "MobileSAM", "EdgeTAM", "EfficientSAM" or "EfficientSAM3".
const char* box_decode_type_sam_model_name(BoxDecodeType type);

/// The mask channel counts a Segment Anything decoder of this type exports: 1 (a single-mask
/// export), 4 (a single-mask token then three multimask tokens; not EfficientSAM) or 3 (three
/// multimask tokens; EfficientSAM and EfficientSAM3). The one rule for admission and the recipe.
bool box_decode_type_sam_exports_masks(BoxDecodeType type, int masks);

/// The counts box_decode_type_sam_exports_masks admits, for messages: "1 or 4", "1 or 3" or
/// "1, 3 or 4".
const char* box_decode_type_sam_mask_counts(BoxDecodeType type);

/// True when the option implies all class-score channels live in the same domain (e.g.,
/// post-sigmoid).
bool is_box_decode_type_option_requires_uniform_score_domain(BoxDecodeTypeOption option);

/// Stable string token for a `BoxDecodeType` (e.g., `"yolov5"`); empty for `Unspecified`.
std::string box_decode_type_token_string(BoxDecodeType type);

/// True when a user-requested decode type can safely consume a contract derived from MPK facts.
///
/// `Unspecified` means "use the contract as-is". Concrete requests are intentionally exact
/// matches: using a detection decoder on a segmentation/pose topology silently corrupts tensor
/// interpretation, so callers should fail early with a diagnostic instead of overriding the MPK.
bool box_decode_type_matches_requested_contract(BoxDecodeType contract_type,
                                                BoxDecodeType requested_type);

/// Stable string token for a `BoxDecodeTypeOption`.
std::string box_decode_type_option_token_string(BoxDecodeTypeOption option);

} // namespace simaai::neat::pipeline_internal::sima
