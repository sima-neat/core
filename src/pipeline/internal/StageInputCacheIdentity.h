#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"

#include <optional>
#include <string>

namespace simaai::neat::pipeline_internal {

// Only an already-qualified non-SuperPoint BoxDecode resolver may specialize original
// image size at submission. FramePassthrough is only for immutable model fragments
// whose child contracts do not consume ingress preprocess geometry: those values
// remain on each frame, not in preparation identity. Exact remains the default.
enum class StageInputCacheGeometryPolicy { Exact, DynamicOriginalSize, FramePassthrough };

// Conservative source-layout identity, not canonical runtime physical authority. The cold
// seeded compiler remains authoritative. Values contain no pointers or retained input loans.
struct StageInputCacheIdentity {
  std::string value;
  bool operator==(const StageInputCacheIdentity&) const = default;
};

// Metadata-only: never maps storage, materializes an envelope or prepares a numerical kernel.
// An unsupported representation returns nullopt (caller must bypass reuse and compile seeded).
// Malformed inspectable metadata throws, rather than admitting an incomplete cache identity.
std::optional<StageInputCacheIdentity> stage_input_cache_identity(
    const Sample& sample, std::string* ineligible_reason = nullptr,
    StageInputCacheGeometryPolicy geometry_policy = StageInputCacheGeometryPolicy::Exact);

} // namespace simaai::neat::pipeline_internal
