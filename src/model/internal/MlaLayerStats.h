#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <span>
#include <vector>

namespace simaai::neat::internal {

// Measures each selected MLA stage's ELF one layer at a time with Scheduler's mla-rt client and
// returns each layer's hardware time. `selected_plugins` holds exact MPK plugin indices, as
// ModelPack::mla_profile_plugin_indices() returns them. Throws with the stage and reason when the
// model, the selection, or a measurement is unusable.
std::vector<MlaLayer> collect_mla_layer_stats(const pipeline_internal::sima::MpkContract& contract,
                                              std::span<const std::size_t> selected_plugins);

} // namespace simaai::neat::internal
