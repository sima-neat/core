#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/PluginCompiledContracts.h"

#include <string>
#include <vector>

#include "gst/SimaPluginStaticManifestAbi.h"
#include <gstsimaaitensorbuffer.h>

namespace simaai::neat::pipeline_internal::packedio {

bool validate_packed_contract(const CompiledRuntimeContract& contract, std::string* err);

bool normalize_shared_parent_input_views(CompiledRuntimeContract* contract, std::string* err);

std::vector<simaai::gst::TensorBufferSelector>
selectors_for_logical_inputs(const CompiledRuntimeContract& contract);

bool prepare_physical_inputs(const CompiledRuntimeContract& contract,
                             const simaai::gst::TensorBufferView& upstream_view,
                             std::vector<std::uint8_t>* out_bytes,
                             std::string* err);

bool publish_logical_outputs(const CompiledRuntimeContract& contract,
                             const std::string& stage_key,
                             simaai::gst::TensorBufferView* out,
                             std::string* err);

} // namespace simaai::neat::pipeline_internal::packedio
