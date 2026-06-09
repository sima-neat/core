#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/runtime/ExecutionGraphPlan.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace simaai::neat::runtime {

nlohmann::ordered_json customer_view_to_json(const ExecutionGraphPlan& plan,
                                             const nlohmann::ordered_json& public_view,
                                             const nlohmann::ordered_json& lowered_view);

nlohmann::ordered_json
fallback_customer_view_from_lowered(const nlohmann::ordered_json& lowered_view,
                                    std::string topology_source, std::vector<std::string> warnings);

} // namespace simaai::neat::runtime
