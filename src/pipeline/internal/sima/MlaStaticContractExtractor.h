#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

struct MlaStaticContract {
  std::string stage_id;
  std::string node_name;
  std::vector<TensorStaticSpec> inputs;
  std::vector<TensorStaticSpec> outputs;
  std::vector<QuantStaticSpec> output_quant;
};

std::optional<MlaStaticContract> extract_mla_static_contract(const nlohmann::json& config_root,
                                                             std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima
