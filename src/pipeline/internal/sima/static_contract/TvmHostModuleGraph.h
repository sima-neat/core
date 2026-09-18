#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

// Safe structural evidence extracted from the embedded TVM GraphExecutor JSON.
// The object file is read as data; no model code is loaded or executed.
struct TvmHostModuleGraph {
  std::vector<std::string> input_names;
  std::vector<HostTensorTypeSpec> input_types;
  std::vector<HostTensorTypeSpec> output_types;
  // Per output, -1 means a real graph-executor materialization. Otherwise the
  // graph proves an exact __nop alias of the indexed input.
  std::vector<std::int32_t> output_alias_input;
};

std::optional<TvmHostModuleGraph> read_tvm_host_module_graph(const std::filesystem::path& module,
                                                             std::string* error = nullptr) noexcept;

} // namespace simaai::neat::pipeline_internal::sima::static_contract
