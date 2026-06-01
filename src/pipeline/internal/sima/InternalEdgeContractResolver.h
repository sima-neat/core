/**
 * @file
 * @ingroup internal_sima
 * @brief Internal producer->consumer edge contract resolver.
 *
 * Public output publication and internal consumer-edge contracts are intentionally different
 * questions:
 *   - public publication asks "what should the application see at a terminal endpoint?"
 *   - this resolver asks "what exact producer output feeds this consumer input?"
 *
 * The resolver is a behavior-preserving utility.  It centralizes pointer resolution across
 * `StageStaticSpec::input_bindings`, consumer logical inputs, producer logical outputs, and
 * producer/consumer physical buffers without changing pipeline rendering or runtime behavior.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::edgecontract {

struct ResolvedEdgeContract {
  std::size_t consumer_stage_index = 0U;
  std::size_t producer_stage_index = 0U;
  std::size_t binding_index = 0U;

  const StageStaticSpec* consumer_stage = nullptr;
  const StageStaticSpec* producer_stage = nullptr;
  const InputBindingStaticSpec* binding = nullptr;

  const LogicalInputStaticSpec* consumer_logical_input = nullptr;
  const LogicalTensorStaticSpec* producer_logical_output = nullptr;
  const PhysicalBufferStaticSpec* consumer_physical_input = nullptr;
  const PhysicalBufferStaticSpec* producer_physical_output = nullptr;

  // True when the consumer-side contract is not a direct logical mirror of the producer output.
  // This covers slice/offset/stride/repack views and other edge-local memory transformations.
  bool consumer_requires_view_contract = false;
};

std::optional<ResolvedEdgeContract>
resolve_edge_contract_for_binding(const SimaPluginStaticManifest& manifest,
                                  std::size_t consumer_stage_index, std::size_t binding_index,
                                  std::string* error_message = nullptr);

std::optional<ResolvedEdgeContract> resolve_edge_contract_for_binding(
    const SimaPluginStaticManifest& manifest, std::size_t consumer_stage_index,
    const InputBindingStaticSpec& binding, std::string* error_message = nullptr);

std::vector<ResolvedEdgeContract>
resolve_consumer_edge_contracts(const SimaPluginStaticManifest& manifest,
                                std::size_t consumer_stage_index,
                                std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::edgecontract
