#pragma once

#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/ModelStorageRequirements.h"

#include <algorithm>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sima_test {

// Test fixtures supply semantic values, storage and operation configuration
// explicitly. This adapter only invokes production lowering; it never interprets
// an MPK, infers geometry, selects a firmware ID or constructs a second oracle.
struct ProcessCvuPlanFixture {
  simaai::neat::pipeline_internal::sima::static_contract::ModelExecutionPlan semantic;
  simaai::neat::pipeline_internal::sima::static_contract::PhysicalExecutionPlan physical;
  simaai::neat::pipeline_internal::sima::static_contract::ModelStorageRequirements storage;
  simaai::neat::CompiledProcessCvuContract contract;

  [[nodiscard]] bool
  materializes(simaai::neat::pipeline_internal::sima::static_contract::ValueId value) const {
    const auto carrier = storage.value_carriers.at(value);
    return carrier && std::any_of(storage.carriers.begin(), storage.carriers.end(),
                                  [&](const auto& item) { return item.carrier_id == *carrier; });
  }
};

inline ProcessCvuPlanFixture compile_processcvu_fixture(
    simaai::neat::pipeline_internal::sima::static_contract::ModelExecutionPlan semantic,
    std::string_view operation) {
  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
  std::string error;
  auto physical = sc::PhysicalExecutionLowerer::lower(semantic, &error);
  if (!physical)
    throw std::invalid_argument("fixture physical plan: " + error);
  std::vector<sc::PhysicalCommandId> commands;
  for (const auto& command : physical->commands) {
    if (command.engine == sc::PhysicalEngine::Cvu && command.implementation_id == operation)
      commands.push_back(command.id);
  }
  if (commands.empty())
    throw std::invalid_argument("fixture has no physical operation '" + std::string(operation) +
                                "'");
  auto storage = sc::collect_model_storage_requirements(semantic, *physical, 4096, &error);
  if (!storage)
    throw std::invalid_argument("fixture carrier requirements: " + error);
  auto contract =
      sc::build_dmabuf_plan_processcvu_command_contract(semantic, *physical, commands, &error);
  if (!contract)
    throw std::invalid_argument("fixture command contract: " + error);
  return {std::move(semantic), std::move(*physical), std::move(*storage), std::move(*contract)};
}

inline ProcessCvuPlanFixture compile_processcvu_fixture(
    simaai::neat::pipeline_internal::sima::static_contract::ModelExecutionPlanData data,
    std::string_view operation) {
  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
  std::string error;
  auto semantic = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!semantic)
    throw std::invalid_argument("fixture semantic plan: " + error);
  return compile_processcvu_fixture(std::move(*semantic), operation);
}

} // namespace sima_test
