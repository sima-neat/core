#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <simaai/scheduler/client/kernel.h>

#include <optional>
#include <stdexcept>

namespace simaai::neat::pipeline_internal {

// Cold graph composition: logical NHWC/CHW labels do not describe physical
// storage. Forward the checked producer specification, selected by backend id.
inline std::optional<::simaai::scheduler::client::TensorSpec>
compiled_kernel_output_tensor(const CompiledNodeContract* producer, int backend_index) {
  namespace rt = ::simaai::scheduler::client;
  if (!producer || !producer->kernel || !producer->kernel->recipe || backend_index < 0)
    return std::nullopt;
  for (const auto& member : producer->kernel->recipe->members) {
    for (std::size_t i = 0; i < member.outputs.size(); ++i) {
      if (member.outputs[i].source != rt::TensorStagePortSource::Output ||
          member.outputs[i].tensor != static_cast<std::size_t>(backend_index))
        continue;
      auto inspected = rt::inspect_kernel(member.executable);
      if (const auto* error = std::get_if<rt::Error>(&inspected))
        throw std::invalid_argument("producer kernel inspection: " + error->message);
      const auto& checked = std::get<rt::KernelInspection>(inspected);
      if (i >= checked.outputs().size())
        throw std::invalid_argument("producer kernel has no checked output specification");
      return checked.outputs()[i];
    }
  }
  return std::nullopt;
}

} // namespace simaai::neat::pipeline_internal
