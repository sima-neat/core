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
// storage. Forward a unique whole-output writer's checked specification. A
// backend assembled from disjoint member writes (for example a serialized
// prefix plus a mask suffix) has no single numerical tensor descriptor; keep
// its published contract authoritative rather than selecting its first writer.
inline std::optional<::simaai::scheduler::client::TensorSpec>
compiled_kernel_output_tensor(const CompiledNodeContract* producer, int backend_index) {
  namespace rt = ::simaai::scheduler::client;
  if (!producer || !producer->kernel || !producer->kernel->recipe || backend_index < 0)
    return std::nullopt;
  const rt::TensorStageMember* writer = nullptr;
  std::size_t output_index = 0;
  for (const auto& member : producer->kernel->recipe->members) {
    for (std::size_t i = 0; i < member.outputs.size(); ++i) {
      const auto& port = member.outputs[i];
      if (port.source != rt::TensorStagePortSource::Output ||
          port.tensor != static_cast<std::size_t>(backend_index))
        continue;
      if (writer || port.byte_offset != 0)
        return std::nullopt;
      writer = &member;
      output_index = i;
    }
  }
  if (!writer)
    return std::nullopt;
  auto inspected = rt::inspect_kernel(writer->executable);
  if (const auto* error = std::get_if<rt::Error>(&inspected))
    throw std::invalid_argument("producer kernel inspection: " + error->message);
  const auto& checked = std::get<rt::KernelInspection>(inspected);
  if (output_index >= checked.outputs().size())
    throw std::invalid_argument("producer kernel has no checked output specification");
  return checked.outputs()[output_index];
}

} // namespace simaai::neat::pipeline_internal
