#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/stage.h>
#include <simaai/scheduler/client/metrics.h>
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

namespace simaai::neat::pipeline_internal::sima {
// Retained only by the in-process graph. The manifest's stable transport ABI
// carries placement and metadata; checked numerical recipes never serialize.
struct KernelStageRecipe {
  ::simaai::scheduler::client::Hardware target = ::simaai::scheduler::client::Hardware::A65;
  std::vector<::simaai::scheduler::client::TensorStageMember> members;
  using Resolver = ::simaai::scheduler::client::VariantResolver;
  struct ResolverFactory {
    std::size_t member = 0;
    std::function<Resolver()> instantiate;
  };
  // Factories capture immutable checked policy only, never an adapter/session,
  // admitted operation or current producer loan. Member descriptors have no
  // live resolver. Each cold request instantiates fresh state exactly once.
  std::vector<ResolverFactory> resolver_factories;
  std::function<::simaai::scheduler::client::Result<
      std::shared_ptr<::simaai::scheduler::client::TensorStageFrameAdapter>>()>
      frame_adapter_factory;
  std::vector<std::byte> constant_data;
  std::optional<::simaai::scheduler::client::TensorStagePublication> publication;
  // Producer-owned wire identity, applied after final arena layout is known.
  // Empty for ordinary numerical tensors; never inferred from their bytes.
  std::string output_format;
  std::shared_ptr<const ::simaai::scheduler::client::TensorStageImagePublication> image_publication;
  // Internal tensor regions share the stage arena but are not graph outputs.
  // Cold preparation makes these available to dependent members, then removes
  // their metadata routes from the published tensor set.
  std::vector<LogicalTensorStaticSpec> private_outputs;
  std::vector<PhysicalBufferStaticSpec> private_buffers;
  // Optional cold semantic boundaries. No numerical ABI or per-frame metadata is changed.
  std::shared_ptr<const ::simaai::scheduler::client::StageMetrics> metrics;
};
// Transfer immutable factories into a fresh request. Scheduler instantiates them
// only after validating the entire stopped cohort and resource budgets.
[[nodiscard]] ::simaai::scheduler::client::Status
wire_kernel_stage_factories(const KernelStageRecipe&,
                            ::simaai::scheduler::client::TensorStageRequest&);
} // namespace simaai::neat::pipeline_internal::sima
