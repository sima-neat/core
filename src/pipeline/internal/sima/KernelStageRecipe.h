#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/runtime/gstreamer/prepared_tensor_stage.h>
#include <simaai/runtime/gstreamer/metrics.h>
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

namespace simaai::neat::pipeline_internal::sima {
// Retained only by the in-process graph. The manifest's stable transport ABI
// carries placement and metadata; checked numerical recipes never serialize.
struct KernelStageRecipe {
  ::simaai::runtime::Hardware target = ::simaai::runtime::Hardware::A65;
  std::vector<::simaai::runtime::gstreamer::TensorStageMember> members;
  using Resolver = decltype(::simaai::runtime::gstreamer::TensorStageMember::resolve);
  struct ResolverFactory {
    std::size_t member = 0;
    std::function<Resolver()> instantiate;
  };
  // Factories capture immutable checked policy only, never an adapter/session,
  // admitted operation or current producer loan. Member descriptors have no
  // live resolver. Each cold request instantiates fresh state exactly once.
  std::vector<ResolverFactory> resolver_factories;
  std::function<::simaai::runtime::Result<
      std::shared_ptr<::simaai::runtime::gstreamer::TensorStageFrameAdapter>>()>
      frame_adapter_factory;
  std::vector<std::byte> constant_data;
  std::optional<::simaai::runtime::gstreamer::TensorStagePublication> publication;
  // Internal tensor regions share the stage arena but are not graph outputs.
  // Cold preparation makes these available to dependent members, then removes
  // their metadata routes from the published tensor set.
  std::vector<LogicalTensorStaticSpec> private_outputs;
  std::vector<PhysicalBufferStaticSpec> private_buffers;
  // Optional cold semantic boundaries. No numerical ABI or per-frame metadata is changed.
  std::shared_ptr<const ::simaai::runtime::gstreamer::StageMetrics> metrics;
};
// Install one fresh execution-state instance into an uninstantiated request.
// Copies of that request share this single admission; independent preparations
// must call this helper with a fresh request, never reset an active adapter.
[[nodiscard]] ::simaai::runtime::Status
instantiate_kernel_stage_state(const KernelStageRecipe&,
                               ::simaai::runtime::gstreamer::TensorStageRequest&);
} // namespace simaai::neat::pipeline_internal::sima
