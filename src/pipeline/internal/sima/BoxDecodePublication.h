#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/host_task.h>

namespace simaai::neat::pipeline_internal::sima {

// Core's application wire formats, not numerical kernel output layouts.
enum class BoxDecodeWireFormat {
  Boxes,
  Pose,
  Masks,
  MasksPose,
  RcnnProposals,
  FeaturePointsV1,
  LegacyFeaturePointsV0
};
struct BoxDecodePublication {
  ::simaai::scheduler::client::KernelExecutable executable;
  ::simaai::scheduler::client::TensorSpec output;
  std::size_t bytes = 0;
};

// Cold, checked Core-owned serializer, scheduled as an ordinary dependent A65
// member after the numerical kernel. It only formats existing results: no
// decoding, NMS, descriptor math or hidden source-task tensor copies.
// Throws invalid_argument for incompatible canonical tensor contracts.
BoxDecodePublication
prepare_boxdecode_publication(BoxDecodeWireFormat,
                              std::span<const ::simaai::scheduler::client::TensorSpec> inputs);

} // namespace simaai::neat::pipeline_internal::sima
