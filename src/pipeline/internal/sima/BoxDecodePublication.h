#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/host_task.h>

#include <cstddef>
#include <memory>
#include <span>

namespace simaai::neat::pipeline_internal::sima {

// Core's application wire formats, not numerical kernel output layouts.
enum class BoxDecodeWireFormat {
  Boxes,
  OrientedBoxesV1,
  Pose,
  Masks,
  MasksPose,
  RcnnProposals,
  FeaturePointsV1,
  LegacyFeaturePointsV0,
  // SegmentationMasksHeaderV1 prefix (header, records, padding). The fifth
  // input spec describes masks the producer writes in place after the prefix.
  MasksV1
};
// The serializer as a plain call: `plan` is BoxDecodePublication::plan, `inputs` the numerical
// kernel's outputs in its order, `output` the wire. What the host task's callback runs.
using BoxDecodeWireWriter = ::simaai::scheduler::client::HostStatus (*)(
    const void* plan, std::span<const ::simaai::scheduler::client::ConstBytes> inputs,
    ::simaai::scheduler::client::MutableBytes output);
struct BoxDecodePublication {
  ::simaai::scheduler::client::KernelExecutable executable;
  ::simaai::scheduler::client::TensorSpec output;
  std::size_t bytes = 0;
  // The same serializer for a backend that runs it outside a Scheduler program: no allocation,
  // no state beyond the immutable plan.
  std::shared_ptr<const void> plan;
  BoxDecodeWireWriter write = nullptr;
};

// Cold, checked Core-owned serializer, scheduled as an ordinary dependent A65
// member after the numerical kernel. It only formats existing results: no
// decoding, NMS, descriptor math or hidden source-task tensor copies.
// Throws invalid_argument for incompatible canonical tensor contracts.
BoxDecodePublication
prepare_boxdecode_publication(BoxDecodeWireFormat,
                              std::span<const ::simaai::scheduler::client::TensorSpec> inputs);

} // namespace simaai::neat::pipeline_internal::sima
