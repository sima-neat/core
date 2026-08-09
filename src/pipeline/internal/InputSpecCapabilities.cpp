#include "pipeline/internal/InputSpecCapabilities.h"

#include "gst/internal/ElementCapability.h"
#include "nodes/groups/internal/VideoSenderRawIngress.h"

namespace simaai::neat::pipeline_internal {

simaai::neat::internal::InputSpecSpecializationContext
discover_input_spec_specialization_context() {
  simaai::neat::internal::InputSpecSpecializationContext context;
  const bool layout_aware =
      simaai::neat::internal::element_boolean_capability("neatencoder", "input-layout-aware")
          .value_or(false);
  context.set_capability(nodes::groups::internal::kNeatEncoderInputLayoutAwareCapability,
                         layout_aware);
  return context;
}

} // namespace simaai::neat::pipeline_internal
