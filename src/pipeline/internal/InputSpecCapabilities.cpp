#include "pipeline/internal/InputSpecCapabilities.h"

#include "gst/internal/ElementCapability.h"
#include "nodes/groups/internal/VideoSenderRawIngress.h"
#include "nodes/sima/internal/SimaEncode.h"

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

bool node_has_input_spec_specialization(const Node& node) {
  if (dynamic_cast<const internal::InputSpecSpecializer*>(&node)) {
    return true;
  }
  const auto* encoder = dynamic_cast<const SimaEncode*>(&node);
  return encoder && internal::SimaEncodeAccess::has_input_adapter(*encoder);
}

internal::SpecializedNodeSequence
specialize_pipeline_nodes_for_input(std::span<const std::shared_ptr<Node>> nodes,
                                    const OutputSpec& input,
                                    const internal::InputSpecSpecializationContext& context) {
  internal::SpecializedNodeSequence result;
  result.nodes.reserve(nodes.size());
  result.output_spec = input;
  for (const auto& node : nodes) {
    auto selected = node;
    if (const auto* encoder = dynamic_cast<const SimaEncode*>(node.get());
        encoder && internal::SimaEncodeAccess::has_input_adapter(*encoder)) {
      selected =
          internal::SimaEncodeAccess::specialize_for_input(*encoder, result.output_spec, context);
    }
    auto step =
        internal::specialize_nodes_for_input(std::span(&selected, 1), result.output_spec, context);
    result.nodes.push_back(std::move(step.nodes.front()));
    result.output_spec = std::move(step.output_spec);
  }
  return result;
}

} // namespace simaai::neat::pipeline_internal
