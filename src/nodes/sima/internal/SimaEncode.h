#pragma once

#include "nodes/sima/SimaEncode.h"

#include <utility>

namespace simaai::neat::internal {

class InputSpecSpecializationContext;

void validate_encode_options(const SimaEncodeOptions& options);
std::string encoder_fragment(const SimaEncodeOptions& options, int node_index);

// VideoSender already owns the typed, serializable raw input adapter.
struct SimaEncodeAccess {
  static bool has_input_adapter(const SimaEncode& node) {
    return node.input_adapter_ != nullptr;
  }
  static std::shared_ptr<Node> specialize_for_input(const SimaEncode& node, const OutputSpec& input,
                                                    const InputSpecSpecializationContext& context);

  static std::shared_ptr<Node> prepared_input(SimaEncodeOptions options) {
    return std::shared_ptr<Node>(new SimaEncode(std::move(options), false));
  }
};

} // namespace simaai::neat::internal
