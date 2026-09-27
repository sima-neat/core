#pragma once

#include "nodes/sima/SimaEncode.h"

#include <utility>

namespace simaai::neat::internal {

void validate_encode_options(const SimaEncodeOptions& options);
std::string encoder_fragment(const SimaEncodeOptions& options, int node_index);

// VideoSender already owns the typed, serializable raw input adapter.
struct SimaEncodeAccess {
  static std::shared_ptr<Node> prepared_input(SimaEncodeOptions options) {
    return std::shared_ptr<Node>(new SimaEncode(std::move(options), false));
  }
};

} // namespace simaai::neat::internal
