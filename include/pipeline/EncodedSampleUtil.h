#pragma once

#include "pipeline/SessionOptions.h"
#include "pipeline/TensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

simaai::neat::EncodedSpec::Codec caps_to_codec(const std::string& caps_string);

Sample make_encoded_sample(std::vector<uint8_t> bytes, std::string caps_string, int64_t pts_ns = -1,
                           int64_t dts_ns = -1, int64_t duration_ns = -1);

} // namespace simaai::neat
