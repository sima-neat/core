#pragma once

#include "pipeline/Tensor.h"

#include <cstdint>
#include <vector>

namespace simaai::neat {

struct PcmAudio {
  std::vector<float> samples;
  std::uint32_t sample_rate = 0;
};

PcmAudio tensor_to_pcm_audio(const Tensor& tensor);

} // namespace simaai::neat
