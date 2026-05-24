#include "pipeline/TensorAudio.h"

#include <cstring>
#include <stdexcept>

namespace simaai::neat {

PcmAudio tensor_to_pcm_audio(const Tensor& tensor) {
  if (!tensor.semantic.audio.has_value()) {
    throw std::runtime_error("GenerationRequest::audio tensor is missing audio semantics");
  }
  const AudioSpec& spec = *tensor.semantic.audio;
  if (spec.sample_rate != 16000) {
    throw std::runtime_error("GenerationRequest::audio requires 16 kHz PCM");
  }
  if (spec.channels != 1) {
    throw std::runtime_error("GenerationRequest::audio requires mono PCM");
  }
  if (!spec.interleaved) {
    throw std::runtime_error("GenerationRequest::audio requires interleaved PCM");
  }
  if (tensor.dtype != TensorDType::Float32) {
    throw std::runtime_error("GenerationRequest::audio requires Float32 PCM");
  }
  if (!tensor.is_dense()) {
    throw std::runtime_error("GenerationRequest::audio requires a dense tensor");
  }
  if (tensor.shape.size() != 1U || tensor.shape.front() <= 0) {
    throw std::runtime_error("GenerationRequest::audio requires shape [num_samples]");
  }

  const Tensor cpu = tensor.to_cpu_if_needed();
  std::vector<uint8_t> bytes = cpu.copy_dense_bytes_tight();
  if (bytes.empty() || bytes.size() % sizeof(float) != 0U) {
    throw std::runtime_error("GenerationRequest::audio has invalid PCM byte size");
  }

  PcmAudio out;
  out.sample_rate = static_cast<std::uint32_t>(spec.sample_rate);
  out.samples.resize(bytes.size() / sizeof(float));
  std::memcpy(out.samples.data(), bytes.data(), bytes.size());
  return out;
}

} // namespace simaai::neat
