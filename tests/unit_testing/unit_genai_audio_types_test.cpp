#include "genai/GenAIInternal.h"
#include "pipeline/TensorAudio.h"
#include "test_main.h"

#include <stdexcept>
#include <string>
#include <filesystem>
#include <vector>

namespace {

simaai::neat::Tensor make_audio_tensor(std::vector<float> data, int sample_rate = 16000,
                                       int channels = 1) {
  simaai::neat::Tensor tensor = simaai::neat::Tensor::from_vector(
      data, {static_cast<int64_t>(data.size())}, simaai::neat::TensorMemory::CPU);
  tensor.semantic.audio = simaai::neat::AudioSpec{
      .sample_rate = sample_rate,
      .channels = channels,
      .interleaved = true,
  };
  return tensor;
}

void require_throws_contains(auto&& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) == std::string::npos) {
      throw std::runtime_error("unexpected exception text: " + std::string(e.what()));
    }
    return;
  }
  throw std::runtime_error("expected exception containing: " + needle);
}

} // namespace

// Verifies strict GenAI ASR audio tensor validation before tensors are passed to
// llima WhisperModel::run_model_from_pcm(...).
RUN_TEST("unit_genai_audio_types_test", ([] {
           using simaai::neat::AudioSpec;
           using simaai::neat::Tensor;
           using simaai::neat::TensorDType;
           using simaai::neat::TensorLayout;
           using simaai::neat::TensorMemory;
           using simaai::neat::tensor_to_pcm_audio;
           using simaai::neat::genai::GenerationRequest;

           const Tensor valid = make_audio_tensor({0.0F, 0.25F, -0.25F});
           const auto pcm = tensor_to_pcm_audio(valid);
           require(pcm.sample_rate == 16000U, "sample rate mismatch");
           require(pcm.samples.size() == 3U, "sample count mismatch");
           require(pcm.samples[1] == 0.25F, "sample payload mismatch");

           Tensor missing_semantic =
               Tensor::from_vector(std::vector<float>{0.0F}, {1}, TensorMemory::CPU);
           require_throws_contains([&] { (void)tensor_to_pcm_audio(missing_semantic); },
                                   "missing audio semantics");

           Tensor wrong_rate = make_audio_tensor({0.0F}, 8000, 1);
           require_throws_contains([&] { (void)tensor_to_pcm_audio(wrong_rate); }, "16 kHz");

           Tensor wrong_channels = make_audio_tensor({0.0F, 0.0F}, 16000, 2);
           require_throws_contains([&] { (void)tensor_to_pcm_audio(wrong_channels); }, "mono");

           Tensor wrong_dtype =
               Tensor::from_vector(std::vector<uint8_t>{0}, {1}, TensorMemory::CPU);
           wrong_dtype.semantic.audio = AudioSpec{.sample_rate = 16000, .channels = 1};
           require_throws_contains([&] { (void)tensor_to_pcm_audio(wrong_dtype); }, "Float32");

           Tensor empty;
           empty.dtype = TensorDType::Float32;
           empty.layout = TensorLayout::Unknown;
           empty.shape = {0};
           empty.strides_bytes = {sizeof(float)};
           empty.semantic.audio = AudioSpec{.sample_rate = 16000, .channels = 1};
           require_throws_contains([&] { (void)tensor_to_pcm_audio(empty); }, "shape");

           GenerationRequest no_audio;
           require_throws_contains(
               [&] { simaai::neat::genai::internal::validate_asr_generation_request(no_audio); },
               "requires audio");

           GenerationRequest mixed;
           mixed.audio = valid;
           mixed.audio_file = std::filesystem::path{"audio.wav"};
           require_throws_contains(
               [&] { simaai::neat::genai::internal::validate_asr_generation_request(mixed); },
               "exactly one");

           GenerationRequest text_field;
           text_field.audio = valid;
           text_field.prompt = std::string{"hello"};
           require_throws_contains(
               [&] { simaai::neat::genai::internal::validate_asr_generation_request(text_field); },
               "text fields");
         }));
