#include <neat.h>

#include "test_main.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path qwen3_tts_model_path() {
  const char* value = std::getenv("SIMANEAT_QWEN3_TTS_MODEL");
  if (value == nullptr || *value == '\0') {
    throw SkipTest("set SIMANEAT_QWEN3_TTS_MODEL to run the Qwen3-TTS DevKit smoke test");
  }
  const std::filesystem::path path(value);
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("SIMANEAT_QWEN3_TTS_MODEL is not a model directory: " + path.string());
  }
  return path;
}

} // namespace

RUN_TEST("genai_qwen3_tts_smoke_test", ([] {
           using simaai::neat::genai::GenAIModel;
           using simaai::neat::genai::GenAITask;
           using simaai::neat::genai::TextToSpeechRequest;

           GenAIModel model(qwen3_tts_model_path());
           require(model.task() == GenAITask::TextToSpeech,
                   "model was not detected as TextToSpeech");
           require(model.accepts_text(), "TextToSpeech model should accept text input");

           TextToSpeechRequest request;
           request.prompt = "Good morning. This is a Core Qwen three text to speech smoke test.";
           request.max_frames = 128;

           const auto result = model.run(request);
           require(result.audio.sample_rate > 0, "Core returned audio with no sample rate");
           require(!result.audio.samples.empty(), "Core returned no audio samples");
           bool has_signal = false;
           for (const float sample : result.audio.samples) {
             if (std::fabs(sample) > 1.0e-5F) {
               has_signal = true;
               break;
             }
           }
           require(has_signal, "Core returned silent Qwen3-TTS audio");
         }));
