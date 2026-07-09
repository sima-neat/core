#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai_test_utils.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

// Exercises direct ASRModel file and PCM transcription against a real LLiMa ASR
// model.
// Model fixture:
//   export LLIMA_MODELS_PATH=/media/nvme/llima/models
//   export SIMA_TEST_LLIMA_ASR_MODEL=whisper-small-a16w8
//   tests/tools/prepare_genai_models.sh
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";
constexpr const char* kExpectedTranscript = "tell me a joke please";
constexpr const char* kExpectedSilenceTranscript = "you";

fs::path resolve_model_dir() {
  return simaai::neat::test::resolve_genai_model_dir(kModelEnv,
                                                     simaai::neat::test::kDefaultAsrModelName,
                                                     "LLiMa ASR", "devkit/whisper_config.json");
}

fs::path audio_fixture(const char* repo_root_arg) {
  fs::path path = fs::path(repo_root_arg) / "tests/assets/genai/audio.wav";
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    throw std::runtime_error("missing ASR audio fixture: " + path.string());
  }
  return path;
}

fs::path pcm_fixture(const char* repo_root_arg) {
  fs::path path = fs::path(repo_root_arg) / "tests/assets/genai/audio_16k_mono_f32le.raw";
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    throw std::runtime_error("missing ASR PCM fixture: " + path.string());
  }
  return path;
}

std::vector<float> read_pcm_fixture(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to read ASR PCM fixture: " + path.string());
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.empty() || bytes.size() % sizeof(float) != 0U) {
    throw std::runtime_error("ASR PCM fixture has invalid size: " + path.string());
  }

  std::vector<float> pcm(bytes.size() / sizeof(float));
  std::memcpy(pcm.data(), bytes.data(), bytes.size());
  return pcm;
}

simaai::neat::Tensor make_pcm_tensor(const std::vector<float>& pcm) {
  simaai::neat::Tensor tensor = simaai::neat::Tensor::from_vector(
      pcm, {static_cast<int64_t>(pcm.size())}, simaai::neat::TensorMemory::CPU);
  tensor.semantic.audio = simaai::neat::AudioSpec{
      .sample_rate = 16000,
      .channels = 1,
      .interleaved = true,
  };
  return tensor;
}

std::vector<float> make_silence_pcm() {
  constexpr std::size_t kSampleRate = 16000;
  constexpr std::size_t kDurationSeconds = 2;
  return std::vector<float>(kSampleRate * kDurationSeconds, 0.0F);
}

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string normalize_transcript(const std::string& value) {
  std::string out;
  bool pending_space = false;
  for (unsigned char ch : value) {
    if (std::isalnum(ch)) {
      if (pending_space && !out.empty()) {
        out.push_back(' ');
      }
      out.push_back(static_cast<char>(std::tolower(ch)));
      pending_space = false;
    } else if (std::isspace(ch) || std::ispunct(ch)) {
      pending_space = true;
    }
  }
  return out;
}

std::string consume_stream(simaai::neat::genai::GenerationStream& stream,
                           const std::string& label) {
  std::string text;
  bool saw_final = false;
  while (auto token = stream.next()) {
    if (token->is_final) {
      saw_final = true;
      require(token->finish_reason == "stop", label + " finish_reason should be stop");
      break;
    }
    if (!token->text.empty()) {
      std::cout << label << "_CHUNK text=\n" << token->text << "\n";
      text += token->text;
    }
  }
  require(saw_final, label + " stream should emit final sample");
  return text;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_asr_run_test requires repository root argument");
    }
    const fs::path model_dir = resolve_model_dir();
    const fs::path audio_path = audio_fixture(argv[1]);
    const fs::path pcm_path = pcm_fixture(argv[1]);

    std::cout << "GENAI_ASR model_dir=" << model_dir << "\n";
    std::cout << "GENAI_ASR audio=" << audio_path << "\n";
    std::cout << "GENAI_ASR pcm=" << pcm_path << "\n";

    simaai::neat::genai::ASRModel model(model_dir);
    require(model.accepts_audio(), "ASR model should accept audio");

    simaai::neat::genai::GenerationRequest file_request;
    file_request.audio_file = audio_path;
    const auto file_result = model.run(file_request);
    std::cout << "GENAI_ASR_FILE text=\n" << file_result.text << "\n";
    require(normalize_transcript(file_result.text) == kExpectedTranscript,
            "ASR file transcript mismatch: " + trim_text(file_result.text));
    require(file_result.finish_reason == "stop", "ASR file finish_reason should be stop");

    simaai::neat::genai::GenerationRequest pcm_request;
    pcm_request.audio = make_pcm_tensor(read_pcm_fixture(pcm_path));
    const auto pcm_result = model.run(pcm_request);
    std::cout << "GENAI_ASR_PCM text=\n" << pcm_result.text << "\n";
    require(normalize_transcript(pcm_result.text) == kExpectedTranscript,
            "ASR PCM transcript mismatch: " + trim_text(pcm_result.text));
    require(pcm_result.finish_reason == "stop", "ASR PCM finish_reason should be stop");

    simaai::neat::genai::GenerationRequest silence_request;
    silence_request.audio = make_pcm_tensor(make_silence_pcm());
    const auto silence_result = model.run(silence_request);
    std::cout << "GENAI_ASR_SILENCE text=\n" << silence_result.text << "\n";
    require(normalize_transcript(silence_result.text) == kExpectedSilenceTranscript,
            "ASR silence transcript mismatch: " + trim_text(silence_result.text));
    require(silence_result.finish_reason == "stop", "ASR silence finish_reason should be stop");

    auto file_stream = model.stream(file_request);
    const std::string file_stream_text = consume_stream(file_stream, "GENAI_ASR_FILE_STREAM");
    std::cout << "GENAI_ASR_FILE_STREAM text=\n" << file_stream_text << "\n";
    require(normalize_transcript(file_stream_text) == kExpectedTranscript,
            "ASR file stream transcript mismatch: " + trim_text(file_stream_text));

    auto silence_stream = model.stream(silence_request);
    const std::string silence_stream_text =
        consume_stream(silence_stream, "GENAI_ASR_SILENCE_STREAM");
    std::cout << "GENAI_ASR_SILENCE_STREAM text=\n" << silence_stream_text << "\n";
    require(normalize_transcript(silence_stream_text) == kExpectedSilenceTranscript,
            "ASR silence stream transcript mismatch: " + trim_text(silence_stream_text));

    simaai::neat::genai::GenAIModel generic_model(model_dir);
    require(generic_model.task() == simaai::neat::genai::GenAITask::ASR,
            "GenAIModel ASR task mismatch");
    require(!generic_model.accepts_text(), "GenAIModel ASR should not accept text");
    require(!generic_model.accepts_image(), "GenAIModel ASR should not accept images");
    require(generic_model.accepts_audio(), "GenAIModel ASR should accept audio");

    const auto generic_result = generic_model.run(file_request);
    std::cout << "GENAI_MODEL_ASR_FILE text=\n" << generic_result.text << "\n";
    require(normalize_transcript(generic_result.text) == kExpectedTranscript,
            "GenAIModel ASR file transcript mismatch: " + trim_text(generic_result.text));

    auto generic_stream = generic_model.stream(file_request);
    const std::string generic_stream_text =
        consume_stream(generic_stream, "GENAI_MODEL_ASR_FILE_STREAM");
    std::cout << "GENAI_MODEL_ASR_FILE_STREAM text=\n" << generic_stream_text << "\n";
    require(normalize_transcript(generic_stream_text) == kExpectedTranscript,
            "GenAIModel ASR stream transcript mismatch: " + trim_text(generic_stream_text));

    simaai::neat::genai::GenerationRequest bad_text_request;
    bad_text_request.prompt = std::string{"What is this audio?"};
    bool rejected_text = false;
    try {
      (void)generic_model.run(bad_text_request);
    } catch (const std::exception&) {
      rejected_text = true;
    }
    require(rejected_text, "GenAIModel ASR should reject text requests");

    simaai::neat::genai::GenerationRequest bad_image_request;
    bad_image_request.images = {simaai::neat::Tensor::from_text("not-an-image")};
    bool rejected_image = false;
    try {
      (void)generic_model.run(bad_image_request);
    } catch (const std::exception&) {
      rejected_image = true;
    }
    require(rejected_image, "GenAIModel ASR should reject image requests");

    std::cout << "[OK] genai_asr_run_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
