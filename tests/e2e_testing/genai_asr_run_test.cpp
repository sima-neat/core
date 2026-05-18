#include "genai/ASRModel.h"
#include "test_utils.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

// Exercises direct ASRModel file and PCM transcription against a real LLiMa
// Whisper model when SIMA_TEST_LLIMA_ASR_MODEL is set.
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";
constexpr const char* kExpectedTranscript = "tell me a joke please";
constexpr const char* kExpectedSilenceTranscript = "you";

std::string shell_quote(const fs::path& path) {
  std::string out = "'";
  for (char c : path.string()) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool command_exists(const char* command) {
  std::string cmd = "command -v ";
  cmd += command;
  cmd += " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

std::string trim_env_value(const char* value) {
  if (value == nullptr) {
    return {};
  }
  std::string out(value);
  const auto first = out.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

bool has_llima_whisper_config(const fs::path& model_dir) {
  std::error_code ec;
  return fs::is_regular_file(model_dir / "devkit" / "whisper_config.json", ec) && !ec;
}

fs::path resolve_model_dir() {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (env_model_dir.empty()) {
    skip_long_test_exception(std::string(kModelEnv) + " is not set");
  }
  fs::path model_dir(env_model_dir);
  if (!has_llima_whisper_config(model_dir)) {
    skip_long_test_exception(std::string(kModelEnv) +
                             " does not point to a LLiMa Whisper model directory");
  }
  return model_dir;
}

fs::path audio_fixture(const char* repo_root_arg) {
  fs::path path = fs::path(repo_root_arg) / "tests/assets/genai/audio.wav";
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    throw std::runtime_error("missing ASR audio fixture: " + path.string());
  }
  return path;
}

std::vector<float> decode_fixture_to_pcm(const fs::path& wav_path) {
  if (!command_exists("ffmpeg")) {
    skip_long_test_exception("missing ffmpeg command for ASR PCM fixture conversion");
  }
  const fs::path raw_path =
      fs::temp_directory_path() /
      ("neat-genai-asr-pcm-" + std::to_string(static_cast<long long>(::getpid())) + ".raw");
  std::ostringstream cmd;
  cmd << "ffmpeg -v error -y -i " << shell_quote(wav_path) << " -ac 1 -ar 16000 -f f32le "
      << shell_quote(raw_path);
  if (std::system(cmd.str().c_str()) != 0) {
    skip_long_test_exception("ffmpeg failed to convert ASR fixture to PCM");
  }

  std::ifstream in(raw_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to read converted ASR PCM fixture");
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::error_code ec;
  fs::remove(raw_path, ec);
  if (bytes.empty() || bytes.size() % sizeof(float) != 0U) {
    throw std::runtime_error("converted ASR PCM fixture has invalid size");
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

    std::cout << "GENAI_ASR model_dir=" << model_dir << "\n";
    std::cout << "GENAI_ASR audio=" << audio_path << "\n";

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
    pcm_request.audio = make_pcm_tensor(decode_fixture_to_pcm(audio_path));
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
