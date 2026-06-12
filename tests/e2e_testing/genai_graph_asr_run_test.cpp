#include "genai/ASRModel.h"
#include "genai/GraphFragments.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "test_utils.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

// Exercises graph ASR transcription from audio_path and PCM audio samples
// against a real LLiMa Whisper model when SIMA_TEST_LLIMA_ASR_MODEL is set.
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";
constexpr const char* kExpectedTranscript = "tell me a joke please";

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
      ("neat-genai-graph-asr-pcm-" + std::to_string(static_cast<long long>(::getpid())) + ".raw");
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

simaai::neat::Sample make_audio_path_input(const fs::path& audio_path, int64_t frame_id) {
  simaai::neat::Sample sample = simaai::neat::make_tensor_sample(
      "audio_path", simaai::neat::Tensor::from_text(audio_path.string()));
  sample.frame_id = frame_id;
  sample.stream_id = "genai-graph-asr";
  sample.pts_ns = frame_id * 1000;
  sample.duration_ns = 1000;
  return sample;
}

simaai::neat::Sample make_audio_input(const std::vector<float>& pcm, int64_t frame_id) {
  simaai::neat::Sample sample = simaai::neat::make_tensor_sample("audio", make_pcm_tensor(pcm));
  sample.frame_id = frame_id;
  sample.stream_id = "genai-graph-asr";
  sample.pts_ns = frame_id * 1000;
  sample.duration_ns = 1000;
  return sample;
}

simaai::neat::Sample make_invalid_audio_input() {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample("audio", simaai::neat::Tensor::from_text("not audio"));
  sample.frame_id = 99;
  sample.stream_id = "genai-graph-asr";
  return sample;
}

std::string sample_text(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    require(sample.tensor.has_value(), "Tensor sample missing tensor");
    return sample.tensor->to_text();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    require(sample.tensors.size() == 1U, "TensorSet sample should carry one tensor");
    return sample.tensors.front().to_text();
  }
  throw std::runtime_error("sample is not text");
}

struct GraphOutputs {
  std::string tokens;
  std::string error;
  bool saw_done = false;
  bool saw_error = false;
};

GraphOutputs pull_graph_outputs(simaai::neat::Run& run, bool stop_on_error = false) {
  GraphOutputs outputs;
  for (int i = 0; i < 64; ++i) {
    if (auto sample = run.pull("tokens", 500)) {
      outputs.tokens += sample_text(*sample);
      continue;
    }
    if (auto sample = run.pull("done", 10)) {
      outputs.saw_done = true;
      break;
    }
    if (auto sample = run.pull("error", 10)) {
      outputs.error = sample_text(*sample);
      outputs.saw_error = true;
      if (stop_on_error) {
        break;
      }
    }
  }
  return outputs;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_graph_asr_run_test requires repository root argument");
    }
    const fs::path model_dir = resolve_model_dir();
    const fs::path audio_path = audio_fixture(argv[1]);
    const std::vector<float> pcm = decode_fixture_to_pcm(audio_path);

    std::cout << "GENAI_GRAPH_ASR model_dir=" << model_dir << "\n";
    std::cout << "GENAI_GRAPH_ASR audio=" << audio_path << "\n";

    auto model = std::make_shared<simaai::neat::genai::ASRModel>(model_dir);
    require(model->accepts_audio(), "ASR model should accept audio");

    simaai::neat::Graph graph;
    graph.add(simaai::neat::genai::graphs::SpeechTranscriber(
        model, simaai::neat::genai::SpeechTranscriberOptions{.language = "en"},
        "speech_transcriber"));

    simaai::neat::Run run = graph.build();

    require(run.push("audio_path", make_audio_path_input(audio_path, 1)),
            "Run::push audio_path failed");
    const GraphOutputs file_outputs = pull_graph_outputs(run);
    require(file_outputs.saw_done, "ASR graph audio_path did not emit done");
    require(!file_outputs.saw_error, "ASR graph audio_path emitted error: " + file_outputs.error);
    std::cout << "GENAI_GRAPH_ASR_FILE text=\n" << file_outputs.tokens << "\n";
    require(normalize_transcript(file_outputs.tokens) == kExpectedTranscript,
            "ASR graph audio_path transcript mismatch: " + trim_text(file_outputs.tokens));

    require(run.push("audio", make_audio_input(pcm, 2)), "Run::push audio failed");
    const GraphOutputs pcm_outputs = pull_graph_outputs(run);
    require(pcm_outputs.saw_done, "ASR graph PCM did not emit done");
    require(!pcm_outputs.saw_error, "ASR graph PCM emitted error: " + pcm_outputs.error);
    std::cout << "GENAI_GRAPH_ASR_PCM text=\n" << pcm_outputs.tokens << "\n";
    require(normalize_transcript(pcm_outputs.tokens) == kExpectedTranscript,
            "ASR graph PCM transcript mismatch: " + trim_text(pcm_outputs.tokens));

    require(run.push("audio", make_invalid_audio_input()), "Run::push invalid audio failed");
    const GraphOutputs error_outputs = pull_graph_outputs(run, true);
    require(error_outputs.saw_error, "ASR graph invalid audio should emit error");
    require(!error_outputs.error.empty(), "ASR graph error text should be non-empty");

    run.stop();

    simaai::neat::Graph sync_graph;
    sync_graph.add(simaai::neat::genai::graphs::SpeechTranscriber(
        model,
        simaai::neat::genai::SpeechTranscriberOptions{
            .language = "en",
            .streaming = false,
        },
        "speech_transcriber_sync"));

    simaai::neat::Run sync_run = sync_graph.build();
    require(sync_run.push("audio_path", make_audio_path_input(audio_path, 3)),
            "Run::push sync audio_path failed");
    const GraphOutputs sync_outputs = pull_graph_outputs(sync_run);
    require(sync_outputs.saw_done, "ASR graph sync audio_path did not emit done");
    require(!sync_outputs.saw_error,
            "ASR graph sync audio_path emitted error: " + sync_outputs.error);
    std::cout << "GENAI_GRAPH_ASR_SYNC_FILE text=\n" << sync_outputs.tokens << "\n";
    require(normalize_transcript(sync_outputs.tokens) == kExpectedTranscript,
            "ASR graph sync audio_path transcript mismatch: " + trim_text(sync_outputs.tokens));
    sync_run.stop();

    std::cout << "[OK] genai_graph_asr_run_test passed\n";
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
