#include "genai/TextToSpeechModel.h"

#include "genai/GenAIInternal.h"

#include <algorithm>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace simaai::neat::genai {
namespace {

constexpr std::string_view kRuntimeExecutable = "qwen3tts";
constexpr std::string_view kRuntimeExecutableEnv = "SIMA_LMM_QWEN3TTS_EXECUTABLE";

template <typename T> T read_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  if (offset + sizeof(T) > bytes.size()) {
    throw std::runtime_error("Malformed Qwen3-TTS WAV: truncated chunk");
  }
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

PcmAudio read_pcm16_wav(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Qwen3-TTS did not produce WAV output: " + path.string());
  }
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(input), {}};
  if (bytes.size() < 12 ||
      std::string_view(reinterpret_cast<const char*>(bytes.data()), 4) != "RIFF" ||
      std::string_view(reinterpret_cast<const char*>(bytes.data() + 8), 4) != "WAVE") {
    throw std::runtime_error("Qwen3-TTS output is not a RIFF/WAVE file: " + path.string());
  }

  std::uint16_t format = 0;
  std::uint16_t channels = 0;
  std::uint16_t bits_per_sample = 0;
  std::uint32_t sample_rate = 0;
  std::optional<std::pair<std::size_t, std::size_t>> pcm_data;
  for (std::size_t offset = 12; offset + 8 <= bytes.size();) {
    const std::string_view id(reinterpret_cast<const char*>(bytes.data() + offset), 4);
    const std::uint32_t size = read_le<std::uint32_t>(bytes, offset + 4);
    const std::size_t payload = offset + 8;
    if (payload + size > bytes.size()) {
      throw std::runtime_error("Malformed Qwen3-TTS WAV: invalid chunk size");
    }
    if (id == "fmt ") {
      if (size < 16) {
        throw std::runtime_error("Malformed Qwen3-TTS WAV: short fmt chunk");
      }
      format = read_le<std::uint16_t>(bytes, payload);
      channels = read_le<std::uint16_t>(bytes, payload + 2);
      sample_rate = read_le<std::uint32_t>(bytes, payload + 4);
      bits_per_sample = read_le<std::uint16_t>(bytes, payload + 14);
    } else if (id == "data") {
      pcm_data = std::pair{payload, static_cast<std::size_t>(size)};
    }
    offset = payload + size + (size % 2);
  }
  if (format != 1 || channels != 1 || bits_per_sample != 16 || sample_rate == 0 || !pcm_data) {
    throw std::runtime_error("Qwen3-TTS WAV must be mono PCM16 audio");
  }
  if (pcm_data->second % sizeof(std::int16_t) != 0) {
    throw std::runtime_error("Malformed Qwen3-TTS WAV: incomplete PCM16 sample");
  }

  PcmAudio audio;
  audio.sample_rate = sample_rate;
  audio.samples.reserve(pcm_data->second / sizeof(std::int16_t));
  for (std::size_t offset = pcm_data->first; offset < pcm_data->first + pcm_data->second;
       offset += sizeof(std::int16_t)) {
    audio.samples.push_back(static_cast<float>(read_le<std::int16_t>(bytes, offset)) / 32768.0F);
  }
  return audio;
}

void run_raw_tts(const std::filesystem::path& package_root, const TextToSpeechRequest& request,
                 const std::filesystem::path& wav_path) {
  const char* configured_executable = std::getenv(kRuntimeExecutableEnv.data());
  const auto executable = configured_executable == nullptr || *configured_executable == '\0'
                              ? std::filesystem::path("/usr/bin") / std::string(kRuntimeExecutable)
                              : std::filesystem::path(configured_executable);

  const auto model_dir = package_root / "qwen3_model";
  const auto components_dir = package_root / "qwen3_components";
  if (!std::filesystem::is_regular_file(executable)) {
    throw std::runtime_error("Qwen3-TTS runner is not installed. Install sima-lmm-cli or set " +
                             std::string(kRuntimeExecutableEnv));
  }
  std::vector<std::string> args = {
      executable.string(),
      "--model-dir",
      model_dir.string(),
      "--components-dir",
      components_dir.string(),
      "--prompt",
      request.prompt,
      "--speaker",
      request.speaker,
      "--language",
      request.language,
      "--seed",
      std::to_string(request.seed),
      "--max-frames",
      std::to_string(request.max_frames),
      "--prefill-mode",
      "prefix_kv",
      "--out-wav",
      wav_path.string(),
      request.do_sample ? "--sample" : "--no-sample",
      request.subtalker_do_sample ? "--subtalker-sample" : "--subtalker-no-sample",
  };
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    throw std::system_error(errno, std::generic_category(), "fork Qwen3-TTS runtime");
  }
  if (child == 0) {
    execv(argv.front(), argv.data());
    _exit(127);
  }
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::system_error(errno, std::generic_category(), "wait for Qwen3-TTS runtime");
    }
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("Qwen3-TTS raw runtime failed with exit status " +
                             (WIFEXITED(status) ? std::to_string(WEXITSTATUS(status)) : "signal"));
  }
}

} // namespace

struct TextToSpeechModel::Impl {
  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))) {
    if (info.task != GenAITask::TextToSpeech) {
      throw std::runtime_error("GenAI model directory is not a TextToSpeech model: " +
                               info.root.string());
    }
  }

  TextToSpeechResult run(const TextToSpeechRequest& request) {
    internal::validate_text_to_speech_request(request);
    std::lock_guard<std::mutex> lock(run_mutex);

    const bool persistent_wav = request.output_wav.has_value();
    std::filesystem::path temp_dir;
    std::filesystem::path wav_path;
    if (persistent_wav) {
      wav_path = std::filesystem::absolute(*request.output_wav);
      std::filesystem::create_directories(wav_path.parent_path());
    } else {
      std::array<char, 64> template_path{};
      const std::string prefix = "/tmp/sima-neat-qwen3-XXXXXX";
      std::copy(prefix.begin(), prefix.end(), template_path.begin());
      char* created = mkdtemp(template_path.data());
      if (created == nullptr) {
        throw std::system_error(errno, std::generic_category(),
                                "create Qwen3-TTS temporary directory");
      }
      temp_dir = created;
      wav_path = temp_dir / "output.wav";
    }

    try {
      run_raw_tts(info.package_root, request, wav_path);
      TextToSpeechResult result;
      result.audio = read_pcm16_wav(wav_path);
      if (persistent_wav) {
        result.output_wav = wav_path;
      }
      if (!temp_dir.empty()) {
        std::filesystem::remove_all(temp_dir);
      }
      return result;
    } catch (...) {
      if (!temp_dir.empty()) {
        std::error_code ignored;
        std::filesystem::remove_all(temp_dir, ignored);
      }
      throw;
    }
  }

  internal::ModelDirectoryInfo info;
  std::mutex run_mutex;
};

TextToSpeechModel::TextToSpeechModel(std::filesystem::path model_dir)
    : impl_(std::make_shared<Impl>(std::move(model_dir))) {}

TextToSpeechModel::~TextToSpeechModel() = default;
TextToSpeechModel::TextToSpeechModel(TextToSpeechModel&&) noexcept = default;
TextToSpeechModel& TextToSpeechModel::operator=(TextToSpeechModel&&) noexcept = default;

bool TextToSpeechModel::accepts_text() const {
  return true;
}

std::string TextToSpeechModel::model_id() const {
  return internal::model_id_from_path(impl_->info.package_root);
}

TextToSpeechResult TextToSpeechModel::run(const TextToSpeechRequest& request) {
  return impl_->run(request);
}

} // namespace simaai::neat::genai
