#include "genai/ASRModel.h"
#include "genai/GenAIInternal.h"
#include "pipeline/TensorAudio.h"

#include <sima_lmm/whisper_model.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

namespace simaai::neat::genai {
namespace {

const char* asr_task_name(ASRTask task) {
  switch (task) {
  case ASRTask::Transcribe:
    return "transcribe";
  case ASRTask::Translate:
    return "translate";
  }
  throw std::invalid_argument("Unsupported ASR task");
}

} // namespace

struct ASRModel::Impl {
  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))) {
    if (info.task != GenAITask::ASR) {
      throw std::runtime_error("GenAI model directory is not an ASR model: " + info.root.string());
    }
  }

  void load() {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (whisper_model) {
      return;
    }

    internal::ensure_llima_runtime_connected();
    whisper_model = std::make_unique<simaai::llima::WhisperModel>(info.root);
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_asr_generation_request(request);
    load();

    std::lock_guard<std::mutex> run_lock(run_mutex);
    const std::string language = request.language.empty() ? "auto" : request.language;
    const char* task = asr_task_name(request.asr_task);
    simaai::llima::WhisperModel::TranscriptionResult transcription;
    if (request.audio_file.has_value()) {
      transcription = whisper_model->run_model(*request.audio_file, language, task);
    } else {
      const PcmAudio audio = tensor_to_pcm_audio(*request.audio);
      transcription = whisper_model->run_model_from_pcm(
          std::span<const float>{audio.samples.data(), audio.samples.size()}, audio.sample_rate,
          language, task);
    }

    GenerationResult result;
    result.text = std::move(transcription.text);
    result.language = std::move(transcription.language);
    result.no_speech_prob = transcription.no_speech_prob;
    result.avg_logprob = transcription.avg_logprob;
    result.finish_reason = "stop";
    return result;
  }

  internal::ModelDirectoryInfo info;
  std::mutex load_mutex;
  std::mutex run_mutex;
  std::unique_ptr<simaai::llima::WhisperModel> whisper_model;
};

ASRModel::ASRModel(std::filesystem::path model_dir)
    : impl_(std::make_shared<Impl>(std::move(model_dir))) {}

ASRModel::~ASRModel() = default;

ASRModel::ASRModel(ASRModel&&) noexcept = default;

ASRModel& ASRModel::operator=(ASRModel&&) noexcept = default;

bool ASRModel::accepts_audio() const {
  return true;
}

std::string ASRModel::model_id() const {
  return internal::model_id_from_path(impl_->info.root);
}

GenerationResult ASRModel::run(const GenerationRequest& request) {
  return impl_->run(request);
}

GenerationStream ASRModel::stream(const GenerationRequest& request) {
  internal::validate_asr_generation_request(request);
  impl_->load();
  return GenerationStream(
      [model = impl_, request](GenerationStream::Producer& producer) {
        std::lock_guard<std::mutex> run_lock(model->run_mutex);
        struct CallbackGuard {
          std::shared_ptr<Impl> model;
          ~CallbackGuard() {
            if (model && model->whisper_model) {
              model->whisper_model->set_info_callback([](const std::string&, double) {});
              model->whisper_model->set_text_callback([](const std::string&, bool, bool) {});
            }
          }
        } callback_guard{model};

        model->whisper_model->set_info_callback(
            [&producer](const std::string& metric, double value) {
              producer.record_metric(metric, value);
            });
        model->whisper_model->set_text_callback(
            [&producer](const std::string& text, bool stream_end, bool) {
              producer.record_text(text, stream_end);
            });

        const std::string language = request.language.empty() ? "auto" : request.language;
        const char* task = asr_task_name(request.asr_task);
        simaai::llima::WhisperModel::TranscriptionResult transcription;
        if (request.audio_file.has_value()) {
          transcription = model->whisper_model->run_model(*request.audio_file, language, task);
        } else {
          const PcmAudio audio = tensor_to_pcm_audio(*request.audio);
          transcription = model->whisper_model->run_model_from_pcm(
              std::span<const float>{audio.samples.data(), audio.samples.size()}, audio.sample_rate,
              language, task);
        }
        producer.finish(producer.cancelled() ? "interrupted" : "stop",
                        std::optional<std::uint32_t>(0), transcription.language,
                        transcription.no_speech_prob, transcription.avg_logprob);
      },
      [model = impl_] {
        if (model && model->whisper_model) {
          model->whisper_model->stop_model();
        }
      });
}

} // namespace simaai::neat::genai
