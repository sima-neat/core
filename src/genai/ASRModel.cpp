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
    whisper_model = std::make_unique<simaai::llima::WhisperModel>(info.root, true);
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_asr_generation_request(request);
    load();

    std::lock_guard<std::mutex> run_lock(run_mutex);
    const std::string language = request.language.empty() ? "en" : request.language;
    std::string text;
    if (request.audio_file.has_value()) {
      text = whisper_model->run_model(*request.audio_file, language);
    } else {
      const PcmAudio audio = tensor_to_pcm_audio(*request.audio);
      text = whisper_model->run_model_from_pcm(
          std::span<const float>{audio.samples.data(), audio.samples.size()}, audio.sample_rate,
          language);
    }

    GenerationResult result;
    result.text = std::move(text);
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

std::string ASRModel::describe() const {
  return "ASRModel(" + impl_->info.root.string() + ")";
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
              model->whisper_model->set_text_callback([](const std::string&, bool) {});
            }
          }
        } callback_guard{model};

        model->whisper_model->set_info_callback(
            [&producer](const std::string& metric, double value) {
              producer.record_metric(metric, value);
            });
        model->whisper_model->set_text_callback(
            [&producer](const std::string& text, bool stream_end) {
              producer.record_text(text, stream_end);
            });

        const std::string language = request.language.empty() ? "en" : request.language;
        if (request.audio_file.has_value()) {
          (void)model->whisper_model->run_model(*request.audio_file, language);
        } else {
          const PcmAudio audio = tensor_to_pcm_audio(*request.audio);
          (void)model->whisper_model->run_model_from_pcm(
              std::span<const float>{audio.samples.data(), audio.samples.size()},
              audio.sample_rate, language);
        }
        producer.finish(producer.cancelled() ? "interrupted" : "stop",
                        std::optional<std::uint32_t>(0));
      },
      [model = impl_] {
        if (model && model->whisper_model) {
          model->whisper_model->stop_model();
        }
      });
}

} // namespace simaai::neat::genai
