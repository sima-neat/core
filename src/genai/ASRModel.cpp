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
  explicit Impl(internal::ModelLoadContext context)
      : info(std::move(context.info)), files(std::move(context.files)) {
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
    whisper_model = std::make_unique<simaai::llima::WhisperModel>(info.root, files);
    whisper_model->set_performance_summary_enabled(false);
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_asr_generation_request(request);
    load();

    std::lock_guard<std::mutex> run_lock(run_mutex);
    auto transcription = transcribe(request, {.text = [](const std::string&, bool, bool) {}});

    GenerationResult result;
    result.text = std::move(transcription.text);
    result.language = std::move(transcription.language);
    result.no_speech_prob = transcription.no_speech_prob;
    result.avg_logprob = transcription.avg_logprob;
    result.finish_reason = "stop";
    return result;
  }

  // Called after load(), with run_mutex held by the request.
  simaai::llima::WhisperModel::TranscriptionResult
  transcribe(const GenerationRequest& request, simaai::llima::StreamCallbacks callbacks) {
    const simaai::llima::TranscriptionOptions options{
        .language = request.language.empty() ? "auto" : request.language,
        .task = asr_task_name(request.asr_task),
        .callbacks = std::move(callbacks)};
    if (request.audio_file.has_value())
      return whisper_model->run_model(*request.audio_file, options);
    const PcmAudio audio = tensor_to_pcm_audio(*request.audio);
    return whisper_model->run_model_from_pcm(
        std::span<const float>{audio.samples.data(), audio.samples.size()}, audio.sample_rate,
        options);
  }

  internal::ModelDirectoryInfo info;
  std::shared_ptr<simaai::llima::FileProvider> files;
  std::mutex load_mutex;
  std::mutex run_mutex;
  std::unique_ptr<simaai::llima::WhisperModel> whisper_model;
};

ASRModel::ASRModel(std::filesystem::path model_dir)
    : ASRModel(internal::local_model_context(model_dir)) {}

ASRModel internal::ModelAccess::asr(ModelLoadContext context) {
  return ASRModel(std::move(context));
}

ASRModel::ASRModel(internal::ModelLoadContext context)
    : impl_(std::make_shared<Impl>(std::move(context))) {}

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
        simaai::llima::StreamCallbacks callbacks;
        callbacks.info = [&producer](const std::string& metric, double value) {
          producer.record_metric(metric, value);
        };
        callbacks.text = [&producer](const std::string& text, bool stream_end, bool) {
          producer.record_text(text, stream_end);
        };
        auto transcription = model->transcribe(request, std::move(callbacks));
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
