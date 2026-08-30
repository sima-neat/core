#include "genai/GenAIModel.h"
#include "genai/ASRModel.h"
#include "genai/GenAIInternal.h"
#include "genai/VisionLanguageModel.h"
#include "genai/TextToSpeechModel.h"

#include <stdexcept>
#include <utility>
#include <variant>
#include <type_traits>

namespace simaai::neat::genai {

struct GenAIModel::Impl {
  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))), model(make_model(info)) {}

  using ModelVariant = std::variant<VisionLanguageModel, ASRModel, TextToSpeechModel>;

  static ModelVariant make_model(const internal::ModelDirectoryInfo& info) {
    switch (info.task) {
    case GenAITask::VisionLanguage:
      return VisionLanguageModel(info.package_root);
    case GenAITask::ASR:
      return ASRModel(info.root);
    case GenAITask::TextToSpeech:
      return TextToSpeechModel(info.package_root);
    }
    throw std::runtime_error("Unsupported GenAI task");
  }

  internal::ModelDirectoryInfo info;
  ModelVariant model;
};

GenAIModel::GenAIModel(std::filesystem::path model_dir)
    : impl_(std::make_unique<Impl>(std::move(model_dir))) {}

GenAIModel::~GenAIModel() = default;

GenAIModel::GenAIModel(GenAIModel&&) noexcept = default;

GenAIModel& GenAIModel::operator=(GenAIModel&&) noexcept = default;

GenAITask GenAIModel::task() const {
  return impl_->info.task;
}

bool GenAIModel::accepts_text() const {
  return impl_->info.accepts_text;
}

bool GenAIModel::accepts_image() const {
  return impl_->info.accepts_image;
}

bool GenAIModel::accepts_audio() const {
  return impl_->info.accepts_audio;
}

bool GenAIModel::supports_thinking() const {
  const auto* model = std::get_if<VisionLanguageModel>(&impl_->model);
  return model && model->supports_thinking();
}

std::string GenAIModel::model_id() const {
  return internal::model_id_from_path(impl_->info.package_root);
}

void GenAIModel::set_lora(const std::string& adapter_name) {
  auto* model = std::get_if<VisionLanguageModel>(&impl_->model);
  if (!model) {
    throw std::invalid_argument("Dynamic LoRA is not supported for ASR models");
  }
  model->set_lora(adapter_name);
}

void GenAIModel::unset_lora() {
  auto* model = std::get_if<VisionLanguageModel>(&impl_->model);
  if (!model) {
    throw std::invalid_argument("Dynamic LoRA is not supported for ASR models");
  }
  model->unset_lora();
}

GenerationResult GenAIModel::run(const GenerationRequest& request) {
  return std::visit(
      [&](auto& model) -> GenerationResult {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, TextToSpeechModel>) {
          throw std::invalid_argument("Use TextToSpeechRequest with a TextToSpeech GenAIModel");
        } else {
          return model.run(request);
        }
      },
      impl_->model);
}

TextToSpeechResult GenAIModel::run(const TextToSpeechRequest& request) {
  return std::visit(
      [&](auto& model) -> TextToSpeechResult {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, TextToSpeechModel>) {
          return model.run(request);
        } else {
          throw std::invalid_argument("TextToSpeechRequest requires a TextToSpeech GenAIModel");
        }
      },
      impl_->model);
}

GenerationStream GenAIModel::stream(const GenerationRequest& request) {
  return std::visit(
      [&](auto& model) -> GenerationStream {
        using Model = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<Model, TextToSpeechModel>) {
          throw std::invalid_argument("Streaming is not supported for TextToSpeech models");
        } else {
          return model.stream(request);
        }
      },
      impl_->model);
}

} // namespace simaai::neat::genai
