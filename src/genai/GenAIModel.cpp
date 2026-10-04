#include "genai/GenAIModel.h"
#include "genai/ASRModel.h"
#include "genai/GenAIInternal.h"
#include "genai/VisionLanguageModel.h"

#include <stdexcept>
#include <utility>
#include <variant>

namespace simaai::neat::genai {

struct GenAIModel::Impl {
  explicit Impl(internal::ModelLoadContext context)
      : info(context.info), model(make_model(std::move(context))) {}

  using ModelVariant = std::variant<VisionLanguageModel, ASRModel>;

  static ModelVariant make_model(internal::ModelLoadContext context) {
    switch (context.info.task) {
    case GenAITask::VisionLanguage:
      return internal::ModelAccess::vision(std::move(context));
    case GenAITask::ASR:
      return internal::ModelAccess::asr(std::move(context));
    }
    throw std::runtime_error("Unsupported GenAI task");
  }

  internal::ModelDirectoryInfo info;
  ModelVariant model;
};

GenAIModel::GenAIModel(std::filesystem::path model_dir)
    : GenAIModel(internal::local_model_context(model_dir)) {}

GenAIModel::GenAIModel(internal::ModelLoadContext context)
    : impl_(std::make_unique<Impl>(std::move(context))) {}

GenAIModel internal::ModelAccess::create(ModelLoadContext context) {
  return GenAIModel(std::move(context));
}

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
  return std::visit([&](auto& model) { return model.run(request); }, impl_->model);
}

GenerationStream GenAIModel::stream(const GenerationRequest& request) {
  return std::visit([&](auto& model) { return model.stream(request); }, impl_->model);
}

} // namespace simaai::neat::genai
