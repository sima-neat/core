#include "genai/GenAIModel.h"
#include "genai/ASRModel.h"
#include "genai/GenAIInternal.h"
#include "genai/VisionLanguageModel.h"

#include <stdexcept>
#include <utility>
#include <variant>

namespace simaai::neat::genai {

struct GenAIModel::Impl {
  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))), model(make_model(info)) {}

  using ModelVariant = std::variant<VisionLanguageModel, ASRModel>;

  static ModelVariant make_model(const internal::ModelDirectoryInfo& info) {
    switch (info.task) {
    case GenAITask::VisionLanguage:
      return VisionLanguageModel(info.root);
    case GenAITask::ASR:
      return ASRModel(info.root);
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

std::string GenAIModel::model_id() const {
  return internal::model_id_from_path(impl_->info.root);
}

GenerationResult GenAIModel::run(const GenerationRequest& request) {
  return std::visit([&](auto& model) { return model.run(request); }, impl_->model);
}

GenerationStream GenAIModel::stream(const GenerationRequest& request) {
  return std::visit([&](auto& model) { return model.stream(request); }, impl_->model);
}

} // namespace simaai::neat::genai
