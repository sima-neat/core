#include "genai/GenAIModel.h"
#include "genai/GenAIInternal.h"

#include <utility>

namespace simaai::neat::genai {

struct GenAIModel::Impl {
  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))) {}

  internal::ModelDirectoryInfo info;
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

std::string GenAIModel::describe() const {
  return "GenAIModel(" + impl_->info.root.string() + ")";
}

} // namespace simaai::neat::genai
