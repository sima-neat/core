#include "genai/ASRModel.h"
#include "genai/GenAIInternal.h"

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

  internal::ModelDirectoryInfo info;
};

ASRModel::ASRModel(std::filesystem::path model_dir)
    : impl_(std::make_unique<Impl>(std::move(model_dir))) {}

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
  (void)request;
  throw std::logic_error("ASRModel::run is not implemented yet");
}

} // namespace simaai::neat::genai
