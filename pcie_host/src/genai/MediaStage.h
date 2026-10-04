#pragma once
#include "simaai/neat/pcie/genai/GenAIModel.h"
namespace simaai::neat::pcie::genai::internal {
class MediaStage {
public:
  MediaStage(const ConnectionOptions& options, const std::string& session, uint64_t request);
  ~MediaStage();
  Json encode(const GenerationRequest& request);

private:
  void prepare();
  Json file(const std::filesystem::path& source);
  Json tensor(const Tensor& tensor, bool audio, uint32_t rate);
  Json descriptor(const std::filesystem::path& path);
  Json images(const std::vector<Tensor>& tensors, const std::vector<std::filesystem::path>& paths);
  std::filesystem::path root_, directory_;
  std::string serve_;
  bool prepared_ = false;
  std::size_t count_ = 0, bytes_ = 0;
};
} // namespace simaai::neat::pcie::genai::internal
