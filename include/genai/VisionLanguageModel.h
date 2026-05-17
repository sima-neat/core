/**
 * @file
 * @brief Public NEAT handle for LLiMa text and vision-language models.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::genai {

class VisionLanguageModel {
public:
  explicit VisionLanguageModel(std::filesystem::path model_dir);
  ~VisionLanguageModel();

  VisionLanguageModel(VisionLanguageModel&&) noexcept;
  VisionLanguageModel& operator=(VisionLanguageModel&&) noexcept;

  VisionLanguageModel(const VisionLanguageModel&) = delete;
  VisionLanguageModel& operator=(const VisionLanguageModel&) = delete;

  bool accepts_image() const;
  std::string model_id() const;
  std::string describe() const;
  std::size_t cached_image_count() const;
  bool encode(const Tensor& image);
  bool encode(const std::vector<Tensor>& images);
#if defined(SIMA_WITH_OPENCV)
  bool encode(const cv::Mat& image);
  bool encode(const std::vector<cv::Mat>& images);
#endif
  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class GenerationStream;
};

} // namespace simaai::neat::genai
