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

namespace internal {
struct ModelLoadContext;
struct ModelAccess;
} // namespace internal

/// A conversation with its own LLiMa generation cache. Requests supply their full message history.
class GenAISession {
public:
  ~GenAISession();
  GenAISession(GenAISession&&) noexcept;
  GenAISession& operator=(GenAISession&&) noexcept;
  GenAISession(const GenAISession&) = delete;
  GenAISession& operator=(const GenAISession&) = delete;

  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);
  void reset();

private:
  struct Impl;
  explicit GenAISession(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class VisionLanguageModel;
};

class VisionLanguageModel {
public:
  explicit VisionLanguageModel(std::filesystem::path model_dir);
  ~VisionLanguageModel();

  VisionLanguageModel(VisionLanguageModel&&) noexcept;
  VisionLanguageModel& operator=(VisionLanguageModel&&) noexcept;

  VisionLanguageModel(const VisionLanguageModel&) = delete;
  VisionLanguageModel& operator=(const VisionLanguageModel&) = delete;

  GenAISession create_session();
  bool accepts_image() const;
  std::string model_id() const;
  void set_lora(const std::string& adapter_name);
  void unset_lora();
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
  explicit VisionLanguageModel(internal::ModelLoadContext context);
  friend struct internal::ModelAccess;
  bool supports_thinking() const;

  struct Impl;
  std::shared_ptr<Impl> impl_;

  static GenerationStream stream_request(std::shared_ptr<Impl> model,
                                         const GenerationRequest& request,
                                         std::shared_ptr<GenAISession::Impl> session);
  friend class GenAISession;
  friend class GenAIModel;
  friend class GenerationStream;
};

} // namespace simaai::neat::genai
