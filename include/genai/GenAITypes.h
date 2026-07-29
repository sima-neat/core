/**
 * @file
 * @brief Shared public vocabulary for NEAT GenAI APIs.
 */
#pragma once

#include "pipeline/Tensor.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::genai {

using Json = nlohmann::ordered_json;

class ASRModel;
class GenAIModel;
class GenAIServer;
class VisionLanguageModel;

/**
 * @brief High-level task family for a deployed LLiMa model directory.
 *
 * Text-only LLMs and image-capable VLMs both use VisionLanguage; distinguish
 * them with model capability queries such as VisionLanguageModel::accepts_image().
 */
enum class GenAITask {
  VisionLanguage,
  ASR,
};

/// Whisper decoding task for ASR requests.
enum class ASRTask {
  Transcribe,
  Translate,
};

/**
 * @brief Ordered image inputs for GenAI requests and chat messages.
 *
 * Tensor inputs are interpreted from their Tensor image metadata and VLM requests
 * require uint8 HWC RGB tensors. OpenCV cv::Mat overloads follow the existing
 * NEAT/OpenCV convention: 3-channel matrices are BGR and are converted to RGB
 * before they are stored in the request.
 */
class ImageList {
public:
  ImageList() = default;
  ImageList(std::initializer_list<Tensor> images);
  explicit ImageList(std::vector<Tensor> images);

  ImageList& operator=(std::initializer_list<Tensor> images);
  ImageList& operator=(std::vector<Tensor> images);

#if defined(SIMA_WITH_OPENCV)
  ImageList(std::initializer_list<cv::Mat> images);
  explicit ImageList(const std::vector<cv::Mat>& images);

  ImageList& operator=(std::initializer_list<cv::Mat> images);
  ImageList& operator=(const std::vector<cv::Mat>& images);
#endif

  bool empty() const;
  std::size_t size() const;
  const std::vector<Tensor>& tensors() const;
  std::vector<Tensor>& tensors();

private:
  std::vector<Tensor> images_;
};

struct ChatMessage {
  std::string role;
  std::string content;
  ImageList images;
  bool use_cached_images = false;
  Json tool_calls = Json::array();
  std::optional<std::string> tool_call_id;
  std::optional<std::string> name;
};

struct GenerationMetrics {
  std::uint32_t generated_tokens = 0;
  double time_to_first_token_s = 0.0;
  double tokens_per_second = 0.0;
};

struct GenerationRequest {
  std::optional<std::string> prompt;
  std::optional<std::string> system_prompt;
  std::vector<ChatMessage> messages;
  ImageList images;
  bool use_cached_images = false;
  std::optional<Tensor> audio;
  std::optional<std::filesystem::path> audio_file;
  /// ASR source language code/name, or `auto` to detect it.
  std::string language = "auto";
  /// Whisper decoding task. Ignored by non-ASR models.
  ASRTask asr_task = ASRTask::Transcribe;
  std::uint32_t max_new_tokens = 0;
  Json tools = Json::array();
  Json tool_choice = nullptr;
};

struct GenerationResult {
  std::string text;
  GenerationMetrics metrics;
  std::string finish_reason;
  /// Detected or explicitly selected ASR source language.
  std::string language;
  /// Probability that the ASR input contains no speech.
  std::optional<float> no_speech_prob;
  /// Mean log probability over generated ASR tokens.
  std::optional<float> avg_logprob;
  Json tool_calls = Json::array();
};

struct TokenSample {
  std::string text;
  GenerationMetrics metrics;
  bool is_final = false;
  std::string finish_reason;
  /// Detected or explicitly selected ASR source language on the final sample.
  std::string language;
  /// Probability that the ASR input contains no speech, set on the final sample.
  std::optional<float> no_speech_prob;
  /// Mean log probability over generated ASR tokens, set on the final sample.
  std::optional<float> avg_logprob;
  Json tool_calls = Json::array();
};

class GenerationStream {
public:
  class iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = TokenSample;
    using difference_type = std::ptrdiff_t;
    using pointer = const TokenSample*;
    using reference = const TokenSample&;

    iterator() = default;

    reference operator*() const;
    pointer operator->() const;
    iterator& operator++();
    void operator++(int);

    friend bool operator==(const iterator& lhs, const iterator& rhs);
    friend bool operator!=(const iterator& lhs, const iterator& rhs) {
      return !(lhs == rhs);
    }

  private:
    explicit iterator(GenerationStream* stream);
    void advance();

    GenerationStream* stream_ = nullptr;
    std::optional<TokenSample> current_;

    friend class GenerationStream;
  };

  ~GenerationStream();

  GenerationStream(GenerationStream&&) noexcept;
  GenerationStream& operator=(GenerationStream&&) noexcept;

  GenerationStream(const GenerationStream&) = delete;
  GenerationStream& operator=(const GenerationStream&) = delete;

  std::optional<TokenSample> next();
  void cancel();
  iterator begin();
  iterator end();

private:
  struct Impl;
  class Producer {
  public:
    void record_metric(const std::string& metric, double value);
    void record_text(const std::string& text, bool stream_end);
    void push(TokenSample sample);
    void finish(std::string finish_reason, std::optional<std::uint32_t> generated_tokens,
                std::string language = {}, std::optional<float> no_speech_prob = std::nullopt,
                std::optional<float> avg_logprob = std::nullopt);
    bool cancelled() const;
    GenerationMetrics current_metrics() const;

  private:
    explicit Producer(Impl& impl);

    Impl& impl_;

    friend struct Impl;
  };
  using ProducerFn = std::function<void(Producer&)>;
  using CancelFn = std::function<void()>;

  explicit GenerationStream(std::unique_ptr<Impl> impl);
  GenerationStream(ProducerFn producer, CancelFn cancel);

  std::unique_ptr<Impl> impl_;

  friend class ASRModel;
  friend class VisionLanguageModel;
};

} // namespace simaai::neat::genai
