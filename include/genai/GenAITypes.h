/**
 * @file
 * @brief Shared public vocabulary for NEAT GenAI APIs.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::genai {

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

struct ChatMessage {
  std::string role;
  std::string content;
};

struct GenerationMetrics {
  std::uint32_t generated_tokens = 0;
  double time_to_first_token_s = 0.0;
  double tokens_per_second = 0.0;
};

struct GenerationRequest {
  std::optional<std::string> prompt;
  std::optional<std::string> formatted_prompt;
  std::optional<std::string> system_prompt;
  std::vector<ChatMessage> messages;
  std::uint32_t max_new_tokens = 0;
  float temperature = 1.0F;
  float top_p = 1.0F;
};

struct GenerationResult {
  std::string text;
  GenerationMetrics metrics;
  std::string finish_reason;
};

struct TokenSample {
  std::string text;
  GenerationMetrics metrics;
  bool is_final = false;
  std::string finish_reason;
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

  explicit GenerationStream(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend class VisionLanguageModel;
};

} // namespace simaai::neat::genai
