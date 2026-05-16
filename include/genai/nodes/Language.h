/**
 * @file
 * @brief Graph node factory for text-only GenAI language generation.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace simaai::neat::graph {
class Node;
} // namespace simaai::neat::graph

namespace simaai::neat::genai {

class VisionLanguageModel;

namespace nodes {

struct LanguageOptions {
  std::string system_prompt;
  std::uint32_t max_new_tokens = 0;
  float temperature = 1.0F;
  float top_p = 1.0F;
};

std::shared_ptr<graph::Node> Language(std::shared_ptr<VisionLanguageModel> model,
                                      LanguageOptions options = {}, std::string label = "language");

} // namespace nodes
} // namespace simaai::neat::genai
