/**
 * @file
 * @brief Graph node factory for GenAI text and vision-language generation.
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

struct VisionLanguageOptions {
  std::string system_prompt;
  std::uint32_t max_new_tokens = 0;
  float temperature = 1.0F;
  float top_p = 1.0F;
  bool streaming = true;
  bool encode_images_on_input = true;
};

std::shared_ptr<graph::Node> VisionLanguage(std::shared_ptr<VisionLanguageModel> model,
                                            VisionLanguageOptions options = {},
                                            std::string label = "vision_language");

} // namespace nodes
} // namespace simaai::neat::genai
