/**
 * @file
 * @brief Graph node factory for GenAI text and vision-language generation.
 */
#pragma once

#include "genai/GenAIOptions.h"

#include <memory>
#include <string>

namespace simaai::neat::graph {
class Node;
} // namespace simaai::neat::graph

namespace simaai::neat::genai {

class VisionLanguageModel;

namespace nodes {

std::shared_ptr<graph::Node> VisionLanguage(std::shared_ptr<VisionLanguageModel> model,
                                            VisionLanguageOptions options = {},
                                            std::string label = "vision_language");

} // namespace nodes
} // namespace simaai::neat::genai
