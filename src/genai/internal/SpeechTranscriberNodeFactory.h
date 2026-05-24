/**
 * @file
 * @brief Graph node factory for GenAI speech-to-text transcription.
 */
#pragma once

#include "genai/GenAIOptions.h"

#include <memory>
#include <string>

namespace simaai::neat::graph {
class Node;
} // namespace simaai::neat::graph

namespace simaai::neat::genai {

class ASRModel;

namespace nodes {

std::shared_ptr<graph::Node> SpeechTranscriber(std::shared_ptr<ASRModel> model,
                                               SpeechTranscriberOptions options = {},
                                               std::string label = "speech_transcriber");

} // namespace nodes
} // namespace simaai::neat::genai
