/**
 * @file
 * @brief Graph node factory for GenAI speech-to-text transcription.
 */
#pragma once

#include <memory>
#include <string>

namespace simaai::neat::graph {
class Node;
} // namespace simaai::neat::graph

namespace simaai::neat::genai {

class ASRModel;

namespace nodes {

struct SpeechTranscriberOptions {
  std::string language = "en";
  bool streaming = true;
};

std::shared_ptr<graph::Node> SpeechTranscriber(std::shared_ptr<ASRModel> model,
                                               SpeechTranscriberOptions options = {},
                                               std::string label = "speech_transcriber");

} // namespace nodes
} // namespace simaai::neat::genai
