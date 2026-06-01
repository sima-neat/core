/**
 * @file
 * @brief Public Graph fragments for GenAI / LLiMa stages.
 */
#pragma once

#include "genai/GenAIOptions.h"
#include "pipeline/Graph.h"

#include <memory>
#include <string>

namespace simaai::neat::genai {

class ASRModel;
class VisionLanguageModel;

namespace graphs {

/// Build a public `Graph` fragment exposing a Vision/Language stage through named endpoints.
Graph VisionLanguage(std::shared_ptr<VisionLanguageModel> model, VisionLanguageOptions options = {},
                     std::string name = "vision_language");

/// Build a public `Graph` fragment exposing an ASR stage through named endpoints.
Graph SpeechTranscriber(std::shared_ptr<ASRModel> model, SpeechTranscriberOptions options = {},
                        std::string name = "speech_transcriber");

} // namespace graphs
} // namespace simaai::neat::genai
