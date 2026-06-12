/**
 * @file
 * @brief Public GenAI Graph-fragment option structs.
 */
#pragma once

#include <cstdint>
#include <string>

namespace simaai::neat::genai {

/// Options for `genai::graphs::VisionLanguage(...)`.
struct VisionLanguageOptions {
  std::string system_prompt;
  std::uint32_t max_new_tokens = 0;
  bool streaming = true;
  bool encode_images_on_input = true;
};

/// Options for `genai::graphs::SpeechTranscriber(...)`.
struct SpeechTranscriberOptions {
  std::string language = "en";
  bool streaming = true;
};

namespace nodes {
// Internal low-level runtime-node factories use the same option types. Keep aliases here so
// source-tree internals do not need duplicate structs. The factories themselves live under src/
// and are intentionally not part of the installed public API.
using VisionLanguageOptions = ::simaai::neat::genai::VisionLanguageOptions;
using SpeechTranscriberOptions = ::simaai::neat::genai::SpeechTranscriberOptions;
} // namespace nodes

} // namespace simaai::neat::genai
