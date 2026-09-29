/**
 * @file
 * @brief The card's answer to a chat command (reset / print history).
 */
#pragma once

#include <string>

namespace simaai::neat::pcie::genai {

struct ChatReply {
  bool ok = false;
  std::string text; ///< print: the history JSON; not ok: why
};

} // namespace simaai::neat::pcie::genai
