/**
 * @file
 * @brief Classify an interactive REPL line (the LLiMa devkit chat commands),
 *        plus the devkit help page and clear messages. Pure and header-light
 *        so it can be unit-tested without readline or a tty.
 */
#pragma once

#include <string>
#include <string_view>

namespace simaai::neat::pcie::genai::tools {

enum class ReplKind {
  Prompt,
  AddImage,
  ClearImage,
  ClearHistory,
  PrintHistory,
  SetSystem,
  ClearSystem,
  EnableThinking,
  DisableThinking,
  Help,
};

struct ReplLine {
  ReplKind kind;
  std::string text; // AddImage: the trimmed path; SetSystem: the prompt; Prompt: the line; else ""
};

// Devkit rules: "add image <path>" / "set system <prompt>" need their argument,
// otherwise the line is a Prompt. Other commands must match exactly.
ReplLine classify_repl_line(std::string_view line);

// The devkit help page, limited to the commands this CLI has.
std::string repl_help();

// What to print after a question when the card cleared the conversation
// (devkit cli.cpp texts): cancelled = Ctrl-C, failed = the question errored,
// neither = the model gave no final answer.
std::string history_cleared_message(bool cancelled, bool failed);

} // namespace simaai::neat::pcie::genai::tools
