#include "pcie_genai_repl.h"

namespace simaai::neat::pcie::genai::tools {

namespace {
std::string_view trim(std::string_view s) {
  const auto ws = " \t";
  const auto b = s.find_first_not_of(ws);
  if (b == std::string_view::npos)
    return {};
  const auto e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}
} // namespace

ReplLine classify_repl_line(std::string_view line) {
  constexpr std::string_view kAddImage = "add image ";
  constexpr std::string_view kSetSystem = "set system ";
  if (line == "clear image")
    return {ReplKind::ClearImage, ""};
  if (line == "clear history")
    return {ReplKind::ClearHistory, ""};
  if (line == "print history")
    return {ReplKind::PrintHistory, ""};
  if (line == "clear system")
    return {ReplKind::ClearSystem, ""};
  if (line == "enable-thinking")
    return {ReplKind::EnableThinking, ""};
  if (line == "disable-thinking")
    return {ReplKind::DisableThinking, ""};
  if (line == "help")
    return {ReplKind::Help, ""};
  if (line.rfind(kAddImage, 0) == 0) {
    const std::string_view path = trim(line.substr(kAddImage.size()));
    if (!path.empty()) {
      return {ReplKind::AddImage, std::string(path)};
    }
  }
  if (line.rfind(kSetSystem, 0) == 0 && line.size() > kSetSystem.size()) {
    return {ReplKind::SetSystem, std::string(line.substr(kSetSystem.size()))};
  }
  return {ReplKind::Prompt, std::string(line)};
}

std::string repl_help() {
  return "add image <fn>     : add an image.\n"
         "set system <prompt>: set system prompt.\n"
         "clear system       : clear system prompt, chat history and images.\n"
         "clear history      : clear prompts, responses and images; keep system prompt.\n"
         "print history      : print chat history.\n"
         "enable-thinking    : enable thinking mode and clear chat history.\n"
         "disable-thinking   : disable thinking mode and clear chat history.\n"
         "quit               : quit.\n"
         "help               : print this page.";
}

std::string history_cleared_message(bool cancelled, bool failed) {
  if (cancelled) {
    return "User interrupt received. Cleared chat history.\nType quit to quit.";
  }
  if (failed) {
    return "Cleared chat history.";
  }
  return "No final answer generated. Cleared chat history.";
}

} // namespace simaai::neat::pcie::genai::tools
