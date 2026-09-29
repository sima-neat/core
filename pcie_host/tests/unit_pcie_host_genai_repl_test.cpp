#include "pcie_genai_repl.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace simaai::neat::pcie::genai::tools;

namespace {
void require(bool c, const std::string& m) {
  if (!c)
    throw std::runtime_error(m);
}
} // namespace

int main() {
  try {
    require(classify_repl_line("add image /a/b.jpg").kind == ReplKind::AddImage, "add image kind");
    require(classify_repl_line("add image /a/b.jpg").text == "/a/b.jpg", "add image path");
    require(classify_repl_line("add image   /a/b.jpg  ").text == "/a/b.jpg", "add image trims");
    require(classify_repl_line("clear image").kind == ReplKind::ClearImage, "clear image kind");
    require(classify_repl_line("what is on the image").kind == ReplKind::Prompt, "prompt kind");
    require(classify_repl_line("what is on the image").text == "what is on the image",
            "prompt text unchanged");
    // not a command: "add image" with no path is just a prompt
    require(classify_repl_line("add image").kind == ReplKind::Prompt,
            "bare 'add image' is a prompt");
    // The devkit chat commands.
    require(classify_repl_line("clear history").kind == ReplKind::ClearHistory, "clear history");
    require(classify_repl_line("print history").kind == ReplKind::PrintHistory, "print history");
    require(classify_repl_line("set system You are a pirate.").kind == ReplKind::SetSystem,
            "set system kind");
    require(classify_repl_line("set system You are a pirate.").text == "You are a pirate.",
            "set system text");
    require(classify_repl_line("set system").kind == ReplKind::Prompt,
            "bare 'set system' is a prompt (like the devkit)");
    require(classify_repl_line("clear system").kind == ReplKind::ClearSystem, "clear system");
    require(classify_repl_line("enable-thinking").kind == ReplKind::EnableThinking, "enable");
    require(classify_repl_line("disable-thinking").kind == ReplKind::DisableThinking, "disable");
    require(classify_repl_line("help").kind == ReplKind::Help, "help");
    require(classify_repl_line("help me").kind == ReplKind::Prompt, "'help me' is a prompt");

    const std::string help = repl_help();
    for (const char* line :
         {"add image <fn>", "set system <prompt>", "clear system", "clear history", "print history",
          "enable-thinking", "disable-thinking", "quit", "help"}) {
      require(help.find(line) != std::string::npos, std::string("help lists ") + line);
    }
    require(help.find("set audio") == std::string::npos, "help lists no audio commands");

    require(history_cleared_message(true, false) ==
                "User interrupt received. Cleared chat history.\nType quit to quit.",
            "cancel message (devkit text)");
    require(history_cleared_message(false, false) ==
                "No final answer generated. Cleared chat history.",
            "empty-answer message (devkit text)");
    require(history_cleared_message(false, true) == "Cleared chat history.", "error message");

    std::cout << "[PASS] pcie-genai repl classifier\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
