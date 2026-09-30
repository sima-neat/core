#include "pcie_genai_cli_args.h"

#include <stdexcept>

namespace simaai::neat::pcie::genai::tools {

namespace {

int to_int(const std::string& flag, const std::string& value) {
  std::size_t used = 0;
  int number = 0;
  try {
    number = std::stoi(value, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != value.size()) {
    throw std::invalid_argument(flag + " needs an integer, got '" + value + "'");
  }
  return number;
}

} // namespace

CliArgs parse_cli_args(const std::vector<std::string>& args) {
  CliArgs a;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& flag = args[i];
    const auto value = [&]() -> const std::string& {
      if (i + 1 >= args.size()) {
        throw std::invalid_argument(flag + " needs a value");
      }
      return args[++i];
    };
    if (flag == "-h" || flag == "--help") {
      a.help = true;
    } else if (flag == "--model") {
      a.model = value();
    } else if (flag == "--prompt") {
      a.prompt = value();
    } else if (flag == "--image") {
      a.images.push_back(value());
    } else if (flag == "--system-prompt") {
      a.system_prompt = value();
    } else if (flag == "--max-new-tokens") {
      const int n = to_int(flag, value());
      if (n < 0) {
        throw std::invalid_argument("--max-new-tokens must not be negative");
      }
      a.max_new_tokens = static_cast<std::uint32_t>(n);
    } else if (flag == "--card-id") {
      a.card_id = to_int(flag, value());
    } else if (flag == "--card-host") {
      a.card_host = value();
    } else if (flag == "--user") {
      a.user = value();
    } else if (flag == "--queue") {
      a.queue = to_int(flag, value());
    } else if (flag == "--ready-timeout-s") {
      a.ready_timeout_s = to_int(flag, value());
    } else if (flag == "--card-program") {
      a.card_program = value();
    } else if (flag == "--list") {
      a.list = true;
    } else if (flag == "--conf") {
      a.conf_path = value();
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }
  if (a.help || a.list) {
    // --list only reads the host config; it needs no model and no card.
    return a;
  }
  if (a.model.empty()) {
    throw std::invalid_argument("--model is required");
  }
  if (a.queue < 0 || a.queue > 3) {
    throw std::invalid_argument("--queue must be 0..3");
  }
  if (a.card_id < 0) {
    throw std::invalid_argument("--card-id must not be negative");
  }
  return a;
}

std::string cli_usage() {
  return "usage: pcie-genai --model <name> [--prompt <text>] [--image <path>]...\n"
         "                  [--system-prompt <text>] [--max-new-tokens N]\n"
         "                  [--card-id N] [--card-host HOST] [--user USER]\n"
         "                  [--queue 0..3] [--ready-timeout-s N] [--card-program NAME]\n"
         "       pcie-genai --list [--conf <path>]\n"
         "\n"
         "Starts pcie-genai-backend on the card over SSH, waits until the model has\n"
         "loaded over PCIe (READY), then prints the answer as it is generated.\n"
         "<name> is the model folder under the host 'models' serve root.\n"
         "Without --prompt: one prompt per stdin line, or a '>>> ' prompt on a terminal.\n"
         "The model remembers the conversation (text and images) until 'clear history'\n"
         "or exit, like the LLiMa devkit CLI. Type 'help' for the chat commands\n"
         "('add image <path>', 'clear history', 'print history', 'set system <text>', ...).\n"
         "--image <path> (repeatable) attaches images to --prompt.\n"
         "Ctrl-C during an answer cancels it and clears the history; 'quit' or Ctrl-D exits.\n"
         "\n"
         "--list prints the models the host serves over PCIe (read from the\n"
         "simaai-mla-daemon config, default /etc/simaai/simaai-mla-daemon.conf) and\n"
         "the folder to copy new models into. It needs no card and opens no SSH.\n";
}

} // namespace simaai::neat::pcie::genai::tools
