#pragma once

#include <simaai/neat/pcie/genai/GenAIModel.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tutorial {

namespace genai = simaai::neat::pcie::genai;

struct Args {
  std::string model;
  std::string media;
  std::string prompt = "Explain PCIe in one sentence.";
  std::string language = "auto";
  genai::ConnectionOptions connection;
  int max_tokens = 128;
  bool stream = false;
  bool translate = false;
};

inline Args parse_args(int argc, char** argv, const std::string& task) {
  Args args;
  args.connection.startup_timeout_ms = 900000;
  args.connection.request_timeout_ms = 300000;
  if (task == "vlm")
    args.prompt = "Describe this image briefly.";
  const std::string media_option = task == "vlm" ? "--image" : "--audio";
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      std::cout << "Usage: " << argv[0] << " --model HOST_DIRECTORY";
      if (task != "llm")
        std::cout << ' ' << media_option << " HOST_FILE";
      std::cout << " [--card 0] [--card-host IP] [--user sima] [--ssh-key FILE]";
      if (task == "whisper")
        std::cout << " [--language auto] [--translate]";
      else
        std::cout << " [--prompt TEXT] [--max-tokens 128]";
      if (task == "llm")
        std::cout << " [--stream]";
      std::cout << '\n';
      std::exit(0);
    }
    if (option == "--stream" && task == "llm") {
      args.stream = true;
      continue;
    }
    if (option == "--translate" && task == "whisper") {
      args.translate = true;
      continue;
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + option);
    const std::string value = argv[++i];
    if (option == "--model")
      args.model = value;
    else if (option == "--card")
      args.connection.card_id = std::stoi(value);
    else if (option == "--card-host")
      args.connection.card_host = value;
    else if (option == "--user")
      args.connection.user = value;
    else if (option == "--ssh-key")
      args.connection.ssh_key = value;
    else if (option == media_option && task != "llm")
      args.media = value;
    else if (option == "--language" && task == "whisper")
      args.language = value;
    else if (option == "--prompt" && task != "whisper")
      args.prompt = value;
    else if (option == "--max-tokens" && task != "whisper")
      args.max_tokens = std::stoi(value);
    else
      throw std::invalid_argument("unknown option: " + option);
  }
  if (args.model.empty() || (task != "llm" && args.media.empty()))
    throw std::invalid_argument(
        "--model and the task's media file (if any) are required; use --help");
  if (args.connection.card_id < 0 || args.max_tokens <= 0)
    throw std::invalid_argument("--card must be nonnegative and --max-tokens must be positive");
  return args;
}

} // namespace tutorial
