#pragma once

#include <simaai/neat/pcie/genai/GenAIModel.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

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

// The API takes UTF-8 paths. On Windows, main() receives its arguments in the ANSI code page,
// which turns characters outside it into '?', so read the UTF-16 command line instead.
inline std::vector<std::string> utf8_arguments([[maybe_unused]] int argc,
                                               [[maybe_unused]] char** argv) {
#ifdef _WIN32
  int count = 0;
  const std::unique_ptr<LPWSTR, decltype(&LocalFree)> wide(
      CommandLineToArgvW(GetCommandLineW(), &count), &LocalFree);
  if (!wide)
    throw std::runtime_error("cannot read the command line");
  std::vector<std::string> arguments;
  for (int i = 0; i < count; ++i) {
    const auto utf8 = std::filesystem::path(wide.get()[i]).u8string();
    arguments.emplace_back(utf8.begin(), utf8.end());
  }
  return arguments;
#else
  return {argv, argv + argc};
#endif
}

// Without this, std::filesystem would read the UTF-8 bytes in the ANSI code page on Windows.
inline std::filesystem::path utf8_path(const std::string& path) {
  return std::u8string(path.begin(), path.end());
}

inline Args parse_args(int argc, char** argv, const std::string& task) {
  const auto arguments = utf8_arguments(argc, argv);
  Args args;
  args.connection.startup_timeout_ms = 900000;
  args.connection.request_timeout_ms = 300000;
  if (task == "vlm")
    args.prompt = "Describe this image briefly.";
  const std::string media_option = task == "vlm" ? "--image" : "--audio";
  for (std::size_t i = 1; i < arguments.size(); ++i) {
    const std::string& option = arguments[i];
    if (option == "--help" || option == "-h") {
      std::cout << "Usage: " << arguments[0] << " --model HOST_DIRECTORY";
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
    if (i + 1 >= arguments.size())
      throw std::invalid_argument("missing value for " + option);
    const std::string& value = arguments[++i];
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
