/**
 * @file
 * @brief Command-line options of the pcie-genai host tool.
 *
 * Kept apart from pcie_genai_cli.cpp so the parsing can be unit-tested with
 * no card and no PCIe library. Defaults: queue 3 (GenAI's own slot) and a
 * 900 s READY timeout, because the model load over PCIe can take minutes.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::genai::tools {

struct CliArgs {
  std::string model;                 ///< subfolder under the host "models" serve root (required)
  std::optional<std::string> prompt; ///< one-shot prompt; otherwise stdin / >>> REPL
  std::vector<std::string> images;   ///< one-shot: attach these images to --prompt (VLM)
  std::optional<std::string> system_prompt;
  std::uint32_t max_new_tokens = 0; ///< 0 = model default
  int card_id = 0;
  std::string card_host; ///< empty = 10.0.<card_id>.2
  std::string user = "root";
  int queue = 3;             ///< GenAI's own lifecycle slot (tensor pipeline uses 0)
  int ready_timeout_s = 900; ///< model load over PCIe can take minutes
  std::string card_program = "pcie-genai-backend"; ///< the card runs /usr/bin/<card_program>
  bool list = false;      ///< list the models the host serves, then exit (no card, no SSH)
  std::string conf_path = ///< simaai-mla-daemon config read by --list and the model check
      "/etc/simaai/simaai-mla-daemon.conf";
  bool help = false;
};

/// Parse argv (without the program name). Throws std::invalid_argument.
CliArgs parse_cli_args(const std::vector<std::string>& args);

std::string cli_usage();

} // namespace simaai::neat::pcie::genai::tools
