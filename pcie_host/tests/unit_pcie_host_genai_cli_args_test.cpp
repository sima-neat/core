#include "pcie_genai_cli_args.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using simaai::neat::pcie::genai::tools::CliArgs;
using simaai::neat::pcie::genai::tools::parse_cli_args;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool refuses(const std::vector<std::string>& args) {
  try {
    parse_cli_args(args);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

} // namespace

int main() {
  try {
    {
      const CliArgs a = parse_cli_args({"--model", "Llama-3.2-3B-Instruct-a16w4"});
      require(a.model == "Llama-3.2-3B-Instruct-a16w4", "model");
      require(!a.prompt.has_value() && !a.system_prompt.has_value(), "no prompt by default");
      require(a.queue == 3, "GenAI defaults to its own queue 3");
      require(a.user == "root" && a.card_id == 0 && a.card_host.empty(), "connection defaults");
      require(a.ready_timeout_s == 900 && a.max_new_tokens == 0, "timeouts/tokens defaults");
      require(!a.help, "flag defaults");
      require(a.images.empty(), "no image by default");
    }
    {
      const CliArgs a = parse_cli_args(
          {"--model", "m", "--prompt", "hi", "--image", "/p/x.jpg", "--image", "/p/y.png"});
      require(a.images.size() == 2 && a.images[0] == "/p/x.jpg" && a.images[1] == "/p/y.png",
              "--image is repeatable, in order");
    }
    {
      const CliArgs a = parse_cli_args(
          {"--model", "m", "--prompt", "Why is the sky blue?", "--system-prompt", "Be brief.",
           "--max-new-tokens", "128", "--card-id", "1", "--card-host", "mla-hl83.sjc.sima.ai",
           "--user", "sima", "--queue", "2", "--ready-timeout-s", "60"});
      require(a.prompt == "Why is the sky blue?" && a.system_prompt == "Be brief.", "prompts");
      require(a.max_new_tokens == 128 && a.card_id == 1 && a.queue == 2, "numbers");
      require(a.card_host == "mla-hl83.sjc.sima.ai" && a.user == "sima", "connection");
      require(a.ready_timeout_s == 60, "timeout");
    }
    require(parse_cli_args({"--help"}).help, "--help needs no --model");
    {
      const CliArgs a = parse_cli_args({"--list"});
      require(a.list, "--list is set");
      require(a.conf_path == "/etc/simaai/simaai-mla-daemon.conf", "--conf default");
    }
    require(parse_cli_args({"--list", "--conf", "/x/y.conf"}).conf_path == "/x/y.conf",
            "--conf overrides the config path");
    require(refuses({}), "--model is required");
    require(refuses({"--model"}), "a flag without its value is refused");
    require(refuses({"--model", "m", "--bogus"}), "unknown flags are refused");
    require(refuses({"--model", "m", "--queue", "4"}), "queue must be 0..3");
    require(refuses({"--model", "m", "--queue", "two"}), "non-numbers are refused");
    require(refuses({"--model", "m", "--max-new-tokens", "-1"}),
            "negative token counts are refused");
    // A negative card id was clamped to card 0 for SSH but became 4294967295 for
    // the svc client, so the CLI started card 0 and then failed to connect.
    require(refuses({"--model", "m", "--card-id", "-1"}), "negative card ids are refused");
    require(parse_cli_args({"--model", "m", "--card-id", "0"}).card_id == 0, "card id 0 is fine");
    // --card-program picks the /usr/bin start script on the card (several can be installed).
    require(parse_cli_args({"--model", "m"}).card_program == "pcie-genai-backend",
            "default card program");
    require(
        parse_cli_args({"--model", "m", "--card-program", "pcie-genai-backend-p7"}).card_program ==
            "pcie-genai-backend-p7",
        "--card-program is parsed");
    // One CLI = one backend = one conversation, so these are gone.
    require(refuses({"--model", "m", "--attach"}), "--attach was removed");
    require(refuses({"--model", "m", "--keep-running"}), "--keep-running was removed");

    std::cout << "[PASS] pcie-genai CLI argument parsing\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
