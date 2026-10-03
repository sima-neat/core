// pcie-genai: the host front-end for PCIe GenAI.
// Starts pcie-genai-backend on the card over SSH, waits for READY, sends each
// prompt over simaai_svc and prints the tokens as they arrive.
// Chain: this CLI -> GenAIModel -> SvcTransport -> DlSvcClient ->
// simaai_mla_daemon -> PCIe -> pcie-genai-backend on the card.
// Rule: the CLI always starts the backend and stops it on every exit path
// (normal end, error, Ctrl-C during the load, SIGTERM/SIGHUP, closed stdout).

#include "pcie_genai_cancel_watcher.h"
#include "pcie_genai_cli_args.h"
#include "pcie_genai_models.h"
#include "pcie_genai_readline.h"
#include "pcie_genai_repl.h"

#include "RemoteRuntime.h"
#include "genai/DlSvcClient.h"
#include "genai/GenAIModelInternal.h"
#include "genai/SvcTransport.h"
#include "simaai/neat/pcie/genai/GenAIModel.h"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace pcie = simaai::neat::pcie;
namespace pgenai = simaai::neat::pcie::genai;
namespace pgi = simaai::neat::pcie::genai::internal;
using pgenai::tools::CliArgs;

namespace {

// Any of SIGINT/SIGTERM/SIGHUP stops the current answer and the READY wait.
// SIGINT (Ctrl-C) during a session cancels only the current answer; the
// session goes on. SIGTERM/SIGHUP mean "go away", so they also end it.
std::atomic<bool> g_interrupt{false};
// SIGTERM/SIGHUP (or a closed stdout) also end the session, so stop() runs.
std::atomic<bool> g_terminate{false};

void on_signal(int sig) {
  if (sig == SIGTERM || sig == SIGHUP) {
    g_terminate.store(true);
  }
  g_interrupt.store(true);
}

void install_signal_handlers() {
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0; // no SA_RESTART: Ctrl-C at the ">>> " prompt ends getline()
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGHUP, &sa, nullptr);
  // A closed stdout (e.g. `| head -3`) must not kill us before stop() runs;
  // run_prompt sees the failed write instead.
  std::signal(SIGPIPE, SIG_IGN);
}

// The [serve] name the GenAI backend reads from (its --serve-root default), so
// the host lists and checks models under the same root the card will pull from.
constexpr const char* kServeName = "models";

// The [serve] root a VLM image is staged under (the card pulls it from here).
// The image goes in a "pcie-genai" subfolder of this root.
constexpr const char* kImageServeName = "data";

// Read a whole file; nullopt if it cannot be opened (missing/no permission).
std::optional<std::string> read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Where the host stages a VLM image so the card can pull it: the "pcie-genai"
// folder under the daemon's [serve] "data" root. Empty if there is no such root
// (then image features are off and "add image" reports it).
std::string image_stage_dir_from_conf(const CliArgs& args) {
  if (const std::optional<std::string> conf = read_file(args.conf_path)) {
    if (const std::optional<std::string> data =
            pgenai::tools::serve_root_path(*conf, kImageServeName)) {
      return (std::filesystem::path(*data) / "pcie-genai").string();
    }
  }
  return {};
}

// `--list`: print the models the host serves over PCIe. Host-only: it reads the
// daemon config and the local serve directory, and never touches the card.
int run_list(const CliArgs& args) {
  const std::optional<std::string> conf = read_file(args.conf_path);
  if (!conf) {
    std::cerr << "pcie-genai: cannot read " << args.conf_path
              << " (the simaai-mla-daemon config). Pass --conf <path>.\n";
    return 1;
  }
  const std::optional<std::string> root = pgenai::tools::serve_root_path(*conf, kServeName);
  if (!root) {
    std::cerr << "pcie-genai: " << args.conf_path << " has no [serve] '" << kServeName
              << "' entry, so the host serves no models over PCIe.\n"
                 "Add e.g. 'models = /srv/simaai/models' under [serve] and restart the daemon.\n";
    return 1;
  }
  std::cout << pgenai::tools::format_model_listing(pgenai::tools::list_models_in(*root), *root);
  return 0;
}

// Fail fast when the model folder is not under the serve root, instead of
// starting the backend and waiting minutes for it to fail on the card. Only
// checks when the serve root exists on this host (e.g. rack1-pc2); otherwise it
// cannot know and stays quiet so unusual setups still run.
std::optional<int> reject_unknown_model(const CliArgs& args) {
  const std::optional<std::string> conf = read_file(args.conf_path);
  if (!conf) {
    return std::nullopt;
  }
  const std::optional<std::string> root = pgenai::tools::serve_root_path(*conf, kServeName);
  if (!root) {
    return std::nullopt;
  }
  std::error_code ec;
  if (!std::filesystem::is_directory(*root, ec)) {
    return std::nullopt; // serve root not local to this host: cannot check
  }
  if (std::filesystem::is_directory(std::filesystem::path(*root) / args.model, ec)) {
    return std::nullopt; // model is there
  }
  std::cerr << "pcie-genai: model '" << args.model << "' is not under " << *root
            << ".\nRun 'pcie-genai --list' to see the models, or copy the folder there first.\n";
  return 2;
}

// The session settings the CLI sends with every question. They match the
// card's Chat, so a question never resets it by accident.
struct ChatState {
  std::optional<std::string> system_prompt; // nullopt = model default, "" = none
  bool enable_thinking = false;
};

// Send one prompt and stream the answer to stdout; stats go to stderr so
// stdout stays pure answer text (a script can pipe or save just the answer).
// Ctrl-C cancels this answer only; SIGTERM/SIGHUP or a stdout that can no
// longer be written also end the session. Returns true if Ctrl-C cancelled it.
bool run_prompt(pgenai::GenAIModel& model, const CliArgs& args, const ChatState& chat,
                const std::string& prompt, const std::vector<std::string>& images) {
  pgenai::GenerationRequest request;
  request.prompt = prompt;
  request.system_prompt = chat.system_prompt;
  request.enable_thinking = chat.enable_thinking;
  request.max_new_tokens = args.max_new_tokens;

  pgenai::PcieRequestOptions options;
  options.image_files.assign(images.begin(), images.end());

  g_interrupt.store(g_terminate.load()); // a pending SIGTERM/SIGHUP is never cleared
  pgenai::GenerationStream stream = model.stream(request, options);
  // stream.next() blocks until the card sends something, so Ctrl-C is watched
  // on its own thread: it cancels at once, even while no token arrives (long
  // prefill, or a stuck card). Declared after `stream`, so it is destroyed first.
  const pgenai::tools::CancelWatcher watcher([] { return g_interrupt.load(); },
                                             [&stream] { stream.cancel(); });
  const auto stop_if_stdout_gone = [] {
    if (!std::cout && !g_terminate.load()) {
      // stdout is gone (SIGPIPE ignored, write failed): nobody reads the answer.
      std::cerr << "pcie-genai: stdout closed; stopping\n";
      g_terminate.store(true);
      g_interrupt.store(true); // the watcher cancels the answer
    }
  };
  while (const std::optional<pgenai::TokenSample> sample = stream.next()) {
    stop_if_stdout_gone();
    if (!sample->is_final) {
      if (!g_interrupt.load()) { // after Ctrl-C, print no more answer text
        std::cout << sample->text << std::flush;
        // Check at once: the next next() can block for a long time if the card
        // stalls, and the watcher only cancels once g_interrupt is set.
        stop_if_stdout_gone();
      }
      continue;
    }
    std::cout << '\n' << std::flush;
    std::cerr << pgenai::tools::format_final_stats(sample->metrics, model.last_run_dropped_events(),
                                                   sample->finish_reason)
              << "\n";
  }
  if (watcher.fired()) {
    std::cerr << "[cancelled]\n";
  }
  return watcher.fired();
}

} // namespace

int main(int argc, char** argv) {
  CliArgs args;
  try {
    args = pgenai::tools::parse_cli_args(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::invalid_argument& e) {
    std::cerr << "pcie-genai: " << e.what() << "\n\n" << pgenai::tools::cli_usage();
    return 2;
  }
  if (args.help) {
    std::cout << pgenai::tools::cli_usage();
    return 0;
  }
  if (args.list) {
    return run_list(args);
  }
  if (const std::optional<int> rc = reject_unknown_model(args)) {
    return *rc;
  }

  pcie::ConnectionOptions connection;
  connection.card_host = args.card_host;
  connection.card_id = args.card_id;
  connection.user = args.user;
  connection.queue = args.queue;

  int rc = 0;
  int pid = -1;
  std::optional<pcie::internal::RemoteRuntime> runtime;
  try {
    runtime.emplace(connection, args.card_program);
    // Before start(): the model load can take minutes. A Ctrl-C during the
    // READY wait must end the wait (should_abort below) so the code after the
    // try block still stops the backend. Without our handler, Ctrl-C would
    // kill this process and leave the backend running on the card.
    install_signal_handlers();
    std::cerr << "Starting " << args.card_program << " on " << runtime->endpoint() << " (queue "
              << args.queue << ")...\n";
    pid = runtime->start(args.queue, args.model, std::nullopt);
    std::cerr << "Waiting for READY (the model loads over PCIe; this can take minutes)...\n";
    runtime->wait_ready(args.queue, pid, args.ready_timeout_s * 1000,
                        [] { return g_interrupt.load(); });
    std::cerr << "READY.\n";

    const std::string image_stage_dir = image_stage_dir_from_conf(args);
    pgi::SvcTransportOptions options;
    options.model_id = args.model;
    options.image_stage_dir = image_stage_dir;
    pgenai::GenAIModel model = pgi::make_model_with_transport(std::make_unique<pgi::SvcTransport>(
        std::make_unique<pgi::DlSvcClient>(static_cast<std::uint32_t>(args.card_id)), options));

    // A card backend without chat support has no genai.chat. It would answer
    // without images and without memory, and say nothing. Stop here instead.
    // Two ways this preflight fails: the call throws (no genai.chat at all, an
    // older backend), or it returns ok == false (a backend that answers
    // genai.chat but refuses the operation, e.g. a compatibility mismatch).
    pgenai::ChatReply preflight{false, ""};
    try {
      preflight = model.chat_history();
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string(e.what()) + "\nThe card backend (" + args.card_program +
          ") does not support chat: it is older than this pcie-genai. Update it, "
          "or pick another card program with --card-program.");
    }
    if (!preflight.ok) {
      throw std::runtime_error(
          "The card backend (" + args.card_program +
          ") refused the chat preflight: " + preflight.text +
          "\nIt cannot keep the conversation (memory or images), so stopping here.");
    }

    ChatState chat{args.system_prompt, false};
    if (args.prompt.has_value()) {
      run_prompt(model, args, chat, *args.prompt, args.images);
    } else {
      const bool interactive = ::isatty(STDIN_FILENO) != 0;
      // Interactive prompts get arrow-key editing and up-arrow recall of past
      // prompts, saved across sessions in ~/.pcie_genai_history (like the LLiMa
      // console). A piped stdin keeps the plain std::getline path.
      std::optional<pgenai::tools::ReadlineSession> editor;
      if (interactive) {
        editor.emplace([] { return g_interrupt.load(); }, [] { return g_terminate.load(); });
      }
      std::string line;
      // The card keeps the conversation. Images added with
      // 'add image' go with the next question, then stay in the card's history.
      std::vector<std::string> pending_images;
      // A chat command's result: the devkit text on success, the card's text otherwise.
      const auto report = [](const pgenai::ChatReply& reply, const char* ok_text) {
        std::cerr << (reply.ok ? std::string(ok_text) : reply.text) << "\n";
      };
      // Stop on SIGTERM/SIGHUP or once stdout is closed; stop() below then runs.
      while (!g_terminate.load() && std::cout) {
        std::optional<std::string> input;
        if (interactive) {
          // A pending Ctrl-C left over from cancelling the previous line must
          // not cancel the next one; clear it, but keep a pending terminate.
          g_interrupt.store(g_terminate.load());
          input = editor->read_line(">>> ");
        } else if (std::getline(std::cin, line)) {
          input = line;
        }
        if (!input || g_terminate.load() || *input == "quit" || *input == "exit") {
          break;
        }
        if (input->empty()) {
          continue;
        }
        if (interactive) {
          editor->record(*input);
        }
        using pgenai::tools::ReplKind;
        const pgenai::tools::ReplLine cmd = pgenai::tools::classify_repl_line(*input);
        if (cmd.kind != ReplKind::Prompt) {
          try {
            switch (cmd.kind) {
            case ReplKind::AddImage: {
              std::error_code ec;
              if (!std::filesystem::is_regular_file(cmd.text, ec)) {
                std::cerr << "Image file not found: " << cmd.text << "\n";
              } else if (image_stage_dir.empty()) {
                std::cerr << "pcie-genai: images need a [serve] 'data' root in " << args.conf_path
                          << "\n";
              } else {
                pending_images.push_back(cmd.text); // silent, like the devkit
              }
              break;
            }
            case ReplKind::ClearImage:
              std::cerr << "The clear image command is not supported; use clear history instead.\n";
              break;
            case ReplKind::ClearHistory: {
              const pgenai::ChatReply r =
                  model.reset_chat(chat.system_prompt, chat.enable_thinking);
              if (r.ok) {
                pending_images.clear();
              }
              report(r, "Cleared chat history.");
              break;
            }
            case ReplKind::SetSystem: {
              const pgenai::ChatReply r = model.reset_chat(cmd.text, chat.enable_thinking);
              if (r.ok) {
                chat.system_prompt = cmd.text;
                pending_images.clear();
              }
              report(r, "Set system message and cleared chat history.");
              break;
            }
            case ReplKind::ClearSystem: {
              const pgenai::ChatReply r = model.reset_chat(std::string(), chat.enable_thinking);
              if (r.ok) {
                chat.system_prompt = std::string();
                pending_images.clear();
              }
              report(r, "Cleared system message and chat history.");
              break;
            }
            case ReplKind::EnableThinking:
            case ReplKind::DisableThinking: {
              const bool on = cmd.kind == ReplKind::EnableThinking;
              const pgenai::ChatReply r = model.reset_chat(chat.system_prompt, on);
              if (r.ok) {
                chat.enable_thinking = on;
                pending_images.clear();
              }
              report(r, on ? "Enabled thinking and cleared chat history."
                           : "Disabled thinking and cleared chat history.");
              break;
            }
            case ReplKind::PrintHistory: {
              const pgenai::ChatReply r = model.chat_history();
              if (r.ok) {
                std::cout << r.text << std::endl;
              } else {
                std::cerr << r.text << "\n";
              }
              break;
            }
            case ReplKind::Help:
              std::cout << pgenai::tools::repl_help() << std::endl;
              break;
            case ReplKind::Prompt:
              break;
            }
          } catch (const std::exception& e) {
            std::cerr << "pcie-genai: " << e.what() << "\n";
            if (!interactive) {
              throw;
            }
          }
          continue;
        }
        const std::vector<std::string> images = std::exchange(pending_images, {});
        try {
          const bool cancelled = run_prompt(model, args, chat, *input, images);
          if (model.last_run_cleared_history()) {
            std::cerr << pgenai::tools::history_cleared_message(cancelled, false) << "\n";
          }
        } catch (const std::exception& e) {
          // One failed prompt must not end an interactive session.
          std::cerr << "pcie-genai: " << e.what() << "\n";
          if (model.last_run_cleared_history()) {
            std::cerr << pgenai::tools::history_cleared_message(false, true) << "\n";
          }
          if (!interactive) {
            throw;
          }
        }
      }
    }
  } catch (const pcie::internal::RemoteStartError& e) {
    std::cerr << "pcie-genai: " << e.what() << "\n";
    rc = 1;
    // start() failed. The pid never reached the variable below, so the cleanup
    // there is skipped; but the script may already have started the backend
    // (nohup) before it failed. Stop that pid directly if we got one. The card
    // guards the kill with a /proc/<pid>/cmdline check, so it is safe.
    if (runtime && e.launched_pid() > 0) {
      try {
        runtime->stop_launched_pid(e.launched_pid());
      } catch (const std::exception& stop_err) {
        std::cerr << "pcie-genai: could not stop the backend after a failed start: "
                  << stop_err.what() << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "pcie-genai: " << e.what() << "\n";
    rc = 1;
  }

  if (runtime && pid > 0) {
    try {
      runtime->stop(args.queue, pid);
    } catch (const std::exception& e) {
      std::cerr << "pcie-genai: could not stop the backend: " << e.what() << "\n";
      rc = rc != 0 ? rc : 1;
    }
  }
  return rc;
}
