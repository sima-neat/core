/**
 * @file
 * @brief GNU readline line-editing + persistent history for the pcie-genai
 *        interactive REPL, matching the LLiMa devkit console.
 *
 * Gives the interactive prompt arrow-key editing and up-arrow recall of past
 * prompts, saved across sessions in ~/.pcie_genai_history (see
 * pcie_genai_history.h). Used only on the interactive (isatty) path; a piped
 * stdin still reads with std::getline and never touches this.
 *
 * Signals: this deliberately turns readline's own signal handling OFF
 * (rl_catch_signals = 0) so the CLI's existing SIGINT/SIGTERM/SIGHUP handlers
 * -- which stop the backend on the card -- stay in control. A custom getc hook
 * reacts to the CLI's interrupt/terminate flags: Ctrl-C discards the current
 * line and re-prompts (LLiMa's feel); SIGTERM/SIGHUP make readline return EOF
 * so the CLI's cleanup runs. rl_catch_sigwinch stays on so resize still works.
 */
#pragma once

#include <functional>
#include <optional>
#include <string>

namespace simaai::neat::pcie::genai::tools {

// Owns readline's global configuration and the on-disk history file for the
// lifetime of one interactive session. Construct one before the REPL loop and
// let it destruct after: the destructor writes the history back. Only one may
// exist at a time (it drives readline's process-global state).
class ReadlineSession {
public:
  // `interrupt_pending` returns true when Ctrl-C was seen (cancel the current
  // line). `terminate_pending` returns true on SIGTERM/SIGHUP or a closed
  // stdout (end input so the CLI can stop the backend and exit).
  ReadlineSession(std::function<bool()> interrupt_pending, std::function<bool()> terminate_pending);
  ~ReadlineSession();

  ReadlineSession(const ReadlineSession&) = delete;
  ReadlineSession& operator=(const ReadlineSession&) = delete;

  // Read one edited line. Returns the line (without the trailing newline), or
  // std::nullopt on EOF (Ctrl-D) / terminate. A Ctrl-C returns an empty string.
  std::optional<std::string> read_line(const std::string& prompt);

  // Add a line to the history if it is worth keeping (see should_record_history)
  // and append it to the history file so it survives a crash.
  void record(const std::string& line);

private:
  std::string history_path_;
  std::string last_recorded_;
};

} // namespace simaai::neat::pcie::genai::tools
