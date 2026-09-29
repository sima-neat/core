#include "pcie_genai_readline.h"

#include "pcie_genai_history.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

#include <unistd.h>

#include <readline/history.h>
#include <readline/readline.h>

namespace simaai::neat::pcie::genai::tools {

namespace {

// The most history lines we keep on disk, matching the LLiMa console.
constexpr int kMaxHistoryEntries = 1000;

// readline's getc hook is a plain C function pointer with no captured state, so
// the session's predicates live here. Only one ReadlineSession exists at a
// time, so a single set of file-static predicates is enough.
std::function<bool()> g_interrupt_pending;
std::function<bool()> g_terminate_pending;

// Read one byte for readline, but let the CLI's signals win. readline's own
// signal handling is off (rl_catch_signals = 0), so a SIGINT/SIGTERM/SIGHUP
// interrupts read() with EINTR (the CLI installs its handlers without
// SA_RESTART) and we act on the flags the CLI's handler set.
int genai_rl_getc(FILE* stream) {
  for (;;) {
    if (g_terminate_pending && g_terminate_pending()) {
      // SIGTERM/SIGHUP/closed stdout: end input so readline returns EOF and the
      // CLI's cleanup stops the backend on the card.
      return EOF;
    }
    if (g_interrupt_pending && g_interrupt_pending()) {
      // Ctrl-C: throw away the typed line and accept an empty one, so readline
      // returns "" and the REPL draws a fresh prompt instead of exiting.
      rl_replace_line("", 0);
      return '\n';
    }
    errno = 0;
    unsigned char c = 0;
    const ssize_t n = ::read(fileno(stream), &c, 1);
    if (n == 1) {
      return static_cast<int>(c);
    }
    if (n == 0) {
      return EOF; // real end of input (Ctrl-D on an empty line)
    }
    if (errno == EINTR) {
      continue; // a signal fired; re-check the flags at the top of the loop
    }
    return EOF; // read error
  }
}

} // namespace

ReadlineSession::ReadlineSession(std::function<bool()> interrupt_pending,
                                 std::function<bool()> terminate_pending)
    : history_path_(history_file_path().string()) {
  g_interrupt_pending = std::move(interrupt_pending);
  g_terminate_pending = std::move(terminate_pending);

  rl_readline_name = "pcie-genai";
  rl_catch_signals = 0;    // the CLI owns SIGINT/SIGTERM/SIGHUP (it stops the card)
  rl_catch_sigwinch = 1;   // but let readline keep track of terminal resizes
  rl_getc_function = genai_rl_getc;

  using_history();
  read_history(history_path_.c_str()); // no-op if the file does not exist yet
}

ReadlineSession::~ReadlineSession() {
  write_history(history_path_.c_str());
  history_truncate_file(history_path_.c_str(), kMaxHistoryEntries);
  g_interrupt_pending = nullptr;
  g_terminate_pending = nullptr;
  rl_getc_function = nullptr;
}

std::optional<std::string> ReadlineSession::read_line(const std::string& prompt) {
  char* raw = readline(prompt.c_str());
  if (raw == nullptr) {
    return std::nullopt; // EOF (Ctrl-D) or terminate
  }
  std::string line(raw);
  std::free(raw);
  return line;
}

void ReadlineSession::record(const std::string& line) {
  if (!should_record_history(line, last_recorded_)) {
    return;
  }
  add_history(line.c_str());
  append_history(1, history_path_.c_str());
  last_recorded_ = line;
}

} // namespace simaai::neat::pcie::genai::tools
