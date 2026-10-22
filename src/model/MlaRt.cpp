#include "model/internal/MlaRt.h"

#include <glib.h>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <stdexcept>

namespace simaai::neat {
namespace {

[[noreturn]] void mla_rt_error(const std::string& message) {
  throw std::runtime_error(message);
}

// Set once an mla-rt run is killed or crashes. MLA may then still hold unfinished work, so this
// process starts no further mla-rt run.
std::atomic<bool> mla_needs_recovery{false};

// A started mla-rt. Destruction before it finishes kills its process group, then reaps it.
struct MlaRtProcess {
  GPid pid = 0;
  GPid group = 0;
  int output = -1;
  ~MlaRtProcess() {
    if (group > 0) {
      (void)::kill(-group, SIGKILL);
      mla_needs_recovery = true;
    }
    if (pid > 0) {
      while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      g_spawn_close_pid(pid);
    }
    if (output >= 0)
      ::close(output);
  }
};

// Runs in the child before exec, so it may only make async-signal-safe calls. The new process
// group lets a timeout stop anything mla-rt started; stderr joins stdout on one pipe.
void prepare_mla_rt(gpointer) {
  if (::setpgid(0, 0) != 0 || ::dup2(STDOUT_FILENO, STDERR_FILENO) < 0)
    _exit(127);
}

} // namespace

std::string internal::run_mla_rt(const std::filesystem::path& mla_rt,
                                 const std::vector<std::string>& args,
                                 std::chrono::milliseconds timeout) {
  if (timeout <= std::chrono::milliseconds::zero())
    mla_rt_error("mla-rt timeout must be positive");
  if (mla_needs_recovery)
    mla_rt_error("skipped: an earlier mla-rt run was killed or crashed, so MLA needs recovery");
  std::vector<std::string> command{mla_rt.string()};
  command.insert(command.end(), args.begin(), args.end());
  std::vector<gchar*> argv;
  for (auto& arg : command)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  MlaRtProcess process;
  GError* error = nullptr;
  if (!g_spawn_async_with_pipes(nullptr, argv.data(), nullptr, G_SPAWN_DO_NOT_REAP_CHILD,
                                prepare_mla_rt, nullptr, &process.pid, nullptr, &process.output,
                                nullptr, &error)) {
    const std::string detail = error ? error->message : "unknown error";
    g_clear_error(&error);
    mla_rt_error("cannot start mla-rt: " + detail);
  }
  // Set the group here too, so a kill right after the spawn still reaches mla-rt.
  (void)::setpgid(process.pid, process.pid);
  process.group = process.pid;

  constexpr std::size_t max_output = 4U * 1024U;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::string output;
  int status = 0;
  // Read until mla-rt has exited and closed its output.
  while (process.pid > 0 || process.output >= 0) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero())
      mla_rt_error("mla-rt timed out after " + std::to_string(timeout.count()) + " ms\n" + output);
    // With the pipe closed, poll only waits before the next exit check.
    pollfd ready{process.output, POLLIN, 0};
    const auto wait_ms = static_cast<int>(std::min<std::int64_t>(remaining.count(), 50));
    if (::poll(&ready, process.output >= 0 ? 1 : 0, wait_ms) < 0 && errno != EINTR)
      mla_rt_error("cannot wait for mla-rt output");
    if (ready.revents != 0) {
      char buffer[4096];
      const auto bytes = ::read(process.output, buffer, sizeof(buffer));
      if (bytes > 0) {
        output.append(buffer, static_cast<std::size_t>(bytes));
        if (output.size() > max_output)
          output.erase(0, output.size() - max_output);
      } else if (bytes == 0) {
        ::close(process.output);
        process.output = -1;
      } else if (errno != EINTR) {
        mla_rt_error("cannot read mla-rt output");
      }
    }
    if (process.pid > 0) {
      const auto waited = ::waitpid(process.pid, &status, WNOHANG);
      if (waited == process.pid) {
        g_spawn_close_pid(process.pid);
        process.pid = 0;
      } else if (waited < 0 && errno != EINTR) {
        mla_rt_error("cannot wait for mla-rt");
      }
    }
  }
  process.group = 0;
  if (WIFSIGNALED(status))
    mla_needs_recovery = true;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string result = WIFEXITED(status) ? "exit " + std::to_string(WEXITSTATUS(status))
                                                 : "signal " + std::to_string(WTERMSIG(status));
    mla_rt_error("mla-rt failed (" + result + ")\n" + output);
  }
  return output;
}

} // namespace simaai::neat
