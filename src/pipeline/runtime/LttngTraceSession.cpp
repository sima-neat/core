#include "LttngMetricsCollector.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace simaai::neat::pipeline_internal {
namespace {

std::atomic_bool g_lttng_collector_active{false};

struct CommandResult {
  int exit_code = -1;
  std::string output;
};

#if defined(__linux__)
CommandResult run_command(const std::vector<std::string>& argv) {
  CommandResult result;
  if (argv.empty()) {
    result.output = "empty command";
    return result;
  }

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    result.output = "pipe() failed";
    return result;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    result.output = "fork() failed";
    return result;
  }

  if (pid == 0) {
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);

    std::vector<char*> args;
    args.reserve(argv.size() + 1U);
    for (const std::string& arg : argv) {
      args.push_back(const_cast<char*>(arg.c_str()));
    }
    args.push_back(nullptr);
    execvp(args[0], args.data());
    _exit(127);
  }

  close(pipefd[1]);
  char buf[4096];
  while (true) {
    const ssize_t n = read(pipefd[0], buf, sizeof(buf));
    if (n > 0) {
      result.output.append(buf, static_cast<std::size_t>(n));
    } else {
      break;
    }
  }
  close(pipefd[0]);

  int status = 0;
  if (waitpid(pid, &status, 0) == pid) {
    if (WIFEXITED(status)) {
      result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      result.exit_code = 128 + WTERMSIG(status);
    }
  }
  return result;
}
#else
CommandResult run_command(const std::vector<std::string>& argv) {
  (void)argv;
  return {-1, "LTTng metrics are supported only on Linux"};
}
#endif

bool command_exists(const std::string& command) {
#if defined(__linux__)
  const char* path = std::getenv("PATH");
  if (!path) {
    return false;
  }
  std::stringstream ss(path);
  std::string dir;
  while (std::getline(ss, dir, ':')) {
    std::filesystem::path p = dir.empty() ? std::filesystem::path(".") : std::filesystem::path(dir);
    p /= command;
    if (access(p.c_str(), X_OK) == 0) {
      return true;
    }
  }
#endif
  return false;
}

std::string default_session_name() {
#if defined(__linux__)
  const long pid = static_cast<long>(getpid());
#else
  const long pid = 0;
#endif
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return "sima-neat-metrics-" + std::to_string(pid) + "-" + std::to_string(now);
}

std::string make_trace_dir(const std::string& parent_or_empty, bool* owns) {
  std::filesystem::path parent = parent_or_empty.empty() ? std::filesystem::temp_directory_path()
                                                         : std::filesystem::path(parent_or_empty);
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  std::filesystem::path templ = parent / "sima-neat-metrics-XXXXXX";
  std::string mutable_template = templ.string();
#if defined(__linux__)
  std::vector<char> data(mutable_template.begin(), mutable_template.end());
  data.push_back('\0');
  char* made = mkdtemp(data.data());
  if (made) {
    if (owns) {
      *owns = true;
    }
    return std::string(made);
  }
#endif
  std::filesystem::path fallback =
      parent / ("sima-neat-metrics-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(fallback, ec);
  if (owns) {
    *owns = true;
  }
  return fallback.string();
}

bool run_or_error(const std::vector<std::string>& argv, std::string* err) {
  const CommandResult r = run_command(argv);
  if (r.exit_code == 0) {
    return true;
  }
  if (err) {
    std::ostringstream os;
    os << "command failed (" << r.exit_code << "):";
    for (const std::string& arg : argv) {
      os << ' ' << arg;
    }
    if (!r.output.empty()) {
      os << "\n" << r.output;
    }
    *err = os.str();
  }
  return false;
}

std::string read_command_output(const std::vector<std::string>& argv, int* exit_code) {
  const CommandResult r = run_command(argv);
  if (exit_code) {
    *exit_code = r.exit_code;
  }
  return r.output;
}

} // namespace

std::uint64_t stable_trace_hash(std::string_view value) {
  std::uint64_t h = 1469598103934665603ull;
  for (unsigned char c : value) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ull;
  }
  return h == 0 ? 1 : h;
}

LttngMetricsCollector::LttngMetricsCollector(InternalMetricsTraceOptions opt,
                                             TraceIdentityContext trace_context)
    : opt_(std::move(opt)), trace_context_(trace_context) {
  session_name_ = opt_.session_name.empty() ? default_session_name() : opt_.session_name;
  trace_dir_ = opt_.trace_dir.empty() ? make_trace_dir({}, &owns_trace_dir_)
                                      : make_trace_dir(opt_.trace_dir, &owns_trace_dir_);
}

LttngMetricsCollector::~LttngMetricsCollector() noexcept {
  cleanup_noexcept();
}

bool LttngMetricsCollector::available(std::string* reason) const {
  if (!command_exists("lttng")) {
    if (reason) {
      *reason = "lttng command not found";
    }
    return false;
  }
  if (!command_exists("babeltrace2")) {
    if (reason) {
      *reason = "babeltrace2 command not found";
    }
    return false;
  }
  return true;
}

bool LttngMetricsCollector::start(std::string* err) {
  std::string reason;
  if (!available(&reason)) {
    if (err) {
      *err = reason;
    }
    return false;
  }

  bool expected = false;
  if (!g_lttng_collector_active.compare_exchange_strong(expected, true)) {
    if (err) {
      *err = "another LTTng metrics collector is already active in this process";
    }
    return false;
  }

  if (!run_or_error({"lttng", "create", session_name_, "--output", trace_dir_}, err)) {
    g_lttng_collector_active.store(false);
    return false;
  }
  created_ = true;

  const std::string subbuf = std::to_string(std::max(1, opt_.subbuf_size_kb)) + "K";
  if (!run_or_error({"lttng", "enable-channel", "--userspace", "--session", session_name_,
                     "--discard", "--blocking-timeout=0", "--subbuf-size", subbuf, "--num-subbuf",
                     std::to_string(std::max(2, opt_.num_subbuf)), "sima-neat-metrics"},
                    err)) {
    cleanup_noexcept();
    return false;
  }

  std::vector<std::string> context_cmd = {"lttng",       "add-context",
                                          "--userspace", "--session",
                                          session_name_, "--channel=sima-neat-metrics",
                                          "--type=vpid", "--type=vtid"};
  if (opt_.add_procname_context) {
    context_cmd.push_back("--type=procname");
  }
  if (!run_or_error(context_cmd, err)) {
    cleanup_noexcept();
    return false;
  }

#if defined(__linux__)
  if (!run_or_error({"lttng", "track", "--userspace", "--session", session_name_, "--vpid",
                     std::to_string(static_cast<long>(getpid()))},
                    err)) {
    cleanup_noexcept();
    return false;
  }
#endif

  if (!run_or_error({"lttng", "enable-event", "--userspace", "--session", session_name_,
                     "--channel=sima-neat-metrics", "sima_neat_plugin:*"},
                    err)) {
    cleanup_noexcept();
    return false;
  }
  if (opt_.enable_message_events) {
    if (!run_or_error({"lttng", "enable-event", "--userspace", "--session", session_name_,
                       "--channel=sima-neat-metrics", "sima_neat_edge:*"},
                      err)) {
      cleanup_noexcept();
      return false;
    }
  }
  if (opt_.enable_pipeline_fallback) {
    if (!run_or_error({"lttng", "enable-event", "--userspace", "--session", session_name_,
                       "--channel=sima-neat-metrics", "pipeline:*"},
                      err)) {
      cleanup_noexcept();
      return false;
    }
  }
  if (opt_.enable_remote_core_debug) {
    if (!run_or_error({"lttng", "enable-event", "--userspace", "--session", session_name_,
                       "--channel=sima-neat-metrics", "remote_core:*"},
                      err)) {
      cleanup_noexcept();
      return false;
    }
  }

  if (!run_or_error({"lttng", "start", session_name_}, err)) {
    cleanup_noexcept();
    return false;
  }
  started_ = true;
  return true;
}

bool LttngMetricsCollector::stop_and_destroy(std::string* err) {
  bool ok = true;
  std::string local_err;
  if (started_) {
    if (!run_or_error({"lttng", "stop", session_name_}, &local_err)) {
      ok = false;
      if (err) {
        *err = local_err;
      }
    }
    started_ = false;
  }
  if (created_) {
    local_err.clear();
    if (!run_or_error({"lttng", "destroy", session_name_}, &local_err)) {
      ok = false;
      if (err && err->empty()) {
        *err = local_err;
      }
    }
    created_ = false;
  }
  g_lttng_collector_active.store(false);
  return ok;
}

LttngParseResult LttngMetricsCollector::parse(std::string* err) const {
  int exit_code = -1;
  const std::string text =
      read_command_output({"babeltrace2", "--color=never", "--no-delta", "--clock-seconds",
                           "--names=context,payload", trace_dir_},
                          &exit_code);
  if (exit_code != 0) {
    if (err) {
      *err = "babeltrace2 failed with exit code " + std::to_string(exit_code) + "\n" + text;
    }
    LttngParseResult failed;
    failed.trace_dir = trace_dir_;
    return failed;
  }
  LttngParseResult parsed =
      parse_lttng_trace_text(text, trace_context_.run_id_hash, trace_context_.graph_id_hash,
                             opt_.enable_pipeline_fallback);
  parsed.trace_dir = trace_dir_;
  return parsed;
}

void LttngMetricsCollector::cleanup_noexcept() {
  try {
    std::string ignored;
    if (started_ || created_) {
      stop_and_destroy(&ignored);
    }
    g_lttng_collector_active.store(false);
    if (owns_trace_dir_ && !opt_.retain_trace && !trace_dir_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(trace_dir_, ec);
      owns_trace_dir_ = false;
    }
  } catch (...) {
  }
}

} // namespace simaai::neat::pipeline_internal
