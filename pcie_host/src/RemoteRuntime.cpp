#include "RemoteRuntime.h"

#include "SshRunner.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include <unistd.h>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {
namespace {

constexpr int kSshPort = 22;
constexpr int kConnectTimeoutSec = 10;
constexpr int kCommandTimeoutSec = 30;
// Start command exit code: the card program exited before the host saw it
// claim its queue (it failed at once; its status file may say why).
constexpr int kExitBeforeQueueClaim = 15;
constexpr const char* kRemoteModelDir = "/tmp";
constexpr const char* kDefaultCardProgram = "pcie-pipeline-builder";
constexpr const char* kRemoteBinDir = "/usr/bin/";
constexpr const char* kDefaultIdentityFile = ".ssh/sima_neat_pcie_ed25519";
constexpr const char* kDefaultMlashmCtrlIoTimeout = "MLASHM_CTRL_IO_TIMEOUT_MS=5000";

std::optional<std::string> default_identity_file() {
  const char* home = std::getenv("HOME");
  if (!home || !*home) {
    return std::nullopt;
  }
  std::filesystem::path path = std::filesystem::path(home) / kDefaultIdentityFile;
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return std::nullopt;
  }
  return path.string();
}

std::string derive_card_host(const ConnectionOptions& opt) {
  if (!opt.card_host.empty()) {
    return opt.card_host;
  }
  const int safe_id = std::max(0, opt.card_id);
  return "10.0." + std::to_string(safe_id) + ".2";
}

int json_int_or(const nlohmann::json& object, const char* key, const int fallback) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    return fallback;
  }
  if (it->is_number_integer()) {
    return it->get<int>();
  }
  return fallback;
}

std::string json_string_or(const nlohmann::json& object, const char* key,
                           const std::string& fallback = {}) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    return fallback;
  }
  if (it->is_string()) {
    return it->get<std::string>();
  }
  return fallback;
}

std::string default_card_gst_debug_file(const int queue) {
  return "/var/log/sima-neat/pcie/q" + std::to_string(queue) + ".gst.log";
}

bool is_env_name_char(const char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

std::vector<std::string> split_card_env(const std::string& raw) {
  std::vector<std::string> out;
  std::istringstream stream(raw);
  std::string token;
  while (stream >> token) {
    const std::size_t eq = token.find('=');
    if (eq == std::string::npos || eq == 0) {
      throw std::runtime_error("card_env entries must be NAME=VALUE assignments");
    }
    for (std::size_t i = 0; i < eq; ++i) {
      const char ch = token[i];
      if ((i == 0 && std::isdigit(static_cast<unsigned char>(ch))) || !is_env_name_char(ch)) {
        throw std::runtime_error("card_env contains invalid environment variable name: " +
                                 token.substr(0, eq));
      }
    }
    out.push_back(std::move(token));
  }
  return out;
}

bool env_contains_name(const std::vector<std::string>& env, const std::string& name) {
  const std::string prefix = name + "=";
  return std::any_of(env.begin(), env.end(),
                     [&prefix](const std::string& entry) { return entry.rfind(prefix, 0) == 0; });
}

void validate_endpoint_component(const std::string& value, const char* name,
                                 const bool allow_empty) {
  if (value.empty()) {
    if (allow_empty) {
      return;
    }
    throw std::invalid_argument(std::string(name) + " must not be empty");
  }
  if (value.front() == '-' || value.find('@') != std::string::npos ||
      std::any_of(value.begin(), value.end(),
                  [](const unsigned char ch) { return std::isspace(ch) != 0; })) {
    throw std::invalid_argument(std::string(name) + " is not a valid SSH endpoint component");
  }
}

// The /proc/<pid>/cmdline guard: does the cmdline contain the program name?
// -F: a fixed string, not a regex, so a '.' in a name like pcie.genai-backend
// does not match any character (pcieXgenai-backend is another program).
// The name is already checked to be [A-Za-z0-9._-] and not to start with '-'.
std::string cmdline_grep(const std::string& program) {
  return "grep -qF -- '" + program + "'";
}

// The busy check at start: is a live queue owner ANY card program, not only
// the one we start? The queue can be owned by the tensor pipeline
// (pcie-pipeline-builder) or GenAI (pcie-genai-backend, or a custom name that
// contains it). A live owner of another kind must not be taken for stale: its
// pid and status files would be erased, and its own stop() could no longer
// find it. A live pid that is none of these is a reused pid (stale).
std::string any_card_program_grep(const std::string& program) {
  return "grep -qF -e '" + program + "' -e '" + std::string(kDefaultCardProgram) +
         "' -e 'pcie-genai-backend'";
}

void validate_card_program(const std::string& value) {
  // Empty is allowed: it selects the default program. A non-empty name is placed
  // inside a shell single-quoted grep pattern and appended to a launch path, so
  // restrict it to a safe basename charset to keep the command injection-free.
  if (value.empty()) {
    return;
  }
  const bool safe = std::all_of(value.begin(), value.end(), [](const unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '.' || ch == '_' || ch == '-';
  });
  if (!safe || value.front() == '-') {
    throw std::invalid_argument("card_program must match [A-Za-z0-9._-] and not start with '-'");
  }
}

} // namespace

RemoteRuntime::RemoteRuntime(ConnectionOptions connection, std::string card_program)
    : connection_(std::move(connection)), card_program_(std::move(card_program)) {
  validate_endpoint_component(connection_.user, "user", false);
  validate_endpoint_component(connection_.card_host, "card_host", true);
  validate_card_program(card_program_);
}

std::string RemoteRuntime::endpoint() const {
  return connection_.user + "@" + derive_card_host(connection_);
}

std::string RemoteRuntime::status_path(const int queue) const {
  return "/run/sima-neat/pcie/q" + std::to_string(queue) + ".status";
}

std::string RemoteRuntime::pid_path(const int queue) const {
  return "/run/sima-neat/pcie/q" + std::to_string(queue) + ".pid";
}

std::string RemoteRuntime::card_program() const {
  return card_program_.empty() ? std::string(kDefaultCardProgram) : card_program_;
}

std::string RemoteRuntime::remote_helper_path() const {
  return std::string(kRemoteBinDir) + card_program();
}

std::string RemoteRuntime::child_cleanup_shell_function() {
  return "child_exited() { "
         "! kill -0 \"$launched_pid\" >/dev/null 2>&1 || "
         "grep -q '^State:[[:space:]]*Z' \"/proc/$launched_pid/status\" 2>/dev/null; "
         "}; "
         "terminate_launched() { "
         "if child_exited; then wait \"$launched_pid\" >/dev/null 2>&1 || true; return 0; fi; "
         "kill -TERM \"$launched_pid\" >/dev/null 2>&1 || true; "
         "for i in $(seq 1 20); do "
         "if child_exited; then wait \"$launched_pid\" >/dev/null 2>&1 || true; return 0; fi; "
         "sleep 0.05; done; "
         "kill -KILL \"$launched_pid\" >/dev/null 2>&1 || true; "
         "for i in $(seq 1 20); do "
         "if child_exited; then wait \"$launched_pid\" >/dev/null 2>&1 || true; return 0; fi; "
         "sleep 0.05; done; "
         "return 1; "
         "}; ";
}

std::string RemoteRuntime::build_start_command(
    const int queue, const std::string& remote_model_path,
    const std::optional<std::string>& remote_model_options_path) const {
  const std::string helper = remote_helper_path();
  // Single-quoted grep pattern; card_program() is validated to a safe charset.
  const std::string cmdline_match = cmdline_grep(card_program());
  const std::string any_owner_match = any_card_program_grep(card_program());
  const std::string start_lock_path =
      "/run/sima-neat/pcie/q" + std::to_string(queue) + ".start.lock";
  std::ostringstream ss;
  ss << child_cleanup_shell_function() << "[ -x " << SshRunner::shell_escape(helper)
     << " ] || { echo missing_builder; exit 10; }; "
     << "[ -d /run/sima-neat/pcie ] || { echo missing_run_dir; exit 11; }; "
     << "[ -d /var/log/sima-neat/pcie ] || { echo missing_log_dir; exit 12; }; "
     << "startlock=" << SshRunner::shell_escape(start_lock_path) << "; "
     << "exec 9>\"$startlock\"; " << "flock -w 30 9 || { echo start_lock_timeout; exit 14; }; "
     << "pidfile=" << SshRunner::shell_escape(pid_path(queue)) << "; "
     << "statusfile=" << SshRunner::shell_escape(status_path(queue)) << "; "
     << "if [ -f \"$pidfile\" ]; then " << "pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then "
     << "if tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | " << any_owner_match
     << "; "
        "then "
     << "echo queue_busy; exit 9; " << "fi; " << "fi; " << "rm -f \"$pidfile\" \"$statusfile\"; "
     << "else " << "rm -f \"$statusfile\"; " << "fi; ";
  ss << "nohup ";
  std::vector<std::string> card_env = split_card_env(connection_.card_env);
  if (!env_contains_name(card_env, "MLASHM_CTRL_IO_TIMEOUT_MS")) {
    card_env.emplace_back(kDefaultMlashmCtrlIoTimeout);
  }
  const bool needs_env = !card_env.empty() || !connection_.card_gst_debug.empty();
  if (needs_env) {
    ss << "env ";
    for (const auto& entry : card_env) {
      ss << SshRunner::shell_escape(entry) << " ";
    }
  }
  if (!connection_.card_gst_debug.empty()) {
    ss << "GST_DEBUG=" << SshRunner::shell_escape(connection_.card_gst_debug) << " ";
    ss << "GST_DEBUG_NO_COLOR=1 ";
    const std::string debug_file = connection_.card_gst_debug_file.empty()
                                       ? default_card_gst_debug_file(queue)
                                       : connection_.card_gst_debug_file;
    if (!debug_file.empty()) {
      ss << "GST_DEBUG_FILE=" << SshRunner::shell_escape(debug_file) << " ";
    }
  }
  ss << SshRunner::shell_escape(helper) << " --model " << SshRunner::shell_escape(remote_model_path)
     << " --queue " << queue;
  if (remote_model_options_path.has_value()) {
    ss << " --model-options " << SshRunner::shell_escape(*remote_model_options_path);
  }
  ss << " 9>&- >/dev/null 2>&1 & " << "launched_pid=$!; " << "echo \"launched_pid=$launched_pid\"; "
     << "for i in $(seq 1 200); do " << "owner_pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ \"$owner_pid\" = \"$launched_pid\" ]; then exit 0; fi; "
     << "if [ -n \"$owner_pid\" ] && kill -0 \"$owner_pid\" >/dev/null 2>&1 && "
     << "tr '\\0' ' ' < \"/proc/$owner_pid/cmdline\" 2>/dev/null | " << cmdline_match << "; then "
     << "if terminate_launched; then echo queue_busy; exit 9; fi; "
     << "echo queue_busy_cleanup_failed; exit 17; fi; " << "if child_exited; then "
     << "wait \"$launched_pid\"; child_rc=$?; "
     << "echo builder_exited_before_queue_claim:$child_rc; exit " << kExitBeforeQueueClaim
     << "; fi; " << "sleep 0.05; " << "done; "
     << "if terminate_launched; then echo queue_claim_timeout; exit 16; fi; "
     << "echo queue_claim_cleanup_failed; exit 18";
  return ss.str();
}

std::string RemoteRuntime::build_stop_command(const int queue, const int expected_pid) const {
  const std::string cmdline_match = cmdline_grep(card_program());
  std::ostringstream ss;
  ss << "expected_pid=" << expected_pid << "; " << "pid=''; " << "if [ -f "
     << SshRunner::shell_escape(pid_path(queue)) << " ]; then " << "pid=$(cat "
     << SshRunner::shell_escape(pid_path(queue)) << "); " << "elif [ -f "
     << SshRunner::shell_escape(status_path(queue)) << " ]; then "
     << "pid=$(sed -n 's/.*\"pid\"[[:space:]]*:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p' "
     << SshRunner::shell_escape(status_path(queue)) << " | head -n1); " << "fi; "
     << "if [ -n \"$pid\" ] && [ \"$pid\" != \"$expected_pid\" ]; then exit 0; fi; "
     << "if [ -n \"$pid\" ]; then " << "kill -0 \"$pid\" >/dev/null 2>&1 || { rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; }; "
     << "tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | " << cmdline_match
     << " || "
        "exit 0; "
     << "kill -TERM \"$pid\" >/dev/null 2>&1 || true; "
     << "for i in $(seq 1 20); do kill -0 \"$pid\" >/dev/null 2>&1 || { rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; }; sleep 0.25; done; "
     << "echo still_running_after_sigterm; exit 13; " << "fi";
  return ss.str();
}

std::string RemoteRuntime::build_stop_launched_pid_command(const int launched_pid) const {
  const std::string cmdline_match = cmdline_grep(card_program());
  std::ostringstream ss;
  // No queue pid file to consult: this pid failed to claim a queue. Guard the
  // kill with the same /proc/<pid>/cmdline check as build_stop_command(), so a
  // reused pid that is now some other program is left untouched.
  // A zombie (dead, not yet reaped) counts as gone, as in child_exited().
  ss << "pid=" << launched_pid << "; " << "gone() { ! kill -0 \"$pid\" >/dev/null 2>&1 || "
     << "grep -q '^State:[[:space:]]*Z' \"/proc/$pid/status\" 2>/dev/null; }; "
     << "gone && exit 0; " << "tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | "
     << cmdline_match << " || exit 0; " << "kill -TERM \"$pid\" >/dev/null 2>&1 || true; "
     << "for i in $(seq 1 20); do gone && exit 0; sleep 0.25; done; "
     << "kill -KILL \"$pid\" >/dev/null 2>&1 || true; exit 0";
  return ss.str();
}

std::string RemoteRuntime::unique_remote_upload_path(const std::string& local_path) {
  static std::atomic<std::uint64_t> sequence{0};
  const fs::path local(local_path);
  const std::string filename = local.filename().string();
  if (filename.empty()) {
    throw std::invalid_argument("local upload path must have a filename");
  }
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string(kRemoteModelDir) + "/sima-neat-pcie-" +
         std::to_string(static_cast<long long>(getpid())) + "-" + std::to_string(timestamp) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + "-" + filename;
}

bool RemoteRuntime::is_managed_upload_path(const std::string& remote_path) {
  const fs::path path(remote_path);
  const std::string filename = path.filename().string();
  return path == path.lexically_normal() && path.parent_path() == kRemoteModelDir &&
         filename.rfind("sima-neat-pcie-", 0) == 0 &&
         filename.size() > std::string("sima-neat-pcie-").size();
}

std::string RemoteRuntime::start_failure_message(const std::string& program,
                                                 const CommandResult& result,
                                                 const RemoteStatus& status) {
  const std::string plain =
      "remote " + program + " start failed (exit=" + std::to_string(result.exit_code) +
      ", timed_out=" + (result.timed_out ? "true" : "false") + "): " + result.output;
  if (result.timed_out || result.exit_code != kExitBeforeQueueClaim || status.state != "failed" ||
      status.message.empty()) {
    return plain;
  }
  try {
    if (!status_owner_matches(status, parse_launched_pid(result.output))) {
      return plain; // a status left by another, older run
    }
  } catch (const std::exception&) {
    return plain;
  }
  return "remote " + program + " failed at start: " + status.message;
}

int RemoteRuntime::parse_launched_pid(const std::string& output) {
  constexpr const char* marker = "launched_pid=";
  const std::size_t marker_pos = output.rfind(marker);
  if (marker_pos == std::string::npos) {
    throw std::runtime_error("remote start output does not contain a launched PID");
  }
  const std::size_t begin = marker_pos + std::char_traits<char>::length(marker);
  std::size_t end = begin;
  while (end < output.size() && std::isdigit(static_cast<unsigned char>(output[end]))) {
    ++end;
  }
  if (end == begin) {
    throw std::runtime_error("remote start output contains an invalid launched PID");
  }
  const long long parsed = std::stoll(output.substr(begin, end - begin));
  if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error("remote start output contains an out-of-range launched PID");
  }
  return static_cast<int>(parsed);
}

bool RemoteRuntime::status_owner_matches(const RemoteStatus& status, const int expected_pid) {
  return status.state.empty() || status.state == "malformed" || status.pid == expected_pid;
}

bool RemoteRuntime::start_failure_cleanup_safe(const int exit_code, const bool timed_out) {
  if (timed_out) {
    return false;
  }
  switch (exit_code) {
  case 9:  // queue busy before launch, or launched child terminated and reaped
  case 10: // missing builder, before launch
  case 11: // missing runtime directory, before launch
  case 12: // missing log directory, before launch
  case 14: // start-lock timeout, before launch
  case 15: // launched child exited and was reaped
  case 16: // queue-claim timeout after launched child was terminated and reaped
    return true;
  default:
    return false;
  }
}

std::vector<std::string> RemoteRuntime::ssh_base() const {
  std::vector<std::string> cmd;
  cmd.push_back("ssh");
  cmd.insert(cmd.end(), {"-p", std::to_string(kSshPort)});
  if (const auto identity = default_identity_file(); identity.has_value()) {
    cmd.insert(cmd.end(), {"-i", *identity});
  }
  cmd.insert(cmd.end(), {"-o", "BatchMode=yes"}); // fail fast, never prompt for a password
  cmd.insert(cmd.end(), {"-o", "StrictHostKeyChecking=accept-new"});
  cmd.insert(cmd.end(), {"-o", "ConnectTimeout=" + std::to_string(kConnectTimeoutSec)});
  cmd.push_back(endpoint());
  return cmd;
}

std::vector<std::string> RemoteRuntime::scp_base() const {
  std::vector<std::string> cmd;
  cmd.push_back("scp");
  cmd.insert(cmd.end(), {"-P", std::to_string(kSshPort)});
  if (const auto identity = default_identity_file(); identity.has_value()) {
    cmd.insert(cmd.end(), {"-i", *identity});
  }
  cmd.insert(cmd.end(), {"-o", "BatchMode=yes"}); // fail fast, never prompt for a password
  cmd.insert(cmd.end(), {"-o", "StrictHostKeyChecking=accept-new"});
  cmd.insert(cmd.end(), {"-o", "ConnectTimeout=" + std::to_string(kConnectTimeoutSec)});
  return cmd;
}

void RemoteRuntime::run_or_throw(const std::vector<std::string>& args, const int timeout_sec,
                                 const std::string& context) const {
  const CommandResult res = SshRunner::run(args, timeout_sec);
  if (res.timed_out || res.exit_code != 0) {
    throw std::runtime_error(context + " failed (exit=" + std::to_string(res.exit_code) +
                             ", timed_out=" + (res.timed_out ? "true" : "false") +
                             "): " + res.output);
  }
}

std::string RemoteRuntime::upload_file(const std::string& local_path) const {
  const fs::path local(local_path);
  if (!fs::exists(local)) {
    throw std::runtime_error("local file does not exist: " + local_path);
  }
  const fs::path absolute_local = fs::absolute(local).lexically_normal();

  {
    std::vector<std::string> mkdir_cmd = ssh_base();
    mkdir_cmd.push_back("mkdir -p " + SshRunner::shell_escape(kRemoteModelDir));
    run_or_throw(mkdir_cmd, kCommandTimeoutSec, "remote mkdir");
  }

  const std::string remote_path = unique_remote_upload_path(absolute_local.string());
  std::vector<std::string> scp_cmd = scp_base();
  scp_cmd.push_back(absolute_local.string());
  scp_cmd.push_back(endpoint() + ":" + remote_path);
  run_or_throw(scp_cmd, kCommandTimeoutSec + 30, "scp upload");
  return remote_path;
}

int RemoteRuntime::start(const int queue, const std::string& remote_model_path,
                         const std::optional<std::string>& remote_model_options_path) const {
  const std::string program = card_program();
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(build_start_command(queue, remote_model_path, remote_model_options_path));
  CommandResult result;
  try {
    result = SshRunner::run(cmd, kCommandTimeoutSec);
  } catch (const std::exception& e) {
    throw RemoteStartError("remote " + program + " start failed: " + e.what(), true);
  }
  if (result.timed_out || result.exit_code != 0) {
    std::optional<int> launched_pid;
    try {
      launched_pid = parse_launched_pid(result.output);
    } catch (const std::exception&) {
    }
    const bool cleanup_safe = start_failure_cleanup_safe(result.exit_code, result.timed_out);
    RemoteStatus status;
    if (!result.timed_out && result.exit_code == kExitBeforeQueueClaim) {
      try {
        status = read_status(queue, std::chrono::seconds(5));
      } catch (const std::exception&) {
        // Keep the plain exit-code message.
      }
    }
    throw RemoteStartError(start_failure_message(program, result, status), cleanup_safe, cleanup_safe ? std::nullopt : launched_pid);
  }
  try {
    return parse_launched_pid(result.output);
  } catch (const std::exception& e) {
    throw RemoteStartError(
        "remote " + program + " start returned an invalid owner PID: " + e.what(), false);
  }
}

RemoteStatus RemoteRuntime::parse_status(const std::string& body, const int queue) {
  if (body.empty()) {
    return {};
  }
  try {
    const auto root = nlohmann::json::parse(body);
    RemoteStatus out;
    out.state = json_string_or(root, "state");
    out.queue = json_int_or(root, "queue", queue);
    out.pid = json_int_or(root, "pid", -1);
    out.message = json_string_or(root, "message");
    out.error_code = json_string_or(root, "error_code");
    if (root.contains("output_buffer_bytes")) {
      const auto& bytes = root.at("output_buffer_bytes");
      if (!bytes.is_number_unsigned() || bytes.get<std::uint64_t>() > 128U * 1024U * 1024U)
        throw std::runtime_error("invalid PCIe output_buffer_bytes in builder status");
      out.output_buffer_bytes = bytes.get<std::size_t>();
    }
    return out;
  } catch (const std::exception& e) {
    RemoteStatus out;
    out.queue = queue;
    out.state = "malformed";
    out.message = e.what();
    return out;
  }
}

RemoteStatus RemoteRuntime::read_status(const int queue,
                                        const std::chrono::milliseconds timeout) const {
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back("cat " + SshRunner::shell_escape(status_path(queue)) + " 2>/dev/null");
  const CommandResult res = SshRunner::run_for(cmd, timeout);
  if (res.timed_out || res.exit_code != 0 || res.output.empty()) {
    return {};
  }
  return parse_status(res.output, queue);
}

namespace {
constexpr const char* kAliveMarker = "sima_neat_pid_alive=";
} // namespace

std::string RemoteRuntime::build_ready_probe_command(const int queue,
                                                     const int expected_pid) const {
  // Status first, liveness second: a program that wrote "failed" and exited is
  // reported with its own error, not as "died". One ssh call per poll, as before.
  std::ostringstream ss;
  ss << "cat " << SshRunner::shell_escape(status_path(queue)) << " 2>/dev/null; " << "if kill -0 "
     << expected_pid << " >/dev/null 2>&1; then alive=1; else alive=0; fi; " << "echo; echo \""
     << kAliveMarker << "$alive\"";
  return ss.str();
}

ReadyProbe RemoteRuntime::parse_ready_probe(const std::string& output, const int queue) {
  ReadyProbe probe;
  const std::size_t marker = output.rfind(kAliveMarker);
  if (marker == std::string::npos) {
    return probe; // no marker (e.g. ssh noise only): status unknown, alive unknown
  }
  const std::size_t value = marker + std::char_traits<char>::length(kAliveMarker);
  probe.alive = !(value < output.size() && output[value] == '0');
  std::string body = output.substr(0, marker);
  const auto not_space = [](const unsigned char ch) { return std::isspace(ch) == 0; };
  body.erase(std::find_if(body.rbegin(), body.rend(), not_space).base(), body.end());
  body.erase(body.begin(), std::find_if(body.begin(), body.end(), not_space));
  probe.status = parse_status(body, queue);
  return probe;
}

std::string RemoteRuntime::startup_log_path(const int queue) const {
  return "/var/log/sima-neat/pcie/q" + std::to_string(queue) + ".log";
}

RemoteStatus RemoteRuntime::wait_ready(const int queue, const int expected_pid,
                                       const int readiness_timeout_ms,
                                       const std::function<bool()>& should_abort) const {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(readiness_timeout_ms);
  RemoteStatus last;

  while (std::chrono::steady_clock::now() < deadline) {
    // should_abort lets the caller (pcie-genai on Ctrl-C/SIGTERM) end a wait
    // that can take minutes. We throw, so the caller's cleanup stops the card
    // program. Checked before each poll and after each sleep.
    if (should_abort && should_abort()) {
      throw std::runtime_error("interrupted while waiting for READY");
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero()) {
      break;
    }
    std::vector<std::string> cmd = ssh_base();
    cmd.push_back(build_ready_probe_command(queue, expected_pid));
    // should_abort is passed down too: a probe that hangs (a broken link after
    // ssh connected) would otherwise keep Ctrl-C waiting until `remaining`
    // ran out, which can be the whole READY timeout.
    const CommandResult res = SshRunner::run_for(cmd, remaining, should_abort);
    if (res.aborted) {
      throw std::runtime_error("interrupted while waiting for READY");
    }
    ReadyProbe probe;
    if (!res.timed_out && res.exit_code == 0) {
      probe = parse_ready_probe(res.output, queue);
    }
    last = probe.status;
    if (!status_owner_matches(last, expected_pid)) {
      throw std::runtime_error("remote pipeline queue ownership changed while waiting for "
                               "readiness: expected pid=" +
                               std::to_string(expected_pid) +
                               " observed pid=" + std::to_string(last.pid));
    }
    if (last.state == "ready") {
      return last;
    }
    if (last.state == "failed" || last.state == "exited" || last.state == "malformed") {
      throw std::runtime_error("remote pipeline startup failed: state=" + last.state +
                               " error=" + last.error_code + " message=" + last.message);
    }
    if (!probe.alive) {
      // A dead program can never become ready: fail now, not at the timeout.
      // Example: the backend crashed or was killed during the model load and
      // had no chance to write "failed". Without this check we would wait for
      // the full readiness timeout (up to 15 minutes for pcie-genai).
      throw std::runtime_error("card program " + card_program() + " (pid " +
                               std::to_string(expected_pid) + ") exited during startup; see " +
                               startup_log_path(queue));
    }
    const auto sleep_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (sleep_remaining > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(std::min(std::chrono::milliseconds(250), sleep_remaining));
      if (should_abort && should_abort()) {
        throw std::runtime_error("interrupted while waiting for READY");
      }
    }
  }

  throw std::runtime_error("remote pipeline readiness timed out; last state=" + last.state +
                           " message=" + last.message);
}

void RemoteRuntime::stop(const int queue, const int expected_pid) const {
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(build_stop_command(queue, expected_pid));
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote " + card_program() + " stop");
}

void RemoteRuntime::stop_process(const int expected_pid,
                                 const std::string& expected_model_path) const {
  if (expected_pid <= 0) {
    throw std::invalid_argument("remote process PID must be positive");
  }
  if (!is_managed_upload_path(expected_model_path)) {
    throw std::invalid_argument("remote process model path is not a managed upload");
  }

  std::ostringstream ss;
  ss << "pid=" << expected_pid << "; " << "kill -0 \"$pid\" >/dev/null 2>&1 || exit 0; "
     << "command=$(tr '\\0' '\\n' < \"/proc/$pid/cmdline\" 2>/dev/null | head -n1); "
     << "[ \"$command\" = " << SshRunner::shell_escape(kRemoteHelper)
     << " ] || { echo unexpected_process; exit 19; }; "
     << "tr '\\0' '\\n' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -Fx -- "
     << SshRunner::shell_escape(expected_model_path)
     << " >/dev/null || { echo unexpected_model; exit 19; }; "
     << "kill -TERM \"$pid\" >/dev/null 2>&1 || true; " << "for i in $(seq 1 20); do "
     << "kill -0 \"$pid\" >/dev/null 2>&1 || exit 0; "
     << "grep -q '^State:[[:space:]]*Z' \"/proc/$pid/status\" 2>/dev/null && exit 0; "
     << "sleep 0.25; done; " << "kill -KILL \"$pid\" >/dev/null 2>&1 || true; "
     << "for i in $(seq 1 20); do " << "kill -0 \"$pid\" >/dev/null 2>&1 || exit 0; "
     << "grep -q '^State:[[:space:]]*Z' \"/proc/$pid/status\" 2>/dev/null && exit 0; "
     << "sleep 0.25; done; " << "echo still_running_after_sigkill; exit 20";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote unclaimed pcie-pipeline-builder stop");
}

void RemoteRuntime::stop_launched_pid(const int launched_pid) const {
  if (launched_pid <= 0) {
    return;
  }
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(build_stop_launched_pid_command(launched_pid));
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote " + card_program() + " stop by pid");
}

} // namespace simaai::neat::pcie::internal
