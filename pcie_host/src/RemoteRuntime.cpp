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
constexpr const char* kRemoteModelDir = "/tmp";
constexpr const char* kRemoteHelper = "/usr/bin/pcie-pipeline-builder";
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

} // namespace

RemoteRuntime::RemoteRuntime(ConnectionOptions connection) : connection_(std::move(connection)) {
  validate_endpoint_component(connection_.user, "user", false);
  validate_endpoint_component(connection_.card_host, "card_host", true);
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

void RemoteRuntime::remove_upload(const std::string& remote_path) const {
  if (!is_managed_upload_path(remote_path)) {
    throw std::invalid_argument("refusing to remove unmanaged remote upload path: " + remote_path);
  }
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back("rm -f -- " + SshRunner::shell_escape(remote_path));
  run_or_throw(cmd, kCommandTimeoutSec, "remote upload cleanup");
}

std::vector<std::string> RemoteRuntime::ssh_base() const {
  std::vector<std::string> cmd;
  cmd.push_back("ssh");
  cmd.insert(cmd.end(), {"-p", std::to_string(kSshPort)});
  if (const auto identity = default_identity_file(); identity.has_value()) {
    cmd.insert(cmd.end(), {"-i", *identity});
  }
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
  const std::string start_lock_path =
      "/run/sima-neat/pcie/q" + std::to_string(queue) + ".start.lock";
  std::ostringstream ss;
  ss << "[ -x " << SshRunner::shell_escape(kRemoteHelper)
     << " ] || { echo missing_builder; exit 10; }; "
     << "[ -d /run/sima-neat/pcie ] || { echo missing_run_dir; exit 11; }; "
     << "[ -d /var/log/sima-neat/pcie ] || { echo missing_log_dir; exit 12; }; "
     << "startlock=" << SshRunner::shell_escape(start_lock_path) << "; "
     << "exec 9>\"$startlock\"; " << "flock -w 30 9 || { echo start_lock_timeout; exit 14; }; "
     << "pidfile=" << SshRunner::shell_escape(pid_path(queue)) << "; "
     << "statusfile=" << SshRunner::shell_escape(status_path(queue)) << "; "
     << "if [ -f \"$pidfile\" ]; then " << "pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then "
     << "if tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -q 'pcie-pipeline-builder'; "
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
  ss << SshRunner::shell_escape(kRemoteHelper) << " --model "
     << SshRunner::shell_escape(remote_model_path) << " --queue " << queue;
  if (remote_model_options_path.has_value()) {
    ss << " --model-options " << SshRunner::shell_escape(*remote_model_options_path);
  }
  ss << " 9>&- >/dev/null 2>&1 & " << "launched_pid=$!; " << "for i in $(seq 1 200); do "
     << "owner_pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ \"$owner_pid\" = \"$launched_pid\" ]; then "
     << "echo \"launched_pid=$launched_pid\"; exit 0; fi; "
     << "if [ -n \"$owner_pid\" ] && kill -0 \"$owner_pid\" >/dev/null 2>&1 && "
     << "tr '\\0' ' ' < \"/proc/$owner_pid/cmdline\" 2>/dev/null | "
        "grep -q 'pcie-pipeline-builder'; then echo queue_busy; exit 9; fi; "
     << "if ! kill -0 \"$launched_pid\" >/dev/null 2>&1; then "
     << "wait \"$launched_pid\"; child_rc=$?; "
     << "echo builder_exited_before_queue_claim:$child_rc; exit 15; fi; " << "sleep 0.05; "
     << "done; " << "kill -TERM \"$launched_pid\" >/dev/null 2>&1 || true; "
     << "echo queue_claim_timeout; exit 16";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  CommandResult result;
  try {
    result = SshRunner::run(cmd, kCommandTimeoutSec);
  } catch (const std::exception& e) {
    throw RemoteStartError(std::string("remote pcie-pipeline-builder start failed: ") + e.what(),
                           true);
  }
  if (result.timed_out || result.exit_code != 0) {
    throw RemoteStartError(
        "remote pcie-pipeline-builder start failed (exit=" + std::to_string(result.exit_code) +
            ", timed_out=" + (result.timed_out ? "true" : "false") + "): " + result.output,
        !result.timed_out);
  }
  try {
    return parse_launched_pid(result.output);
  } catch (const std::exception& e) {
    throw RemoteStartError(std::string("remote pcie-pipeline-builder start returned an invalid "
                                       "owner PID: ") +
                               e.what(),
                           false);
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

  try {
    const auto root = nlohmann::json::parse(res.output);
    RemoteStatus out;
    out.state = json_string_or(root, "state");
    out.queue = json_int_or(root, "queue", queue);
    out.pid = json_int_or(root, "pid", -1);
    out.message = json_string_or(root, "message");
    out.error_code = json_string_or(root, "error_code");
    return out;
  } catch (const std::exception& e) {
    RemoteStatus out;
    out.queue = queue;
    out.state = "malformed";
    out.message = e.what();
    return out;
  }
}

RemoteStatus RemoteRuntime::wait_ready(const int queue, const int expected_pid,
                                       const int readiness_timeout_ms) const {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(readiness_timeout_ms);
  RemoteStatus last;

  while (std::chrono::steady_clock::now() < deadline) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero()) {
      break;
    }
    last = read_status(queue, remaining);
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
    const auto sleep_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (sleep_remaining > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(std::min(std::chrono::milliseconds(250), sleep_remaining));
    }
  }

  throw std::runtime_error("remote pipeline readiness timed out; last state=" + last.state +
                           " message=" + last.message);
}

void RemoteRuntime::stop(const int queue, const int expected_pid) const {
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
     << "tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -q 'pcie-pipeline-builder' || "
        "exit 0; "
     << "kill -TERM \"$pid\" >/dev/null 2>&1 || true; "
     << "for i in $(seq 1 20); do kill -0 \"$pid\" >/dev/null 2>&1 || { rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; }; sleep 0.25; done; "
     << "echo still_running_after_sigterm; exit 13; " << "fi";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote pcie-pipeline-builder stop");
}

} // namespace simaai::neat::pcie::internal
