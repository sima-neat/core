#include "RemoteRuntime.h"

#include "SshRunner.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

namespace simaai::neat::pcie::internal {
namespace {

constexpr int kSshPort = 22;
constexpr int kConnectTimeoutSec = 10;
constexpr int kCommandTimeoutSec = 30;
constexpr const char* kRemoteModelDir = "/tmp";
constexpr const char* kRemoteHelper = "/usr/bin/pcie-pipeline-builder";
constexpr const char* kDefaultIdentityFile = ".ssh/sima_neat_pcie_ed25519";

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

} // namespace

RemoteRuntime::RemoteRuntime(ConnectionOptions connection) : connection_(std::move(connection)) {}

std::string RemoteRuntime::endpoint() const {
  return connection_.user + "@" + derive_card_host(connection_);
}

std::string RemoteRuntime::status_path(const int queue) const {
  return "/run/sima-neat/pcie/q" + std::to_string(queue) + ".status";
}

std::string RemoteRuntime::pid_path(const int queue) const {
  return "/run/sima-neat/pcie/q" + std::to_string(queue) + ".pid";
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

  {
    std::vector<std::string> mkdir_cmd = ssh_base();
    mkdir_cmd.push_back("mkdir -p " + SshRunner::shell_escape(kRemoteModelDir));
    run_or_throw(mkdir_cmd, kCommandTimeoutSec, "remote mkdir");
  }

  const std::string remote_path = std::string(kRemoteModelDir) + "/" + local.filename().string();
  std::vector<std::string> scp_cmd = scp_base();
  scp_cmd.push_back(local.string());
  scp_cmd.push_back(endpoint() + ":" + remote_path);
  run_or_throw(scp_cmd, kCommandTimeoutSec + 30, "scp upload");
  return remote_path;
}

void RemoteRuntime::start(const int queue, const bool accelerator,
                          const std::string& remote_model_path,
                          const std::optional<std::string>& remote_model_options_path) const {
  std::ostringstream ss;
  ss << "[ -x " << SshRunner::shell_escape(kRemoteHelper) << " ] || { echo missing_builder; exit 10; }; "
     << "[ -d /run/sima-neat/pcie ] || { echo missing_run_dir; exit 11; }; "
     << "[ -d /var/log/sima-neat/pcie ] || { echo missing_log_dir; exit 12; }; "
     << "pidfile=" << SshRunner::shell_escape(pid_path(queue)) << "; "
     << "statusfile=" << SshRunner::shell_escape(status_path(queue)) << "; "
     << "if [ -f \"$pidfile\" ]; then "
     << "pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then "
     << "if tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -q 'pcie-pipeline-builder'; then "
     << "echo queue_busy; exit 9; "
     << "fi; "
     << "fi; "
     << "rm -f \"$pidfile\" \"$statusfile\"; "
     << "else "
     << "rm -f \"$statusfile\"; "
     << "fi; ";
  ss << "nohup ";
  if (!connection_.card_gst_debug.empty()) {
    ss << "env GST_DEBUG=" << SshRunner::shell_escape(connection_.card_gst_debug) << " ";
    if (connection_.card_gst_debug_no_color) {
      ss << "GST_DEBUG_NO_COLOR=1 ";
    }
    const std::string debug_file = connection_.card_gst_debug_file.empty()
                                       ? default_card_gst_debug_file(queue)
                                       : connection_.card_gst_debug_file;
    if (!debug_file.empty()) {
      ss << "GST_DEBUG_FILE=" << SshRunner::shell_escape(debug_file) << " ";
    }
  }
  ss << SshRunner::shell_escape(kRemoteHelper)
     << " --model " << SshRunner::shell_escape(remote_model_path)
     << " --queue " << queue;
  if (accelerator) {
    ss << " --accelerator";
  }
  if (remote_model_options_path.has_value()) {
    ss << " --model-options " << SshRunner::shell_escape(*remote_model_options_path);
  }
  ss << " >/dev/null 2>&1 & echo $!";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  run_or_throw(cmd, kCommandTimeoutSec, "remote pcie-pipeline-builder start");
}

RemoteStatus RemoteRuntime::read_status(const int queue) const {
  std::vector<std::string> cmd = ssh_base();
  cmd.push_back("cat " + SshRunner::shell_escape(status_path(queue)) + " 2>/dev/null");
  const CommandResult res = SshRunner::run(cmd, kCommandTimeoutSec);
  if (res.timed_out || res.exit_code != 0 || res.output.empty()) {
    return {};
  }

  try {
    const auto root = nlohmann::json::parse(res.output);
    RemoteStatus out;
    out.state = json_string_or(root, "state");
    out.queue = json_int_or(root, "queue", queue);
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

RemoteStatus RemoteRuntime::wait_ready(const int queue, const int readiness_timeout_ms) const {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(readiness_timeout_ms);
  RemoteStatus last;

  while (std::chrono::steady_clock::now() < deadline) {
    last = read_status(queue);
    if (last.state == "ready") {
      return last;
    }
    if (last.state == "failed" || last.state == "exited" || last.state == "malformed") {
      throw std::runtime_error("remote pipeline startup failed: state=" + last.state +
                               " error=" + last.error_code + " message=" + last.message);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }

  throw std::runtime_error("remote pipeline readiness timed out; last state=" + last.state +
                           " message=" + last.message);
}

void RemoteRuntime::stop(const int queue) const {
  std::ostringstream ss;
  ss << "pid=''; "
     << "if [ -f " << SshRunner::shell_escape(pid_path(queue)) << " ]; then "
     << "pid=$(cat " << SshRunner::shell_escape(pid_path(queue)) << "); "
     << "elif [ -f " << SshRunner::shell_escape(status_path(queue)) << " ]; then "
     << "pid=$(sed -n 's/.*\"pid\"[[:space:]]*:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p' "
     << SshRunner::shell_escape(status_path(queue)) << " | head -n1); "
     << "fi; "
     << "if [ -n \"$pid\" ]; then "
     << "kill -0 \"$pid\" >/dev/null 2>&1 || { rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; }; "
     << "tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -q 'pcie-pipeline-builder' || exit 0; "
     << "kill -TERM \"$pid\" >/dev/null 2>&1 || true; "
     << "for i in $(seq 1 20); do kill -0 \"$pid\" >/dev/null 2>&1 || { rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; }; sleep 0.25; done; "
     << "echo still_running_after_sigterm; exit 13; "
     << "fi";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote pcie-pipeline-builder stop");
}

} // namespace simaai::neat::pcie::internal
