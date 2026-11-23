#include "RemoteRuntime.h"

#include "CardEnvironment.h"
#include "HostPlatform.h"
#include "SshRunner.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
constexpr const char* kRemoteRunDir = "/run/sima-neat/pcie";
constexpr const char* kRemoteLogDir = "/var/log/sima-neat/pcie";
constexpr const char* kDefaultIdentityFile = "sima_neat_pcie_ed25519";
constexpr const char* kDefaultMlashmCtrlIoTimeout = "MLASHM_CTRL_IO_TIMEOUT_MS=5000";
constexpr const char* kHelperPathEnv = "SIMA_NEAT_PCIE_HELPER_PATH";
constexpr const char* kRunDirEnv = "SIMA_NEAT_PCIE_RUN_DIR";
constexpr const char* kLogDirEnv = "SIMA_NEAT_PCIE_LOG_DIR";
constexpr const char* kInstallOnCard = "Install the matching Neat release on the card (see Install "
                                       "Neat on the card).";

std::optional<std::string> default_identity_file() {
  const auto home = host_home_directory();
  if (!home.has_value()) {
    return std::nullopt;
  }
  // Joined per component, so that the printed path uses the host's own separators.
  std::filesystem::path path = *home / ".ssh" / kDefaultIdentityFile;
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return std::nullopt;
  }
  return host_path_string(path);
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

// The major.minor release of a Neat version such as 0.6.2 or 0.6.2+develop.1a2b3c, or "".
std::string neat_minor_release(const std::string& version) {
  std::size_t end = 0;
  for (int part = 0; part < 2; ++part) {
    const std::size_t begin = end;
    while (end < version.size() && std::isdigit(static_cast<unsigned char>(version[end]))) {
      ++end;
    }
    if (end == begin || (part == 0 && (end == version.size() || version[end++] != '.'))) {
      return "";
    }
  }
  return version.substr(0, end);
}

std::string output_value(const std::string& output, const std::string& key) {
  std::istringstream lines(output);
  for (std::string line; std::getline(lines, line);) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.rfind(key + "=", 0) == 0) {
      return line.substr(key.size() + 1);
    }
  }
  return "";
}

} // namespace

RemoteRuntime::RemoteRuntime(ConnectionOptions connection) : connection_(std::move(connection)) {
  validate_endpoint_component(connection_.user, "user", false);
  validate_endpoint_component(connection_.card_host, "card_host", true);
  const auto card_env = parse_card_environment(connection_.card_env);
  helper_path_ = card_environment_path_or(card_env, kHelperPathEnv, kRemoteHelper);
  // Card-side liveness checks match the worker by this file name.
  if (helper_path_.substr(helper_path_.rfind('/') + 1) != "pcie-pipeline-builder") {
    throw std::invalid_argument(std::string(kHelperPathEnv) +
                                " must name a pcie-pipeline-builder executable");
  }
  run_dir_ = card_environment_path_or(card_env, kRunDirEnv, kRemoteRunDir);
  log_dir_ = card_environment_path_or(card_env, kLogDirEnv, kRemoteLogDir);
}

std::string RemoteRuntime::endpoint() const {
  return connection_.user + "@" + derive_card_host(connection_);
}

std::string RemoteRuntime::status_path(const int queue) const {
  return run_dir_ + "/q" + std::to_string(queue) + ".status";
}

std::string RemoteRuntime::pid_path(const int queue) const {
  return run_dir_ + "/q" + std::to_string(queue) + ".pid";
}

std::string RemoteRuntime::owner_path(const int queue) const {
  return run_dir_ + "/q" + std::to_string(queue) + ".owner";
}

std::string RemoteRuntime::host_neat_version() {
#ifdef SIMA_NEAT_PCIE_HOST_VERSION
  return SIMA_NEAT_PCIE_HOST_VERSION;
#else
  return "unknown";
#endif
}

std::optional<QueueOwner> RemoteRuntime::parse_queue_owner(const std::string& output) {
  constexpr std::string_view prefix = "owner.";
  QueueOwner owner;
  std::istringstream lines(output);
  for (std::string line; std::getline(lines, line);) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto equals = line.find('=');
    if (line.rfind(prefix, 0) != 0 || equals == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(prefix.size(), equals - prefix.size());
    const std::string value = line.substr(equals + 1);
    try {
      if (key == "builder_pid") {
        owner.builder_pid = std::stoi(value);
      } else if (key == "host_id") {
        owner.host.host_id = value;
      } else if (key == "host") {
        owner.host.host = value;
      } else if (key == "host_pid") {
        owner.host.pid = std::stoull(value);
      } else if (key == "host_start") {
        owner.host.start = value;
      } else if (key == "upload") {
        owner.uploads.push_back(value);
      }
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  if (owner.builder_pid <= 0 || owner.host.pid == 0) {
    return std::nullopt;
  }
  return owner;
}

std::string RemoteRuntime::unique_remote_upload_path(const std::string& local_path) {
  static std::atomic<std::uint64_t> sequence{0};
  const fs::path local = host_path_from_string(local_path);
  const std::string filename = host_path_string(local.filename());
  if (filename.empty()) {
    throw std::invalid_argument("local upload path must have a filename");
  }
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string(kRemoteModelDir) + "/sima-neat-pcie-" + std::to_string(host_process_id()) +
         "-" + std::to_string(timestamp) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + "-" + filename;
}

bool RemoteRuntime::is_managed_upload_path(const std::string& remote_path) {
  const fs::path path = host_path_from_string(remote_path);
  const std::string filename = host_path_string(path.filename());
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
  case 21: // card Neat release differs from the host's, before launch
    return true;
  default:
    return false;
  }
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

std::string RemoteRuntime::detached_child_redirections() {
  // Closing the lock descriptor is not sufficient to detach the worker from
  // the SSH channel. An inherited stdin can keep the OpenSSH client alive after
  // the launcher shell exits.
  return " 9>&- </dev/null >/dev/null 2>&1 & ";
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
  cmd.push_back(open_ssh_executable("ssh"));
  cmd.insert(cmd.end(), {"-p", std::to_string(kSshPort)});
  if (const auto identity = default_identity_file(); identity.has_value()) {
    cmd.insert(cmd.end(), {"-i", *identity});
#ifdef _WIN32
    // Windows only: the ssh-agent service commonly offers several keys before the Neat key.
    cmd.insert(cmd.end(), {"-o", "IdentitiesOnly=yes"});
#endif
  }
  cmd.insert(cmd.end(), {"-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"});
  cmd.insert(cmd.end(), {"-o", "ConnectTimeout=" + std::to_string(kConnectTimeoutSec)});
  cmd.push_back(endpoint());
  return cmd;
}

std::vector<std::string> RemoteRuntime::scp_base() const {
  std::vector<std::string> cmd;
  cmd.push_back(open_ssh_executable("scp"));
  cmd.insert(cmd.end(), {"-P", std::to_string(kSshPort)});
  if (const auto identity = default_identity_file(); identity.has_value()) {
    cmd.insert(cmd.end(), {"-i", *identity});
#ifdef _WIN32
    // Windows only: the ssh-agent service commonly offers several keys before the Neat key.
    cmd.insert(cmd.end(), {"-o", "IdentitiesOnly=yes"});
#endif
  }
  cmd.insert(cmd.end(), {"-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"});
  cmd.insert(cmd.end(), {"-o", "ConnectTimeout=" + std::to_string(kConnectTimeoutSec)});
  return cmd;
}

void RemoteRuntime::run_or_throw(const std::vector<std::string>& args, const int timeout_sec,
                                 const std::string& context) const {
  const CommandResult res = SshRunner::run(args, timeout_sec);
  if (res.timed_out || res.exit_code != 0) {
    throw std::runtime_error(context + " failed " + SshRunner::failure_detail(args, res));
  }
}

std::string RemoteRuntime::upload_file(const std::string& local_path) const {
  const fs::path local = host_path_from_string(local_path);
  if (!fs::exists(local)) {
    throw std::runtime_error("local file does not exist: " + local_path);
  }
  const fs::path absolute_local = fs::absolute(local).lexically_normal();

  {
    std::vector<std::string> mkdir_cmd = ssh_base();
    mkdir_cmd.push_back("mkdir -p " + SshRunner::shell_escape(kRemoteModelDir));
    run_or_throw(mkdir_cmd, kCommandTimeoutSec, "remote mkdir");
  }

  const std::string remote_path = unique_remote_upload_path(host_path_string(absolute_local));
  std::vector<std::string> scp_cmd = scp_base();
  scp_cmd.push_back(host_path_string(absolute_local));
  scp_cmd.push_back(endpoint() + ":" + remote_path);
  run_or_throw(scp_cmd, kCommandTimeoutSec + 30, "scp upload");
  return remote_path;
}

std::string RemoteRuntime::start_failure_remedy(const std::string& output, const int queue) const {
  const std::string q = std::to_string(queue);
  // Name the Neat key as the docs do: where only it is enrolled, plain ssh asks for a password.
  const auto identity = default_identity_file();
  const std::string stop_worker =
      "ssh " + (identity ? "-i \"" + *identity + "\" " : std::string()) + endpoint() +
      " \"pkill -f '[p]cie-pipeline-builder.*--queue " + q + "( |$)'\"";
  std::istringstream lines(output);
  for (std::string line; std::getline(lines, line);) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line == "missing_builder") {
      return "\nNeat is not installed on the card at " + endpoint() +
             ". Install the Neat release that matches this host package (Neat " +
             host_neat_version() + ") on the card (see Install Neat on the card).";
    }
    if (line == "neat_version_mismatch") {
      return "\nThe card at " + endpoint() + " runs Neat " +
             output_value(output, "card_neat_version") + " and this host runs Neat " +
             host_neat_version() + "; the card needs the same Neat minor release (" +
             neat_minor_release(host_neat_version()) + ") as the host. " + kInstallOnCard;
    }
    if (line == "missing_run_dir") {
      return "\nThe card's Neat runtime directory " + run_dir_ +
             " is missing. Install Neat on the card, or restart the card to recreate it.";
    }
    if (line == "queue_busy") {
      const auto owner = parse_queue_owner(output);
      if (!owner.has_value()) {
        return "\nAnother card worker holds PCIe queue " + q +
               ". If no application uses this queue, for example after a host application "
               "ended without closing its model, stop the leftover worker and retry: " +
               stop_worker;
      }
      const std::string worker = " (card worker PID " + std::to_string(owner->builder_pid) + ")";
      const std::string process = "process " + std::to_string(owner->host.pid);
      if (owner->host.host_id.empty() ||
          owner->host.host_id != current_host_process_owner().host_id) {
        return "\nPCIe queue " + q + " is in use by " + process + " on host " + owner->host.host +
               worker +
               ". Close the model in that application or choose another queue. If that "
               "application has ended, stop the leftover worker and retry: " +
               stop_worker;
      }
      if (host_process_ended(owner->host)) {
        return "\nPCIe queue " + q + " is held by a leftover worker" + worker + " of " + process +
               " on this host, which has ended. Stop the leftover worker and retry: " + stop_worker;
      }
      return "\nPCIe queue " + q + " is in use by " + process + " on this host (" +
             owner->host.host + ")" + worker +
             ", which is still running. Close the model in that application, end the "
             "application, or choose another queue. If that application has ended, stop the "
             "leftover worker and retry: " +
             stop_worker;
    }
  }
  return "";
}

std::vector<std::string>
RemoteRuntime::start_command(const int queue, const std::string& remote_model_path,
                             const std::optional<std::string>& remote_model_options_path) const {
  const std::string& helper = helper_path_;
  const std::string& resolved_run_dir = run_dir_;
  const std::string& resolved_log_dir = log_dir_;
  const std::string start_lock_path =
      resolved_run_dir + "/q" + std::to_string(queue) + ".start.lock";
  std::vector<std::string> card_env = parse_card_environment(connection_.card_env);
  if (!card_environment_contains(card_env, "MLASHM_CTRL_IO_TIMEOUT_MS")) {
    card_env.emplace_back(kDefaultMlashmCtrlIoTimeout);
  }
  std::string helper_env = "env ";
  for (const auto& entry : card_env) {
    helper_env += SshRunner::shell_escape(entry) + " ";
  }

  // Compare the card's package version, which like the host's comes from deps/manifest.json
  // (sima-neat-version is a tag name on tag builds). Builders that print no package version
  // report nothing here, and the start goes ahead as before. Patch releases may differ.
  std::ostringstream version_check;
  version_check << "card_version=$(" << helper_env << SshRunner::shell_escape(helper)
                << " --version 2>/dev/null | "
                << "sed -n 's/^sima-neat-package-version=//p' | head -n 1); "
                << "[ -z \"$card_version\" ] || echo \"card_neat_version=$card_version\"; ";
  if (const std::string release = neat_minor_release(host_neat_version()); !release.empty()) {
    const std::string quoted = SshRunner::shell_escape(release);
    version_check << "case \"$card_version\" in ''|[!0-9]*|" << quoted << "|" << quoted
                  << "[!0-9]*) ;; *) echo neat_version_mismatch; exit 21;; esac; ";
  }

  // Record this process with the claim, so that a later start can tell whether the worker was
  // left by a process that has ended. Without a record, a busy queue is reported as before.
  const HostProcessOwner self = current_host_process_owner();
  std::vector<std::string> record = {"host_id=" + self.host_id, "host=" + self.host,
                                     "host_pid=" + std::to_string(self.pid),
                                     "host_start=" + self.start, "upload=" + remote_model_path};
  if (remote_model_options_path.has_value()) {
    record.push_back("upload=" + *remote_model_options_path);
  }
  std::string write_owner;
  if (!self.host_id.empty() && std::none_of(record.begin(), record.end(), [](const auto& line) {
        return line.find_first_of("\r\n") != std::string::npos;
      })) {
    write_owner = "{ echo \"builder_pid=$launched_pid\"; printf '%s\\n'";
    for (const auto& line : record) {
      write_owner += " " + SshRunner::shell_escape(line);
    }
    write_owner += "; } > \"$ownerfile.tmp\" 2>/dev/null && "
                   "mv -f \"$ownerfile.tmp\" \"$ownerfile\" 2>/dev/null; ";
  }

  std::ostringstream ss;
  ss << child_cleanup_shell_function() << "[ -x " << SshRunner::shell_escape(helper)
     << " ] || { echo missing_builder; exit 10; }; " << version_check.str() << "[ -d "
     << SshRunner::shell_escape(resolved_run_dir) << " ] || { echo missing_run_dir; exit 11; }; "
     << "[ -d " << SshRunner::shell_escape(resolved_log_dir)
     << " ] || { echo missing_log_dir; exit 12; }; "
     << "startlock=" << SshRunner::shell_escape(start_lock_path) << "; "
     << "exec 9>\"$startlock\"; " << "flock -w 30 9 || { echo start_lock_timeout; exit 14; }; "
     << "pidfile=" << SshRunner::shell_escape(pid_path(queue)) << "; "
     << "statusfile=" << SshRunner::shell_escape(status_path(queue)) << "; "
     << "ownerfile=" << SshRunner::shell_escape(owner_path(queue)) << "; "
     << "print_owner() { grep -qx \"builder_pid=$1\" \"$ownerfile\" 2>/dev/null && "
        "sed 's/^/owner./' \"$ownerfile\"; }; "
     << "if [ -f \"$pidfile\" ]; then " << "pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ -n \"$pid\" ] && kill -0 \"$pid\" >/dev/null 2>&1; then "
     << "if tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -q 'pcie-pipeline-builder'; "
        "then "
     << "print_owner \"$pid\"; echo queue_busy; exit 9; " << "fi; " << "fi; "
     << "rm -f \"$pidfile\" \"$statusfile\" \"$ownerfile\"; " << "else "
     << "rm -f \"$statusfile\" \"$ownerfile\"; " << "fi; ";
  ss << "nohup ";
  ss << helper_env;
  if (!connection_.card_gst_debug.empty()) {
    ss << "GST_DEBUG=" << SshRunner::shell_escape(connection_.card_gst_debug) << " ";
    ss << "GST_DEBUG_NO_COLOR=1 ";
    const std::string debug_file =
        connection_.card_gst_debug_file.empty()
            ? (resolved_log_dir + "/q" + std::to_string(queue) + ".gst.log")
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
  ss << detached_child_redirections() << "launched_pid=$!; "
     << "echo \"launched_pid=$launched_pid\"; " << "for i in $(seq 1 200); do "
     << "owner_pid=$(cat \"$pidfile\" 2>/dev/null || true); "
     << "if [ \"$owner_pid\" = \"$launched_pid\" ]; then " << write_owner << "exit 0; fi; "
     << "if [ -n \"$owner_pid\" ] && kill -0 \"$owner_pid\" >/dev/null 2>&1 && "
     << "tr '\\0' ' ' < \"/proc/$owner_pid/cmdline\" 2>/dev/null | "
        "grep -q 'pcie-pipeline-builder'; then "
     << "print_owner \"$owner_pid\"; "
     << "if terminate_launched; then echo queue_busy; exit 9; fi; "
     << "echo queue_busy_cleanup_failed; exit 17; fi; " << "if child_exited; then "
     << "wait \"$launched_pid\"; child_rc=$?; "
     << "echo builder_exited_before_queue_claim:$child_rc; exit 15; fi; " << "sleep 0.05; "
     << "done; " << "if terminate_launched; then echo queue_claim_timeout; exit 16; fi; "
     << "echo queue_claim_cleanup_failed; exit 18";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  return cmd;
}

CommandResult RemoteRuntime::run_start(const std::vector<std::string>& cmd) const {
  try {
    return SshRunner::run(cmd, kCommandTimeoutSec);
  } catch (const std::exception& e) {
    throw RemoteStartError(std::string("remote pcie-pipeline-builder start failed: ") + e.what(),
                           true);
  }
}

int RemoteRuntime::start(const int queue, const std::string& remote_model_path,
                         const std::optional<std::string>& remote_model_options_path) const {
  const std::vector<std::string> cmd =
      start_command(queue, remote_model_path, remote_model_options_path);
  CommandResult result = run_start(cmd);
  std::string reclaim_failure;
  // A worker left by an ended process of this host, for example after Ctrl+C or a crash, holds
  // the queue: stop it, remove its uploads and start once more. A worker whose recorded process
  // still runs, or was started from another host, is never stopped.
  if (!result.timed_out && result.exit_code == 9) {
    const auto owner = parse_queue_owner(result.output);
    if (owner.has_value() && host_process_ended(owner->host)) {
      try {
        if (owner->uploads.empty()) {
          throw std::runtime_error("its owner record names no model upload");
        }
        stop_process(owner->builder_pid, owner->uploads.front());
      } catch (const std::exception& e) {
        reclaim_failure = std::string("\nStopping the leftover worker failed: ") + e.what();
      }
      if (reclaim_failure.empty()) {
        for (const auto& upload : owner->uploads) {
          try {
            remove_upload(upload);
          } catch (const std::exception&) {
          }
        }
        result = run_start(cmd);
      }
    }
  }
  if (result.timed_out || result.exit_code != 0) {
    std::optional<int> launched_pid;
    try {
      launched_pid = parse_launched_pid(result.output);
    } catch (const std::exception&) {
    }
    const bool cleanup_safe = start_failure_cleanup_safe(result.exit_code, result.timed_out);
    throw RemoteStartError("remote pcie-pipeline-builder start failed " +
                               SshRunner::failure_detail(cmd, result) +
                               start_failure_remedy(result.output, queue) + reclaim_failure,
                           cleanup_safe, cleanup_safe ? std::nullopt : launched_pid);
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
     << "for i in $(seq 1 20); do if ! kill -0 \"$pid\" >/dev/null 2>&1 || "
     << "grep -q '^State:[[:space:]]*Z' \"/proc/$pid/status\" 2>/dev/null; then rm -f "
     << SshRunner::shell_escape(pid_path(queue)) << "; exit 0; fi; sleep 0.25; done; "
     << "echo still_running_after_sigterm; exit 13; " << "fi";

  std::vector<std::string> cmd = ssh_base();
  cmd.push_back(ss.str());
  run_or_throw(cmd, kCommandTimeoutSec + 10, "remote pcie-pipeline-builder stop");
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
     << "[ \"$command\" = " << SshRunner::shell_escape(helper_path_)
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

} // namespace simaai::neat::pcie::internal
