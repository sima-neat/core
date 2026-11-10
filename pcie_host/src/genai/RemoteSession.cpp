#include "genai/RemoteSession.h"
#include "CardEnvironment.h"
#include "HostPlatform.h"
#include "SshRunner.h"
#include "Protocol.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace simaai::neat::pcie::genai::internal {
using Runner = simaai::neat::pcie::internal::SshRunner;
using simaai::neat::pcie::internal::card_environment_path_or;
using simaai::neat::pcie::internal::host_home_directory;
using simaai::neat::pcie::internal::host_path_string;
using simaai::neat::pcie::internal::host_secure_random;
using simaai::neat::pcie::internal::open_ssh_executable;
using simaai::neat::pcie::internal::parse_card_environment;
namespace {
constexpr const char* kRemoteWorker = "/usr/bin/neat-pcie-genai-worker";
constexpr const char* kWorkerPathEnvironment = "SIMA_NEAT_PCIE_GENAI_HELPER_PATH";
constexpr const char* kCardEnvironment = "SIMA_NEAT_PCIE_GENAI_CARD_ENV";
} // namespace
RemoteSession::RemoteSession(ConnectionOptions options) : options_(std::move(options)) {
  if (options_.card_host.empty())
    options_.card_host = "10.0." + std::to_string(options_.card_id) + ".2";
  auto endpoint_ok = [](const std::string& s) {
    return !s.empty() && s.front() != '-' && s.find_first_of(" \t\r\n@") == std::string::npos &&
           s.find('\0') == std::string::npos;
  };
  if (options_.card_id < 0 || !endpoint_ok(options_.card_host) || !endpoint_ok(options_.user) ||
      options_.user.find(':') != std::string::npos || options_.startup_timeout_ms <= 0 ||
      options_.request_timeout_ms <= 0)
    throw std::invalid_argument("Invalid GenAI connection options");
  // Qualification can select the exact staged card worker without changing the
  // public GenAI ConnectionOptions layout used by existing Linux applications.
  if (const auto* value = std::getenv(kCardEnvironment))
    card_environment_ = value;
  const auto card_environment = parse_card_environment(card_environment_);
  (void)card_environment_path_or(card_environment, kWorkerPathEnvironment, kRemoteWorker);
  std::array<unsigned char, 12> bytes{};
  host_secure_random(bytes.data(), bytes.size());
  std::ostringstream out;
  for (auto b : bytes)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
  id_ = out.str();
}
RemoteSession::~RemoteSession() {
  try {
    stop();
  } catch (...) {
  }
}
std::vector<std::string> RemoteSession::ssh(const std::string& script) const {
  std::vector<std::string> args{
      open_ssh_executable("ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-p", "22"};
  std::string key = options_.ssh_key;
  if (key.empty()) {
    if (const auto home = host_home_directory(); home.has_value()) {
      auto candidate = *home / ".ssh/sima_neat_pcie_ed25519";
      if (std::filesystem::exists(candidate))
        key = host_path_string(candidate);
    }
  }
  if (!key.empty()) {
    args.push_back("-i");
    args.push_back(key);
  }
  args.push_back(options_.user + "@" + options_.card_host);
  args.push_back("sh -c " + Runner::shell_escape(script));
  return args;
}
void RemoteSession::start(const std::string& target_directory, const std::string& draft_directory) {
  if (!target_directory.empty())
    wire::relative_name(target_directory);
  if (!draft_directory.empty())
    wire::relative_name(draft_directory);
  const auto q = Runner::shell_escape;
  const auto card_environment = parse_card_environment(card_environment_);
  const auto worker =
      card_environment_path_or(card_environment, kWorkerPathEnvironment, kRemoteWorker);
  // Session-specific directory is also the ownership record if SSH disconnects
  // before returning the PID. No tensor qN.pid files are used.
  std::string script = "set -eu; umask 077; d=\"$HOME/.cache/neat-genai/" + id_ +
                       "\"; mkdir -p \"$HOME/.cache/neat-genai\"; mkdir \"$d\"; "
                       "nohup ";
  if (!card_environment.empty()) {
    script += "env ";
    for (const auto& entry : card_environment)
      script += q(entry) + " ";
  }
  script += q(worker) + " " + q(id_) + " " + q("/tmp") + " " +
            q(std::to_string(std::max(options_.startup_timeout_ms, options_.request_timeout_ms))) +
            " " + q(target_directory) + " " + q(draft_directory) +
            " >\"$d/worker.log\" 2>&1 </dev/null & p=$!; echo \"$p\" >\"$d/pid\"; "
            "awk '{print $22}' /proc/$p/stat >\"$d/start\"; echo launched";
  launched_ = true;
  const auto result = Runner::run(ssh(script), 15);
  if (result.exit_code || result.timed_out)
    throw std::runtime_error("Could not launch GenAI worker: " + result.output);
}
std::string RemoteSession::stop_script(const std::string& session_id) {
  wire::validate_session(session_id);
  // The PCIe close message may make the worker exit between owned() and kill.
  // Only tolerate a failed kill when the session no longer owns that process.
  return "set -eu; d=\"$HOME/.cache/neat-genai/" + session_id +
         "\"; test -f \"$d/pid\" || exit 0; p=$(cat \"$d/pid\"); "
         "case $p in ''|*[!0-9]*) exit 1;; esac; "
         "owned() { test -r /proc/$p/stat && test -f \"$d/start\" && "
         "test \"$(awk '{print $22}' /proc/$p/stat)\" = \"$(cat \"$d/start\")\" && "
         "tr '\\0' '\\n' </proc/$p/cmdline | grep -Fxq " +
         Runner::shell_escape(session_id) +
         "; }; if owned; then kill -TERM \"$p\" || { if owned; then exit 1; fi; }; "
         "i=0; while owned && test $i -lt 50; do sleep 0.1; i=$((i+1)); done; "
         "if owned; then kill -KILL \"$p\" || { if owned; then exit 1; fi; }; "
         "i=0; while owned && test $i -lt 10; do sleep 0.1; i=$((i+1)); done; fi; fi; "
         "if owned; then exit 1; fi; rm -rf -- \"$d\"";
}
void RemoteSession::stop() {
  if (!launched_)
    return;
  const auto script = stop_script(id_);
  const auto result = Runner::run(ssh(script), 15);
  if (result.exit_code || result.timed_out)
    throw std::runtime_error(
        "GenAI worker shutdown could not be confirmed (exit=" + std::to_string(result.exit_code) +
        ", timed_out=" + (result.timed_out ? "true" : "false") + "): " + result.output);
  launched_ = false;
}
} // namespace simaai::neat::pcie::genai::internal
