#include "genai/RemoteSession.h"
#include "SshRunner.h"
#include "Protocol.h"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace simaai::neat::pcie::genai::internal {
using Runner = simaai::neat::pcie::internal::SshRunner;
RemoteSession::RemoteSession(std::string model, ConnectionOptions options)
    : model_(wire::relative_name(model)), options_(std::move(options)) {
  auto endpoint_ok = [](const std::string& s) {
    return !s.empty() && s.front() != '-' && s.find_first_of(" \t\r\n@") == std::string::npos &&
           s.find('\0') == std::string::npos;
  };
  if (options_.card_id < 0 || !endpoint_ok(options_.card_host) || !endpoint_ok(options_.user) ||
      options_.user.find(':') != std::string::npos || options_.startup_timeout_ms <= 0 ||
      options_.request_timeout_ms <= 0)
    throw std::invalid_argument("Invalid GenAI connection options");
  std::ifstream random("/dev/urandom", std::ios::binary);
  unsigned char bytes[12];
  if (!random.read(reinterpret_cast<char*>(bytes), sizeof(bytes)))
    throw std::runtime_error("Cannot create session identity");
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
  std::vector<std::string> args{"ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-p", "22"};
  std::string key = options_.ssh_key;
  if (key.empty()) {
    if (const char* home = std::getenv("HOME")) {
      auto candidate = std::filesystem::path(home) / ".ssh/sima_neat_pcie_ed25519";
      if (std::filesystem::exists(candidate))
        key = candidate.string();
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
void RemoteSession::start() {
  const auto q = Runner::shell_escape;
  // Session-specific directory is also the ownership record if SSH disconnects
  // before returning the PID. No tensor qN.pid files are used.
  std::string script = "set -eu; umask 077; d=\"$HOME/.cache/neat-genai/" + id_ +
                       "\"; mkdir -p \"$HOME/.cache/neat-genai\"; mkdir \"$d\"; "
                       "nohup /usr/bin/neat-pcie-genai-worker " +
                       q(id_) + " " + q(options_.model_serve_root) + " " + q(model_) + " " +
                       q(options_.card_receive_directory.string()) +
                       " >\"$d/worker.log\" 2>&1 </dev/null & p=$!; echo \"$p\" >\"$d/pid\"; "
                       "awk '{print $22}' /proc/$p/stat >\"$d/start\"; echo launched";
  launched_ = true;
  const auto result = Runner::run(ssh(script), 15);
  if (result.exit_code || result.timed_out)
    throw std::runtime_error("Could not launch GenAI worker: " + result.output);
}
void RemoteSession::stop() {
  if (!launched_)
    return;
  std::string script = "set -eu; d=\"$HOME/.cache/neat-genai/" + id_ +
                       "\"; test -f \"$d/pid\" || exit 0; p=$(cat \"$d/pid\"); "
                       "case $p in ''|*[!0-9]*) exit 1;; esac; "
                       "owned() { test -r /proc/$p/stat && test -f \"$d/start\" && "
                       "test \"$(awk '{print $22}' /proc/$p/stat)\" = \"$(cat \"$d/start\")\" && "
                       "tr '\\0' '\\n' </proc/$p/cmdline | grep -Fxq " +
                       Runner::shell_escape(id_) +
                       "; }; if owned; then kill -TERM \"$p\"; "
                       "i=0; while owned && test $i -lt 50; do sleep 0.1; i=$((i+1)); done; "
                       "if owned; then kill -KILL \"$p\"; fi; fi";
  const auto result = Runner::run(ssh(script), 15);
  if (result.exit_code || result.timed_out)
    throw std::runtime_error("GenAI worker shutdown could not be confirmed");
  launched_ = false;
}
} // namespace simaai::neat::pcie::genai::internal
