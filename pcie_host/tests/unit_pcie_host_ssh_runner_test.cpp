#include "HostPlatform.h"
#include "SshRunner.h"
#include "RemoteRuntime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_child_cleanup_shell() {
#ifndef _WIN32
  const std::string cleanup = pcie_internal::RemoteRuntime::child_cleanup_shell_function();
  const auto normal = pcie_internal::SshRunner::run(
      {"/bin/bash", "-c",
       cleanup + "sleep 30 & launched_pid=$!; owner_pid=$$; "
                 "terminate_launched; rc=$?; "
                 "kill -0 \"$owner_pid\" >/dev/null 2>&1 || exit 21; exit $rc"},
      5);
  require(normal.exit_code == 0 && !normal.timed_out,
          "normal launched child was not terminated and reaped");

  const auto ignores_term = pcie_internal::SshRunner::run(
      {"/bin/bash", "-c",
       cleanup + "( trap \"\" TERM; exec sleep 30 ) & launched_pid=$!; "
                 "sleep 0.1; terminate_launched"},
      5);
  require(ignores_term.exit_code == 0 && !ignores_term.timed_out,
          "SIGTERM-ignoring child was not killed and reaped");

  const auto preserves_owner = pcie_internal::SshRunner::run(
      {"/bin/bash", "-c",
       cleanup + "sleep 30 & owner_pid=$!; sleep 30 & launched_pid=$!; "
                 "terminate_launched; rc=$?; "
                 "kill -0 \"$owner_pid\" >/dev/null 2>&1 || exit 22; "
                 "kill -TERM \"$owner_pid\"; wait \"$owner_pid\" 2>/dev/null; "
                 "exit $rc"},
      5);
  require(preserves_owner.exit_code == 0 && !preserves_owner.timed_out,
          "launched-child cleanup affected the existing owner");
#endif
}

void test_detached_child_redirections() {
  const std::string redirections = pcie_internal::RemoteRuntime::detached_child_redirections();
  require(redirections.find("9>&-") != std::string::npos,
          "detached worker must close the launch lock descriptor");
  require(redirections.find("</dev/null") != std::string::npos,
          "detached worker must not inherit the SSH channel stdin");
  require(redirections.find(">/dev/null 2>&1") != std::string::npos,
          "detached worker must not inherit the SSH channel output streams");
  require(redirections.find(" & ") != std::string::npos,
          "detached worker must be launched in the background");
}

void test_command_runner(const std::string& executable) {
  const auto ok = pcie_internal::SshRunner::run({executable, "--emit", "hello"}, 2);
  require(ok.exit_code == 0 && ok.output == "hello\n", "expected child output capture");

  const auto bad = pcie_internal::SshRunner::run({executable, "--exit", "7"}, 2);
  require(bad.exit_code == 7, "expected child exit code 7");
  // A command that fails without output still names its exit code and executable.
  require(pcie_internal::SshRunner::failure_detail({executable, "--exit", "7"}, bad) ==
              "(exit=7, timed_out=false): " + executable + " printed no output",
          "silent command failure was not described");
  require(pcie_internal::SshRunner::failure_detail({executable}, ok) ==
              "(exit=0, timed_out=false): hello\n",
          "command output was not kept in the failure detail");

  const auto round_trip = pcie_internal::SshRunner::run(
      {executable, "--echo-args", "two words", "quote\"inside", "C:\\path with space\\"}, 2);
  require(round_trip.exit_code == 0 &&
              round_trip.output == "ARG=two words\nARG=quote\"inside\nARG=C:\\path with space\\\n",
          "command arguments did not round-trip exactly: " + round_trip.output);

  const auto flood = pcie_internal::SshRunner::run({executable, "--emit-bytes", "131072"}, 2);
  require(flood.exit_code == 0 && flood.output.size() == 131072 &&
              std::all_of(flood.output.begin(), flood.output.end(),
                          [](const char byte) { return byte == 'x'; }),
          "large child output was truncated or corrupted");

  const auto timed = pcie_internal::SshRunner::run_for({executable, "--sleep-ms", "2000"},
                                                       std::chrono::milliseconds(50));
  require(timed.timed_out && timed.exit_code == 124,
          "expected bounded command timeout and timeout exit code");
}

#ifdef _WIN32
std::optional<std::string> environment_variable(const char* name) {
  const DWORD size = GetEnvironmentVariableA(name, nullptr, 0);
  if (size == 0) {
    return std::nullopt;
  }
  std::string value(size, '\0');
  value.resize(GetEnvironmentVariableA(name, value.data(), size));
  return value;
}

class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(const char* name, const char* value)
      : name_(name), previous_(environment_variable(name)) {
    require(SetEnvironmentVariableA(name, value) != FALSE, std::string("could not set ") + name);
  }
  ~ScopedEnvironmentVariable() {
    SetEnvironmentVariableA(name_, previous_ ? previous_->c_str() : nullptr);
  }
  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

private:
  const char* name_;
  std::optional<std::string> previous_;
};

// A non-terminal Windows OpenSSH server session gives every descendant this
// variable; ssh.exe children that inherit it never exit after printing.
void test_child_environment_omits_openssh_fd_state(const std::string& executable) {
  constexpr const char* openssh_fd_state = "c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE";
  constexpr const char* ordinary = "SIMA_NEAT_PCIE_RUNNER_TEST_INHERITED";
  const ScopedEnvironmentVariable fd_state(openssh_fd_state, "AAAAAAICAgA=");
  const ScopedEnvironmentVariable inherited(ordinary, "kept");

  const auto child =
      pcie_internal::SshRunner::run({executable, "--print-env", openssh_fd_state, ordinary}, 2);
  require(child.exit_code == 0 && !child.timed_out &&
              child.output == std::string(openssh_fd_state) + " unset\n" + ordinary + "=kept\n",
          "child environment must omit the OpenSSH descriptor state and keep the rest: " +
              child.output);
  require(environment_variable(openssh_fd_state) == "AAAAAAICAgA=",
          "runner must not change the parent process environment");
}

// OpenSSH outside System32\OpenSSH (e.g. C:\Program Files\OpenSSH) is found through PATH and
// returned as a full path; a bare name would make CreateProcessW use the current directory.
void test_open_ssh_executable_searches_path() {
  pcie_internal::ScopedTempDirectory directory("sima-neat-pcie-path-test");
  const auto tool = directory.path() / "sima-neat-path-probe.exe";
  std::ofstream(tool).put('\0');
  const auto path = "relative;" + pcie_internal::host_path_string(directory.path());
  const ScopedEnvironmentVariable scoped_path("PATH", path.c_str());
  require(pcie_internal::open_ssh_executable("sima-neat-path-probe") ==
              pcie_internal::host_path_string(tool),
          "OpenSSH lookup did not return the full PATH match");
  // A tool outside System32\OpenSSH and PATH fails, even when the current directory holds a
  // same-named executable that CreateProcessW would run for a bare name.
  pcie_internal::ScopedTempDirectory current("sima-neat-pcie-cwd-test");
  std::ofstream(current.path() / "sima-neat-cwd-probe.exe").put('\0');
  const auto previous = std::filesystem::current_path();
  std::filesystem::current_path(current.path());
  std::string error;
  try {
    (void)pcie_internal::open_ssh_executable("sima-neat-cwd-probe");
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  std::filesystem::current_path(previous);
  require(error.find("OpenSSH Client") != std::string::npos,
          "OpenSSH lookup must fail with the fix instead of a bare name: " + error);
}
#endif

void test_windows_argument_quoting() {
  using Runner = pcie_internal::SshRunner;
  require(Runner::windows_quote_argument("plain") == "plain", "plain argument was quoted");
  require(Runner::windows_quote_argument("") == "\"\"", "empty argument quoting mismatch");
  require(Runner::windows_quote_argument("two words") == "\"two words\"",
          "spaced argument quoting mismatch");
  require(Runner::windows_quote_argument("quote\"inside") == "\"quote\\\"inside\"",
          "embedded quote escaping mismatch");
  require(Runner::windows_quote_argument("C:\\path with space\\") == "\"C:\\path with space\\\\\"",
          "trailing backslash escaping mismatch");
}

void test_host_platform() {
  require(pcie_internal::host_process_id() > 0, "host process ID must be positive");
  std::array<unsigned char, 32> random{};
  pcie_internal::host_secure_random(random.data(), random.size());
  require(std::any_of(random.begin(), random.end(), [](const unsigned char byte) { return byte; }),
          "secure host randomness unexpectedly returned only zero bytes");
  require(pcie_internal::host_path_string(std::filesystem::path("path") / "file") ==
              (std::filesystem::path("path") / "file").string(),
          "host path conversion changed an ASCII path");
  const std::string unicode_path = "customer models \xCE\xB4/model archive.tar.gz";
  require(pcie_internal::host_path_string(pcie_internal::host_path_from_string(unicode_path)) ==
              unicode_path,
          "host path conversion did not preserve a UTF-8 path");
#ifdef _WIN32
  require(pcie_internal::same_host_path("c:\\Media Root", "C:\\media root"),
          "Windows path comparison must ignore case");
#else
  require(!pcie_internal::same_host_path("/media/root", "/Media/Root"),
          "POSIX path comparison must keep case");
#endif
#ifdef _WIN32
  const auto ssh = pcie_internal::open_ssh_executable("ssh");
  require(ssh.size() >= std::string("ssh.exe").size() &&
              ssh.substr(ssh.size() - std::string("ssh.exe").size()) == "ssh.exe",
          "Windows OpenSSH command must name ssh.exe");
#else
  require(pcie_internal::open_ssh_executable("ssh") == "ssh", "POSIX OpenSSH command name changed");
#endif
  std::filesystem::path first_path;
  {
    pcie_internal::ScopedTempDirectory first("sima-neat-pcie-platform-test");
    pcie_internal::ScopedTempDirectory second("sima-neat-pcie-platform-test");
    first_path = first.path();
    require(first.path() != second.path() && std::filesystem::is_directory(first.path()) &&
                std::filesystem::is_directory(second.path()),
            "temporary directories must be distinct and exist for their scoped lifetime");
#ifndef _WIN32
    const auto permissions = std::filesystem::status(first.path()).permissions();
    require((permissions & (std::filesystem::perms::group_all |
                            std::filesystem::perms::others_all)) == std::filesystem::perms::none,
            "temporary directory must not grant group or other access");
#endif
  }
  require(!std::filesystem::exists(first_path),
          "temporary directory must be removed at the end of its scoped lifetime");
}

void test_start_failure_cleanup_classification() {
  for (const int exit_code : {9, 10, 11, 12, 14, 15, 16, 21}) {
    require(pcie_internal::RemoteRuntime::start_failure_cleanup_safe(exit_code, false),
            "known cleanup-safe start failure was not classified as safe");
  }
  for (const int exit_code : {1, 17, 18, 255}) {
    require(!pcie_internal::RemoteRuntime::start_failure_cleanup_safe(exit_code, false),
            "ambiguous start failure was classified as cleanup-safe");
  }
  require(!pcie_internal::RemoteRuntime::start_failure_cleanup_safe(9, true),
          "timed-out start was classified as cleanup-safe");

  const pcie_internal::RemoteStartError error("cleanup failed", false, 1234);
  require(!error.cleanup_safe() && error.launched_pid() == 1234,
          "remote start failure did not preserve its launched PID");
}

void test_remote_runtime_paths() {
  simaai::neat::pcie::ConnectionOptions connection;
  connection.card_env = "SIMA_NEAT_PCIE_HELPER_PATH=/home/sima/neat/bin/pcie-pipeline-builder "
                        "SIMA_NEAT_PCIE_RUN_DIR=/home/sima/neat/run "
                        "SIMA_NEAT_PCIE_LOG_DIR=/home/sima/neat/log";
  const pcie_internal::RemoteRuntime runtime(connection);
  require(runtime.helper_path() == "/home/sima/neat/bin/pcie-pipeline-builder",
          "custom remote helper path was not retained");
  require(runtime.pid_path(3) == "/home/sima/neat/run/q3.pid",
          "custom remote PID path was not retained");
  require(runtime.status_path(3) == "/home/sima/neat/run/q3.status",
          "custom remote status path was not retained");

  for (const std::string card_env : {"SIMA_NEAT_PCIE_RUN_DIR=relative/path",
                                     "SIMA_NEAT_PCIE_HELPER_PATH=/opt/neat/bin/worker"}) {
    simaai::neat::pcie::ConnectionOptions invalid;
    invalid.card_env = card_env;
    bool rejected = false;
    try {
      (void)pcie_internal::RemoteRuntime(invalid);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "invalid remote runtime path was accepted: " + card_env);
  }
}

bool has_ssh_option(const std::vector<std::string>& command, const std::string& option) {
  for (std::size_t index = 1; index < command.size(); ++index) {
    if (command[index - 1] == "-o" && command[index] == option) {
      return true;
    }
  }
  return false;
}

void test_remote_transport_requires_known_host_keys() {
  const pcie_internal::RemoteRuntime runtime(simaai::neat::pcie::ConnectionOptions{});
  require(has_ssh_option(runtime.ssh_base(), "StrictHostKeyChecking=yes") &&
              has_ssh_option(runtime.scp_base(), "StrictHostKeyChecking=yes"),
          "tensor ssh and scp must refuse an unknown card host key");
  require(has_ssh_option(runtime.ssh_base(), "BatchMode=yes") &&
              has_ssh_option(runtime.scp_base(), "BatchMode=yes"),
          "tensor ssh and scp must fail instead of waiting on a hidden prompt");
}

void test_failures_name_the_next_step() {
  using Runner = pcie_internal::SshRunner;
#ifdef _WIN32
  const std::string setup = "pcie-setup.cmd";
#else
  const std::string setup = "pcie-setup.sh";
#endif
  for (const std::string output :
       {"@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@\n",
        "Host key verification failed.\r\n", "sima@10.0.0.2: Permission denied (publickey).\n"}) {
    const auto detail = Runner::failure_detail({"ssh"}, {255, false, output});
    require(detail.find(output) != std::string::npos && detail.find(setup) != std::string::npos,
            "SSH trust failure did not name the setup tool: " + detail);
  }
  require(Runner::failure_remedy("ssh: connect to host 10.0.0.2 port 22: Connection timed out")
                  .find("ping") != std::string::npos,
          "unreachable card did not name the link check");
  require(Runner::failure_remedy("some other failure").empty(),
          "an unknown failure must not get a guessed remedy");

  const pcie_internal::RemoteRuntime runtime(simaai::neat::pcie::ConnectionOptions{});
  const auto busy = runtime.start_failure_remedy("launched_pid=42\r\nqueue_busy\r\n", 3);
  require(busy.find("\nAnother card worker holds PCIe queue 3.") == 0 &&
              busy.find(": ssh ") != std::string::npos &&
              busy.find(runtime.endpoint() +
                        " \"pkill -f '[p]cie-pipeline-builder.*--queue 3( |$)'\"") !=
                  std::string::npos,
          "busy queue did not give the worker stop command: " + busy);
  require(runtime.start_failure_remedy("queue_busy_cleanup_failed\n", 3).empty(),
          "a failed cleanup is not a leftover worker");
  const auto missing = runtime.start_failure_remedy("missing_builder\n", 0);
  require(missing.find("Install the Neat release") != std::string::npos &&
              missing.find("(Neat " + pcie_internal::RemoteRuntime::host_neat_version() + ")") !=
                  std::string::npos,
          "missing card worker did not name the card install and host release: " + missing);
  const auto mismatch =
      runtime.start_failure_remedy("card_neat_version=0.5.1\r\nneat_version_mismatch\r\n", 0);
  require(mismatch.find("runs Neat 0.5.1 and this host runs Neat " +
                        pcie_internal::RemoteRuntime::host_neat_version()) != std::string::npos &&
              mismatch.find("Install the matching Neat release on the card (see Install Neat "
                            "on the card).") != std::string::npos,
          "a different card release did not name both releases and the install: " + mismatch);

  const auto owner = pcie_internal::RemoteRuntime::parse_queue_owner(
      "owner.builder_pid=42\r\nowner.host_id=linux:x\r\nowner.host=build-02\r\n"
      "owner.host_pid=7\r\nowner.host_start=99\r\nowner.upload=/tmp/sima-neat-pcie-a\r\n"
      "owner.upload=/tmp/sima-neat-pcie-b\r\nqueue_busy\r\n");
  require(owner.has_value() && owner->builder_pid == 42 && owner->host.host_id == "linux:x" &&
              owner->host.host == "build-02" && owner->host.pid == 7 && owner->host.start == "99" &&
              owner->uploads.size() == 2 && owner->uploads[1] == "/tmp/sima-neat-pcie-b",
          "queue owner record was not parsed");
  require(!pcie_internal::RemoteRuntime::parse_queue_owner("queue_busy\n").has_value() &&
              !pcie_internal::RemoteRuntime::parse_queue_owner("owner.builder_pid=x\n").has_value(),
          "a missing or malformed owner record was accepted");
  require(runtime.start_failure_remedy("missing_run_dir\n", 0).find("/run/sima-neat/pcie") !=
              std::string::npos,
          "missing run directory was not named");
}

void set_test_environment(const char* name, const std::optional<std::string>& value) {
#ifdef _WIN32
  _putenv_s(name, value ? value->c_str() : "");
#else
  value ? setenv(name, value->c_str(), 1) : unsetenv(name);
#endif
}

void test_remote_transport_offers_the_neat_key() {
#ifdef _WIN32
  const char* home_variable = "USERPROFILE";
#else
  const char* home_variable = "HOME";
#endif
  const auto home = std::filesystem::temp_directory_path() /
                    ("neat-pcie-identity-" + std::to_string(pcie_internal::host_process_id()));
  std::filesystem::create_directories(home / ".ssh");
  std::ofstream(home / ".ssh" / "sima_neat_pcie_ed25519") << "fixture";
  const char* previous = std::getenv(home_variable);
  const std::optional<std::string> saved =
      previous ? std::optional<std::string>(previous) : std::nullopt;
  set_test_environment(home_variable, home.string());
  const pcie_internal::RemoteRuntime runtime(simaai::neat::pcie::ConnectionOptions{});
  const auto ssh = runtime.ssh_base();
  const auto scp = runtime.scp_base();
  const auto busy = runtime.start_failure_remedy("queue_busy\n", 3);
  set_test_environment(home_variable, saved);
  // The printed stop command names the key, as the troubleshooting docs do.
  const std::string key = pcie_internal::host_path_string(home / ".ssh" / "sima_neat_pcie_ed25519");
  require(busy.find("ssh -i \"" + key + "\" " + runtime.endpoint() + " \"pkill -f ") !=
              std::string::npos,
          "busy queue stop command must use the Neat key: " + busy);
  std::filesystem::remove_all(home);
#ifdef _WIN32
  // Windows offers only the Neat key: the ssh-agent service offers its keys first.
  const bool identities_only = true;
#else
  // Linux keeps agent and configured keys beside the Neat key, as on develop.
  const bool identities_only = false;
#endif
  for (const auto* command : {&ssh, &scp}) {
    require(std::find(command->begin(), command->end(), "-i") != command->end() &&
                has_ssh_option(*command, "IdentitiesOnly=yes") == identities_only,
            "tensor ssh and scp must offer the Neat key, and only it on Windows");
  }
}

} // namespace

int main(const int argc, char** argv) {
  if (argc >= 2 && std::string_view(argv[1]) == "--emit" && argc == 3) {
    std::cout << argv[2] << "\n";
    return 0;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "--exit" && argc == 3) {
    return std::stoi(argv[2]);
  }
  if (argc >= 2 && std::string_view(argv[1]) == "--echo-args") {
    for (int index = 2; index < argc; ++index) {
      std::cout << "ARG=" << argv[index] << "\n";
    }
    return 0;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "--emit-bytes" && argc == 3) {
    std::cout << std::string(static_cast<std::size_t>(std::stoul(argv[2])), 'x');
    return 0;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "--sleep-ms" && argc == 3) {
    std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(argv[2])));
    return 0;
  }
#ifdef _WIN32
  if (argc >= 2 && std::string_view(argv[1]) == "--print-env") {
    for (int index = 2; index < argc; ++index) {
      const auto value = environment_variable(argv[index]);
      std::cout << argv[index] << (value ? "=" + *value : std::string(" unset")) << "\n";
    }
    return 0;
  }
#endif

  try {
    const std::string escaped = pcie_internal::SshRunner::shell_escape("ab'cd");
    if (escaped != "'ab'\\''cd'") {
      throw std::runtime_error("shell escape mismatch");
    }
    test_windows_argument_quoting();
    test_host_platform();
    test_command_runner(argv[0]);
#ifdef _WIN32
    test_child_environment_omits_openssh_fd_state(argv[0]);
    test_open_ssh_executable_searches_path();
#endif
    const std::string first =
        pcie_internal::RemoteRuntime::unique_remote_upload_path("/first/model.tar.gz");
    const std::string second =
        pcie_internal::RemoteRuntime::unique_remote_upload_path("/second/model.tar.gz");
    if (first == second) {
      throw std::runtime_error("remote upload paths must be unique");
    }
    if (first.find("/tmp/sima-neat-pcie-") != 0 ||
        first.rfind("-model.tar.gz") != first.size() - std::string("-model.tar.gz").size()) {
      throw std::runtime_error("unexpected remote upload path: " + first);
    }
    if (!pcie_internal::RemoteRuntime::is_managed_upload_path(first)) {
      throw std::runtime_error("generated remote upload path must be managed");
    }
    // Host paths are UTF-8 on every platform, including a non-ASCII Windows profile.
    const std::string unicode_leaf = "model \xCE\xB4.tar.gz";
    const std::string unicode = pcie_internal::RemoteRuntime::unique_remote_upload_path(
        "/customer models \xCE\xB4/" + unicode_leaf);
    if (unicode.rfind("-" + unicode_leaf) != unicode.size() - unicode_leaf.size() - 1 ||
        !pcie_internal::RemoteRuntime::is_managed_upload_path(unicode)) {
      throw std::runtime_error("remote upload path did not preserve a UTF-8 leaf: " + unicode);
    }
    const int launched_pid = pcie_internal::RemoteRuntime::parse_launched_pid(
        "Warning: known host added\nlaunched_pid=12345\n");
    if (launched_pid != 12345) {
      throw std::runtime_error("launched PID parsing mismatch");
    }
    bool rejected_missing_pid = false;
    try {
      (void)pcie_internal::RemoteRuntime::parse_launched_pid("queue ready\n");
    } catch (const std::runtime_error&) {
      rejected_missing_pid = true;
    }
    if (!rejected_missing_pid) {
      throw std::runtime_error("remote start output without a launched PID must be rejected");
    }
    const pcie_internal::RemoteStatus matching_status{.state = "ready", .pid = launched_pid};
    const pcie_internal::RemoteStatus replacement_status{.state = "ready", .pid = launched_pid + 1};
    if (!pcie_internal::RemoteRuntime::status_owner_matches(matching_status, launched_pid) ||
        pcie_internal::RemoteRuntime::status_owner_matches(replacement_status, launched_pid)) {
      throw std::runtime_error("remote readiness must remain bound to the launched PID");
    }
    for (const std::string path : {"/tmp/model.tar.gz", "/tmp/sima-neat-pcie-../model.tar.gz",
                                   "/var/tmp/sima-neat-pcie-model.tar.gz"}) {
      if (pcie_internal::RemoteRuntime::is_managed_upload_path(path)) {
        throw std::runtime_error("unsafe remote upload path accepted: " + path);
      }
    }
    test_child_cleanup_shell();
    test_detached_child_redirections();
    test_start_failure_cleanup_classification();
    test_remote_runtime_paths();
    test_remote_transport_requires_known_host_keys();
    test_failures_name_the_next_step();
    test_remote_transport_offers_the_neat_key();
    std::cout << "[PASS] ssh runner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
