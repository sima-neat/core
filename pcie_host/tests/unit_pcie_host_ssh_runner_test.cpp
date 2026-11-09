#include "HostPlatform.h"
#include "SshRunner.h"
#include "RemoteRuntime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

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

void test_command_runner(const std::string& executable) {
  const auto ok = pcie_internal::SshRunner::run({executable, "--emit", "hello"}, 2);
  require(ok.exit_code == 0 && ok.output == "hello\n", "expected child output capture");

  const auto bad = pcie_internal::SshRunner::run({executable, "--exit", "7"}, 2);
  require(bad.exit_code == 7, "expected child exit code 7");

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
  for (const int exit_code : {9, 10, 11, 12, 14, 15, 16}) {
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

  simaai::neat::pcie::ConnectionOptions invalid;
  invalid.card_env = "SIMA_NEAT_PCIE_RUN_DIR=relative/path";
  bool rejected = false;
  try {
    (void)pcie_internal::RemoteRuntime(invalid);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "relative remote runtime path was accepted");
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

  try {
    const std::string escaped = pcie_internal::SshRunner::shell_escape("ab'cd");
    if (escaped != "'ab'\\''cd'") {
      throw std::runtime_error("shell escape mismatch");
    }
    test_windows_argument_quoting();
    test_host_platform();
    test_command_runner(argv[0]);
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
    test_start_failure_cleanup_classification();
    test_remote_runtime_paths();
    std::cout << "[PASS] ssh runner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
