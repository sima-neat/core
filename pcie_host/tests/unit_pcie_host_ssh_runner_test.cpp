#include "SshRunner.h"
#include "RemoteRuntime.h"

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_child_cleanup_shell() {
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

} // namespace

int main() {
  try {
    const std::string escaped = pcie_internal::SshRunner::shell_escape("ab'cd");
    if (escaped != "'ab'\\''cd'") {
      throw std::runtime_error("shell escape mismatch");
    }
    const auto ok = pcie_internal::SshRunner::run({"/bin/sh", "-c", "echo hello"}, 2);
    if (ok.exit_code != 0 || ok.output.find("hello") == std::string::npos) {
      throw std::runtime_error("expected echo command to succeed");
    }
    const auto bad = pcie_internal::SshRunner::run({"/bin/sh", "-c", "exit 7"}, 2);
    if (bad.exit_code != 7) {
      throw std::runtime_error("expected exit code 7");
    }
    const auto timed =
        pcie_internal::SshRunner::run_for({"/bin/sleep", "2"}, std::chrono::milliseconds(50));
    if (!timed.timed_out) {
      throw std::runtime_error("expected millisecond command timeout");
    }
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
    std::cout << "[PASS] ssh runner\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
