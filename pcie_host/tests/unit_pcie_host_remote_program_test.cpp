#include "RemoteRuntime.h"
#include "SshRunner.h"

#include "simaai/neat/pcie/Model.h"

#include <csignal>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace pcie = simaai::neat::pcie;
namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

int count_occurrences(const std::string& haystack, const std::string& needle) {
  int count = 0;
  std::size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

} // namespace

int main() {
  try {
    // Default: the tensor pipeline program is unchanged.
    {
      pcie::ConnectionOptions options; // user defaults to "sima"
      const pcie_internal::RemoteRuntime runtime(options);
      require(runtime.card_program() == "pcie-pipeline-builder",
              "default card program must remain pcie-pipeline-builder");
      require(runtime.remote_helper_path() == "/usr/bin/pcie-pipeline-builder",
              "default helper path must remain /usr/bin/pcie-pipeline-builder");
    }

    // Override: GenAI selects its own program name.
    {
      pcie::ConnectionOptions options;
      options.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime(options);
      require(runtime.card_program() == "pcie-genai-backend",
              "override card program must be honoured");
      require(runtime.remote_helper_path() == "/usr/bin/pcie-genai-backend",
              "override helper path must follow the configured name");
    }

    // The start command must use the configured name in ALL of: the launch path
    // and the two /proc/<pid>/cmdline safety checks (busy check + claim check).
    {
      pcie::ConnectionOptions options;
      const pcie_internal::RemoteRuntime runtime(options);
      const std::string cmd = runtime.build_start_command(0, "/tmp/model", std::nullopt);
      require(contains(cmd, "/usr/bin/pcie-pipeline-builder"),
              "default start must launch the helper path");
      require(count_occurrences(cmd, "pcie-pipeline-builder") >= 3,
              "default name must appear in launch + both start-time checks");
    }

    // The safety interlock: an overridden name must appear in every spot and the
    // stale default must appear NOWHERE (else the host cannot confirm or stop it).
    {
      pcie::ConnectionOptions options;
      options.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime(options);
      const std::string cmd = runtime.build_start_command(2, "/tmp/model", std::nullopt);
      require(contains(cmd, "/usr/bin/pcie-genai-backend"),
              "override start must launch the new helper path");
      require(count_occurrences(cmd, "pcie-genai-backend") >= 3,
              "override name must appear in launch + both start-time checks");
      require(!contains(cmd, "pcie-pipeline-builder"),
              "override start must not carry the stale default name anywhere");
    }

    // The stop command guards the kill with the same cmdline check.
    {
      const pcie_internal::RemoteRuntime runtime_default(pcie::ConnectionOptions{});
      require(contains(runtime_default.build_stop_command(0, 123), "pcie-pipeline-builder"),
              "default stop must guard the kill with the default name");

      pcie::ConnectionOptions options;
      options.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime_override(options);
      const std::string stop_cmd = runtime_override.build_stop_command(0, 123);
      require(contains(stop_cmd, "pcie-genai-backend"),
              "override stop must guard the kill with the new name");
      require(!contains(stop_cmd, "pcie-pipeline-builder"),
              "override stop must not guard with the stale default name");
    }

    // An unsafe program name must be rejected: it lands in a shell grep pattern.
    {
      for (const std::string bad : {"a b", "bad;rm -rf /", "quote'name", "-leadingdash"}) {
        pcie::ConnectionOptions options;
        options.card_program = bad;
        bool threw = false;
        try {
          const pcie_internal::RemoteRuntime runtime(options);
        } catch (const std::invalid_argument&) {
          threw = true;
        }
        require(threw, "unsafe card_program must be rejected: '" + bad + "'");
      }
      // A safe basename is accepted.
      pcie::ConnectionOptions ok;
      ok.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime(ok);
      require(runtime.card_program() == "pcie-genai-backend", "safe card_program accepted");
    }

    // wait_ready's poll: one ssh command reads the status AND checks the pid lives,
    // so a program that dies during the load fails fast instead of at the timeout.
    {
      pcie::ConnectionOptions options;
      options.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime(options);
      const std::string cmd = runtime.build_ready_probe_command(3, 4242);
      require(contains(cmd, "/run/sima-neat/pcie/q3.status"), "probe must read the status file");
      require(contains(cmd, "kill -0 4242"), "probe must check the launched pid");

      // Run the probe for real (no status file here): our own pid is alive.
      const auto self = pcie_internal::SshRunner::run(
          {"/bin/sh", "-c", runtime.build_ready_probe_command(3, static_cast<int>(::getpid()))}, 5);
      const auto live = pcie_internal::RemoteRuntime::parse_ready_probe(self.output, 3);
      require(self.exit_code == 0 && live.alive && live.status.state.empty(),
              "live pid with no status file: alive, empty state");

      // A reaped child's pid is dead.
      const pid_t child = ::fork();
      if (child == 0) {
        _exit(0);
      }
      ::waitpid(child, nullptr, 0);
      const auto gone = pcie_internal::SshRunner::run(
          {"/bin/sh", "-c", runtime.build_ready_probe_command(3, static_cast<int>(child))}, 5);
      require(!pcie_internal::RemoteRuntime::parse_ready_probe(gone.output, 3).alive,
              "a dead pid must be reported as not alive");

      const auto starting = pcie_internal::RemoteRuntime::parse_ready_probe(
          "{\"state\":\"starting\",\"pid\":4242}\nsima_neat_pid_alive=0\n", 3);
      require(!starting.alive && starting.status.state == "starting" && starting.status.pid == 4242,
              "status body and liveness parse together");
      const auto ready = pcie_internal::RemoteRuntime::parse_ready_probe(
          "{\"state\":\"ready\",\"pid\":4242}\n\nsima_neat_pid_alive=1\n", 3);
      require(ready.alive && ready.status.state == "ready", "ready + alive parses");
      require(pcie_internal::RemoteRuntime::parse_ready_probe("garbage", 3).alive,
              "no marker (unknown) must not be treated as dead");
    }

    // A card program that fails before the host sees its queue claim (exit 15):
    // its own "failed" status message is shown, not the bare exit code. A status
    // of another pid, another exit code or a timeout keeps the plain message.
    {
      using RR = pcie_internal::RemoteRuntime;
      pcie_internal::CommandResult early;
      early.exit_code = 15;
      early.output = "launched_pid=4242\nbuilder_exited_before_queue_claim:1\n";
      pcie_internal::RemoteStatus failed;
      failed.state = "failed";
      failed.pid = 4242;
      failed.message = "another pcie-genai-backend is already running on this card";
      require(RR::start_failure_message("pcie-genai-backend", early, failed) ==
                  "remote pcie-genai-backend failed at start: another pcie-genai-backend is "
                  "already running on this card",
              "an early failure shows the card's own reason");
      pcie_internal::RemoteStatus other = failed;
      other.pid = 7;
      require(contains(RR::start_failure_message("pcie-genai-backend", early, other), "exit=15"),
              "a status of another pid is not trusted");
      pcie_internal::CommandResult busy = early;
      busy.exit_code = 9;
      require(contains(RR::start_failure_message("pcie-genai-backend", busy, failed), "exit=9"),
              "other exit codes keep the plain message");
      require(contains(RR::start_failure_message("pcie-genai-backend", early, {}), "exit=15"),
              "no status keeps the plain message");
    }

    // A failed start can leave the launched backend running. RemoteStartError
    // carries its pid, and stop-by-pid kills it only if its cmdline is ours.
    {
      pcie_internal::RemoteStartError with_pid("start failed", true, 4242);
      require(with_pid.launched_pid() == 4242, "RemoteStartError carries the launched pid");
      pcie_internal::RemoteStartError no_pid("start failed", true);
      require(no_pid.launched_pid() == -1, "no launched pid defaults to -1");

      pcie::ConnectionOptions options;
      options.card_program = "pcie-genai-backend";
      const pcie_internal::RemoteRuntime runtime(options);
      const std::string cmd = runtime.build_stop_launched_pid_command(4242);
      require(contains(cmd, "pid=4242"), "stop-by-pid targets the launched pid");
      require(contains(cmd, "grep -q 'pcie-genai-backend'"),
              "stop-by-pid guards the kill with the program name");

      // Run it for real on a live child that is NOT our program: it must live.
      const pid_t other = ::fork();
      if (other == 0) {
        ::sleep(30);
        _exit(0);
      }
      const auto kept = pcie_internal::SshRunner::run(
          {"/bin/sh", "-c", runtime.build_stop_launched_pid_command(static_cast<int>(other))}, 10);
      require(kept.exit_code == 0 && ::kill(other, 0) == 0,
              "stop-by-pid must not kill a pid whose cmdline is another program");
      ::kill(other, SIGKILL);
      ::waitpid(other, nullptr, 0);

      // A process whose cmdline IS our program is stopped. argv[0] makes
      // /proc/<pid>/cmdline read "pcie-genai-backend 30".
      const pid_t ours = ::fork();
      if (ours == 0) {
        ::execl("/bin/sleep", "pcie-genai-backend", "30", static_cast<char*>(nullptr));
        _exit(127);
      }
      ::usleep(200 * 1000); // let execl() replace the cmdline
      const auto stopped = pcie_internal::SshRunner::run(
          {"/bin/sh", "-c", runtime.build_stop_launched_pid_command(static_cast<int>(ours))}, 10);
      int ours_status = 0;
      require(stopped.exit_code == 0 && ::waitpid(ours, &ours_status, WNOHANG) == ours &&
                  WIFSIGNALED(ours_status),
              "stop-by-pid must stop a launched backend");

      // A dead pid is a no-op that succeeds.
      const auto gone = pcie_internal::SshRunner::run(
          {"/bin/sh", "-c", runtime.build_stop_launched_pid_command(static_cast<int>(other))}, 5);
      require(gone.exit_code == 0, "stop-by-pid on a dead pid succeeds");
    }

    std::cout << "[PASS] remote card-program name generalization\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
