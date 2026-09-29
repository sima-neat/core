#pragma once

#include "simaai/neat/pcie/Model.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::pcie::internal {

struct RemoteStatus {
  std::string state;
  int queue = -1;
  int pid = -1;
  std::string message;
  std::string error_code;
  std::size_t output_buffer_bytes = 0;
};

/// One wait_ready poll: the status file plus whether the launched pid still lives.
struct ReadyProbe {
  RemoteStatus status;
  bool alive = true; // unknown (ssh failed, no marker) counts as alive
};

class RemoteStartError final : public std::runtime_error {
public:
  RemoteStartError(std::string message, bool cleanup_safe,
                   std::optional<int> launched_pid = std::nullopt)
      : std::runtime_error(std::move(message)), cleanup_safe_(cleanup_safe),
        launched_pid_(launched_pid) {}

  bool cleanup_safe() const noexcept {
    return cleanup_safe_;
  }

  std::optional<int> launched_pid() const noexcept {
    return launched_pid_;
  }

private:
  bool cleanup_safe_ = false;
  std::optional<int> launched_pid_;
};

class RemoteRuntime {
public:
  explicit RemoteRuntime(ConnectionOptions connection);

  std::string upload_file(const std::string& local_path) const;
  int start(int queue, const std::string& remote_model_path,
            const std::optional<std::string>& remote_model_options_path) const;
  RemoteStatus wait_ready(int queue, int expected_pid, int readiness_timeout_ms,
                          const std::function<bool()>& should_abort = {}) const;
  void stop(int queue, int expected_pid) const;
  void stop_process(int expected_pid, const std::string& expected_model_path) const;
  RemoteStatus read_status(int queue, std::chrono::milliseconds timeout) const;

  std::string endpoint() const;
  std::string status_path(int queue) const;
  std::string pid_path(int queue) const;

  /// Resolved card-side program name. Empty ConnectionOptions::card_program
  /// selects the default "pcie-pipeline-builder" (the tensor pipeline).
  std::string card_program() const;
  /// Absolute launch path for the card program: "/usr/bin/<card_program()>".
  std::string remote_helper_path() const;
  /// Build the SSH command that starts the card program on a queue. Pure (no I/O)
  /// so the name/interlock wiring is unit-testable without a card.
  std::string
  build_start_command(int queue, const std::string& remote_model_path,
                      const std::optional<std::string>& remote_model_options_path) const;
  /// Build the SSH command that stops the card program on a queue. Pure (no I/O).
  std::string build_stop_command(int queue, int expected_pid) const;
  /// Build the SSH command for one wait_ready poll: cat the status file, then
  /// report `kill -0 <expected_pid>` on a marker line. Pure (no I/O).
  std::string build_ready_probe_command(int queue, int expected_pid) const;
  /// Parse the output of build_ready_probe_command(). Pure (no I/O).
  static ReadyProbe parse_ready_probe(const std::string& output, int queue);
  /// Parse a status file body; empty body -> empty state, bad JSON -> "malformed".
  static RemoteStatus parse_status(const std::string& body, int queue);
  void remove_upload(const std::string& remote_path) const;
  static bool is_managed_upload_path(const std::string& remote_path);
  static std::string unique_remote_upload_path(const std::string& local_path);
  static int parse_launched_pid(const std::string& output);
  static bool status_owner_matches(const RemoteStatus& status, int expected_pid);
  static bool start_failure_cleanup_safe(int exit_code, bool timed_out);
  /// Emit shell helpers child_exited()/terminate_launched() used by
  /// build_start_command() to reap the launched child. Pure (no I/O).
  static std::string child_cleanup_shell_function();

private:
  ConnectionOptions connection_;

  std::vector<std::string> ssh_base() const;
  std::vector<std::string> scp_base() const;
  std::string startup_log_path(int queue) const;
  void run_or_throw(const std::vector<std::string>& args, int timeout_sec,
                    const std::string& context) const;
};

} // namespace simaai::neat::pcie::internal
