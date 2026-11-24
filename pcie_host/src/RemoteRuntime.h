#pragma once

#include "HostPlatform.h"
#include "SshRunner.h"
#include "simaai/neat/pcie/Model.h"

#include <chrono>
#include <filesystem>
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

// The host process that started a queue's card worker, as the start script records it on the
// card in q<N>.owner beside the queue's PID file.
struct QueueOwner {
  int builder_pid = -1;
  HostProcessOwner host;
  std::vector<std::string> uploads;
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
  RemoteStatus wait_ready(int queue, int expected_pid, int readiness_timeout_ms) const;
  void stop(int queue, int expected_pid) const;
  void stop_process(int expected_pid, const std::string& expected_model_path) const;
  RemoteStatus read_status(int queue, std::chrono::milliseconds timeout) const;

  std::string endpoint() const;
  const std::string& helper_path() const {
    return helper_path_;
  }
  std::string status_path(int queue) const;
  std::string pid_path(int queue) const;
  std::string owner_path(int queue) const;
  void remove_upload(const std::string& remote_path) const;
  static bool is_managed_upload_path(const std::string& remote_path);
  static std::string unique_remote_upload_path(const std::string& local_path);
  static int parse_launched_pid(const std::string& output);
  static bool status_owner_matches(const RemoteStatus& status, int expected_pid);
  static bool start_failure_cleanup_safe(int exit_code, bool timed_out);
  // The next step for a card-side start failure the user can fix, or "".
  std::string start_failure_remedy(const std::string& output, int queue) const;
  static std::optional<QueueOwner> parse_queue_owner(const std::string& output);
  // This host package's Neat release, which the card's Neat release must match.
  static std::string host_neat_version();
  static std::string child_cleanup_shell_function();
  static std::string detached_child_redirections();
  std::vector<std::string> ssh_base() const;
  std::vector<std::string> scp_base() const;

private:
  ConnectionOptions connection_;
  std::string helper_path_;
  std::string run_dir_;
  std::string log_dir_;

  void run_or_throw(const std::vector<std::string>& args, int timeout_sec,
                    const std::string& context) const;
  std::vector<std::string> start_command(int queue, const std::string& remote_model_path,
                                         const std::optional<std::string>& options_path) const;
  CommandResult run_start(const std::vector<std::string>& cmd) const;
};

} // namespace simaai::neat::pcie::internal
