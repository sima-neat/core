#pragma once

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
};

class RemoteStartError final : public std::runtime_error {
public:
  RemoteStartError(std::string message, bool cleanup_safe)
      : std::runtime_error(std::move(message)), cleanup_safe_(cleanup_safe) {}

  bool cleanup_safe() const noexcept {
    return cleanup_safe_;
  }

private:
  bool cleanup_safe_ = false;
};

class RemoteRuntime {
public:
  explicit RemoteRuntime(ConnectionOptions connection);

  std::string upload_file(const std::string& local_path) const;
  int start(int queue, const std::string& remote_model_path,
            const std::optional<std::string>& remote_model_options_path) const;
  RemoteStatus wait_ready(int queue, int expected_pid, int readiness_timeout_ms) const;
  void stop(int queue, int expected_pid) const;
  RemoteStatus read_status(int queue, std::chrono::milliseconds timeout) const;

  std::string endpoint() const;
  std::string status_path(int queue) const;
  std::string pid_path(int queue) const;
  void remove_upload(const std::string& remote_path) const;
  static bool is_managed_upload_path(const std::string& remote_path);
  static std::string unique_remote_upload_path(const std::string& local_path);
  static int parse_launched_pid(const std::string& output);
  static bool status_owner_matches(const RemoteStatus& status, int expected_pid);

private:
  ConnectionOptions connection_;

  std::vector<std::string> ssh_base() const;
  std::vector<std::string> scp_base() const;
  void run_or_throw(const std::vector<std::string>& args, int timeout_sec,
                    const std::string& context) const;
};

} // namespace simaai::neat::pcie::internal
