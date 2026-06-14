#pragma once

#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::internal {

struct RemoteStatus {
  std::string state;
  int queue = -1;
  std::string message;
  std::string error_code;
};

class RemoteRuntime {
public:
  explicit RemoteRuntime(ConnectionOptions connection);

  std::string upload_file(const std::string& local_path) const;
  void start(int queue, bool accelerator, const std::string& remote_model_path,
             const std::optional<std::string>& remote_model_options_path) const;
  RemoteStatus wait_ready(int queue, int readiness_timeout_ms) const;
  void stop(int queue) const;
  RemoteStatus read_status(int queue) const;

  std::string endpoint() const;
  std::string status_path(int queue) const;
  std::string pid_path(int queue) const;

private:
  ConnectionOptions connection_;

  std::vector<std::string> ssh_base() const;
  std::vector<std::string> scp_base() const;
  void run_or_throw(const std::vector<std::string>& args, int timeout_sec,
                    const std::string& context) const;
};

} // namespace simaai::neat::pcie::internal
