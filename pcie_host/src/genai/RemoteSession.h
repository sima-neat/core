#pragma once
#include "simaai/neat/pcie/genai/GenAIModel.h"
namespace simaai::neat::pcie::genai::internal {
class RemoteSession {
public:
  explicit RemoteSession(ConnectionOptions options);
  ~RemoteSession();
  void start();
  void stop();
  static std::string stop_script(const std::string& session_id);
  const std::string& id() const {
    return id_;
  }

private:
  std::vector<std::string> ssh(const std::string& script) const;
  std::string id_;
  ConnectionOptions options_;
  bool launched_ = false;
};
} // namespace simaai::neat::pcie::genai::internal
