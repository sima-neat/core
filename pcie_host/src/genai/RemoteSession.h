#pragma once
#include "simaai/neat/pcie/genai/GenAIModel.h"
namespace simaai::neat::pcie::genai::internal {
class RemoteSession {
public:
  RemoteSession(std::string model, ConnectionOptions options);
  ~RemoteSession();
  void start();
  void stop();
  const std::string& id() const {
    return id_;
  }

private:
  std::vector<std::string> ssh(const std::string& script) const;
  std::string model_, id_;
  ConnectionOptions options_;
  bool launched_ = false;
};
} // namespace simaai::neat::pcie::genai::internal
