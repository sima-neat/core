#pragma once
#include "Service.h"
#include "Protocol.h"
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

namespace simaai::neat::pcie::genai::internal {
std::filesystem::path host_model_path(const std::filesystem::path& model);
std::optional<std::filesystem::path> model_asset_path(const std::filesystem::path& model,
                                                      const std::string& name);
// Handles one on-demand file at a time, independently of token/control traffic.
class ModelAssets {
public:
  ModelAssets(std::filesystem::path model, std::string session, int card);
  ~ModelAssets();
  void close();
  void check() const;

private:
  void receive();
  std::filesystem::path model_;
  std::string session_;
  wire::Service service_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::exception_ptr error_;
};
} // namespace simaai::neat::pcie::genai::internal
