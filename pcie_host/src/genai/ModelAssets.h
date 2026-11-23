#pragma once
#include "Service.h"
#include "Protocol.h"
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

namespace simaai::neat::pcie::genai::internal {
std::filesystem::path host_model_path(const std::filesystem::path& model);
std::optional<std::filesystem::path> model_asset_path(const std::filesystem::path& model,
                                                      const std::string& name);
// The worker's name for a directory under the model: UTF-8 with '/' separators, empty for the root.
std::string model_subdirectory(const std::filesystem::path& model,
                               const std::filesystem::path& directory);
// Handles one on-demand file at a time, independently of token/control traffic.
class ModelAssets {
public:
  ModelAssets(std::filesystem::path model, std::string session, int card);
  ~ModelAssets();
  void close();
  void check() const;

private:
  struct State;
  std::shared_ptr<State> state_;
  std::thread thread_;
};
} // namespace simaai::neat::pcie::genai::internal
