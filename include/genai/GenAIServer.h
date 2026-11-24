/**
 * @file
 * @brief Multi-model HTTP server for NEAT GenAI models.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::genai {

class GenAIModel;

struct GenAIServerOptions {
  std::string host = "0.0.0.0";
  std::uint16_t port = 9998;
};

class GenAIServer {
public:
  explicit GenAIServer(GenAIServerOptions options = {});
  ~GenAIServer();

  GenAIServer(GenAIServer&&) noexcept;
  GenAIServer& operator=(GenAIServer&&) noexcept;

  GenAIServer(const GenAIServer&) = delete;
  GenAIServer& operator=(const GenAIServer&) = delete;

  std::string add_model(std::filesystem::path model_dir);
  std::string add_model(std::filesystem::path model_dir, std::string served_name);
  void add_model(std::string served_name, std::shared_ptr<GenAIModel> model);
  bool remove_model(const std::string& served_name);
  std::vector<std::string> model_names() const;

  void serve();
  /**
   * @brief Start serving in the background and return once the server accepts connections.
   *
   * Registered models are warmed up on the calling thread before the port is bound.
   * @throws std::runtime_error if the server cannot listen on `host:port` (for example, the
   *         port is already in use), if warm-up fails, or if it is already started or running.
   *         The registered models are kept and start() can be called again.
   */
  void start();
  void stop();

private:
  static bool model_supports_thinking(const GenAIModel& model);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::genai
