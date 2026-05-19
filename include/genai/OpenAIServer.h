/**
 * @file
 * @brief Multi-model OpenAI-compatible HTTP server for NEAT GenAI models.
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

struct OpenAIServerOptions {
  std::string host = "0.0.0.0";
  std::uint16_t port = 9998;
};

class OpenAIServer {
public:
  explicit OpenAIServer(OpenAIServerOptions options = {});
  ~OpenAIServer();

  OpenAIServer(OpenAIServer&&) noexcept;
  OpenAIServer& operator=(OpenAIServer&&) noexcept;

  OpenAIServer(const OpenAIServer&) = delete;
  OpenAIServer& operator=(const OpenAIServer&) = delete;

  std::string add_model(std::filesystem::path model_dir);
  std::string add_model(std::filesystem::path model_dir, std::string served_name);
  void add_model(std::string served_name, std::shared_ptr<GenAIModel> model);
  bool remove_model(const std::string& served_name);
  std::vector<std::string> model_names() const;

  void serve();
  void start();
  void stop();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::genai
