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
  std::string add_model(std::filesystem::path model_dir, GenAIModelOptions model_options);
  std::string add_model(std::filesystem::path model_dir, std::string served_name);
  std::string add_model(std::filesystem::path model_dir, std::string served_name,
                        GenAIModelOptions model_options);
  void add_model(std::string served_name, std::shared_ptr<GenAIModel> model);
  bool remove_model(const std::string& served_name);
  std::vector<std::string> model_names() const;
  std::size_t kv_cache_count(const std::string& served_name) const;
  bool remove_kv_cache(const std::string& served_name, const std::string& cache_id);
  void clear_kv_caches(const std::string& served_name);

  void serve();
  void start();
  void stop();

private:
  static bool model_supports_thinking(const GenAIModel& model);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::genai
