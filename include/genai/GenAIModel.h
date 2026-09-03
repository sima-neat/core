/**
 * @file
 * @brief Auto-detecting GenAI model handle.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <filesystem>
#include <memory>
#include <string>

namespace simaai::neat::genai {

class GenAIModel {
public:
  explicit GenAIModel(std::filesystem::path model_dir);
  GenAIModel(std::filesystem::path model_dir, GenAIModelOptions options);
  ~GenAIModel();

  GenAIModel(GenAIModel&&) noexcept;
  GenAIModel& operator=(GenAIModel&&) noexcept;

  GenAIModel(const GenAIModel&) = delete;
  GenAIModel& operator=(const GenAIModel&) = delete;

  GenAITask task() const;
  bool accepts_text() const;
  bool accepts_image() const;
  bool accepts_audio() const;
  std::string model_id() const;
  void set_lora(const std::string& adapter_name);
  void unset_lora();
  std::size_t kv_cache_count() const;
  bool remove_kv_cache(const std::string& cache_id);
  void clear_kv_caches();
  std::size_t kv_cache_bytes_per_slot() const;
  GenerationResult run(const GenerationRequest& request);
  GenerationStream stream(const GenerationRequest& request);

private:
  bool supports_thinking() const;

  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class GenAIServer;
};

} // namespace simaai::neat::genai
