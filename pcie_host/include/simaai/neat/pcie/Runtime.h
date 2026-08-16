#pragma once

#include "simaai/neat/pcie/Model.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie {

using ModelId = std::int32_t;
using RequestId = std::int32_t;

struct ModelConfig {
  std::string path;
  ModelOptions options;
};

struct Completion {
  ModelId model_id = 0;
  RequestId request_id = 0;
  TensorList outputs;
};

enum class EnqueueResult {
  Accepted,
  Full,
};

// A Runtime represents one Modalix card. It assigns one queue to each loaded
// model and hides that physical mapping behind stable model IDs.
class Runtime {
public:
  explicit Runtime(ConnectionOptions connection = {});
  ~Runtime() noexcept;

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  ModelId load(std::string model_path, ModelOptions options = {},
               int readiness_timeout_ms = 180000);
  std::vector<ModelId> load_models(const std::vector<ModelConfig>& models,
                                   int readiness_timeout_ms = 180000);

  EnqueueResult try_enqueue(ModelId model_id, RequestId request_id, const Tensor& tensor);
  EnqueueResult try_enqueue(ModelId model_id, RequestId request_id, const TensorList& tensors);
  std::optional<Completion> retrieve(int timeout_ms = -1);

  void unload(ModelId model_id, int drain_timeout_ms = 30000);
  void close();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat::pcie
