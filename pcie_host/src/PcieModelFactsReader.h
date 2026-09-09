#pragma once

#include "simaai/neat/pcie/Model.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::internal {

struct PcieTensorFact {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0;
  int tensor_index = -1;
  int physical_index = -1;
  std::int64_t byte_offset = 0;
  std::size_t payload_offset = 0;
  std::vector<std::int64_t> transport_strides_bytes;
  std::size_t transport_size_bytes = 0;
  std::size_t dense_offset = 0;
  std::optional<QuantParams> quant;
};

struct PcieModelFacts {
  std::vector<PcieTensorFact> inputs;
  std::vector<PcieTensorFact> outputs;
  std::size_t packed_input_bytes = 0;
  std::size_t packed_output_bytes = 0;
  bool has_preprocess = false;
  bool has_boxdecode = false;
  std::size_t dense_output_bytes = 0;
};

PcieModelFacts read_model_facts(const std::string& model_path, const ModelOptions& options = {});
ModelInfo to_public_model_info(const PcieModelFacts& facts);

} // namespace simaai::neat::pcie::internal
