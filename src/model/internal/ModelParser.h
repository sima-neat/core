#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/internal/ModelPack.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::internal {

struct ModelSemantics;

enum class ParsedKernelKind : std::uint8_t {
  Unknown = 0,
  Preproc = 1,
  Quantize = 2,
  Tessellate = 3,
  QuantTess = 4,
  Cast = 5,
  DetessDequant = 6,
  Detessellate = 7,
  Dequantize = 8,
  BoxDecode = 9,
};

struct ParsedKernelStage {
  std::string name;
  std::string plugin_id;
  std::string processor;
  std::string kernel;
  ParsedKernelKind kind = ParsedKernelKind::Unknown;
  int sequence = -1;
  bool before_mla = false;
  bool after_mla = false;
};

struct ParsedPhysicalOutput {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0U;
};

struct ParsedLogicalOutput {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t size_bytes = 0U;
  int physical_index = -1;
  std::string source_plugin;
  std::string source_kernel;
  int source_sequence = -1;
};

struct ParsedOutputTopology {
  std::vector<ParsedPhysicalOutput> physical;
  std::vector<ParsedLogicalOutput> logical;
  bool packed_output = false;
};

struct ParsedModelNeeds {
  bool tess_needed = false;
  bool quant_needed = false;
  bool pre_quantization = false;
  bool pre_tessellation = false;
  bool pre_cast = false;
  bool post_detessellation = false;
  bool post_dequantization = false;
  bool post_cast = false;
};

struct ParsedRouteCapabilities {
  bool has_pre_quantization = false;
  bool has_pre_tessellation = false;
  bool has_pre_cast = false;
  bool has_post_detessellation = false;
  bool has_post_dequantization = false;
  bool has_post_cast = false;
  bool has_post_boxdecode = false;
};

struct ParsedModelInfo {
  std::string mpk_json_path;
  std::string model_name;
  std::vector<ParsedKernelStage> plugins;
  std::vector<pipeline_internal::sima::MpkContractEdge> edges;
  std::vector<std::size_t> execution_order;
  int mla_plugin_index = -1;
  ParsedModelNeeds needs;
  ParsedRouteCapabilities capabilities;
  ParsedOutputTopology outputs;
  std::vector<std::string> pre_kernels;
  std::vector<std::string> post_kernels;
  std::vector<std::string> warnings;
};

ParsedModelInfo parse_model_from_pack(const ModelPack& pack);
std::vector<std::string> validate_parsed_model_contract(const ParsedModelInfo& parsed);
bool parse_model_semantics_from_pack(const ModelPack& pack, ModelSemantics* out);

} // namespace simaai::neat::internal
