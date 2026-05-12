#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/EnvUtil.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string normalize_dtype_local(const std::string& raw_dtype);
std::optional<std::string> infer_dtype_from_shape_and_size(const MpkTensorContract& tensor);
std::optional<std::vector<std::int64_t>>
primary_input_shape_local(const MpkPluginIoContract& stage);
std::optional<std::vector<std::int64_t>>
primary_output_shape_local(const MpkPluginIoContract& stage);
std::optional<int> resolved_batch_size_local(const MpkPluginIoContract& stage);
bool nhwc_dims_local(const std::vector<std::int64_t>& shape, int* out_h, int* out_w, int* out_c);

std::string lower_copy_local(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string upper_copy_local(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

bool mpk_contract_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_MPK_CONTRACT_DEBUG", false);
}

bool mpk_contract_compare_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* raw = std::getenv("SIMA_MPK_CONTRACT_COMPARE");
  cached = (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  return cached == 1;
}

bool mpk_graph_dump_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* raw = std::getenv("SIMA_MPK_GRAPH_DUMP");
  cached = (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  return cached == 1;
}

bool mpk_graph_exit_after_dump_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* raw = std::getenv("SIMA_MPK_GRAPH_EXIT_AFTER_DUMP");
  cached = (raw && *raw && std::strcmp(raw, "0") != 0) ? 1 : 0;
  return cached == 1;
}

bool env_flag_enabled_local(const char* name, const bool default_enabled) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_enabled;
  }
  return std::strcmp(raw, "0") != 0;
}

bool mpk_graph_fuse_quanttess_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  cached = env_flag_enabled_local("SIMA_MPK_GRAPH_FUSE_QUANTTESS", true) ? 1 : 0;
  return cached == 1;
}

bool mpk_graph_fuse_detessdequant_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  cached = env_flag_enabled_local("SIMA_MPK_GRAPH_FUSE_DETESSDEQUANT", true) ? 1 : 0;
  return cached == 1;
}

bool mpk_graph_fuse_preproc_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  cached = env_flag_enabled_local("SIMA_MPK_GRAPH_FUSE_PREPROC", false) ? 1 : 0;
  return cached == 1;
}

bool mpk_graph_fuse_boxdecode_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  cached = env_flag_enabled_local("SIMA_MPK_GRAPH_FUSE_BOXDECODE", false) ? 1 : 0;
  return cached == 1;
}

const char* mpk_plugin_name_dbg(const MpkPluginIoContract& plugin) {
  if (!plugin.name.empty()) {
    return plugin.name.c_str();
  }
  if (!plugin.plugin_id.empty()) {
    return plugin.plugin_id.c_str();
  }
  return "<unnamed>";
}

const char* mpk_shape_semantics_dbg_local(const MpkShapeSemantics semantics) {
  switch (semantics) {
  case MpkShapeSemantics::Unknown:
    return "Unknown";
  case MpkShapeSemantics::Geometry:
    return "Geometry";
  case MpkShapeSemantics::PackedExtent:
    return "PackedExtent";
  }
  return "Unknown";
}

std::string json_compact_dbg_local(const json& value) {
  try {
    return value.dump();
  } catch (const std::exception&) {
    return "\"<json-dump-error>\"";
  }
}

std::string canonical_token_local(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  return out;
}

std::string graph_id_token_local(const std::string& raw) {
  if (raw.empty()) {
    return "unnamed";
  }
  std::string out;
  out.reserve(raw.size());
  for (unsigned char c : raw) {
    if (std::isalnum(c)) {
      out.push_back(static_cast<char>(std::tolower(c)));
      continue;
    }
    out.push_back('_');
  }
  return out;
}

bool is_logical_consumer_transform_kernel(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  if (token.empty()) {
    return false;
  }
  return token.find("detess") != std::string::npos || token.find("dequant") != std::string::npos ||
         token.find("unpack") != std::string::npos || token.find("slice") != std::string::npos ||
         token.find("cast") != std::string::npos || token.find("flatten") != std::string::npos ||
         token.find("reshape") != std::string::npos ||
         token.find("transpose") != std::string::npos ||
         token.find("permute") != std::string::npos || token.find("squeeze") != std::string::npos;
}

bool is_dtype_preserving_transform_kernel(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  if (token.empty()) {
    return false;
  }
  if (token.find("dequant") != std::string::npos || token.find("quantize") != std::string::npos ||
      token.find("cast") != std::string::npos) {
    return false;
  }
  return token.find("detess") != std::string::npos || token.find("unpack") != std::string::npos ||
         token.find("slice") != std::string::npos || token.find("flatten") != std::string::npos ||
         token.find("reshape") != std::string::npos ||
         token.find("transpose") != std::string::npos ||
         token.find("permute") != std::string::npos || token.find("squeeze") != std::string::npos;
}

bool is_tessellate_producer_kernel(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  if (token.empty()) {
    return false;
  }
  if (token.find("detess") != std::string::npos) {
    return false;
  }
  return token.find("tessellate") != std::string::npos ||
         token.find("tessellation") != std::string::npos ||
         token.find("quanttess") != std::string::npos;
}

bool is_tess_like_source_kernel(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  if (token.empty()) {
    return false;
  }
  if (token.find("detess") != std::string::npos) {
    return false;
  }
  return token.find("tess") != std::string::npos || token.find("quanttess") != std::string::npos;
}

bool is_geometry_shape_semantics_local(const MpkShapeSemantics semantics) {
  return semantics == MpkShapeSemantics::Geometry;
}

MpkShapeSemantics classify_mpk_tensor_shape_semantics_local(const MpkPluginIoContract& stage,
                                                            const bool is_input) {
  const std::string kernel = canonical_token_local(stage.kernel);
  if (kernel.empty()) {
    return MpkShapeSemantics::Unknown;
  }
  if (kernel == "mla" || kernel.find("mla") != std::string::npos) {
    return MpkShapeSemantics::PackedExtent;
  }
  if (kernel.find("detess") != std::string::npos) {
    return is_input ? MpkShapeSemantics::PackedExtent : MpkShapeSemantics::Geometry;
  }
  if (kernel.find("unpack") != std::string::npos) {
    return MpkShapeSemantics::PackedExtent;
  }
  if (kernel.find("slice") != std::string::npos) {
    return MpkShapeSemantics::Geometry;
  }
  if (kernel.find("quanttess") != std::string::npos) {
    return is_input ? MpkShapeSemantics::Geometry : MpkShapeSemantics::PackedExtent;
  }
  if (kernel.find("tess") != std::string::npos) {
    return is_input ? MpkShapeSemantics::Geometry : MpkShapeSemantics::PackedExtent;
  }
  if (kernel.find("preproc") != std::string::npos || kernel.find("resize") != std::string::npos ||
      kernel.find("quant") != std::string::npos || kernel.find("dequant") != std::string::npos ||
      kernel.find("cast") != std::string::npos || kernel.find("boxdecode") != std::string::npos ||
      kernel.find("flatten") != std::string::npos || kernel.find("reshape") != std::string::npos ||
      kernel.find("transpose") != std::string::npos ||
      kernel.find("permute") != std::string::npos || kernel.find("squeeze") != std::string::npos) {
    return MpkShapeSemantics::Geometry;
  }
  return MpkShapeSemantics::Unknown;
}

bool is_placeholder_kernel_token(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  return token.empty() || token == "kernelnametbd" || token == "tbd" || token == "unknown";
}

std::string infer_kernel_from_stage_metadata(const MpkPluginIoContract& stage) {
  const auto classify = [](const std::string& raw) -> std::string {
    const std::string token = canonical_token_local(raw);
    if (token.empty()) {
      return {};
    }
    if (token.find("detessdequant") != std::string::npos ||
        (token.find("detess") != std::string::npos && token.find("dequant") != std::string::npos)) {
      return "detessdequant";
    }
    if (token.find("detess") != std::string::npos ||
        token.find("detessellate") != std::string::npos ||
        token.find("detessellation") != std::string::npos) {
      return "detessellate";
    }
    if (token.find("dequant") != std::string::npos) {
      return "dequantize";
    }
    if (token.find("quanttess") != std::string::npos) {
      return "quanttess";
    }
    if (token.find("tessellate") != std::string::npos ||
        token.find("tessellation") != std::string::npos) {
      return "tessellate";
    }
    if (token.find("quantize") != std::string::npos ||
        token.find("quantization") != std::string::npos) {
      return "quantize";
    }
    if (token.find("preproc") != std::string::npos || token.find("resize") != std::string::npos) {
      return "preproc";
    }
    if (token.find("cast") != std::string::npos) {
      return "cast";
    }
    if (token.find("slice") != std::string::npos) {
      return "slice";
    }
    if (token.find("mla") != std::string::npos) {
      return "mla";
    }
    return {};
  };

  if (const std::string k = classify(stage.name); !k.empty()) {
    return k;
  }
  if (const std::string k = classify(stage.plugin_id); !k.empty()) {
    return k;
  }
  if (const std::string k = classify(stage.executable); !k.empty()) {
    return k;
  }
  return {};
}

bool ends_with_local(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

bool read_json_file_local(const fs::path& path, json* out) {
  if (!out || path.empty()) {
    return false;
  }
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return false;
  }
  try {
    in >> *out;
  } catch (const std::exception&) {
    return false;
  }
  return out->is_object();
}

std::optional<int> read_int_local(const json& value) {
  if (value.is_number_integer()) {
    const auto raw = value.get<std::int64_t>();
    if (raw >= static_cast<std::int64_t>(std::numeric_limits<int>::min()) &&
        raw <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      return static_cast<int>(raw);
    }
    return std::nullopt;
  }
  if (value.is_number()) {
    const double raw = value.get<double>();
    if (raw >= static_cast<double>(std::numeric_limits<int>::min()) &&
        raw <= static_cast<double>(std::numeric_limits<int>::max())) {
      return static_cast<int>(raw);
    }
  }
  return std::nullopt;
}

std::optional<bool> read_bool_local(const json& value) {
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_number_integer()) {
    return value.get<std::int64_t>() != 0;
  }
  if (value.is_number()) {
    return value.get<double>() != 0.0;
  }
  if (value.is_string()) {
    std::string token = lower_copy_local(value.get<std::string>());
    if (token == "1" || token == "true" || token == "yes" || token == "on") {
      return true;
    }
    if (token == "0" || token == "false" || token == "no" || token == "off") {
      return false;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> read_size_local(const json& value) {
  if (value.is_number_integer()) {
    const auto raw = value.get<std::int64_t>();
    if (raw > 0 && static_cast<std::uint64_t>(raw) <=
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return static_cast<std::size_t>(raw);
    }
    return std::nullopt;
  }
  if (value.is_number()) {
    const double raw = value.get<double>();
    if (raw > 0.0 && raw <= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
      return static_cast<std::size_t>(raw);
    }
  }
  return std::nullopt;
}

template <typename T> std::vector<T> read_numeric_values_any(const json& value) {
  std::vector<T> out;
  if (value.is_array()) {
    out.reserve(value.size());
    for (const auto& item : value) {
      if (item.is_number_integer()) {
        out.push_back(static_cast<T>(item.get<std::int64_t>()));
      } else if (item.is_number()) {
        out.push_back(static_cast<T>(item.get<double>()));
      }
    }
    return out;
  }
  if (value.is_number_integer()) {
    out.push_back(static_cast<T>(value.get<std::int64_t>()));
  } else if (value.is_number()) {
    out.push_back(static_cast<T>(value.get<double>()));
  }
  return out;
}

std::vector<std::string> read_string_values_any(const json& value) {
  std::vector<std::string> out;
  if (value.is_array()) {
    out.reserve(value.size());
    for (const auto& item : value) {
      if (item.is_string()) {
        out.push_back(item.get<std::string>());
      }
    }
    return out;
  }
  if (value.is_string()) {
    out.push_back(value.get<std::string>());
  }
  return out;
}

std::optional<std::string> read_string_alias(const json& obj,
                                             std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key || !obj.contains(key)) {
      continue;
    }
    if (!obj.at(key).is_string()) {
      continue;
    }
    const std::string value = obj.at(key).get<std::string>();
    if (!value.empty()) {
      return value;
    }
  }
  return std::nullopt;
}

std::vector<std::string> read_string_alias_values(const json& obj,
                                                  std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key || !obj.contains(key)) {
      continue;
    }
    const auto values = read_string_values_any(obj.at(key));
    if (!values.empty()) {
      return values;
    }
  }
  return {};
}

std::vector<std::vector<std::int64_t>> read_shape_values_any(const json& value) {
  std::vector<std::vector<std::int64_t>> out;
  if (!value.is_array()) {
    return out;
  }
  out.reserve(value.size());
  for (const auto& item : value) {
    if (item.is_array()) {
      auto shape = read_numeric_values_any<std::int64_t>(item);
      if (!shape.empty()) {
        out.push_back(std::move(shape));
      }
    } else {
      auto shape = read_numeric_values_any<std::int64_t>(item);
      if (!shape.empty()) {
        out.push_back(std::move(shape));
      }
    }
  }
  return out;
}

std::vector<std::vector<std::int64_t>> read_shape_alias(const json& obj,
                                                        std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key || !obj.contains(key)) {
      continue;
    }
    const auto shapes = read_shape_values_any(obj.at(key));
    if (!shapes.empty()) {
      return shapes;
    }
  }
  return {};
}

std::vector<std::int64_t> read_shape_vector_values_any(const json& value) {
  if (!value.is_array()) {
    return {};
  }
  if (!value.empty() && value.front().is_array()) {
    return read_numeric_values_any<std::int64_t>(value.front());
  }
  return read_numeric_values_any<std::int64_t>(value);
}

std::vector<std::int64_t> read_shape_vector_alias(const json& obj,
                                                  std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (!key || !*key || !obj.contains(key)) {
      continue;
    }
    auto shape = read_shape_vector_values_any(obj.at(key));
    if (!shape.empty()) {
      return shape;
    }
  }
  return {};
}

bool canonical_nhwc_from_shape(const std::vector<std::int64_t>& shape, int* out_n, int* out_h,
                               int* out_w, int* out_c) {
  if (!out_n || !out_h || !out_w || !out_c || shape.empty()) {
    return false;
  }
  auto to_pos = [](std::int64_t value) -> int {
    if (value <= 0) {
      return 0;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
  };

  *out_n = 1;
  *out_h = 0;
  *out_w = 0;
  *out_c = 0;
  if (shape.size() >= 4U) {
    *out_n = to_pos(shape[shape.size() - 4U]);
    *out_h = to_pos(shape[shape.size() - 3U]);
    *out_w = to_pos(shape[shape.size() - 2U]);
    *out_c = to_pos(shape[shape.size() - 1U]);
  } else if (shape.size() == 3U) {
    *out_h = to_pos(shape[0]);
    *out_w = to_pos(shape[1]);
    *out_c = to_pos(shape[2]);
  } else if (shape.size() == 2U) {
    *out_h = to_pos(shape[0]);
    *out_w = to_pos(shape[1]);
    *out_c = 1;
  } else {
    *out_h = 1;
    *out_w = to_pos(shape[0]);
    *out_c = 1;
  }
  if (*out_n <= 0) {
    *out_n = 1;
  }
  return *out_h > 0 && *out_w > 0 && *out_c > 0;
}

bool canonical_slice_dhwc_from_shape(const std::vector<std::int64_t>& shape, int* out_d, int* out_h,
                                     int* out_w, int* out_c) {
  if (!out_d || !out_h || !out_w || !out_c) {
    return false;
  }
  *out_d = 1;
  *out_h = 0;
  *out_w = 0;
  *out_c = 0;
  auto to_pos = [](std::int64_t value) -> int {
    if (value <= 0) {
      return 0;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
  };
  if (shape.size() == 4U) {
    *out_d = to_pos(shape[0]);
    *out_h = to_pos(shape[1]);
    *out_w = to_pos(shape[2]);
    *out_c = to_pos(shape[3]);
  } else if (shape.size() == 3U) {
    *out_d = 1;
    *out_h = to_pos(shape[0]);
    *out_w = to_pos(shape[1]);
    *out_c = to_pos(shape[2]);
  } else {
    return false;
  }
  if (*out_d <= 0) {
    *out_d = 1;
  }
  return *out_h > 0 && *out_w > 0 && *out_c > 0;
}

std::string normalize_dtype_local(const std::string& raw_dtype) {
  if (raw_dtype.empty()) {
    return {};
  }
  const std::string up = upper_copy_local(raw_dtype);
  if (up.find("BFLOAT16") != std::string::npos || up.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (up.find("FLOAT32") != std::string::npos || up == "FP32" || up == "F32") {
    return "FP32";
  }
  if (up.find("FLOAT16") != std::string::npos || up == "FP16" || up == "F16") {
    return "FP16";
  }
  if (up.find("UINT8") != std::string::npos || up == "U8") {
    return "UINT8";
  }
  if (up.find("INT8") != std::string::npos || up == "S8") {
    return "INT8";
  }
  if (up.find("UINT16") != std::string::npos || up == "U16") {
    return "UINT16";
  }
  if (up.find("INT16") != std::string::npos || up == "S16") {
    return "INT16";
  }
  if (up.find("UINT32") != std::string::npos || up == "U32") {
    return "UINT32";
  }
  if (up.find("INT32") != std::string::npos || up == "S32") {
    return "INT32";
  }
  return up;
}

std::optional<std::string> infer_dtype_from_shape_and_size(const MpkTensorContract& tensor) {
  if (!tensor.dtype.empty() || tensor.mpk_shape.empty() || tensor.size_bytes == 0U ||
      !is_geometry_shape_semantics_local(tensor.shape_semantics)) {
    return std::nullopt;
  }
  std::size_t elements = 1U;
  for (const auto dim : tensor.mpk_shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (elements > std::numeric_limits<std::size_t>::max() / u_dim) {
      return std::nullopt;
    }
    elements *= u_dim;
  }
  if (elements == 0U || tensor.size_bytes % elements != 0U) {
    return std::nullopt;
  }
  const std::size_t bytes_per_element = tensor.size_bytes / elements;
  if (bytes_per_element == 1U) {
    return std::string("INT8");
  }
  if (bytes_per_element == 2U) {
    return std::string("BF16");
  }
  if (bytes_per_element == 4U) {
    return std::string("FP32");
  }
  return std::nullopt;
}

void finalize_tensor_contract(MpkTensorContract* tensor) {
  if (!tensor) {
    return;
  }
  tensor->dtype = normalize_dtype_local(tensor->dtype);
  if (tensor->dtype.empty()) {
    if (const auto inferred = infer_dtype_from_shape_and_size(*tensor); inferred.has_value()) {
      tensor->dtype = *inferred;
    }
  }
}

std::size_t dtype_size_bytes_local(const std::string& raw_dtype) {
  const std::string dtype = normalize_dtype_local(raw_dtype);
  if (dtype == "BF16" || dtype == "FP16" || dtype == "INT16" || dtype == "UINT16") {
    return 2U;
  }
  if (dtype == "FP32" || dtype == "INT32" || dtype == "UINT32") {
    return 4U;
  }
  return 1U;
}

std::optional<std::size_t> dense_shape_size_bytes_local(const std::vector<std::int64_t>& shape,
                                                        const std::string& dtype) {
  if (shape.empty()) {
    return std::nullopt;
  }
  std::size_t total = dtype_size_bytes_local(dtype);
  for (const auto dim : shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (u_dim > 0U && total > std::numeric_limits<std::size_t>::max() / u_dim) {
      return std::nullopt;
    }
    total *= u_dim;
  }
  return total;
}

bool shape_starts_with_batch_local(const std::vector<std::int64_t>& shape, const int batch) {
  return batch > 1 && !shape.empty() && shape.front() == static_cast<std::int64_t>(batch);
}

std::vector<std::int64_t> prepend_actual_batch_local(std::vector<std::int64_t> shape,
                                                     const int actual_batch) {
  if (actual_batch <= 1 || shape.empty() || shape_starts_with_batch_local(shape, actual_batch)) {
    return shape;
  }
  if (shape.front() == 1 && shape.size() > 1U) {
    shape.front() = actual_batch;
    return shape;
  }
  shape.insert(shape.begin(), actual_batch);
  return shape;
}

void normalize_batched_view_stage_outputs_local(MpkPluginIoContract* stage) {
  if (!stage || stage->batch_sz_model <= 1 || stage->output_tensors.empty()) {
    return;
  }
  const std::string token =
      canonical_token_local(!stage->kernel.empty() ? stage->kernel : stage->name);
  const bool is_slice_stage = token.find("slice") != std::string::npos;
  const bool is_batch_flatten_stage = token.find("batchflatten") != std::string::npos;
  if (!is_slice_stage && !is_batch_flatten_stage) {
    return;
  }
  if (is_slice_stage && !stage->input_tensors.empty() &&
      !shape_starts_with_batch_local(stage->input_tensors.front().mpk_shape,
                                     stage->batch_sz_model)) {
    return;
  }

  for (auto& tensor : stage->output_tensors) {
    if (tensor.mpk_shape.empty() ||
        shape_starts_with_batch_local(tensor.mpk_shape, stage->batch_sz_model)) {
      continue;
    }
    const auto per_item_bytes = dense_shape_size_bytes_local(tensor.mpk_shape, tensor.dtype);
    if (!per_item_bytes.has_value() || tensor.size_bytes != *per_item_bytes) {
      continue;
    }
    tensor.mpk_shape = prepend_actual_batch_local(tensor.mpk_shape, stage->batch_sz_model);
    tensor.size_bytes = *per_item_bytes * static_cast<std::size_t>(stage->batch_sz_model);
    tensor.shape_semantics = MpkShapeSemantics::Geometry;
  }
}

std::uint64_t round_up_to_multiple_local(const std::uint64_t value, const std::uint64_t multiple) {
  if (multiple == 0U) {
    return value;
  }
  const std::uint64_t remainder = value % multiple;
  if (remainder == 0U) {
    return value;
  }
  return value + (multiple - remainder);
}

std::uint64_t expected_detess_packed_input_size_bytes_local(const MpkPluginIoContract& stage,
                                                            const std::string& dtype) {
  if (stage.frame_shape.empty() || dtype.empty()) {
    return 0U;
  }

  std::vector<std::int64_t> shape = stage.frame_shape;
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  if (shape.size() < 3U) {
    return 0U;
  }

  std::uint64_t batch = 1U;
  for (std::size_t i = 0; i + 3U < shape.size(); ++i) {
    const auto dim = static_cast<std::uint64_t>(std::max<std::int64_t>(shape[i], 1));
    if (dim == 0U || batch > std::numeric_limits<std::uint64_t>::max() / dim) {
      return 0U;
    }
    batch *= dim;
  }

  const std::int64_t height = shape[shape.size() - 3U];
  const std::int64_t width = shape[shape.size() - 2U];
  const std::int64_t depth = shape.back();
  if (height <= 0 || width <= 0 || depth <= 0) {
    return 0U;
  }

  std::uint64_t channels = static_cast<std::uint64_t>(depth);
  if ((stage.has_align_c16 && stage.align_c16) || (stage.has_cblock && stage.cblock)) {
    channels = round_up_to_multiple_local(channels, 16U);
  }

  const std::uint64_t elem_bytes = dtype_size_bytes_local(dtype);
  if (elem_bytes == 0U) {
    return 0U;
  }
  const std::uint64_t factors[] = {batch, static_cast<std::uint64_t>(height),
                                   static_cast<std::uint64_t>(width), channels, elem_bytes};
  std::uint64_t total = 1U;
  for (const std::uint64_t factor : factors) {
    if (factor == 0U || total > std::numeric_limits<std::uint64_t>::max() / factor) {
      return 0U;
    }
    total *= factor;
  }
  return total;
}

std::vector<std::int64_t>
canonical_detess_transport_shape_local(const MpkPluginIoContract& stage,
                                       const MpkTensorContract& tensor,
                                       const std::string& dtype_override) {
  const std::string dtype =
      normalize_dtype_local(dtype_override.empty() ? stage.frame_type : dtype_override);
  if (stage.frame_shape.empty() || dtype.empty()) {
    throw std::runtime_error("detess transport shape requires frame_shape and frame_type for '" +
                             stage.name + "'");
  }

  std::vector<std::int64_t> shape = stage.frame_shape;
  if (!shape.empty() && shape.front() == 1 && shape.size() > 1U) {
    shape.erase(shape.begin());
  }
  if (shape.size() < 3U) {
    throw std::runtime_error("detess transport shape requires canonical frame geometry for '" +
                             stage.name + "'");
  }

  std::uint64_t batch = 1U;
  for (std::size_t i = 0; i + 3U < shape.size(); ++i) {
    const auto dim = static_cast<std::uint64_t>(std::max<std::int64_t>(shape[i], 1));
    batch *= dim;
  }

  const std::int64_t height = shape[shape.size() - 3U];
  const std::int64_t width = shape[shape.size() - 2U];
  const std::int64_t logical_channels = shape.back();
  if (height <= 0 || width <= 0 || logical_channels <= 0) {
    throw std::runtime_error("detess transport shape contains non-positive dimensions for '" +
                             stage.name + "'");
  }

  std::uint64_t packed_channels = 0U;
  const auto elem_bytes = static_cast<std::uint64_t>(dtype_size_bytes_local(dtype));
  std::uint64_t spatial_elems = batch;
  if (static_cast<std::uint64_t>(height) > 0U &&
      spatial_elems >
          std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(height)) {
    throw std::runtime_error("detess transport shape overflow for '" + stage.name + "'");
  }
  spatial_elems *= static_cast<std::uint64_t>(height);
  if (static_cast<std::uint64_t>(width) > 0U &&
      spatial_elems >
          std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(width)) {
    throw std::runtime_error("detess transport shape overflow for '" + stage.name + "'");
  }
  spatial_elems *= static_cast<std::uint64_t>(width);
  if (elem_bytes > 0U && spatial_elems > std::numeric_limits<std::uint64_t>::max() / elem_bytes) {
    throw std::runtime_error("detess transport shape overflow for '" + stage.name + "'");
  }
  spatial_elems *= elem_bytes;
  if (spatial_elems == 0U) {
    throw std::runtime_error("detess transport shape overflow for '" + stage.name + "'");
  }
  if (tensor.size_bytes > 0U) {
    if (tensor.size_bytes % spatial_elems != 0U) {
      throw std::runtime_error(
          "detess transport shape bytes are not divisible by frame geometry for '" + stage.name +
          "'");
    }
    packed_channels = static_cast<std::uint64_t>(tensor.size_bytes / spatial_elems);
  } else {
    packed_channels = static_cast<std::uint64_t>(logical_channels);
    if ((stage.has_align_c16 && stage.align_c16) || (stage.has_cblock && stage.cblock)) {
      packed_channels = round_up_to_multiple_local(packed_channels, 16U);
    }
  }
  if (packed_channels < static_cast<std::uint64_t>(logical_channels)) {
    throw std::runtime_error(
        "detess transport shape packed channels are smaller than logical channels for '" +
        stage.name + "'");
  }
  if (((stage.has_align_c16 && stage.align_c16) || (stage.has_cblock && stage.cblock)) &&
      (packed_channels % 16U) != 0U) {
    throw std::runtime_error("detess transport shape expected c16-aligned packed channels for '" +
                             stage.name + "'");
  }

  std::vector<std::int64_t> out;
  if (batch != 1U) {
    out.push_back(static_cast<std::int64_t>(batch));
  }
  out.push_back(height);
  out.push_back(width);
  out.push_back(static_cast<std::int64_t>(packed_channels));

  const auto dense_bytes = dense_shape_size_bytes_local(out, dtype);
  if (!dense_bytes.has_value() || *dense_bytes != tensor.size_bytes) {
    throw std::runtime_error(
        "detess transport shape bytes mismatch for '" + stage.name +
        "': shape-derived=" + std::to_string(dense_bytes.has_value() ? *dense_bytes : 0U) +
        " tensor=" + std::to_string(tensor.size_bytes));
  }
  return out;
}

std::vector<std::int64_t> contiguous_stride_bytes_local(const std::vector<std::int64_t>& shape,
                                                        const std::string& dtype) {
  std::vector<std::int64_t> strides;
  if (shape.empty()) {
    return strides;
  }
  strides.assign(shape.size(), 0);
  std::int64_t running = static_cast<std::int64_t>(dtype_size_bytes_local(dtype));
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = running;
    const auto dim = shape[i];
    if (dim > 0 && running <= std::numeric_limits<std::int64_t>::max() / dim) {
      running *= dim;
    } else if (dim > 0) {
      running = std::numeric_limits<std::int64_t>::max();
    }
  }
  return strides;
}

std::string ints_dbg_local(const std::vector<std::int64_t>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

std::string ints32_dbg_local(const std::vector<int>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

std::string doubles_dbg_local(const std::vector<double>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string strings_dbg_local(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << "\"" << values[i] << "\"";
  }
  out << "]";
  return out.str();
}

std::string nested_shapes_dbg_local(const std::vector<std::vector<std::int64_t>>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << ints_dbg_local(values[i]);
  }
  out << "]";
  return out.str();
}

std::string sizes_dbg_local(const std::vector<std::size_t>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0U) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

std::string tensor_dbg_local(const MpkTensorContract& tensor) {
  std::ostringstream out;
  out << "{tensor_index=" << tensor.tensor_index << ",physical_index=" << tensor.physical_index
      << ",source_physical_index=" << tensor.source_physical_index << ",name=\"" << tensor.name
      << "\""
      << ",segment_name=\"" << tensor.segment_name << "\""
      << ",kind=\"" << tensor.kind << "\""
      << ",dtype=\"" << tensor.dtype << "\""
      << ",mpk_shape=" << ints_dbg_local(tensor.mpk_shape)
      << ",shape_semantics=" << mpk_shape_semantics_dbg_local(tensor.shape_semantics)
      << ",size_bytes=" << tensor.size_bytes << ",byte_offset=" << tensor.byte_offset
      << ",source_byte_offset=" << tensor.source_byte_offset
      << ",stride_bytes=" << ints_dbg_local(tensor.stride_bytes)
      << ",logical_shape=" << ints_dbg_local(tensor.logical_shape) << ",logical_dtype=\""
      << tensor.logical_dtype << "\""
      << ",logical_source_plugin=\"" << tensor.logical_source_plugin << "\""
      << ",logical_source_kernel=\"" << tensor.logical_source_kernel << "\""
      << ",logical_source_sequence=" << tensor.logical_source_sequence << "}";
  return out.str();
}

std::string quant_dbg_local(const std::optional<MpkQuantContract>& quant) {
  if (!quant.has_value()) {
    return "<none>";
  }
  std::ostringstream out;
  out << "{scales=" << doubles_dbg_local(quant->scales)
      << ",zero_points=" << ints_dbg_local(quant->zero_points) << ",axis=" << quant->axis << "}";
  return out.str();
}

std::string plugin_dbg_local(const MpkPluginIoContract& stage) {
  std::ostringstream out;
  out << "{name=\"" << stage.name << "\""
      << ",plugin_id=\"" << stage.plugin_id << "\""
      << ",processor=\"" << stage.processor << "\""
      << ",kernel=\"" << stage.kernel << "\""
      << ",executable=\"" << stage.executable << "\""
      << ",batch_size=" << stage.batch_size << ",batch_sz_model=" << stage.batch_sz_model
      << ",sequence=" << stage.sequence << ",slice_shape=" << ints_dbg_local(stage.slice_shape)
      << ",slice_begin=" << ints_dbg_local(stage.slice_begin)
      << ",slice_end=" << ints_dbg_local(stage.slice_end)
      << ",frame_shape=" << ints_dbg_local(stage.frame_shape) << ",frame_type=\""
      << stage.frame_type << "\""
      << ",round_off=\"" << stage.round_off << "\""
      << ",canonical_contract=" << (stage.has_canonical_processcvu_contract ? 1 : 0)
      << ",out_shape_raw=" << ints_dbg_local(stage.out_shape_raw) << ",canonical_input_dtype=\""
      << stage.canonical_input_dtype << "\""
      << ",canonical_output_dtype=\"" << stage.canonical_output_dtype << "\""
      << ",has_align_c16=" << (stage.has_align_c16 ? 1 : 0)
      << ",align_c16=" << (stage.align_c16 ? 1 : 0) << ",has_cblock=" << (stage.has_cblock ? 1 : 0)
      << ",cblock=" << (stage.cblock ? 1 : 0) << ",quant=" << quant_dbg_local(stage.quant) << "}";
  return out.str();
}

void dump_tensor_list_compare_local(const char* label, const std::string& stage_name,
                                    const json* raw_nodes,
                                    const std::vector<MpkTensorContract>& tensors) {
  const std::size_t raw_count = (raw_nodes && raw_nodes->is_array()) ? raw_nodes->size() : 0U;
  const std::size_t count = std::max(raw_count, tensors.size());
  for (std::size_t i = 0; i < count; ++i) {
    const std::string raw_value = (raw_nodes && raw_nodes->is_array() && i < raw_nodes->size())
                                      ? json_compact_dbg_local(raw_nodes->at(i))
                                      : "<missing>";
    const std::string parsed_value =
        (i < tensors.size()) ? tensor_dbg_local(tensors[i]) : "<missing>";
    std::fprintf(stderr, "[mpk-compare] scope=final stage=%s list=%s index=%zu raw=%s parsed=%s\n",
                 stage_name.c_str(), label, i, raw_value.c_str(), parsed_value.c_str());
  }
}

void dump_named_tensor_view_list_local(const char* label,
                                       const std::vector<MpkTensorContract>& tensors) {
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    std::fprintf(stderr, "[mpk-compare] scope=final view=%s index=%zu parsed=%s\n", label, i,
                 tensor_dbg_local(tensors[i]).c_str());
  }
}

std::vector<std::int64_t>
normalize_stride_rank_to_shape_local(const std::vector<std::int64_t>& strides,
                                     const std::vector<std::int64_t>& source_shape,
                                     const std::vector<std::int64_t>& target_shape) {
  if (strides.empty() || target_shape.empty() || strides.size() == target_shape.size()) {
    return strides;
  }
  if (source_shape.size() != strides.size() || source_shape.size() < target_shape.size()) {
    return strides;
  }

  const std::size_t drop = source_shape.size() - target_shape.size();
  for (std::size_t i = 0; i < drop; ++i) {
    if (source_shape[i] != 1) {
      return strides;
    }
  }
  for (std::size_t i = 0; i < target_shape.size(); ++i) {
    if (target_shape[i] <= 0 || source_shape[i + drop] < target_shape[i]) {
      return strides;
    }
  }
  return std::vector<std::int64_t>(strides.begin() + static_cast<std::ptrdiff_t>(drop),
                                   strides.end());
}

std::vector<std::int64_t>
flatten_batched_view_stride_local(const std::vector<std::int64_t>& strides,
                                  const std::vector<std::int64_t>& source_shape,
                                  const std::vector<std::int64_t>& target_shape) {
  if (strides.size() == source_shape.size() && target_shape.size() == 2U &&
      source_shape.size() >= 2U && !source_shape.empty() &&
      source_shape.front() == target_shape.front()) {
    return {strides.front(), strides.back()};
  }
  return normalize_stride_rank_to_shape_local(strides, source_shape, target_shape);
}

void recompute_dense_tensor_contract_geometry_local(MpkTensorContract* tensor,
                                                    const std::size_t expected_size_bytes,
                                                    const std::string& context) {
  if (!tensor) {
    throw std::runtime_error("dense tensor contract recompute requires a tensor for '" + context +
                             "'");
  }
  finalize_tensor_contract(tensor);
  const auto dense_bytes = dense_shape_size_bytes_local(tensor->mpk_shape, tensor->dtype);
  if (!dense_bytes.has_value()) {
    throw std::runtime_error(
        "dense tensor contract recompute requires canonical shape/dtype for '" + context + "'");
  }
  if (expected_size_bytes > 0U && *dense_bytes != expected_size_bytes) {
    throw std::runtime_error("dense tensor contract bytes mismatch for '" + context +
                             "': shape-derived=" + std::to_string(*dense_bytes) +
                             " source=" + std::to_string(expected_size_bytes));
  }
  tensor->size_bytes = *dense_bytes;
  tensor->stride_bytes = contiguous_stride_bytes_local(tensor->mpk_shape, tensor->dtype);
}

void set_dense_tensor_contract_size_preserve_stride_local(MpkTensorContract* tensor,
                                                          const std::size_t expected_size_bytes,
                                                          const std::string& context) {
  if (!tensor) {
    throw std::runtime_error("dense tensor contract size update requires a tensor for '" + context +
                             "'");
  }
  finalize_tensor_contract(tensor);
  const auto dense_bytes = dense_shape_size_bytes_local(tensor->mpk_shape, tensor->dtype);
  if (!dense_bytes.has_value()) {
    throw std::runtime_error(
        "dense tensor contract size update requires canonical shape/dtype for '" + context + "'");
  }
  if (expected_size_bytes > 0U && *dense_bytes != expected_size_bytes) {
    throw std::runtime_error("dense tensor contract bytes mismatch for '" + context +
                             "': shape-derived=" + std::to_string(*dense_bytes) +
                             " source=" + std::to_string(expected_size_bytes));
  }
  tensor->size_bytes = *dense_bytes;
  if (tensor->stride_bytes.empty()) {
    tensor->stride_bytes = contiguous_stride_bytes_local(tensor->mpk_shape, tensor->dtype);
  }
}

void validate_mla_boundary_tensor_contract_local(
    const MpkTensorContract& tensor, const std::string& context, const bool transport_view = false,
    const MpkPluginIoContract* boundary_stage = nullptr) {
  if (tensor.mpk_shape.empty()) {
    throw std::runtime_error("MLA boundary tensor is missing shape for '" + context + "'");
  }
  if (tensor.size_bytes == 0U) {
    throw std::runtime_error("MLA boundary tensor is missing size_bytes for '" + context + "'");
  }
  if (tensor.byte_offset < 0) {
    throw std::runtime_error("MLA boundary tensor has negative byte_offset for '" + context + "'");
  }
  if (transport_view) {
    if (!boundary_stage) {
      throw std::runtime_error("MLA boundary transport tensor is missing stage metadata for '" +
                               context + "'");
    }
    const std::uint64_t transport_bytes =
        expected_detess_packed_input_size_bytes_local(*boundary_stage, tensor.dtype);
    if (transport_bytes == 0U || transport_bytes != tensor.size_bytes) {
      throw std::runtime_error("MLA boundary transport tensor byte span mismatch for '" + context +
                               "': expected=" + std::to_string(transport_bytes) +
                               " tensor=" + std::to_string(tensor.size_bytes));
    }
    if (!tensor.stride_bytes.empty() && tensor.stride_bytes.size() != tensor.mpk_shape.size()) {
      throw std::runtime_error("MLA boundary transport tensor stride rank mismatch for '" +
                               context + "': shape=" + ints_dbg_local(tensor.mpk_shape) +
                               " stride=" + ints_dbg_local(tensor.stride_bytes));
    }
    return;
  }
  const auto dense_bytes = dense_shape_size_bytes_local(tensor.mpk_shape, tensor.dtype);
  const bool detess_logical_boundary_preserves_packed_span =
      !transport_view && boundary_stage != nullptr && dense_bytes.has_value() &&
      *dense_bytes <= tensor.size_bytes &&
      expected_detess_packed_input_size_bytes_local(*boundary_stage, tensor.dtype) ==
          tensor.size_bytes;
  if ((!dense_bytes.has_value() || *dense_bytes != tensor.size_bytes) &&
      !detess_logical_boundary_preserves_packed_span) {
    throw std::runtime_error(
        "MLA boundary tensor shape/dtype bytes mismatch for '" + context +
        "': shape-derived=" + std::to_string(dense_bytes.has_value() ? *dense_bytes : 0U) +
        " tensor=" + std::to_string(tensor.size_bytes));
  }
  if (!tensor.stride_bytes.empty() && tensor.stride_bytes.size() != tensor.mpk_shape.size()) {
    throw std::runtime_error("MLA boundary tensor stride rank mismatch for '" + context +
                             "': shape=" + ints_dbg_local(tensor.mpk_shape) +
                             " stride=" + ints_dbg_local(tensor.stride_bytes));
  }
}

std::int64_t
projected_slice_begin_offset_bytes_local(const std::vector<std::int64_t>& begin,
                                         const std::vector<std::int64_t>& stride_bytes) {
  if (begin.empty() || stride_bytes.empty()) {
    return 0;
  }
  const std::size_t count = std::min(begin.size(), stride_bytes.size());
  std::int64_t total = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (begin[i] <= 0 || stride_bytes[i] <= 0) {
      continue;
    }
    if (begin[i] > std::numeric_limits<std::int64_t>::max() / stride_bytes[i]) {
      return std::numeric_limits<std::int64_t>::max();
    }
    const std::int64_t delta = begin[i] * stride_bytes[i];
    if (delta > std::numeric_limits<std::int64_t>::max() - total) {
      return std::numeric_limits<std::int64_t>::max();
    }
    total += delta;
  }
  return total;
}

std::vector<MpkTensorContract>
parse_tensor_nodes(const json& nodes, const std::vector<std::vector<std::int64_t>>& shapes,
                   const std::vector<std::string>& dtypes, const std::string& fallback_dtype,
                   const MpkShapeSemantics shape_semantics) {
  std::vector<MpkTensorContract> out;
  if (!nodes.is_array()) {
    return out;
  }
  out.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i].is_object()) {
      continue;
    }
    MpkTensorContract tensor;
    tensor.tensor_index = static_cast<int>(i);
    if (nodes[i].contains("name") && nodes[i]["name"].is_string()) {
      tensor.name = nodes[i]["name"].get<std::string>();
    }
    if (nodes[i].contains("type") && nodes[i]["type"].is_string()) {
      tensor.kind = nodes[i]["type"].get<std::string>();
    }
    if (nodes[i].contains("size")) {
      if (const auto size = read_size_local(nodes[i]["size"]); size.has_value()) {
        tensor.size_bytes = *size;
      }
    }
    if (nodes[i].contains("input_range")) {
      tensor.input_range = read_numeric_values_any<double>(nodes[i]["input_range"]);
    }
    tensor.shape_semantics = shape_semantics;
    if (!shapes.empty()) {
      tensor.mpk_shape = (i < shapes.size()) ? shapes[i] : shapes.front();
    }
    if (!dtypes.empty()) {
      tensor.dtype = normalize_dtype_local((i < dtypes.size()) ? dtypes[i] : dtypes.front());
    } else {
      tensor.dtype = normalize_dtype_local(fallback_dtype);
    }
    finalize_tensor_contract(&tensor);
    out.push_back(std::move(tensor));
  }
  return out;
}

std::optional<MpkQuantContract> parse_quant_from_params(const json& params) {
  if (!params.is_object()) {
    return std::nullopt;
  }
  std::vector<double> scales;
  for (const char* key : {"q_scale", "quant_scale", "dq_scale"}) {
    if (params.contains(key)) {
      scales = read_numeric_values_any<double>(params.at(key));
      if (!scales.empty()) {
        break;
      }
    }
  }
  std::vector<std::int64_t> zero_points;
  for (const char* key : {"q_zp", "zero_point", "dq_zp"}) {
    if (params.contains(key)) {
      zero_points = read_numeric_values_any<std::int64_t>(params.at(key));
      if (!zero_points.empty()) {
        break;
      }
    }
  }
  if ((scales.empty() || zero_points.empty()) && params.contains("channel_params") &&
      params.at("channel_params").is_array()) {
    const auto& channel_params = params.at("channel_params");
    std::vector<double> channel_scales;
    std::vector<std::int64_t> channel_zero_points;
    channel_scales.reserve(channel_params.size());
    channel_zero_points.reserve(channel_params.size());
    for (const auto& channel : channel_params) {
      if (channel.is_array() && channel.size() >= 2U) {
        const auto read_scale = read_numeric_values_any<double>(channel[0]);
        const auto read_zp = read_numeric_values_any<std::int64_t>(channel[1]);
        if (!read_scale.empty() && !read_zp.empty()) {
          channel_scales.push_back(read_scale.front());
          channel_zero_points.push_back(read_zp.front());
          continue;
        }
      }
      if (channel.is_object()) {
        std::vector<double> channel_scale;
        std::vector<std::int64_t> channel_zp;
        for (const char* key : {"q_scale", "quant_scale", "dq_scale", "scale"}) {
          if (channel.contains(key)) {
            channel_scale = read_numeric_values_any<double>(channel.at(key));
            if (!channel_scale.empty()) {
              break;
            }
          }
        }
        for (const char* key : {"q_zp", "zero_point", "dq_zp", "quant_zero_point"}) {
          if (channel.contains(key)) {
            channel_zp = read_numeric_values_any<std::int64_t>(channel.at(key));
            if (!channel_zp.empty()) {
              break;
            }
          }
        }
        if (!channel_scale.empty() && !channel_zp.empty()) {
          channel_scales.push_back(channel_scale.front());
          channel_zero_points.push_back(channel_zp.front());
        }
      }
    }
    if (scales.empty() && !channel_scales.empty()) {
      scales = std::move(channel_scales);
    }
    if (zero_points.empty() && !channel_zero_points.empty()) {
      zero_points = std::move(channel_zero_points);
    }
  }
  if (scales.empty() || zero_points.empty()) {
    return std::nullopt;
  }
  MpkQuantContract quant;
  quant.scales = std::move(scales);
  quant.zero_points = std::move(zero_points);
  if (params.contains("q_axis")) {
    if (const auto axis = read_int_local(params.at("q_axis")); axis.has_value()) {
      quant.axis = *axis;
    }
  } else if (params.contains("axis")) {
    if (const auto axis = read_int_local(params.at("axis")); axis.has_value()) {
      quant.axis = *axis;
    }
  }
  return quant;
}

void derive_mla_output_quant_from_downstream(MpkContract* contract) {
  if (!contract || contract->plugins.empty()) {
    return;
  }

  std::optional<std::size_t> mla_index;
  for (std::size_t i = 0; i < contract->plugins.size(); ++i) {
    const auto& plugin = contract->plugins[i];
    const bool by_processor = lower_copy_local(plugin.processor) == "mla";
    const bool by_kernel = canonical_token_local(plugin.kernel) == "mla";
    if (by_processor || by_kernel) {
      if (mla_index.has_value()) {
        return;
      }
      mla_index = i;
    }
  }
  if (!mla_index.has_value()) {
    return;
  }

  auto& mla = contract->plugins[*mla_index];
  if (mla.output_tensors.empty()) {
    return;
  }
  if (mla.quant.has_value() && !mla.quant->scales.empty() && !mla.quant->zero_points.empty()) {
    return;
  }

  const std::size_t output_count = mla.output_tensors.size();
  if (output_count == 1U) {
    for (const auto& edge : contract->edges) {
      if (edge.src_plugin_index != *mla_index || edge.src_output_index != 0 ||
          edge.dst_plugin_index >= contract->plugins.size()) {
        continue;
      }
      const auto& downstream = contract->plugins[edge.dst_plugin_index];
      const std::string downstream_kernel = canonical_token_local(downstream.kernel);
      const bool dequant_like = downstream_kernel.find("dequant") != std::string::npos ||
                                downstream_kernel.find("detessdequant") != std::string::npos;
      if (!dequant_like || !downstream.quant.has_value() || downstream.quant->scales.empty() ||
          downstream.quant->zero_points.empty()) {
        continue;
      }
      mla.quant = downstream.quant;
      if (mpk_contract_debug_enabled()) {
        std::fprintf(
            stderr,
            "[mpk-contract] derived mla_output_quant from downstream=%s scales=%zu zps=%zu\n",
            mpk_plugin_name_dbg(downstream), mla.quant->scales.size(),
            mla.quant->zero_points.size());
      }
      return;
    }
    return;
  }

  std::vector<double> per_output_scales(output_count, 0.0);
  std::vector<std::int64_t> per_output_zero_points(output_count, 0);
  std::vector<bool> per_output_found(output_count, false);

  for (const auto& edge : contract->edges) {
    if (edge.src_plugin_index != *mla_index || edge.src_output_index < 0 ||
        edge.dst_plugin_index >= contract->plugins.size()) {
      continue;
    }
    const auto out_idx = static_cast<std::size_t>(edge.src_output_index);
    if (out_idx >= output_count || per_output_found[out_idx]) {
      continue;
    }
    const auto& downstream = contract->plugins[edge.dst_plugin_index];
    const std::string downstream_kernel = canonical_token_local(downstream.kernel);
    const bool dequant_like = downstream_kernel.find("dequant") != std::string::npos ||
                              downstream_kernel.find("detessdequant") != std::string::npos;
    if (!dequant_like || !downstream.quant.has_value() || downstream.quant->scales.empty() ||
        downstream.quant->zero_points.empty()) {
      continue;
    }
    per_output_scales[out_idx] = downstream.quant->scales.front();
    per_output_zero_points[out_idx] = downstream.quant->zero_points.front();
    per_output_found[out_idx] = true;
  }

  if (!std::all_of(per_output_found.begin(), per_output_found.end(),
                   [](bool found) { return found; })) {
    return;
  }

  MpkQuantContract quant;
  quant.scales = std::move(per_output_scales);
  quant.zero_points = std::move(per_output_zero_points);
  quant.axis = -1;
  mla.quant = std::move(quant);

  if (mpk_contract_debug_enabled()) {
    std::fprintf(stderr, "[mpk-contract] derived mla_output_quant per-output scales=%zu zps=%zu\n",
                 mla.quant->scales.size(), mla.quant->zero_points.size());
  }
}

std::optional<fs::path> find_mpk_contract_path(const fs::path& package_root) {
  if (package_root.empty()) {
    return std::nullopt;
  }
  std::vector<fs::path> candidates;
  std::error_code ec;
  fs::recursive_directory_iterator it(package_root, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    if (it.depth() > 2) {
      it.disable_recursion_pending();
      continue;
    }
    if (!it->is_regular_file()) {
      continue;
    }
    const std::string filename = it->path().filename().string();
    if (ends_with_local(filename, "_mpk.json")) {
      candidates.push_back(it->path());
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const fs::path& a, const fs::path& b) { return a.string() < b.string(); });
  return candidates.front();
}

std::size_t plugin_order_key(const MpkPluginIoContract& stage, std::size_t idx) {
  if (stage.sequence >= 0) {
    return static_cast<std::size_t>(stage.sequence);
  }
  return static_cast<std::size_t>(1000000U + idx);
}

std::vector<std::size_t> plugins_in_order_internal(const MpkContract& contract) {
  std::vector<std::size_t> order;
  order.reserve(contract.plugins.size());
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const std::size_t ka = plugin_order_key(contract.plugins[a], a);
    const std::size_t kb = plugin_order_key(contract.plugins[b], b);
    if (ka != kb) {
      return ka < kb;
    }
    return a < b;
  });
  return order;
}

bool is_session_ingress_input(const MpkTensorContract& input, const MpkPluginIoContract& consumer,
                              std::size_t consumer_rank, const MpkContract& contract) {
  if (consumer_rank == 0U) {
    return true;
  }
  const std::string input_kind = canonical_token_local(input.kind);
  const std::string input_name = canonical_token_local(input.name);
  if (input_kind.find("input") != std::string::npos ||
      input_kind.find("ingress") != std::string::npos) {
    return true;
  }
  if (input_name == "images" || input_name == "image" || input_name == "input" ||
      input_name == "inputimage" || input_name == "ifm0" || input_name == "ifm") {
    return true;
  }
  if (input_name.rfind("ifm", 0U) == 0U) {
    return true;
  }
  if (!input_name.empty()) {
    for (const auto& ingress : contract.ingress_tensors) {
      if (canonical_token_local(ingress.name) == input_name) {
        return true;
      }
    }
  }
  const std::string proc = lower_copy_local(consumer.processor);
  return proc == "a65" && input_name.empty();
}

bool resolve_contract_edges_strict(MpkContract* contract, std::string* error_message) {
  if (!contract) {
    if (error_message) {
      *error_message = "null contract";
    }
    return false;
  }
  contract->edges.clear();
  contract->errors.clear();
  if (contract->plugins.empty()) {
    return true;
  }

  struct ProducerCandidate {
    std::size_t plugin_index = 0U;
    int output_index = -1;
    std::size_t order = 0U;
  };

  const auto order = plugins_in_order_internal(*contract);
  std::unordered_map<std::size_t, std::size_t> rank_by_plugin;
  rank_by_plugin.reserve(order.size());
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    rank_by_plugin[order[rank]] = rank;
  }
  if (mpk_contract_debug_enabled()) {
    std::fprintf(stderr, "[mpk-contract] strict_resolve plugins=%zu ordered=%zu\n",
                 contract->plugins.size(), order.size());
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
      const std::size_t idx = order[rank];
      if (idx >= contract->plugins.size()) {
        continue;
      }
      const auto& plugin = contract->plugins[idx];
      std::fprintf(
          stderr,
          "[mpk-contract]   order rank=%zu idx=%zu plugin=%s kernel=%s inputs=%zu outputs=%zu\n",
          rank, idx, mpk_plugin_name_dbg(plugin), plugin.kernel.c_str(),
          plugin.input_tensors.size(), plugin.output_tensors.size());
    }
  }

  std::unordered_map<std::string, std::vector<ProducerCandidate>> producers_by_tensor_name;
  for (std::size_t pi = 0; pi < contract->plugins.size(); ++pi) {
    const auto& plugin = contract->plugins[pi];
    const std::size_t plugin_order = plugin_order_key(plugin, pi);
    for (std::size_t oi = 0; oi < plugin.output_tensors.size(); ++oi) {
      const auto& out = plugin.output_tensors[oi];
      if (out.name.empty()) {
        continue;
      }
      producers_by_tensor_name[out.name].push_back(
          ProducerCandidate{pi, static_cast<int>(oi), plugin_order});
    }
  }
  for (auto& kv : producers_by_tensor_name) {
    auto& vec = kv.second;
    std::stable_sort(vec.begin(), vec.end(),
                     [](const ProducerCandidate& a, const ProducerCandidate& b) {
                       if (a.order != b.order) {
                         return a.order < b.order;
                       }
                       if (a.plugin_index != b.plugin_index) {
                         return a.plugin_index < b.plugin_index;
                       }
                       return a.output_index < b.output_index;
                     });
  }

  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const std::size_t consumer_idx = order[rank];
    auto& consumer = contract->plugins[consumer_idx];
    const std::size_t consumer_order = plugin_order_key(consumer, consumer_idx);
    for (std::size_t ii = 0; ii < consumer.input_tensors.size(); ++ii) {
      auto& input = consumer.input_tensors[ii];
      if (input.name.empty()) {
        continue;
      }
      const auto prod_it = producers_by_tensor_name.find(input.name);
      if (prod_it == producers_by_tensor_name.end()) {
        if (is_session_ingress_input(input, consumer, rank, *contract)) {
          if (mpk_contract_debug_enabled()) {
            std::fprintf(
                stderr,
                "[mpk-contract] ingress_input consumer=%s rank=%zu tensor=%s input_index=%zu\n",
                mpk_plugin_name_dbg(consumer), rank, input.name.c_str(), ii);
          }
          continue;
        }
        std::ostringstream oss;
        oss << "missing_producer_for_tensor: tensor='" << input.name << "' consumer='"
            << consumer.name << "' input_index=" << ii;
        const std::string msg = oss.str();
        if (mpk_contract_debug_enabled()) {
          std::fprintf(stderr, "[mpk-contract] strict_resolve_error %s\n", msg.c_str());
        }
        contract->errors.push_back({MpkContractErrorCode::MissingProducerForTensor, msg});
        if (error_message) {
          *error_message = msg;
        }
        return false;
      }

      std::vector<ProducerCandidate> candidates;
      candidates.reserve(prod_it->second.size());
      for (const auto& cand : prod_it->second) {
        if (cand.order < consumer_order) {
          candidates.push_back(cand);
        }
      }
      if (candidates.empty()) {
        if (is_session_ingress_input(input, consumer, rank, *contract)) {
          if (mpk_contract_debug_enabled()) {
            std::fprintf(stderr,
                         "[mpk-contract] ingress_input_no_prior_producer consumer=%s rank=%zu "
                         "tensor=%s input_index=%zu\n",
                         mpk_plugin_name_dbg(consumer), rank, input.name.c_str(), ii);
          }
          continue;
        }
        std::ostringstream oss;
        oss << "missing_producer_for_tensor: tensor='" << input.name << "' consumer='"
            << consumer.name << "' input_index=" << ii << " (no producer before consumer order)";
        const std::string msg = oss.str();
        if (mpk_contract_debug_enabled()) {
          std::fprintf(stderr, "[mpk-contract] strict_resolve_error %s\n", msg.c_str());
        }
        contract->errors.push_back({MpkContractErrorCode::MissingProducerForTensor, msg});
        if (error_message) {
          *error_message = msg;
        }
        return false;
      }

      const auto best_it =
          std::max_element(candidates.begin(), candidates.end(),
                           [](const ProducerCandidate& a, const ProducerCandidate& b) {
                             if (a.order != b.order) {
                               return a.order < b.order;
                             }
                             if (a.plugin_index != b.plugin_index) {
                               return a.plugin_index < b.plugin_index;
                             }
                             return a.output_index < b.output_index;
                           });
      const std::size_t selected_order = best_it->order;
      std::vector<ProducerCandidate> same_order;
      for (const auto& cand : candidates) {
        if (cand.order == selected_order) {
          same_order.push_back(cand);
        }
      }
      if (same_order.size() > 1U) {
        std::ostringstream oss;
        oss << "ambiguous_producer_for_tensor: tensor='" << input.name << "' consumer='"
            << consumer.name << "' input_index=" << ii << " candidates=" << same_order.size();
        const std::string msg = oss.str();
        if (mpk_contract_debug_enabled()) {
          std::fprintf(stderr, "[mpk-contract] strict_resolve_error %s\n", msg.c_str());
          for (const auto& cand : same_order) {
            if (cand.plugin_index >= contract->plugins.size()) {
              continue;
            }
            const auto& cand_plugin = contract->plugins[cand.plugin_index];
            std::fprintf(stderr,
                         "[mpk-contract]   ambiguous_candidate plugin_idx=%zu plugin=%s "
                         "output_index=%d order=%zu\n",
                         cand.plugin_index, mpk_plugin_name_dbg(cand_plugin), cand.output_index,
                         cand.order);
          }
        }
        contract->errors.push_back({MpkContractErrorCode::AmbiguousProducerForTensor, msg});
        if (error_message) {
          *error_message = msg;
        }
        return false;
      }

      const ProducerCandidate selected = same_order.front();
      if (selected.plugin_index >= contract->plugins.size()) {
        continue;
      }
      auto& producer = contract->plugins[selected.plugin_index];
      if (selected.output_index < 0 ||
          static_cast<std::size_t>(selected.output_index) >= producer.output_tensors.size()) {
        continue;
      }
      auto& source = producer.output_tensors[static_cast<std::size_t>(selected.output_index)];
      if (mpk_contract_debug_enabled()) {
        std::fprintf(stderr,
                     "[mpk-contract] edge_select tensor=%s consumer=%s input=%zu producer=%s "
                     "output=%d producer_order=%zu consumer_order=%zu candidates=%zu\n",
                     input.name.c_str(), mpk_plugin_name_dbg(consumer), ii,
                     mpk_plugin_name_dbg(producer), selected.output_index, selected.order,
                     consumer_order, candidates.size());
      }

      if (input.mpk_shape.empty() && !source.mpk_shape.empty()) {
        input.mpk_shape = source.mpk_shape;
      }
      if (input.dtype.empty() && !source.dtype.empty()) {
        input.dtype = source.dtype;
      }
      if (input.size_bytes == 0U && source.size_bytes > 0U) {
        input.size_bytes = source.size_bytes;
      }
      finalize_tensor_contract(&input);
      if (source.mpk_shape.empty() && !input.mpk_shape.empty()) {
        source.mpk_shape = input.mpk_shape;
      }
      if (source.dtype.empty() && !input.dtype.empty()) {
        source.dtype = input.dtype;
      }
      if (source.size_bytes == 0U && input.size_bytes > 0U) {
        source.size_bytes = input.size_bytes;
      }
      finalize_tensor_contract(&source);

      MpkContractEdge edge;
      edge.src_plugin_index = selected.plugin_index;
      edge.src_output_index = selected.output_index;
      edge.dst_plugin_index = consumer_idx;
      edge.dst_input_index = static_cast<int>(ii);
      edge.src_plugin = producer.name;
      edge.dst_plugin = consumer.name;
      edge.tensor_name = input.name;
      contract->edges.push_back(std::move(edge));
    }
  }

  std::stable_sort(contract->edges.begin(), contract->edges.end(),
                   [&](const MpkContractEdge& a, const MpkContractEdge& b) {
                     const std::size_t adst_order = plugin_order_key(
                         contract->plugins[a.dst_plugin_index], a.dst_plugin_index);
                     const std::size_t bdst_order = plugin_order_key(
                         contract->plugins[b.dst_plugin_index], b.dst_plugin_index);
                     if (adst_order != bdst_order) {
                       return adst_order < bdst_order;
                     }
                     if (a.dst_plugin_index != b.dst_plugin_index) {
                       return a.dst_plugin_index < b.dst_plugin_index;
                     }
                     if (a.dst_input_index != b.dst_input_index) {
                       return a.dst_input_index < b.dst_input_index;
                     }
                     if (a.src_plugin_index != b.src_plugin_index) {
                       return a.src_plugin_index < b.src_plugin_index;
                     }
                     return a.src_output_index < b.src_output_index;
                   });
  if (mpk_contract_debug_enabled()) {
    std::fprintf(stderr, "[mpk-contract] strict_resolve_done edges=%zu\n", contract->edges.size());
    for (const auto& edge : contract->edges) {
      std::fprintf(stderr,
                   "[mpk-contract]   edge src_idx=%zu src_out=%d dst_idx=%zu dst_in=%d tensor=%s "
                   "src=%s dst=%s\n",
                   edge.src_plugin_index, edge.src_output_index, edge.dst_plugin_index,
                   edge.dst_input_index, edge.tensor_name.c_str(), edge.src_plugin.c_str(),
                   edge.dst_plugin.c_str());
    }
  }
  return true;
}

void derive_logical_output_contracts(MpkContract* contract) {
  if (!contract) {
    return;
  }

  auto normalize_shape = [](std::vector<std::int64_t> shape) {
    if (shape.size() >= 4U && shape.front() == 1) {
      shape.erase(shape.begin());
    }
    return shape;
  };
  auto geometry_shape_from_tensor = [&](const MpkTensorContract& tensor) {
    if (!tensor.logical_shape.empty()) {
      return normalize_shape(tensor.logical_shape);
    }
    if (is_geometry_shape_semantics_local(tensor.shape_semantics) && !tensor.mpk_shape.empty()) {
      return normalize_shape(tensor.mpk_shape);
    }
    return std::vector<std::int64_t>{};
  };

  auto pick_consumer_output = [](const MpkPluginIoContract& consumer,
                                 std::size_t input_index) -> const MpkTensorContract* {
    if (consumer.output_tensors.empty()) {
      return nullptr;
    }
    if (consumer.output_tensors.size() == 1U) {
      return &consumer.output_tensors.front();
    }
    if (input_index < consumer.output_tensors.size()) {
      return &consumer.output_tensors[input_index];
    }
    return &consumer.output_tensors.front();
  };
  auto pick_consumer_output_index = [](const MpkPluginIoContract& consumer,
                                       std::size_t input_index) -> std::size_t {
    if (consumer.output_tensors.empty()) {
      return 0U;
    }
    if (consumer.output_tensors.size() == 1U) {
      return 0U;
    }
    if (input_index < consumer.output_tensors.size()) {
      return input_index;
    }
    return 0U;
  };

  std::unordered_map<std::uint64_t, std::vector<const MpkContractEdge*>> outgoing_edges;
  outgoing_edges.reserve(contract->edges.size());
  auto output_key = [](std::size_t plugin_index, int output_index) -> std::uint64_t {
    return (static_cast<std::uint64_t>(plugin_index) << 32U) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_index));
  };
  for (const auto& edge : contract->edges) {
    outgoing_edges[output_key(edge.src_plugin_index, edge.src_output_index)].push_back(&edge);
  }

  for (std::size_t producer_idx = 0; producer_idx < contract->plugins.size(); ++producer_idx) {
    auto& producer = contract->plugins[producer_idx];
    const std::size_t producer_order = plugin_order_key(producer, producer_idx);
    const bool producer_is_tessellate = is_tessellate_producer_kernel(producer.kernel);
    const bool producer_is_mla = (lower_copy_local(producer.processor) == "mla") ||
                                 (canonical_token_local(producer.kernel) == "mla");

    for (std::size_t oi = 0; oi < producer.output_tensors.size(); ++oi) {
      auto& output = producer.output_tensors[oi];
      output.logical_shape.clear();
      output.logical_dtype.clear();
      output.logical_source_plugin.clear();
      output.logical_source_kernel.clear();
      output.logical_source_sequence = -1;

      if (producer_is_tessellate) {
        const MpkTensorContract* producer_input = nullptr;
        if (!producer.input_tensors.empty()) {
          if (output.tensor_index >= 0 &&
              static_cast<std::size_t>(output.tensor_index) < producer.input_tensors.size()) {
            producer_input = &producer.input_tensors[static_cast<std::size_t>(output.tensor_index)];
          } else {
            producer_input = &producer.input_tensors.front();
          }
        }
        if (producer_input) {
          output.logical_shape = geometry_shape_from_tensor(*producer_input);
        }
        if (!output.logical_shape.empty()) {
          const std::string dtype =
              !producer.frame_type.empty()
                  ? producer.frame_type
                  : ((producer_input && !producer_input->dtype.empty()) ? producer_input->dtype
                                                                        : output.dtype);
          output.logical_dtype = normalize_dtype_local(dtype);
          output.logical_source_plugin = producer.name;
          output.logical_source_kernel = producer.kernel;
          output.logical_source_sequence = producer.sequence;
          continue;
        }
      }

      const std::vector<std::int64_t> physical_shape = geometry_shape_from_tensor(output);
      const bool physical_shape_structured = !physical_shape.empty();
      const auto edge_it = outgoing_edges.find(output_key(producer_idx, static_cast<int>(oi)));
      if (edge_it == outgoing_edges.end() || edge_it->second.empty()) {
        if (physical_shape_structured) {
          output.logical_shape = physical_shape;
          output.logical_dtype = normalize_dtype_local(output.dtype);
        }
        continue;
      }

      const MpkContractEdge* best_edge = nullptr;
      std::size_t best_order = std::numeric_limits<std::size_t>::max();
      for (const auto* edge : edge_it->second) {
        if (!edge || edge->dst_plugin_index >= contract->plugins.size()) {
          continue;
        }
        const std::size_t order =
            plugin_order_key(contract->plugins[edge->dst_plugin_index], edge->dst_plugin_index);
        if (order <= producer_order) {
          continue;
        }
        if (order < best_order) {
          best_order = order;
          best_edge = edge;
        }
      }
      if (!best_edge) {
        if (physical_shape_structured) {
          output.logical_shape = physical_shape;
          output.logical_dtype = normalize_dtype_local(output.dtype);
        }
        continue;
      }

      const auto& consumer = contract->plugins[best_edge->dst_plugin_index];
      const MpkTensorContract* best_input_tensor = nullptr;
      if (best_edge->dst_input_index >= 0 &&
          static_cast<std::size_t>(best_edge->dst_input_index) < consumer.input_tensors.size()) {
        best_input_tensor =
            &consumer.input_tensors[static_cast<std::size_t>(best_edge->dst_input_index)];
      }
      if (!best_input_tensor && !consumer.input_tensors.empty()) {
        best_input_tensor = &consumer.input_tensors.front();
      }
      const bool transform_consumer = is_logical_consumer_transform_kernel(consumer.kernel);
      const bool dtype_preserving_consumer = is_dtype_preserving_transform_kernel(consumer.kernel);
      const MpkTensorContract* logical_source = nullptr;
      if (transform_consumer) {
        const std::size_t input_index = (best_edge->dst_input_index >= 0)
                                            ? static_cast<std::size_t>(best_edge->dst_input_index)
                                            : 0U;
        const std::string consumer_kernel_token = canonical_token_local(consumer.kernel);
        logical_source = pick_consumer_output(consumer, input_index);
        if (consumer_kernel_token.find("unpack") != std::string::npos &&
            best_edge->dst_plugin_index < contract->plugins.size() &&
            !consumer.output_tensors.empty()) {
          const std::size_t consumer_output_index =
              pick_consumer_output_index(consumer, input_index);
          const auto unpack_edge_it = outgoing_edges.find(
              output_key(best_edge->dst_plugin_index, static_cast<int>(consumer_output_index)));
          if (unpack_edge_it != outgoing_edges.end() && !unpack_edge_it->second.empty()) {
            const std::size_t consumer_order = plugin_order_key(
                contract->plugins[best_edge->dst_plugin_index], best_edge->dst_plugin_index);
            const MpkContractEdge* best_unpack_edge = nullptr;
            std::size_t best_unpack_order = std::numeric_limits<std::size_t>::max();
            for (const auto* unpack_edge : unpack_edge_it->second) {
              if (!unpack_edge || unpack_edge->dst_plugin_index >= contract->plugins.size()) {
                continue;
              }
              const std::size_t unpack_order = plugin_order_key(
                  contract->plugins[unpack_edge->dst_plugin_index], unpack_edge->dst_plugin_index);
              if (unpack_order <= consumer_order || unpack_order >= best_unpack_order) {
                continue;
              }
              best_unpack_order = unpack_order;
              best_unpack_edge = unpack_edge;
            }
            if (best_unpack_edge) {
              const auto& unpack_consumer = contract->plugins[best_unpack_edge->dst_plugin_index];
              const std::size_t unpack_input_index =
                  (best_unpack_edge->dst_input_index >= 0)
                      ? static_cast<std::size_t>(best_unpack_edge->dst_input_index)
                      : 0U;
              if (const MpkTensorContract* unpack_logical_source =
                      pick_consumer_output(unpack_consumer, unpack_input_index);
                  unpack_logical_source &&
                  !geometry_shape_from_tensor(*unpack_logical_source).empty()) {
                logical_source = unpack_logical_source;
              }
            }
          }
        }
      }
      if (!logical_source) {
        logical_source = best_input_tensor;
      }

      if (logical_source) {
        output.logical_shape = geometry_shape_from_tensor(*logical_source);
      }
      if (!producer_is_mla && dtype_preserving_consumer && logical_source &&
          !logical_source->dtype.empty()) {
        output.logical_dtype = normalize_dtype_local(logical_source->dtype);
      }

      if (output.logical_shape.empty() && physical_shape_structured) {
        output.logical_shape = physical_shape;
      }
      if (producer_is_mla && !output.dtype.empty()) {
        // MLA boundary contracts are physical; cast/dequant are separate nodes.
        output.logical_dtype = normalize_dtype_local(output.dtype);
      }
      if (output.logical_dtype.empty() && !output.dtype.empty() && !output.logical_shape.empty()) {
        output.logical_dtype = normalize_dtype_local(output.dtype);
      }

      output.logical_source_plugin = consumer.name;
      output.logical_source_kernel = consumer.kernel;
      output.logical_source_sequence = consumer.sequence;
    }
  }
}

void derive_logical_input_contracts(MpkContract* contract) {
  if (!contract) {
    return;
  }

  auto normalize_shape = [](std::vector<std::int64_t> shape) {
    if (shape.size() >= 4U && shape.front() == 1) {
      shape.erase(shape.begin());
    }
    return shape;
  };
  auto geometry_shape_from_tensor = [&](const MpkTensorContract& tensor) {
    if (!tensor.logical_shape.empty()) {
      return normalize_shape(tensor.logical_shape);
    }
    if (is_geometry_shape_semantics_local(tensor.shape_semantics) && !tensor.mpk_shape.empty()) {
      return normalize_shape(tensor.mpk_shape);
    }
    return std::vector<std::int64_t>{};
  };

  std::unordered_map<std::uint64_t, const MpkContractEdge*> incoming_edge_by_input;
  incoming_edge_by_input.reserve(contract->edges.size());
  auto input_key = [](std::size_t plugin_index, int input_index) -> std::uint64_t {
    return (static_cast<std::uint64_t>(plugin_index) << 32U) |
           static_cast<std::uint64_t>(static_cast<std::uint32_t>(input_index));
  };
  for (const auto& edge : contract->edges) {
    incoming_edge_by_input[input_key(edge.dst_plugin_index, edge.dst_input_index)] = &edge;
  }

  for (std::size_t pi = 0; pi < contract->plugins.size(); ++pi) {
    auto& plugin = contract->plugins[pi];
    for (std::size_t ii = 0; ii < plugin.input_tensors.size(); ++ii) {
      auto& input = plugin.input_tensors[ii];
      input.logical_shape.clear();
      input.logical_dtype.clear();
      input.logical_source_plugin.clear();
      input.logical_source_kernel.clear();
      input.logical_source_sequence = -1;

      const std::vector<std::int64_t> input_geometry_shape = geometry_shape_from_tensor(input);
      const bool input_shape_structured = !input_geometry_shape.empty();
      if (input.name.empty()) {
        if (input_shape_structured) {
          input.logical_shape = input_geometry_shape;
          input.logical_dtype = normalize_dtype_local(input.dtype);
        }
        continue;
      }

      const auto edge_it = incoming_edge_by_input.find(input_key(pi, static_cast<int>(ii)));
      if (edge_it == incoming_edge_by_input.end() || !edge_it->second) {
        if (input_shape_structured) {
          input.logical_shape = input_geometry_shape;
          input.logical_dtype = normalize_dtype_local(input.dtype);
        }
        continue;
      }

      const auto* edge = edge_it->second;
      if (edge->src_plugin_index >= contract->plugins.size()) {
        if (input_shape_structured) {
          input.logical_shape = input_geometry_shape;
          input.logical_dtype = normalize_dtype_local(input.dtype);
        }
        continue;
      }
      const auto& producer = contract->plugins[edge->src_plugin_index];
      const MpkTensorContract* source = nullptr;
      if (edge->src_output_index >= 0 &&
          static_cast<std::size_t>(edge->src_output_index) < producer.output_tensors.size()) {
        source = &producer.output_tensors[static_cast<std::size_t>(edge->src_output_index)];
      }
      if (!source && !producer.output_tensors.empty()) {
        source = &producer.output_tensors.front();
      }
      if (!source) {
        continue;
      }

      input.logical_shape = geometry_shape_from_tensor(*source);
      if (input.logical_shape.empty() && input_shape_structured) {
        input.logical_shape = input_geometry_shape;
      }

      if (!source->logical_dtype.empty()) {
        input.logical_dtype = normalize_dtype_local(source->logical_dtype);
      } else if (!source->dtype.empty()) {
        input.logical_dtype = normalize_dtype_local(source->dtype);
      } else if (!input.dtype.empty()) {
        input.logical_dtype = normalize_dtype_local(input.dtype);
      }

      input.logical_source_plugin = producer.name;
      input.logical_source_kernel = producer.kernel;
      input.logical_source_sequence = producer.sequence;
    }
  }
}

void dump_mpk_contract_compare_local(const MpkContract& contract, const json& root_json) {
  std::fprintf(stderr,
               "[mpk-compare] scope=final root mpk_json_path=%s model_name=\"%s\" "
               "model_path=\"%s\" plugins=%zu ingress=%zu edges=%zu errors=%zu\n",
               contract.mpk_json_path.c_str(), contract.model_name.c_str(),
               contract.model_path.c_str(), contract.plugins.size(),
               contract.ingress_tensors.size(), contract.edges.size(), contract.errors.size());
  std::fprintf(stderr, "[mpk-compare] scope=final root input_nodes_raw=%s\n",
               root_json.contains("input_nodes")
                   ? json_compact_dbg_local(root_json.at("input_nodes")).c_str()
                   : "<missing>");
  if (root_json.contains("input_nodes") && root_json.at("input_nodes").is_array()) {
    dump_tensor_list_compare_local("ingress", "<root>", &root_json.at("input_nodes"),
                                   contract.ingress_tensors);
  }

  const json* plugins_json = nullptr;
  if (root_json.contains("plugins") && root_json.at("plugins").is_array()) {
    plugins_json = &root_json.at("plugins");
  }
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    const auto& stage = contract.plugins[i];
    const json* raw_plugin =
        (plugins_json && i < plugins_json->size() && plugins_json->at(i).is_object())
            ? &plugins_json->at(i)
            : nullptr;
    const std::string raw_plugin_json =
        raw_plugin ? json_compact_dbg_local(*raw_plugin) : "<missing>";
    std::fprintf(stderr, "[mpk-compare] scope=final plugin_index=%zu stage=%s raw_plugin=%s\n", i,
                 mpk_plugin_name_dbg(stage), raw_plugin_json.c_str());
    std::fprintf(stderr, "[mpk-compare] scope=final plugin_index=%zu stage=%s parsed=%s\n", i,
                 mpk_plugin_name_dbg(stage), plugin_dbg_local(stage).c_str());
    if (raw_plugin && raw_plugin->contains("config_params")) {
      std::fprintf(stderr,
                   "[mpk-compare] scope=final plugin_index=%zu stage=%s raw_config_params=%s\n", i,
                   mpk_plugin_name_dbg(stage),
                   json_compact_dbg_local(raw_plugin->at("config_params")).c_str());
    }
    if (raw_plugin && raw_plugin->contains("input_nodes")) {
      dump_tensor_list_compare_local("input_nodes", mpk_plugin_name_dbg(stage),
                                     &raw_plugin->at("input_nodes"), stage.input_tensors);
    } else {
      dump_tensor_list_compare_local("input_nodes", mpk_plugin_name_dbg(stage), nullptr,
                                     stage.input_tensors);
    }
    if (raw_plugin && raw_plugin->contains("output_nodes")) {
      dump_tensor_list_compare_local("output_nodes", mpk_plugin_name_dbg(stage),
                                     &raw_plugin->at("output_nodes"), stage.output_tensors);
    } else {
      dump_tensor_list_compare_local("output_nodes", mpk_plugin_name_dbg(stage), nullptr,
                                     stage.output_tensors);
    }
  }

  dump_named_tensor_view_list_local("mla_boundary_physical_inputs",
                                    get_mla_boundary_physical_inputs_contract(contract));
  dump_named_tensor_view_list_local("mla_boundary_physical_outputs",
                                    get_mla_boundary_physical_outputs_contract(contract));
  dump_named_tensor_view_list_local("mla_published_outputs",
                                    get_mla_published_outputs_contract(contract));
  dump_named_tensor_view_list_local("mla_logical_outputs",
                                    get_mla_logical_outputs_contract(contract));

  for (std::size_t i = 0; i < contract.edges.size(); ++i) {
    const auto& edge = contract.edges[i];
    std::fprintf(stderr,
                 "[mpk-compare] scope=final edge index=%zu src_plugin_index=%zu "
                 "src_output_index=%d dst_plugin_index=%zu dst_input_index=%d src_plugin=\"%s\" "
                 "dst_plugin=\"%s\" tensor_name=\"%s\"\n",
                 i, edge.src_plugin_index, edge.src_output_index, edge.dst_plugin_index,
                 edge.dst_input_index, edge.src_plugin.c_str(), edge.dst_plugin.c_str(),
                 edge.tensor_name.c_str());
  }
}

const char* mpk_graph_node_kind_name_local(const MpkGraphNodeKind kind) {
  switch (kind) {
  case MpkGraphNodeKind::Unknown:
    return "unknown";
  case MpkGraphNodeKind::IngressTensor:
    return "ingress_tensor";
  case MpkGraphNodeKind::Plugin:
    return "plugin";
  case MpkGraphNodeKind::FusedPreproc:
    return "fused_preproc";
  case MpkGraphNodeKind::FusedBoxDecode:
    return "fused_boxdecode";
  case MpkGraphNodeKind::FusedQuantTess:
    return "fused_quanttess";
  case MpkGraphNodeKind::FusedDetessDequant:
    return "fused_detessdequant";
  }
  return "unknown";
}

const char* mpk_graph_edge_kind_name_local(const MpkGraphEdgeKind kind) {
  switch (kind) {
  case MpkGraphEdgeKind::Unknown:
    return "unknown";
  case MpkGraphEdgeKind::CandidateTensorMatch:
    return "candidate_tensor_match";
  case MpkGraphEdgeKind::FusedRoute:
    return "fused_route";
  }
  return "unknown";
}

std::string canonical_mpk_graph_op_local(const std::string& raw_kernel, const std::string& raw_name,
                                         const std::string& raw_processor) {
  (void)raw_processor;
  std::string token = canonical_token_local(raw_kernel);
  if (token.empty()) {
    token = canonical_token_local(raw_name);
  }
  if (token.empty()) {
    return "unknown";
  }
  if (token.find("detessdequant") != std::string::npos ||
      (token.find("detess") != std::string::npos && token.find("dequant") != std::string::npos)) {
    return "detessdequant";
  }
  if (token.find("boxdecode") != std::string::npos) {
    return "boxdecode";
  }
  if (token.find("preproc") != std::string::npos || token.find("preprocess") != std::string::npos) {
    return "preproc";
  }
  if (token.find("quanttess") != std::string::npos ||
      (token.find("quant") != std::string::npos && token.find("tess") != std::string::npos)) {
    return "quanttess";
  }
  if (token.find("dequant") != std::string::npos) {
    return "dequantize";
  }
  if (token.find("detess") != std::string::npos) {
    return "detess";
  }
  if (token.find("quant") != std::string::npos) {
    return "quant";
  }
  if (token.find("tess") != std::string::npos) {
    return "tess";
  }
  if (token.find("cast") != std::string::npos) {
    return "cast";
  }
  if (token.find("packtransform") != std::string::npos ||
      token.find("bufferconcat") != std::string::npos ||
      token.find("concatenat") != std::string::npos) {
    return "pack";
  }
  if (token.find("unpack") != std::string::npos) {
    return "unpack";
  }
  if (token.find("passthrough") != std::string::npos ||
      token.find("passthru") != std::string::npos ||
      token.find("pass_through") != std::string::npos) {
    return "pass_through";
  }
  if (token.find("mla") != std::string::npos || token == "infer") {
    return "mla";
  }
  if (token.find("slice") != std::string::npos) {
    return "slice";
  }
  return "unknown";
}

MpkGraphKernelField make_kernel_field_local(const MpkGraphKernelFieldKind kind,
                                            const std::string& name) {
  MpkGraphKernelField field;
  field.kind = kind;
  field.name = name;
  field.value = "unknown";
  field.known = false;
  return field;
}

void add_kernel_fields_local(MpkGraphKernelContract* contract, const MpkGraphKernelFieldKind kind,
                             std::initializer_list<const char*> names) {
  if (!contract) {
    return;
  }
  for (const char* name : names) {
    if (!name || !*name) {
      continue;
    }
    contract->fields.push_back(make_kernel_field_local(kind, name));
  }
}

void set_kernel_field_value_local(MpkGraphKernelContract* contract,
                                  const MpkGraphKernelFieldKind kind, const std::string& name,
                                  const std::string& value) {
  if (!contract || name.empty() || value.empty()) {
    return;
  }
  for (auto& field : contract->fields) {
    if (field.kind != kind || field.name != name) {
      continue;
    }
    field.value = value;
    field.known = true;
    return;
  }
}

std::optional<std::size_t> shape_element_count_local(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return std::nullopt;
  }
  std::size_t total = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return std::nullopt;
    }
    const auto u_dim = static_cast<std::size_t>(dim);
    if (u_dim > 0U && total > std::numeric_limits<std::size_t>::max() / u_dim) {
      return std::nullopt;
    }
    total *= u_dim;
  }
  return total;
}

MpkGraphKernelContract kernel_contract_template_local(const std::string& actual_kernel,
                                                      const std::string& canonical_op) {
  MpkGraphKernelContract contract;
  contract.kernel_name = !actual_kernel.empty() ? actual_kernel : canonical_op;

  const std::string op = canonical_op.empty() ? "unknown" : canonical_op;
  if (op == "quant") {
    contract.contract_type = "quantize_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"num_in_tensor", "input_shapes", "q_scale_array", "q_zp_array",
                             "q_scale", "q_zp", "batch_size"});
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Parameter,
        {"round_off_array", "out_dtype_array", "round_off", "out_dtype", "output_shape", "debug"});
  } else if (op == "tess") {
    contract.contract_type = "tessellate_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"num_in_tensor", "input_shapes", "slice_shape", "batch_size"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"input_dtype_array", "byte_align_array", "input_dtype", "byte_align",
                             "output_shape", "debug"});
  } else if (op == "quanttess") {
    contract.contract_type = "quanttess_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"input_shape", "q_scale", "q_zp", "slice_shape", "batch_size"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"out_dtype", "output_shape", "debug"});
  } else if (op == "cast") {
    contract.kernel_name = "bf16_to_fp32_bits/fp32_bits_to_bf16";
    contract.contract_type = "cast bit-conversion kernel";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"in", "out"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value, {"count"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"output_dtype", "output_shape"});
  } else if (op == "detess") {
    contract.contract_type = "detessellate_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"num_in_tensor", "batch_size", "input_shape", "slice_shape"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"input_dtype", "output_format", "output_shape", "debug"});
  } else if (op == "dequantize") {
    contract.contract_type = "dequantize_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"num_in_tensor", "input_shapes", "q_scale_array", "q_zp_array",
                             "q_scale", "q_zp", "batch_size"});
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Parameter,
        {"in_dtype_array", "out_dtype_array", "in_dtype", "out_dtype", "output_shape", "debug"});
  } else if (op == "detessdequant") {
    contract.contract_type = "detessdequant_config";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"input_tensor", "output_tensor", "metadata"});
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Value,
        {"num_in_tensor", "batch_size", "input_shape", "slice_shape", "dq_scale", "dq_zp"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"data_type", "fp16_out_en", "output_format", "output_shape", "debug"});
  } else if (op == "preproc") {
    contract.contract_type = "preproc_config";
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Argument,
        {"input_image", "output_rgb_image", "output_tessellated_image", "metadata"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"input_shape", "input_offset", "input_stride", "output_stride",
                             "scaled_width", "scaled_height", "slice_shape", "q_scale", "q_zp",
                             "channel_mean", "channel_stddev", "batch_size"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"aspect_ratio", "tessellate", "normalize", "input_img_type",
                             "output_img_type", "output_dtype", "output_shapes", "scaling_type",
                             "padding_type", "debug"});
  } else if (op == "boxdecode") {
    contract.kernel_name = "simaai_boxdecode_configure_from_runtime_v2/run";
    contract.contract_type = "configure:SimaBoxDecodeRuntimeConfigV2 + run(...)";
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Argument,
        {"instance_id", "in_data", "in_data_size", "out_data", "out_data_size"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value,
                            {"topk", "detection_threshold", "nms_iou_threshold", "num_classes",
                             "original_width", "original_height", "model_input_shape",
                             "num_in_tensor", "input_shapes", "slice_shapes", "dq_scale", "dq_zp"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"decode_type", "decode_type_option", "data_type"});
  } else if (op == "mla") {
    contract.kernel_name = "MLA_load_model + MLA_run_model/mla_run_model_phys";
    contract.contract_type = "dispatcher/client MLA call surface";
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument,
                            {"path", "mla_model", "iaddr", "oaddr", "in_addr", "out_addr"});
    add_kernel_fields_local(
        &contract, MpkGraphKernelFieldKind::Value,
        {"phys", "batch_size", "in_tensor_size", "out_tensor_size", "out_tensor_sizes"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"in_addr_list", "out_addr_list"});
  } else if (op == "pack") {
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"inputs"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value, {"input_shapes"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter, {"output_shape"});
  } else if (op == "unpack") {
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"input_tensor"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Value, {"input_shape"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter, {"output_shapes"});
  } else if (op == "slice") {
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"input_tensor"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter,
                            {"slice_begin", "slice_end", "slice_shape"});
  } else if (op == "pass_through") {
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"inputs"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter, {"output_names"});
  } else {
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Argument, {"input_tensor"});
    add_kernel_fields_local(&contract, MpkGraphKernelFieldKind::Parameter, {"kernel_params"});
  }

  return contract;
}

std::string bool_dbg_local(const bool value) {
  return value ? "true" : "false";
}

std::string shapes_dbg_local(const std::vector<MpkTensorContract>& tensors) {
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const auto& tensor : tensors) {
    if (tensor.mpk_shape.empty()) {
      continue;
    }
    if (!first) {
      out << ",";
    }
    out << ints_dbg_local(tensor.mpk_shape);
    first = false;
  }
  out << "]";
  return out.str();
}

std::optional<int> resolved_batch_size_local(const MpkPluginIoContract& stage) {
  int batch_size = stage.batch_sz_model > 0 ? stage.batch_sz_model : stage.batch_size;
  if (batch_size > 0) {
    return batch_size;
  }
  if (const auto output_shape = primary_output_shape_local(stage);
      output_shape.has_value() && !output_shape->empty() && output_shape->front() > 0) {
    return static_cast<int>(output_shape->front());
  }
  if (const auto input_shape = primary_input_shape_local(stage);
      input_shape.has_value() && !input_shape->empty() && input_shape->front() > 0) {
    return static_cast<int>(input_shape->front());
  }
  return std::nullopt;
}

const MpkPluginIoContract* find_stage_by_name_local(const MpkContract& contract,
                                                    const std::string& stage_name) {
  if (stage_name.empty()) {
    return nullptr;
  }
  for (const auto& stage : contract.plugins) {
    if (stage.name == stage_name || (!stage.plugin_id.empty() && stage.plugin_id == stage_name)) {
      return &stage;
    }
  }
  return nullptr;
}

std::vector<const MpkPluginIoContract*> resolve_member_stages_local(const MpkContract& contract,
                                                                    const MpkGraphNode& node) {
  std::vector<const MpkPluginIoContract*> out;
  std::unordered_set<std::string> seen;
  for (const auto& member_name : node.member_node_ids) {
    if (member_name.empty() || !seen.insert(member_name).second) {
      continue;
    }
    if (const auto* stage = find_stage_by_name_local(contract, member_name); stage != nullptr) {
      out.push_back(stage);
    }
  }
  return out;
}

std::string graph_fill_node_name_local(const MpkGraphNode& node) {
  if (!node.label.empty()) {
    return node.label;
  }
  if (!node.name.empty()) {
    return node.name;
  }
  return node.node_id;
}

[[noreturn]] void throw_graph_fill_error_local(MpkContract* contract, const MpkGraphNode& node,
                                               const std::string& detail) {
  const std::string message =
      "mpk graph fill failed for node '" + graph_fill_node_name_local(node) + "': " + detail;
  if (contract) {
    contract->errors.push_back({MpkContractErrorCode::GraphFillFailed, message});
  }
  throw std::runtime_error(message);
}

const MpkPluginIoContract& require_stage_for_graph_node_local(MpkContract* contract,
                                                              const MpkGraphNode& node) {
  const auto* stage = find_stage_by_name_local(*contract, node.name);
  if (!stage) {
    throw_graph_fill_error_local(contract, node,
                                 "could not resolve backing MPK plugin stage '" + node.name + "'");
  }
  return *stage;
}

const MpkPluginIoContract&
require_member_stage_by_op_local(MpkContract* contract, const MpkGraphNode& node,
                                 const std::vector<const MpkPluginIoContract*>& members,
                                 const std::string& op) {
  for (const auto* stage : members) {
    if (!stage) {
      continue;
    }
    if (canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor) == op) {
      return *stage;
    }
  }
  throw_graph_fill_error_local(
      contract, node, "could not resolve fused member stage with canonical op '" + op + "'");
}

const MpkTensorContract& require_first_input_tensor_local(MpkContract* contract,
                                                          const MpkGraphNode& node,
                                                          const MpkPluginIoContract& stage) {
  if (stage.input_tensors.empty()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing input tensor metadata");
  }
  return stage.input_tensors.front();
}

const MpkTensorContract& require_first_output_tensor_local(MpkContract* contract,
                                                           const MpkGraphNode& node,
                                                           const MpkPluginIoContract& stage) {
  if (stage.output_tensors.empty()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing output tensor metadata");
  }
  return stage.output_tensors.front();
}

std::vector<std::int64_t>
require_primary_input_shape_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                           const MpkPluginIoContract& stage) {
  const auto shape = primary_input_shape_local(stage);
  if (!shape.has_value()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing a primary input shape");
  }
  return *shape;
}

std::vector<std::int64_t>
require_primary_output_shape_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                            const MpkPluginIoContract& stage) {
  const auto shape = primary_output_shape_local(stage);
  if (!shape.has_value()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing a primary output shape");
  }
  return *shape;
}

int require_resolved_batch_size_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                               const MpkPluginIoContract& stage) {
  const auto batch = resolved_batch_size_local(stage);
  if (!batch.has_value()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing a resolved batch size");
  }
  return *batch;
}

const MpkQuantContract& require_quant_contract_for_fill_local(MpkContract* contract,
                                                              const MpkGraphNode& node,
                                                              const MpkPluginIoContract& stage) {
  if (!stage.quant.has_value()) {
    throw_graph_fill_error_local(contract, node,
                                 "stage '" + stage.name + "' is missing quant parameters");
  }
  return *stage.quant;
}

std::vector<std::int64_t> preferred_tensor_shape_local(const MpkTensorContract& tensor);

std::vector<std::vector<std::int64_t>> require_tensor_shapes_for_fill_local(
    MpkContract* contract, const MpkGraphNode& node, const MpkPluginIoContract& stage,
    const std::vector<MpkTensorContract>& tensors, const std::string& context) {
  std::vector<std::vector<std::int64_t>> shapes;
  shapes.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto shape = preferred_tensor_shape_local(tensors[i]);
    if (shape.empty()) {
      throw_graph_fill_error_local(contract, node,
                                   "stage '" + stage.name + "' has a " + context +
                                       " tensor without semantic shape at index " +
                                       std::to_string(i));
    }
    shapes.push_back(shape);
  }
  return shapes;
}

void require_nhwc_dims_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                      const std::vector<std::int64_t>& shape,
                                      const std::string& shape_name, int* h, int* w, int* c) {
  if (!nhwc_dims_local(shape, h, w, c)) {
    throw_graph_fill_error_local(contract, node,
                                 shape_name + " does not map cleanly to NHWC geometry");
  }
}

void require_slice_dhwc_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                       const std::vector<std::int64_t>& shape,
                                       const std::string& shape_name, int* d, int* h, int* w,
                                       int* c) {
  if (!canonical_slice_dhwc_from_shape(shape, d, h, w, c)) {
    throw_graph_fill_error_local(contract, node,
                                 shape_name + " does not map cleanly to slice DHWC geometry");
  }
}

void require_quanttess_tile_hwd_for_fill_local(MpkContract* contract, const MpkGraphNode& node,
                                               const std::vector<std::int64_t>& shape,
                                               const std::string& shape_name, int* h, int* w,
                                               int* d) {
  if (!h || !w || !d || shape.size() != 3U) {
    throw_graph_fill_error_local(contract, node,
                                 shape_name + " does not map cleanly to quanttess HWD geometry");
  }
  auto to_pos = [](std::int64_t value) -> int {
    if (value <= 0) {
      return 0;
    }
    if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
  };
  *h = to_pos(shape[0]);
  *w = to_pos(shape[1]);
  *d = to_pos(shape[2]);
  if (*h <= 0 || *w <= 0 || *d <= 0) {
    throw_graph_fill_error_local(contract, node,
                                 shape_name + " does not map cleanly to quanttess HWD geometry");
  }
}

std::optional<std::vector<std::int64_t>>
primary_input_shape_local(const MpkPluginIoContract& stage) {
  if (!stage.input_tensors.empty()) {
    if (!stage.input_tensors.front().logical_shape.empty()) {
      return stage.input_tensors.front().logical_shape;
    }
    if (!stage.input_tensors.front().mpk_shape.empty()) {
      return stage.input_tensors.front().mpk_shape;
    }
  }
  return std::nullopt;
}

std::optional<std::vector<std::int64_t>>
primary_output_shape_local(const MpkPluginIoContract& stage) {
  if (!stage.output_tensors.empty() && !stage.output_tensors.front().mpk_shape.empty()) {
    return stage.output_tensors.front().mpk_shape;
  }
  if (!stage.out_shape_raw.empty()) {
    return stage.out_shape_raw;
  }
  return std::nullopt;
}

std::string primary_output_dtype_local(const MpkPluginIoContract& stage) {
  if (!stage.output_tensors.empty() && !stage.output_tensors.front().dtype.empty()) {
    return stage.output_tensors.front().dtype;
  }
  return stage.canonical_output_dtype;
}

bool nhwc_dims_local(const std::vector<std::int64_t>& shape, int* out_h, int* out_w, int* out_c) {
  int n = 0;
  return canonical_nhwc_from_shape(shape, &n, out_h, out_w, out_c);
}

std::optional<std::vector<std::int64_t>>
contract_model_input_shape_local(const MpkContract& contract) {
  for (const auto& ingress : contract.ingress_tensors) {
    for (const auto& stage : contract.plugins) {
      if (stage.input_tensors.empty()) {
        continue;
      }
      if (stage.input_tensors.front().name != ingress.name) {
        continue;
      }
      if (const auto shape = primary_input_shape_local(stage); shape.has_value()) {
        return shape;
      }
    }
  }
  for (const auto& stage : contract.plugins) {
    if (const auto shape = primary_input_shape_local(stage); shape.has_value()) {
      return shape;
    }
  }
  return std::nullopt;
}

constexpr const char* kMlaTransportByteDtypeLocal = "BYTE";

bool is_transport_only_graph_op_local(const std::string& canonical_op) {
  return canonical_op == "pack" || canonical_op == "unpack" || canonical_op == "slice" ||
         canonical_op == "pass_through";
}

bool is_all_zero_shape_local(const std::vector<std::int64_t>& values) {
  return !values.empty() && std::all_of(values.begin(), values.end(),
                                        [](const std::int64_t value) { return value == 0; });
}

std::vector<std::int64_t>
normalize_slice_begin_for_graph_local(const std::vector<std::int64_t>& begin) {
  if (begin.empty() || is_all_zero_shape_local(begin)) {
    return {};
  }
  return begin;
}

std::size_t mapped_output_index_for_input_local(const MpkPluginIoContract& stage, int input_index) {
  if (stage.output_tensors.empty()) {
    return 0U;
  }
  if (stage.output_tensors.size() == 1U) {
    return 0U;
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.output_tensors.size()) {
    return static_cast<std::size_t>(input_index);
  }
  return 0U;
}

std::size_t mapped_input_index_for_output_local(const MpkPluginIoContract& stage,
                                                int output_index) {
  if (stage.input_tensors.empty()) {
    return 0U;
  }
  if (stage.input_tensors.size() == 1U) {
    return 0U;
  }
  if (output_index >= 0 && static_cast<std::size_t>(output_index) < stage.input_tensors.size()) {
    return static_cast<std::size_t>(output_index);
  }
  return 0U;
}

std::optional<std::size_t> plugin_index_from_stage_ref_local(const MpkContract& contract,
                                                             const MpkPluginIoContract& stage) {
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == &stage) {
      return i;
    }
  }
  return std::nullopt;
}

const MpkContractEdge* incoming_edge_for_stage_input_local(const MpkContract& contract,
                                                           const std::size_t dst_plugin_index,
                                                           const int dst_input_index) {
  for (const auto& edge : contract.edges) {
    if (edge.dst_plugin_index == dst_plugin_index && edge.dst_input_index == dst_input_index) {
      return &edge;
    }
  }
  return nullptr;
}

std::vector<std::int64_t> preferred_tensor_shape_local(const MpkTensorContract& tensor);

const MpkPluginIoContract* nearest_producer_stage_for_tensor_name_local(
    const MpkContract& contract, const std::size_t consumer_stage_index,
    const std::string& tensor_name, int* out_output_index) {
  if (out_output_index) {
    *out_output_index = -1;
  }
  if (consumer_stage_index >= contract.plugins.size() || tensor_name.empty()) {
    return nullptr;
  }

  const std::size_t consumer_order =
      plugin_order_key(contract.plugins[consumer_stage_index], consumer_stage_index);
  const MpkPluginIoContract* best = nullptr;
  std::size_t best_order = 0U;
  int best_output_index = -1;
  for (std::size_t pi = 0; pi < contract.plugins.size(); ++pi) {
    if (pi == consumer_stage_index) {
      continue;
    }
    const std::size_t order = plugin_order_key(contract.plugins[pi], pi);
    if (order >= consumer_order) {
      continue;
    }
    for (std::size_t oi = 0; oi < contract.plugins[pi].output_tensors.size(); ++oi) {
      if (contract.plugins[pi].output_tensors[oi].name != tensor_name) {
        continue;
      }
      if (!best || order >= best_order) {
        best = &contract.plugins[pi];
        best_order = order;
        best_output_index = static_cast<int>(oi);
      }
    }
  }
  if (best && out_output_index) {
    *out_output_index = best_output_index;
  }
  return best;
}

const MpkPluginIoContract*
nearest_consumer_stage_for_tensor_name_local(const MpkContract& contract,
                                             const std::size_t producer_stage_index,
                                             const std::string& tensor_name, int* out_input_index) {
  if (out_input_index) {
    *out_input_index = -1;
  }
  if (producer_stage_index >= contract.plugins.size() || tensor_name.empty()) {
    return nullptr;
  }

  const std::size_t producer_order =
      plugin_order_key(contract.plugins[producer_stage_index], producer_stage_index);
  const MpkPluginIoContract* best = nullptr;
  std::size_t best_order = std::numeric_limits<std::size_t>::max();
  int best_input_index = -1;
  for (std::size_t pi = 0; pi < contract.plugins.size(); ++pi) {
    if (pi == producer_stage_index) {
      continue;
    }
    const std::size_t order = plugin_order_key(contract.plugins[pi], pi);
    if (order <= producer_order) {
      continue;
    }
    for (std::size_t ii = 0; ii < contract.plugins[pi].input_tensors.size(); ++ii) {
      if (contract.plugins[pi].input_tensors[ii].name != tensor_name) {
        continue;
      }
      if (!best || order < best_order) {
        best = &contract.plugins[pi];
        best_order = order;
        best_input_index = static_cast<int>(ii);
      }
    }
  }
  if (best && out_input_index) {
    *out_input_index = best_input_index;
  }
  return best;
}

const MpkContractEdge* earliest_outgoing_edge_for_stage_output_local(
    const MpkContract& contract, const std::size_t src_plugin_index, const int src_output_index) {
  const MpkContractEdge* best = nullptr;
  std::size_t best_order = std::numeric_limits<std::size_t>::max();
  for (const auto& edge : contract.edges) {
    if (edge.src_plugin_index != src_plugin_index || edge.src_output_index != src_output_index ||
        edge.dst_plugin_index >= contract.plugins.size()) {
      continue;
    }
    const std::size_t order =
        plugin_order_key(contract.plugins[edge.dst_plugin_index], edge.dst_plugin_index);
    if (!best || order < best_order) {
      best = &edge;
      best_order = order;
    }
  }
  return best;
}

std::string stage_output_semantic_dtype_hint_local(const MpkPluginIoContract& stage) {
  const std::string op = canonical_mpk_graph_op_local(stage.kernel, stage.name, stage.processor);
  if ((op == "tess" || op == "detess") && !stage.frame_type.empty()) {
    return normalize_dtype_local(stage.frame_type);
  }
  if ((op == "quant" || op == "quanttess" || op == "cast" || op == "dequantize" ||
       op == "detessdequant" || op == "preproc") &&
      !stage.canonical_output_dtype.empty()) {
    return normalize_dtype_local(stage.canonical_output_dtype);
  }
  if (!stage.output_tensors.empty()) {
    const auto& output = stage.output_tensors.front();
    if (!output.logical_dtype.empty()) {
      return normalize_dtype_local(output.logical_dtype);
    }
    if (!output.dtype.empty() && !is_transport_only_graph_op_local(op) && op != "mla") {
      return normalize_dtype_local(output.dtype);
    }
  }
  return {};
}

std::string stage_input_semantic_dtype_hint_local(const MpkPluginIoContract& stage,
                                                  const std::size_t input_index) {
  const std::string op = canonical_mpk_graph_op_local(stage.kernel, stage.name, stage.processor);
  if ((op == "tess" || op == "detess" || op == "detessdequant") && !stage.frame_type.empty()) {
    return normalize_dtype_local(stage.frame_type);
  }
  if (!stage.canonical_input_dtype.empty() && (op == "quant" || op == "quanttess" || op == "cast" ||
                                               op == "dequantize" || op == "boxdecode")) {
    return normalize_dtype_local(stage.canonical_input_dtype);
  }
  if (input_index < stage.input_tensors.size()) {
    const auto& input = stage.input_tensors[input_index];
    if (!input.logical_dtype.empty()) {
      return normalize_dtype_local(input.logical_dtype);
    }
    if (!input.dtype.empty() && !is_transport_only_graph_op_local(op) && op != "mla") {
      return normalize_dtype_local(input.dtype);
    }
  }
  if (!stage.canonical_input_dtype.empty()) {
    return normalize_dtype_local(stage.canonical_input_dtype);
  }
  if (!stage.frame_type.empty()) {
    return normalize_dtype_local(stage.frame_type);
  }
  return {};
}

std::string infer_semantic_output_dtype_from_stage_output_local(
    const MpkContract& contract, const std::size_t stage_index, const int output_index,
    std::unordered_set<std::uint64_t>* visited) {
  if (stage_index >= contract.plugins.size() || !visited) {
    return {};
  }
  const std::uint64_t visit_key =
      (static_cast<std::uint64_t>(stage_index) << 32U) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_index));
  if (!visited->insert(visit_key).second) {
    return {};
  }

  const auto& stage = contract.plugins[stage_index];
  if (const std::string direct = stage_output_semantic_dtype_hint_local(stage); !direct.empty()) {
    return direct;
  }

  const std::string op = canonical_mpk_graph_op_local(stage.kernel, stage.name, stage.processor);
  if (!is_transport_only_graph_op_local(op)) {
    return {};
  }

  const std::size_t input_index = mapped_input_index_for_output_local(stage, output_index);
  const auto* incoming_edge =
      incoming_edge_for_stage_input_local(contract, stage_index, static_cast<int>(input_index));
  if (!incoming_edge) {
    return {};
  }
  return infer_semantic_output_dtype_from_stage_output_local(
      contract, incoming_edge->src_plugin_index, incoming_edge->src_output_index, visited);
}

std::string infer_semantic_input_dtype_from_downstream_local(
    const MpkContract& contract, const std::size_t stage_index, const int output_index,
    std::unordered_set<std::uint64_t>* visited) {
  if (stage_index >= contract.plugins.size() || !visited) {
    return {};
  }
  const std::uint64_t visit_key =
      (static_cast<std::uint64_t>(stage_index) << 32U) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_index));
  if (!visited->insert(visit_key).second) {
    return {};
  }

  const auto* outgoing_edge =
      earliest_outgoing_edge_for_stage_output_local(contract, stage_index, output_index);
  if (!outgoing_edge || outgoing_edge->dst_plugin_index >= contract.plugins.size()) {
    return {};
  }

  const auto& consumer = contract.plugins[outgoing_edge->dst_plugin_index];
  const std::string consumer_op =
      canonical_mpk_graph_op_local(consumer.kernel, consumer.name, consumer.processor);
  const std::size_t input_index = outgoing_edge->dst_input_index >= 0
                                      ? static_cast<std::size_t>(outgoing_edge->dst_input_index)
                                      : 0U;
  if (const std::string direct = stage_input_semantic_dtype_hint_local(consumer, input_index);
      !direct.empty() && !is_transport_only_graph_op_local(consumer_op)) {
    return direct;
  }
  if (!is_transport_only_graph_op_local(consumer_op)) {
    return {};
  }
  const std::size_t consumer_output_index =
      mapped_output_index_for_input_local(consumer, outgoing_edge->dst_input_index);
  return infer_semantic_input_dtype_from_downstream_local(
      contract, outgoing_edge->dst_plugin_index, static_cast<int>(consumer_output_index), visited);
}

std::vector<std::int64_t> preferred_tensor_shape_local(const MpkTensorContract& tensor) {
  if (!tensor.logical_shape.empty()) {
    return tensor.logical_shape;
  }
  return tensor.mpk_shape;
}

bool looks_like_packed_extent_shape_local(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return true;
  }
  if (shape.size() == 1U) {
    return true;
  }
  if (shape.size() == 2U && shape.front() == 1) {
    return true;
  }
  if (shape.size() == 3U && shape.front() == 1 && shape.back() == 1) {
    return true;
  }
  return false;
}

std::vector<std::int64_t> infer_semantic_input_shape_from_downstream_local(
    const MpkContract& contract, const std::size_t stage_index, const int output_index,
    std::unordered_set<std::uint64_t>* visited) {
  if (stage_index >= contract.plugins.size() || !visited) {
    return {};
  }
  const std::uint64_t visit_key =
      (static_cast<std::uint64_t>(stage_index) << 32U) |
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(output_index));
  if (!visited->insert(visit_key).second) {
    return {};
  }

  const auto* outgoing_edge =
      earliest_outgoing_edge_for_stage_output_local(contract, stage_index, output_index);
  if (!outgoing_edge || outgoing_edge->dst_plugin_index >= contract.plugins.size()) {
    return {};
  }

  const auto& consumer = contract.plugins[outgoing_edge->dst_plugin_index];
  const std::string consumer_op =
      canonical_mpk_graph_op_local(consumer.kernel, consumer.name, consumer.processor);
  const std::size_t input_index = outgoing_edge->dst_input_index >= 0
                                      ? static_cast<std::size_t>(outgoing_edge->dst_input_index)
                                      : 0U;
  if (input_index < consumer.input_tensors.size()) {
    const auto shape = preferred_tensor_shape_local(consumer.input_tensors[input_index]);
    if (!shape.empty()) {
      return shape;
    }
  }
  if (!is_transport_only_graph_op_local(consumer_op)) {
    return {};
  }
  const std::size_t consumer_output_index =
      mapped_output_index_for_input_local(consumer, outgoing_edge->dst_input_index);
  return infer_semantic_input_shape_from_downstream_local(
      contract, outgoing_edge->dst_plugin_index, static_cast<int>(consumer_output_index), visited);
}

struct ResolvedMlaGraphOutputLocal {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> slice_begin;
  std::size_t size_bytes = 0U;
  std::string semantic_dtype;
};

ResolvedMlaGraphOutputLocal resolve_mla_graph_output_local(const MpkContract& contract,
                                                           const std::size_t producer_stage_index,
                                                           const int producer_output_index,
                                                           const MpkTensorContract& base_output) {
  ResolvedMlaGraphOutputLocal resolved;
  resolved.name = base_output.name;
  resolved.shape = preferred_tensor_shape_local(base_output);
  resolved.size_bytes = base_output.size_bytes;
  if (!base_output.logical_dtype.empty()) {
    resolved.semantic_dtype = normalize_dtype_local(base_output.logical_dtype);
  }

  int consumer_input_index = -1;
  const auto* consumer = nearest_consumer_stage_for_tensor_name_local(
      contract, producer_stage_index, base_output.name, &consumer_input_index);
  if (!consumer) {
    return resolved;
  }

  const std::string consumer_op =
      canonical_mpk_graph_op_local(consumer->kernel, consumer->name, consumer->processor);
  if (consumer_op == "slice") {
    const auto slice_output_index =
        mapped_output_index_for_input_local(*consumer, consumer_input_index);
    if (slice_output_index < consumer->output_tensors.size()) {
      const auto& slice_output = consumer->output_tensors[slice_output_index];
      if (!slice_output.name.empty()) {
        resolved.name = slice_output.name;
      }
      if (const auto slice_shape = preferred_tensor_shape_local(slice_output);
          !slice_shape.empty()) {
        resolved.shape = slice_shape;
      }
      if (slice_output.size_bytes > 0U) {
        resolved.size_bytes = slice_output.size_bytes;
      }
    }
    resolved.slice_begin = normalize_slice_begin_for_graph_local(consumer->slice_begin);
    if (const auto slice_stage_index = plugin_index_from_stage_ref_local(contract, *consumer);
        slice_stage_index.has_value()) {
      std::unordered_set<std::uint64_t> visited;
      resolved.semantic_dtype = infer_semantic_input_dtype_from_downstream_local(
          contract, *slice_stage_index, static_cast<int>(slice_output_index), &visited);
    }
    return resolved;
  }

  if (consumer_input_index >= 0) {
    if (static_cast<std::size_t>(consumer_input_index) < consumer->input_tensors.size()) {
      const auto consumer_shape = preferred_tensor_shape_local(
          consumer->input_tensors[static_cast<std::size_t>(consumer_input_index)]);
      if (!consumer_shape.empty()) {
        resolved.shape = consumer_shape;
      }
    }
    resolved.semantic_dtype = stage_input_semantic_dtype_hint_local(
        *consumer, static_cast<std::size_t>(consumer_input_index));
  }
  if (looks_like_packed_extent_shape_local(resolved.shape)) {
    std::unordered_set<std::uint64_t> visited;
    const auto inferred_shape = infer_semantic_input_shape_from_downstream_local(
        contract, producer_stage_index, producer_output_index, &visited);
    if (!inferred_shape.empty()) {
      resolved.shape = inferred_shape;
    }
  }
  return resolved;
}

std::string summarized_semantic_dtype_local(const std::vector<std::string>& values) {
  std::vector<std::string> unique;
  std::unordered_set<std::string> seen;
  for (const auto& value : values) {
    if (value.empty()) {
      continue;
    }
    if (seen.insert(value).second) {
      unique.push_back(value);
    }
  }
  if (unique.empty()) {
    return {};
  }
  if (unique.size() == 1U) {
    return unique.front();
  }
  return strings_dbg_local(unique);
}

void fill_cast_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const auto element_count = shape_element_count_local(output_shape);
  if (!element_count.has_value()) {
    throw_graph_fill_error_local(
        contract, *node, "stage '" + stage.name + "' has a non-dense or invalid cast output shape");
  }
  if (out.dtype.empty() && stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing cast output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "in",
                               in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "out",
                               out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "count",
                               std::to_string(*element_count));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_dtype",
                               !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = output_shape;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_quant_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, stage);
  const auto input_shapes = require_tensor_shapes_for_fill_local(
      contract, *node, stage, stage.input_tensors, "quant input");
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name +
                                     "' is missing quant scalar scale/zero-point values");
  }
  if (stage.round_off.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing round_off");
  }
  if (stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing quant output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(stage.input_tensors.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shapes", nested_shapes_dbg_local(input_shapes));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "q_scale_array", doubles_dbg_local(quant.scales));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp_array",
                               ints_dbg_local(quant.zero_points));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "round_off_array", "[\"" + stage.round_off + "\"]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype_array", "[\"" + stage.canonical_output_dtype + "\"]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "round_off", stage.round_off);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype", stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_tess_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto input_shapes = require_tensor_shapes_for_fill_local(contract, *node, stage,
                                                                 stage.input_tensors, "tess input");
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing tess slice_shape");
  }
  if (stage.frame_type.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing tess frame_type");
  }
  if (!stage.has_align_c16) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing tess align_c16");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(stage.input_tensors.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shapes", nested_shapes_dbg_local(input_shapes));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "input_dtype_array", "[\"" + stage.frame_type + "\"]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "byte_align_array", "[" + bool_dbg_local(stage.align_c16) + "]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "input_dtype", stage.frame_type);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "byte_align", bool_dbg_local(stage.align_c16));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_quanttess_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, stage);
  const auto input_shape = require_primary_input_shape_for_fill_local(contract, *node, stage);
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing quanttess slice_shape");
  }
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name +
                                     "' is missing quanttess scalar scale/zero-point values");
  }
  if (stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing quanttess output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(input_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype", stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_detess_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing detess slice_shape");
  }
  if (stage.frame_shape.empty() && !primary_output_shape_local(stage).has_value()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing detess frame/output shape");
  }
  if (stage.frame_type.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing detess frame_type");
  }

  const auto frame_shape =
      !stage.frame_shape.empty()
          ? stage.frame_shape
          : require_primary_output_shape_for_fill_local(contract, *node, stage);

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(stage.input_tensors.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(frame_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "input_dtype", stage.frame_type);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(frame_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : frame_shape;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_dequantize_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, stage);
  const auto input_shapes = require_tensor_shapes_for_fill_local(
      contract, *node, stage, stage.input_tensors, "dequant input");
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name +
                                     "' is missing dequant scalar scale/zero-point values");
  }
  if (stage.canonical_input_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing dequant input dtype");
  }
  if (stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing dequant output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(stage.input_tensors.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shapes", nested_shapes_dbg_local(input_shapes));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "q_scale_array", doubles_dbg_local(quant.scales));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp_array",
                               ints_dbg_local(quant.zero_points));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "in_dtype_array", "[\"" + stage.canonical_input_dtype + "\"]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype_array", "[\"" + stage.canonical_output_dtype + "\"]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "in_dtype", stage.canonical_input_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype", stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_detessdequant_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing detessdequant slice_shape");
  }
  if (stage.frame_shape.empty() && !primary_output_shape_local(stage).has_value()) {
    throw_graph_fill_error_local(
        contract, *node, "stage '" + stage.name + "' is missing detessdequant frame/output shape");
  }
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(
        contract, *node, "stage '" + stage.name + "' is missing detessdequant quant values");
  }

  const auto frame_shape =
      !stage.frame_shape.empty()
          ? stage.frame_shape
          : require_primary_output_shape_for_fill_local(contract, *node, stage);

  const std::string data_type =
      !stage.canonical_output_dtype.empty() ? stage.canonical_output_dtype : stage.frame_type;
  if (data_type.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing detessdequant data_type");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(stage.input_tensors.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(frame_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "data_type", data_type);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(frame_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : frame_shape;
  node->dtype = !out.dtype.empty() ? out.dtype : data_type;
}

void fill_preproc_plugin_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  const auto input_shape = require_primary_input_shape_for_fill_local(contract, *node, stage);
  const auto output_shapes = require_tensor_shapes_for_fill_local(
      contract, *node, stage, stage.output_tensors, "preproc output");
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing preproc output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_image", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_rgb_image", out.name);
  if (stage.output_tensors.size() > 1U && !stage.output_tensors[1].name.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                                 "output_tessellated_image", stage.output_tensors[1].name);
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(input_shape));
  if (!stage.slice_shape.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "slice_shape", ints_dbg_local(stage.slice_shape));
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_dtype", stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shapes", nested_shapes_dbg_local(output_shapes));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = output_shape;
  node->dtype = stage.canonical_output_dtype;
}

void fill_mla_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  if (stage.executable.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing MLA executable path");
  }
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, stage);
  if (stage.input_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing MLA input tensors");
  }
  if (stage.output_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing MLA output tensors");
  }

  auto semantic_dtype_from_tensor_local = [&](const MpkTensorContract& tensor,
                                              const MpkPluginIoContract* producer,
                                              const int producer_output_index) {
    std::string semantic_dtype;
    if (producer != nullptr) {
      semantic_dtype = stage_output_semantic_dtype_hint_local(*producer);
      if (semantic_dtype.empty() && producer_output_index >= 0 &&
          static_cast<std::size_t>(producer_output_index) < producer->output_tensors.size()) {
        const auto& producer_output =
            producer->output_tensors[static_cast<std::size_t>(producer_output_index)];
        if (!producer_output.logical_dtype.empty()) {
          semantic_dtype = normalize_dtype_local(producer_output.logical_dtype);
        } else if (!producer_output.dtype.empty() &&
                   canonical_mpk_graph_op_local(producer->kernel, producer->name,
                                                producer->processor) != "mla") {
          semantic_dtype = normalize_dtype_local(producer_output.dtype);
        }
      }
    }
    if (semantic_dtype.empty() && !tensor.logical_dtype.empty()) {
      semantic_dtype = normalize_dtype_local(tensor.logical_dtype);
    }
    if (semantic_dtype.empty()) {
      semantic_dtype = "UNKNOWN";
    }
    return semantic_dtype;
  };

  node->mla_metadata = {};
  node->mla_metadata.present = true;
  node->mla_metadata.input_transport_dtype = kMlaTransportByteDtypeLocal;
  node->mla_metadata.output_transport_dtype = kMlaTransportByteDtypeLocal;

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "path",
                               stage.executable);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  if (stage.input_tensors.size() == 1U) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "in_tensor_size",
                                 std::to_string(stage.input_tensors.front().size_bytes));
  }
  if (stage.output_tensors.size() == 1U) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "out_tensor_size",
                                 std::to_string(stage.output_tensors.front().size_bytes));
  } else {
    std::vector<std::int64_t> output_sizes;
    for (const auto& tensor : stage.output_tensors) {
      output_sizes.push_back(static_cast<std::int64_t>(tensor.size_bytes));
    }
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "out_tensor_sizes", ints_dbg_local(output_sizes));
  }

  const auto stage_index = plugin_index_from_stage_ref_local(*contract, stage);
  if (stage_index.has_value()) {
    const MpkPluginIoContract* explicit_pack_stage = nullptr;
    if (stage.input_tensors.size() == 1U && !stage.input_tensors.front().name.empty()) {
      int producer_output_index = -1;
      if (const auto* producer = nearest_producer_stage_for_tensor_name_local(
              *contract, *stage_index, stage.input_tensors.front().name, &producer_output_index);
          producer != nullptr && canonical_mpk_graph_op_local(producer->kernel, producer->name,
                                                              producer->processor) == "pack") {
        explicit_pack_stage = producer;
      }
    }

    node->mla_metadata.pack.present = explicit_pack_stage != nullptr;

    const auto& mla_input_parts =
        explicit_pack_stage != nullptr ? explicit_pack_stage->input_tensors : stage.input_tensors;
    node->mla_metadata.input_sizes.reserve(mla_input_parts.size());
    node->mla_metadata.input_semantic_dtypes.reserve(mla_input_parts.size());
    bool any_pre_mla_tess_input = false;
    for (std::size_t ii = 0; ii < mla_input_parts.size(); ++ii) {
      const auto& input_tensor = mla_input_parts[ii];
      node->mla_metadata.input_sizes.push_back(input_tensor.size_bytes);

      int producer_output_index = -1;
      const MpkPluginIoContract* producer = nullptr;
      if (!input_tensor.name.empty()) {
        producer = nearest_producer_stage_for_tensor_name_local(
            *contract,
            explicit_pack_stage != nullptr
                ? plugin_index_from_stage_ref_local(*contract, *explicit_pack_stage)
                      .value_or(*stage_index)
                : *stage_index,
            input_tensor.name, &producer_output_index);
      }
      if (producer != nullptr && canonical_mpk_graph_op_local(producer->kernel, producer->name,
                                                              producer->processor) == "tess") {
        any_pre_mla_tess_input = true;
      }
      node->mla_metadata.input_semantic_dtypes.push_back(
          semantic_dtype_from_tensor_local(input_tensor, producer, producer_output_index));
    }

    // Path (1) defensive gate: this BF16-doubling workaround is for the
    // monolithic-IFM compilation path (explicit pack producer + single MLA
    // input). It must NOT fire when the .elf is a multi-IFM compilation —
    // those placeholder sizes are already correct as declared in the MPK and
    // the dispatcher addresses each one independently.
    const bool multi_ifm_consumer = mla_consumer_keeps_distinct_physical_inputs(*contract);
    if (!multi_ifm_consumer && explicit_pack_stage != nullptr && stage.input_tensors.size() == 1U &&
        !node->mla_metadata.input_sizes.empty() && !any_pre_mla_tess_input) {
      const bool all_bf16_inputs =
          std::all_of(node->mla_metadata.input_semantic_dtypes.begin(),
                      node->mla_metadata.input_semantic_dtypes.end(),
                      [](const std::string& dtype) { return dtype == "BF16"; });
      std::size_t logical_input_bytes = 0U;
      for (const auto size_bytes : node->mla_metadata.input_sizes) {
        logical_input_bytes += size_bytes;
      }
      const std::size_t packed_input_bytes = stage.input_tensors.front().size_bytes;
      // Compiler bug workaround: some explicit BF16 pack stages without pre-MLA tess
      // report logical part sizes at half the packed carrier byte count.
      if (all_bf16_inputs && logical_input_bytes > 0U &&
          packed_input_bytes == logical_input_bytes * 2U) {
        for (auto& size_bytes : node->mla_metadata.input_sizes) {
          size_bytes *= 2U;
        }
      }
    }

    const auto* unpack_stage = get_mla_unpack_stage_io_contract(*contract);
    const bool explicit_unpack = unpack_stage != nullptr;
    std::vector<MpkTensorContract> implicit_unpack_like_outputs;
    if (!explicit_unpack) {
      implicit_unpack_like_outputs.reserve(stage.output_tensors.size());
      for (std::size_t oi = 0; oi < stage.output_tensors.size(); ++oi) {
        MpkTensorContract output = stage.output_tensors[oi];
        if (output.mpk_shape.empty()) {
          output.mpk_shape = preferred_tensor_shape_local(output);
        }
        if (output.mpk_shape.empty()) {
          std::unordered_set<std::uint64_t> visited;
          output.mpk_shape = infer_semantic_input_shape_from_downstream_local(
              *contract, *stage_index, static_cast<int>(oi), &visited);
        }
        implicit_unpack_like_outputs.push_back(std::move(output));
      }
    }
    const bool implicit_published_outputs =
        !explicit_unpack && !implicit_unpack_like_outputs.empty();
    node->mla_metadata.unpack.present =
        explicit_unpack || implicit_published_outputs || stage.output_tensors.size() > 1U;
    node->mla_metadata.unpack.explicit_from_mpk = explicit_unpack;
    node->mla_metadata.unpack.source_stage =
        explicit_unpack
            ? (!unpack_stage->name.empty() ? unpack_stage->name : unpack_stage->plugin_id)
            : (!stage.name.empty() ? stage.name : stage.plugin_id);

    if (explicit_unpack) {
      if (!unpack_stage->input_tensors.empty()) {
        const auto unpack_input_shape =
            preferred_tensor_shape_local(unpack_stage->input_tensors.front());
        if (!unpack_input_shape.empty()) {
          node->mla_metadata.unpack.input_shape = unpack_input_shape;
        }
      }
      node->mla_metadata.unpack.output_count =
          static_cast<int>(unpack_stage->output_tensors.size());
      node->mla_metadata.output_semantic_dtypes.reserve(unpack_stage->output_tensors.size());
      for (std::size_t oi = 0; oi < unpack_stage->output_tensors.size(); ++oi) {
        const auto& output = unpack_stage->output_tensors[oi];
        const auto resolved =
            resolve_mla_graph_output_local(*contract, *stage_index, static_cast<int>(oi), output);
        node->mla_metadata.unpack.output_names.push_back(!resolved.name.empty() ? resolved.name
                                                                                : output.name);
        node->mla_metadata.unpack.output_shapes.push_back(
            !resolved.shape.empty() ? resolved.shape : preferred_tensor_shape_local(output));
        node->mla_metadata.unpack.output_slice_begins.push_back(resolved.slice_begin);
        node->mla_metadata.unpack.output_sizes.push_back(
            resolved.size_bytes > 0U ? resolved.size_bytes : output.size_bytes);

        std::string semantic_dtype = resolved.semantic_dtype;
        if (semantic_dtype.empty() && !output.logical_dtype.empty()) {
          semantic_dtype = normalize_dtype_local(output.logical_dtype);
        }
        node->mla_metadata.output_semantic_dtypes.push_back(
            semantic_dtype.empty() ? std::string("UNKNOWN") : semantic_dtype);
      }
    } else {
      if (stage.output_tensors.size() == 1U) {
        const auto stage_output_shape = preferred_tensor_shape_local(stage.output_tensors.front());
        if (!stage_output_shape.empty()) {
          node->mla_metadata.unpack.input_shape = stage_output_shape;
        }
      }
      const auto& unpack_like_outputs =
          implicit_published_outputs ? implicit_unpack_like_outputs : stage.output_tensors;
      node->mla_metadata.unpack.output_count = static_cast<int>(unpack_like_outputs.size());
      node->mla_metadata.output_semantic_dtypes.reserve(unpack_like_outputs.size());
      for (std::size_t oi = 0; oi < unpack_like_outputs.size(); ++oi) {
        const auto& output = unpack_like_outputs[oi];
        const auto resolved =
            resolve_mla_graph_output_local(*contract, *stage_index, static_cast<int>(oi), output);
        node->mla_metadata.unpack.output_names.push_back(!resolved.name.empty() ? resolved.name
                                                                                : output.name);
        node->mla_metadata.unpack.output_shapes.push_back(
            !resolved.shape.empty() ? resolved.shape : output.mpk_shape);
        node->mla_metadata.unpack.output_slice_begins.push_back(resolved.slice_begin);
        node->mla_metadata.unpack.output_sizes.push_back(
            resolved.size_bytes > 0U ? resolved.size_bytes : output.size_bytes);

        std::string semantic_dtype = resolved.semantic_dtype;
        if (semantic_dtype.empty() && !output.logical_dtype.empty()) {
          semantic_dtype = normalize_dtype_local(output.logical_dtype);
        }
        node->mla_metadata.output_semantic_dtypes.push_back(
            semantic_dtype.empty() ? std::string("UNKNOWN") : semantic_dtype);
      }
    }
  }

  const auto& out = stage.output_tensors.front();
  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = kMlaTransportByteDtypeLocal;
}

void fill_pack_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  if (stage.input_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing pack input tensors");
  }
  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  std::vector<std::string> input_names;
  std::vector<std::string> input_shapes;
  for (const auto& tensor : stage.input_tensors) {
    if (tensor.name.empty()) {
      throw_graph_fill_error_local(contract, *node,
                                   "stage '" + stage.name + "' has a pack input without name");
    }
    if (tensor.mpk_shape.empty()) {
      throw_graph_fill_error_local(contract, *node,
                                   "stage '" + stage.name + "' has a pack input without shape");
    }
    input_names.push_back(tensor.name);
    input_shapes.push_back(ints_dbg_local(tensor.mpk_shape));
  }
  const auto output_shape = require_primary_output_shape_for_fill_local(contract, *node, stage);

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "inputs",
                               strings_dbg_local(input_names));
  set_kernel_field_value_local(
      &node->kernel_contract, MpkGraphKernelFieldKind::Value,
      "input_shapes", "[" + [&input_shapes]() {
        std::ostringstream out;
        bool first = true;
        for (const auto& shape : input_shapes) {
          if (!first) {
            out << ",";
          }
          out << shape;
          first = false;
        }
        return out.str();
      }() + "]");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = output_shape;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_unpack_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  if (in.mpk_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing unpack input shape");
  }
  if (stage.output_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing unpack outputs");
  }
  std::vector<std::string> output_shapes;
  for (const auto& tensor : stage.output_tensors) {
    if (tensor.mpk_shape.empty()) {
      throw_graph_fill_error_local(contract, *node,
                                   "stage '" + stage.name + "' has an unpack output without shape");
    }
    output_shapes.push_back(ints_dbg_local(tensor.mpk_shape));
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(in.mpk_shape));
  set_kernel_field_value_local(
      &node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
      "output_shapes", "[" + [&output_shapes]() {
        std::ostringstream out;
        bool first = true;
        for (const auto& shape : output_shapes) {
          if (!first) {
            out << ",";
          }
          out << shape;
          first = false;
        }
        return out.str();
      }() + "]");

  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_slice_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  const auto& in = require_first_input_tensor_local(contract, *node, stage);
  if (stage.slice_begin.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing slice_begin");
  }
  if (stage.slice_end.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing slice_end");
  }
  if (stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing slice_shape");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "slice_begin", ints_dbg_local(stage.slice_begin));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "slice_end", ints_dbg_local(stage.slice_end));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "slice_shape", ints_dbg_local(stage.slice_shape));

  const auto& out = require_first_output_tensor_local(contract, *node, stage);
  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_pass_through_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto& stage = require_stage_for_graph_node_local(contract, *node);
  if (stage.input_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing passthrough inputs");
  }
  if (stage.output_tensors.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "stage '" + stage.name + "' is missing passthrough outputs");
  }
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  for (const auto& input : stage.input_tensors) {
    if (input.name.empty()) {
      throw_graph_fill_error_local(
          contract, *node, "stage '" + stage.name + "' has a passthrough input without name");
    }
    inputs.push_back(input.name);
  }
  for (const auto& output : stage.output_tensors) {
    if (output.name.empty()) {
      throw_graph_fill_error_local(
          contract, *node, "stage '" + stage.name + "' has a passthrough output without name");
    }
    outputs.push_back(output.name);
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "inputs",
                               strings_dbg_local(inputs));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_names", strings_dbg_local(outputs));

  const auto& out = stage.output_tensors.front();
  node->tensor_kind = out.kind;
  node->size_bytes = out.size_bytes;
  node->mpk_shape = !out.mpk_shape.empty() ? out.mpk_shape : stage.out_shape_raw;
  node->dtype = !out.dtype.empty() ? out.dtype : stage.canonical_output_dtype;
}

void fill_preproc_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto members = resolve_member_stages_local(*contract, *node);
  const auto* cast_stage = [&members]() -> const MpkPluginIoContract* {
    for (const auto* stage : members) {
      if (!stage) {
        continue;
      }
      if (canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor) == "cast") {
        return stage;
      }
    }
    return nullptr;
  }();
  const auto* quant_stage = [&members]() -> const MpkPluginIoContract* {
    for (const auto* stage : members) {
      if (!stage) {
        continue;
      }
      if (canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor) == "quant") {
        return stage;
      }
    }
    return nullptr;
  }();
  const auto* quanttess_stage = [&members]() -> const MpkPluginIoContract* {
    for (const auto* stage : members) {
      if (!stage) {
        continue;
      }
      if (canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor) ==
          "quanttess") {
        return stage;
      }
    }
    return nullptr;
  }();
  const auto* tess_stage = [&members]() -> const MpkPluginIoContract* {
    for (const auto* stage : members) {
      if (!stage) {
        continue;
      }
      const auto op = canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor);
      if (op == "tess" || op == "quanttess") {
        return stage;
      }
    }
    return nullptr;
  }();
  const MpkPluginIoContract* input_stage = nullptr;
  if (cast_stage) {
    input_stage = cast_stage;
  } else if (quant_stage) {
    input_stage = quant_stage;
  } else if (quanttess_stage) {
    input_stage = quanttess_stage;
  }
  if (!input_stage) {
    throw_graph_fill_error_local(contract, *node,
                                 "fused preproc is missing a cast/quant/quanttess member stage");
  }
  const MpkPluginIoContract* output_stage = nullptr;
  if (tess_stage) {
    output_stage = tess_stage;
  } else if (quanttess_stage) {
    output_stage = quanttess_stage;
  } else if (quant_stage) {
    output_stage = quant_stage;
  } else {
    output_stage = input_stage;
  }

  const auto& input_tensor = require_first_input_tensor_local(contract, *node, *input_stage);
  const auto& output_tensor = require_first_output_tensor_local(contract, *node, *output_stage);
  const auto input_shape =
      require_primary_input_shape_for_fill_local(contract, *node, *input_stage);
  std::vector<std::vector<std::int64_t>> output_shapes;
  if (cast_stage) {
    output_shapes.push_back(
        require_primary_output_shape_for_fill_local(contract, *node, *cast_stage));
  }
  if (output_stage) {
    const auto output_shape =
        require_primary_output_shape_for_fill_local(contract, *node, *output_stage);
    if (output_shapes.empty() || output_shapes.back() != output_shape) {
      output_shapes.push_back(output_shape);
    }
  }
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, *input_stage);

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_image", input_tensor.name);
  if (cast_stage) {
    const auto& cast_out = require_first_output_tensor_local(contract, *node, *cast_stage);
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                                 "output_rgb_image", cast_out.name);
  }
  if (output_stage) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                                 "output_tessellated_image", output_tensor.name);
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(input_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));

  const auto* tile_stage = tess_stage ? tess_stage : quanttess_stage;
  if (tile_stage && !tile_stage->slice_shape.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "slice_shape", ints_dbg_local(tile_stage->slice_shape));
  }

  if (quant_stage && quant_stage->quant.has_value()) {
    const auto& quant = *quant_stage->quant;
    if (!quant.scales.empty()) {
      set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                   "q_scale", std::to_string(quant.scales.front()));
    }
    if (!quant.zero_points.empty()) {
      set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                                   std::to_string(quant.zero_points.front()));
    }
  }
  if (quanttess_stage && quanttess_stage->quant.has_value()) {
    const auto& quant = *quanttess_stage->quant;
    if (!quant.scales.empty()) {
      set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                   "q_scale", std::to_string(quant.scales.front()));
    }
    if (!quant.zero_points.empty()) {
      set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                                   std::to_string(quant.zero_points.front()));
    }
  }

  std::string output_dtype;
  if (cast_stage) {
    output_dtype = !cast_stage->canonical_output_dtype.empty()
                       ? cast_stage->canonical_output_dtype
                       : primary_output_dtype_local(*cast_stage);
  } else if (quant_stage) {
    output_dtype = !quant_stage->canonical_output_dtype.empty()
                       ? quant_stage->canonical_output_dtype
                       : primary_output_dtype_local(*quant_stage);
  } else if (quanttess_stage) {
    output_dtype = !quanttess_stage->canonical_output_dtype.empty()
                       ? quanttess_stage->canonical_output_dtype
                       : primary_output_dtype_local(*quanttess_stage);
  }
  if (output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused preproc is missing output dtype");
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_dtype", output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shapes", nested_shapes_dbg_local(output_shapes));

  node->tensor_kind = output_tensor.kind;
  node->size_bytes = output_tensor.size_bytes;
  node->mpk_shape =
      !output_tensor.mpk_shape.empty() ? output_tensor.mpk_shape : output_stage->out_shape_raw;
  node->dtype =
      !output_tensor.dtype.empty() ? output_tensor.dtype : output_stage->canonical_output_dtype;
}

void fill_boxdecode_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto members = resolve_member_stages_local(*contract, *node);
  std::vector<const MpkPluginIoContract*> cast_stages;
  std::vector<const MpkPluginIoContract*> dequant_stages;
  std::vector<const MpkPluginIoContract*> detess_stages;
  const MpkPluginIoContract* pass_through_stage = nullptr;
  for (const auto* stage : members) {
    if (!stage) {
      continue;
    }
    const std::string op =
        canonical_mpk_graph_op_local(stage->kernel, stage->name, stage->processor);
    if (op == "cast") {
      cast_stages.push_back(stage);
    } else if (op == "dequantize") {
      dequant_stages.push_back(stage);
    } else if (op == "detess" || op == "detessdequant") {
      detess_stages.push_back(stage);
    } else if (op == "pass_through") {
      pass_through_stage = stage;
    }
  }
  const std::vector<const MpkPluginIoContract*>* output_stages = nullptr;
  if (!cast_stages.empty()) {
    output_stages = &cast_stages;
  } else if (!dequant_stages.empty()) {
    output_stages = &dequant_stages;
  }
  if (!output_stages) {
    throw_graph_fill_error_local(contract, *node,
                                 "fused boxdecode is missing cast/dequant member stages");
  }

  auto by_sequence = [](const MpkPluginIoContract* a, const MpkPluginIoContract* b) {
    return a->sequence < b->sequence;
  };
  std::sort(cast_stages.begin(), cast_stages.end(), by_sequence);
  std::sort(dequant_stages.begin(), dequant_stages.end(), by_sequence);
  std::sort(detess_stages.begin(), detess_stages.end(), by_sequence);

  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_sizes;
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::vector<std::int64_t>> slice_shapes;
  std::vector<double> dq_scales;
  std::vector<std::int64_t> dq_zero_points;
  std::vector<std::string> input_dtypes;

  for (const auto* stage : *output_stages) {
    const auto& out = require_first_output_tensor_local(contract, *node, *stage);
    const auto shape = require_primary_output_shape_for_fill_local(contract, *node, *stage);
    input_names.push_back(out.name);
    input_sizes.push_back(static_cast<std::int64_t>(out.size_bytes));
    input_shapes.push_back(shape);
    const std::string dtype = !out.dtype.empty() ? out.dtype : stage->canonical_output_dtype;
    if (dtype.empty()) {
      throw_graph_fill_error_local(
          contract, *node, "boxdecode output stage '" + stage->name + "' is missing output dtype");
    }
    input_dtypes.push_back(dtype);
  }

  for (const auto* stage : detess_stages) {
    if (stage->slice_shape.empty()) {
      throw_graph_fill_error_local(
          contract, *node, "boxdecode detess stage '" + stage->name + "' is missing slice_shape");
    }
    slice_shapes.push_back(stage->slice_shape);
  }

  for (const auto* stage : dequant_stages) {
    const auto& quant = require_quant_contract_for_fill_local(contract, *node, *stage);
    if (quant.scales.empty() || quant.zero_points.empty()) {
      throw_graph_fill_error_local(
          contract, *node, "boxdecode dequant stage '" + stage->name + "' is missing quant values");
    }
    dq_scales.push_back(quant.scales.front());
    dq_zero_points.push_back(static_cast<std::int64_t>(quant.zero_points.front()));
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument, "in_data",
                               strings_dbg_local(input_names));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "in_data_size", ints_dbg_local(input_sizes));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", std::to_string(input_names.size()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shapes", nested_shapes_dbg_local(input_shapes));
  if (!slice_shapes.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                                 "slice_shapes", nested_shapes_dbg_local(slice_shapes));
  }
  if (!dq_scales.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_scale",
                                 doubles_dbg_local(dq_scales));
  }
  if (!dq_zero_points.empty()) {
    set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_zp",
                                 ints_dbg_local(dq_zero_points));
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "data_type", strings_dbg_local(input_dtypes));

  const auto model_input_shape = contract_model_input_shape_local(*contract);
  if (!model_input_shape.has_value()) {
    throw_graph_fill_error_local(contract, *node, "fused boxdecode is missing model input shape");
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "model_input_shape", ints_dbg_local(*model_input_shape));

  if (pass_through_stage && !pass_through_stage->output_tensors.empty()) {
    const auto& out = pass_through_stage->output_tensors.front();
    node->tensor_kind = out.kind;
    node->size_bytes = out.size_bytes;
    node->mpk_shape = out.mpk_shape;
    node->dtype = !out.dtype.empty() ? out.dtype : pass_through_stage->canonical_output_dtype;
  }
}

void fill_quanttess_fused_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto members = resolve_member_stages_local(*contract, *node);
  const auto& quant_stage = require_member_stage_by_op_local(contract, *node, members, "quant");
  const auto& tess_stage = require_member_stage_by_op_local(contract, *node, members, "tess");

  const auto& quant_in = require_first_input_tensor_local(contract, *node, quant_stage);
  const auto& tess_out = require_first_output_tensor_local(contract, *node, tess_stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, quant_stage);
  const auto input_shape = require_primary_input_shape_for_fill_local(contract, *node, quant_stage);
  const auto output_shape =
      require_primary_output_shape_for_fill_local(contract, *node, tess_stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, quant_stage);
  if (tess_stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused quanttess is missing tess slice_shape");
  }
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused quanttess is missing quant values");
  }
  if (quant_stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused quanttess is missing quant output dtype");
  }
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", quant_in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", tess_out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(input_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "q_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(tess_stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "out_dtype", quant_stage.canonical_output_dtype);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "output_shape", ints_dbg_local(output_shape));

  node->tensor_kind = tess_out.kind;
  node->size_bytes = tess_out.size_bytes;
  node->mpk_shape = !tess_out.mpk_shape.empty() ? tess_out.mpk_shape : tess_stage.out_shape_raw;
  node->dtype = !tess_out.dtype.empty() ? tess_out.dtype : tess_stage.canonical_output_dtype;
}

void fill_detessdequant_fused_node_from_mpk_local(MpkContract* contract, MpkGraphNode* node) {
  if (!contract || !node) {
    return;
  }
  const auto members = resolve_member_stages_local(*contract, *node);
  const auto& detess_stage = require_member_stage_by_op_local(contract, *node, members, "detess");
  const auto& dequant_stage =
      require_member_stage_by_op_local(contract, *node, members, "dequantize");

  const auto& detess_in = require_first_input_tensor_local(contract, *node, detess_stage);
  const auto& dequant_out = require_first_output_tensor_local(contract, *node, dequant_stage);
  const auto& quant = require_quant_contract_for_fill_local(contract, *node, dequant_stage);
  const int batch = require_resolved_batch_size_for_fill_local(contract, *node, detess_stage);
  if (detess_stage.slice_shape.empty()) {
    throw_graph_fill_error_local(contract, *node,
                                 "fused detessdequant is missing detess slice_shape");
  }
  const auto frame_shape =
      !detess_stage.frame_shape.empty()
          ? detess_stage.frame_shape
          : require_primary_output_shape_for_fill_local(contract, *node, detess_stage);
  if (quant.scales.empty() || quant.zero_points.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused detessdequant is missing dequant values");
  }
  if (dequant_stage.canonical_output_dtype.empty()) {
    throw_graph_fill_error_local(contract, *node, "fused detessdequant is missing output dtype");
  }

  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "input_tensor", detess_in.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Argument,
                               "output_tensor", dequant_out.name);
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "num_in_tensor", "1");
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "batch_size",
                               std::to_string(batch));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "input_shape", ints_dbg_local(frame_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value,
                               "slice_shape", ints_dbg_local(detess_stage.slice_shape));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_scale",
                               std::to_string(quant.scales.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Value, "dq_zp",
                               std::to_string(quant.zero_points.front()));
  set_kernel_field_value_local(&node->kernel_contract, MpkGraphKernelFieldKind::Parameter,
                               "data_type", dequant_stage.canonical_output_dtype);
  set_kernel_field_value_local(
      &node->kernel_contract, MpkGraphKernelFieldKind::Parameter, "output_shape",
      ints_dbg_local(require_primary_output_shape_for_fill_local(contract, *node, dequant_stage)));

  node->tensor_kind = dequant_out.kind;
  node->size_bytes = dequant_out.size_bytes;
  node->mpk_shape =
      !dequant_out.mpk_shape.empty() ? dequant_out.mpk_shape : dequant_stage.out_shape_raw;
  node->dtype =
      !dequant_out.dtype.empty() ? dequant_out.dtype : dequant_stage.canonical_output_dtype;
}

void fill_graph_nodes_from_mpk_local(std::vector<MpkGraphNode>* nodes, MpkContract* contract) {
  if (!nodes || !contract) {
    return;
  }
  for (auto& node : *nodes) {
    if (node.kind == MpkGraphNodeKind::FusedPreproc) {
      fill_preproc_node_from_mpk_local(contract, &node);
      continue;
    }
    if (node.kind == MpkGraphNodeKind::FusedBoxDecode) {
      fill_boxdecode_node_from_mpk_local(contract, &node);
      continue;
    }
    if (node.kind == MpkGraphNodeKind::FusedQuantTess) {
      fill_quanttess_fused_node_from_mpk_local(contract, &node);
      continue;
    }
    if (node.kind == MpkGraphNodeKind::FusedDetessDequant) {
      fill_detessdequant_fused_node_from_mpk_local(contract, &node);
      continue;
    }
    if (node.canonical_op == "cast") {
      fill_cast_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "quant") {
      fill_quant_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "tess") {
      fill_tess_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "quanttess") {
      fill_quanttess_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "detess") {
      fill_detess_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "dequantize") {
      fill_dequantize_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "detessdequant") {
      fill_detessdequant_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "preproc") {
      fill_preproc_plugin_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "mla") {
      fill_mla_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "pack") {
      fill_pack_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "unpack") {
      fill_unpack_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "slice") {
      fill_slice_node_from_mpk_local(contract, &node);
    } else if (node.canonical_op == "pass_through") {
      fill_pass_through_node_from_mpk_local(contract, &node);
    }
  }
}

void merge_graph_requirements_local(MpkGraphFusionRequirements* dst,
                                    const MpkGraphFusionRequirements& src) {
  if (!dst) {
    return;
  }
  dst->preproc = dst->preproc || src.preproc;
  dst->boxdecode = dst->boxdecode || src.boxdecode;
  dst->quantization = dst->quantization || src.quantization;
  dst->tessellation = dst->tessellation || src.tessellation;
  dst->detessellation = dst->detessellation || src.detessellation;
  dst->dequantization = dst->dequantization || src.dequantization;
  dst->cast = dst->cast || src.cast;
}

void mark_requirement_for_op_local(const std::string& canonical_op,
                                   MpkGraphFusionRequirements* requirements) {
  if (!requirements) {
    return;
  }
  if (canonical_op == "preproc") {
    requirements->preproc = true;
  } else if (canonical_op == "boxdecode") {
    requirements->boxdecode = true;
  } else if (canonical_op == "quant" || canonical_op == "quanttess") {
    requirements->quantization = true;
  }
  if (canonical_op == "tess" || canonical_op == "quanttess") {
    requirements->tessellation = true;
  }
  if (canonical_op == "detess" || canonical_op == "detessdequant") {
    requirements->detessellation = true;
  }
  if (canonical_op == "dequantize" || canonical_op == "detessdequant") {
    requirements->dequantization = true;
  }
  if (canonical_op == "cast") {
    requirements->cast = true;
  }
}

std::string requirements_summary_local(const MpkGraphFusionRequirements& requirements) {
  std::vector<std::string> tokens;
  if (requirements.preproc) {
    tokens.emplace_back("preproc");
  }
  if (requirements.boxdecode) {
    tokens.emplace_back("boxdecode");
  }
  if (requirements.quantization) {
    tokens.emplace_back("quantization");
  }
  if (requirements.tessellation) {
    tokens.emplace_back("tessellation");
  }
  if (requirements.detessellation) {
    tokens.emplace_back("detessellation");
  }
  if (requirements.dequantization) {
    tokens.emplace_back("dequantization");
  }
  if (requirements.cast) {
    tokens.emplace_back("cast");
  }
  if (tokens.empty()) {
    return "none";
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << tokens[i];
  }
  return oss.str();
}

std::string mermaid_escape_local(const std::string& raw) {
  std::string out;
  out.reserve(raw.size());
  for (const char c : raw) {
    switch (c) {
    case '&':
      out += "&amp;";
      break;
    case '"':
      out += "&quot;";
      break;
    case '\n':
      out += "<br/>";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

std::string mermaid_label_lines_local(const std::vector<std::string>& parts) {
  std::string out;
  bool first = true;
  for (const auto& part : parts) {
    if (part.empty()) {
      continue;
    }
    if (!first) {
      out += "<br/>";
    }
    out += mermaid_escape_local(part);
    first = false;
  }
  return out;
}

std::string mermaid_label_lines_local(std::initializer_list<std::string> parts) {
  return mermaid_label_lines_local(std::vector<std::string>(parts));
}

void append_kernel_contract_label_parts_local(const MpkGraphKernelContract& kernel_contract,
                                              std::vector<std::string>* parts) {
  if (!parts || kernel_contract.kernel_name.empty()) {
    return;
  }
  parts->push_back("kernel=" + kernel_contract.kernel_name);
  if (!kernel_contract.contract_type.empty()) {
    parts->push_back("contract=" + kernel_contract.contract_type);
  }

  auto append_group = [&](const char* prefix, const MpkGraphKernelFieldKind kind) {
    std::vector<std::string> entries;
    for (const auto& field : kernel_contract.fields) {
      if (field.kind != kind || field.name.empty()) {
        continue;
      }
      entries.push_back(field.name + "=" + (field.value.empty() ? "unknown" : field.value));
    }
    if (entries.empty()) {
      return;
    }
    constexpr std::size_t kChunkSize = 4U;
    for (std::size_t i = 0; i < entries.size(); i += kChunkSize) {
      std::ostringstream oss;
      oss << prefix << "=";
      for (std::size_t j = i; j < std::min(entries.size(), i + kChunkSize); ++j) {
        if (j > i) {
          oss << ",";
        }
        oss << entries[j];
      }
      parts->push_back(oss.str());
    }
  };

  append_group("args", MpkGraphKernelFieldKind::Argument);
  append_group("values", MpkGraphKernelFieldKind::Value);
  append_group("params", MpkGraphKernelFieldKind::Parameter);
}

void append_mla_metadata_label_parts_local(const MpkGraphNode& node,
                                           std::vector<std::string>* parts) {
  if (!parts || !node.mla_metadata.present) {
    return;
  }

  const auto& mla = node.mla_metadata;
  const auto& unpack = mla.unpack;
  parts->push_back("mla.input_transport_dtype=" + (mla.input_transport_dtype.empty()
                                                       ? std::string("unknown")
                                                       : mla.input_transport_dtype));
  parts->push_back("mla.output_transport_dtype=" + (mla.output_transport_dtype.empty()
                                                        ? std::string("unknown")
                                                        : mla.output_transport_dtype));
  parts->push_back("mla.pack.present=" + bool_dbg_local(mla.pack.present));
  if (!mla.input_semantic_dtypes.empty()) {
    parts->push_back("mla.input_semantic_dtypes=" + strings_dbg_local(mla.input_semantic_dtypes));
  }
  if (!mla.input_sizes.empty()) {
    parts->push_back("mla.input_sizes=" + sizes_dbg_local(mla.input_sizes));
  }
  if (!mla.output_semantic_dtypes.empty()) {
    parts->push_back("mla.output_semantic_dtypes=" + strings_dbg_local(mla.output_semantic_dtypes));
  }
  parts->push_back("mla.unpack.present=" + bool_dbg_local(unpack.present));
  if (unpack.present) {
    parts->push_back("mla.unpack.explicit_from_mpk=" + bool_dbg_local(unpack.explicit_from_mpk));
    if (!unpack.source_stage.empty()) {
      parts->push_back("mla.unpack.source_stage=" + unpack.source_stage);
    }
    if (!unpack.input_shape.empty()) {
      parts->push_back("mla.unpack.input_shape=" + ints_dbg_local(unpack.input_shape));
    }
    if (!unpack.output_names.empty()) {
      parts->push_back("mla.unpack.output_names=" + strings_dbg_local(unpack.output_names));
    }
    if (!unpack.output_shapes.empty()) {
      parts->push_back("mla.unpack.output_shapes=" + nested_shapes_dbg_local(unpack.output_shapes));
    }
    if (std::any_of(unpack.output_slice_begins.begin(), unpack.output_slice_begins.end(),
                    [](const std::vector<std::int64_t>& begin) { return !begin.empty(); })) {
      parts->push_back("mla.unpack.output_slice_begins=" +
                       nested_shapes_dbg_local(unpack.output_slice_begins));
    }
    if (!unpack.output_sizes.empty()) {
      parts->push_back("mla.unpack.output_sizes=" + sizes_dbg_local(unpack.output_sizes));
    }
    parts->push_back("mla.unpack.output_count=" + std::to_string(unpack.output_count));
  }
}

std::string mermaid_class_for_node_local(const MpkGraphNode& node) {
  switch (node.kind) {
  case MpkGraphNodeKind::IngressTensor:
    return "ingress";
  case MpkGraphNodeKind::FusedPreproc:
    return "fusedPreproc";
  case MpkGraphNodeKind::FusedBoxDecode:
    return "fusedBox";
  case MpkGraphNodeKind::FusedQuantTess:
    return "fusedPair";
  case MpkGraphNodeKind::FusedDetessDequant:
    return "fusedPair";
  case MpkGraphNodeKind::Plugin:
  case MpkGraphNodeKind::Unknown:
    return "plugin";
  }
  return "plugin";
}

std::string mermaid_node_definition_local(const std::string& mermaid_id, const MpkGraphNode& node) {
  const std::string requirements = requirements_summary_local(node.requirements);
  std::string label;
  if (node.kind == MpkGraphNodeKind::IngressTensor) {
    label = mermaid_label_lines_local(
        {"INGRESS", node.label,
         node.tensor_kind.empty() ? std::string() : "kind=" + node.tensor_kind,
         node.dtype.empty() ? std::string() : "dtype=" + node.dtype,
         node.size_bytes == 0U ? std::string() : "size=" + std::to_string(node.size_bytes)});
    return "  " + mermaid_id + "([\"" + label + "\"])\n";
  }

  std::vector<std::string> label_parts = {
      node.label,
      node.synthetic ? "synthetic=true" : std::string(),
      node.canonical_op.empty() ? std::string() : "op=" + node.canonical_op,
      requirements == "none" ? std::string() : "requires=" + requirements,
      node.branch_count > 1 ? "branches=" + std::to_string(node.branch_count) : std::string(),
      !node.member_node_ids.empty() ? "members=" + std::to_string(node.member_node_ids.size())
                                    : std::string(),
      node.sequence >= 0 ? "sequence=" + std::to_string(node.sequence) : std::string(),
  };
  append_kernel_contract_label_parts_local(node.kernel_contract, &label_parts);
  append_mla_metadata_label_parts_local(node, &label_parts);
  label = mermaid_label_lines_local(label_parts);
  return "  " + mermaid_id + "[\"" + label + "\"]\n";
}

std::string render_mermaid_graph_section_local(const std::string& title,
                                               const std::vector<MpkGraphNode>& nodes,
                                               const std::vector<MpkGraphEdge>& edges) {
  std::ostringstream oss;
  oss << "## " << title << "\n\n";
  oss << "```mermaid\n";
  oss << "flowchart LR\n";
  oss << "  classDef ingress fill:#dbeafe,stroke:#2563eb,color:#111827;\n";
  oss << "  classDef plugin fill:#f3f4f6,stroke:#6b7280,color:#111827;\n";
  oss << "  classDef fusedPreproc fill:#dcfce7,stroke:#16a34a,color:#111827;\n";
  oss << "  classDef fusedBox fill:#fef3c7,stroke:#d97706,color:#111827;\n";
  oss << "  classDef fusedPair fill:#ede9fe,stroke:#7c3aed,color:#111827;\n";

  std::unordered_map<std::string, std::string> mermaid_id_by_node;
  mermaid_id_by_node.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    mermaid_id_by_node.emplace(nodes[i].node_id, "n" + std::to_string(i));
  }

  for (const auto& node : nodes) {
    const auto id_it = mermaid_id_by_node.find(node.node_id);
    if (id_it == mermaid_id_by_node.end()) {
      continue;
    }
    oss << mermaid_node_definition_local(id_it->second, node);
    oss << "  class " << id_it->second << " " << mermaid_class_for_node_local(node) << ";\n";
  }

  oss << "\n";
  for (const auto& edge : edges) {
    const auto src_it = mermaid_id_by_node.find(edge.src_node_id);
    const auto dst_it = mermaid_id_by_node.find(edge.dst_node_id);
    if (src_it == mermaid_id_by_node.end() || dst_it == mermaid_id_by_node.end()) {
      continue;
    }
    const std::string label = mermaid_label_lines_local(
        {edge.tensor_name, "kind=" + std::string(mpk_graph_edge_kind_name_local(edge.kind))});
    oss << "  " << src_it->second << " -->|\"" << label << "\"| " << dst_it->second << "\n";
  }
  oss << "```\n\n";
  return oss.str();
}

std::string render_synthetic_node_details_local(const std::vector<MpkGraphNode>& nodes) {
  std::ostringstream oss;
  oss << "## Fusion Details\n\n";
  bool wrote_any = false;
  for (const auto& node : nodes) {
    if (!node.synthetic) {
      continue;
    }
    wrote_any = true;
    oss << "- `" << node.label << "` kind=`" << mpk_graph_node_kind_name_local(node.kind)
        << "` requires=`" << requirements_summary_local(node.requirements) << "`";
    if (!node.member_node_ids.empty()) {
      oss << " members=`";
      for (std::size_t i = 0; i < node.member_node_ids.size(); ++i) {
        if (i > 0U) {
          oss << ", ";
        }
        oss << node.member_node_ids[i];
      }
      oss << "`";
    }
    oss << "\n";
  }
  if (!wrote_any) {
    oss << "- none\n";
  }
  oss << "\n";
  return oss.str();
}

std::string render_mpk_graph_markdown_local(const MpkGraph& graph, const std::string& title) {
  std::ostringstream oss;
  oss << "# " << (title.empty() ? std::string("MPK Graph") : title) << "\n\n";
  oss << "Model: `" << (graph.model_name.empty() ? std::string("mpk_graph") : graph.model_name)
      << "`  \n";
  oss << "MPK JSON: `" << graph.mpk_json_path << "`\n\n";
  if (!graph.raw_nodes.empty()) {
    oss << render_mermaid_graph_section_local("Raw Graph", graph.raw_nodes, graph.raw_edges);
  }
  if (!graph.nodes.empty()) {
    oss << render_mermaid_graph_section_local("Graph", graph.nodes, graph.edges);
    oss << render_synthetic_node_details_local(graph.nodes);
  }
  return oss.str();
}

bool write_mpk_graph_dump_local(const MpkGraph& graph, const fs::path& output_path,
                                std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }
  if (output_path.empty()) {
    if (error_message) {
      *error_message = "empty output path";
    }
    return false;
  }
  std::error_code ec;
  if (!output_path.parent_path().empty()) {
    fs::create_directories(output_path.parent_path(), ec);
    if (ec) {
      if (error_message) {
        *error_message = "failed to create graph dump directory: " + ec.message();
      }
      return false;
    }
  }
  std::ofstream out(output_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    if (error_message) {
      *error_message = "failed to open graph dump path";
    }
    return false;
  }
  out << render_mpk_graph_markdown_local(graph, "MPK Graph");
  return out.good();
}

fs::path mpk_graph_output_path_local(const fs::path& package_root, const MpkContract& contract) {
  if (const char* raw = std::getenv("SIMA_MPK_GRAPH_OUTPUT_PATH"); raw && *raw) {
    return fs::path(raw);
  }
  std::string base_name;
  if (!contract.model_name.empty()) {
    base_name = graph_id_token_local(contract.model_name);
  } else if (!contract.mpk_json_path.empty()) {
    base_name = graph_id_token_local(fs::path(contract.mpk_json_path).stem().string());
  } else {
    base_name = "mpk";
  }
  return package_root / (base_name + "_graph.md");
}

} // namespace

std::string render_mpk_graph_markdown(const MpkGraph& graph, const std::string& title) {
  return render_mpk_graph_markdown_local(graph, title);
}

namespace {

struct WorkingGraphNode {
  MpkGraphNode node;
  bool removed = false;
};

struct WorkingGraphEdge {
  MpkGraphEdge edge;
  bool removed = false;
};

struct WorkingGraph {
  std::vector<WorkingGraphNode> nodes;
  std::vector<WorkingGraphEdge> edges;
  std::size_t next_synthetic_node = 0U;
  std::size_t next_synthetic_edge = 0U;
};

std::optional<std::size_t> working_node_index_local(const WorkingGraph& graph,
                                                    const std::string& node_id) {
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    if (!graph.nodes[i].removed && graph.nodes[i].node.node_id == node_id) {
      return i;
    }
  }
  return std::nullopt;
}

std::vector<std::size_t> working_incoming_edges_local(const WorkingGraph& graph,
                                                      const std::string& node_id) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    if (!graph.edges[i].removed && graph.edges[i].edge.dst_node_id == node_id) {
      out.push_back(i);
    }
  }
  return out;
}

std::vector<std::size_t> working_outgoing_edges_local(const WorkingGraph& graph,
                                                      const std::string& node_id) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    if (!graph.edges[i].removed && graph.edges[i].edge.src_node_id == node_id) {
      out.push_back(i);
    }
  }
  return out;
}

void mark_incident_edges_removed_local(WorkingGraph* graph, const std::string& node_id) {
  if (!graph) {
    return;
  }
  for (auto& edge : graph->edges) {
    if (edge.removed) {
      continue;
    }
    if (edge.edge.src_node_id == node_id || edge.edge.dst_node_id == node_id) {
      edge.removed = true;
    }
  }
}

void add_working_edge_local(WorkingGraph* graph, const std::string& src_node_id,
                            const std::string& dst_node_id, const std::string& tensor_name,
                            const MpkGraphEdgeKind kind) {
  if (!graph) {
    return;
  }
  MpkGraphEdge edge;
  edge.edge_id = "fused_edge:" + std::to_string(graph->next_synthetic_edge++);
  edge.src_node_id = src_node_id;
  edge.dst_node_id = dst_node_id;
  edge.tensor_name = tensor_name;
  edge.kind = kind;
  graph->edges.push_back(WorkingGraphEdge{std::move(edge), false});
}

void replace_working_output_tensor_name_local(MpkGraphNode* node, const std::string& old_name,
                                              const std::string& new_name) {
  if (!node || old_name.empty() || new_name.empty()) {
    return;
  }
  bool replaced = false;
  for (auto& output_name : node->output_tensor_names) {
    if (output_name == old_name) {
      output_name = new_name;
      replaced = true;
    }
  }
  if (!replaced) {
    node->output_tensor_names.push_back(new_name);
  }
  std::vector<std::string> deduped;
  deduped.reserve(node->output_tensor_names.size());
  std::unordered_set<std::string> seen;
  for (const auto& output_name : node->output_tensor_names) {
    if (output_name.empty() || !seen.insert(output_name).second) {
      continue;
    }
    deduped.push_back(output_name);
  }
  node->output_tensor_names = std::move(deduped);
}

void append_readable_member_labels_local(const MpkGraphNode& node, std::vector<std::string>* out) {
  if (!out) {
    return;
  }
  if (!node.member_node_ids.empty()) {
    out->insert(out->end(), node.member_node_ids.begin(), node.member_node_ids.end());
    return;
  }
  out->push_back(!node.label.empty() ? node.label : node.node_id);
}

MpkGraphNode make_fused_graph_node_local(WorkingGraph* graph, const MpkGraphNodeKind kind,
                                         const std::string& label, const std::string& canonical_op,
                                         const std::vector<std::size_t>& member_indices,
                                         const int branch_count) {
  MpkGraphNode node;
  node.kind = kind;
  node.synthetic = true;
  node.label = label;
  node.canonical_op = canonical_op;
  node.kernel = canonical_op;
  node.kernel_contract = kernel_contract_template_local(node.kernel, node.canonical_op);
  node.node_id =
      "fused:" + canonical_op + ":" + std::to_string(graph ? graph->next_synthetic_node++ : 0U);
  node.branch_count = branch_count;
  node.plugin_index = static_cast<std::size_t>(-1);
  node.tensor_index = -1;
  node.sequence = -1;
  for (const auto member_index : member_indices) {
    if (!graph || member_index >= graph->nodes.size() || graph->nodes[member_index].removed) {
      continue;
    }
    const auto& member = graph->nodes[member_index].node;
    append_readable_member_labels_local(member, &node.member_node_ids);
    merge_graph_requirements_local(&node.requirements, member.requirements);
    mark_requirement_for_op_local(member.canonical_op, &node.requirements);
    if (member.sequence >= 0) {
      node.sequence = std::max(node.sequence, member.sequence);
    }
  }
  if (kind == MpkGraphNodeKind::FusedPreproc) {
    node.requirements.preproc = true;
  } else if (kind == MpkGraphNodeKind::FusedBoxDecode) {
    node.requirements.boxdecode = true;
  }
  mark_requirement_for_op_local(canonical_op, &node.requirements);
  return node;
}

bool is_preproc_absorbable_op_local(const std::string& canonical_op) {
  return canonical_op == "preproc" || canonical_op == "quant" || canonical_op == "tess" ||
         canonical_op == "quanttess" || canonical_op == "cast";
}

bool is_boxdecode_absorbable_op_local(const std::string& canonical_op) {
  return canonical_op == "detess" || canonical_op == "dequantize" ||
         canonical_op == "detessdequant" || canonical_op == "cast";
}

void accumulate_transitive_requirements_local(const WorkingGraph& graph, const std::string& node_id,
                                              MpkGraphFusionRequirements* requirements,
                                              std::unordered_set<std::string>* visited) {
  if (!requirements || !visited) {
    return;
  }
  if (!visited->insert(node_id).second) {
    return;
  }
  const auto node_index = working_node_index_local(graph, node_id);
  if (!node_index.has_value()) {
    return;
  }
  const auto& node = graph.nodes[*node_index].node;
  merge_graph_requirements_local(requirements, node.requirements);
  mark_requirement_for_op_local(node.canonical_op, requirements);
  for (const auto edge_index : working_incoming_edges_local(graph, node.node_id)) {
    if (edge_index >= graph.edges.size() || graph.edges[edge_index].removed) {
      continue;
    }
    accumulate_transitive_requirements_local(graph, graph.edges[edge_index].edge.src_node_id,
                                             requirements, visited);
  }
}

bool fuse_linear_pair_pattern_local(WorkingGraph* graph, const std::string& lhs_op,
                                    const std::string& rhs_op, const MpkGraphNodeKind fused_kind,
                                    const std::string& fused_label,
                                    const std::string& fused_canonical_op) {
  if (!graph) {
    return false;
  }
  bool progress = false;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
      if (graph->nodes[i].removed || graph->nodes[i].node.canonical_op != lhs_op) {
        continue;
      }
      const auto outgoing = working_outgoing_edges_local(*graph, graph->nodes[i].node.node_id);
      if (outgoing.size() != 1U) {
        continue;
      }
      const auto& edge = graph->edges[outgoing.front()].edge;
      const auto dst_index = working_node_index_local(*graph, edge.dst_node_id);
      if (!dst_index.has_value() || graph->nodes[*dst_index].removed ||
          graph->nodes[*dst_index].node.canonical_op != rhs_op) {
        continue;
      }
      const auto incoming_dst = working_incoming_edges_local(*graph, edge.dst_node_id);
      if (incoming_dst.size() != 1U) {
        continue;
      }

      const auto incoming_lhs = working_incoming_edges_local(*graph, graph->nodes[i].node.node_id);
      const auto outgoing_rhs =
          working_outgoing_edges_local(*graph, graph->nodes[*dst_index].node.node_id);
      MpkGraphNode fused_node = make_fused_graph_node_local(
          graph, fused_kind, fused_label, fused_canonical_op, {i, *dst_index}, /*branch_count=*/0);
      const std::string fused_node_id = fused_node.node_id;
      graph->nodes.push_back(WorkingGraphNode{std::move(fused_node), false});
      for (const auto incoming_edge_index : incoming_lhs) {
        if (incoming_edge_index >= graph->edges.size() ||
            graph->edges[incoming_edge_index].removed) {
          continue;
        }
        const auto& incoming_edge = graph->edges[incoming_edge_index].edge;
        add_working_edge_local(graph, incoming_edge.src_node_id, fused_node_id,
                               incoming_edge.tensor_name, MpkGraphEdgeKind::FusedRoute);
      }
      for (const auto outgoing_edge_index : outgoing_rhs) {
        if (outgoing_edge_index >= graph->edges.size() ||
            graph->edges[outgoing_edge_index].removed) {
          continue;
        }
        const auto& outgoing_edge = graph->edges[outgoing_edge_index].edge;
        add_working_edge_local(graph, fused_node_id, outgoing_edge.dst_node_id,
                               outgoing_edge.tensor_name, MpkGraphEdgeKind::FusedRoute);
      }

      mark_incident_edges_removed_local(graph, graph->nodes[i].node.node_id);
      mark_incident_edges_removed_local(graph, graph->nodes[*dst_index].node.node_id);
      graph->nodes[i].removed = true;
      graph->nodes[*dst_index].removed = true;
      progress = true;
      changed = true;
      break;
    }
  }
  return progress;
}

bool fold_post_mla_slice_views_local(WorkingGraph* graph) {
  if (!graph) {
    return false;
  }
  bool progress = false;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
      if (graph->nodes[i].removed || graph->nodes[i].node.canonical_op != "slice") {
        continue;
      }
      const auto incoming = working_incoming_edges_local(*graph, graph->nodes[i].node.node_id);
      if (incoming.size() != 1U) {
        continue;
      }
      const auto outgoing = working_outgoing_edges_local(*graph, graph->nodes[i].node.node_id);
      if (outgoing.empty()) {
        continue;
      }

      const auto& producer_edge = graph->edges[incoming.front()].edge;
      const auto producer_index = working_node_index_local(*graph, producer_edge.src_node_id);
      if (!producer_index.has_value() || graph->nodes[*producer_index].removed) {
        continue;
      }
      auto& producer_node = graph->nodes[*producer_index].node;
      if (producer_node.canonical_op != "mla") {
        continue;
      }

      std::string folded_output_name = !graph->nodes[i].node.output_tensor_names.empty()
                                           ? graph->nodes[i].node.output_tensor_names.front()
                                           : std::string();
      if (folded_output_name.empty()) {
        folded_output_name = graph->edges[outgoing.front()].edge.tensor_name;
      }
      replace_working_output_tensor_name_local(&producer_node, producer_edge.tensor_name,
                                               folded_output_name);

      for (const auto edge_index : outgoing) {
        if (edge_index >= graph->edges.size() || graph->edges[edge_index].removed) {
          continue;
        }
        const auto& edge = graph->edges[edge_index].edge;
        add_working_edge_local(graph, producer_node.node_id, edge.dst_node_id, edge.tensor_name,
                               MpkGraphEdgeKind::FusedRoute);
      }

      mark_incident_edges_removed_local(graph, graph->nodes[i].node.node_id);
      graph->nodes[i].removed = true;
      progress = true;
      changed = true;
      break;
    }
  }
  return progress;
}

bool fold_pre_mla_pack_views_local(WorkingGraph* graph) {
  if (!graph) {
    return false;
  }
  bool progress = false;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
      if (graph->nodes[i].removed || graph->nodes[i].node.canonical_op != "pack") {
        continue;
      }
      const auto incoming = working_incoming_edges_local(*graph, graph->nodes[i].node.node_id);
      const auto outgoing = working_outgoing_edges_local(*graph, graph->nodes[i].node.node_id);
      if (incoming.empty() || outgoing.size() != 1U) {
        continue;
      }

      const auto& mla_edge = graph->edges[outgoing.front()].edge;
      const auto mla_index = working_node_index_local(*graph, mla_edge.dst_node_id);
      if (!mla_index.has_value() || graph->nodes[*mla_index].removed) {
        continue;
      }
      auto& mla_node = graph->nodes[*mla_index].node;
      if (mla_node.canonical_op != "mla") {
        continue;
      }

      mla_node.input_tensor_names.clear();
      mla_node.input_tensor_names.reserve(incoming.size());
      for (const auto incoming_edge_index : incoming) {
        if (incoming_edge_index >= graph->edges.size() ||
            graph->edges[incoming_edge_index].removed) {
          continue;
        }
        const auto& incoming_edge = graph->edges[incoming_edge_index].edge;
        mla_node.input_tensor_names.push_back(incoming_edge.tensor_name);
        add_working_edge_local(graph, incoming_edge.src_node_id, mla_node.node_id,
                               incoming_edge.tensor_name, MpkGraphEdgeKind::FusedRoute);
      }

      mark_incident_edges_removed_local(graph, graph->nodes[i].node.node_id);
      graph->nodes[i].removed = true;
      progress = true;
      changed = true;
      break;
    }
  }
  return progress;
}

bool fuse_preproc_paths_local(WorkingGraph* graph) {
  if (!graph) {
    return false;
  }
  bool progress = false;
  for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
    if (graph->nodes[i].removed || graph->nodes[i].node.kind != MpkGraphNodeKind::IngressTensor) {
      continue;
    }
    const auto ingress_outgoing =
        working_outgoing_edges_local(*graph, graph->nodes[i].node.node_id);
    if (ingress_outgoing.size() != 1U) {
      continue;
    }

    std::vector<std::size_t> path_nodes;
    const auto& first_edge = graph->edges[ingress_outgoing.front()].edge;
    std::string boundary_target = first_edge.dst_node_id;
    std::string boundary_tensor = first_edge.tensor_name;
    while (true) {
      const auto current_index = working_node_index_local(*graph, boundary_target);
      if (!current_index.has_value() || graph->nodes[*current_index].removed) {
        break;
      }
      const auto& current_node = graph->nodes[*current_index].node;
      if (!is_preproc_absorbable_op_local(current_node.canonical_op)) {
        break;
      }
      if (working_incoming_edges_local(*graph, current_node.node_id).size() != 1U) {
        break;
      }
      path_nodes.push_back(*current_index);
      const auto current_outgoing = working_outgoing_edges_local(*graph, current_node.node_id);
      if (current_outgoing.size() != 1U) {
        boundary_target.clear();
        boundary_tensor.clear();
        break;
      }
      const auto& next_edge = graph->edges[current_outgoing.front()].edge;
      boundary_target = next_edge.dst_node_id;
      boundary_tensor = next_edge.tensor_name;
      const auto next_index = working_node_index_local(*graph, boundary_target);
      if (!next_index.has_value() || graph->nodes[*next_index].removed) {
        break;
      }
      if (!is_preproc_absorbable_op_local(graph->nodes[*next_index].node.canonical_op) ||
          working_incoming_edges_local(*graph, boundary_target).size() > 1U) {
        break;
      }
    }

    if (path_nodes.empty()) {
      continue;
    }

    MpkGraphNode fused_node =
        make_fused_graph_node_local(graph, MpkGraphNodeKind::FusedPreproc, "Preproc", "preproc",
                                    path_nodes, /*branch_count=*/1);
    std::string fused_node_id = fused_node.node_id;
    graph->nodes.push_back(WorkingGraphNode{std::move(fused_node), false});
    add_working_edge_local(graph, graph->nodes[i].node.node_id, fused_node_id,
                           first_edge.tensor_name, MpkGraphEdgeKind::FusedRoute);
    if (!boundary_target.empty()) {
      add_working_edge_local(graph, fused_node_id, boundary_target, boundary_tensor,
                             MpkGraphEdgeKind::FusedRoute);
    }
    for (const auto path_index : path_nodes) {
      if (path_index >= graph->nodes.size() || graph->nodes[path_index].removed) {
        continue;
      }
      mark_incident_edges_removed_local(graph, graph->nodes[path_index].node.node_id);
      graph->nodes[path_index].removed = true;
    }
    progress = true;
  }
  return progress;
}

bool fuse_boxdecode_paths_local(WorkingGraph* graph) {
  if (!graph) {
    return false;
  }
  bool progress = false;
  for (std::size_t sink_index = 0; sink_index < graph->nodes.size(); ++sink_index) {
    if (graph->nodes[sink_index].removed) {
      continue;
    }
    const auto incoming_sink =
        working_incoming_edges_local(*graph, graph->nodes[sink_index].node.node_id);
    if (incoming_sink.empty()) {
      continue;
    }
    const auto outgoing_sink =
        working_outgoing_edges_local(*graph, graph->nodes[sink_index].node.node_id);
    const auto& sink_node = graph->nodes[sink_index].node;
    if (sink_node.kind == MpkGraphNodeKind::FusedBoxDecode) {
      continue;
    }
    const bool sink_is_terminal = outgoing_sink.empty();
    const bool sink_self_absorbable = is_boxdecode_absorbable_op_local(sink_node.canonical_op);
    const bool sink_is_supported = sink_node.canonical_op == "pass_through" ||
                                   sink_node.canonical_op == "boxdecode" ||
                                   (sink_is_terminal && sink_self_absorbable);
    if (!sink_is_supported) {
      continue;
    }

    struct BranchInput {
      std::string src_node_id;
      std::string tensor_name;
    };

    std::vector<std::size_t> absorbed_indices;
    std::vector<BranchInput> branch_inputs;
    branch_inputs.reserve(incoming_sink.size());

    for (const auto incoming_edge_index : incoming_sink) {
      if (incoming_edge_index >= graph->edges.size() || graph->edges[incoming_edge_index].removed) {
        continue;
      }
      const auto& sink_input_edge = graph->edges[incoming_edge_index].edge;
      std::string downstream_node_id = sink_node.node_id;
      std::string branch_source_id = sink_input_edge.src_node_id;
      std::string branch_tensor_name = sink_input_edge.tensor_name;
      std::vector<std::size_t> branch_absorbed;

      while (true) {
        const auto current_index = working_node_index_local(*graph, branch_source_id);
        if (!current_index.has_value() || graph->nodes[*current_index].removed) {
          break;
        }
        const auto& current_node = graph->nodes[*current_index].node;
        if (!is_boxdecode_absorbable_op_local(current_node.canonical_op)) {
          break;
        }
        const auto current_outgoing = working_outgoing_edges_local(*graph, current_node.node_id);
        if (current_outgoing.size() != 1U ||
            graph->edges[current_outgoing.front()].edge.dst_node_id != downstream_node_id) {
          break;
        }
        branch_absorbed.push_back(*current_index);
        const auto current_incoming = working_incoming_edges_local(*graph, current_node.node_id);
        if (current_incoming.size() != 1U) {
          branch_source_id.clear();
          break;
        }
        const auto& predecessor_edge = graph->edges[current_incoming.front()].edge;
        branch_source_id = predecessor_edge.src_node_id;
        branch_tensor_name = predecessor_edge.tensor_name;
        downstream_node_id = current_node.node_id;
        const auto predecessor_index = working_node_index_local(*graph, branch_source_id);
        if (!predecessor_index.has_value() || graph->nodes[*predecessor_index].removed) {
          break;
        }
        if (!is_boxdecode_absorbable_op_local(graph->nodes[*predecessor_index].node.canonical_op) ||
            working_outgoing_edges_local(*graph, branch_source_id).size() != 1U) {
          break;
        }
      }

      absorbed_indices.insert(absorbed_indices.end(), branch_absorbed.begin(),
                              branch_absorbed.end());
      if (!branch_source_id.empty()) {
        branch_inputs.push_back(BranchInput{branch_source_id, branch_tensor_name});
      } else {
        branch_inputs.push_back(
            BranchInput{sink_input_edge.src_node_id, sink_input_edge.tensor_name});
      }
    }

    if (branch_inputs.empty()) {
      continue;
    }
    if (absorbed_indices.empty() && sink_node.canonical_op != "pass_through" &&
        sink_node.canonical_op != "boxdecode" && !(sink_is_terminal && sink_self_absorbable)) {
      continue;
    }

    std::vector<std::size_t> fused_members = absorbed_indices;
    fused_members.push_back(sink_index);
    MpkGraphNode fused_node = make_fused_graph_node_local(graph, MpkGraphNodeKind::FusedBoxDecode,
                                                          "BoxDecode", "boxdecode", fused_members,
                                                          static_cast<int>(branch_inputs.size()));
    {
      std::unordered_set<std::string> visited;
      for (const auto& branch_input : branch_inputs) {
        accumulate_transitive_requirements_local(*graph, branch_input.src_node_id,
                                                 &fused_node.requirements, &visited);
      }
      for (const auto absorbed_index : absorbed_indices) {
        if (absorbed_index < graph->nodes.size() && !graph->nodes[absorbed_index].removed) {
          accumulate_transitive_requirements_local(*graph,
                                                   graph->nodes[absorbed_index].node.node_id,
                                                   &fused_node.requirements, &visited);
        }
      }
      fused_node.requirements.boxdecode = true;
    }
    const std::string fused_node_id = fused_node.node_id;
    graph->nodes.push_back(WorkingGraphNode{std::move(fused_node), false});
    for (const auto& branch_input : branch_inputs) {
      add_working_edge_local(graph, branch_input.src_node_id, fused_node_id,
                             branch_input.tensor_name, MpkGraphEdgeKind::FusedRoute);
    }
    for (const auto outgoing_edge_index : outgoing_sink) {
      if (outgoing_edge_index >= graph->edges.size() || graph->edges[outgoing_edge_index].removed) {
        continue;
      }
      const auto& outgoing_edge = graph->edges[outgoing_edge_index].edge;
      add_working_edge_local(graph, fused_node_id, outgoing_edge.dst_node_id,
                             outgoing_edge.tensor_name, MpkGraphEdgeKind::FusedRoute);
    }

    for (const auto absorbed_index : absorbed_indices) {
      if (absorbed_index >= graph->nodes.size() || graph->nodes[absorbed_index].removed) {
        continue;
      }
      mark_incident_edges_removed_local(graph, graph->nodes[absorbed_index].node.node_id);
      graph->nodes[absorbed_index].removed = true;
    }
    mark_incident_edges_removed_local(graph, sink_node.node_id);
    graph->nodes[sink_index].removed = true;
    progress = true;
  }
  return progress;
}

WorkingGraph make_working_graph_local(const MpkGraph& graph) {
  WorkingGraph working;
  working.nodes.reserve(graph.nodes.size());
  for (const auto& node : graph.nodes) {
    working.nodes.push_back(WorkingGraphNode{node, false});
  }
  working.edges.reserve(graph.edges.size());
  for (const auto& edge : graph.edges) {
    working.edges.push_back(WorkingGraphEdge{edge, false});
  }
  working.next_synthetic_node = graph.nodes.size();
  working.next_synthetic_edge = graph.edges.size();
  return working;
}

void export_working_graph_local(const WorkingGraph& working, MpkGraph* graph) {
  if (!graph) {
    return;
  }
  graph->nodes.clear();
  graph->edges.clear();
  for (const auto& node : working.nodes) {
    if (!node.removed) {
      graph->nodes.push_back(node.node);
    }
  }
  for (const auto& edge : working.edges) {
    if (!edge.removed) {
      graph->edges.push_back(edge.edge);
    }
  }
}

} // namespace

void graph_fuser(MpkContract* contract) {
  if (!contract) {
    return;
  }
  WorkingGraph working = make_working_graph_local(contract->graph);
  (void)fold_pre_mla_pack_views_local(&working);
  (void)fold_post_mla_slice_views_local(&working);
  if (mpk_graph_fuse_quanttess_enabled()) {
    (void)fuse_linear_pair_pattern_local(
        &working, "quant", "tess", MpkGraphNodeKind::FusedQuantTess, "QuantTess", "quanttess");
  }
  if (mpk_graph_fuse_detessdequant_enabled()) {
    (void)fuse_linear_pair_pattern_local(&working, "detess", "dequantize",
                                         MpkGraphNodeKind::FusedDetessDequant, "DetessDequant",
                                         "detessdequant");
  }
  if (mpk_graph_fuse_preproc_enabled()) {
    (void)fuse_preproc_paths_local(&working);
  }
  if (mpk_graph_fuse_boxdecode_enabled()) {
    (void)fuse_boxdecode_paths_local(&working);
  }
  export_working_graph_local(working, &contract->graph);
}

void fill_graph_from_mpk(MpkContract* contract) {
  if (!contract) {
    return;
  }
  fill_graph_nodes_from_mpk_local(&contract->graph.raw_nodes, contract);
  fill_graph_nodes_from_mpk_local(&contract->graph.nodes, contract);
}

void graph_mpk_creation(MpkContract* contract) {
  if (!contract) {
    return;
  }

  MpkGraph graph;
  graph.mpk_json_path = contract->mpk_json_path;
  graph.model_name = contract->model_name;
  graph.model_path = contract->model_path;
  graph.nodes.reserve(contract->ingress_tensors.size() + contract->plugins.size());
  const MpkPluginIoContract* mla_unpack_stage = get_mla_unpack_stage_io_contract(*contract);
  const auto mla_unpack_plugin_index =
      mla_unpack_stage ? plugin_index_from_stage_ref_local(*contract, *mla_unpack_stage)
                       : std::nullopt;

  struct IngressRef {
    std::string node_id;
    int tensor_index = -1;
  };
  struct PluginOutputRef {
    std::string node_id;
    std::size_t plugin_index = static_cast<std::size_t>(-1);
    int tensor_index = -1;
  };

  std::unordered_map<std::string, std::vector<IngressRef>> ingress_by_name;
  ingress_by_name.reserve(contract->ingress_tensors.size());
  for (std::size_t i = 0; i < contract->ingress_tensors.size(); ++i) {
    const auto& ingress = contract->ingress_tensors[i];
    MpkGraphNode node;
    node.kind = MpkGraphNodeKind::IngressTensor;
    node.node_id = "ingress:" + std::to_string(i) + ":" + graph_id_token_local(ingress.name);
    node.label = ingress.name.empty() ? ("ingress_" + std::to_string(i)) : ingress.name;
    node.name = ingress.name;
    node.tensor_kind = ingress.kind;
    node.dtype = ingress.dtype;
    node.mpk_shape = ingress.mpk_shape;
    node.tensor_index = ingress.tensor_index;
    node.size_bytes = ingress.size_bytes;
    node.canonical_op = "ingress";
    graph.nodes.push_back(node);
    if (!ingress.name.empty()) {
      ingress_by_name[ingress.name].push_back(IngressRef{node.node_id, ingress.tensor_index});
    }
  }

  std::unordered_map<std::string, std::vector<PluginOutputRef>> outputs_by_name;
  for (std::size_t pi = 0; pi < contract->plugins.size(); ++pi) {
    if (mla_unpack_plugin_index.has_value() && pi == *mla_unpack_plugin_index) {
      continue;
    }
    const auto& plugin = contract->plugins[pi];
    MpkGraphNode node;
    node.kind = MpkGraphNodeKind::Plugin;
    node.node_id = "plugin:" + std::to_string(pi) + ":" +
                   graph_id_token_local(!plugin.name.empty() ? plugin.name : plugin.plugin_id);
    node.label =
        !plugin.name.empty()
            ? plugin.name
            : (!plugin.plugin_id.empty() ? plugin.plugin_id : ("plugin_" + std::to_string(pi)));
    node.name = plugin.name;
    node.plugin_id = plugin.plugin_id;
    node.processor = plugin.processor;
    node.kernel = plugin.kernel;
    node.canonical_op = canonical_mpk_graph_op_local(plugin.kernel, plugin.name, plugin.processor);
    node.kernel_contract = kernel_contract_template_local(node.kernel, node.canonical_op);
    mark_requirement_for_op_local(node.canonical_op, &node.requirements);
    node.plugin_index = pi;
    node.sequence = plugin.sequence;
    node.input_tensor_names.reserve(plugin.input_tensors.size());
    for (const auto& input : plugin.input_tensors) {
      node.input_tensor_names.push_back(input.name);
    }
    node.output_tensor_names.reserve(plugin.output_tensors.size());
    for (const auto& output : plugin.output_tensors) {
      node.output_tensor_names.push_back(output.name);
      if (!output.name.empty()) {
        outputs_by_name[output.name].push_back(
            PluginOutputRef{node.node_id, pi, output.tensor_index});
      }
    }
    if (node.canonical_op == "mla" && mla_unpack_stage) {
      for (const auto& unpack_output : mla_unpack_stage->output_tensors) {
        if (!unpack_output.name.empty()) {
          node.output_tensor_names.push_back(unpack_output.name);
          outputs_by_name[unpack_output.name].push_back(
              PluginOutputRef{node.node_id, pi, unpack_output.tensor_index});
        }
      }
    }
    graph.nodes.push_back(std::move(node));
  }

  std::size_t edge_counter = 0U;
  for (std::size_t pi = 0; pi < contract->plugins.size(); ++pi) {
    if (mla_unpack_plugin_index.has_value() && pi == *mla_unpack_plugin_index) {
      continue;
    }
    const auto& plugin = contract->plugins[pi];
    const std::string plugin_node_id =
        "plugin:" + std::to_string(pi) + ":" +
        graph_id_token_local(!plugin.name.empty() ? plugin.name : plugin.plugin_id);
    for (const auto& input : plugin.input_tensors) {
      if (input.name.empty()) {
        continue;
      }
      if (const auto ingress_it = ingress_by_name.find(input.name);
          ingress_it != ingress_by_name.end()) {
        for (const auto& ingress : ingress_it->second) {
          MpkGraphEdge edge;
          edge.edge_id = "edge:" + std::to_string(edge_counter++);
          edge.src_node_id = ingress.node_id;
          edge.dst_node_id = plugin_node_id;
          edge.tensor_name = input.name;
          edge.kind = MpkGraphEdgeKind::CandidateTensorMatch;
          edge.src_tensor_index = ingress.tensor_index;
          edge.dst_plugin_index = pi;
          edge.dst_tensor_index = input.tensor_index;
          graph.edges.push_back(std::move(edge));
        }
      }
      if (const auto producer_it = outputs_by_name.find(input.name);
          producer_it != outputs_by_name.end()) {
        for (const auto& producer : producer_it->second) {
          MpkGraphEdge edge;
          edge.edge_id = "edge:" + std::to_string(edge_counter++);
          edge.src_node_id = producer.node_id;
          edge.dst_node_id = plugin_node_id;
          edge.tensor_name = input.name;
          edge.kind = MpkGraphEdgeKind::CandidateTensorMatch;
          edge.src_plugin_index = producer.plugin_index;
          edge.dst_plugin_index = pi;
          edge.src_tensor_index = producer.tensor_index;
          edge.dst_tensor_index = input.tensor_index;
          graph.edges.push_back(std::move(edge));
        }
      }
    }
  }

  graph.raw_nodes = graph.nodes;
  graph.raw_edges = graph.edges;
  contract->graph = std::move(graph);
  graph_fuser(contract);
}

std::optional<MpkContract> load_mpk_contract_from_pack_root(const std::string& package_root,
                                                            std::string* error_message) {
  if (error_message) {
    error_message->clear();
  }
  const fs::path root(package_root);
  if (package_root.empty() || !fs::exists(root) || !fs::is_directory(root)) {
    if (error_message) {
      *error_message = "invalid package root";
    }
    return std::nullopt;
  }

  const auto mpk_path = find_mpk_contract_path(root);
  if (!mpk_path.has_value()) {
    if (error_message) {
      *error_message = "no *_mpk.json found";
    }
    return std::nullopt;
  }

  json root_json;
  if (!read_json_file_local(*mpk_path, &root_json)) {
    if (error_message) {
      *error_message = "failed to parse mpk json";
    }
    return std::nullopt;
  }
  if (!root_json.contains("plugins") || !root_json["plugins"].is_array()) {
    if (error_message) {
      *error_message = "mpk json missing plugins[]";
    }
    return std::nullopt;
  }

  MpkContract contract;
  contract.mpk_json_path = mpk_path->string();
  if (root_json.contains("name") && root_json["name"].is_string()) {
    contract.model_name = root_json["name"].get<std::string>();
  }
  if (root_json.contains("model_path") && root_json["model_path"].is_string()) {
    contract.model_path = root_json["model_path"].get<std::string>();
  }
  if (root_json.contains("input_nodes")) {
    contract.ingress_tensors =
        parse_tensor_nodes(root_json["input_nodes"], {}, {}, "", MpkShapeSemantics::Unknown);
  }

  const auto& plugins = root_json["plugins"];
  contract.plugins.reserve(plugins.size());
  for (const auto& plugin : plugins) {
    if (!plugin.is_object()) {
      continue;
    }
    MpkPluginIoContract stage;
    if (plugin.contains("name") && plugin["name"].is_string()) {
      stage.name = plugin["name"].get<std::string>();
    }
    if (plugin.contains("pluginId") && plugin["pluginId"].is_string()) {
      stage.plugin_id = plugin["pluginId"].get<std::string>();
    } else if (plugin.contains("plugin_id") && plugin["plugin_id"].is_string()) {
      stage.plugin_id = plugin["plugin_id"].get<std::string>();
    }
    if (plugin.contains("processor") && plugin["processor"].is_string()) {
      stage.processor = plugin["processor"].get<std::string>();
    }
    if (plugin.contains("sequence")) {
      if (const auto seq = read_int_local(plugin["sequence"]); seq.has_value()) {
        stage.sequence = *seq;
      }
    }
    if (plugin.contains("resources") && plugin["resources"].is_object() &&
        plugin["resources"].contains("executable") &&
        plugin["resources"]["executable"].is_string()) {
      stage.executable = plugin["resources"]["executable"].get<std::string>();
    }

    std::vector<std::vector<std::int64_t>> input_shapes;
    std::vector<std::vector<std::int64_t>> output_shapes;
    std::vector<std::string> input_dtypes;
    std::vector<std::string> output_dtypes;
    std::string fallback_input_dtype;
    std::string fallback_output_dtype;
    int config_actual_batch_size = 0;

    const json* params = nullptr;
    if (plugin.contains("config_params") && plugin["config_params"].is_object()) {
      const auto& cfg = plugin["config_params"];
      if (const auto kernel = read_string_alias(cfg, {"kernel"}); kernel.has_value()) {
        stage.kernel = *kernel;
      }
      for (const char* key : {"actual_batch_size", "batch_sz_model", "batch_size_model"}) {
        if (cfg.contains(key)) {
          if (const auto batch_model = read_int_local(cfg.at(key)); batch_model.has_value()) {
            config_actual_batch_size = *batch_model;
            break;
          }
        }
      }
      if (cfg.contains("params") && cfg["params"].is_object()) {
        params = &cfg["params"];
      } else {
        params = &cfg;
      }
    }
    if (params && params->is_object()) {
      for (const char* key : {"desired_batch_size", "batch_size"}) {
        if (params->contains(key)) {
          if (const auto batch = read_int_local(params->at(key)); batch.has_value()) {
            stage.batch_size = *batch;
            break;
          }
        }
      }
      for (const char* key : {"actual_batch_size", "batch_sz_model", "batch_size_model"}) {
        if (params->contains(key)) {
          if (const auto batch_model = read_int_local(params->at(key)); batch_model.has_value()) {
            if (stage.batch_sz_model <= 0) {
              stage.batch_sz_model = *batch_model;
            }
            break;
          }
        }
      }
      input_shapes = read_shape_alias(*params, {"input_shapes", "in_shapes", "input_shape"});
      output_shapes = read_shape_alias(
          *params, {"output_shapes", "tensor_shapes", "out_shapes", "output_shape"});
      stage.slice_shape = read_shape_vector_alias(*params, {"slice_shape", "slice_shapes"});
      stage.slice_begin = read_shape_vector_alias(*params, {"begin", "slice_begin"});
      stage.slice_end = read_shape_vector_alias(*params, {"end", "slice_end"});
      stage.frame_shape = read_shape_vector_alias(*params, {"frame_shape", "frame_shapes"});
      stage.frame_type = read_string_alias(*params, {"frame_type"}).value_or("");
      stage.round_off = read_string_alias(*params, {"round_off", "rounding"}).value_or("");
      stage.has_canonical_processcvu_contract = true;
      if (!output_shapes.empty()) {
        stage.out_shape_raw = output_shapes.front();
      }
      if (stage.slice_shape.empty() &&
          canonical_mpk_graph_op_local(stage.kernel, stage.name, stage.processor) == "slice" &&
          !stage.out_shape_raw.empty()) {
        stage.slice_shape = stage.out_shape_raw;
      }
      if (const auto align =
              read_bool_local(params->contains("align_c16") ? params->at("align_c16") : json{});
          align.has_value()) {
        stage.has_align_c16 = true;
        stage.align_c16 = *align;
      }
      if (const auto cblock =
              read_bool_local(params->contains("cblock") ? params->at("cblock") : json{});
          cblock.has_value()) {
        stage.has_cblock = true;
        stage.cblock = *cblock;
      }
      input_dtypes = read_string_alias_values(
          *params, {"input_types", "input_dtype", "in_dtype", "input_data_type"});
      output_dtypes =
          read_string_alias_values(*params, {"tensor_types", "output_types", "output_dtype",
                                             "out_dtype", "output_data_type", "data_type"});
      fallback_input_dtype =
          read_string_alias(*params, {"in_dtype", "input_dtype", "input_data_type", "frame_type"})
              .value_or("");
      fallback_output_dtype =
          read_string_alias(
              *params, {"out_dtype", "output_dtype", "output_data_type", "data_type", "frame_type"})
              .value_or("");
      if (!fallback_input_dtype.empty()) {
        stage.canonical_input_dtype = normalize_dtype_local(fallback_input_dtype);
      }
      if (!fallback_output_dtype.empty()) {
        stage.canonical_output_dtype = normalize_dtype_local(fallback_output_dtype);
      }
      if (auto quant = parse_quant_from_params(*params); quant.has_value()) {
        stage.quant = std::move(quant);
      }
    }

    if (is_placeholder_kernel_token(stage.kernel)) {
      if (const std::string inferred_kernel = infer_kernel_from_stage_metadata(stage);
          !inferred_kernel.empty()) {
        stage.kernel = inferred_kernel;
      }
    }

    if (plugin.contains("input_nodes")) {
      const MpkShapeSemantics input_shape_semantics =
          classify_mpk_tensor_shape_semantics_local(stage, true);
      stage.input_tensors = parse_tensor_nodes(plugin["input_nodes"], input_shapes, input_dtypes,
                                               fallback_input_dtype, input_shape_semantics);
    }
    if (plugin.contains("output_nodes")) {
      const MpkShapeSemantics output_shape_semantics =
          classify_mpk_tensor_shape_semantics_local(stage, false);
      stage.output_tensors =
          parse_tensor_nodes(plugin["output_nodes"], output_shapes, output_dtypes,
                             fallback_output_dtype, output_shape_semantics);
    }
    if (stage.batch_sz_model <= 0 && config_actual_batch_size > 1) {
      const std::string kernel_token =
          canonical_token_local(!stage.kernel.empty() ? stage.kernel : stage.name);
      if (kernel_token.find("slice") != std::string::npos ||
          kernel_token.find("batchflatten") != std::string::npos) {
        stage.batch_sz_model = config_actual_batch_size;
      }
    }
    normalize_batched_view_stage_outputs_local(&stage);
    if (stage.canonical_input_dtype.empty() && !stage.input_tensors.empty()) {
      stage.canonical_input_dtype = normalize_dtype_local(stage.input_tensors.front().dtype);
    }
    if (stage.canonical_output_dtype.empty() && !stage.output_tensors.empty()) {
      stage.canonical_output_dtype = normalize_dtype_local(stage.output_tensors.front().dtype);
    }
    if (!stage.has_canonical_processcvu_contract) {
      stage.has_canonical_processcvu_contract = !stage.input_tensors.empty() ||
                                                !stage.output_tensors.empty() ||
                                                !stage.slice_shape.empty();
    }
    contract.plugins.push_back(std::move(stage));
  }

  try {
    graph_mpk_creation(&contract);
  } catch (const std::exception& e) {
    if (error_message) {
      *error_message = e.what();
    }
    return std::nullopt;
  }
  std::string resolve_error;
  if (!resolve_contract_edges_strict(&contract, &resolve_error)) {
    if (error_message) {
      *error_message =
          resolve_error.empty() ? std::string("failed to resolve strict mpk edges") : resolve_error;
    }
    return std::nullopt;
  }
  derive_mla_output_quant_from_downstream(&contract);
  derive_logical_output_contracts(&contract);
  derive_logical_input_contracts(&contract);
  fill_graph_from_mpk(&contract);
  if (mpk_graph_dump_enabled() || mpk_graph_exit_after_dump_enabled()) {
    const fs::path graph_output_path = mpk_graph_output_path_local(root, contract);
    std::string graph_dump_error;
    if (!write_mpk_graph_dump_local(contract.graph, graph_output_path, &graph_dump_error)) {
      if (error_message) {
        *error_message = graph_dump_error.empty() ? std::string("failed to write mpk graph dump")
                                                  : graph_dump_error;
      }
      return std::nullopt;
    }
    std::fprintf(stderr,
                 "[mpk-graph] package_root=%s mpk_json_path=%s output=%s model_name=\"%s\" "
                 "ingress_nodes=%zu plugin_nodes=%zu edges=%zu\n",
                 package_root.c_str(), contract.mpk_json_path.c_str(),
                 graph_output_path.string().c_str(), contract.model_name.c_str(),
                 contract.ingress_tensors.size(), contract.plugins.size(),
                 contract.graph.edges.size());
    if (mpk_graph_exit_after_dump_enabled()) {
      std::fprintf(stderr, "[mpk-graph] exit_after_dump=1\n");
      std::fflush(stderr);
      std::exit(0);
    }
  }
  if (mpk_contract_compare_enabled()) {
    dump_mpk_contract_compare_local(contract, root_json);
  }
  return contract;
}

const MpkPluginIoContract* get_stage_io_contract(const MpkContract& contract,
                                                 const std::string& plugin_name_or_id) {
  const auto idx = find_plugin_index_by_name_or_id(contract, plugin_name_or_id);
  if (!idx.has_value() || *idx >= contract.plugins.size()) {
    return nullptr;
  }
  return &contract.plugins[*idx];
}

const MpkPluginIoContract* get_mla_stage_io_contract(const MpkContract& contract) {
  const MpkPluginIoContract* mla_stage = nullptr;
  for (const auto& plugin : contract.plugins) {
    const bool by_processor = lower_copy_local(plugin.processor) == "mla";
    const bool by_kernel = canonical_token_local(plugin.kernel) == "mla";
    if (!(by_processor || by_kernel)) {
      continue;
    }
    if (mla_stage != nullptr) {
      return nullptr;
    }
    mla_stage = &plugin;
  }
  return mla_stage;
}

const MpkPluginIoContract* get_mla_unpack_stage_io_contract(const MpkContract& contract) {
  const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
  if (!mla) {
    return nullptr;
  }

  std::optional<std::size_t> mla_idx;
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == mla) {
      mla_idx = i;
      break;
    }
  }
  if (!mla_idx.has_value()) {
    return nullptr;
  }

  const auto ordered = plugins_in_order_internal(contract);
  std::unordered_map<std::size_t, std::size_t> rank_by_index;
  rank_by_index.reserve(ordered.size());
  for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
    rank_by_index.emplace(ordered[pos], pos);
  }

  const auto mla_rank_it = rank_by_index.find(*mla_idx);
  if (mla_rank_it == rank_by_index.end()) {
    return nullptr;
  }
  const std::size_t mla_rank = mla_rank_it->second;
  std::unordered_set<std::string> mla_output_names;
  mla_output_names.reserve(mla->output_tensors.size() + 1U);
  for (const auto& output : mla->output_tensors) {
    if (!output.name.empty()) {
      mla_output_names.insert(output.name);
    }
  }
  if (!mla->name.empty()) {
    mla_output_names.insert(mla->name);
  }

  const MpkPluginIoContract* unpack = nullptr;
  std::size_t unpack_rank = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (i == *mla_idx) {
      continue;
    }
    const auto rank_it = rank_by_index.find(i);
    if (rank_it == rank_by_index.end() || rank_it->second <= mla_rank) {
      continue;
    }
    const std::string kernel_source =
        !contract.plugins[i].kernel.empty() ? contract.plugins[i].kernel : contract.plugins[i].name;
    const std::string kernel = canonical_token_local(kernel_source);
    if (kernel.find("unpack") == std::string::npos) {
      continue;
    }
    bool consumes_mla = false;
    for (const auto& edge : contract.edges) {
      if (edge.dst_plugin_index == i && edge.src_plugin_index == *mla_idx) {
        consumes_mla = true;
        break;
      }
    }
    if (!consumes_mla) {
      for (const auto& input : contract.plugins[i].input_tensors) {
        if (!input.name.empty() && mla_output_names.count(input.name) > 0U) {
          consumes_mla = true;
          break;
        }
      }
    }
    if (!consumes_mla) {
      continue;
    }
    if (!unpack || rank_it->second < unpack_rank) {
      unpack = &contract.plugins[i];
      unpack_rank = rank_it->second;
    }
  }
  return unpack;
}

bool mla_consumer_keeps_distinct_physical_inputs(const MpkContract& contract) {
  const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
  if (!mla || mla->input_tensors.size() <= 1U) {
    return false;
  }
  const auto mla_idx_opt = plugin_index_from_stage_ref_local(contract, *mla);
  if (!mla_idx_opt.has_value()) {
    return false;
  }
  const std::size_t mla_idx = *mla_idx_opt;
  for (const auto& input_tensor : mla->input_tensors) {
    if (input_tensor.name.empty()) {
      continue;
    }
    int producer_output_index = -1;
    const MpkPluginIoContract* producer = nearest_producer_stage_for_tensor_name_local(
        contract, mla_idx, input_tensor.name, &producer_output_index);
    if (producer != nullptr && canonical_mpk_graph_op_local(producer->kernel, producer->name,
                                                            producer->processor) == "pack") {
      // An explicit pack producer means the .elf is the monolithic-IFM variant
      // and the existing collapse paths are correct.
      return false;
    }
  }
  // Validator log: surface the topology decision so DevKit / CI runs can
  // confirm the heuristic is firing for the four known failing variants and
  // not accidentally for monolithic models. Mirrors the existing
  // SIMA_MPK_CONTRACT_DEBUG-gated logs throughout this file.
  if (mpk_contract_debug_enabled()) {
    std::fprintf(
        stderr,
        "[mpk-contract] consumer_keeps_distinct_physical_inputs=true mla_stage=%s n_inputs=%zu\n",
        mla->name.empty() ? "<unnamed>" : mla->name.c_str(), mla->input_tensors.size());
  }
  return true;
}

std::optional<std::size_t> plugin_index_from_ptr_local(const MpkContract& contract,
                                                       const MpkPluginIoContract* stage) {
  if (!stage) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == stage) {
      return i;
    }
  }
  return std::nullopt;
}

const MpkContractEdge* earliest_edge_from_local(const MpkContract& contract,
                                                std::size_t src_plugin_index,
                                                int src_output_index) {
  const MpkContractEdge* best = nullptr;
  std::size_t best_order = std::numeric_limits<std::size_t>::max();
  for (const auto& edge : contract.edges) {
    const bool output_matches = edge.src_output_index == src_output_index ||
                                (src_output_index == 0 && edge.src_output_index < 0);
    if (edge.src_plugin_index != src_plugin_index || !output_matches) {
      continue;
    }
    if (edge.dst_plugin_index >= contract.plugins.size()) {
      continue;
    }
    const std::size_t order =
        plugin_order_key(contract.plugins[edge.dst_plugin_index], edge.dst_plugin_index);
    if (!best || order < best_order) {
      best = &edge;
      best_order = order;
    }
  }
  return best;
}

const MpkTensorContract* pick_stage_output_for_input_local(const MpkPluginIoContract& stage,
                                                           int input_index) {
  if (stage.output_tensors.empty()) {
    return nullptr;
  }
  if (stage.output_tensors.size() == 1U) {
    return &stage.output_tensors.front();
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.output_tensors.size()) {
    return &stage.output_tensors[static_cast<std::size_t>(input_index)];
  }
  return &stage.output_tensors.front();
}

std::size_t pick_stage_output_index_for_input_local(const MpkPluginIoContract& stage,
                                                    int input_index) {
  if (stage.output_tensors.empty()) {
    return 0U;
  }
  if (stage.output_tensors.size() == 1U) {
    return 0U;
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.output_tensors.size()) {
    return static_cast<std::size_t>(input_index);
  }
  return 0U;
}

bool is_mla_boundary_view_kernel_local(const std::string& kernel) {
  const std::string token = canonical_token_local(kernel);
  if (token.empty()) {
    return false;
  }
  if (token.find("unpack") != std::string::npos || token.find("slice") != std::string::npos ||
      token.find("batchflatten") != std::string::npos) {
    return true;
  }
  return token.find("detess") != std::string::npos;
}

const MpkTensorContract* pick_stage_input_for_binding_local(const MpkPluginIoContract& stage,
                                                            int input_index) {
  if (stage.input_tensors.empty()) {
    return nullptr;
  }
  if (stage.input_tensors.size() == 1U) {
    return &stage.input_tensors.front();
  }
  if (input_index >= 0 && static_cast<std::size_t>(input_index) < stage.input_tensors.size()) {
    return &stage.input_tensors[static_cast<std::size_t>(input_index)];
  }
  return &stage.input_tensors.front();
}

std::vector<MpkTensorContract>
get_mla_boundary_physical_inputs_contract(const MpkContract& contract) {
  const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
  if (!mla) {
    return {};
  }
  if (mla->input_tensors.empty()) {
    return {};
  }
  if (mla->input_tensors.size() > 1U) {
    return mla->input_tensors;
  }

  const auto mla_idx = plugin_index_from_ptr_local(contract, mla);
  if (!mla_idx.has_value()) {
    return mla->input_tensors;
  }

  const auto& mla_input = mla->input_tensors.front();

  std::vector<const MpkContractEdge*> incoming;
  incoming.reserve(contract.edges.size());
  for (const auto& edge : contract.edges) {
    if (edge.dst_plugin_index == *mla_idx) {
      incoming.push_back(&edge);
    }
  }
  if (incoming.empty()) {
    return mla->input_tensors;
  }

  std::sort(incoming.begin(), incoming.end(),
            [](const MpkContractEdge* lhs, const MpkContractEdge* rhs) {
              if (lhs->dst_input_index != rhs->dst_input_index) {
                return lhs->dst_input_index < rhs->dst_input_index;
              }
              if (lhs->src_plugin_index != rhs->src_plugin_index) {
                return lhs->src_plugin_index < rhs->src_plugin_index;
              }
              return lhs->src_output_index < rhs->src_output_index;
            });

  auto resolve_upstream_output_parts =
      [&](std::size_t consumer_plugin_index) -> std::vector<MpkTensorContract> {
    std::vector<const MpkContractEdge*> upstream_edges;
    upstream_edges.reserve(contract.edges.size());
    for (const auto& edge : contract.edges) {
      if (edge.dst_plugin_index == consumer_plugin_index) {
        upstream_edges.push_back(&edge);
      }
    }
    if (upstream_edges.empty()) {
      return {};
    }
    std::sort(upstream_edges.begin(), upstream_edges.end(),
              [](const MpkContractEdge* lhs, const MpkContractEdge* rhs) {
                if (lhs->dst_input_index != rhs->dst_input_index) {
                  return lhs->dst_input_index < rhs->dst_input_index;
                }
                if (lhs->src_plugin_index != rhs->src_plugin_index) {
                  return lhs->src_plugin_index < rhs->src_plugin_index;
                }
                return lhs->src_output_index < rhs->src_output_index;
              });
    std::vector<MpkTensorContract> parts;
    parts.reserve(upstream_edges.size());
    for (const auto* upstream_edge : upstream_edges) {
      if (!upstream_edge || upstream_edge->src_plugin_index >= contract.plugins.size()) {
        return {};
      }
      const auto& upstream_producer = contract.plugins[upstream_edge->src_plugin_index];
      const MpkTensorContract* tensor = nullptr;
      if (upstream_edge->src_output_index >= 0 &&
          static_cast<std::size_t>(upstream_edge->src_output_index) <
              upstream_producer.output_tensors.size()) {
        tensor = &upstream_producer
                      .output_tensors[static_cast<std::size_t>(upstream_edge->src_output_index)];
      } else if (!upstream_producer.output_tensors.empty()) {
        tensor = &upstream_producer.output_tensors.front();
      }
      if (!tensor) {
        return {};
      }
      parts.push_back(*tensor);
    }
    return parts;
  };

  auto synthesize_joined_carrier =
      [&](const MpkPluginIoContract& producer,
          const std::vector<MpkTensorContract>& parts) -> std::vector<MpkTensorContract> {
    if (parts.empty()) {
      return mla->input_tensors;
    }
    std::uint64_t total_bytes = 0U;
    for (const auto& part : parts) {
      if (part.size_bytes == 0U || total_bytes > (std::numeric_limits<std::uint64_t>::max() -
                                                  static_cast<std::uint64_t>(part.size_bytes))) {
        return mla->input_tensors;
      }
      total_bytes += static_cast<std::uint64_t>(part.size_bytes);
    }
    if (total_bytes == 0U ||
        total_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      return mla->input_tensors;
    }

    MpkTensorContract carrier = mla_input;
    carrier.tensor_index = 0;
    carrier.physical_index = 0;
    carrier.source_physical_index = 0;
    carrier.byte_offset = 0;
    carrier.source_byte_offset = 0;
    // This helper publishes the physical carrier that will be bound at MLA ingress. Its byte span
    // must match the concrete packed carrier, not any stale semantic/logical size recorded on the
    // MLA input tensor contract. Preserve the MPK logical shape elsewhere; for the physical carrier
    // contract we use the resolved joined span unconditionally.
    carrier.size_bytes = static_cast<std::size_t>(total_bytes);
    carrier.shape_semantics = MpkShapeSemantics::PackedExtent;
    // The joined carrier is a physical MLA ingress span, not a semantic tensor. Keep the MPK
    // semantic/logical tensor shape if we already have it; otherwise inherit a real geometry
    // shape from the upstream logical boundary instead of fabricating a byte-shaped pseudo tensor.
    if (carrier.logical_shape.empty()) {
      if (!mla_input.logical_shape.empty()) {
        carrier.logical_shape = mla_input.logical_shape;
      } else if (is_geometry_shape_semantics_local(mla_input.shape_semantics) &&
                 !mla_input.mpk_shape.empty()) {
        carrier.logical_shape = mla_input.mpk_shape;
      } else {
        for (const auto& part : parts) {
          if (!part.logical_shape.empty()) {
            carrier.logical_shape = part.logical_shape;
            break;
          }
          if (is_geometry_shape_semantics_local(part.shape_semantics) && !part.mpk_shape.empty()) {
            carrier.logical_shape = part.mpk_shape;
            break;
          }
        }
      }
    }
    carrier.mpk_shape.clear();
    carrier.stride_bytes.clear();
    if (carrier.dtype.empty()) {
      carrier.dtype = "INT8";
    }
    if (carrier.logical_dtype.empty()) {
      carrier.logical_dtype = carrier.dtype;
    }
    if (carrier.segment_name.empty()) {
      carrier.segment_name = !carrier.name.empty() ? carrier.name : producer.name;
    }
    if (carrier.name.empty()) {
      carrier.name =
          !carrier.segment_name.empty() ? carrier.segment_name : std::string("mla_input_0");
    }
    return {std::move(carrier)};
  };

  if (incoming.size() == 1U) {
    const auto* edge = incoming.front();
    if (edge->src_plugin_index < contract.plugins.size()) {
      const auto& producer = contract.plugins[edge->src_plugin_index];
      const std::string producer_kernel =
          canonical_token_local(!producer.kernel.empty() ? producer.kernel : producer.name);
      if (producer_kernel.find("pack") != std::string::npos) {
        const auto upstream_parts = resolve_upstream_output_parts(edge->src_plugin_index);
        if (!upstream_parts.empty()) {
          return synthesize_joined_carrier(producer, upstream_parts);
        }
        if (!producer.input_tensors.empty()) {
          return synthesize_joined_carrier(producer, producer.input_tensors);
        }
      }
      if (edge->src_output_index >= 0 &&
          static_cast<std::size_t>(edge->src_output_index) < producer.output_tensors.size()) {
        return {producer.output_tensors[static_cast<std::size_t>(edge->src_output_index)]};
      }
      if (!producer.output_tensors.empty()) {
        return {producer.output_tensors.front()};
      }
    }
  }

  if (incoming.size() > 1U) {
    std::vector<MpkTensorContract> parts;
    parts.reserve(incoming.size());
    for (const auto* edge : incoming) {
      if (!edge || edge->src_plugin_index >= contract.plugins.size()) {
        return mla->input_tensors;
      }
      const auto& producer = contract.plugins[edge->src_plugin_index];
      const MpkTensorContract* tensor = nullptr;
      if (edge->src_output_index >= 0 &&
          static_cast<std::size_t>(edge->src_output_index) < producer.output_tensors.size()) {
        tensor = &producer.output_tensors[static_cast<std::size_t>(edge->src_output_index)];
      } else if (!producer.output_tensors.empty()) {
        tensor = &producer.output_tensors.front();
      }
      if (!tensor) {
        return mla->input_tensors;
      }
      parts.push_back(*tensor);
    }
    const auto& producer = contract.plugins[incoming.front()->src_plugin_index];
    return synthesize_joined_carrier(producer, parts);
  }

  if (mla_input.shape_semantics != MpkShapeSemantics::PackedExtent) {
    return mla->input_tensors;
  }

  return mla->input_tensors;
}

std::vector<MpkTensorContract>
get_mla_boundary_logical_inputs_contract(const MpkContract& contract) {
  const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
  if (!mla) {
    return {};
  }
  if (mla->input_tensors.empty()) {
    return {};
  }

  const auto mla_idx = plugin_index_from_ptr_local(contract, mla);
  if (!mla_idx.has_value()) {
    return mla->input_tensors;
  }

  std::vector<const MpkContractEdge*> incoming;
  incoming.reserve(contract.edges.size());
  for (const auto& edge : contract.edges) {
    if (edge.dst_plugin_index == *mla_idx) {
      incoming.push_back(&edge);
    }
  }
  if (incoming.empty()) {
    return mla->input_tensors;
  }

  std::sort(incoming.begin(), incoming.end(),
            [](const MpkContractEdge* lhs, const MpkContractEdge* rhs) {
              if (lhs->dst_input_index != rhs->dst_input_index) {
                return lhs->dst_input_index < rhs->dst_input_index;
              }
              if (lhs->src_plugin_index != rhs->src_plugin_index) {
                return lhs->src_plugin_index < rhs->src_plugin_index;
              }
              return lhs->src_output_index < rhs->src_output_index;
            });

  auto resolve_upstream_output_parts =
      [&](std::size_t consumer_plugin_index) -> std::vector<MpkTensorContract> {
    std::vector<const MpkContractEdge*> upstream_edges;
    upstream_edges.reserve(contract.edges.size());
    for (const auto& edge : contract.edges) {
      if (edge.dst_plugin_index == consumer_plugin_index) {
        upstream_edges.push_back(&edge);
      }
    }
    if (upstream_edges.empty()) {
      return {};
    }
    std::sort(upstream_edges.begin(), upstream_edges.end(),
              [](const MpkContractEdge* lhs, const MpkContractEdge* rhs) {
                if (lhs->dst_input_index != rhs->dst_input_index) {
                  return lhs->dst_input_index < rhs->dst_input_index;
                }
                if (lhs->src_plugin_index != rhs->src_plugin_index) {
                  return lhs->src_plugin_index < rhs->src_plugin_index;
                }
                return lhs->src_output_index < rhs->src_output_index;
              });
    std::vector<MpkTensorContract> parts;
    parts.reserve(upstream_edges.size());
    for (const auto* upstream_edge : upstream_edges) {
      if (!upstream_edge || upstream_edge->src_plugin_index >= contract.plugins.size()) {
        return {};
      }
      const auto& upstream_producer = contract.plugins[upstream_edge->src_plugin_index];
      const MpkTensorContract* tensor = nullptr;
      if (upstream_edge->src_output_index >= 0 &&
          static_cast<std::size_t>(upstream_edge->src_output_index) <
              upstream_producer.output_tensors.size()) {
        tensor = &upstream_producer
                      .output_tensors[static_cast<std::size_t>(upstream_edge->src_output_index)];
      } else if (!upstream_producer.output_tensors.empty()) {
        tensor = &upstream_producer.output_tensors.front();
      }
      if (!tensor) {
        return {};
      }
      parts.push_back(*tensor);
    }
    return parts;
  };

  if (incoming.size() == 1U) {
    const auto* edge = incoming.front();
    if (!edge || edge->src_plugin_index >= contract.plugins.size()) {
      return mla->input_tensors;
    }
    const auto& producer = contract.plugins[edge->src_plugin_index];
    const std::string producer_kernel =
        canonical_token_local(!producer.kernel.empty() ? producer.kernel : producer.name);
    if (producer_kernel.find("pack") != std::string::npos) {
      if (!producer.input_tensors.empty()) {
        return producer.input_tensors;
      }
      if (const auto upstream_parts = resolve_upstream_output_parts(edge->src_plugin_index);
          !upstream_parts.empty()) {
        return upstream_parts;
      }
    }
    if (edge->src_output_index >= 0 &&
        static_cast<std::size_t>(edge->src_output_index) < producer.output_tensors.size()) {
      return {producer.output_tensors[static_cast<std::size_t>(edge->src_output_index)]};
    }
    if (!producer.output_tensors.empty()) {
      return {producer.output_tensors.front()};
    }
    return mla->input_tensors;
  }

  std::vector<MpkTensorContract> logical_inputs;
  logical_inputs.reserve(incoming.size());
  for (const auto* edge : incoming) {
    if (!edge || edge->src_plugin_index >= contract.plugins.size()) {
      return mla->input_tensors;
    }
    const auto& producer = contract.plugins[edge->src_plugin_index];
    const MpkTensorContract* tensor = nullptr;
    if (edge->src_output_index >= 0 &&
        static_cast<std::size_t>(edge->src_output_index) < producer.output_tensors.size()) {
      tensor = &producer.output_tensors[static_cast<std::size_t>(edge->src_output_index)];
    } else if (!producer.output_tensors.empty()) {
      tensor = &producer.output_tensors.front();
    }
    if (!tensor) {
      return mla->input_tensors;
    }
    logical_inputs.push_back(*tensor);
  }
  return logical_inputs;
}

struct MlaBoundaryTensorView {
  const MpkPluginIoContract* boundary_stage = nullptr;
  const MpkTensorContract* unpack_tensor = nullptr;
  const MpkTensorContract* boundary_tensor = nullptr;
  std::size_t unpack_output_index = 0U;
  bool publish_transport_tensor = false;
  std::string boundary_name;
  std::string boundary_dtype;
};

std::uint64_t boundary_parent_span_bytes_local(const MlaBoundaryTensorView& view) {
  if (view.publish_transport_tensor && view.boundary_stage) {
    const std::uint64_t transport_span =
        expected_detess_packed_input_size_bytes_local(*view.boundary_stage, view.boundary_dtype);
    if (transport_span > 0U) {
      return transport_span;
    }
  }
  if (view.boundary_stage &&
      canonical_token_local(view.boundary_stage->kernel).find("slice") != std::string::npos &&
      view.unpack_tensor && view.unpack_tensor->size_bytes > 0U) {
    return static_cast<std::uint64_t>(view.unpack_tensor->size_bytes);
  }
  const MpkTensorContract* preferred = view.boundary_tensor;
  if (preferred && preferred->size_bytes > 0U) {
    return static_cast<std::uint64_t>(preferred->size_bytes);
  }
  if (view.unpack_tensor && view.unpack_tensor->size_bytes > 0U) {
    return static_cast<std::uint64_t>(view.unpack_tensor->size_bytes);
  }
  if (view.boundary_tensor && view.boundary_tensor->size_bytes > 0U) {
    return static_cast<std::uint64_t>(view.boundary_tensor->size_bytes);
  }
  return 0U;
}

std::vector<MlaBoundaryTensorView>
resolve_mla_boundary_tensor_views_local(const MpkContract& contract,
                                        const MpkPluginIoContract& mla) {
  std::vector<MlaBoundaryTensorView> out;
  const MpkPluginIoContract* unpack = get_mla_unpack_stage_io_contract(contract);
  const MpkPluginIoContract* boundary_root = unpack ? unpack : &mla;
  if (!boundary_root || boundary_root->output_tensors.empty()) {
    return out;
  }

  const auto root_idx = plugin_index_from_ptr_local(contract, boundary_root);
  if (!root_idx.has_value()) {
    return out;
  }

  out.reserve(boundary_root->output_tensors.size());
  for (std::size_t oi = 0; oi < boundary_root->output_tensors.size(); ++oi) {
    const auto& root_tensor = boundary_root->output_tensors[oi];

    MlaBoundaryTensorView view;
    view.boundary_stage = boundary_root;
    view.unpack_tensor = &root_tensor;
    view.boundary_tensor = &root_tensor;
    view.unpack_output_index = oi;
    view.boundary_name =
        !root_tensor.name.empty() ? root_tensor.name : ("ofm" + std::to_string(oi));
    if (!root_tensor.dtype.empty()) {
      view.boundary_dtype = root_tensor.dtype;
    } else if (oi < mla.output_tensors.size() && !mla.output_tensors[oi].dtype.empty()) {
      view.boundary_dtype = mla.output_tensors[oi].dtype;
    } else if (!mla.output_tensors.empty()) {
      view.boundary_dtype = mla.output_tensors.front().dtype;
    }

    const MpkPluginIoContract* current_stage = boundary_root;
    auto current_stage_index = root_idx;
    std::size_t current_output_index = oi;
    while (current_stage_index.has_value()) {
      const MpkContractEdge* boundary_edge = earliest_edge_from_local(
          contract, *current_stage_index, static_cast<int>(current_output_index));
      if (!boundary_edge || boundary_edge->dst_plugin_index >= contract.plugins.size()) {
        break;
      }
      const auto& boundary_consumer = contract.plugins[boundary_edge->dst_plugin_index];
      if (!is_mla_boundary_view_kernel_local(boundary_consumer.kernel)) {
        // Once we have walked through an explicit view stage (for example unpack -> slice),
        // the MLA boundary is already the view output that feeds the first real post op.
        // Do not replace that boundary with the post op input, or we lose the view stage's
        // slice offset/stride metadata and accidentally describe the packed MLA parent as
        // a dense concatenation.
        if (current_stage != boundary_root) {
          break;
        }
        if (const MpkTensorContract* candidate = pick_stage_input_for_binding_local(
                boundary_consumer, boundary_edge->dst_input_index);
            candidate != nullptr) {
          view.boundary_stage = &boundary_consumer;
          view.boundary_tensor = candidate;
          if (!candidate->name.empty()) {
            view.boundary_name = candidate->name;
          }
          if (!candidate->dtype.empty()) {
            view.boundary_dtype = candidate->dtype;
          }
        }
        break;
      }

      const std::string consumer_kernel = canonical_token_local(boundary_consumer.kernel);
      const bool consumer_is_slice = consumer_kernel.find("slice") != std::string::npos;
      const bool consumer_is_batch_flatten =
          consumer_kernel.find("batchflatten") != std::string::npos;
      const bool consumer_is_detess = consumer_kernel.find("detess") != std::string::npos;
      if (consumer_is_slice || consumer_is_batch_flatten) {
        view.boundary_stage = &boundary_consumer;
        if (const MpkTensorContract* candidate = pick_stage_output_for_input_local(
                boundary_consumer, boundary_edge->dst_input_index);
            candidate != nullptr) {
          view.boundary_tensor = candidate;
          if (!candidate->name.empty()) {
            view.boundary_name = candidate->name;
          }
          if (!candidate->dtype.empty()) {
            view.boundary_dtype = candidate->dtype;
          }
        }
        current_stage = &boundary_consumer;
        current_stage_index = plugin_index_from_ptr_local(contract, current_stage);
        current_output_index = pick_stage_output_index_for_input_local(
            boundary_consumer, boundary_edge->dst_input_index);
        continue;
      }

      if (consumer_is_detess) {
        // MLA boundary publication is graph-final: the external input boundary for any detess
        // family consumer must be the consumer input tensor, including fused detess+dequant/cast
        // variants. The packed transport span is still derived from the detess contract.
        view.publish_transport_tensor = true;
        view.boundary_stage = &boundary_consumer;
        if (const MpkTensorContract* candidate = pick_stage_input_for_binding_local(
                boundary_consumer, boundary_edge->dst_input_index);
            candidate != nullptr) {
          view.boundary_tensor = candidate;
          if (!candidate->dtype.empty()) {
            view.boundary_dtype = candidate->dtype;
          }
        }
      }
      break;
    }

    out.push_back(std::move(view));
  }
  return out;
}

const std::vector<MpkTensorContract>* get_mla_input_contract(const MpkContract& contract) {
  const MpkPluginIoContract* stage = get_mla_stage_io_contract(contract);
  if (!stage) {
    return nullptr;
  }
  return &stage->input_tensors;
}

const std::vector<MpkTensorContract>* get_mla_outputs_contract(const MpkContract& contract) {
  const MpkPluginIoContract* stage = get_mla_stage_io_contract(contract);
  if (!stage) {
    return nullptr;
  }
  return &stage->output_tensors;
}

std::vector<MpkTensorContract>
get_mla_boundary_physical_outputs_contract(const MpkContract& contract) {
  if (const MpkPluginIoContract* unpack = get_mla_unpack_stage_io_contract(contract);
      unpack && !unpack->input_tensors.empty()) {
    return unpack->input_tensors;
  }
  if (const auto* outputs = get_mla_outputs_contract(contract); outputs != nullptr) {
    return *outputs;
  }
  return {};
}

std::vector<MpkTensorContract> get_mla_published_outputs_contract(const MpkContract& contract) {
  auto build_outputs = [&](const bool publish_transport_boundary_views) {
    std::vector<MpkTensorContract> out;
    const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
    if (!mla) {
      return out;
    }
    const auto mla_idx = plugin_index_from_ptr_local(contract, mla);

    auto derive_geometry_size_bytes = [&](MpkTensorContract* tensor) {
      if (!tensor || tensor->size_bytes > 0U || tensor->mpk_shape.empty() ||
          !is_geometry_shape_semantics_local(tensor->shape_semantics)) {
        return;
      }
      std::size_t total = 1U;
      for (const auto dim : tensor->mpk_shape) {
        if (dim <= 0) {
          return;
        }
        const auto u_dim = static_cast<std::size_t>(dim);
        if (total > std::numeric_limits<std::size_t>::max() / u_dim) {
          return;
        }
        total *= u_dim;
      }
      const auto dtype_bytes = normalize_dtype_local(tensor->dtype);
      if (dtype_bytes == "FP32" || dtype_bytes == "INT32") {
        total *= 4U;
      } else if (dtype_bytes == "BF16" || dtype_bytes == "FP16" || dtype_bytes == "INT16" ||
                 dtype_bytes == "UINT16") {
        total *= 2U;
      }
      tensor->size_bytes = total;
    };

    auto canonical_tensor = [&](const MpkTensorContract& src, int tensor_index,
                                const std::string& dtype_override = std::string(),
                                const bool prefer_logical_shape = true) {
      MpkTensorContract tensor = src;
      tensor.tensor_index = tensor_index;
      if (!src.mpk_shape.empty()) {
        tensor.mpk_shape = src.mpk_shape;
        tensor.shape_semantics = src.shape_semantics;
      } else if (prefer_logical_shape && !src.logical_shape.empty()) {
        tensor.mpk_shape = src.logical_shape;
        tensor.shape_semantics = MpkShapeSemantics::Geometry;
      }
      if (!dtype_override.empty()) {
        tensor.dtype = dtype_override;
      } else if (!src.dtype.empty()) {
        tensor.dtype = src.dtype;
      } else if (prefer_logical_shape && !src.logical_dtype.empty()) {
        tensor.dtype = src.logical_dtype;
      }
      if (prefer_logical_shape && !tensor.mpk_shape.empty()) {
        tensor.size_bytes = 0U;
      }
      if (!prefer_logical_shape) {
        tensor.logical_shape.clear();
        tensor.logical_dtype.clear();
        tensor.logical_source_plugin.clear();
        tensor.logical_source_kernel.clear();
        tensor.logical_source_sequence = -1;
      }
      finalize_tensor_contract(&tensor);
      derive_geometry_size_bytes(&tensor);
      return tensor;
    };

    // Packed MLA exports fan out through an unpack stage. Lift boundary tensors from that
    // fanout and keep the output view anchored at the MLA boundary. For published outputs,
    // detess-bound heads must stay transport-shaped so downstream consumers bind the packed
    // MLA sub-buffer directly instead of a logical reinterpretation.
    const auto boundary_views = resolve_mla_boundary_tensor_views_local(contract, *mla);
    if (!boundary_views.empty()) {
      out.reserve(boundary_views.size());
      std::vector<std::uint64_t> unpack_raw_offsets(boundary_views.size(), 0U);
      std::uint64_t running_unpack_offset = 0U;
      for (std::size_t oi = 0; oi < boundary_views.size(); ++oi) {
        unpack_raw_offsets[oi] = running_unpack_offset;
        running_unpack_offset += boundary_parent_span_bytes_local(boundary_views[oi]);
      }

      for (std::size_t oi = 0; oi < boundary_views.size(); ++oi) {
        const auto& view = boundary_views[oi];
        const bool publish_transport_view =
            publish_transport_boundary_views && view.publish_transport_tensor;
        const MpkTensorContract* semantic_boundary_tensor =
            (view.boundary_stage != nullptr && view.boundary_stage != mla && view.boundary_tensor)
                ? view.boundary_tensor
                : (view.unpack_tensor ? view.unpack_tensor : view.boundary_tensor);
        const MpkTensorContract* selected_boundary_tensor =
            publish_transport_view
                ? (view.boundary_tensor ? view.boundary_tensor : semantic_boundary_tensor)
                : (semantic_boundary_tensor ? semantic_boundary_tensor : view.boundary_tensor);
        if (!selected_boundary_tensor) {
          continue;
        }
        const std::string boundary_context =
            (view.boundary_stage && !view.boundary_stage->name.empty())
                ? (view.boundary_stage->name + "[output " + std::to_string(oi) + "]")
                : ("MLA boundary output " + std::to_string(oi));
        const bool prefer_logical_boundary = !publish_transport_view;
        MpkTensorContract tensor = canonical_tensor(*selected_boundary_tensor, static_cast<int>(oi),
                                                    view.boundary_dtype, prefer_logical_boundary);
        if (prefer_logical_boundary && mla_idx.has_value() && oi < mla->output_tensors.size()) {
          const auto resolved = resolve_mla_graph_output_local(
              contract, *mla_idx, static_cast<int>(oi), mla->output_tensors[oi]);
          if (tensor.mpk_shape.empty() && !resolved.shape.empty()) {
            tensor.mpk_shape = resolved.shape;
            tensor.shape_semantics = MpkShapeSemantics::Geometry;
          }
          if (tensor.dtype.empty() && !resolved.semantic_dtype.empty()) {
            tensor.dtype = resolved.semantic_dtype;
          }
          if (tensor.size_bytes == 0U && resolved.size_bytes > 0U) {
            tensor.size_bytes = resolved.size_bytes;
          }
          derive_geometry_size_bytes(&tensor);
        }
        if (!prefer_logical_boundary) {
          const std::uint64_t transport_span_bytes = boundary_parent_span_bytes_local(view);
          if (transport_span_bytes == 0U) {
            throw std::runtime_error(
                "MLA boundary transport view is missing packed byte span for '" + boundary_context +
                "'");
          }
          tensor.size_bytes = static_cast<std::size_t>(transport_span_bytes);
          tensor.shape_semantics = MpkShapeSemantics::PackedExtent;
          tensor.stride_bytes.clear();
        }
        const bool packed_single_physical =
            mla->output_tensors.size() == 1U && boundary_views.size() > 1U;
        std::size_t physical_index = 0U;
        bool have_physical_index = false;
        if (tensor.physical_index >= 0 &&
            static_cast<std::size_t>(tensor.physical_index) < mla->output_tensors.size()) {
          physical_index = static_cast<std::size_t>(tensor.physical_index);
          have_physical_index = true;
        } else if (!tensor.segment_name.empty()) {
          for (std::size_t pi = 0; pi < mla->output_tensors.size(); ++pi) {
            const auto& physical_candidate = mla->output_tensors[pi];
            const std::string physical_segment = !physical_candidate.segment_name.empty()
                                                     ? physical_candidate.segment_name
                                                     : physical_candidate.name;
            if (!physical_segment.empty() && physical_segment == tensor.segment_name) {
              physical_index = pi;
              have_physical_index = true;
              break;
            }
          }
        }
        if (!have_physical_index) {
          physical_index = packed_single_physical
                               ? 0U
                               : std::min<std::size_t>(oi, mla->output_tensors.size() - 1U);
        }

        const auto published_name = !view.boundary_name.empty()
                                        ? view.boundary_name
                                        : ("mla_logical_output_" + std::to_string(oi));
        const auto unpack_dtype =
            !view.boundary_dtype.empty()
                ? view.boundary_dtype
                : (view.unpack_tensor ? view.unpack_tensor->dtype : tensor.dtype);
        const auto unpack_shape = (view.unpack_tensor && !view.unpack_tensor->mpk_shape.empty())
                                      ? view.unpack_tensor->mpk_shape
                                      : tensor.mpk_shape;
        const auto unpack_stride_bytes = contiguous_stride_bytes_local(unpack_shape, unpack_dtype);
        const std::string boundary_kernel = view.boundary_stage != nullptr
                                                ? canonical_token_local(view.boundary_stage->kernel)
                                                : std::string{};
        const bool boundary_is_slice = boundary_kernel.find("slice") != std::string::npos;
        const bool boundary_is_batch_flatten =
            boundary_kernel.find("batchflatten") != std::string::npos;
        const bool boundary_is_offset_view = boundary_is_slice || boundary_is_batch_flatten;
        std::optional<std::int64_t> explicit_boundary_byte_offset;
        if (view.boundary_tensor && view.boundary_tensor->byte_offset > 0) {
          explicit_boundary_byte_offset = view.boundary_tensor->byte_offset;
        } else if (view.unpack_tensor && view.unpack_tensor->byte_offset > 0 &&
                   packed_single_physical) {
          explicit_boundary_byte_offset = view.unpack_tensor->byte_offset;
        }
        std::int64_t boundary_byte_offset =
            explicit_boundary_byte_offset.has_value()
                ? *explicit_boundary_byte_offset
                : (packed_single_physical ? static_cast<std::int64_t>(unpack_raw_offsets[oi]) : 0);
        if (boundary_is_slice && !explicit_boundary_byte_offset.has_value()) {
          const auto slice_offset = projected_slice_begin_offset_bytes_local(
              view.boundary_stage->slice_begin, unpack_stride_bytes);
          if (slice_offset != std::numeric_limits<std::int64_t>::max() &&
              boundary_byte_offset <= std::numeric_limits<std::int64_t>::max() - slice_offset) {
            boundary_byte_offset += slice_offset;
          } else {
            boundary_byte_offset = std::numeric_limits<std::int64_t>::max();
          }
          tensor.stride_bytes = normalize_stride_rank_to_shape_local(
              unpack_stride_bytes, unpack_shape, tensor.mpk_shape);
        } else if (boundary_is_batch_flatten) {
          tensor.stride_bytes = flatten_batched_view_stride_local(unpack_stride_bytes, unpack_shape,
                                                                  tensor.mpk_shape);
        } else {
          tensor.stride_bytes = view.publish_transport_tensor
                                    ? std::vector<std::int64_t>{}
                                    : contiguous_stride_bytes_local(tensor.mpk_shape, tensor.dtype);
        }

        const MpkTensorContract* parent_tensor =
            packed_single_physical ? &mla->output_tensors[physical_index] : view.unpack_tensor;
        if (!parent_tensor && physical_index < mla->output_tensors.size()) {
          parent_tensor = &mla->output_tensors[physical_index];
        }
        const std::string parent_segment_name =
            parent_tensor && !parent_tensor->segment_name.empty() ? parent_tensor->segment_name
            : parent_tensor && !parent_tensor->name.empty()       ? parent_tensor->name
                                                            : std::string("mla_output_tensor");
        const bool publish_separate_semantic_view =
            !view.publish_transport_tensor && (packed_single_physical || boundary_is_offset_view);
        if (publish_separate_semantic_view) {
          tensor.source_physical_index = parent_tensor && parent_tensor->physical_index >= 0
                                             ? parent_tensor->physical_index
                                             : static_cast<int>(physical_index);
          tensor.source_byte_offset = boundary_byte_offset;
          tensor.physical_index = static_cast<int>(oi);
          tensor.segment_name = published_name;
          tensor.byte_offset = 0;
          tensor.materialization_kind = MpkTensorMaterializationKind::OffsetView;
        } else if (packed_single_physical || boundary_is_offset_view) {
          tensor.source_physical_index = parent_tensor && parent_tensor->physical_index >= 0
                                             ? parent_tensor->physical_index
                                             : static_cast<int>(physical_index);
          // A packed single-parent MLA export keeps one parent physical buffer.
          // The logical child view moves via byte_offset; source_byte_offset must stay anchored
          // at the parent base so downstream consumers do not split one parent into many
          // synthetic physical inputs.
          tensor.source_byte_offset = 0;
          tensor.physical_index = tensor.source_physical_index;
          tensor.segment_name = parent_segment_name;
          tensor.byte_offset = boundary_byte_offset;
          tensor.materialization_kind = MpkTensorMaterializationKind::OffsetView;
        } else {
          tensor.source_physical_index = static_cast<int>(physical_index);
          tensor.source_byte_offset = 0;
          tensor.physical_index = static_cast<int>(oi);
          tensor.segment_name = published_name;
          tensor.byte_offset = 0;
          tensor.materialization_kind = MpkTensorMaterializationKind::Direct;
        }
        // INT8 cblock-aligned MLA boundary: the MLA elf produces tile-packed
        // output where each 16-byte stride contains 8 valid INT8 values plus
        // 8 bytes of zero padding (verified via OFM byte dumps showing the
        // pattern: "61 81 80 93 88 81 80 80 | 00 00 00 00 00 00 00 00 ...").
        // Without a separate detessellate/lane-split stage on the slice ->
        // dequantize chain (the multibuff variants), downstream dequantize
        // reads ALL 16 bytes per stride and dequantizes the padding zeros as
        // probabilities, garbling segmentation output. Flag the boundary
        // tensor with Bf16LaneSplitRepack — the existing Bf16-named enum is
        // reused as a generic "lane-split" hint that downstream consumers
        // already plumb through (PreparedRuntimeBuild, ProcessCvuStageSemantics,
        // PreparedRuntimeBridge).
        if (mla && ((mla->has_align_c16 && mla->align_c16) || (mla->has_cblock && mla->cblock))) {
          const auto dtype_lower = canonical_token_local(tensor.dtype);
          const bool is_int8 = dtype_lower == "int8" || dtype_lower == "i8";
          const bool is_bf16 = dtype_lower == "bf16" || dtype_lower == "bfloat16";
          if (is_int8 || is_bf16) {
            tensor.materialization_kind = MpkTensorMaterializationKind::Bf16LaneSplitRepack;
          }
        }
        if (!published_name.empty()) {
          tensor.name = published_name;
        } else if (tensor.name.empty()) {
          tensor.name = published_name;
        }
        if (!view.publish_transport_tensor) {
          set_dense_tensor_contract_size_preserve_stride_local(&tensor, 0U, boundary_context);
        }
        validate_mla_boundary_tensor_contract_local(
            tensor, boundary_context, view.publish_transport_tensor, view.boundary_stage);
        out.push_back(std::move(tensor));
      }
      if (!out.empty()) {
        return out;
      }
    }

    out.reserve(mla->output_tensors.size());
    for (std::size_t i = 0; i < mla->output_tensors.size(); ++i) {
      MpkTensorContract tensor = canonical_tensor(mla->output_tensors[i], static_cast<int>(i));
      if (mla_idx.has_value()) {
        const auto resolved = resolve_mla_graph_output_local(
            contract, *mla_idx, static_cast<int>(i), mla->output_tensors[i]);
        if (tensor.mpk_shape.empty() && !resolved.shape.empty()) {
          tensor.mpk_shape = resolved.shape;
          tensor.shape_semantics = MpkShapeSemantics::Geometry;
        }
        if (tensor.dtype.empty() && !resolved.semantic_dtype.empty()) {
          tensor.dtype = resolved.semantic_dtype;
        }
        if (tensor.size_bytes == 0U && resolved.size_bytes > 0U) {
          tensor.size_bytes = resolved.size_bytes;
        }
      }
      derive_geometry_size_bytes(&tensor);
      tensor.physical_index = static_cast<int>(i);
      tensor.segment_name =
          !mla->output_tensors[i].name.empty()
              ? mla->output_tensors[i].name
              : ((mla->output_tensors.size() == 1U) ? "mla_output_tensor"
                                                    : ("ofm" + std::to_string(i)));
      tensor.byte_offset = 0;
      out.push_back(std::move(tensor));
    }
    return out;
  };

  return build_outputs(true);
}

std::vector<MpkTensorContract> get_mla_logical_outputs_contract(const MpkContract& contract) {
  auto build_outputs = [&](const bool publish_transport_boundary_views) {
    std::vector<MpkTensorContract> out;
    const MpkPluginIoContract* mla = get_mla_stage_io_contract(contract);
    if (!mla) {
      return out;
    }

    auto derive_geometry_size_bytes = [&](MpkTensorContract* tensor) {
      if (!tensor || tensor->size_bytes > 0U || tensor->mpk_shape.empty() ||
          !is_geometry_shape_semantics_local(tensor->shape_semantics)) {
        return;
      }
      std::size_t total = 1U;
      for (const auto dim : tensor->mpk_shape) {
        if (dim <= 0) {
          return;
        }
        const auto u_dim = static_cast<std::size_t>(dim);
        if (total > std::numeric_limits<std::size_t>::max() / u_dim) {
          return;
        }
        total *= u_dim;
      }
      const auto dtype_bytes = normalize_dtype_local(tensor->dtype);
      if (dtype_bytes == "FP32" || dtype_bytes == "INT32") {
        total *= 4U;
      } else if (dtype_bytes == "BF16" || dtype_bytes == "FP16" || dtype_bytes == "INT16" ||
                 dtype_bytes == "UINT16") {
        total *= 2U;
      }
      tensor->size_bytes = total;
    };

    auto canonical_tensor = [&](const MpkTensorContract& src, int tensor_index,
                                const std::string& dtype_override = std::string(),
                                const bool prefer_logical_shape = true) {
      MpkTensorContract tensor = src;
      tensor.tensor_index = tensor_index;
      if (prefer_logical_shape && !src.logical_shape.empty()) {
        tensor.mpk_shape = src.logical_shape;
        tensor.shape_semantics = MpkShapeSemantics::Geometry;
      } else if (!src.mpk_shape.empty()) {
        tensor.mpk_shape = src.mpk_shape;
        tensor.shape_semantics = src.shape_semantics;
      }
      if (!dtype_override.empty()) {
        tensor.dtype = dtype_override;
      } else if (prefer_logical_shape && !src.logical_dtype.empty()) {
        tensor.dtype = src.logical_dtype;
      }
      if (prefer_logical_shape && !tensor.mpk_shape.empty()) {
        tensor.size_bytes = 0U;
      }
      if (!prefer_logical_shape) {
        tensor.logical_shape.clear();
        tensor.logical_dtype.clear();
        tensor.logical_source_plugin.clear();
        tensor.logical_source_kernel.clear();
        tensor.logical_source_sequence = -1;
      }
      finalize_tensor_contract(&tensor);
      derive_geometry_size_bytes(&tensor);
      return tensor;
    };

    // Packed MLA exports fan out through an unpack stage. Lift boundary tensors from that
    // fanout and keep the output view anchored at the MLA boundary. For published outputs,
    // detess-bound heads must stay transport-shaped so downstream consumers bind the packed
    // MLA sub-buffer directly instead of a logical reinterpretation.
    const auto mla_idx = plugin_index_from_ptr_local(contract, mla);
    const auto boundary_views = resolve_mla_boundary_tensor_views_local(contract, *mla);
    if (!boundary_views.empty()) {
      out.reserve(boundary_views.size());
      std::vector<std::uint64_t> unpack_raw_offsets(boundary_views.size(), 0U);
      std::uint64_t running_unpack_offset = 0U;
      for (std::size_t oi = 0; oi < boundary_views.size(); ++oi) {
        unpack_raw_offsets[oi] = running_unpack_offset;
        running_unpack_offset += boundary_parent_span_bytes_local(boundary_views[oi]);
      }

      for (std::size_t oi = 0; oi < boundary_views.size(); ++oi) {
        const auto& view = boundary_views[oi];
        const bool publish_transport_view =
            publish_transport_boundary_views && view.publish_transport_tensor;
        const MpkTensorContract* semantic_boundary_tensor =
            (view.boundary_stage != nullptr && view.boundary_stage != mla && view.boundary_tensor)
                ? view.boundary_tensor
                : (view.unpack_tensor ? view.unpack_tensor : view.boundary_tensor);
        const MpkTensorContract* selected_boundary_tensor =
            publish_transport_view
                ? (view.boundary_tensor ? view.boundary_tensor : semantic_boundary_tensor)
                : (semantic_boundary_tensor ? semantic_boundary_tensor : view.boundary_tensor);
        if (!selected_boundary_tensor) {
          continue;
        }
        const std::string boundary_context =
            (view.boundary_stage && !view.boundary_stage->name.empty())
                ? (view.boundary_stage->name + "[output " + std::to_string(oi) + "]")
                : ("MLA boundary output " + std::to_string(oi));
        const bool prefer_logical_boundary = !publish_transport_view;
        const std::uint64_t boundary_transport_span_bytes = boundary_parent_span_bytes_local(view);
        MpkTensorContract tensor = canonical_tensor(*selected_boundary_tensor, static_cast<int>(oi),
                                                    view.boundary_dtype, prefer_logical_boundary);
        if (prefer_logical_boundary && mla_idx.has_value() && oi < mla->output_tensors.size()) {
          const auto resolved = resolve_mla_graph_output_local(
              contract, *mla_idx, static_cast<int>(oi), mla->output_tensors[oi]);
          if (tensor.mpk_shape.empty() && !resolved.shape.empty()) {
            tensor.mpk_shape = resolved.shape;
            tensor.shape_semantics = MpkShapeSemantics::Geometry;
          }
          if (tensor.dtype.empty() && !resolved.semantic_dtype.empty()) {
            tensor.dtype = resolved.semantic_dtype;
          }
          if (tensor.size_bytes == 0U && resolved.size_bytes > 0U) {
            tensor.size_bytes = resolved.size_bytes;
          }
          derive_geometry_size_bytes(&tensor);
        }
        if (!prefer_logical_boundary) {
          MpkTensorContract transport_tensor = tensor;
          if (boundary_transport_span_bytes > 0U) {
            transport_tensor.size_bytes = static_cast<std::size_t>(boundary_transport_span_bytes);
          }
          tensor.mpk_shape = canonical_detess_transport_shape_local(
              *view.boundary_stage, transport_tensor, view.boundary_dtype);
          tensor.shape_semantics = MpkShapeSemantics::Geometry;
          recompute_dense_tensor_contract_geometry_local(&tensor, boundary_transport_span_bytes,
                                                         boundary_context);
        }
        const bool packed_single_physical =
            mla->output_tensors.size() == 1U && boundary_views.size() > 1U;
        std::size_t physical_index = 0U;
        bool have_physical_index = false;
        if (tensor.physical_index >= 0 &&
            static_cast<std::size_t>(tensor.physical_index) < mla->output_tensors.size()) {
          physical_index = static_cast<std::size_t>(tensor.physical_index);
          have_physical_index = true;
        } else if (!tensor.segment_name.empty()) {
          for (std::size_t pi = 0; pi < mla->output_tensors.size(); ++pi) {
            const auto& physical_candidate = mla->output_tensors[pi];
            const std::string physical_segment = !physical_candidate.segment_name.empty()
                                                     ? physical_candidate.segment_name
                                                     : physical_candidate.name;
            if (!physical_segment.empty() && physical_segment == tensor.segment_name) {
              physical_index = pi;
              have_physical_index = true;
              break;
            }
          }
        }
        if (!have_physical_index) {
          physical_index = packed_single_physical
                               ? 0U
                               : std::min<std::size_t>(oi, mla->output_tensors.size() - 1U);
        }

        const auto published_name = !view.boundary_name.empty()
                                        ? view.boundary_name
                                        : ("mla_logical_output_" + std::to_string(oi));
        const auto unpack_dtype =
            !view.boundary_dtype.empty()
                ? view.boundary_dtype
                : (view.unpack_tensor ? view.unpack_tensor->dtype : tensor.dtype);
        const auto unpack_shape = (view.unpack_tensor && !view.unpack_tensor->mpk_shape.empty())
                                      ? view.unpack_tensor->mpk_shape
                                      : tensor.mpk_shape;
        const auto unpack_stride_bytes = contiguous_stride_bytes_local(unpack_shape, unpack_dtype);
        const std::string boundary_kernel = view.boundary_stage != nullptr
                                                ? canonical_token_local(view.boundary_stage->kernel)
                                                : std::string{};
        const bool boundary_is_slice = boundary_kernel.find("slice") != std::string::npos;
        const bool boundary_is_batch_flatten =
            boundary_kernel.find("batchflatten") != std::string::npos;
        const bool boundary_is_offset_view = boundary_is_slice || boundary_is_batch_flatten;
        std::optional<std::int64_t> explicit_boundary_byte_offset;
        if (view.boundary_tensor && view.boundary_tensor->byte_offset > 0) {
          explicit_boundary_byte_offset = view.boundary_tensor->byte_offset;
        } else if (view.unpack_tensor && view.unpack_tensor->byte_offset > 0 &&
                   packed_single_physical) {
          explicit_boundary_byte_offset = view.unpack_tensor->byte_offset;
        }
        std::int64_t boundary_byte_offset =
            explicit_boundary_byte_offset.has_value()
                ? *explicit_boundary_byte_offset
                : (packed_single_physical ? static_cast<std::int64_t>(unpack_raw_offsets[oi]) : 0);
        if (boundary_is_slice && !explicit_boundary_byte_offset.has_value()) {
          const auto slice_offset = projected_slice_begin_offset_bytes_local(
              view.boundary_stage->slice_begin, unpack_stride_bytes);
          if (slice_offset != std::numeric_limits<std::int64_t>::max() &&
              boundary_byte_offset <= std::numeric_limits<std::int64_t>::max() - slice_offset) {
            boundary_byte_offset += slice_offset;
          } else {
            boundary_byte_offset = std::numeric_limits<std::int64_t>::max();
          }
          tensor.stride_bytes = normalize_stride_rank_to_shape_local(
              unpack_stride_bytes, unpack_shape, tensor.mpk_shape);
        } else if (boundary_is_batch_flatten) {
          tensor.stride_bytes = flatten_batched_view_stride_local(unpack_stride_bytes, unpack_shape,
                                                                  tensor.mpk_shape);
        } else {
          tensor.stride_bytes = view.publish_transport_tensor
                                    ? std::vector<std::int64_t>{}
                                    : contiguous_stride_bytes_local(tensor.mpk_shape, tensor.dtype);
        }

        const MpkTensorContract* parent_tensor =
            packed_single_physical ? &mla->output_tensors[physical_index] : view.unpack_tensor;
        if (!parent_tensor && physical_index < mla->output_tensors.size()) {
          parent_tensor = &mla->output_tensors[physical_index];
        }
        const std::string parent_segment_name =
            parent_tensor && !parent_tensor->segment_name.empty() ? parent_tensor->segment_name
            : parent_tensor && !parent_tensor->name.empty()       ? parent_tensor->name
                                                            : std::string("mla_output_tensor");
        const bool publish_separate_semantic_view =
            !view.publish_transport_tensor && (packed_single_physical || boundary_is_offset_view);
        if (publish_separate_semantic_view) {
          tensor.source_physical_index = parent_tensor && parent_tensor->physical_index >= 0
                                             ? parent_tensor->physical_index
                                             : static_cast<int>(physical_index);
          tensor.source_byte_offset = boundary_byte_offset;
          tensor.physical_index = static_cast<int>(oi);
          tensor.segment_name = published_name;
          tensor.byte_offset = 0;
        } else if (packed_single_physical || boundary_is_offset_view) {
          tensor.source_physical_index = parent_tensor && parent_tensor->physical_index >= 0
                                             ? parent_tensor->physical_index
                                             : static_cast<int>(physical_index);
          tensor.source_byte_offset = 0;
          tensor.physical_index = tensor.source_physical_index;
          tensor.segment_name = parent_segment_name;
          tensor.byte_offset = boundary_byte_offset;
        } else {
          tensor.source_physical_index = static_cast<int>(physical_index);
          tensor.source_byte_offset = 0;
          tensor.physical_index = static_cast<int>(oi);
          tensor.segment_name = published_name;
          tensor.byte_offset = 0;
        }
        if (!published_name.empty()) {
          tensor.name = published_name;
        } else if (tensor.name.empty()) {
          tensor.name = published_name;
        }
        // get_mla_logical_outputs_contract() returns the runtime semantic tensor contract.
        // Do not preserve a stale upstream logical_shape/logical_dtype pair from the source
        // unpack tensor here; downstream publication must see one canonical semantic shape.
        tensor.logical_shape = tensor.mpk_shape;
        tensor.logical_dtype = normalize_dtype_local(tensor.dtype);
        set_dense_tensor_contract_size_preserve_stride_local(&tensor, 0U, boundary_context);
        validate_mla_boundary_tensor_contract_local(tensor, boundary_context, false,
                                                    view.boundary_stage);
        out.push_back(std::move(tensor));
      }
      if (!out.empty()) {
        return out;
      }
    }

    out.reserve(mla->output_tensors.size());
    for (std::size_t i = 0; i < mla->output_tensors.size(); ++i) {
      MpkTensorContract tensor = canonical_tensor(mla->output_tensors[i], static_cast<int>(i));
      tensor.physical_index = static_cast<int>(i);
      tensor.segment_name =
          !mla->output_tensors[i].name.empty()
              ? mla->output_tensors[i].name
              : ((mla->output_tensors.size() == 1U) ? "mla_output_tensor"
                                                    : ("ofm" + std::to_string(i)));
      tensor.byte_offset = 0;
      out.push_back(std::move(tensor));
    }
    return out;
  };

  return build_outputs(false);
}

std::optional<MpkQuantContract> get_quant_params_contract(const MpkContract& contract,
                                                          const std::string& plugin_name_or_id) {
  const MpkPluginIoContract* stage = get_stage_io_contract(contract, plugin_name_or_id);
  if (!stage || !stage->quant.has_value()) {
    return std::nullopt;
  }
  return stage->quant;
}

std::optional<std::size_t> find_plugin_index_by_name_or_id(const MpkContract& contract,
                                                           const std::string& plugin_name_or_id) {
  if (plugin_name_or_id.empty()) {
    return std::nullopt;
  }
  const std::string want = lower_copy_local(plugin_name_or_id);
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    const auto& plugin = contract.plugins[i];
    if (lower_copy_local(plugin.name) == want || lower_copy_local(plugin.plugin_id) == want) {
      return i;
    }
  }
  return std::nullopt;
}

std::vector<std::size_t> plugins_in_execution_order(const MpkContract& contract) {
  return plugins_in_order_internal(contract);
}

} // namespace simaai::neat::pipeline_internal::sima
