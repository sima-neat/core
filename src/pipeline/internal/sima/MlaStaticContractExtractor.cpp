#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/EnvUtil.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string lower_copy_local(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool mla_contract_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_MLA_CONTRACT_DEBUG", false);
}

bool mla_contract_dirty_disable_segment_mode() {
  const char* raw = std::getenv("SIMA_DIRTY_AB_DISABLE_SEGMENT_MODE");
  if (!raw) {
    return false;
  }
  const std::string v = lower_copy_local(std::string(raw));
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

const char* debug_string_or_empty(const std::string& value) {
  return value.empty() ? "<empty>" : value.c_str();
}

const nlohmann::json* params_or_root(const nlohmann::json& root) {
  if (root.contains("simaai__params") && root["simaai__params"].is_object()) {
    return &root["simaai__params"];
  }
  return &root;
}

const nlohmann::json* find_field(const nlohmann::json& root, const nlohmann::json& params,
                                 const char* key) {
  if (params.contains(key))
    return &params.at(key);
  if (root.contains(key))
    return &root.at(key);
  return nullptr;
}

template <typename T> std::vector<T> read_numeric_array_any(const nlohmann::json& value) {
  std::vector<T> out;
  if (value.is_array()) {
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

int read_first_int_any(const nlohmann::json& value) {
  const auto values = read_numeric_array_any<std::int64_t>(value);
  if (values.empty()) {
    return 0;
  }
  return static_cast<int>(values[0]);
}

std::vector<std::string> read_string_array_any(const nlohmann::json& value) {
  std::vector<std::string> out;
  if (value.is_array()) {
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

int to_non_negative_int(std::int64_t value) {
  if (value < 0)
    return 0;
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  return static_cast<int>(value);
}

std::string normalize_layout(const std::string& layout_raw) {
  return tensorsemantics::normalize_layout_token(layout_raw);
}

std::string normalize_video_format_token(std::string token) {
  token = upper_copy(token);
  if (token == "GRAY")
    return "GRAY8";
  if (token == "IYUV")
    return "I420";
  return token;
}

std::vector<std::string> split_format_values(std::string values) {
  for (char& c : values) {
    if (c == '(' || c == ')' || c == '[' || c == ']') {
      c = ' ';
    }
  }
  std::vector<std::string> out;
  std::string cur;
  for (char c : values) {
    if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
      if (!cur.empty()) {
        out.push_back(normalize_video_format_token(cur));
        cur.clear();
      }
      continue;
    }
    cur.push_back(c);
  }
  if (!cur.empty()) {
    out.push_back(normalize_video_format_token(cur));
  }
  return out;
}

bool read_u16_le(const std::vector<std::uint8_t>& bytes, std::size_t off, std::uint16_t* out) {
  if (!out || off + 2U > bytes.size()) {
    return false;
  }
  *out =
      static_cast<std::uint16_t>(bytes[off]) | (static_cast<std::uint16_t>(bytes[off + 1U]) << 8U);
  return true;
}

bool read_u32_le(const std::vector<std::uint8_t>& bytes, std::size_t off, std::uint32_t* out) {
  if (!out || off + 4U > bytes.size()) {
    return false;
  }
  *out = static_cast<std::uint32_t>(bytes[off]) |
         (static_cast<std::uint32_t>(bytes[off + 1U]) << 8U) |
         (static_cast<std::uint32_t>(bytes[off + 2U]) << 16U) |
         (static_cast<std::uint32_t>(bytes[off + 3U]) << 24U);
  return true;
}

bool read_u64_le(const std::vector<std::uint8_t>& bytes, std::size_t off, std::uint64_t* out) {
  if (!out || off + 8U > bytes.size()) {
    return false;
  }
  *out = static_cast<std::uint64_t>(bytes[off]) |
         (static_cast<std::uint64_t>(bytes[off + 1U]) << 8U) |
         (static_cast<std::uint64_t>(bytes[off + 2U]) << 16U) |
         (static_cast<std::uint64_t>(bytes[off + 3U]) << 24U) |
         (static_cast<std::uint64_t>(bytes[off + 4U]) << 32U) |
         (static_cast<std::uint64_t>(bytes[off + 5U]) << 40U) |
         (static_cast<std::uint64_t>(bytes[off + 6U]) << 48U) |
         (static_cast<std::uint64_t>(bytes[off + 7U]) << 56U);
  return true;
}

std::string read_cstr_from_blob(const std::vector<std::uint8_t>& bytes, std::size_t off,
                                std::size_t limit) {
  if (off >= bytes.size() || off >= limit) {
    return {};
  }
  const std::size_t end = std::min(limit, bytes.size());
  std::size_t cur = off;
  while (cur < end && bytes[cur] != 0U) {
    ++cur;
  }
  if (cur <= off) {
    return {};
  }
  return std::string(reinterpret_cast<const char*>(bytes.data() + off), cur - off);
}

std::optional<int> parse_output_index_from_ofm_section_name(const std::string& name) {
  const std::string persistent_key = "persistent.output_";
  const auto pos = name.find(persistent_key);
  if (pos != std::string::npos) {
    std::size_t idx_pos = pos + persistent_key.size();
    std::size_t idx_end = idx_pos;
    while (idx_end < name.size() && std::isdigit(static_cast<unsigned char>(name[idx_end]))) {
      ++idx_end;
    }
    if (idx_end > idx_pos) {
      try {
        return std::stoi(name.substr(idx_pos, idx_end - idx_pos));
      } catch (const std::exception&) {
      }
    }
  }
  const std::string b_key = "data.ofm.b";
  if (name.rfind(b_key, 0) == 0U) {
    const std::string suffix = name.substr(b_key.size());
    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; })) {
      try {
        return std::stoi(suffix);
      } catch (const std::exception&) {
      }
    }
  }
  return std::nullopt;
}

struct ElfOfmSectionInfo {
  std::string section_name;
  std::size_t byte_size = 0;
  std::optional<int> output_index;
};

[[maybe_unused]] std::vector<ElfOfmSectionInfo>
read_mla_ofm_sections_from_elf(const std::string& elf_path) {
  std::vector<ElfOfmSectionInfo> out;
  if (elf_path.empty()) {
    return out;
  }
  std::ifstream in(elf_path, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return out;
  }
  const std::vector<std::uint8_t> blob{std::istreambuf_iterator<char>(in),
                                       std::istreambuf_iterator<char>()};
  if (blob.size() < 64U) {
    return out;
  }
  if (!(blob[0] == 0x7F && blob[1] == 'E' && blob[2] == 'L' && blob[3] == 'F')) {
    return out;
  }
  // Only parse ELF64 little-endian section headers.
  if (blob[4] != 2U || blob[5] != 1U) {
    return out;
  }

  std::uint64_t shoff = 0;
  std::uint16_t shentsize = 0;
  std::uint16_t shnum = 0;
  std::uint16_t shstrndx = 0;
  if (!read_u64_le(blob, 40U, &shoff) || !read_u16_le(blob, 58U, &shentsize) ||
      !read_u16_le(blob, 60U, &shnum) || !read_u16_le(blob, 62U, &shstrndx)) {
    return out;
  }
  if (shoff == 0U || shentsize < 64U || shnum == 0U || shstrndx >= shnum) {
    return out;
  }
  if (shoff > blob.size()) {
    return out;
  }
  const std::uint64_t sh_table_bytes =
      static_cast<std::uint64_t>(shentsize) * static_cast<std::uint64_t>(shnum);
  if (shoff + sh_table_bytes > static_cast<std::uint64_t>(blob.size())) {
    return out;
  }

  const auto section_offset = [&](std::uint16_t idx) -> std::size_t {
    return static_cast<std::size_t>(shoff + static_cast<std::uint64_t>(idx) * shentsize);
  };

  const std::size_t shstr_shoff = section_offset(shstrndx);
  std::uint64_t shstr_off = 0;
  std::uint64_t shstr_size = 0;
  if (!read_u64_le(blob, shstr_shoff + 24U, &shstr_off) ||
      !read_u64_le(blob, shstr_shoff + 32U, &shstr_size)) {
    return out;
  }
  if (shstr_off >= blob.size() ||
      shstr_off + shstr_size > static_cast<std::uint64_t>(blob.size())) {
    return out;
  }

  std::map<int, ElfOfmSectionInfo> indexed;
  std::vector<ElfOfmSectionInfo> unordered;
  std::set<std::string> seen;
  for (std::uint16_t i = 0; i < shnum; ++i) {
    const std::size_t off = section_offset(i);
    std::uint32_t name_off = 0;
    std::uint64_t size = 0;
    if (!read_u32_le(blob, off, &name_off) || !read_u64_le(blob, off + 32U, &size)) {
      continue;
    }
    const std::string name =
        read_cstr_from_blob(blob, static_cast<std::size_t>(shstr_off) + name_off,
                            static_cast<std::size_t>(shstr_off + shstr_size));
    if (name.empty() || name.rfind("data.ofm", 0) != 0U) {
      continue;
    }
    if (!seen.insert(name).second) {
      continue;
    }
    ElfOfmSectionInfo info;
    info.section_name = name;
    info.output_index = parse_output_index_from_ofm_section_name(name);
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      info.byte_size = std::numeric_limits<std::size_t>::max();
    } else {
      info.byte_size = static_cast<std::size_t>(size);
    }
    if (info.output_index.has_value()) {
      indexed[*info.output_index] = info;
    } else {
      unordered.push_back(std::move(info));
    }
  }

  out.reserve(indexed.size() + unordered.size());
  for (const auto& it : indexed) {
    out.push_back(it.second);
  }
  std::sort(unordered.begin(), unordered.end(),
            [](const ElfOfmSectionInfo& a, const ElfOfmSectionInfo& b) {
              return a.section_name < b.section_name;
            });
  out.insert(out.end(), unordered.begin(), unordered.end());
  return out;
}

[[maybe_unused]] std::string resolve_model_elf_path(const nlohmann::json& root,
                                                    const nlohmann::json& params,
                                                    const std::string& config_path) {
  const nlohmann::json* model_path_json = find_field(root, params, "model_path");
  if (!model_path_json || !model_path_json->is_string()) {
    return {};
  }
  const std::string model_path = model_path_json->get<std::string>();
  if (model_path.empty()) {
    return {};
  }
  namespace fs = std::filesystem;
  std::vector<fs::path> candidates;
  const fs::path raw(model_path);
  if (raw.is_absolute()) {
    candidates.push_back(raw);
  } else {
    if (!config_path.empty()) {
      const fs::path cfg(config_path);
      const fs::path cfg_dir = cfg.parent_path();
      if (!cfg_dir.empty()) {
        candidates.push_back(cfg_dir / raw);
        candidates.push_back(cfg_dir.parent_path() / "share" / raw);
        candidates.push_back(cfg_dir.parent_path() / raw);
      }
    }
    candidates.push_back(raw);
  }
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (candidate.empty()) {
      continue;
    }
    if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
      return candidate.string();
    }
  }
  return {};
}

std::string normalize_tensor_dtype_token(std::string raw) {
  raw = upper_copy(raw);
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw == "FP16" || raw == "HALF") {
    return "FP16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw == "FP32") {
    return "FP32";
  }
  if (raw.find("UINT8") != std::string::npos || raw == "U8") {
    return "UINT8";
  }
  if (raw.find("INT8") != std::string::npos || raw == "S8") {
    return "INT8";
  }
  if (raw.find("INT16") != std::string::npos || raw == "S16") {
    return "INT16";
  }
  if (raw.find("INT32") != std::string::npos || raw == "S32") {
    return "INT32";
  }
  return raw;
}

void assign_tensor_dims_from_shape(const std::vector<std::int64_t>& shape, TensorStaticSpec* t) {
  if (!t || shape.empty()) {
    return;
  }
  const std::string layout = normalize_layout(t->layout);
  if (shape.size() >= 3U) {
    if (layout == "CHW") {
      t->max_h = to_non_negative_int(shape[shape.size() - 2U]);
      t->max_w = to_non_negative_int(shape.back());
    } else if (layout == "HWC") {
      t->max_h = to_non_negative_int(shape[shape.size() - 3U]);
      t->max_w = to_non_negative_int(shape[shape.size() - 2U]);
    }
  } else if (shape.size() == 2U) {
    t->max_h = to_non_negative_int(shape[0]);
    t->max_w = to_non_negative_int(shape[1]);
  } else if (shape.size() == 1U) {
    t->max_h = 1;
    t->max_w = to_non_negative_int(shape[0]);
  }
}

std::vector<std::int64_t> normalize_tensor_shape_local(std::vector<std::int64_t> shape) {
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  return shape;
}

std::vector<std::int64_t>
mla_input_logical_shape_from_mpk_tensor_local(const MpkTensorContract& src) {
  const std::vector<std::int64_t> physical_shape = normalize_tensor_shape_local(src.mpk_shape);
  const std::vector<std::int64_t> logical_shape = normalize_tensor_shape_local(src.logical_shape);
  const bool has_logical_shape = !logical_shape.empty();

  if (has_logical_shape) {
    return logical_shape;
  }
  return physical_shape;
}

TensorDType tensor_dtype_from_contract_token_local(std::string token) {
  token = upper_copy(token);
  if (token.find("BFLOAT16") != std::string::npos || token.find("BF16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (token.find("FLOAT16") != std::string::npos || token.find("FP16") != std::string::npos) {
    return TensorDType::Int16;
  }
  if (token.find("FLOAT64") != std::string::npos || token.find("FP64") != std::string::npos) {
    return TensorDType::Float64;
  }
  if (token.find("FLOAT32") != std::string::npos || token.find("FP32") != std::string::npos) {
    return TensorDType::Float32;
  }
  if (token.find("UINT16") != std::string::npos) {
    return TensorDType::UInt16;
  }
  if (token.find("INT16") != std::string::npos) {
    return TensorDType::Int16;
  }
  if (token.find("INT32") != std::string::npos) {
    return TensorDType::Int32;
  }
  if (token.find("UINT8") != std::string::npos) {
    return TensorDType::UInt8;
  }
  return TensorDType::Int8;
}

std::uint64_t tensor_logical_size_bytes_from_static_spec_local(const TensorStaticSpec& tensor) {
  if (tensor.shape.empty()) {
    return tensor.max_stride > 0 ? static_cast<std::uint64_t>(tensor.max_stride) : 0U;
  }
  const std::size_t elem_bytes =
      pipeline_internal::dtype_bytes(tensor_dtype_from_contract_token_local(
          tensor.dtype.empty() ? std::string("INT8") : tensor.dtype));
  if (elem_bytes == 0U) {
    return tensor.max_stride > 0 ? static_cast<std::uint64_t>(tensor.max_stride) : 0U;
  }
  std::size_t total = 1U;
  for (const auto dim : tensor.shape) {
    if (dim <= 0) {
      return 0U;
    }
    std::size_t next = 0U;
    if (!pipeline_internal::safe_mul(total, static_cast<std::size_t>(dim), &next)) {
      return 0U;
    }
    total = next;
  }
  std::size_t total_bytes = 0U;
  if (!pipeline_internal::safe_mul(total, elem_bytes, &total_bytes)) {
    return 0U;
  }
  return static_cast<std::uint64_t>(total_bytes);
}

std::vector<std::int64_t>
tensor_stride_bytes_from_static_spec_local(const TensorStaticSpec& tensor) {
  if (tensor.shape.empty()) {
    return {};
  }
  return pipeline_internal::contiguous_strides_bytes(
      tensor.shape, pipeline_internal::dtype_bytes(tensor_dtype_from_contract_token_local(
                        tensor.dtype.empty() ? std::string("INT8") : tensor.dtype)));
}

std::vector<std::int64_t>
tensor_stride_bytes_from_mpk_tensor_local(const MpkTensorContract& src,
                                          const TensorStaticSpec& tensor) {
  if (!src.stride_bytes.empty()) {
    auto normalized = pipeline_internal::normalize_strides_rank_to_shape(
        src.stride_bytes, src.mpk_shape, tensor.shape, true);
    if (normalized.size() == tensor.shape.size()) {
      return normalized;
    }
    normalized = pipeline_internal::normalize_strides_rank_to_shape(
        src.stride_bytes, src.logical_shape, tensor.shape, true);
    if (normalized.size() == tensor.shape.size()) {
      return normalized;
    }
    return src.stride_bytes;
  }
  return tensor_stride_bytes_from_static_spec_local(tensor);
}

std::uint64_t
tensor_physical_span_bytes_from_shape_strides_local(const std::vector<std::int64_t>& shape,
                                                    const std::vector<std::int64_t>& stride_bytes,
                                                    const std::string& dtype) {
  if (shape.empty() || stride_bytes.empty()) {
    return 0U;
  }
  if (shape.size() != stride_bytes.size()) {
    return 0U;
  }
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(
      tensor_dtype_from_contract_token_local(dtype.empty() ? std::string("INT8") : dtype));
  std::uint64_t max_offset = 0U;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || stride_bytes[i] < 0) {
      return 0U;
    }
    if (shape[i] == 1) {
      continue;
    }
    const auto dim = static_cast<std::uint64_t>(shape[i] - 1);
    const auto stride = static_cast<std::uint64_t>(stride_bytes[i]);
    std::uint64_t delta = 0U;
    if (!pipeline_internal::safe_mul(dim, stride, &delta) ||
        !pipeline_internal::safe_add(max_offset, delta, &max_offset)) {
      return 0U;
    }
  }
  std::uint64_t span = 0U;
  if (!pipeline_internal::safe_add(max_offset, static_cast<std::uint64_t>(elem_bytes), &span)) {
    return 0U;
  }
  return span;
}

TensorStaticSpec tensor_spec_from_mpk_tensor(const MpkTensorContract& src, int tensor_index,
                                             const std::string& default_name,
                                             const std::string& default_dtype) {
  TensorStaticSpec t;
  t.tensor_index = tensor_index;
  t.dtype = normalize_tensor_dtype_token(src.dtype.empty() ? default_dtype : src.dtype);
  t.shape = !src.logical_shape.empty() ? normalize_tensor_shape_local(src.logical_shape)
                                       : normalize_tensor_shape_local(src.mpk_shape);
  assign_tensor_dims_from_shape(t.shape, &t);
  t.semantic_tag = src.name.empty() ? default_name : src.name;
  if (src.size_bytes > 0U) {
    t.max_stride = to_non_negative_int(static_cast<std::int64_t>(std::min<std::size_t>(
        src.size_bytes, static_cast<std::size_t>(std::numeric_limits<int>::max()))));
  }
  return t;
}

void assign_output_quant(MlaStaticContract* contract,
                         const std::optional<MpkQuantContract>& quant) {
  if (!contract || !quant.has_value() || quant->scales.empty() || quant->zero_points.empty()) {
    return;
  }

  const bool per_output_scalars = contract->logical_outputs.size() > 1U &&
                                  quant->scales.size() >= contract->logical_outputs.size() &&
                                  quant->zero_points.size() >= contract->logical_outputs.size();
  if (per_output_scalars) {
    contract->output_quant.reserve(contract->logical_outputs.size());
    for (std::size_t i = 0; i < contract->logical_outputs.size(); ++i) {
      QuantStaticSpec q;
      q.granularity = QuantGranularity::PerTensor;
      q.axis = -1;
      q.scales.push_back(quant->scales[i]);
      q.zero_points.push_back(quant->zero_points[i]);
      contract->output_quant.push_back(std::move(q));
    }
  } else {
    QuantStaticSpec q;
    q.granularity = (quant->scales.size() > 1 || quant->zero_points.size() > 1)
                        ? QuantGranularity::PerAxis
                        : QuantGranularity::PerTensor;
    q.axis = quant->axis;
    q.scales = quant->scales;
    q.zero_points = quant->zero_points;
    contract->output_quant.push_back(std::move(q));
  }
}

bool validate_output_contracts(const MlaStaticContract& contract) {
  if (contract.logical_outputs.empty()) {
    if (mla_contract_debug_enabled()) {
      std::fprintf(stderr, "[mla-contract] invalid output contract: logical_outputs empty\n");
    }
    return false;
  }
  for (const auto& physical : contract.physical_outputs) {
    if (physical.source_physical_index < 0 ||
        static_cast<std::size_t>(physical.source_physical_index) >=
            contract.dispatcher_physical_outputs.size() ||
        physical.size_bytes == 0U || physical.source_byte_offset < 0) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(stderr,
                     "[mla-contract] invalid physical_output idx=%d src_physical_index=%d "
                     "size_bytes=%llu source_byte_offset=%lld dispatcher_count=%zu\n",
                     physical.physical_index, physical.source_physical_index,
                     static_cast<unsigned long long>(physical.size_bytes),
                     static_cast<long long>(physical.source_byte_offset),
                     contract.dispatcher_physical_outputs.size());
      }
      return false;
    }
    const auto& dispatcher =
        contract
            .dispatcher_physical_outputs[static_cast<std::size_t>(physical.source_physical_index)];
    const auto source_offset = static_cast<std::uint64_t>(physical.source_byte_offset);
    if (source_offset > std::numeric_limits<std::uint64_t>::max() - physical.size_bytes) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(stderr,
                     "[mla-contract] invalid physical_output idx=%d overflow source_offset=%llu "
                     "size_bytes=%llu\n",
                     physical.physical_index, static_cast<unsigned long long>(source_offset),
                     static_cast<unsigned long long>(physical.size_bytes));
      }
      return false;
    }
    const std::uint64_t source_end = source_offset + physical.size_bytes;
    if (source_end > dispatcher.size_bytes) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(stderr,
                     "[mla-contract] invalid physical_output idx=%d source_end=%llu "
                     "dispatcher_size=%llu segment=%s source_segment=%s\n",
                     physical.physical_index, static_cast<unsigned long long>(source_end),
                     static_cast<unsigned long long>(dispatcher.size_bytes),
                     debug_string_or_empty(physical.segment_name),
                     debug_string_or_empty(dispatcher.segment_name));
      }
      return false;
    }
  }
  const bool packed_single_physical =
      contract.physical_outputs.size() == 1U && contract.logical_outputs.size() > 1U &&
      std::all_of(contract.logical_outputs.begin(), contract.logical_outputs.end(),
                  [](const LogicalTensorStaticSpec& logical) {
                    return logical.layout.empty() && logical.shape.size() == 1U &&
                           logical.dtype == "INT8";
                  });
  std::uint64_t previous_end = 0U;
  for (std::size_t i = 0; i < contract.logical_outputs.size(); ++i) {
    const auto& logical = contract.logical_outputs[i];
    if (logical.physical_index < 0 ||
        static_cast<std::size_t>(logical.physical_index) >= contract.physical_outputs.size()) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(
            stderr,
            "[mla-contract] invalid logical_output idx=%zu physical_index=%d physical_count=%zu\n",
            i, logical.physical_index, contract.physical_outputs.size());
      }
      return false;
    }
    if (logical.size_bytes == 0U || logical.byte_offset < 0) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(
            stderr,
            "[mla-contract] invalid logical_output idx=%zu size_bytes=%llu byte_offset=%lld\n", i,
            static_cast<unsigned long long>(logical.size_bytes),
            static_cast<long long>(logical.byte_offset));
      }
      return false;
    }
    const auto& physical =
        contract.physical_outputs[static_cast<std::size_t>(logical.physical_index)];
    if (!logical.segment_name.empty() && !physical.segment_name.empty() &&
        logical.segment_name != physical.segment_name) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(stderr,
                     "[mla-contract] invalid logical_output idx=%zu segment_mismatch logical=%s "
                     "physical=%s\n",
                     i, debug_string_or_empty(logical.segment_name),
                     debug_string_or_empty(physical.segment_name));
      }
      return false;
    }
    const std::uint64_t logical_physical_span = tensor_physical_span_bytes_from_shape_strides_local(
        logical.shape, logical.stride_bytes, logical.dtype);
    const std::uint64_t span_bytes =
        logical_physical_span > 0U ? logical_physical_span : logical.size_bytes;
    const std::uint64_t logical_end = static_cast<std::uint64_t>(logical.byte_offset) + span_bytes;
    if (logical_end > physical.size_bytes) {
      if (mla_contract_debug_enabled()) {
        std::fprintf(
            stderr,
            "[mla-contract] invalid logical_output idx=%zu logical_end=%llu physical_size=%llu "
            "byte_offset=%lld span_bytes=%llu logical_size=%llu dtype=%s layout=%s segment=%s\n",
            i, static_cast<unsigned long long>(logical_end),
            static_cast<unsigned long long>(physical.size_bytes),
            static_cast<long long>(logical.byte_offset),
            static_cast<unsigned long long>(span_bytes),
            static_cast<unsigned long long>(logical.size_bytes),
            debug_string_or_empty(logical.dtype), debug_string_or_empty(logical.layout),
            debug_string_or_empty(logical.segment_name));
      }
      return false;
    }
    if (packed_single_physical) {
      if (i > 0U && static_cast<std::uint64_t>(logical.byte_offset) < previous_end) {
        if (mla_contract_debug_enabled()) {
          std::fprintf(
              stderr,
              "[mla-contract] invalid logical_output idx=%zu packed overlap byte_offset=%lld "
              "previous_end=%llu span_bytes=%llu logical_size=%llu dtype=%s layout=%s segment=%s\n",
              i, static_cast<long long>(logical.byte_offset),
              static_cast<unsigned long long>(previous_end),
              static_cast<unsigned long long>(span_bytes),
              static_cast<unsigned long long>(logical.size_bytes),
              debug_string_or_empty(logical.dtype), debug_string_or_empty(logical.layout),
              debug_string_or_empty(logical.segment_name));
        }
        return false;
      }
      previous_end = logical_end;
    }
  }
  return true;
}

std::optional<int>
resolve_logical_output_physical_index(const MpkTensorContract& logical_src,
                                      const std::vector<PhysicalBufferStaticSpec>& physical_outputs,
                                      std::size_t logical_index, bool packed_single_physical) {
  if (logical_src.physical_index >= 0 &&
      static_cast<std::size_t>(logical_src.physical_index) < physical_outputs.size()) {
    return logical_src.physical_index;
  }
  if (!logical_src.segment_name.empty()) {
    const auto physical_it =
        std::find_if(physical_outputs.begin(), physical_outputs.end(),
                     [&](const PhysicalBufferStaticSpec& physical) {
                       return physical.segment_name == logical_src.segment_name;
                     });
    if (physical_it != physical_outputs.end()) {
      return physical_it->physical_index;
    }
  }
  if (packed_single_physical && !physical_outputs.empty()) {
    return 0;
  }
  if (physical_outputs.size() == 1U) {
    return 0;
  }
  if (logical_index < physical_outputs.size() && physical_outputs.size() > 1U) {
    return static_cast<int>(logical_index);
  }
  return std::nullopt;
}

void build_output_contracts(MlaStaticContract* contract,
                            const std::vector<MpkTensorContract>& physical_outputs,
                            const std::vector<MpkTensorContract>& logical_outputs,
                            const std::optional<MpkQuantContract>& quant,
                            const std::string& canonical_output_dtype) {
  if (!contract) {
    return;
  }

  contract->dispatcher_physical_outputs.clear();
  contract->physical_outputs.clear();
  contract->logical_outputs.clear();

  contract->dispatcher_physical_outputs.reserve(physical_outputs.size());
  for (std::size_t i = 0; i < physical_outputs.size(); ++i) {
    const auto& src = physical_outputs[i];
    PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(i);
    physical.allocator_index = static_cast<int>(i);
    physical.source_physical_index = static_cast<int>(i);
    physical.source_byte_offset = 0;
    physical.device_kind = DeviceKind::Mla;
    physical.segment_name =
        src.name.empty()
            ? ((physical_outputs.size() == 1U) ? "mla_output_tensor" : ("ofm" + std::to_string(i)))
            : src.name;
    physical.size_bytes = src.size_bytes;
    contract->dispatcher_physical_outputs.push_back(std::move(physical));
  }

  struct PublishedPhysicalSeed {
    int published_index = -1;
    int allocator_index = -1;
    int source_physical_index = -1;
    std::int64_t source_byte_offset = 0;
    std::uint64_t size_bytes = 0;
    std::string segment_name;
  };

  std::vector<PublishedPhysicalSeed> published_seed;
  published_seed.reserve(logical_outputs.size());
  std::vector<int> published_index_for_logical(logical_outputs.size(), -1);
  const bool packed_single_parent = physical_outputs.size() == 1U && logical_outputs.size() > 1U;
  if (packed_single_parent) {
    std::string packed_segment_name;
    std::uint64_t packed_size_bytes = 0U;
    if (!contract->dispatcher_physical_outputs.empty()) {
      packed_segment_name = contract->dispatcher_physical_outputs[0].segment_name;
      packed_size_bytes = contract->dispatcher_physical_outputs[0].size_bytes;
    }
    for (const auto& logical : logical_outputs) {
      if (!logical.segment_name.empty()) {
        packed_segment_name = logical.segment_name;
        break;
      }
    }
    PublishedPhysicalSeed seed;
    seed.published_index = 0;
    seed.allocator_index = 0;
    seed.source_physical_index = 0;
    seed.source_byte_offset = 0;
    seed.size_bytes = packed_size_bytes;
    seed.segment_name = std::move(packed_segment_name);
    published_seed.push_back(std::move(seed));
    std::fill(published_index_for_logical.begin(), published_index_for_logical.end(), 0);
  } else {
    for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
      const auto& src = logical_outputs[i];
      const int requested_physical =
          src.physical_index >= 0 ? src.physical_index : static_cast<int>(i);
      const std::string segment_name =
          !src.segment_name.empty()
              ? src.segment_name
              : ((logical_outputs.size() == 1U) ? std::string("mla_output_tensor")
                                                : ("ofm" + std::to_string(i)));
      auto existing = std::find_if(
          published_seed.begin(), published_seed.end(), [&](const PublishedPhysicalSeed& seed) {
            return seed.published_index == requested_physical ||
                   (!segment_name.empty() && seed.segment_name == segment_name);
          });
      if (existing == published_seed.end()) {
        PublishedPhysicalSeed seed;
        seed.published_index = requested_physical;
        seed.allocator_index = requested_physical;
        seed.source_physical_index = src.source_physical_index >= 0
                                         ? src.source_physical_index
                                         : (src.physical_index >= 0 ? src.physical_index : 0);
        seed.source_byte_offset =
            src.source_byte_offset != 0 ? src.source_byte_offset : src.byte_offset;
        if (seed.source_physical_index >= 0 &&
            static_cast<std::size_t>(seed.source_physical_index) <
                contract->dispatcher_physical_outputs.size()) {
          const auto& dispatcher = contract->dispatcher_physical_outputs[static_cast<std::size_t>(
              seed.source_physical_index)];
          const auto source_offset =
              static_cast<std::uint64_t>(std::max<std::int64_t>(seed.source_byte_offset, 0));
          if (dispatcher.size_bytes > source_offset) {
            seed.size_bytes = dispatcher.size_bytes - source_offset;
          }
        }
        seed.segment_name = segment_name;
        published_seed.push_back(std::move(seed));
        existing = std::prev(published_seed.end());
      }
      published_index_for_logical[i] = existing->published_index;
    }
  }

  std::sort(published_seed.begin(), published_seed.end(),
            [](const PublishedPhysicalSeed& lhs, const PublishedPhysicalSeed& rhs) {
              if (lhs.published_index != rhs.published_index) {
                return lhs.published_index < rhs.published_index;
              }
              return lhs.segment_name < rhs.segment_name;
            });
  std::unordered_map<int, int> published_reindex;
  contract->physical_outputs.reserve(published_seed.size());
  for (std::size_t i = 0; i < published_seed.size(); ++i) {
    auto seed = published_seed[i];
    published_reindex[seed.published_index] = static_cast<int>(i);
    PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(i);
    physical.allocator_index = static_cast<int>(i);
    physical.source_physical_index = seed.source_physical_index;
    physical.source_byte_offset = seed.source_byte_offset;
    physical.device_kind = DeviceKind::Mla;
    physical.segment_name = seed.segment_name;
    physical.size_bytes = seed.size_bytes;
    contract->physical_outputs.push_back(std::move(physical));
  }

  contract->logical_outputs.reserve(logical_outputs.size());
  for (std::size_t i = 0; i < logical_outputs.size(); ++i) {
    const auto& src = logical_outputs[i];
    const std::string default_name =
        (logical_outputs.size() == 1U) ? "mla_output_tensor" : ("mla_output_" + std::to_string(i));
    const bool packed_byte_view = packed_single_parent &&
                                  src.shape_semantics == MpkShapeSemantics::PackedExtent &&
                                  src.logical_shape.empty();
    const std::string logical_output_dtype =
        packed_byte_view ? std::string("INT8") : canonical_output_dtype;
    TensorStaticSpec tensor =
        tensor_spec_from_mpk_tensor(src, static_cast<int>(i), default_name, logical_output_dtype);
    if (packed_byte_view) {
      tensor.dtype = logical_output_dtype;
    }

    const auto published_it = published_reindex.find(published_index_for_logical[i]);
    if (published_it == published_reindex.end()) {
      contract->dispatcher_physical_outputs.clear();
      contract->physical_outputs.clear();
      contract->logical_outputs.clear();
      contract->output_quant.clear();
      return;
    }

    LogicalTensorStaticSpec logical;
    const bool packed_extent_output =
        src.shape_semantics == MpkShapeSemantics::PackedExtent && src.logical_shape.empty();
    logical.logical_index = static_cast<int>(i);
    logical.backend_output_index = (src.tensor_index >= 0) ? src.tensor_index : static_cast<int>(i);
    logical.physical_index = published_it->second;
    logical.output_slot = static_cast<int>(i);
    logical.tensor_index = logical.backend_output_index;
    logical.byte_offset = src.byte_offset;
    if (logical.byte_offset < 0) {
      logical.byte_offset = 0;
    }
    logical.size_bytes = packed_extent_output && src.size_bytes > 0U
                             ? static_cast<std::uint64_t>(src.size_bytes)
                             : tensor_logical_size_bytes_from_static_spec_local(tensor);
    logical.shape = tensor.shape;
    logical.dtype = packed_extent_output ? std::string("INT8") : tensor.dtype;
    logical.dtype_source = packed_extent_output ? DTypeSource::InternalContract : src.dtype_source;
    if (packed_extent_output) {
      if (!src.stride_bytes.empty()) {
        logical.stride_bytes = src.stride_bytes;
      } else {
        logical.stride_bytes = pipeline_internal::contiguous_strides_bytes(
            logical.shape,
            pipeline_internal::dtype_bytes(tensor_dtype_from_contract_token_local(logical.dtype)));
      }
    } else {
      logical.stride_bytes = tensor_stride_bytes_from_mpk_tensor_local(src, tensor);
    }
    logical.layout = tensor.layout;
    logical.logical_name = tensor.semantic_tag.empty() ? default_name : tensor.semantic_tag;
    logical.backend_name = "ofm" + std::to_string(logical.backend_output_index);
    logical.segment_name =
        contract->physical_outputs[static_cast<std::size_t>(published_it->second)].segment_name;
    contract->logical_outputs.push_back(std::move(logical));

    auto& physical = contract->physical_outputs[static_cast<std::size_t>(published_it->second)];
    const auto& logical_ref = contract->logical_outputs.back();
    const std::uint64_t logical_physical_span = tensor_physical_span_bytes_from_shape_strides_local(
        logical_ref.shape, logical_ref.stride_bytes, logical_ref.dtype);
    const std::uint64_t logical_published_span =
        std::max<std::uint64_t>(logical_physical_span, logical_ref.size_bytes);
    const std::uint64_t logical_end =
        static_cast<std::uint64_t>(logical_ref.byte_offset) + logical_published_span;
    if (!packed_single_parent) {
      physical.size_bytes = std::max<std::uint64_t>(physical.size_bytes, logical_end);
    }
  }

  if (!validate_output_contracts(*contract)) {
    if (mla_contract_debug_enabled()) {
      std::fprintf(stderr,
                   "[mla-contract] rejecting invalid MLA output contract logical_outputs=%zu "
                   "physical_outputs=%zu\n",
                   contract->logical_outputs.size(), contract->physical_outputs.size());
    }
    contract->physical_outputs.clear();
    contract->logical_outputs.clear();
    contract->output_quant.clear();
    return;
  }

  assign_output_quant(contract, quant);
  for (std::size_t i = 0; i < contract->logical_outputs.size(); ++i) {
    const std::size_t quant_index =
        (i < contract->output_quant.size())
            ? i
            : (contract->output_quant.size() == 1U ? 0U : contract->output_quant.size());
    if (quant_index < contract->output_quant.size()) {
      contract->logical_outputs[i].quant = contract->output_quant[quant_index];
    }
  }

  if (mla_contract_debug_enabled()) {
    for (std::size_t i = 0; i < contract->physical_outputs.size(); ++i) {
      const auto& physical = contract->physical_outputs[i];
      std::fprintf(stderr,
                   "[mla-contract] published_output idx=%zu physical_index=%d segment=%s "
                   "size_bytes=%llu source_physical_index=%d source_byte_offset=%lld\n",
                   i, physical.physical_index, debug_string_or_empty(physical.segment_name),
                   static_cast<unsigned long long>(physical.size_bytes),
                   physical.source_physical_index,
                   static_cast<long long>(physical.source_byte_offset));
    }
  }
}

MlaStaticContract build_mla_static_contract_from_mpk_stage_local(
    const MpkPluginIoContract& mla, const std::vector<MpkTensorContract>& logical_outputs,
    const std::vector<MpkTensorContract>& physical_outputs, const std::string& node_name_hint,
    const std::vector<MpkTensorContract>* boundary_inputs_override) {
  MlaStaticContract contract;
  contract.node_name = node_name_hint.empty() ? mla.name : node_name_hint;
  contract.stage_id = contract.node_name;
  contract.model_path = mla.executable;
  contract.batch_size = mla.batch_size;
  contract.batch_sz_model = mla.batch_sz_model;

  auto build_input_spec = [&](const MpkTensorContract& src, const std::vector<std::int64_t>& shape,
                              const std::string& dtype) {
    TensorStaticSpec spec;
    spec.dtype = dtype.empty() ? "FP32" : normalize_tensor_dtype_token(dtype);
    spec.shape = normalize_tensor_shape_local(shape);
    assign_tensor_dims_from_shape(spec.shape, &spec);
    spec.semantic_tag = src.name.empty() ? "mla_input" : src.name;
    if (src.size_bytes > 0U) {
      spec.max_stride = to_non_negative_int(static_cast<std::int64_t>(std::min<std::size_t>(
          src.size_bytes, static_cast<std::size_t>(std::numeric_limits<int>::max()))));
    }
    return spec;
  };

  const auto& input_tensors =
      boundary_inputs_override ? *boundary_inputs_override : mla.input_tensors;
  contract.inputs.reserve(input_tensors.size());
  contract.logical_inputs.reserve(input_tensors.size());
  contract.physical_inputs.reserve(input_tensors.size());
  for (std::size_t i = 0; i < input_tensors.size(); ++i) {
    const auto& src = input_tensors[i];

    TensorStaticSpec physical = build_input_spec(src, src.mpk_shape, src.dtype);
    physical.tensor_index = static_cast<int>(i);
    contract.inputs.push_back(physical);

    PhysicalBufferStaticSpec physical_input;
    physical_input.physical_index = static_cast<int>(i);
    physical_input.allocator_index = static_cast<int>(i);
    physical_input.source_physical_index =
        src.source_physical_index >= 0
            ? src.source_physical_index
            : (src.physical_index >= 0 ? src.physical_index : static_cast<int>(i));
    physical_input.source_byte_offset = src.source_byte_offset != 0
                                            ? src.source_byte_offset
                                            : std::max<std::int64_t>(0, src.byte_offset);
    physical_input.device_kind = DeviceKind::Unknown;
    physical_input.segment_name = !src.segment_name.empty() ? src.segment_name : src.name;
    physical_input.size_bytes =
        src.size_bytes > 0U
            ? static_cast<std::uint64_t>(src.size_bytes)
            : tensor_logical_size_bytes_from_static_spec_local(contract.inputs.back());
    contract.physical_inputs.push_back(std::move(physical_input));

    const std::vector<std::int64_t> logical_shape =
        mla_input_logical_shape_from_mpk_tensor_local(src);
    const std::string& logical_dtype = !src.logical_dtype.empty() ? src.logical_dtype : src.dtype;
    TensorStaticSpec logical = build_input_spec(src, logical_shape, logical_dtype);
    logical.tensor_index = static_cast<int>(i);
    contract.logical_inputs.push_back(std::move(logical));

    if (mla_contract_debug_enabled()) {
      const auto shape_csv = [](const std::vector<std::int64_t>& shape) {
        std::string out;
        for (std::size_t idx = 0; idx < shape.size(); ++idx) {
          if (idx) {
            out += ",";
          }
          out += std::to_string(shape[idx]);
        }
        return out.empty() ? std::string("<empty>") : out;
      };
      std::fprintf(stderr,
                   "[mla-contract] input idx=%zu stage=%s src_dtype=%s src_logical_dtype=%s "
                   "src_shape=%s src_logical_shape=%s derived_logical_shape=%s physical_shape=%s "
                   "logical_shape=%s "
                   "physical_bytes=%llu physical_segment=%s "
                   "source_physical_index=%d source_byte_offset=%lld\n",
                   i, debug_string_or_empty(contract.node_name), debug_string_or_empty(src.dtype),
                   debug_string_or_empty(src.logical_dtype), shape_csv(src.mpk_shape).c_str(),
                   shape_csv(src.logical_shape).c_str(), shape_csv(logical_shape).c_str(),
                   shape_csv(contract.inputs.back().shape).c_str(),
                   shape_csv(contract.logical_inputs.back().shape).c_str(),
                   static_cast<unsigned long long>(contract.physical_inputs.back().size_bytes),
                   debug_string_or_empty(contract.physical_inputs.back().segment_name),
                   contract.physical_inputs.back().source_physical_index,
                   static_cast<long long>(contract.physical_inputs.back().source_byte_offset));
    }
  }

  build_output_contracts(&contract, physical_outputs, logical_outputs, mla.quant,
                         mla.canonical_output_dtype);

  return contract;
}

} // namespace

MlaStaticContract build_mla_static_contract_from_mpk_stage(
    const MpkPluginIoContract& mla, const std::vector<MpkTensorContract>& logical_outputs,
    const std::vector<MpkTensorContract>& physical_outputs, const std::string& node_name_hint,
    const std::vector<MpkTensorContract>* boundary_inputs_override) {
  return build_mla_static_contract_from_mpk_stage_local(mla, logical_outputs, physical_outputs,
                                                        node_name_hint, boundary_inputs_override);
}

} // namespace simaai::neat::pipeline_internal::sima
