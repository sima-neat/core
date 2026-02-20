#include "pipeline/internal/sima/MlaStaticContractExtractor.h"

#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>

namespace simaai::neat::pipeline_internal::sima {
namespace {

std::string lower_copy_local(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
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

template <typename T>
std::vector<T> read_numeric_field(const nlohmann::json& root, const nlohmann::json& params,
                                  const char* key) {
  const nlohmann::json* value = find_field(root, params, key);
  if (!value)
    return {};
  return read_numeric_array_any<T>(*value);
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

std::vector<std::string> read_string_field(const nlohmann::json& root, const nlohmann::json& params,
                                           const char* key) {
  const nlohmann::json* value = find_field(root, params, key);
  if (!value)
    return {};
  return read_string_array_any(*value);
}

int to_non_negative_int(std::int64_t value) {
  if (value < 0)
    return 0;
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  return static_cast<int>(value);
}

std::string normalize_layout(const std::string& layout_raw) {
  const std::string up = upper_copy(layout_raw);
  if (up.find("NCHW") != std::string::npos || up.find("CHW") != std::string::npos)
    return "CHW";
  if (up.find("NHWC") != std::string::npos || up.find("HWC") != std::string::npos)
    return "HWC";
  if (up.find("HW") != std::string::npos)
    return "HW";
  return {};
}

std::vector<std::int64_t> shape_from_layout(int width, int height, int depth,
                                            const std::string& layout_raw) {
  if (width <= 0 || height <= 0)
    return {};
  const std::string layout = normalize_layout(layout_raw);
  if (layout == "CHW") {
    if (depth <= 0)
      return {};
    return {depth, height, width};
  }
  if (layout == "HW") {
    return {height, width};
  }
  if (depth <= 0)
    return {};
  return {height, width, depth};
}

std::string pick_str_or_default(const std::vector<std::string>& values, std::size_t index,
                                const std::string& def) {
  if (values.empty())
    return def;
  if (index < values.size())
    return values[index];
  return values.front();
}

template <typename T> T pick_or_default(const std::vector<T>& values, std::size_t index, T def) {
  if (values.empty())
    return def;
  if (index < values.size())
    return values[index];
  return values.front();
}

std::size_t tensor_count_from_fields(const std::vector<std::int64_t>& widths,
                                     const std::vector<std::int64_t>& heights,
                                     const std::vector<std::int64_t>& depths,
                                     const std::vector<std::string>& dtypes,
                                     const std::vector<std::string>& formats,
                                     const nlohmann::json& params) {
  std::size_t count = 0;
  count = std::max<std::size_t>(count, widths.size());
  count = std::max<std::size_t>(count, heights.size());
  count = std::max<std::size_t>(count, depths.size());
  count = std::max<std::size_t>(count, dtypes.size());
  count = std::max<std::size_t>(count, formats.size());
  if (params.contains("outputs") && params["outputs"].is_array()) {
    count = std::max<std::size_t>(count, params["outputs"].size());
  }
  if (count == 0)
    count = 1;
  return count;
}

} // namespace

std::optional<MlaStaticContract> extract_mla_static_contract(const nlohmann::json& config_root,
                                                             std::string* error_message) {
  if (error_message)
    error_message->clear();

  if (!config_root.is_object()) {
    if (error_message)
      *error_message = "MLA contract extraction failed: config root is not an object";
    return std::nullopt;
  }

  const nlohmann::json* params_ptr = params_or_root(config_root);
  if (!params_ptr || !params_ptr->is_object()) {
    if (error_message)
      *error_message = "MLA contract extraction failed: missing params object";
    return std::nullopt;
  }
  const nlohmann::json& params = *params_ptr;

  MlaStaticContract contract;
  if (config_root.contains("node_name") && config_root["node_name"].is_string()) {
    contract.node_name = config_root["node_name"].get<std::string>();
  }
  contract.stage_id = contract.node_name;

  const std::vector<std::int64_t> in_w =
      read_numeric_field<std::int64_t>(config_root, params, "input_width");
  const std::vector<std::int64_t> in_h =
      read_numeric_field<std::int64_t>(config_root, params, "input_height");
  std::vector<std::int64_t> in_d =
      read_numeric_field<std::int64_t>(config_root, params, "input_depth");
  if (in_d.empty()) {
    in_d = read_numeric_field<std::int64_t>(config_root, params, "input_channels");
  }
  const std::vector<std::string> in_dtype = read_string_field(config_root, params, "input_dtype");
  const std::vector<std::string> in_format = read_string_field(config_root, params, "input_format");
  const std::vector<std::int64_t> in_stride =
      read_numeric_field<std::int64_t>(config_root, params, "input_stride");

  const std::size_t input_count =
      tensor_count_from_fields(in_w, in_h, in_d, in_dtype, in_format, params);
  contract.inputs.reserve(input_count);
  for (std::size_t i = 0; i < input_count; ++i) {
    TensorStaticSpec tensor;
    tensor.tensor_index = static_cast<int>(i);

    const int width = to_non_negative_int(pick_or_default<std::int64_t>(in_w, i, 0));
    const int height = to_non_negative_int(pick_or_default<std::int64_t>(in_h, i, 0));
    const int depth = to_non_negative_int(pick_or_default<std::int64_t>(in_d, i, 0));
    tensor.dtype = pick_str_or_default(in_dtype, i, "INT8");
    tensor.layout = normalize_layout(pick_str_or_default(in_format, i, "HWC"));
    tensor.max_w = width;
    tensor.max_h = height;
    tensor.max_stride = to_non_negative_int(pick_or_default<std::int64_t>(in_stride, i, 0));
    tensor.semantic_tag = "mla_input";
    tensor.shape = shape_from_layout(width, height, depth, tensor.layout);
    contract.inputs.push_back(std::move(tensor));
  }

  std::vector<std::int64_t> out_w =
      read_numeric_field<std::int64_t>(config_root, params, "output_width");
  std::vector<std::int64_t> out_h =
      read_numeric_field<std::int64_t>(config_root, params, "output_height");
  std::vector<std::int64_t> out_d =
      read_numeric_field<std::int64_t>(config_root, params, "output_depth");
  if (out_w.empty()) {
    out_w = read_numeric_field<std::int64_t>(config_root, params, "slice_width");
  }
  if (out_h.empty()) {
    out_h = read_numeric_field<std::int64_t>(config_root, params, "slice_height");
  }
  if (out_d.empty()) {
    out_d = read_numeric_field<std::int64_t>(config_root, params, "slice_depth");
  }
  const std::vector<std::string> out_dtype = read_string_field(config_root, params, "data_type");
  std::vector<std::string> out_format = read_string_field(config_root, params, "output_format");
  if (out_format.empty()) {
    out_format = read_string_field(config_root, params, "slice_format");
  }
  const std::vector<std::int64_t> out_stride =
      read_numeric_field<std::int64_t>(config_root, params, "output_stride");

  const std::size_t output_count =
      tensor_count_from_fields(out_w, out_h, out_d, out_dtype, out_format, params);
  contract.outputs.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i) {
    TensorStaticSpec tensor;
    tensor.tensor_index = static_cast<int>(i);
    const int width = to_non_negative_int(pick_or_default<std::int64_t>(out_w, i, 0));
    const int height = to_non_negative_int(pick_or_default<std::int64_t>(out_h, i, 0));
    const int depth = to_non_negative_int(pick_or_default<std::int64_t>(out_d, i, 0));
    tensor.dtype = pick_str_or_default(out_dtype, i, "INT8");
    tensor.layout = normalize_layout(pick_str_or_default(out_format, i, "HWC"));
    tensor.max_w = width;
    tensor.max_h = height;
    tensor.max_stride = to_non_negative_int(pick_or_default<std::int64_t>(out_stride, i, 0));
    tensor.semantic_tag = "mla_output";
    tensor.shape = shape_from_layout(width, height, depth, tensor.layout);
    contract.outputs.push_back(std::move(tensor));
  }

  std::vector<double> q_scale = read_numeric_field<double>(config_root, params, "q_scale");
  if (q_scale.empty()) {
    q_scale = read_numeric_field<double>(config_root, params, "quant_scale");
  }
  std::vector<std::int64_t> q_zp = read_numeric_field<std::int64_t>(config_root, params, "q_zp");
  if (q_zp.empty()) {
    q_zp = read_numeric_field<std::int64_t>(config_root, params, "zero_point");
  }
  const std::vector<std::int64_t> q_axis =
      read_numeric_field<std::int64_t>(config_root, params, "q_axis");

  if (!q_scale.empty() || !q_zp.empty()) {
    const bool per_output_scalars =
        output_count > 1 && q_scale.size() >= output_count && q_zp.size() >= output_count;
    if (per_output_scalars) {
      for (std::size_t i = 0; i < output_count; ++i) {
        QuantStaticSpec quant;
        quant.granularity = QuantGranularity::PerTensor;
        quant.axis = -1;
        quant.scales.push_back(q_scale[i]);
        quant.zero_points.push_back(q_zp[i]);
        contract.output_quant.push_back(std::move(quant));
      }
    } else {
      QuantStaticSpec quant;
      quant.granularity = (q_scale.size() > 1 || q_zp.size() > 1) ? QuantGranularity::PerAxis
                                                                  : QuantGranularity::PerTensor;
      quant.axis = q_axis.empty() ? -1 : static_cast<int>(q_axis.front());
      quant.scales = std::move(q_scale);
      quant.zero_points = std::move(q_zp);
      contract.output_quant.push_back(std::move(quant));
    }
  }

  if (contract.outputs.empty()) {
    if (error_message) {
      *error_message = "MLA contract extraction failed: no output tensor metadata in config";
    }
    return std::nullopt;
  }

  return contract;
}

} // namespace simaai::neat::pipeline_internal::sima
