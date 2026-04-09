#include "pipeline/internal/StageConfig.h"

#include "builder/ConfigJsonProvider.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat::stages {
namespace {

const nlohmann::json* config_json_from_group(const NodeGroup& group) {
  for (const auto& node : group.nodes()) {
    if (!node)
      continue;
    auto* provider = dynamic_cast<ConfigJsonProvider*>(node.get());
    if (!provider)
      continue;
    const nlohmann::json* cfg = provider->config_json();
    if (cfg)
      return cfg;
  }
  return nullptr;
}

int elem_size_from_string(const std::string& fmt) {
  std::string up;
  up.reserve(fmt.size());
  for (char c : fmt) {
    up.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (up.find("INT8") != std::string::npos || up.find("UINT8") != std::string::npos)
    return 1;
  if (up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos)
    return 2;
  if (up.find("INT16") != std::string::npos || up.find("UINT16") != std::string::npos)
    return 2;
  if (up.find("INT32") != std::string::npos || up.find("FP32") != std::string::npos)
    return 4;
  if (up.find("FP64") != std::string::npos)
    return 8;
  return 1;
}

std::vector<int64_t> read_int_array(const nlohmann::json& v) {
  std::vector<int64_t> out;
  if (v.is_number_integer()) {
    out.push_back(v.get<int64_t>());
    return out;
  }
  if (v.is_number()) {
    out.push_back(static_cast<int64_t>(v.get<double>()));
    return out;
  }
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_number_integer()) {
        out.push_back(entry.get<int64_t>());
      } else if (entry.is_number()) {
        out.push_back(static_cast<int64_t>(entry.get<double>()));
      }
    }
  }
  return out;
}

std::vector<std::string> read_string_array(const nlohmann::json& v) {
  std::vector<std::string> out;
  if (v.is_string()) {
    out.push_back(v.get<std::string>());
    return out;
  }
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_string()) {
        out.push_back(entry.get<std::string>());
      }
    }
  }
  return out;
}

std::string read_string_field(const nlohmann::json& v) {
  if (v.is_string())
    return v.get<std::string>();
  if (v.is_array() && !v.empty() && v[0].is_string()) {
    return v[0].get<std::string>();
  }
  return {};
}

bool read_bool_field(const nlohmann::json& v, bool def_val = false) {
  if (v.is_boolean())
    return v.get<bool>();
  if (v.is_number_integer())
    return v.get<int>() != 0;
  if (v.is_number())
    return v.get<double>() != 0.0;
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "0" || s == "false" || s == "no" || s == "off")
      return false;
    if (s == "1" || s == "true" || s == "yes" || s == "on")
      return true;
  }
  return def_val;
}

int read_first_int(const nlohmann::json& v) {
  if (v.is_number_integer())
    return v.get<int>();
  if (v.is_number())
    return static_cast<int>(v.get<double>());
  if (v.is_array()) {
    for (const auto& entry : v) {
      if (entry.is_number_integer())
        return entry.get<int>();
      if (entry.is_number())
        return static_cast<int>(entry.get<double>());
    }
  }
  return 0;
}

int read_int_field(const nlohmann::json& j, const char* key) {
  if (!j.contains(key))
    return 0;
  return read_first_int(j.at(key));
}

const nlohmann::json* params_from_config(const nlohmann::json& cfg) {
  if (cfg.contains("simaai__params") && cfg["simaai__params"].is_object()) {
    return &cfg["simaai__params"];
  }
  return &cfg;
}

const nlohmann::json* field_from_config(const nlohmann::json& cfg, const nlohmann::json& params,
                                        const char* key) {
  if (params.contains(key))
    return &params.at(key);
  if (cfg.contains(key))
    return &cfg.at(key);
  return nullptr;
}

int read_cfg_int_field(const nlohmann::json& cfg, const char* key) {
  const nlohmann::json* params = params_from_config(cfg);
  if (!params)
    return 0;
  const nlohmann::json* value = field_from_config(cfg, *params, key);
  if (!value)
    return 0;
  return read_first_int(*value);
}

std::string read_cfg_string_field(const nlohmann::json& cfg, const char* key) {
  const nlohmann::json* params = params_from_config(cfg);
  if (!params)
    return {};
  const nlohmann::json* value = field_from_config(cfg, *params, key);
  if (!value)
    return {};
  return read_string_field(*value);
}

std::string upper_copy_local(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return out;
}

TensorLayout layout_from_format(const std::string& fmt) {
  const std::string up = upper_copy_local(fmt);
  if (up.find("CHW") != std::string::npos)
    return TensorLayout::CHW;
  if (up.find("HWC") != std::string::npos)
    return TensorLayout::HWC;
  if (up.find("HW") != std::string::npos)
    return TensorLayout::HW;
  return TensorLayout::Unknown;
}

std::vector<char> dims_from_format(const std::string& fmt) {
  const std::string up = upper_copy_local(fmt);
  if (up.find("NDHWC") != std::string::npos)
    return {'N', 'D', 'H', 'W', 'C'};
  if (up.find("NDCHW") != std::string::npos)
    return {'N', 'D', 'C', 'H', 'W'};
  if (up.find("NHWC") != std::string::npos)
    return {'N', 'H', 'W', 'C'};
  if (up.find("NCHW") != std::string::npos)
    return {'N', 'C', 'H', 'W'};
  if (up.find("DHWC") != std::string::npos)
    return {'D', 'H', 'W', 'C'};
  if (up.find("DCHW") != std::string::npos)
    return {'D', 'C', 'H', 'W'};
  if (up.find("HWC") != std::string::npos)
    return {'H', 'W', 'C'};
  if (up.find("CHW") != std::string::npos)
    return {'C', 'H', 'W'};
  return {};
}

} // namespace

PreprocOutputInfo read_preproc_output_info(const NodeGroup& group) {
  PreprocOutputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;
  if (cfg->contains("tessellate")) {
    const auto& t = (*cfg)["tessellate"];
    if (t.is_boolean())
      info.tessellate = t.get<bool>();
    if (t.is_number_integer())
      info.tessellate = t.get<int>() != 0;
  }
  if (cfg->contains("output_dtype") && (*cfg)["output_dtype"].is_string()) {
    info.output_dtype = (*cfg)["output_dtype"].get<std::string>();
  }
  info.dims.width = read_int_field(*cfg, "output_width");
  info.dims.height = read_int_field(*cfg, "output_height");
  info.dims.depth = read_int_field(*cfg, "output_channels");
  if (info.dims.depth <= 0) {
    info.dims.depth = read_int_field(*cfg, "tile_channels");
  }
  if (cfg->contains("output_memory_order") && (*cfg)["output_memory_order"].is_array()) {
    for (const auto& entry : (*cfg)["output_memory_order"]) {
      if (entry.is_string()) {
        info.output_memory_order.push_back(entry.get<std::string>());
      }
    }
  }
  return info;
}

MlaOutputInfo read_mla_output_info(const NodeGroup& group) {
  MlaOutputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;
  info.data_type = read_cfg_string_field(*cfg, "data_type");
  info.output_format = read_cfg_string_field(*cfg, "output_format");
  info.layout = layout_from_format(info.output_format);
  info.dims.width = read_cfg_int_field(*cfg, "output_width");
  info.dims.height = read_cfg_int_field(*cfg, "output_height");
  info.dims.depth = read_cfg_int_field(*cfg, "output_depth");
  if (info.dims.width <= 0)
    info.dims.width = read_cfg_int_field(*cfg, "slice_width");
  if (info.dims.height <= 0)
    info.dims.height = read_cfg_int_field(*cfg, "slice_height");
  if (info.dims.depth <= 0)
    info.dims.depth = read_cfg_int_field(*cfg, "slice_depth");
  if (cfg->contains("simaai__params") && (*cfg)["simaai__params"].is_object()) {
    const auto& params = (*cfg)["simaai__params"];
    if (params.contains("outputs") && params["outputs"].is_array() && !params["outputs"].empty() &&
        params["outputs"][0].is_object()) {
      const auto& out = params["outputs"][0];
      if (out.contains("size") && out["size"].is_number_integer()) {
        info.size_bytes = out["size"].get<int64_t>();
      } else if (out.contains("size") && out["size"].is_number()) {
        info.size_bytes = static_cast<int64_t>(out["size"].get<double>());
      }
    }
  }
  return info;
}

MlaInputInfo read_mla_input_info(const NodeGroup& group) {
  MlaInputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;

  info.input_format = read_cfg_string_field(*cfg, "input_format");
  info.layout = layout_from_format(info.input_format);
  info.dims.width = read_cfg_int_field(*cfg, "input_width");
  info.dims.height = read_cfg_int_field(*cfg, "input_height");
  info.dims.depth = read_cfg_int_field(*cfg, "input_depth");
  if (info.dims.depth <= 0) {
    info.dims.depth = read_cfg_int_field(*cfg, "input_channels");
  }
  return info;
}

BoxDecodeInputInfo read_boxdecode_input_info(const NodeGroup& group) {
  BoxDecodeInputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;
  info.dims.width = read_int_field(*cfg, "input_width");
  info.dims.height = read_int_field(*cfg, "input_height");
  info.dims.depth = read_int_field(*cfg, "input_depth");
  return info;
}

BoxDecodeExpectedInfo read_boxdecode_expected_info(const NodeGroup& group) {
  BoxDecodeExpectedInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;
  if (cfg->contains("buffers") && (*cfg)["buffers"].is_object()) {
    const auto& buffers = (*cfg)["buffers"];
    if (buffers.contains("input") && buffers["input"].is_array() && !buffers["input"].empty() &&
        buffers["input"][0].is_object()) {
      const auto& in = buffers["input"][0];
      if (in.contains("size") && in["size"].is_number_integer()) {
        info.buffer_size = in["size"].get<int64_t>();
      } else if (in.contains("size") && in["size"].is_number()) {
        info.buffer_size = static_cast<int64_t>(in["size"].get<double>());
      }
    }
  }
  std::vector<int64_t> widths;
  std::vector<int64_t> heights;
  std::vector<int64_t> depths;
  if (cfg->contains("input_width"))
    widths = read_int_array((*cfg)["input_width"]);
  if (cfg->contains("input_height"))
    heights = read_int_array((*cfg)["input_height"]);
  if (cfg->contains("input_depth"))
    depths = read_int_array((*cfg)["input_depth"]);
  const size_t n = std::min(widths.size(), std::min(heights.size(), depths.size()));
  for (size_t i = 0; i < n; ++i) {
    info.total_elems += widths[i] * heights[i] * depths[i];
  }
  std::string dtype;
  if (cfg->contains("data_type")) {
    const auto& dt = (*cfg)["data_type"];
    if (dt.is_array() && !dt.empty() && dt[0].is_string()) {
      dtype = dt[0].get<std::string>();
    } else if (dt.is_string()) {
      dtype = dt.get<std::string>();
    }
  }
  info.elem_size = elem_size_from_string(dtype);
  info.total_bytes = info.total_elems * static_cast<int64_t>(info.elem_size);
  return info;
}

std::string build_boxdecode_caps_override(const NodeGroup& group) {
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return {};

  const std::vector<int64_t> widths =
      cfg->contains("input_width") ? read_int_array((*cfg)["input_width"]) : std::vector<int64_t>{};
  const std::vector<int64_t> heights = cfg->contains("input_height")
                                           ? read_int_array((*cfg)["input_height"])
                                           : std::vector<int64_t>{};
  const std::vector<int64_t> depths =
      cfg->contains("input_depth") ? read_int_array((*cfg)["input_depth"]) : std::vector<int64_t>{};
  if (widths.empty() || heights.empty() || depths.empty())
    return {};

  const std::vector<std::string> dtypes = cfg->contains("data_type")
                                              ? read_string_array((*cfg)["data_type"])
                                              : std::vector<std::string>{};
  const std::vector<int64_t> slice_w =
      cfg->contains("slice_width") ? read_int_array((*cfg)["slice_width"]) : std::vector<int64_t>{};
  const std::vector<int64_t> slice_h = cfg->contains("slice_height")
                                           ? read_int_array((*cfg)["slice_height"])
                                           : std::vector<int64_t>{};
  const std::vector<int64_t> slice_d =
      cfg->contains("slice_depth") ? read_int_array((*cfg)["slice_depth"]) : std::vector<int64_t>{};

  const size_t n = std::min(widths.size(), std::min(heights.size(), depths.size()));
  if (n == 0)
    return {};

  std::ostringstream caps;
  caps << "application/vnd.simaai.tensor,format=MLA";
  if (widths[0] > 0)
    caps << ",width=" << widths[0];
  if (heights[0] > 0)
    caps << ",height=" << heights[0];
  if (depths[0] > 0)
    caps << ",depth=" << depths[0];

  for (size_t i = 0; i < n; ++i) {
    if (i < dtypes.size() && !dtypes[i].empty()) {
      caps << ",data_type__" << i << "=" << dtypes[i];
    }
    caps << ",width__" << i << "=" << widths[i];
    caps << ",height__" << i << "=" << heights[i];
    caps << ",depth__" << i << "=" << depths[i];
    if (i < slice_w.size() && slice_w[i] > 0) {
      caps << ",slice_width__" << i << "=" << slice_w[i];
    }
    if (i < slice_h.size() && slice_h[i] > 0) {
      caps << ",slice_height__" << i << "=" << slice_h[i];
    }
    if (i < slice_d.size() && slice_d[i] > 0) {
      caps << ",slice_depth__" << i << "=" << slice_d[i];
    }
  }

  return caps.str();
}

DetessDequantOutputInfo read_detessdequant_output_info(const NodeGroup& group,
                                                       bool include_batch_axis) {
  DetessDequantOutputInfo info;
  const nlohmann::json* cfg = config_json_from_group(group);
  if (!cfg)
    return info;

  const nlohmann::json* params = params_from_config(*cfg);

  const nlohmann::json* num_in_val = field_from_config(*cfg, *params, "num_in_tensor");
  int num_in = num_in_val ? read_first_int(*num_in_val) : 0;

  const nlohmann::json* batch_val = field_from_config(*cfg, *params, "batch_size");
  int batch_size = batch_val ? read_first_int(*batch_val) : 1;
  if (batch_size <= 0)
    batch_size = 1;

  const nlohmann::json* fmt_val = field_from_config(*cfg, *params, "output_format");
  info.output_format = fmt_val ? read_string_field(*fmt_val) : std::string{};
  info.layout = layout_from_format(info.output_format);

  const nlohmann::json* fp16_val = field_from_config(*cfg, *params, "fp16_out_en");
  const bool fp16 = fp16_val ? read_bool_field(*fp16_val) : false;
  info.dtype = fp16 ? TensorDType::BFloat16 : TensorDType::Float32;

  const nlohmann::json* w_val = field_from_config(*cfg, *params, "input_width");
  const nlohmann::json* h_val = field_from_config(*cfg, *params, "input_height");
  const nlohmann::json* d_val = field_from_config(*cfg, *params, "input_depth");
  const nlohmann::json* c_val = field_from_config(*cfg, *params, "input_channels");

  const std::vector<int64_t> widths = w_val ? read_int_array(*w_val) : std::vector<int64_t>{};
  const std::vector<int64_t> heights = h_val ? read_int_array(*h_val) : std::vector<int64_t>{};
  const std::vector<int64_t> depths = d_val ? read_int_array(*d_val) : std::vector<int64_t>{};
  const std::vector<int64_t> channels = c_val ? read_int_array(*c_val) : std::vector<int64_t>{};

  const size_t max_fields =
      std::max(widths.size(), std::max(heights.size(), std::max(depths.size(), channels.size())));
  if (num_in <= 0)
    num_in = static_cast<int>(max_fields);
  if (num_in <= 0)
    return info;

  const size_t n = std::min(static_cast<size_t>(num_in), std::min(widths.size(), heights.size()));
  if (n == 0)
    return info;

  const bool add_batch = include_batch_axis && batch_size > 1;
  const std::vector<char> fmt_dims = dims_from_format(info.output_format);
  const std::size_t elem_bytes = pipeline_internal::dtype_bytes(info.dtype);

  int64_t offset = 0;
  for (size_t i = 0; i < n; ++i) {
    const int64_t w = widths[i];
    const int64_t h = heights[i];
    int64_t d = (i < depths.size()) ? depths[i] : 0;
    int64_t c = (i < channels.size()) ? channels[i] : 0;
    if (channels.empty()) {
      c = d;
      d = 0;
    }
    if (w <= 0 || h <= 0 || (c <= 0 && d <= 0)) {
      info.outputs.clear();
      return info;
    }

    std::vector<int64_t> shape;
    const auto push_dim = [&](char dim) {
      int64_t v = 0;
      switch (dim) {
      case 'N':
        if (!add_batch)
          return;
        v = batch_size;
        break;
      case 'D':
        v = d;
        break;
      case 'H':
        v = h;
        break;
      case 'W':
        v = w;
        break;
      case 'C':
        v = c;
        break;
      default:
        return;
      }
      if (v > 0)
        shape.push_back(v);
    };

    if (!fmt_dims.empty()) {
      for (char dim : fmt_dims)
        push_dim(dim);
    } else if (info.layout == TensorLayout::CHW) {
      push_dim('C');
      push_dim('H');
      push_dim('W');
    } else {
      push_dim('H');
      push_dim('W');
      push_dim('C');
    }

    if (shape.empty()) {
      info.outputs.clear();
      return info;
    }

    int64_t bytes = static_cast<int64_t>(elem_bytes);
    for (const auto dim : shape) {
      bytes *= dim;
    }

    DetessDequantTensorInfo out;
    out.shape = std::move(shape);
    out.byte_offset = offset;
    info.outputs.push_back(std::move(out));
    offset += bytes;
  }

  return info;
}

} // namespace simaai::neat::stages
