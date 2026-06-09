#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/ProcessCvuFamily.h"
#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include "builder/NodeContractConfigurable.h"
#include "builder/CompiledChildStageProvider.h"
#include "builder/NodeContractProvider.h"
#include "gst/GstHelpers.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TempJsonFileUtil.h"
#include "pipeline/internal/packedio/PackedIoAdapter.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/DequantStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"

#include <nlohmann/json.hpp>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

#include <array>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <signal.h>
#include <unistd.h>

namespace simaai::neat::internal {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kDefaultBaseOutputDir = "/data/simaai/coprocessing/models/";
constexpr const char* kDirConf = "etc";
constexpr const char* kModelPackKeepMarkerFile = ".sima_modelpack_keep";
constexpr std::uint64_t kDefaultExtractFreeReserveBytes = 16ULL * 1024ULL * 1024ULL;

constexpr const char* kDefaultPreviousNodeName = "decoder";

static std::string to_upper(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return s;
}

static std::string to_lower(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

static bool env_truthy_local(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
         std::strcmp(v, "FALSE") != 0;
}

static bool env_enabled_local(const char* name, bool default_value) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return default_value;
  return std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0 &&
         std::strcmp(v, "off") != 0 && std::strcmp(v, "OFF") != 0;
}

static bool modelpack_space_check_enabled() {
  return env_enabled_local("SIMA_NEAT_SPACE_CHECK", true);
}

static std::uint64_t parse_env_u64_bytes(const char* key, std::uint64_t fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw)
    return fallback;
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (errno != 0 || !end || *end != '\0') {
    return fallback;
  }
  return static_cast<std::uint64_t>(value);
}

static std::uint64_t modelpack_extract_free_reserve_bytes() {
  return parse_env_u64_bytes("SIMA_MPK_EXTRACT_MIN_FREE_BYTES", kDefaultExtractFreeReserveBytes);
}

static bool checked_add_u64_local(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
  if (!out)
    return false;
  if (a > std::numeric_limits<std::uint64_t>::max() - b)
    return false;
  *out = a + b;
  return true;
}

static std::uint64_t
required_modelpack_extract_bytes(const simaai::neat::internal::ModelArchiveManifest& manifest,
                                 std::uint64_t reserve_bytes) {
  std::uint64_t required = reserve_bytes;
  for (const auto& entry : manifest.entries) {
    if (entry.type != '-')
      continue;
    if (!checked_add_u64_local(required, entry.size_bytes, &required)) {
      throw std::runtime_error("ModelPack: size_limit_exceeded: extracted archive size plus "
                               "free-space reserve overflows uint64");
    }
  }
  return required;
}

static std::string format_bytes(std::uint64_t bytes) {
  std::ostringstream oss;
  constexpr double kMiB = 1024.0 * 1024.0;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  if (bytes >= static_cast<std::uint64_t>(kGiB)) {
    oss << bytes << " bytes (" << (static_cast<double>(bytes) / kGiB) << " GiB)";
  } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
    oss << bytes << " bytes (" << (static_cast<double>(bytes) / kMiB) << " MiB)";
  } else {
    oss << bytes << " bytes";
  }
  return oss.str();
}

static bool dir_has_available_space(const fs::path& dir, std::uint64_t required_available_bytes,
                                    std::uint64_t* available_out = nullptr) {
  if (!modelpack_space_check_enabled() || required_available_bytes == 0) {
    return true;
  }
  std::error_code ec;
  const auto space = fs::space(dir, ec);
  if (ec)
    return false;
  const std::uint64_t available =
      space.available > static_cast<std::uintmax_t>(std::numeric_limits<std::uint64_t>::max())
          ? std::numeric_limits<std::uint64_t>::max()
          : static_cast<std::uint64_t>(space.available);
  if (available_out)
    *available_out = available;
  return available >= required_available_bytes;
}

static bool processcvu_family_is_fused_local(std::string family) {
  family = to_lower(std::move(family));
  return family.find("tess") != std::string::npos &&
         (family.find("quant") != std::string::npos || family.find("cast") != std::string::npos);
}

static std::string canonical_processcvu_stage_name_local(const std::string& stage_name,
                                                         ExecutionStageKind kind) {
  const std::string family = pipeline_internal::sima::processcvu_graph_family_for_stage_kind(kind);
  if (family.empty() || !processcvu_family_is_fused_local(family)) {
    return stage_name;
  }
  return family;
}

static ExecutionStageKind canonical_execution_stage_kind(std::string raw) {
  raw = to_lower(std::move(raw));
  if (raw.find("detessdequant") != std::string::npos ||
      (raw.find("detess") != std::string::npos && raw.find("dequant") != std::string::npos)) {
    return ExecutionStageKind::DetessDequant;
  }
  if (raw.find("detesscast") != std::string::npos ||
      raw.find("detessellatecast") != std::string::npos ||
      (raw.find("detess") != std::string::npos && raw.find("cast") != std::string::npos)) {
    return ExecutionStageKind::DetessCast;
  }
  if (raw.find("detessellate") != std::string::npos ||
      raw.find("detessellation_transform") != std::string::npos ||
      raw.find("detess") != std::string::npos) {
    return ExecutionStageKind::Detess;
  }
  if (raw.find("dequantize") != std::string::npos ||
      raw.find("dequantization_transform") != std::string::npos ||
      raw.find("dequant") != std::string::npos) {
    return ExecutionStageKind::Dequant;
  }
  if (raw.find("boxdecode") != std::string::npos || raw.find("objectdecode") != std::string::npos) {
    return ExecutionStageKind::BoxDecode;
  }
  if (raw.find("quanttess") != std::string::npos ||
      (raw.find("quant") != std::string::npos && raw.find("tess") != std::string::npos)) {
    return ExecutionStageKind::QuantTess;
  }
  if (raw.find("casttess") != std::string::npos || raw.find("cast_tess") != std::string::npos ||
      raw.find("casttessellate") != std::string::npos ||
      (raw.find("cast") != std::string::npos && raw.find("tess") != std::string::npos)) {
    return ExecutionStageKind::CastTess;
  }
  if (raw.find("quantization_transform") != std::string::npos ||
      raw.find("quantize") != std::string::npos || raw == "quant") {
    return ExecutionStageKind::Quant;
  }
  if (raw.find("tessellation_transform") != std::string::npos ||
      raw.find("tessellate") != std::string::npos || raw == "tess") {
    return ExecutionStageKind::Tess;
  }
  if (raw.find("preproc") != std::string::npos || raw.find("preprocess") != std::string::npos) {
    return ExecutionStageKind::Preproc;
  }
  if (raw.find("cast") != std::string::npos) {
    return ExecutionStageKind::Cast;
  }
  if (raw.find("mla") != std::string::npos || raw == "infer") {
    return ExecutionStageKind::Mla;
  }
  return ExecutionStageKind::Unknown;
}

static const char* default_stage_name(ExecutionStageKind kind) {
  switch (kind) {
  case ExecutionStageKind::Preproc:
    return "preproc";
  case ExecutionStageKind::Quant:
    return "quant";
  case ExecutionStageKind::Tess:
    return "tess";
  case ExecutionStageKind::QuantTess:
    return "quanttess";
  case ExecutionStageKind::CastTess:
    return "casttess";
  case ExecutionStageKind::Mla:
    return "mla";
  case ExecutionStageKind::Detess:
    return "post_detess";
  case ExecutionStageKind::DetessCast:
    return "post_detesscast";
  case ExecutionStageKind::DetessDequant:
    return "post_detessdequant";
  case ExecutionStageKind::Dequant:
    return "post_dequant";
  case ExecutionStageKind::BoxDecode:
    return "boxdecode";
  case ExecutionStageKind::Cast:
    return "post_cast";
  case ExecutionStageKind::Unknown:
    break;
  }
  return "stage";
}

static std::string require_stage_factory(ExecutionStageKind kind) {
  const char* factory = nullptr;
  switch (kind) {
  case ExecutionStageKind::Preproc:
  case ExecutionStageKind::Quant:
  case ExecutionStageKind::Tess:
  case ExecutionStageKind::QuantTess:
  case ExecutionStageKind::CastTess:
  case ExecutionStageKind::Cast:
  case ExecutionStageKind::DetessCast:
  case ExecutionStageKind::DetessDequant:
    factory = "neatprocesscvu";
    break;
  case ExecutionStageKind::Mla:
    factory = "neatprocessmla";
    break;
  case ExecutionStageKind::Detess:
    factory = "neatdetess";
    break;
  case ExecutionStageKind::Dequant:
    factory = "neatprocesscvu";
    break;
  case ExecutionStageKind::BoxDecode:
    factory = "neatboxdecode";
    break;
  case ExecutionStageKind::Unknown:
    break;
  }
  if (!factory || !*factory) {
    throw std::runtime_error("ModelFragment: unresolved factory for typed execution stage (kind=" +
                             std::to_string(static_cast<int>(kind)) +
                             "); no GStreamer element is mapped for this stage kind");
  }
  if (!simaai::neat::element_exists(factory)) {
    throw std::runtime_error(
        std::string("ModelFragment: required NEAT factory not found: '") + factory +
        "' (typed stage kind=" + std::to_string(static_cast<int>(kind)) +
        "). Ensure the NEAT plugin .so is installed and GST_PLUGIN_PATH includes it.");
  }
  return factory;
}

static bool execution_stage_uses_processcvu_contract(ExecutionStageKind kind) {
  return kind == ExecutionStageKind::Preproc || kind == ExecutionStageKind::Quant ||
         kind == ExecutionStageKind::Tess || kind == ExecutionStageKind::QuantTess ||
         kind == ExecutionStageKind::CastTess || kind == ExecutionStageKind::Cast ||
         kind == ExecutionStageKind::Detess || kind == ExecutionStageKind::DetessCast ||
         kind == ExecutionStageKind::DetessDequant || kind == ExecutionStageKind::Dequant;
}

static std::string normalize_format(std::string fmt) {
  fmt = to_upper(fmt);
  if (fmt == "GRAY8")
    fmt = "GRAY";
  if (fmt == "I420")
    fmt = "IYUV";
  return fmt;
}

static bool output_name_looks_generic_local(std::string raw_name) {
  if (raw_name.empty()) {
    return true;
  }
  std::transform(raw_name.begin(), raw_name.end(), raw_name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return raw_name.rfind("pass_through_out_", 0U) == 0U ||
         raw_name.rfind("output_tensor_", 0U) == 0U || raw_name == "output_tensor" ||
         raw_name.rfind("output_", 0U) == 0U;
}

static std::array<float, 3> materialize3(const std::vector<float>& v, float defv) {
  if (v.empty())
    return {defv, defv, defv};
  if (v.size() == 1)
    return {v[0], v[0], v[0]};
  if (v.size() == 3)
    return {v[0], v[1], v[2]};
  throw std::invalid_argument("mean/stddev must have 0, 1, or 3 values.");
}

static std::string append_model_paths_if_exists(json& json_data, const std::string& append_path) {
  if (json_data.contains("simaai__params") && json_data["simaai__params"].contains("model_path")) {
    std::string model_path = json_data["simaai__params"]["model_path"];
    json_data["simaai__params"]["model_path"] = append_path + "/share/" + model_path;
  } else if (json_data.contains("model_info") && json_data["model_info"].contains("path")) {
    std::string model_info_path = json_data["model_info"]["path"];
    json_data["model_info"]["path"] = append_path + "/lib/" + model_info_path;
  }
  return "";
}

struct MlaRuntimeProperties {
  std::string model_path;
  int batch_size = 0;
  int batch_sz_model = 0;
};

struct MpkTensorDims {
  int width = 0;
  int height = 0;
  int depth = 0;
  std::string format = "HWC";
};

static MpkTensorDims mpk_dims_from_shape(const std::vector<std::int64_t>& in_shape) {
  std::vector<std::int64_t> shape = in_shape;
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }

  MpkTensorDims out;
  if (shape.size() >= 3U) {
    out.height = static_cast<int>(shape[shape.size() - 3U]);
    out.width = static_cast<int>(shape[shape.size() - 2U]);
    out.depth = static_cast<int>(shape[shape.size() - 1U]);
    out.format = "HWC";
    return out;
  }
  if (shape.size() == 2U) {
    out.height = static_cast<int>(shape[0]);
    out.width = static_cast<int>(shape[1]);
    out.depth = 1;
    out.format = "HW";
    return out;
  }
  if (shape.size() == 1U) {
    out.width = static_cast<int>(shape[0]);
    out.height = 1;
    out.depth = 1;
    out.format = "HW";
    return out;
  }
  return out;
}

static std::optional<MlaRuntimeProperties>
read_mla_runtime_properties_from_config(const std::string& cfg_path) {
  if (cfg_path.empty()) {
    return std::nullopt;
  }
  std::ifstream in(cfg_path);
  if (!in.is_open()) {
    return std::nullopt;
  }

  json cfg;
  try {
    in >> cfg;
  } catch (const std::exception&) {
    return std::nullopt;
  }
  if (!cfg.is_object()) {
    return std::nullopt;
  }

  const json* params = &cfg;
  if (cfg.contains("simaai__params") && cfg["simaai__params"].is_object()) {
    params = &cfg["simaai__params"];
  }

  auto read_string = [&](const char* key) -> std::string {
    if (!key || !*key) {
      return {};
    }
    if (params->contains(key) && (*params)[key].is_string()) {
      return (*params)[key].get<std::string>();
    }
    if (cfg.contains(key) && cfg[key].is_string()) {
      return cfg[key].get<std::string>();
    }
    return {};
  };

  auto read_positive_int = [&](const char* key) -> int {
    if (!key || !*key) {
      return 0;
    }
    auto read_from = [&](const json& obj) -> int {
      if (!obj.contains(key)) {
        return 0;
      }
      const auto& value = obj.at(key);
      if (value.is_number_integer()) {
        const auto raw = value.get<long long>();
        if (raw > 0 && raw <= std::numeric_limits<int>::max()) {
          return static_cast<int>(raw);
        }
      }
      if (value.is_array() && !value.empty() && value[0].is_number_integer()) {
        const auto raw = value[0].get<long long>();
        if (raw > 0 && raw <= std::numeric_limits<int>::max()) {
          return static_cast<int>(raw);
        }
      }
      return 0;
    };
    int out = read_from(*params);
    if (out > 0) {
      return out;
    }
    return read_from(cfg);
  };

  MlaRuntimeProperties props;
  props.model_path = read_string("model_path");
  props.batch_size = read_positive_int("batch_size");
  props.batch_sz_model = read_positive_int("batch_sz_model");

  if (env_truthy_local("SIMA_MLA_CONTRACT_DEBUG")) {
    const json* params_after = &cfg;
    if (cfg.contains("simaai__params") && cfg["simaai__params"].is_object()) {
      params_after = &cfg["simaai__params"];
    }

    std::ostringstream dbg;
    dbg << "[MLA-CONTRACT][ModelPack] cfg=" << cfg_path
        << " model_path=" << (props.model_path.empty() ? "<empty>" : props.model_path)
        << " batch_size=" << props.batch_size << " batch_sz_model=" << props.batch_sz_model;

    if (params_after->contains("outputs") && (*params_after)["outputs"].is_array()) {
      const auto& outputs = (*params_after)["outputs"];
      dbg << " outputs_len=" << outputs.size();
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto& out = outputs[i];
        std::string out_name;
        std::int64_t out_size = 0;
        if (out.is_object()) {
          if (out.contains("name") && out["name"].is_string()) {
            out_name = out["name"].get<std::string>();
          }
          if (out.contains("size") && out["size"].is_number()) {
            out_size = out["size"].get<std::int64_t>();
          }
        } else if (out.is_string()) {
          out_name = out.get<std::string>();
        }
        dbg << " out[" << i << "]={name=" << (out_name.empty() ? std::string("<empty>") : out_name)
            << ",size=" << out_size << "}";
      }
    } else {
      dbg << " outputs_len=0";
    }

    std::cerr << dbg.str() << "\n";
  }

  if (props.model_path.empty() && props.batch_size <= 0 && props.batch_sz_model <= 0) {
    return std::nullopt;
  }
  return props;
}

static std::optional<MlaRuntimeProperties> read_mla_runtime_properties_from_mpk_contract(
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract) {
  if (!mpk_contract.has_value()) {
    return std::nullopt;
  }

  auto resolve_package_relative_path = [&](const std::string& raw_path,
                                           bool prefer_share_dir) -> std::string {
    if (raw_path.empty()) {
      return {};
    }
    const fs::path raw(raw_path);
    if (raw.is_absolute()) {
      return raw.string();
    }

    fs::path package_root;
    if (!mpk_contract->mpk_json_path.empty()) {
      package_root = fs::path(mpk_contract->mpk_json_path).parent_path();
      if (package_root.filename() == "etc") {
        package_root = package_root.parent_path();
      }
    }

    std::vector<fs::path> candidates;
    if (!package_root.empty()) {
      if (prefer_share_dir) {
        candidates.push_back(package_root / "share" / raw);
      }
      candidates.push_back(package_root / raw);
      if (!prefer_share_dir) {
        candidates.push_back(package_root / "share" / raw);
      }
    }
    candidates.push_back(raw);

    for (const auto& candidate : candidates) {
      std::error_code ec;
      if (candidate.empty()) {
        continue;
      }
      if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
        return candidate.string();
      }
    }
    return candidates.empty() ? raw.string() : candidates.front().string();
  };

  MlaRuntimeProperties props;
  if (const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract);
      mla_stage && !mla_stage->executable.empty()) {
    props.model_path = resolve_package_relative_path(mla_stage->executable, true);
    props.batch_size = mla_stage->batch_size;
    props.batch_sz_model = mla_stage->batch_sz_model;
  }
  if (props.model_path.empty()) {
    return std::nullopt;
  }
  return props;
}

static void
apply_mla_runtime_properties_to_contract(const MlaRuntimeProperties& props,
                                         pipeline_internal::sima::MlaStaticContract* contract) {
  if (!contract) {
    return;
  }
  contract->model_path = props.model_path;
  contract->batch_size = props.batch_size;
  contract->batch_sz_model = props.batch_sz_model;
}

static CompiledTransportContract build_model_managed_transport_contract(
    const std::string& plugin_kind, const std::string& kernel_kind,
    std::optional<pipeline_internal::sima::ProcessCvuStagePayload> processcvu_payload =
        std::nullopt,
    std::optional<CompiledRuntimeContract> runtime_contract = std::nullopt) {
  pipeline_internal::sima::stagesemantics::TransportCanonicalFacts facts;
  facts.plugin_kind = plugin_kind;
  facts.kernel_kind = kernel_kind;
  facts.model_managed_stage = true;
  facts.payload_kind = processcvu_payload.has_value()
                           ? pipeline_internal::sima::StagePayloadKind::ProcessCvu
                           : pipeline_internal::sima::StagePayloadKind::None;
  facts.processcvu_payload = std::move(processcvu_payload);
  facts.runtime_contract = std::move(runtime_contract);
  return pipeline_internal::sima::stagesemantics::build_transport_compiled_contract_from_facts(
      facts);
}

static std::uint64_t tensor_physical_span_bytes(const std::vector<std::int64_t>& shape,
                                                const std::vector<std::int64_t>& stride_bytes,
                                                const std::uint64_t logical_size_bytes,
                                                const std::uint64_t elem_bytes);

static std::uint64_t
tensor_static_logical_size_bytes_local(const pipeline_internal::sima::TensorStaticSpec& tensor) {
  if (tensor.shape.empty()) {
    return tensor.max_stride > 0 ? static_cast<std::uint64_t>(tensor.max_stride) : 0U;
  }
  std::uint64_t elems = 1U;
  for (const auto dim : tensor.shape) {
    if (dim <= 0) {
      return 0U;
    }
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * pipeline_internal::sima::specbuilders::dtype_size_bytes_from_token(tensor.dtype);
}

static CompiledRuntimeContract
transport_runtime_contract_from_processcvu_compiled(const CompiledProcessCvuContract& compiled) {
  CompiledRuntimeContract runtime = pipeline_internal::sima::stagesemantics::
      build_transport_runtime_contract_from_processcvu_compiled(compiled);
  if (env_truthy_local("SIMA_TRANSPORT_CONTRACT_DEBUG")) {
    const std::string graph_family =
        pipeline_internal::sima::stagesemantics::canonical_processcvu_graph_family(
            compiled.payload.graph_family);
    std::fprintf(stderr,
                 "[transport-contract-debug] family=%s runtime_phys=%zu runtime_logical=%zu "
                 "exposed_logical=%zu output_routes=%zu first_segment=%s\n",
                 graph_family.c_str(), runtime.physical_outputs.size(),
                 runtime.logical_outputs.size(),
                 compiled.exposed_view.exposed_logical_outputs.size(), runtime.output_order.size(),
                 (!runtime.physical_outputs.empty() &&
                  !runtime.physical_outputs.front().segment_name.empty())
                     ? runtime.physical_outputs.front().segment_name.c_str()
                     : "<empty>");
  }
  return runtime;
}

static const CompiledRuntimeContract*
resolve_stage_facts_runtime_contract(const ModelFragment::StageFacts& entry,
                                     CompiledRuntimeContract* scratch) {
  if (!scratch) {
    return nullptr;
  }
  if (entry.transport_compiled.has_value()) {
    return &entry.transport_compiled->runtime_contract;
  }
  if (entry.processcvu_contract.has_value()) {
    *scratch = transport_runtime_contract_from_processcvu_compiled(*entry.processcvu_contract);
    return scratch;
  }
  if (entry.mla_compiled.has_value()) {
    return &entry.mla_compiled->runtime_contract;
  }
  if (entry.boxdecode_compiled.has_value()) {
    return &entry.boxdecode_compiled->runtime_contract;
  }
  if (entry.dequant_compiled.has_value()) {
    return &entry.dequant_compiled->runtime_contract;
  }
  return nullptr;
}

static const pipeline_internal::sima::MpkPluginIoContract* find_mpk_stage_for_execution_stage(
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract,
    const ExecutionStage& stage) {
  if (!mpk_contract.has_value()) {
    return nullptr;
  }
  if (stage.mpk_plugin_index.has_value() &&
      *stage.mpk_plugin_index < mpk_contract->plugins.size()) {
    return &mpk_contract->plugins[*stage.mpk_plugin_index];
  }
  if (!stage.stage_name.empty()) {
    if (const auto* found =
            pipeline_internal::sima::get_stage_io_contract(*mpk_contract, stage.stage_name)) {
      return found;
    }
  }
  if (!stage.plugin_id.empty()) {
    if (const auto* found =
            pipeline_internal::sima::get_stage_io_contract(*mpk_contract, stage.plugin_id)) {
      return found;
    }
  }
  return nullptr;
}

static std::string& modelpack_process_root_storage() {
  static std::string root;
  return root;
}

static bool& modelpack_process_cleanup_enabled_storage() {
  static bool enabled = true;
  return enabled;
}

static bool parse_proc_pid(const std::string& name, pid_t* out_pid) {
  if (!out_pid) {
    return false;
  }
  *out_pid = 0;
  const std::string prefix = "proc_";
  if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size()) {
    return false;
  }
  const char* raw = name.c_str() + static_cast<std::ptrdiff_t>(prefix.size());
  char* end = nullptr;
  errno = 0;
  const long pid = std::strtol(raw, &end, 10);
  if (errno != 0 || !end || *end != '\0' || pid <= 0) {
    return false;
  }
  *out_pid = static_cast<pid_t>(pid);
  return true;
}

static bool pid_is_alive(pid_t pid) {
  if (pid <= 0) {
    return false;
  }
  if (::kill(pid, 0) == 0) {
    return true;
  }
  return errno == EPERM;
}

static void cleanup_stale_modelpack_process_roots(const fs::path& base) {
  if (!pipeline_internal::env_bool("SIMA_MPK_EXTRACT_GC_STALE_PROC", true)) {
    return;
  }
  std::error_code ec;
  fs::create_directories(base, ec);
  ec.clear();
  for (const auto& entry : fs::directory_iterator(base, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    pid_t pid = 0;
    if (!parse_proc_pid(name, &pid)) {
      continue;
    }
    if (pid == ::getpid() || pid_is_alive(pid)) {
      continue;
    }
    const fs::path keep_marker = entry.path() / kModelPackKeepMarkerFile;
    if (fs::exists(keep_marker, ec) && !ec) {
      ec.clear();
      continue;
    }
    ec.clear();
    fs::remove_all(entry.path(), ec);
    ec.clear();
  }
}

static void mark_modelpack_process_root_keep(const std::string& root) {
  if (root.empty()) {
    return;
  }
  std::error_code ec;
  const fs::path marker = fs::path(root) / kModelPackKeepMarkerFile;
  if (fs::exists(marker, ec) && !ec) {
    return;
  }
  std::ofstream out(marker, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return;
  }
  out << "keep\n";
}

static bool modelpack_cleanup_enabled_for_request(bool request_enabled) {
  if (!request_enabled) {
    return false;
  }
  return pipeline_internal::env_bool("SIMA_MPK_CLEANUP_EXTRACTED", true);
}

static void cleanup_modelpack_process_root() {
  const std::string& root = modelpack_process_root_storage();
  if (root.empty()) {
    return;
  }
  if (!modelpack_process_cleanup_enabled_storage()) {
    return;
  }
  std::error_code ec;
  fs::remove_all(fs::path(root), ec);
}

static std::string modelpack_output_root(bool cleanup_enabled_request,
                                         std::uint64_t required_available_bytes) {
  if (!modelpack_cleanup_enabled_for_request(cleanup_enabled_request)) {
    modelpack_process_cleanup_enabled_storage() = false;
  }

  auto is_writable_dir = [&](const fs::path& dir, std::string* reason = nullptr) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
      if (reason)
        *reason = "create_directories failed: " + ec.message();
      return false;
    }
    cleanup_stale_modelpack_process_roots(dir);
    const fs::path probe = dir / ".sima_modelpack_write_probe";
    std::ofstream out(probe, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
      if (reason)
        *reason = "write probe failed";
      return false;
    }
    out << "ok";
    out.close();
    fs::remove(probe, ec);
    std::uint64_t available = 0;
    if (!dir_has_available_space(dir, required_available_bytes, &available)) {
      if (reason) {
        *reason = "insufficient free space: required=" + format_bytes(required_available_bytes) +
                  " available=" + format_bytes(available);
      }
      return false;
    }
    return true;
  };

  auto choose_base = [&]() -> fs::path {
    const char* env_root = std::getenv("SIMA_MPK_EXTRACT_ROOT");
    if (env_root && *env_root) {
      std::string reason;
      fs::path explicit_root(env_root);
      if (!is_writable_dir(explicit_root, &reason)) {
        throw std::runtime_error("ModelPack: output_storage_unavailable: explicit "
                                 "SIMA_MPK_EXTRACT_ROOT is not usable: " +
                                 explicit_root.string() + " (" + reason + ")");
      }
      return explicit_root;
    }

    std::vector<std::string> rejected;
    const fs::path preferred(kDefaultBaseOutputDir);
    std::string reason;
    if (is_writable_dir(preferred, &reason))
      return preferred;
    rejected.push_back(preferred.string() + " (" + reason + ")");

    const char* tmpdir = std::getenv("TMPDIR");
    fs::path tmp_base = (tmpdir && *tmpdir) ? fs::path(tmpdir) : fs::path("/tmp");
    tmp_base /= "simaai/coprocessing/models";
    reason.clear();
    if (is_writable_dir(tmp_base, &reason))
      return tmp_base;
    rejected.push_back(tmp_base.string() + " (" + reason + ")");

    fs::path cwd_base = fs::current_path() / "tmp" / "model_extract";
    reason.clear();
    if (is_writable_dir(cwd_base, &reason))
      return cwd_base;
    rejected.push_back(cwd_base.string() + " (" + reason + ")");

    std::ostringstream oss;
    oss << "ModelPack: output_storage_unavailable: no usable extraction root";
    if (required_available_bytes > 0) {
      oss << " with required free space " << format_bytes(required_available_bytes);
    }
    oss << ". Tried:";
    for (const auto& item : rejected) {
      oss << " [" << item << "]";
    }
    oss << ". Set SIMA_MPK_EXTRACT_ROOT to a filesystem with enough space or clean /data and /tmp.";
    throw std::runtime_error(oss.str());
  };

  static std::mutex root_mu;
  static std::string root;
  std::lock_guard<std::mutex> lock(root_mu);
  if (root.empty()) {
    const fs::path base = choose_base();
    const fs::path proc_root =
        base / ("proc_" + std::to_string(static_cast<long long>(::getpid())));
    std::error_code ec;
    fs::create_directories(proc_root, ec);
    if (ec) {
      throw std::runtime_error("ModelPack: output_storage_unavailable: failed to create "
                               "per-process extraction root: " +
                               proc_root.string() + " (" + ec.message() + ")");
    }
    root = proc_root.string();
    modelpack_process_root_storage() = root;
    static const bool registered = []() {
      std::atexit(cleanup_modelpack_process_root);
      return true;
    }();
    (void)registered;
  } else if (!dir_has_available_space(fs::path(root), required_available_bytes)) {
    throw std::runtime_error("ModelPack: output_storage_unavailable: selected extraction root "
                             "does not have enough free space: " +
                             root + " required=" + format_bytes(required_available_bytes));
  }
  if (!modelpack_process_cleanup_enabled_storage()) {
    mark_modelpack_process_root_keep(root);
  }
  return root;
}

static bool directory_has_json(const fs::path& dir) {
  std::error_code ec;
  if (!fs::exists(dir, ec) || ec || !fs::is_directory(dir, ec)) {
    return false;
  }
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    if (entry.path().extension() == ".json")
      return true;
  }
  return false;
}

static void write_json_file_atomic(const fs::path& path, const json& value, const char* label) {
  const std::string payload = value.dump(4);
  const fs::path parent = path.parent_path().empty() ? fs::path(".") : path.parent_path();
  std::error_code ec;
  fs::create_directories(parent, ec);
  if (ec) {
    throw std::runtime_error(std::string(label) +
                             ": output_storage_unavailable: failed to create directory for " +
                             path.string() + " (" + ec.message() + ")");
  }
  const fs::path tmp = parent / (path.filename().string() + ".tmp." +
                                 std::to_string(static_cast<long long>(::getpid())));
  {
    std::ofstream out(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error(std::string(label) +
                               ": output_storage_unavailable: failed to open temp config " +
                               tmp.string());
    }
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    out.flush();
    if (!out.good()) {
      fs::remove(tmp, ec);
      throw std::runtime_error(std::string(label) +
                               ": output_storage_unavailable: failed writing temp config " +
                               tmp.string());
    }
  }
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    throw std::runtime_error(std::string(label) +
                             ": output_storage_unavailable: failed to replace config " +
                             path.string());
  }
}

static bool extracted_layout_ready(const fs::path& package_root) {
  std::error_code ec;
  const fs::path etc_dir = package_root / "etc";
  const fs::path lib_dir = package_root / "lib";
  const fs::path share_dir = package_root / "share";
  if (!fs::exists(etc_dir, ec) || ec || !fs::is_directory(etc_dir, ec)) {
    return false;
  }
  if (!fs::exists(lib_dir, ec) || ec || !fs::is_directory(lib_dir, ec)) {
    return false;
  }
  if (!fs::exists(share_dir, ec) || ec || !fs::is_directory(share_dir, ec)) {
    return false;
  }
  return directory_has_json(etc_dir);
}

static std::string archive_cache_key(const std::string& tar_path) {
  std::error_code ec;
  fs::path p = fs::absolute(fs::path(tar_path), ec);
  if (ec) {
    p = fs::path(tar_path);
    ec.clear();
  }
  const auto size = fs::file_size(p, ec);
  if (ec) {
    return p.string();
  }
  ec.clear();
  const auto mtime = fs::last_write_time(p, ec);
  if (ec) {
    return p.string() + "|" + std::to_string(static_cast<unsigned long long>(size));
  }
  const auto stamp = mtime.time_since_epoch().count();
  return p.string() + "|" + std::to_string(static_cast<unsigned long long>(size)) + "|" +
         std::to_string(static_cast<long long>(stamp));
}

static std::unordered_map<std::string, std::string>& modelpack_extract_cache() {
  static std::unordered_map<std::string, std::string> cache;
  return cache;
}

static std::mutex& modelpack_extract_cache_mutex() {
  static std::mutex mu;
  return mu;
}

static std::string extract_and_organize(const std::string& tar_path,
                                        bool cleanup_extracted_model_data) {
  {
    std::error_code ec;
    const fs::path direct_root(tar_path);
    if (fs::exists(direct_root, ec) && !ec && fs::is_directory(direct_root, ec) && !ec &&
        extracted_layout_ready(direct_root)) {
      return direct_root.string();
    }
  }

  const std::string cache_key = archive_cache_key(tar_path);
  simaai::neat::internal::ModelArchiveLoaderOptions opt;
  // Runtime model packs may include auxiliary build/report artifacts.
  // Keep strict type validation in security/unit tests (default options),
  // but allow these extras in ModelPack runtime extraction.
  opt.reject_unsupported_file_types = false;
  opt.require_pipeline_sequence = false;
  opt.min_output_free_bytes = modelpack_extract_free_reserve_bytes();
  try {
    {
      std::lock_guard<std::mutex> lock(modelpack_extract_cache_mutex());
      auto& cache = modelpack_extract_cache();
      const auto found = cache.find(cache_key);
      if (found != cache.end() && extracted_layout_ready(fs::path(found->second))) {
        return found->second;
      }
    }

    const auto manifest = simaai::neat::internal::ModelArchiveLoader::inspect(tar_path, opt);
    const std::uint64_t required_available_bytes =
        required_modelpack_extract_bytes(manifest, opt.min_output_free_bytes);

    std::lock_guard<std::mutex> lock(modelpack_extract_cache_mutex());
    auto& cache = modelpack_extract_cache();
    const auto found = cache.find(cache_key);
    if (found != cache.end() && extracted_layout_ready(fs::path(found->second))) {
      return found->second;
    }

    const auto extracted = simaai::neat::internal::ModelArchiveLoader::extract(
        tar_path, modelpack_output_root(cleanup_extracted_model_data, required_available_bytes),
        opt);
    const fs::path target_dir(extracted.package_root);

    // Preserve existing behavior: materialize model-relative paths as absolute paths
    // anchored at extracted package root.
    try {
      for (const auto& entry : fs::directory_iterator(extracted.etc_dir)) {
        if (!entry.is_regular_file())
          continue;
        if (entry.path().extension() != ".json")
          continue;
        std::ifstream in(entry.path());
        if (!in.is_open())
          continue;
        json cfg;
        try {
          in >> cfg;
        } catch (const std::exception&) {
          continue;
        }
        append_model_paths_if_exists(cfg, target_dir.string());
        write_json_file_atomic(entry.path(), cfg, "ModelPack");
      }
    } catch (...) {
      std::error_code cleanup_ec;
      fs::remove_all(target_dir, cleanup_ec);
      throw;
    }

    cache[cache_key] = target_dir.string();
    return target_dir.string();
  } catch (const simaai::neat::internal::ModelArchiveError& e) {
    // Surface archive failures as a structured NeatError (not a flat std::runtime_error) so the
    // public Model boundary carries a machine-triage error_code, per the Model.h error contract.
    const ModelArchiveErrorClass cls = e.code();
    const char* code = (cls == ModelArchiveErrorClass::SizeLimitExceeded ||
                        cls == ModelArchiveErrorClass::OutputStorageUnavailable)
                           ? error_codes::kIoOpen
                           : error_codes::kIoParse;
    simaai::neat::pipeline_internal::error_util::throw_session_error(
        code, std::string("ModelPack: ") + model_archive_error_class_name(cls) + ": " + e.what(),
        /*pipeline_string=*/{},
        "Re-export the model archive and verify it contains a valid MPK contract.");
  }
}

static std::string make_temp_json_path(const std::string& dir, const std::string& tag);

static std::string update_input_buffers_name(const std::string& file_path,
                                             const std::string& previous_node_name) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open()) {
    return "Failed to open the JSON file.";
  }

  json json_data;
  json_file >> json_data;

  if (json_data.contains("input_buffers") && json_data["input_buffers"][0].contains("name")) {
    json_data["input_buffers"][0]["name"] = previous_node_name;
    std::ofstream updated_json_file(file_path);
    if (!updated_json_file.is_open()) {
      return "Failed to open the file for writing.";
    }
    updated_json_file << json_data.dump(4);
    return "";
  }

  return "input_buffers->name not found.";
}

static std::string rewrite_node_and_input_names(const std::string& file_path,
                                                const std::string& tag,
                                                const std::string& node_name,
                                                const std::string& previous_node_name,
                                                bool set_next_cpu, int next_cpu) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open()) {
    return file_path;
  }

  json json_data;
  json_file >> json_data;

  if (set_next_cpu && json_data.contains("simaai__params") &&
      json_data["simaai__params"].is_object()) {
    json_data["simaai__params"]["next_cpu"] = next_cpu;
  }

  if (!node_name.empty()) {
    json_data["node_name"] = node_name;
  }

  if (!previous_node_name.empty() && json_data.contains("input_buffers") &&
      json_data["input_buffers"].is_array() && !json_data["input_buffers"].empty() &&
      json_data["input_buffers"][0].is_object()) {
    json_data["input_buffers"][0]["name"] = previous_node_name;
  }

  const std::string out_path = make_temp_json_path("/tmp", tag);
  std::ofstream updated_json_file(out_path);
  if (!updated_json_file.is_open()) {
    return file_path;
  }
  updated_json_file << json_data.dump(4);
  return out_path;
}

static bool parse_mla_next_cpu_override(int& out) {
  const char* v = std::getenv("SIMA_MLA_NEXT_CPU");
  if (!v || !*v)
    return false;
  std::string s(v);
  std::string upper = to_upper(s);
  if (upper == "APU") {
    out = 0;
    return true;
  }
  if (upper == "CVU" || upper == "MLA") {
    out = 1;
    return true;
  }
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (end && *end == '\0') {
    out = static_cast<int>(val);
    return true;
  }
  return false;
}

static void update_mla_next_cpu(const std::string& file_path, int next_cpu) {
  std::ifstream json_file(file_path);
  if (!json_file.is_open())
    return;

  json json_data;
  json_file >> json_data;

  if (!json_data.contains("simaai__params") || !json_data["simaai__params"].is_object()) {
    return;
  }

  json_data["simaai__params"]["next_cpu"] = next_cpu;
  std::ofstream updated_json_file(file_path);
  if (!updated_json_file.is_open())
    return;
  updated_json_file << json_data.dump(4);
}

static std::string make_temp_json_path(const std::string& dir, const std::string& tag) {
  std::string prefix = "sima_mpk";
  if (!tag.empty()) {
    prefix += "_" + tag;
  }
  return pipeline_internal::make_temp_json_path(dir, prefix, "ModelPack");
}

static bool
mpk_quant_contract_complete(const std::optional<pipeline_internal::sima::MpkQuantContract>& quant) {
  return quant.has_value() && !quant->scales.empty() && !quant->zero_points.empty();
}

static pipeline_internal::sima::QuantStaticSpec
quant_static_spec_from_mpk_contract(const pipeline_internal::sima::MpkQuantContract& quant) {
  pipeline_internal::sima::QuantStaticSpec out;
  out.granularity = (quant.scales.size() > 1U || quant.zero_points.size() > 1U)
                        ? pipeline_internal::sima::QuantGranularity::PerAxis
                        : pipeline_internal::sima::QuantGranularity::PerTensor;
  out.axis = quant.axis;
  out.scales = quant.scales;
  out.zero_points = quant.zero_points;
  return out;
}

static std::optional<pipeline_internal::sima::MpkQuantContract>
resolve_model_managed_dequant_quant_contract(
    const pipeline_internal::sima::MpkContract& mpk_contract,
    const pipeline_internal::sima::MpkPluginIoContract& stage) {
  if (mpk_quant_contract_complete(stage.quant)) {
    return stage.quant;
  }

  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(mpk_contract);
  auto find_position =
      [&](const pipeline_internal::sima::MpkPluginIoContract* want) -> std::optional<std::size_t> {
    if (!want) {
      return std::nullopt;
    }
    for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
      const std::size_t idx = ordered[pos];
      if (idx < mpk_contract.plugins.size() && &mpk_contract.plugins[idx] == want) {
        return pos;
      }
    }
    return std::nullopt;
  };

  const auto stage_pos = find_position(&stage);
  if (!stage_pos.has_value()) {
    return std::nullopt;
  }
  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(mpk_contract);
  const auto mla_pos = find_position(mla_stage);
  const std::size_t lower_bound = mla_pos.value_or(0U);
  if (*stage_pos > lower_bound) {
    for (std::size_t pos = *stage_pos; pos > lower_bound; --pos) {
      const std::size_t idx = ordered[pos - 1U];
      if (idx >= mpk_contract.plugins.size()) {
        continue;
      }
      const auto& candidate = mpk_contract.plugins[idx];
      if (mpk_quant_contract_complete(candidate.quant)) {
        return candidate.quant;
      }
    }
  }
  if (lower_bound < ordered.size()) {
    const std::size_t idx = ordered[lower_bound];
    if (idx < mpk_contract.plugins.size() &&
        mpk_quant_contract_complete(mpk_contract.plugins[idx].quant)) {
      return mpk_contract.plugins[idx].quant;
    }
  }
  return std::nullopt;
}

static std::pair<double, std::int64_t>
require_uniform_dequant_params(const pipeline_internal::sima::MpkPluginIoContract& stage) {
  if (!stage.quant.has_value() || stage.quant->scales.empty() || stage.quant->zero_points.empty()) {
    throw std::runtime_error(
        "ModelFragment: dequant stage '" + stage.name +
        "' is missing MPK quant facts (expected non-empty 'scales' and 'zero_points'"
        " in the MPK plugin quant contract).");
  }

  const double scale = stage.quant->scales.front();
  const std::int64_t zp = stage.quant->zero_points.front();
  const auto scale_differs = [&](double candidate) { return std::abs(candidate - scale) > 1e-12; };
  const auto zp_differs = [&](std::int64_t candidate) { return candidate != zp; };
  if (std::any_of(stage.quant->scales.begin(), stage.quant->scales.end(), scale_differs) ||
      std::any_of(stage.quant->zero_points.begin(), stage.quant->zero_points.end(), zp_differs)) {
    throw std::runtime_error(
        "ModelFragment: dequant stage '" + stage.name +
        "' requires unsupported per-channel quant facts"
        " (scales_count=" +
        std::to_string(stage.quant->scales.size()) +
        ", zp_count=" + std::to_string(stage.quant->zero_points.size()) +
        "). Only uniform (single scale/zp) dequantization is supported."
        " Consider using a post-processing stage that handles per-channel dequant.");
  }
  return {scale, zp};
}

static int first_positive(std::initializer_list<int> values) {
  for (const int value : values) {
    if (value > 0) {
      return value;
    }
  }
  return 0;
}

static void require_positive_mpk_fact(int value, const std::string& context,
                                      const char* fact_name) {
  if (value > 0) {
    return;
  }
  throw std::runtime_error(context + " requires explicit MPK " + fact_name + ".");
}

static std::string require_string_mpk_fact(std::string value, const std::string& context,
                                           const char* fact_name) {
  if (!value.empty()) {
    return value;
  }
  throw std::runtime_error(context + " requires explicit MPK " + fact_name + ".");
}

static std::string normalize_dtype_token(std::string raw) {
  raw = to_upper(std::move(raw));
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw.find("FP32") != std::string::npos) {
    return "FP32";
  }
  if (raw.find("FLOAT16") != std::string::npos || raw.find("FP16") != std::string::npos) {
    return "FP16";
  }
  if (raw.find("UINT8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT32") != std::string::npos) {
    return "UINT32";
  }
  if (raw.find("INT32") != std::string::npos) {
    return "INT32";
  }
  return raw;
}

static std::vector<std::size_t>
ordered_plugin_indices(const pipeline_internal::sima::MpkContract& contract) {
  auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  std::vector<bool> seen(contract.plugins.size(), false);
  for (const std::size_t idx : ordered) {
    if (idx < seen.size()) {
      seen[idx] = true;
    }
  }
  for (std::size_t idx = 0; idx < contract.plugins.size(); ++idx) {
    if (!seen[idx]) {
      ordered.push_back(idx);
    }
  }
  return ordered;
}

static std::optional<std::size_t>
mla_rank_in_order(const pipeline_internal::sima::MpkContract& contract,
                  const std::vector<std::size_t>& ordered) {
  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return std::nullopt;
  }
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    const std::size_t idx = ordered[rank];
    if (idx < contract.plugins.size() && &contract.plugins[idx] == mla_stage) {
      return rank;
    }
  }
  return std::nullopt;
}

static std::vector<std::size_t>
collect_plugin_indices_by_kind(const pipeline_internal::sima::MpkContract& contract,
                               const std::vector<std::size_t>& ordered,
                               std::optional<std::size_t> mla_rank, bool before_mla, bool after_mla,
                               std::initializer_list<ExecutionStageKind> wanted) {
  std::vector<std::size_t> matches;
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    if (mla_rank.has_value()) {
      if (before_mla && !(rank < *mla_rank)) {
        continue;
      }
      if (after_mla && !(rank > *mla_rank)) {
        continue;
      }
    }
    const std::size_t idx = ordered[rank];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[idx];
    const ExecutionStageKind kind =
        canonical_execution_stage_kind(!stage.kernel.empty() ? stage.kernel : stage.name);
    if (std::find(wanted.begin(), wanted.end(), kind) != wanted.end()) {
      matches.push_back(idx);
    }
  }
  return matches;
}

static std::optional<std::size_t> first_index(const std::vector<std::size_t>& indices) {
  if (indices.empty()) {
    return std::nullopt;
  }
  return indices.front();
}

static std::string pick_stage_name(const pipeline_internal::sima::MpkContract& contract,
                                   const std::vector<std::size_t>& candidates,
                                   ExecutionStageKind kind) {
  for (const std::size_t idx : candidates) {
    if (idx < contract.plugins.size() && !contract.plugins[idx].name.empty()) {
      return canonical_processcvu_stage_name_local(contract.plugins[idx].name, kind);
    }
  }
  return canonical_processcvu_stage_name_local(default_stage_name(kind), kind);
}

static std::string plugin_id_for_stage_kind(ExecutionStageKind kind) {
  switch (kind) {
  case ExecutionStageKind::Mla:
    return "processmla";
  case ExecutionStageKind::BoxDecode:
    return "boxdecode";
  case ExecutionStageKind::Dequant:
  case ExecutionStageKind::Cast:
    return "processcvu";
  case ExecutionStageKind::Preproc:
  case ExecutionStageKind::Quant:
  case ExecutionStageKind::Tess:
  case ExecutionStageKind::QuantTess:
  case ExecutionStageKind::CastTess:
  case ExecutionStageKind::Detess:
  case ExecutionStageKind::DetessCast:
  case ExecutionStageKind::DetessDequant:
    return "processcvu";
  case ExecutionStageKind::Unknown:
    break;
  }
  return {};
}

static std::string processor_for_stage_kind(ExecutionStageKind kind) {
  if (kind == ExecutionStageKind::Mla) {
    return "MLA";
  }
  return "CVU";
}

static std::string kernel_for_stage_kind(ExecutionStageKind kind) {
  switch (kind) {
  case ExecutionStageKind::Preproc:
    return "preproc";
  case ExecutionStageKind::Quant:
    return "quantize";
  case ExecutionStageKind::Tess:
    return "tessellate";
  case ExecutionStageKind::QuantTess:
    return "quanttess";
  case ExecutionStageKind::CastTess:
    return "casttess";
  case ExecutionStageKind::Mla:
    return "infer";
  case ExecutionStageKind::Detess:
    return "detessellate";
  case ExecutionStageKind::DetessCast:
    return "detesscast";
  case ExecutionStageKind::DetessDequant:
    return "detessdequant";
  case ExecutionStageKind::Dequant:
    return "dequantize";
  case ExecutionStageKind::BoxDecode:
    return "boxdecode";
  case ExecutionStageKind::Cast:
    return "cast";
  case ExecutionStageKind::Unknown:
    break;
  }
  return {};
}

static std::optional<ExecutionStage>
make_pre_stage_from_contract(const pipeline_internal::sima::MpkContract& contract,
                             const std::vector<std::size_t>& ordered,
                             std::optional<std::size_t> mla_rank,
                             PipelineType requested_pipeline_type, std::size_t order_index) {
  const auto quant_indices =
      collect_plugin_indices_by_kind(contract, ordered, mla_rank, true, false,
                                     {ExecutionStageKind::Quant, ExecutionStageKind::QuantTess});
  const auto tess_indices = collect_plugin_indices_by_kind(
      contract, ordered, mla_rank, true, false,
      {ExecutionStageKind::Tess, ExecutionStageKind::QuantTess, ExecutionStageKind::CastTess});
  const auto cast_indices =
      collect_plugin_indices_by_kind(contract, ordered, mla_rank, true, false,
                                     {ExecutionStageKind::Cast, ExecutionStageKind::CastTess});
  const auto preproc_indices = collect_plugin_indices_by_kind(contract, ordered, mla_rank, true,
                                                              false, {ExecutionStageKind::Preproc});

  const bool has_pre_any = !quant_indices.empty() || !tess_indices.empty() ||
                           !cast_indices.empty() || !preproc_indices.empty();
  if (!has_pre_any) {
    return std::nullopt;
  }

  ExecutionStageKind kind = ExecutionStageKind::Unknown;
  switch (requested_pipeline_type) {
  case PipelineType::Preproc:
    kind = ExecutionStageKind::Preproc;
    break;
  case PipelineType::Quant:
    if (!quant_indices.empty()) {
      kind = ExecutionStageKind::Quant;
    }
    break;
  case PipelineType::Tess:
    if (!tess_indices.empty()) {
      kind = ExecutionStageKind::Tess;
    }
    break;
  case PipelineType::QuantTess:
    if (!quant_indices.empty() && !tess_indices.empty()) {
      kind = ExecutionStageKind::QuantTess;
    }
    break;
  case PipelineType::CastTess:
    if (!cast_indices.empty() && !tess_indices.empty()) {
      kind = ExecutionStageKind::CastTess;
    }
    break;
  case PipelineType::Cast:
    if (!cast_indices.empty()) {
      kind = ExecutionStageKind::Cast;
    }
    break;
  }
  if (kind == ExecutionStageKind::Unknown) {
    return std::nullopt;
  }

  std::vector<std::size_t> candidates;
  if (kind == ExecutionStageKind::Preproc) {
    candidates = !preproc_indices.empty() ? preproc_indices
                                          : (!tess_indices.empty() ? tess_indices : quant_indices);
  } else if (kind == ExecutionStageKind::Quant) {
    candidates = quant_indices;
  } else if (kind == ExecutionStageKind::Tess) {
    candidates = tess_indices;
  } else if (kind == ExecutionStageKind::QuantTess) {
    candidates = !tess_indices.empty() ? tess_indices : quant_indices;
  } else if (kind == ExecutionStageKind::CastTess) {
    candidates = !tess_indices.empty() ? tess_indices : cast_indices;
  } else if (kind == ExecutionStageKind::Cast) {
    candidates = cast_indices;
  }

  ExecutionStage stage;
  stage.order_index = order_index;
  stage.mpk_plugin_index = first_index(candidates);
  stage.stage_name = pick_stage_name(contract, candidates, kind);
  stage.factory_name = require_stage_factory(kind);
  stage.plugin_id = plugin_id_for_stage_kind(kind);
  stage.processor = processor_for_stage_kind(kind);
  stage.kernel = kernel_for_stage_kind(kind);
  stage.kind = kind;
  return stage;
}

static std::optional<ExecutionStage>
make_mla_stage_from_contract(const pipeline_internal::sima::MpkContract& contract,
                             std::size_t order_index) {
  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(contract);
  if (!mla_stage) {
    return std::nullopt;
  }
  const auto mla_idx =
      pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, mla_stage->name);
  ExecutionStage stage;
  stage.order_index = order_index;
  stage.mpk_plugin_index = mla_idx;
  stage.stage_name = !mla_stage->name.empty()
                         ? mla_stage->name
                         : std::string(default_stage_name(ExecutionStageKind::Mla));
  stage.factory_name = require_stage_factory(ExecutionStageKind::Mla);
  stage.plugin_id = plugin_id_for_stage_kind(ExecutionStageKind::Mla);
  stage.processor = processor_for_stage_kind(ExecutionStageKind::Mla);
  stage.kernel = kernel_for_stage_kind(ExecutionStageKind::Mla);
  stage.kind = ExecutionStageKind::Mla;
  return stage;
}

static std::vector<ExecutionStage> make_post_stages_from_contract(
    const pipeline_internal::sima::MpkContract& contract, const std::vector<std::size_t>& ordered,
    std::optional<std::size_t> mla_rank,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& route_flags,
    std::size_t order_index) {
  std::vector<ExecutionStage> stages;
  const auto detess_indices =
      collect_plugin_indices_by_kind(contract, ordered, mla_rank, false, true,
                                     {ExecutionStageKind::Detess, ExecutionStageKind::DetessCast,
                                      ExecutionStageKind::DetessDequant});
  const auto dequant_indices = collect_plugin_indices_by_kind(
      contract, ordered, mla_rank, false, true,
      {ExecutionStageKind::Dequant, ExecutionStageKind::DetessDequant});
  const auto boxdecode_indices = collect_plugin_indices_by_kind(
      contract, ordered, mla_rank, false, true, {ExecutionStageKind::BoxDecode});
  const auto cast_indices = collect_plugin_indices_by_kind(contract, ordered, mla_rank, false, true,
                                                           {ExecutionStageKind::Cast});

  ExecutionStageKind kind = ExecutionStageKind::Unknown;
  std::vector<std::size_t> candidates;
  const bool route_requests_boxdecode = route_flags.has_value() && route_flags->boxdecode_selected;
  if (!boxdecode_indices.empty() || route_requests_boxdecode) {
    kind = ExecutionStageKind::BoxDecode;
    if (!boxdecode_indices.empty()) {
      candidates = boxdecode_indices;
    } else {
      candidates.clear();
    }
  } else if (!detess_indices.empty() && !cast_indices.empty() && dequant_indices.empty()) {
    kind = ExecutionStageKind::DetessCast;
    candidates = detess_indices;
  } else if (!detess_indices.empty() && !dequant_indices.empty()) {
    kind = ExecutionStageKind::DetessDequant;
    candidates = !dequant_indices.empty() ? dequant_indices : detess_indices;
  } else if (!detess_indices.empty()) {
    kind = ExecutionStageKind::Detess;
    candidates = detess_indices;
  } else if (!dequant_indices.empty()) {
    kind = ExecutionStageKind::Dequant;
    candidates = dequant_indices;
  } else if (!cast_indices.empty()) {
    kind = ExecutionStageKind::Cast;
    candidates = cast_indices;
  }
  if (kind == ExecutionStageKind::Unknown) {
    return stages;
  }

  ExecutionStage stage;
  stage.order_index = order_index;
  stage.mpk_plugin_index = first_index(candidates);
  stage.stage_name = pick_stage_name(contract, candidates, kind);
  stage.factory_name = require_stage_factory(kind);
  stage.plugin_id = plugin_id_for_stage_kind(kind);
  stage.processor = processor_for_stage_kind(kind);
  stage.kernel = kernel_for_stage_kind(kind);
  stage.kind = kind;
  stages.push_back(std::move(stage));
  return stages;
}

static ExecutionPlan build_execution_plan_from_mpk_contract(
    const pipeline_internal::sima::MpkContract& contract, PipelineType requested_pipeline_type,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& route_flags) {
  ExecutionPlan plan;
  const auto ordered = ordered_plugin_indices(contract);
  const auto mla_rank = mla_rank_in_order(contract, ordered);
  std::size_t stage_order = 0U;
  if (const auto pre = make_pre_stage_from_contract(contract, ordered, mla_rank,
                                                    requested_pipeline_type, stage_order);
      pre.has_value()) {
    plan.pre.push_back(*pre);
    ++stage_order;
  }
  if (const auto mla = make_mla_stage_from_contract(contract, stage_order); mla.has_value()) {
    plan.infer.push_back(*mla);
    ++stage_order;
  }
  const auto post =
      make_post_stages_from_contract(contract, ordered, mla_rank, route_flags, stage_order);
  if (!post.empty()) {
    plan.post.insert(plan.post.end(), post.begin(), post.end());
  }
  return plan;
}

static std::string preferred_tensor_dtype(const pipeline_internal::sima::MpkTensorContract& tensor,
                                          const std::string& fallback = {}) {
  if (!tensor.logical_dtype.empty()) {
    return normalize_dtype_token(tensor.logical_dtype);
  }
  if (!tensor.dtype.empty()) {
    return normalize_dtype_token(tensor.dtype);
  }
  const auto& shape = !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape;
  if (!shape.empty() && tensor.size_bytes > 0U) {
    std::uint64_t elements = 1U;
    for (const auto dim : shape) {
      if (dim <= 0) {
        elements = 0U;
        break;
      }
      const auto u_dim = static_cast<std::uint64_t>(dim);
      if (elements > std::numeric_limits<std::uint64_t>::max() / u_dim) {
        elements = 0U;
        break;
      }
      elements *= u_dim;
    }
    if (elements > 0U && tensor.size_bytes % elements == 0U) {
      const auto bytes_per_element = tensor.size_bytes / elements;
      if (bytes_per_element == 4U) {
        return "FP32";
      }
      if (bytes_per_element == 2U) {
        return "BF16";
      }
      if (bytes_per_element == 1U) {
        return "INT8";
      }
    }
  }
  return normalize_dtype_token(fallback);
}

static std::uint64_t packed_tensor_size_bytes(const std::vector<std::int64_t>& shape,
                                              const std::string& dtype) {
  if (shape.empty()) {
    return 0U;
  }
  std::uint64_t elems = 1U;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems *
         pipeline_internal::sima::stagesemantics::processcvu_dtype_size_bytes_from_token(dtype);
}

static std::uint64_t tensor_physical_span_bytes(const std::vector<std::int64_t>& shape,
                                                const std::vector<std::int64_t>& stride_bytes,
                                                const std::uint64_t logical_size_bytes,
                                                const std::uint64_t elem_bytes) {
  if (shape.empty() || stride_bytes.empty() || shape.size() != stride_bytes.size() ||
      elem_bytes == 0U) {
    return logical_size_bytes;
  }

  std::uint64_t max_offset = 0U;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 || stride_bytes[i] < 0) {
      return logical_size_bytes;
    }
    if (shape[i] == 1) {
      continue;
    }
    const auto dim = static_cast<std::uint64_t>(shape[i] - 1);
    const auto stride = static_cast<std::uint64_t>(stride_bytes[i]);
    const std::uint64_t delta = dim * stride;
    if (dim > 0U && delta / dim != stride) {
      return logical_size_bytes;
    }
    if (max_offset > (std::numeric_limits<std::uint64_t>::max() - delta)) {
      return logical_size_bytes;
    }
    max_offset += delta;
  }
  if (max_offset > (std::numeric_limits<std::uint64_t>::max() - elem_bytes)) {
    return logical_size_bytes;
  }
  return std::max(logical_size_bytes, max_offset + elem_bytes);
}

static std::uint64_t
preferred_mpk_tensor_size_bytes(const pipeline_internal::sima::MpkTensorContract& tensor,
                                const std::string& dtype) {
  if (tensor.size_bytes > 0U) {
    return static_cast<std::uint64_t>(tensor.size_bytes);
  }
  if (!tensor.logical_shape.empty()) {
    return packed_tensor_size_bytes(tensor.logical_shape, dtype);
  }
  if (tensor.shape_semantics == pipeline_internal::sima::MpkShapeSemantics::Geometry) {
    return packed_tensor_size_bytes(tensor.mpk_shape, dtype);
  }
  return 0U;
}

static int positive_tile_channels(const pipeline_internal::sima::MpkPluginIoContract& stage) {
  if (!stage.slice_shape.empty()) {
    int ch = static_cast<int>(stage.slice_shape.back());
    if (ch > 0)
      return ch;
    // fall back to depth dim if present (4-element shape)
    if (stage.slice_shape.size() >= 4) {
      int d = static_cast<int>(stage.slice_shape[0]);
      if (d > 0)
        return d;
    }
  }
  if (!stage.input_tensors.empty()) {
    const auto& shape = !stage.input_tensors.front().logical_shape.empty()
                            ? stage.input_tensors.front().logical_shape
                            : stage.input_tensors.front().mpk_shape;
    if (!shape.empty()) {
      return static_cast<int>(shape.back());
    }
  }
  return 0;
}

static int logical_channels_from_dims(const MpkTensorDims& dims) {
  return to_upper(dims.format) == "HW" ? 1 : std::max(1, dims.depth);
}

static int logical_depth_from_dims(const MpkTensorDims& dims) {
  const std::string format = to_upper(dims.format);
  if (format == "HW" || format == "HWC") {
    return 1;
  }
  return logical_channels_from_dims(dims);
}

static const pipeline_internal::sima::MpkTensorContract*
terminal_output_tensor_for_index(const pipeline_internal::sima::MpkPluginIoContract* terminal_stage,
                                 std::size_t index, std::size_t expected_count) {
  if (!terminal_stage || terminal_stage->output_tensors.empty()) {
    return nullptr;
  }
  if (terminal_stage->output_tensors.size() != expected_count || index >= expected_count) {
    return nullptr;
  }
  return &terminal_stage->output_tensors[index];
}

static const pipeline_internal::sima::MpkPluginIoContract*
find_pre_stage_for_family(const pipeline_internal::sima::MpkContract& contract,
                          std::initializer_list<ExecutionStageKind> preferred) {
  const auto ordered = ordered_plugin_indices(contract);
  const auto mla_rank = mla_rank_in_order(contract, ordered);
  for (const ExecutionStageKind kind : preferred) {
    const auto matches =
        collect_plugin_indices_by_kind(contract, ordered, mla_rank, true, false, {kind});
    if (!matches.empty()) {
      return &contract.plugins[matches.front()];
    }
  }
  return nullptr;
}

static std::vector<const pipeline_internal::sima::MpkPluginIoContract*>
collect_post_stages_for_family(const pipeline_internal::sima::MpkContract& contract,
                               std::initializer_list<ExecutionStageKind> preferred) {
  const auto ordered = ordered_plugin_indices(contract);
  const auto mla_rank = mla_rank_in_order(contract, ordered);
  std::vector<const pipeline_internal::sima::MpkPluginIoContract*> matches;
  for (const std::size_t idx : ordered) {
    std::size_t rank = 0U;
    while (rank < ordered.size() && ordered[rank] != idx) {
      ++rank;
    }
    if (mla_rank.has_value() && !(rank > *mla_rank)) {
      continue;
    }
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[idx];
    const ExecutionStageKind kind =
        canonical_execution_stage_kind(!stage.kernel.empty() ? stage.kernel : stage.name);
    if (std::find(preferred.begin(), preferred.end(), kind) == preferred.end()) {
      continue;
    }
    matches.push_back(&stage);
  }
  return matches;
}

static const pipeline_internal::sima::MpkPluginIoContract* find_terminal_stage_after_outputs(
    const pipeline_internal::sima::MpkContract& contract,
    const std::vector<const pipeline_internal::sima::MpkPluginIoContract*>& producers);

struct DequantPublishedPhysicalInput {
  int local_physical_index = -1;
  int upstream_physical_index = -1;
  std::int64_t upstream_source_offset = 0;
  std::string segment_name;
  std::uint64_t size_bytes = 0U;
};

static std::string resolve_dequant_published_segment_name(
    const pipeline_internal::sima::MpkTensorContract& published_input, std::size_t fallback_index) {
  if (!published_input.segment_name.empty()) {
    return published_input.segment_name;
  }
  if (!published_input.name.empty()) {
    return published_input.name;
  }
  return "input_" + std::to_string(fallback_index);
}

static int resolve_dequant_boundary_physical_index(
    const pipeline_internal::sima::MpkTensorContract& published_input, std::size_t fallback_index) {
  if (published_input.materialization_kind ==
          pipeline_internal::sima::MpkTensorMaterializationKind::OffsetView &&
      published_input.source_physical_index >= 0) {
    return published_input.source_physical_index;
  }
  if (published_input.physical_index >= 0) {
    return published_input.physical_index;
  }
  if (published_input.source_physical_index >= 0) {
    return published_input.source_physical_index;
  }
  return static_cast<int>(fallback_index);
}

static DequantPublishedPhysicalInput*
find_dequant_physical_input(std::vector<DequantPublishedPhysicalInput>* physical_inputs,
                            int upstream_physical_index, std::int64_t upstream_source_offset) {
  if (!physical_inputs) {
    return nullptr;
  }
  auto it = std::find_if(physical_inputs->begin(), physical_inputs->end(),
                         [&](const DequantPublishedPhysicalInput& input) {
                           return input.upstream_physical_index == upstream_physical_index &&
                                  input.upstream_source_offset == upstream_source_offset;
                         });
  return it == physical_inputs->end() ? nullptr : &(*it);
}

static std::optional<std::size_t>
find_contract_plugin_index_local(const pipeline_internal::sima::MpkContract& contract,
                                 const pipeline_internal::sima::MpkPluginIoContract& stage) {
  for (std::size_t i = 0; i < contract.plugins.size(); ++i) {
    if (&contract.plugins[i] == &stage) {
      return i;
    }
  }
  if (!stage.plugin_id.empty()) {
    return pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, stage.plugin_id);
  }
  return pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, stage.name);
}

static const pipeline_internal::sima::MpkContractEdge*
find_stage_input_edge_local(const pipeline_internal::sima::MpkContract& contract,
                            std::size_t dst_plugin_index, int dst_input_index) {
  const pipeline_internal::sima::MpkContractEdge* fallback = nullptr;
  for (const auto& edge : contract.edges) {
    if (edge.dst_plugin_index != dst_plugin_index) {
      continue;
    }
    if (edge.dst_input_index == dst_input_index) {
      return &edge;
    }
    if (!fallback && dst_input_index < 0) {
      fallback = &edge;
    }
  }
  return fallback;
}

static std::vector<std::int64_t>
contiguous_stride_bytes_local(const std::vector<std::int64_t>& shape, const std::string& dtype) {
  std::vector<std::int64_t> strides;
  if (shape.empty()) {
    return strides;
  }
  strides.assign(shape.size(), 0);
  std::int64_t running = static_cast<std::int64_t>(
      pipeline_internal::sima::specbuilders::dtype_size_bytes_from_token(dtype));
  if (running <= 0) {
    return {};
  }
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

static std::vector<std::int64_t>
normalize_stride_rank_to_shape_local(const std::vector<std::int64_t>& strides,
                                     const std::vector<std::int64_t>& source_shape,
                                     const std::vector<std::int64_t>& target_shape) {
  return pipeline_internal::normalize_strides_rank_to_shape(strides, source_shape, target_shape);
}

static std::vector<std::int64_t>
normalize_view_stride_to_shape_local(const std::vector<std::int64_t>& strides,
                                     const std::vector<std::int64_t>& source_shape,
                                     const std::vector<std::int64_t>& alternate_source_shape,
                                     const std::vector<std::int64_t>& target_shape) {
  if (strides.empty() || target_shape.empty() || strides.size() == target_shape.size()) {
    return strides;
  }

  auto normalized = normalize_stride_rank_to_shape_local(strides, source_shape, target_shape);
  if (normalized.size() == target_shape.size()) {
    return normalized;
  }
  normalized = normalize_stride_rank_to_shape_local(strides, alternate_source_shape, target_shape);
  if (normalized.size() == target_shape.size()) {
    return normalized;
  }

  // Some MPK MLA tessellation views carry a leading singleton/batch stride even
  // after the logical tensor shape has been projected to the runtime tensor rank.
  // When the source rank is unavailable, preserving the suffix strides is the
  // tensor-view equivalent of dropping those leading singleton dimensions.  This
  // keeps the consumer strided over the packed parent buffer instead of silently
  // falling back to a dense copy/interpretation.
  return pipeline_internal::normalize_strides_rank_to_shape(strides, {}, target_shape, true);
}

static std::int64_t
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

static std::int64_t cumulative_mla_output_source_offset_bytes_local(
    const pipeline_internal::sima::MpkContract& contract, int output_index) {
  if (output_index <= 0) {
    return 0;
  }
  const auto* mla_outputs = pipeline_internal::sima::get_mla_outputs_contract(contract);
  if (!mla_outputs) {
    return 0;
  }
  std::uint64_t running = 0U;
  const std::size_t end =
      std::min<std::size_t>(static_cast<std::size_t>(output_index), mla_outputs->size());
  for (std::size_t i = 0; i < end; ++i) {
    const auto slot_size = static_cast<std::uint64_t>((*mla_outputs)[i].size_bytes);
    if (running > std::numeric_limits<std::uint64_t>::max() - slot_size) {
      return std::numeric_limits<std::int64_t>::max();
    }
    running += slot_size;
  }
  if (running > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(running);
}

static std::uint64_t
mla_output_slot_size_bytes_local(const pipeline_internal::sima::MpkContract& contract,
                                 int output_index) {
  if (output_index < 0) {
    return 0U;
  }
  const auto mla_outputs =
      pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(contract);
  const auto index = static_cast<std::size_t>(output_index);
  if (index >= mla_outputs.size()) {
    return 0U;
  }
  return static_cast<std::uint64_t>(mla_outputs[index].size_bytes);
}

struct ResolvedDequantInputView {
  int upstream_physical_index = -1;
  std::int64_t upstream_source_offset = 0;
  std::int64_t logical_byte_offset = 0;
  std::vector<std::int64_t> stride_bytes;
  std::uint64_t physical_span_bytes = 0U;
};

static std::optional<ResolvedDequantInputView> resolve_slice_backed_dequant_input_view_local(
    const pipeline_internal::sima::MpkContract& contract,
    const pipeline_internal::sima::MpkPluginIoContract& stage,
    const pipeline_internal::sima::MpkTensorContract& published_input,
    const std::vector<std::int64_t>& input_shape, const std::string& input_dtype,
    std::uint64_t logical_size_bytes, std::size_t fallback_index) {
  const auto stage_index = find_contract_plugin_index_local(contract, stage);
  if (!stage_index.has_value()) {
    return std::nullopt;
  }
  const auto* input_edge = find_stage_input_edge_local(contract, *stage_index, 0);
  if (!input_edge || input_edge->src_plugin_index >= contract.plugins.size()) {
    return std::nullopt;
  }
  const auto& producer = contract.plugins[input_edge->src_plugin_index];
  const std::string producer_token =
      to_upper(!producer.kernel.empty() ? producer.kernel : producer.name);
  if (producer_token.find("SLICE") == std::string::npos || producer.input_tensors.empty()) {
    return std::nullopt;
  }

  const auto& parent_tensor = producer.input_tensors.front();
  const std::vector<std::int64_t> parent_shape =
      !parent_tensor.logical_shape.empty() ? parent_tensor.logical_shape : parent_tensor.mpk_shape;
  if (parent_shape.empty()) {
    return std::nullopt;
  }

  const auto parent_stride_bytes = contiguous_stride_bytes_local(parent_shape, input_dtype);
  if (parent_stride_bytes.empty()) {
    return std::nullopt;
  }

  const auto logical_stride_bytes =
      normalize_stride_rank_to_shape_local(parent_stride_bytes, parent_shape, input_shape);
  const std::int64_t slice_byte_offset =
      projected_slice_begin_offset_bytes_local(producer.slice_begin, parent_stride_bytes);
  if (slice_byte_offset == std::numeric_limits<std::int64_t>::max()) {
    return std::nullopt;
  }

  const int upstream_physical_index =
      resolve_dequant_boundary_physical_index(published_input, fallback_index);
  const std::int64_t upstream_source_offset =
      published_input.source_byte_offset > 0
          ? published_input.source_byte_offset
          : cumulative_mla_output_source_offset_bytes_local(contract, upstream_physical_index);
  if (upstream_source_offset < 0 ||
      upstream_source_offset == std::numeric_limits<std::int64_t>::max()) {
    return std::nullopt;
  }

  const std::uint64_t physical_span_bytes = tensor_physical_span_bytes(
      input_shape, logical_stride_bytes, logical_size_bytes,
      pipeline_internal::sima::specbuilders::dtype_size_bytes_from_token(input_dtype));
  if (physical_span_bytes == 0U) {
    return std::nullopt;
  }

  return ResolvedDequantInputView{
      upstream_physical_index, upstream_source_offset, slice_byte_offset,
      logical_stride_bytes,    physical_span_bytes,
  };
}

static std::optional<CompiledDequantContract>
build_model_managed_dequant_compiled_contract_from_mpk(
    const pipeline_internal::sima::MpkContract& contract, std::string* err = nullptr) {
  const auto dequant_stages =
      collect_post_stages_for_family(contract, {ExecutionStageKind::Dequant});
  if (dequant_stages.empty()) {
    if (err) {
      *err = "missing_dequant_post_stages";
    }
    return std::nullopt;
  }

  const auto mla_published_outputs =
      pipeline_internal::sima::get_mla_published_outputs_contract(contract);
  if (mla_published_outputs.size() != dequant_stages.size()) {
    if (err) {
      *err = "mla_dequant_head_count_mismatch";
    }
    return std::nullopt;
  }

  CompiledDequantContract compiled;
  compiled.runtime_contract.plugin_kind = "dequant";
  const auto* terminal_stage = find_terminal_stage_after_outputs(contract, dequant_stages);
  std::uint64_t packed_output_total_bytes = 0U;
  std::vector<DequantPublishedPhysicalInput> physical_inputs;
  physical_inputs.reserve(mla_published_outputs.size());

  for (std::size_t i = 0; i < dequant_stages.size(); ++i) {
    const auto& stage = *dequant_stages[i];
    if (stage.input_tensors.empty() || stage.output_tensors.empty()) {
      if (err) {
        *err = "dequant_stage_missing_tensor_metadata";
      }
      return std::nullopt;
    }

    const auto quant_contract = resolve_model_managed_dequant_quant_contract(contract, stage);
    if (!quant_contract.has_value() || quant_contract->scales.empty() ||
        quant_contract->zero_points.empty() || quant_contract->scales.front() <= 0.0) {
      if (err) {
        *err = "missing_dequant_quant_contract";
      }
      return std::nullopt;
    }
    const double scale = quant_contract->scales.front();
    const std::int64_t zp = quant_contract->zero_points.front();
    const auto scale_differs = [&](double candidate) {
      return std::abs(candidate - scale) > 1e-12;
    };
    const auto zp_differs = [&](std::int64_t candidate) { return candidate != zp; };
    if (std::any_of(quant_contract->scales.begin(), quant_contract->scales.end(), scale_differs) ||
        std::any_of(quant_contract->zero_points.begin(), quant_contract->zero_points.end(),
                    zp_differs)) {
      if (err) {
        *err = "unsupported_per_channel_dequant_quant";
      }
      return std::nullopt;
    }

    pipeline_internal::sima::QuantStaticSpec input_quant;
    input_quant.granularity = pipeline_internal::sima::QuantGranularity::PerTensor;
    input_quant.axis = -1;
    input_quant.scales = {scale};
    input_quant.zero_points = {zp};

    const auto& published_input = mla_published_outputs[i];
    const auto& input_tensor = stage.input_tensors.front();
    const auto& output_tensor = stage.output_tensors.front();
    const auto* terminal_output_tensor =
        terminal_output_tensor_for_index(terminal_stage, i, dequant_stages.size());

    const std::string input_dtype = preferred_tensor_dtype(
        published_input, preferred_tensor_dtype(input_tensor, stage.canonical_input_dtype));
    std::string output_dtype = normalize_dtype_token(
        terminal_output_tensor && !terminal_output_tensor->logical_dtype.empty()
            ? terminal_output_tensor->logical_dtype
        : terminal_output_tensor && !terminal_output_tensor->dtype.empty()
            ? terminal_output_tensor->dtype
            : preferred_tensor_dtype(output_tensor, stage.canonical_output_dtype));
    if (output_dtype != "FP16" && output_dtype != "FP32") {
      output_dtype = "FP32";
    }
    const auto input_dims = mpk_dims_from_shape(
        !input_tensor.logical_shape.empty() ? input_tensor.logical_shape : input_tensor.mpk_shape);
    const auto output_dims =
        terminal_output_tensor
            ? mpk_dims_from_shape(terminal_output_tensor->logical_shape)
            : mpk_dims_from_shape(!output_tensor.logical_shape.empty() ? output_tensor.logical_shape
                                                                       : output_tensor.mpk_shape);
    const std::string input_layout =
        input_dims.format.empty() ? std::string{} : normalize_format(input_dims.format);
    const std::string output_layout =
        output_dims.format.empty() ? std::string{} : normalize_format(output_dims.format);
    const std::vector<std::int64_t> input_shape =
        !input_tensor.logical_shape.empty()      ? input_tensor.logical_shape
        : !input_tensor.mpk_shape.empty()        ? input_tensor.mpk_shape
        : !published_input.logical_shape.empty() ? published_input.logical_shape
                                                 : published_input.mpk_shape;
    const std::vector<std::int64_t> output_shape =
        terminal_output_tensor && !terminal_output_tensor->logical_shape.empty()
            ? terminal_output_tensor->logical_shape
        : !output_tensor.logical_shape.empty() ? output_tensor.logical_shape
                                               : output_tensor.mpk_shape;
    if (input_shape.empty() || output_shape.empty() || input_dtype.empty() ||
        output_dtype.empty()) {
      if (err) {
        *err = "dequant_stage_missing_shape_or_dtype";
      }
      return std::nullopt;
    }
    if (input_layout.empty() || output_layout.empty()) {
      if (err) {
        *err = "dequant_stage_missing_explicit_layout_semantics";
      }
      return std::nullopt;
    }

    const std::string input_segment_name =
        resolve_dequant_published_segment_name(published_input, i);
    const std::string physical_segment_name =
        !published_input.segment_name.empty() ? published_input.segment_name : input_segment_name;
    const std::string output_name =
        terminal_output_tensor && !output_name_looks_generic_local(terminal_output_tensor->name)
            ? terminal_output_tensor->name
        : !output_tensor.name.empty() && !output_name_looks_generic_local(output_tensor.name)
            ? output_tensor.name
        : !published_input.name.empty() ? published_input.name
                                        : input_segment_name;
    const std::uint64_t input_size_bytes =
        preferred_mpk_tensor_size_bytes(input_tensor, input_dtype) > 0U
            ? preferred_mpk_tensor_size_bytes(input_tensor, input_dtype)
        : published_input.size_bytes > 0U ? static_cast<std::uint64_t>(published_input.size_bytes)
                                          : 0U;
    const std::uint64_t output_size_bytes =
        terminal_output_tensor
            ? preferred_mpk_tensor_size_bytes(*terminal_output_tensor, output_dtype)
            : preferred_mpk_tensor_size_bytes(output_tensor, output_dtype);
    if (input_size_bytes == 0U || output_size_bytes == 0U) {
      if (err) {
        *err = "dequant_stage_missing_tensor_size";
      }
      return std::nullopt;
    }

    const bool has_unpack_stage =
        pipeline_internal::sima::get_mla_unpack_stage_io_contract(contract) != nullptr;
    auto resolved_input_view = resolve_slice_backed_dequant_input_view_local(
        contract, stage, published_input, input_shape, input_dtype, input_size_bytes, i);
    const auto contiguous_input_stride_bytes =
        contiguous_stride_bytes_local(input_shape, input_dtype);
    const bool published_stride_needs_recovery =
        published_input.stride_bytes.empty() ||
        published_input.stride_bytes == contiguous_input_stride_bytes;
    const bool use_slice_logical_view =
        resolved_input_view.has_value() && published_stride_needs_recovery;
    const int upstream_physical_index =
        (!has_unpack_stage && resolved_input_view.has_value())
            ? resolved_input_view->upstream_physical_index
            : resolve_dequant_boundary_physical_index(published_input, i);
    const std::int64_t upstream_source_offset =
        (!has_unpack_stage && resolved_input_view.has_value())
            ? resolved_input_view->upstream_source_offset
            : published_input.source_byte_offset;
    const std::int64_t logical_byte_offset = use_slice_logical_view
                                                 ? resolved_input_view->logical_byte_offset
                                                 : published_input.byte_offset;
    const std::vector<std::int64_t> input_stride_source_shape =
        !published_input.mpk_shape.empty()
            ? published_input.mpk_shape
            : (!published_input.logical_shape.empty() ? published_input.logical_shape
                                                      : input_tensor.mpk_shape);
    const std::vector<std::int64_t> input_stride_alternate_source_shape =
        !input_tensor.mpk_shape.empty()
            ? input_tensor.mpk_shape
            : (!input_tensor.logical_shape.empty() ? input_tensor.logical_shape
                                                   : published_input.logical_shape);
    const std::vector<std::int64_t> input_stride_bytes =
        use_slice_logical_view
            ? normalize_view_stride_to_shape_local(resolved_input_view->stride_bytes, input_shape,
                                                   input_stride_source_shape, input_shape)
            : normalize_view_stride_to_shape_local(
                  published_input.stride_bytes, input_stride_source_shape,
                  input_stride_alternate_source_shape, input_shape);
    const std::uint64_t input_physical_span_bytes =
        use_slice_logical_view
            ? resolved_input_view->physical_span_bytes
            : tensor_physical_span_bytes(
                  input_shape, input_stride_bytes, input_size_bytes,
                  pipeline_internal::sima::specbuilders::dtype_size_bytes_from_token(input_dtype));
    if (logical_byte_offset < 0 || upstream_source_offset < 0) {
      if (err) {
        *err = "dequant_stage_negative_input_offset";
      }
      return std::nullopt;
    }
    const std::uint64_t upstream_physical_size_bytes =
        mla_output_slot_size_bytes_local(contract, upstream_physical_index);
    DequantPublishedPhysicalInput* physical_input = find_dequant_physical_input(
        &physical_inputs, upstream_physical_index, upstream_source_offset);
    if (!physical_input) {
      physical_inputs.push_back(DequantPublishedPhysicalInput{
          static_cast<int>(physical_inputs.size()),
          upstream_physical_index,
          upstream_source_offset,
          physical_segment_name,
          0U,
      });
      physical_input = &physical_inputs.back();
    } else if (physical_input->segment_name.empty() && !physical_segment_name.empty()) {
      physical_input->segment_name = physical_segment_name;
    }
    const std::uint64_t logical_end =
        static_cast<std::uint64_t>(logical_byte_offset) + input_physical_span_bytes;
    physical_input->size_bytes =
        std::max(physical_input->size_bytes, std::max(logical_end, upstream_physical_size_bytes));

    const int logical_index = static_cast<int>(i);
    const auto input_materialization_kind =
        published_input.materialization_kind ==
                pipeline_internal::sima::MpkTensorMaterializationKind::OffsetView
            ? pipeline_internal::sima::TensorMaterializationKind::OffsetView
        : published_input.materialization_kind ==
                pipeline_internal::sima::MpkTensorMaterializationKind::Bf16LaneSplitRepack
            ? pipeline_internal::sima::TensorMaterializationKind::Bf16LaneSplitRepack
            : pipeline_internal::sima::TensorMaterializationKind::Direct;
    auto logical_input = pipeline_internal::sima::specbuilders::build_logical_input_static_spec(
        logical_index, logical_index, physical_input->local_physical_index, input_shape,
        input_dtype, input_layout,
        !input_tensor.name.empty()      ? input_tensor.name
        : !published_input.name.empty() ? published_input.name
                                        : input_segment_name,
        "input_tensor", input_segment_name, logical_byte_offset, 0U, input_materialization_kind,
        input_quant);
    logical_input.size_bytes = input_size_bytes;
    if (!input_stride_bytes.empty()) {
      logical_input.stride_bytes = input_stride_bytes;
    }
    compiled.runtime_contract.logical_inputs.push_back(std::move(logical_input));
    compiled.runtime_contract.input_bindings.push_back(
        pipeline_internal::sima::specbuilders::build_input_binding_static_spec(
            0, logical_index, "input_tensor", input_segment_name, logical_index, logical_index,
            upstream_physical_index,
            upstream_physical_size_bytes > 0U ? upstream_physical_size_bytes
                                              : input_physical_span_bytes,
            logical_byte_offset, true));

    auto logical_output = pipeline_internal::sima::specbuilders::build_logical_output_static_spec(
        logical_index, logical_index, 0, logical_index, logical_index, output_shape, output_dtype,
        output_layout, output_name, output_name, "output_tensor",
        static_cast<std::int64_t>(packed_output_total_bytes), output_size_bytes);
    if (input_materialization_kind ==
            pipeline_internal::sima::TensorMaterializationKind::OffsetView &&
        !input_stride_bytes.empty()) {
      logical_output.stride_bytes = contiguous_stride_bytes_local(output_shape, output_dtype);
    }
    compiled.runtime_contract.logical_outputs.push_back(std::move(logical_output));
    compiled.runtime_contract.output_order.push_back(
        pipeline_internal::sima::specbuilders::build_output_route_static_spec(
            logical_index, logical_index, logical_index, output_name, output_name));

    packed_output_total_bytes += output_size_bytes;
  }

  compiled.runtime_contract.physical_inputs.reserve(physical_inputs.size());
  for (const auto& physical_input : physical_inputs) {
    compiled.runtime_contract.physical_inputs.push_back(
        pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
            physical_input.local_physical_index, physical_input.local_physical_index,
            physical_input.size_bytes, pipeline_internal::sima::DeviceKind::Mla,
            physical_input.segment_name, physical_input.upstream_physical_index,
            physical_input.upstream_source_offset));
  }
  compiled.runtime_contract.physical_outputs.push_back(
      pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
          0, 0, packed_output_total_bytes, pipeline_internal::sima::DeviceKind::Cpu,
          "output_tensor"));
  std::string normalize_err;
  if (!pipeline_internal::packedio::normalize_shared_parent_input_views(&compiled.runtime_contract,
                                                                        &normalize_err)) {
    if (err) {
      *err = normalize_err.empty()
                 ? "invalid_model_managed_dequant_parent_view_contract"
                 : "invalid_model_managed_dequant_parent_view_contract: " + normalize_err;
    }
    return std::nullopt;
  }
  std::string packed_err;
  if (!pipeline_internal::packedio::validate_packed_contract(compiled.runtime_contract,
                                                             &packed_err)) {
    if (err) {
      *err = "invalid_model_managed_dequant_contract: " + packed_err;
    }
    return std::nullopt;
  }
  if (err) {
    err->clear();
  }
  return compiled;
}

static bool
remap_model_managed_dequant_inputs_from_upstream(const CompiledRuntimeContract& upstream_runtime,
                                                 CompiledDequantContract* compiled,
                                                 std::string* err = nullptr) {
  if (!compiled) {
    if (err) {
      *err = "null_compiled_dequant_contract";
    }
    return false;
  }
  auto& runtime = compiled->runtime_contract;
  if (runtime.logical_inputs.size() != upstream_runtime.logical_outputs.size()) {
    if (err) {
      *err = "dequant_upstream_logical_count_mismatch";
    }
    return false;
  }
  if (upstream_runtime.physical_outputs.empty()) {
    if (err) {
      *err = "dequant_upstream_missing_physical_outputs";
    }
    return false;
  }

  runtime.physical_inputs = upstream_runtime.physical_outputs;
  for (std::size_t i = 0; i < runtime.logical_inputs.size(); ++i) {
    const auto& upstream_output = upstream_runtime.logical_outputs[i];
    auto& logical_input = runtime.logical_inputs[i];
    auto& binding = runtime.input_bindings[i];

    logical_input.logical_index =
        upstream_output.logical_index >= 0 ? upstream_output.logical_index : static_cast<int>(i);
    logical_input.backend_input_index = upstream_output.backend_output_index >= 0
                                            ? upstream_output.backend_output_index
                                            : static_cast<int>(i);
    logical_input.physical_index = upstream_output.physical_index;
    logical_input.shape = upstream_output.shape;
    logical_input.stride_bytes = upstream_output.stride_bytes;
    logical_input.byte_offset = upstream_output.byte_offset;
    logical_input.size_bytes = upstream_output.size_bytes;
    if (!upstream_output.dtype.empty()) {
      logical_input.dtype = upstream_output.dtype;
    }
    if (!upstream_output.layout.empty()) {
      logical_input.layout = upstream_output.layout;
    }
    if (!upstream_output.logical_name.empty()) {
      logical_input.logical_name = upstream_output.logical_name;
    }
    if (!upstream_output.backend_name.empty()) {
      logical_input.backend_name = upstream_output.backend_name;
    }
    if (!upstream_output.segment_name.empty()) {
      logical_input.segment_name = upstream_output.segment_name;
    }
    if (upstream_output.quant.has_value()) {
      logical_input.quant = upstream_output.quant;
    }

    const auto physical_it = std::find_if(
        runtime.physical_inputs.begin(), runtime.physical_inputs.end(), [&](const auto& physical) {
          return physical.physical_index == upstream_output.physical_index;
        });
    const std::uint64_t physical_size_bytes = physical_it != runtime.physical_inputs.end()
                                                  ? physical_it->size_bytes
                                                  : upstream_output.size_bytes;

    binding.local_logical_input_index = logical_input.logical_index;
    binding.src_logical_output_index = upstream_output.logical_index;
    binding.src_output_slot = upstream_output.output_slot;
    binding.src_physical_output_index = upstream_output.physical_index;
    binding.src_physical_size_bytes = physical_size_bytes;
    binding.src_physical_byte_offset = upstream_output.byte_offset;
    if (!logical_input.backend_name.empty()) {
      binding.cm_input_name = logical_input.backend_name;
    }
    if (!logical_input.segment_name.empty()) {
      binding.source_segment_name = logical_input.segment_name;
    }
  }

  std::string packed_err;
  if (!pipeline_internal::packedio::validate_packed_contract(runtime, &packed_err)) {
    if (err) {
      *err = "invalid_upstream_remapped_dequant_contract: " + packed_err;
    }
    return false;
  }
  if (err) {
    err->clear();
  }
  return true;
}

static const pipeline_internal::sima::MpkPluginIoContract* find_terminal_stage_after_outputs(
    const pipeline_internal::sima::MpkContract& contract,
    const std::vector<const pipeline_internal::sima::MpkPluginIoContract*>& producers);

static bool should_publish_mla_outputs_as_packed_parent_for_boxdecode(
    const std::vector<ExecutionStage>& stages, std::size_t stage_index,
    const pipeline_internal::sima::MlaStaticContract& contract, bool direct_mla_to_boxdecode) {
  const bool adjacent_boxdecode = stage_index + 1U < stages.size() &&
                                  stages[stage_index + 1U].kind == ExecutionStageKind::BoxDecode;
  if (stages[stage_index].kind != ExecutionStageKind::Mla ||
      (!adjacent_boxdecode && !direct_mla_to_boxdecode)) {
    return false;
  }
  if (contract.dispatcher_physical_outputs.size() <= 1U ||
      contract.logical_outputs.size() != contract.dispatcher_physical_outputs.size()) {
    return false;
  }
  for (const auto& dispatcher : contract.dispatcher_physical_outputs) {
    if (dispatcher.size_bytes == 0U) {
      return false;
    }
  }
  std::vector<bool> dispatcher_seen(contract.dispatcher_physical_outputs.size(), false);
  for (std::size_t i = 0; i < contract.logical_outputs.size(); ++i) {
    const auto& logical = contract.logical_outputs[i];
    const int backend_index =
        logical.backend_output_index >= 0 ? logical.backend_output_index : logical.logical_index;
    if (backend_index < 0 ||
        static_cast<std::size_t>(backend_index) >= contract.dispatcher_physical_outputs.size()) {
      return false;
    }
    const auto dispatcher_index = static_cast<std::size_t>(backend_index);
    if (dispatcher_seen[dispatcher_index]) {
      return false;
    }
    dispatcher_seen[dispatcher_index] = true;
    if (logical.size_bytes == 0U ||
        logical.size_bytes != contract.dispatcher_physical_outputs[dispatcher_index].size_bytes) {
      return false;
    }
  }
  if (std::find(dispatcher_seen.begin(), dispatcher_seen.end(), false) != dispatcher_seen.end()) {
    return false;
  }
  return true;
}

static bool
publish_mla_outputs_as_packed_parent(pipeline_internal::sima::MlaStaticContract* contract) {
  if (!contract || contract->dispatcher_physical_outputs.size() <= 1U ||
      contract->logical_outputs.empty()) {
    return false;
  }

  std::vector<std::uint64_t> dispatcher_offsets(contract->dispatcher_physical_outputs.size(), 0U);
  std::uint64_t total_size = 0U;
  for (std::size_t i = 0; i < contract->dispatcher_physical_outputs.size(); ++i) {
    dispatcher_offsets[i] = total_size;
    const auto size = contract->dispatcher_physical_outputs[i].size_bytes;
    if (size == 0U || total_size > (std::numeric_limits<std::uint64_t>::max() - size)) {
      return false;
    }
    total_size += size;
  }
  if (total_size == 0U) {
    return false;
  }

  constexpr const char* kPackedParentSegmentName = "mla_output_tensor";
  // The dispatcher still has N real OFM outputs.  This published parent is an
  // aggregate runtime allocation over those N outputs, so source_physical_index
  // is only a compatibility anchor for older one-source descriptors; processmla
  // validates the dense aggregate-parent shape explicitly.
  auto parent = pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
      /*physical_index=*/0, /*allocator_index=*/0, /*size_bytes=*/total_size,
      pipeline_internal::sima::DeviceKind::Mla, kPackedParentSegmentName,
      /*source_physical_index=*/0, /*source_byte_offset=*/0);

  for (std::size_t i = 0; i < contract->logical_outputs.size(); ++i) {
    auto& logical = contract->logical_outputs[i];
    const int backend_index =
        logical.backend_output_index >= 0 ? logical.backend_output_index : logical.logical_index;
    if (backend_index < 0 || static_cast<std::size_t>(backend_index) >= dispatcher_offsets.size()) {
      return false;
    }
    const auto dispatcher_index = static_cast<std::size_t>(backend_index);
    logical.physical_index = 0;
    logical.byte_offset = static_cast<std::int64_t>(dispatcher_offsets[dispatcher_index]);
    logical.segment_name = kPackedParentSegmentName;
  }

  contract->physical_outputs.clear();
  contract->physical_outputs.push_back(std::move(parent));
  return true;
}

static std::vector<ModelFragment::StageFacts> build_stage_facts_from_execution_plan(
    const std::vector<ExecutionStage>& stages,
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract,
    const std::optional<bool>& processcvu_preproc_single_output_handoff,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& model_managed_route_flags,
    const std::string& input_format, int input_depth, int max_input_width, int max_input_height,
    bool normalize, const std::vector<float>& mean, const std::vector<float>& stddev,
    const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract,
    ModelStage stage_context, bool direct_mla_to_boxdecode = false) {
  (void)upstream_handoff_contract;
  if (!mpk_contract.has_value()) {
    throw std::runtime_error(
        "ModelFragment: strict MPK contract required for typed execution plan");
  }

  std::vector<ModelFragment::StageFacts> facts;
  facts.reserve(stages.size());
  for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
    const auto& stage = stages[stage_index];
    ModelFragment::StageFacts entry;
    entry.stage_name = stage.stage_name;
    entry.stage_order = stage.order_index;
    const auto* mpk_stage = find_mpk_stage_for_execution_stage(mpk_contract, stage);

    if (execution_stage_uses_processcvu_contract(stage.kind)) {
      if (stage.kind == ExecutionStageKind::Preproc) {
        if (!processcvu_preproc_single_output_handoff.has_value()) {
          throw std::runtime_error("ModelFragment: model-managed preproc stage '" +
                                   stage.stage_name +
                                   "' is missing typed preproc_single_output_handoff fact");
        }
        entry.processcvu_preproc_single_output_handoff = *processcvu_preproc_single_output_handoff;
        entry.processcvu_contract = pipeline_internal::sima::stagesemantics::
            build_processcvu_mpk_compiled_contract_for_stage_kind(
                *mpk_contract, ExecutionStageKind::Preproc, std::nullopt, std::nullopt,
                processcvu_preproc_single_output_handoff, input_format, input_depth,
                max_input_width, max_input_height, normalize, mean, stddev);
      } else if (stage_context == ModelStage::Postprocess) {
        std::optional<std::string> exact_processcvu_stage_name_or_id;
        if (!stage.stage_name.empty()) {
          exact_processcvu_stage_name_or_id = stage.stage_name;
        } else if (!stage.plugin_id.empty()) {
          exact_processcvu_stage_name_or_id = stage.plugin_id;
        }
        entry.processcvu_contract = pipeline_internal::sima::stagesemantics::
            build_processcvu_mpk_compiled_contract_for_stage_kind(
                *mpk_contract, stage.kind, exact_processcvu_stage_name_or_id);
      } else {
        // Determine whether this pre-MLA stage is part of a fan-in topology
        // (native multi-IFM or packer-style). For fan-in we omit
        // exact_processcvu_stage_name_or_id so the renderer's multi-IO branch
        // (the symmetric counterpart of how the post side renders multi-IO
        // detessdequant) collapses N siblings into one compiled contract whose
        // logical_inputs.size() == N. For monolithic single-sibling models we
        // keep the explicit per-stage name to preserve the existing
        // single-stage code path (including the geometry-validator's
        // by-exact-name graph node lookup, which the generic family fallback
        // does not cover for every family — e.g. casttess).
        const auto pre_mla_branch_count = [&]() -> std::size_t {
          const auto* mla = pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract);
          if (mla == nullptr) {
            return 1U;
          }
          if (mla->input_tensors.size() > 1U) {
            // Native multi-IFM: MLA itself consumes N distinct physical inputs.
            return mla->input_tensors.size();
          }
          if (pipeline_internal::sima::mla_consumer_keeps_distinct_physical_inputs(*mpk_contract)) {
            return mla->input_tensors.size();
          }
          // Packer-style: a single upstream pre-MLA stage with input_tensors.size()
          // > 1 (the IFM packer) bundles N branches into MLA's single input.
          const auto ordered_plugins =
              pipeline_internal::sima::plugins_in_execution_order(*mpk_contract);
          const auto mla_rank_opt = mla_rank_in_order(*mpk_contract, ordered_plugins);
          for (std::size_t rank = 0; rank < ordered_plugins.size(); ++rank) {
            if (mla_rank_opt.has_value() && rank >= *mla_rank_opt) {
              break;
            }
            const auto idx = ordered_plugins[rank];
            if (idx >= mpk_contract->plugins.size()) {
              continue;
            }
            if (mpk_contract->plugins[idx].input_tensors.size() > 1U) {
              return mpk_contract->plugins[idx].input_tensors.size();
            }
          }
          return 1U;
        }();
        const bool is_pre_mla_family =
            stage.kind == ExecutionStageKind::Quant || stage.kind == ExecutionStageKind::Tess ||
            stage.kind == ExecutionStageKind::QuantTess || stage.kind == ExecutionStageKind::Cast ||
            stage.kind == ExecutionStageKind::CastTess;
        const bool fan_in_path = is_pre_mla_family && pre_mla_branch_count > 1U;
        std::optional<std::string> exact_processcvu_stage_name_or_id;
        if (!fan_in_path) {
          if (mpk_stage != nullptr) {
            if (!mpk_stage->name.empty()) {
              exact_processcvu_stage_name_or_id = mpk_stage->name;
            } else if (!mpk_stage->plugin_id.empty()) {
              exact_processcvu_stage_name_or_id = mpk_stage->plugin_id;
            }
          }
          if (!exact_processcvu_stage_name_or_id.has_value() ||
              exact_processcvu_stage_name_or_id->empty()) {
            if (!stage.stage_name.empty()) {
              exact_processcvu_stage_name_or_id = stage.stage_name;
            } else if (!stage.plugin_id.empty()) {
              exact_processcvu_stage_name_or_id = stage.plugin_id;
            }
          }
        }
        entry.processcvu_contract = pipeline_internal::sima::stagesemantics::
            build_processcvu_mpk_preadapter_compiled_contract_for_stage_kind(
                *mpk_contract, stage.kind, exact_processcvu_stage_name_or_id, std::nullopt);
      }
    }

    if (stage.kind == ExecutionStageKind::Mla) {
      const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract);
      if (!mla_stage) {
        throw std::runtime_error(
            "ModelFragment: strict MPK MLA contract missing for stage '" + stage.stage_name +
            "'. Ensure the MPK manifest includes an MLA plugin with"
            " input/output tensor contracts (processor='MLA' or kernel='infer').");
      }
      const auto published_outputs =
          pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_contract);
      const auto logical_outputs =
          pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_contract);
      const auto boundary_inputs =
          pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(*mpk_contract);
      const auto physical_outputs =
          pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(*mpk_contract);
      auto mla_contract = pipeline_internal::sima::build_mla_static_contract_from_mpk_stage(
          *mla_stage,
          !published_outputs.empty()
              ? published_outputs
              : (logical_outputs.empty() ? mla_stage->output_tensors : logical_outputs),
          physical_outputs.empty() ? mla_stage->output_tensors : physical_outputs, stage.stage_name,
          boundary_inputs.empty() ? nullptr : &boundary_inputs);
      mla_contract.consumer_keeps_distinct_physical_inputs =
          pipeline_internal::sima::mla_consumer_keeps_distinct_physical_inputs(*mpk_contract);
      const auto mla_props = read_mla_runtime_properties_from_mpk_contract(mpk_contract);
      if (!mla_props.has_value() || mla_props->model_path.empty()) {
        throw std::runtime_error(
            "ModelFragment: strict MPK MLA runtime payload missing for stage '" + stage.stage_name +
            "' (expected 'model_path', 'batch_size' fields in the MPK MLA"
            " config or simaai__params section).");
      }
      apply_mla_runtime_properties_to_contract(*mla_props, &mla_contract);
      if (should_publish_mla_outputs_as_packed_parent_for_boxdecode(
              stages, stage_index, mla_contract, direct_mla_to_boxdecode)) {
        (void)publish_mla_outputs_as_packed_parent(&mla_contract);
      }
      entry.mla_compiled =
          pipeline_internal::sima::stagesemantics::build_mla_compiled_contract(mla_contract);
    }

    if (stage.kind == ExecutionStageKind::BoxDecode) {
      std::optional<pipeline_internal::sima::ModelManagedRouteFlags> resolved_route_flags;
      if (model_managed_route_flags.has_value()) {
        resolved_route_flags = *model_managed_route_flags;
      } else {
        std::string route_flags_error;
        resolved_route_flags =
            pipeline_internal::sima::resolve_model_managed_boxdecode_route_flags_from_mpk(
                *mpk_contract, mpk_stage, &route_flags_error);
        if (!resolved_route_flags.has_value()) {
          throw std::runtime_error(
              "ModelFragment: strict model-managed boxdecode route facts missing for stage '" +
              stage.stage_name + "': " +
              (route_flags_error.empty() ? std::string("missing MPK/upstream route facts")
                                         : route_flags_error));
        }
      }
      resolved_route_flags->quant_contract_required = resolved_route_flags->quant_needed;
      resolved_route_flags->boxdecode_selected = true;
      std::string subset_error;
      auto boxdecode_subset =
          pipeline_internal::sima::plugin_contracts::extract_boxdecode_contract_subset_from_mpk(
              *mpk_contract, *resolved_route_flags, mpk_stage, &subset_error);
      if (!boxdecode_subset.has_value()) {
        throw std::runtime_error(
            "ModelFragment: strict model-managed boxdecode contract missing for stage '" +
            stage.stage_name + "': " +
            (subset_error.empty() ? std::string("missing MPK/upstream facts") : subset_error));
      }
      // No model-managed boxdecode unless the MPK declares an explicit
      // decode_type. Skipping the compiled-contract install here means the
      // resulting Model has no model-managed boxdecode stage; callers that
      // want boxdecode must either author decode_type into the MPK or use
      // the standalone SimaBoxDecode constructor.
      if (simaai::neat::pipeline_internal::sima::is_box_decode_type_specified(
              boxdecode_subset->decode_type)) {
        pipeline_internal::sima::stagesemantics::BoxDecodeCompiledContractOptions compile_options;
        compile_options.decode_type = boxdecode_subset->decode_type;
        if (boxdecode_subset->decode_type_option.has_value()) {
          compile_options.decode_type_option = boxdecode_subset->decode_type_option;
        }
        compile_options.score_activation = boxdecode_subset->score_activation;
        compile_options.model_owned_flags = true;
        compile_options.required_preprocess_meta_fields = default_preprocess_meta_required_fields();
        entry.boxdecode_compiled =
            pipeline_internal::sima::stagesemantics::build_boxdecode_compiled_contract_from_subset(
                *boxdecode_subset, compile_options);
      }
    }

    if (stage.kind == ExecutionStageKind::Dequant && !entry.processcvu_contract.has_value()) {
      std::string dequant_error;
      entry.dequant_compiled =
          build_model_managed_dequant_compiled_contract_from_mpk(*mpk_contract, &dequant_error);
      if (!entry.dequant_compiled.has_value()) {
        throw std::runtime_error(
            "ModelFragment: strict model-managed dequant contract missing for stage '" +
            stage.stage_name + "': " +
            (dequant_error.empty() ? std::string("missing MPK dequant facts") : dequant_error));
      }
      if (!facts.empty() && stage_index > 0U &&
          stages[stage_index - 1U].kind == ExecutionStageKind::Detess) {
        CompiledRuntimeContract scratch_runtime;
        const auto* upstream_runtime =
            facts.back().transport_compiled.has_value()
                ? &facts.back().transport_compiled->runtime_contract
                : resolve_stage_facts_runtime_contract(facts.back(), &scratch_runtime);
        if (!upstream_runtime) {
          throw std::runtime_error(
              "ModelFragment: strict model-managed dequant contract missing upstream detess "
              "runtime facts for stage '" +
              stage.stage_name + "'");
        }
        if (!remap_model_managed_dequant_inputs_from_upstream(
                *upstream_runtime, &*entry.dequant_compiled, &dequant_error)) {
          throw std::runtime_error(
              "ModelFragment: strict model-managed dequant contract could not bind upstream "
              "detess outputs for stage '" +
              stage.stage_name + "': " +
              (dequant_error.empty() ? std::string("invalid upstream runtime contract")
                                     : dequant_error));
        }
      }
    }

    if (stage.kind == ExecutionStageKind::Detess) {
      std::optional<CompiledRuntimeContract> runtime_contract;
      std::optional<pipeline_internal::sima::ProcessCvuStagePayload> processcvu_payload;
      if (entry.processcvu_contract.has_value()) {
        runtime_contract =
            transport_runtime_contract_from_processcvu_compiled(*entry.processcvu_contract);
        runtime_contract->plugin_kind = "neatdetess";
        processcvu_payload = entry.processcvu_contract->payload;
      }
      entry.transport_compiled = build_model_managed_transport_contract(
          "neatdetess", "detess", std::move(processcvu_payload), std::move(runtime_contract));
    }

    facts.push_back(std::move(entry));
  }
  return facts;
}

class ModelFragmentNode final : public Node,
                                public CompiledChildStageProvider,
                                public NodeContractProvider,
                                public NodeContractConfigurable,
                                public ModelLineageProvider {
public:
  ModelFragmentNode(std::string kind, std::string label, std::string fragment,
                    std::vector<std::string> elements,
                    std::vector<ModelFragment::StageFacts> stage_facts,
                    std::shared_ptr<const ModelLineageBinding> model_lineage = nullptr)
      : kind_(std::move(kind)), label_(std::move(label)), fragment_(std::move(fragment)),
        elements_(std::move(elements)), stage_facts_(std::move(stage_facts)),
        model_lineage_(std::move(model_lineage)) {}

  std::string kind() const override {
    return kind_;
  }
  std::string user_label() const override {
    return label_;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  std::string buffer_name_hint(int) const override {
    if (!elements_.empty())
      return elements_.back();
    return "";
  }
  std::string backend_fragment(int) const override {
    return fragment_;
  }
  std::vector<std::string> element_names(int) const override {
    return elements_;
  }

  NodeContractDefinition contract_definition() const override {
    NodeContractDefinition def;
    def.node_kind = kind_;
    def.plugin_kind = "ModelFragment";
    return def;
  }

  bool compile_child_stage_contracts(std::vector<CompiledNodeContract>* out,
                                     std::string* err) const override {
    if (!out) {
      if (err) {
        *err = "ModelFragment contract compile: child stage output is null";
      }
      return false;
    }
    *out = compile_fragment_contracts(err);
    if (out->empty()) {
      if (err && err->empty()) {
        *err = "ModelFragment contract compile: fragment produced no semantic stages";
      }
      return false;
    }
    if (err) {
      err->clear();
    }
    return true;
  }

  void apply_compiled_contract(const CompiledNodeContract&, std::string* err) override {
    if (err) {
      err->clear();
    }
  }

  const ModelLineageBinding* model_lineage_binding() const override {
    return model_lineage_.get();
  }

private:
  const ModelFragment::StageFacts* find_stage_facts(const std::string& stage_name) const {
    if (stage_name.empty()) {
      return nullptr;
    }
    for (const auto& entry : stage_facts_) {
      if (entry.stage_name == stage_name) {
        return &entry;
      }
    }
    return nullptr;
  }

  std::vector<CompiledNodeContract> compile_fragment_contracts(std::string* err) const {
    using namespace simaai::neat::pipeline_internal::sima;

    if (err) {
      err->clear();
    }
    std::vector<CompiledNodeContract> stages;
    const auto elements = parse_pipeline_elements(fragment_);
    auto resolve_stage_facts =
        [&](const PipelineElementSpec& element) -> const ModelFragment::StageFacts* {
      if (!element.stage_id.empty()) {
        if (const auto* entry = find_stage_facts(element.stage_id)) {
          return entry;
        }
      }
      if (!element.element_name.empty()) {
        if (const auto* entry = find_stage_facts(element.element_name)) {
          return entry;
        }
      }
      if (stage_facts_.size() == 1U) {
        return &stage_facts_.front();
      }
      return nullptr;
    };
    auto fragment_stage_definition = [&](const std::string& plugin_kind) {
      NodeContractDefinition def;
      def.node_kind = kind_;
      def.plugin_kind = plugin_kind;
      return def;
    };
    for (const auto& element : elements) {
      const std::string plugin = to_lower(element.plugin);
      const std::string stage_id =
          !element.stage_id.empty()
              ? element.stage_id
              : (!element.element_name.empty() ? element.element_name : label_);
      const std::string element_name =
          !element.element_name.empty() ? element.element_name : stage_id;
      CompiledNodeContract compiled_stage;
      if (plugin.find("queue") != std::string::npos ||
          plugin.find("capsfilter") != std::string::npos ||
          plugin.find("funnel") != std::string::npos) {
        continue;
      }

      if (plugin.find("processcvu") != std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry) {
          if (err) {
            *err = "ModelFragment contract compile: missing processcvu config for '" +
                   element_name + "'";
          }
          return {};
        }
        if (!entry->processcvu_contract.has_value()) {
          if (err) {
            *err = "ModelFragment contract compile: missing cached processcvu contract for '" +
                   element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_processcvu_node_contract(
                kind_, element_name, stage_id, fragment_stage_definition("processcvu"),
                *entry->processcvu_contract, &compiled_stage, err)) {
          return {};
        }
        stages.push_back(std::move(compiled_stage));
        continue;
      }

      if (plugin.find("processmla") != std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry) {
          if (err) {
            *err = "ModelFragment contract compile: missing MLA config for '" + element_name + "'";
          }
          return {};
        }
        if (!entry->mla_compiled.has_value()) {
          if (err) {
            *err = "ModelFragment contract compile: missing cached MLA contract for '" +
                   element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_mla_node_contract(
                kind_, element_name, stage_id, fragment_stage_definition("processmla"),
                *entry->mla_compiled, &compiled_stage, err)) {
          if (err) {
            *err = err->empty()
                       ? "ModelFragment contract compile: failed to build MLA contract for '" +
                             element_name + "'"
                       : *err;
          }
          return {};
        }
        stages.push_back(std::move(compiled_stage));
        continue;
      }

      if (plugin.find("boxdecode") != std::string::npos ||
          plugin.find("objectdecode") != std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry) {
          if (err) {
            *err = "ModelFragment contract compile: missing boxdecode config for '" + element_name +
                   "'";
          }
          return {};
        }
        const std::string plugin_kind =
            plugin.find("objectdecode") != std::string::npos ? "neatobjectdecode" : "neatboxdecode";
        if (!entry->boxdecode_compiled.has_value()) {
          if (err) {
            *err = "ModelFragment contract compile: missing cached boxdecode contract for '" +
                   element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_boxdecode_node_contract(
                kind_, plugin_kind, element_name, stage_id, fragment_stage_definition(plugin_kind),
                *entry->boxdecode_compiled, &compiled_stage, err)) {
          if (err) {
            *err =
                err->empty()
                    ? "ModelFragment contract compile: failed to build boxdecode contract for '" +
                          element_name + "'"
                    : *err;
          }
          return {};
        }
        stages.push_back(std::move(compiled_stage));
        continue;
      }

      if (plugin.find("dequant") != std::string::npos &&
          plugin.find("processcvu") == std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry) {
          if (err) {
            *err =
                "ModelFragment contract compile: missing dequant config for '" + element_name + "'";
          }
          return {};
        }
        if (!entry->dequant_compiled.has_value()) {
          if (err) {
            *err =
                "ModelFragment contract compile: missing cached standalone dequant contract for '" +
                element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_dequant_node_contract(
                kind_, "dequant", element_name, stage_id, fragment_stage_definition("dequant"),
                *entry->dequant_compiled, &compiled_stage, err)) {
          return {};
        }
        stages.push_back(std::move(compiled_stage));
        continue;
      }

      if (plugin.find("detess") != std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry || !entry->transport_compiled.has_value()) {
          if (err) {
            *err =
                "ModelFragment contract compile: missing cached detess transport contract for '" +
                element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_transport_node_contract(
                kind_, element_name, stage_id, fragment_stage_definition("neatdetess"),
                *entry->transport_compiled, &compiled_stage, err)) {
          return {};
        }
        stages.push_back(std::move(compiled_stage));
        continue;
      }
    }
    return stages;
  }

  std::string kind_;
  std::string label_;
  std::string fragment_;
  std::vector<std::string> elements_;
  std::vector<ModelFragment::StageFacts> stage_facts_;
  std::shared_ptr<const ModelLineageBinding> model_lineage_;
};

static std::vector<std::shared_ptr<Node>>
make_fragment_nodes(const ModelFragment& frag, const std::string& label,
                    std::shared_ptr<const ModelLineageBinding> model_lineage = nullptr) {
  std::vector<std::shared_ptr<Node>> nodes;
  nodes.push_back(std::make_shared<ModelFragmentNode>(
      "ModelFragment", label, frag.gst, frag.elements, frag.stage_facts, std::move(model_lineage)));
  return nodes;
}

static bool validate_terminal_mla_metadata(
    const std::vector<ExecutionStage>& infer_seq,
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract) {
  if (infer_seq.empty())
    return false;
  const auto& terminal = infer_seq.back();
  if (terminal.kind != ExecutionStageKind::Mla)
    return false;
  if (!mpk_contract.has_value()) {
    throw std::runtime_error(
        "Terminal MLA metadata validation requires strict MPK contract after infer trimming");
  }

  const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract);
  if (!mla_stage) {
    throw std::runtime_error("Terminal MLA stage missing from strict MPK contract after infer "
                             "trimming: '" +
                             terminal.stage_name + "'");
  }

  const auto published_outputs =
      pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_contract);
  const auto logical_outputs =
      pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_contract);
  const auto& effective_outputs =
      !published_outputs.empty()
          ? published_outputs
          : (!logical_outputs.empty() ? logical_outputs : mla_stage->output_tensors);
  if (effective_outputs.empty()) {
    throw std::runtime_error(
        "Terminal MLA contract missing output tensors after infer trimming: '" +
        terminal.stage_name + "'");
  }

  const auto shape_dbg = [](const std::vector<std::int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i > 0U) {
        oss << ",";
      }
      oss << shape[i];
    }
    oss << "]";
    return oss.str();
  };

  for (std::size_t i = 0; i < effective_outputs.size(); ++i) {
    const auto& tensor = effective_outputs[i];
    const auto& resolved_shape =
        !tensor.logical_shape.empty() ? tensor.logical_shape : tensor.mpk_shape;
    const auto dims = mpk_dims_from_shape(resolved_shape);
    const std::string dtype =
        !tensor.logical_dtype.empty()
            ? tensor.logical_dtype
            : (!tensor.dtype.empty() ? tensor.dtype : mla_stage->canonical_output_dtype);
    if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0 || dtype.empty()) {
      std::ostringstream msg;
      msg << "Terminal MLA contract incomplete after infer trimming"
          << " stage='" << terminal.stage_name << "'"
          << " output_index=" << i << " resolved={w=" << dims.width << ",h=" << dims.height
          << ",d=" << dims.depth << ",dtype=" << (dtype.empty() ? "<empty>" : dtype) << "}"
          << " logical_shape=" << shape_dbg(tensor.logical_shape)
          << " mpk_shape=" << shape_dbg(tensor.mpk_shape)
          << " shape_semantics=" << static_cast<int>(tensor.shape_semantics);
      throw std::runtime_error(msg.str());
    }
  }

  return true;
}

static std::string pipeline_config_name(PipelineType type) {
  switch (type) {
  case PipelineType::Preproc:
    return "0_preproc.json";
  case PipelineType::Quant:
    return "0_quant.json";
  case PipelineType::Tess:
    return "0_tess.json";
  case PipelineType::QuantTess:
    return "0_quanttess.json";
  case PipelineType::CastTess:
    return "0_casttess.json";
  case PipelineType::Cast:
    return "0_cast.json";
  }
  return {};
}

static const pipeline_internal::sima::MpkPluginIoContract* find_terminal_stage_after_outputs(
    const pipeline_internal::sima::MpkContract& contract,
    const std::vector<const pipeline_internal::sima::MpkPluginIoContract*>& producers);

static const pipeline_internal::sima::MpkPluginIoContract* find_terminal_stage_after_outputs(
    const pipeline_internal::sima::MpkContract& contract,
    const std::vector<const pipeline_internal::sima::MpkPluginIoContract*>& producers) {
  if (producers.empty()) {
    return nullptr;
  }
  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  if (ordered.empty()) {
    return nullptr;
  }

  std::optional<std::size_t> anchor_pos;
  for (std::size_t pos = 0; pos < ordered.size(); ++pos) {
    const std::size_t idx = ordered[pos];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    for (const auto* producer : producers) {
      if (producer == &contract.plugins[idx]) {
        anchor_pos = std::max(anchor_pos.value_or(0U), pos);
      }
    }
  }
  if (!anchor_pos.has_value() || *anchor_pos + 1U >= ordered.size()) {
    return nullptr;
  }

  const std::size_t expected_count = producers.size();
  for (std::size_t pos = *anchor_pos + 1U; pos < ordered.size(); ++pos) {
    const std::size_t idx = ordered[pos];
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& candidate = contract.plugins[idx];
    if (candidate.output_tensors.size() == expected_count) {
      return &candidate;
    }
  }

  const auto& terminal = contract.plugins[ordered.back()];
  return terminal.output_tensors.empty() ? nullptr : &terminal;
}

static std::string pipeline_kernel_name(PipelineType type) {
  switch (type) {
  case PipelineType::Preproc:
    return "preproc";
  case PipelineType::Quant:
    return "quant";
  case PipelineType::Tess:
    return "tess";
  case PipelineType::QuantTess:
    return "quanttess";
  case PipelineType::CastTess:
    return "casttess";
  case PipelineType::Cast:
    return "cast";
  }
  return "preproc";
}

static PipelineType get_pipeline_type(PipelineType requested_pipeline_type) {
  if (requested_pipeline_type != PipelineType::Preproc &&
      requested_pipeline_type != PipelineType::Quant &&
      requested_pipeline_type != PipelineType::Tess &&
      requested_pipeline_type != PipelineType::QuantTess &&
      requested_pipeline_type != PipelineType::CastTess &&
      requested_pipeline_type != PipelineType::Cast) {
    throw std::invalid_argument("ModelPack: invalid requested pipeline type");
  }
  return requested_pipeline_type;
}

// Sequence parsing is internal to the model archive loader.

static std::string find_config_by_substr(const std::string& etc_dir, const std::string& needle) {
  if (needle.empty())
    return "";
  const std::string want = to_lower(needle);
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(etc_dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    const fs::path p = entry.path();
    if (p.extension() != ".json")
      continue;
    const std::string name = to_lower(p.filename().string());
    if (name.find(want) != std::string::npos) {
      return p.string();
    }
  }
  return "";
}

static const json* params_or_root_local(const json& cfg) {
  if (cfg.contains("simaai__params") && cfg.at("simaai__params").is_object()) {
    return &cfg.at("simaai__params");
  }
  return &cfg;
}

static bool has_terminal_policy(const InferenceTerminalPolicy& policy) {
  return policy.mla_only || policy.last_stage_index.has_value() ||
         policy.last_stage_name.has_value() || policy.last_plugin_id.has_value() ||
         policy.last_processor.has_value();
}

static std::string infer_stage_summary(const std::vector<ExecutionStage>& seq) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    if (i)
      oss << ", ";
    oss << "#" << i << "{name=" << seq[i].stage_name << ",plugin=" << seq[i].plugin_id
        << ",kernel=" << seq[i].kernel << ",processor=" << seq[i].processor << "}";
  }
  return oss.str();
}

static std::size_t resolve_terminal_index_or_throw(const std::vector<ExecutionStage>& infer_seq,
                                                   const InferenceTerminalPolicy& policy) {
  if (infer_seq.empty()) {
    throw std::runtime_error(
        "Inference terminal policy cannot resolve terminal stage: infer block is empty");
  }

  if (policy.last_stage_index.has_value()) {
    const std::size_t idx = *policy.last_stage_index;
    if (idx >= infer_seq.size()) {
      std::ostringstream msg;
      msg << "Inference terminal policy index out of range: requested=" << idx
          << " infer_size=" << infer_seq.size();
      throw std::runtime_error(msg.str());
    }
    return idx;
  }

  if (policy.last_stage_name.has_value()) {
    const std::string want = *policy.last_stage_name;
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (infer_seq[i].stage_name == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by name='" << want
        << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.last_plugin_id.has_value()) {
    const std::string want = to_lower(*policy.last_plugin_id);
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (to_lower(infer_seq[i].plugin_id) == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by plugin='"
        << *policy.last_plugin_id << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.last_processor.has_value()) {
    const std::string want = to_lower(*policy.last_processor);
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (to_lower(infer_seq[i].processor) == want)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve terminal stage by processor='"
        << *policy.last_processor << "' infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  if (policy.mla_only) {
    for (std::size_t i = infer_seq.size(); i-- > 0;) {
      if (infer_seq[i].kind == ExecutionStageKind::Mla)
        return i;
    }
    std::ostringstream msg;
    msg << "Inference terminal policy could not resolve last MLA stage"
        << " infer_stages=[" << infer_stage_summary(infer_seq) << "]";
    throw std::runtime_error(msg.str());
  }

  throw std::runtime_error(
      "Inference terminal policy requested but no terminal selector was provided");
}

static void validate_infer_sequence_or_throw(const std::vector<ExecutionStage>& infer_seq) {
  if (infer_seq.empty()) {
    throw std::runtime_error("Inference block is empty after terminal policy application");
  }
  for (std::size_t i = 0; i < infer_seq.size(); ++i) {
    const auto& e = infer_seq[i];
    if (e.stage_name.empty()) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << ": empty stage name";
      throw std::runtime_error(msg.str());
    }
    if (e.plugin_id.empty()) {
      std::ostringstream msg;
      msg << "Invalid infer stage at index " << i << " name='" << e.stage_name
          << "': empty plugin_id";
      throw std::runtime_error(msg.str());
    }
  }
}

static const char* stage_label(ModelStage stage) {
  switch (stage) {
  case ModelStage::Preprocess:
    return "preprocess";
  case ModelStage::MlaOnly:
    return "mla_only";
  case ModelStage::Postprocess:
    return "postprocess";
  case ModelStage::Full:
    return "full";
  }
  return "full";
}

static std::vector<ExecutionStage> flatten_execution_plan(const ExecutionPlan& plan,
                                                          ModelStage stage) {
  if (stage == ModelStage::Preprocess) {
    return plan.pre;
  }
  if (stage == ModelStage::MlaOnly) {
    return plan.infer;
  }
  if (stage == ModelStage::Postprocess) {
    return plan.post;
  }
  std::vector<ExecutionStage> out;
  out.reserve(plan.pre.size() + plan.infer.size() + plan.post.size());
  out.insert(out.end(), plan.pre.begin(), plan.pre.end());
  out.insert(out.end(), plan.infer.begin(), plan.infer.end());
  out.insert(out.end(), plan.post.begin(), plan.post.end());
  return out;
}

static std::string upstream_name_for_stage(const ExecutionPlan& plan, ModelStage stage) {
  if (stage == ModelStage::MlaOnly) {
    if (!plan.pre.empty() && !plan.pre.back().stage_name.empty()) {
      return plan.pre.back().stage_name;
    }
    return kDefaultPreviousNodeName;
  }
  if (stage == ModelStage::Postprocess) {
    if (!plan.infer.empty() && !plan.infer.back().stage_name.empty()) {
      return plan.infer.back().stage_name;
    }
    return kDefaultPreviousNodeName;
  }
  return kDefaultPreviousNodeName;
}

static ModelFragment
build_fragment_linear(const std::vector<ExecutionStage>& stages,
                      const std::string& initial_input_name, int num_buffers_cvu,
                      int num_buffers_mla, const std::string& name_suffix,
                      const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract,
                      std::vector<ModelFragment::StageFacts> stage_facts) {
  ModelFragment frag;
  if (stages.empty())
    return frag;

  std::ostringstream pipelineStr;
  std::string previous_node_name =
      initial_input_name.empty() ? kDefaultPreviousNodeName : initial_input_name;
  auto find_stage_facts = [&](const std::string& stage_name) -> const ModelFragment::StageFacts* {
    if (stage_name.empty()) {
      return nullptr;
    }
    for (const auto& facts : stage_facts) {
      if (facts.stage_name == stage_name) {
        return &facts;
      }
    }
    return nullptr;
  };

  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    const std::string plugin =
        stage.factory_name.empty() ? require_stage_factory(stage.kind) : stage.factory_name;
    const std::string base_name =
        stage.stage_name.empty() ? std::string(default_stage_name(stage.kind)) : stage.stage_name;
    const std::string name = name_suffix.empty() ? base_name : (base_name + name_suffix);

    if (i)
      pipelineStr << "! ";
    pipelineStr << plugin << " name=" << name << " ";
    pipelineStr << "stage-id=" << name << " ";
    if (plugin == "neatprocesscvu") {
      if (num_buffers_cvu > 0) {
        pipelineStr << " num-buffers=" << num_buffers_cvu << " ";
      }
    } else if (plugin == "neatprocessmla") {
      const auto props = read_mla_runtime_properties_from_mpk_contract(mpk_contract);
      if (!props.has_value()) {
        throw std::runtime_error("ModelPack: model-managed MLA stage '" + name +
                                 "' is missing MPK runtime properties");
      }
      if (!props->model_path.empty()) {
        pipelineStr << "model-path=\"" << props->model_path << "\" ";
      }
      if (props->batch_size > 0) {
        pipelineStr << "batch-size=" << props->batch_size << " ";
      }
      if (props->batch_sz_model > 0) {
        pipelineStr << "batch-sz-model=" << props->batch_sz_model << " ";
      }
      const bool force_single_pipe = env_truthy_local("SIMA_FORCE_MLA_SINGLE_PIPE");
      const bool use_multi_pipeline = !force_single_pipe && num_buffers_mla > 1;
      pipelineStr << "multi-pipeline=" << (use_multi_pipeline ? "true" : "false") << " ";
      if (num_buffers_mla > 0) {
        pipelineStr << " num-buffers=" << num_buffers_mla << " ";
      }
    }
    if (const auto* facts = find_stage_facts(stage.stage_name); facts != nullptr) {
      for (const auto& [key, value] : facts->fragment_properties) {
        if (!key.empty() && !value.empty()) {
          pipelineStr << key << "=" << value << " ";
        }
      }
    }

    frag.elements.push_back(name);
    previous_node_name = name;
  }

  frag.gst = pipelineStr.str();
  frag.stage_facts = std::move(stage_facts);
  return frag;
}

} // namespace

ModelPack::ModelPack(const std::string& tar_gz) {
  init(tar_gz);
}

ModelPack::ModelPack(const std::string& tar_gz, const std::string& media_type,
                     const std::string& format, int depth, int max_width, int max_height,
                     int max_depth, bool normalize, std::vector<float> mean,
                     std::vector<float> stddev, const std::string& preproc_next_cpu,
                     PipelineType requested_pipeline_type, const std::string& upstream_name,
                     int num_buffers_cvu, int num_buffers_mla, int queue_max_buffers,
                     int64_t queue_max_time_ns, const std::string& queue_leaky,
                     const std::string& name_suffix, const InferenceTerminalPolicy& terminal_policy,
                     bool cleanup_extracted_model_data) {
  Config cfg;
  cfg.normalize = normalize;
  cfg.mean = std::move(mean);
  cfg.stddev = std::move(stddev);
  cfg.input_depth = depth;
  cfg.max_input_width = max_width;
  cfg.max_input_height = max_height;
  cfg.max_input_depth = max_depth;
  cfg.preproc_next_cpu = preproc_next_cpu;
  cfg.requested_pipeline_type = requested_pipeline_type;
  if (!upstream_name.empty())
    cfg.upstream_name = upstream_name;
  cfg.num_buffers_cvu = num_buffers_cvu;
  cfg.num_buffers_mla = num_buffers_mla;
  cfg.queue_max_buffers = queue_max_buffers;
  cfg.queue_max_time_ns = queue_max_time_ns;
  cfg.queue_leaky = queue_leaky;
  cfg.name_suffix = name_suffix;
  cfg.terminal_policy = terminal_policy;
  cfg.cleanup_extracted_model_data = cleanup_extracted_model_data;

  if (!media_type.empty() && media_type != "video/x-raw" &&
      media_type != "application/vnd.simaai.tensor") {
    throw std::invalid_argument("ModelPack: unsupported media_type: " + media_type);
  }
  if (media_type == "application/vnd.simaai.tensor") {
    cfg.input_format = format;
  } else {
    cfg.input_format = format;
  }
  init_from_config(tar_gz, std::move(cfg));
}

#if defined(SIMA_WITH_OPENCV)
ModelPack::ModelPack(const std::string& tar_gz, const cv::Mat& mat, int max_width, int max_height,
                     int max_depth, bool normalize, std::vector<float> mean,
                     std::vector<float> stddev, const std::string& preproc_next_cpu,
                     PipelineType requested_pipeline_type, const std::string& upstream_name,
                     int num_buffers_cvu, int num_buffers_mla, int queue_max_buffers,
                     int64_t queue_max_time_ns, const std::string& queue_leaky,
                     const std::string& name_suffix, const InferenceTerminalPolicy& terminal_policy,
                     bool cleanup_extracted_model_data)
    : ModelPack(tar_gz, "video/x-raw", (mat.channels() == 1) ? "GRAY" : "BGR", mat.channels(),
                max_width, max_height, max_depth, normalize, std::move(mean), std::move(stddev),
                preproc_next_cpu, requested_pipeline_type, upstream_name, num_buffers_cvu,
                num_buffers_mla, queue_max_buffers, queue_max_time_ns, queue_leaky, name_suffix,
                terminal_policy, cleanup_extracted_model_data) {}
#endif

ModelPack ModelPack::clone_with_buffers(int num_buffers_cvu, int num_buffers_mla) const {
  ModelPack out = *this;
  out.options_.num_buffers_cvu = num_buffers_cvu;
  out.options_.num_buffers_mla = num_buffers_mla;
  return out;
}

ModelPack ModelPack::clone_with_overrides(const std::string& upstream_name,
                                          const std::string& name_suffix) const {
  ModelPack out = *this;
  if (!upstream_name.empty()) {
    out.options_.upstream_name = upstream_name;
  }
  if (!name_suffix.empty()) {
    out.options_.name_suffix = name_suffix;
  }
  return out;
}

void ModelPack::set_model_managed_stage_facts(
    std::optional<bool> processcvu_preproc_single_output_handoff,
    std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags) {
  processcvu_preproc_single_output_handoff_ = processcvu_preproc_single_output_handoff;
  model_managed_route_flags_ = std::move(model_managed_route_flags);
}

void ModelPack::init(const std::string& tar_gz) {
  Config cfg;
  init_from_config(tar_gz, std::move(cfg));
}

void ModelPack::init_from_config(const std::string& tar_gz, Config cfg) {
  options_ = std::move(cfg);
  mpk_contract_.reset();
  route_graph_.reset();
  processcvu_preproc_single_output_handoff_.reset();
  model_managed_route_flags_.reset();

  if (options_.num_buffers_cvu != 4 || options_.num_buffers_mla != 4) {
    throw std::runtime_error(
        "ModelPack: num_buffers_cvu/num_buffers_mla must be 4 for model pipelines.");
  }

  std::string fmt = normalize_format(options_.input_format);
  options_.input_format = fmt;

  if (options_.max_input_width <= 0) {
    options_.max_input_width = 1920;
  }
  if (options_.max_input_height <= 0) {
    options_.max_input_height = 1080;
  }
  if (options_.max_input_depth <= 0) {
    options_.max_input_depth = (options_.input_depth > 0) ? options_.input_depth : 0;
  }

  std::string extracted = extract_and_organize(tar_gz, options_.cleanup_extracted_model_data);
  etc_dir_ = (fs::path(extracted) / kDirConf).string();
  {
    std::string contract_error;
    mpk_contract_ =
        pipeline_internal::sima::load_mpk_contract_from_pack_root(extracted, &contract_error);
    if (mpk_contract_.has_value() && env_truthy_local("SIMA_MPK_CONTRACT_DEBUG")) {
      const auto ordered =
          simaai::neat::pipeline_internal::sima::plugins_in_execution_order(*mpk_contract_);
      std::cerr << "[MPK-CONTRACT][ModelPack] root=" << extracted
                << " parse_status=ok plugins=" << mpk_contract_->plugins.size()
                << " edges=" << mpk_contract_->edges.size() << " order_len=" << ordered.size()
                << "\n";
    }
    if (!mpk_contract_.has_value() && env_truthy_local("SIMA_MPK_CONTRACT_DEBUG") &&
        !contract_error.empty()) {
      std::cerr << "[MPK-CONTRACT][ModelPack] root=" << extracted
                << " parse_status=missing error=" << contract_error << "\n";
    }
  }
  if (mpk_contract_.has_value()) {
    if (const auto* mla =
            simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract_);
        mla && !mla->input_tensors.empty()) {
      const auto& in0 = mla->input_tensors.front();
      const MpkTensorDims mla_input_dims = mpk_dims_from_shape(in0.logical_shape);
      if (options_.input_depth <= 0 && mla_input_dims.depth > 0) {
        options_.input_depth = mla_input_dims.depth;
      }
      if (options_.max_input_depth <= 0 && mla_input_dims.depth > 0) {
        options_.max_input_depth = mla_input_dims.depth;
      }
    }
  }
  {
    const std::string mla_cfg = find_config_path_by_processor("MLA");
    if (!mla_cfg.empty()) {
      if (!mpk_contract_.has_value()) {
        throw std::runtime_error(
            "ModelPack: strict MPK contract required for MLA stage but *_mpk.json is missing");
      }
      if (env_truthy_local("SIMA_ROUTE_DEBUG") || env_truthy_local("SIMA_MPK_CONTRACT_DEBUG")) {
        std::fprintf(
            stderr,
            "[ModelPack] strict MPK MLA contract will be sourced directly from MPK for %s; "
            "skipping legacy config patch\n",
            mla_cfg.c_str());
      }
    }
  }

  // Legacy 0_boxdecoder/0_boxdecode static contract ingestion is disabled for
  // model-managed typed sessions. Boxdecode payload is sourced from model
  // semantics + MLA contracts, and non-model-managed/custom flows may still
  // resolve from explicit stage config JSON at manifest build time.

  pipeline_type_ = get_pipeline_type(options_.requested_pipeline_type);
  int out_c = 0;
  std::string out_fmt;

  if (pipeline_type_ == PipelineType::QuantTess || pipeline_type_ == PipelineType::Quant ||
      pipeline_type_ == PipelineType::Tess || pipeline_type_ == PipelineType::CastTess ||
      pipeline_type_ == PipelineType::Cast) {
    const std::string kernel = pipeline_kernel_name(pipeline_type_);
    const std::string config_name = pipeline_config_name(pipeline_type_);
    const fs::path cfg_path = fs::path(etc_dir_) / config_name;
    if (fs::exists(cfg_path)) {
      if (env_truthy_local("SIMA_ROUTE_DEBUG") || env_truthy_local("SIMA_MPK_CONTRACT_DEBUG")) {
        std::fprintf(
            stderr,
            "[ModelPack] MPK shadow geometry enabled: skipping legacy frontend rewrite/patch "
            "for kernel='%s' config=%s\n",
            kernel.c_str(), cfg_path.string().c_str());
      }
    } else {
      // Typed/manifest routing no longer requires legacy 0_quant*.json files to
      // exist at model load time. Keep this path non-fatal and let runtime
      // stage configuration come from MPK/manifest data.
      if (env_truthy_local("SIMA_ROUTE_DEBUG") || env_truthy_local("SIMA_MPK_CONTRACT_DEBUG")) {
        std::fprintf(stderr,
                     "[WARN] ModelPack: legacy frontend config missing for kernel '%s': %s "
                     "(continuing with typed/manifest config path)\n",
                     kernel.c_str(), cfg_path.string().c_str());
      }
    }
  } else if (pipeline_type_ == PipelineType::Preproc) {
    const fs::path preproc_cfg = fs::path(etc_dir_) / "0_preproc.json";
    if (fs::exists(preproc_cfg)) {
      if (env_truthy_local("SIMA_ROUTE_DEBUG") || env_truthy_local("SIMA_MPK_CONTRACT_DEBUG")) {
        std::fprintf(
            stderr,
            "[ModelPack] MPK shape-only mode: skipping legacy preproc frontend rewrite %s\n",
            preproc_cfg.string().c_str());
      }
    }
  }

  if (!out_fmt.empty())
    options_.input_format = out_fmt;
  if (out_c > 0)
    options_.input_depth = out_c;
  if (out_c > 0 && options_.input_format.empty()) {
    options_.input_format = (out_c == 1) ? "GRAY" : "RGB";
  }
}

std::string ModelPack::find_config_path_by_plugin(const std::string& plugin_id) const {
  if (plugin_id.empty())
    return "";
  return find_config_by_substr(etc_dir_, plugin_id);
}

std::string ModelPack::find_config_path_by_processor(const std::string& processor) const {
  if (processor.empty())
    return "";
  return find_config_by_substr(etc_dir_, processor);
}

const pipeline_internal::sima::RouteGraph& ModelPack::route_graph() const {
  if (!mpk_contract_.has_value()) {
    throw std::runtime_error("ModelPack: strict MPK contract required to derive the route graph");
  }
  if (!route_graph_.has_value()) {
    route_graph_ = pipeline_internal::sima::build_route_graph(*mpk_contract_);
  }
  return *route_graph_;
}

ExecutionPlan ModelPack::execution_plan() const {
  if (!mpk_contract_.has_value()) {
    throw std::runtime_error(
        "ModelPack: strict MPK contract required to derive the typed execution plan");
  }
  return build_execution_plan_from_mpk_contract(*mpk_contract_, pipeline_type_,
                                                model_managed_route_flags_);
}

std::vector<ModelFragment::StageFacts> ModelPack::build_stage_facts(
    const std::vector<ExecutionStage>& stages,
    const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract,
    ModelStage stage_context) const {
  return build_stage_facts_from_execution_plan(
      stages, mpk_contract_, processcvu_preproc_single_output_handoff_, model_managed_route_flags_,
      options_.input_format, options_.input_depth, options_.max_input_width,
      options_.max_input_height, options_.normalize, options_.mean, options_.stddev,
      upstream_handoff_contract, stage_context);
}

std::vector<ModelFragment::StageFacts>
ModelPack::stage_facts_for_model_stage(ModelStage stage) const {
  const ExecutionPlan plan = execution_plan();
  if (stage == ModelStage::Preprocess) {
    return build_stage_facts(plan.pre, std::nullopt, ModelStage::Preprocess);
  }
  if (stage == ModelStage::MlaOnly) {
    std::optional<CompiledProcessCvuContract> upstream_handoff_contract;
    const auto pre_stage_facts = build_stage_facts(plan.pre, std::nullopt, ModelStage::Preprocess);
    for (auto it = pre_stage_facts.rbegin(); it != pre_stage_facts.rend(); ++it) {
      if (it->processcvu_contract.has_value()) {
        upstream_handoff_contract = *it->processcvu_contract;
        break;
      }
    }
    const bool direct_mla_to_boxdecode =
        plan.infer.size() == 1U && plan.infer.front().kind == ExecutionStageKind::Mla &&
        plan.post.size() == 1U && plan.post.front().kind == ExecutionStageKind::BoxDecode;
    return build_stage_facts_from_execution_plan(
        plan.infer, mpk_contract_, processcvu_preproc_single_output_handoff_,
        model_managed_route_flags_, options_.input_format, options_.input_depth,
        options_.max_input_width, options_.max_input_height, options_.normalize, options_.mean,
        options_.stddev, upstream_handoff_contract, ModelStage::MlaOnly, direct_mla_to_boxdecode);
  }
  if (stage == ModelStage::Postprocess) {
    return build_stage_facts(plan.post, std::nullopt, ModelStage::Postprocess);
  }
  if (stage == ModelStage::Full) {
    std::vector<ModelFragment::StageFacts> out;
    auto pre_facts = build_stage_facts(plan.pre, std::nullopt, ModelStage::Preprocess);

    std::optional<CompiledProcessCvuContract> upstream_handoff_contract;
    for (auto it = pre_facts.rbegin(); it != pre_facts.rend(); ++it) {
      if (it->processcvu_contract.has_value()) {
        upstream_handoff_contract = *it->processcvu_contract;
        break;
      }
    }

    const bool direct_mla_to_boxdecode =
        plan.infer.size() == 1U && plan.infer.front().kind == ExecutionStageKind::Mla &&
        plan.post.size() == 1U && plan.post.front().kind == ExecutionStageKind::BoxDecode;
    auto infer_facts = build_stage_facts_from_execution_plan(
        plan.infer, mpk_contract_, processcvu_preproc_single_output_handoff_,
        model_managed_route_flags_, options_.input_format, options_.input_depth,
        options_.max_input_width, options_.max_input_height, options_.normalize, options_.mean,
        options_.stddev, upstream_handoff_contract, ModelStage::MlaOnly, direct_mla_to_boxdecode);
    auto post_facts = build_stage_facts(plan.post, std::nullopt, ModelStage::Postprocess);

    out.reserve(pre_facts.size() + infer_facts.size() + post_facts.size());
    out.insert(out.end(), pre_facts.begin(), pre_facts.end());
    out.insert(out.end(), infer_facts.begin(), infer_facts.end());
    out.insert(out.end(), post_facts.begin(), post_facts.end());
    return out;
  }
  return build_stage_facts(flatten_execution_plan(plan, stage), std::nullopt, stage);
}

ModelFragment ModelPack::fragment(ModelStage stage) const {
  const ExecutionPlan plan = execution_plan();
  std::vector<ExecutionStage> sel = flatten_execution_plan(plan, stage);
  if (sel.empty())
    return {};

  std::string upstream;
  if (stage == ModelStage::Preprocess || stage == ModelStage::Full) {
    upstream = options_.upstream_name.empty() ? upstream_name_for_stage(plan, stage)
                                              : options_.upstream_name;
  } else {
    upstream = upstream_name_for_stage(plan, stage);
  }
  std::vector<ModelFragment::StageFacts> stage_facts;
  if (stage == ModelStage::Preprocess || stage == ModelStage::MlaOnly ||
      stage == ModelStage::Postprocess || stage == ModelStage::Full) {
    stage_facts = stage_facts_for_model_stage(stage);
  } else {
    stage_facts = build_stage_facts(sel, std::nullopt, stage);
  }
  return build_fragment_linear(sel, upstream, options_.num_buffers_cvu, options_.num_buffers_mla,
                               options_.name_suffix, mpk_contract_, std::move(stage_facts));
}

std::string ModelPack::backend_fragment(ModelStage stage) const {
  return fragment(stage).gst;
}

std::vector<std::shared_ptr<Node>> ModelPack::to_nodes(ModelStage stage) const {
  ModelFragment frag = fragment(stage);
  if (frag.gst.empty())
    return {};
  const std::string label = stage_label(stage);
  return make_fragment_nodes(frag, label);
}

std::vector<std::shared_ptr<Node>>
ModelPack::infer_block(const std::string& upstream_name,
                       std::shared_ptr<const ModelLineageBinding> model_lineage) const {
  const ExecutionPlan plan = execution_plan();
  if (plan.infer.empty()) {
    throw std::runtime_error("ModelPack::infer_block: pipeline has no infer stages");
  }
  std::vector<ExecutionStage> infer_seq = plan.infer;

  if (internal::has_terminal_policy(options_.terminal_policy)) {
    const std::size_t terminal_idx =
        resolve_terminal_index_or_throw(infer_seq, options_.terminal_policy);
    if (terminal_idx + 1 < infer_seq.size()) {
      std::ostringstream log;
      log << "[ModelPack] inference terminal stage index=" << terminal_idx
          << " name=" << infer_seq[terminal_idx].stage_name
          << " plugin=" << infer_seq[terminal_idx].plugin_id
          << " processor=" << infer_seq[terminal_idx].processor
          << " dropped_tail=" << (infer_seq.size() - (terminal_idx + 1));
      std::cerr << log.str() << "\n";
    }
    infer_seq.resize(terminal_idx + 1);
    if (infer_seq.empty()) {
      throw std::runtime_error("Inference terminal policy removed all infer stages");
    }
  }
  validate_infer_sequence_or_throw(infer_seq);

  // Keep terminal MLA stage self-consistent after infer truncation.
  // The terminal MLA stage must remain fully described by the strict MPK
  // contract after infer trimming.
  if (internal::has_terminal_policy(options_.terminal_policy)) {
    (void)validate_terminal_mla_metadata(infer_seq, mpk_contract_);
  }

  std::string upstream = upstream_name;
  if (upstream.empty()) {
    upstream = options_.upstream_name.empty() ? kDefaultPreviousNodeName : options_.upstream_name;
  }

  std::optional<CompiledProcessCvuContract> upstream_handoff_contract;
  const auto pre_stage_facts = build_stage_facts(plan.pre, std::nullopt, ModelStage::Preprocess);
  for (auto it = pre_stage_facts.rbegin(); it != pre_stage_facts.rend(); ++it) {
    if (it->processcvu_contract.has_value()) {
      upstream_handoff_contract = *it->processcvu_contract;
      break;
    }
  }
  const bool direct_mla_to_boxdecode =
      infer_seq.size() == 1U && infer_seq.front().kind == ExecutionStageKind::Mla &&
      plan.post.size() == 1U && plan.post.front().kind == ExecutionStageKind::BoxDecode;
  auto stage_facts = build_stage_facts_from_execution_plan(
      infer_seq, mpk_contract_, processcvu_preproc_single_output_handoff_,
      model_managed_route_flags_, options_.input_format, options_.input_depth,
      options_.max_input_width, options_.max_input_height, options_.normalize, options_.mean,
      options_.stddev, upstream_handoff_contract, ModelStage::MlaOnly, direct_mla_to_boxdecode);
  ModelFragment frag =
      build_fragment_linear(infer_seq, upstream, options_.num_buffers_cvu, options_.num_buffers_mla,
                            options_.name_suffix, mpk_contract_, std::move(stage_facts));
  if (frag.gst.empty())
    return {};
  return make_fragment_nodes(frag, "infer", std::move(model_lineage));
}

std::string ModelPack::apply_name_suffix(const std::string& base) const {
  if (options_.name_suffix.empty())
    return base;
  return base + options_.name_suffix;
}

bool ModelPack::has_terminal_policy() const {
  return internal::has_terminal_policy(options_.terminal_policy);
}

InputOptions ModelPack::input_appsrc_options(bool tensor_mode) const {
  InputOptions opt;

  if (tensor_mode) {
    const bool quant_like_pre =
        pipeline_type_ == PipelineType::Quant || pipeline_type_ == PipelineType::Tess ||
        pipeline_type_ == PipelineType::QuantTess || pipeline_type_ == PipelineType::CastTess ||
        pipeline_type_ == PipelineType::Cast;
    std::string tensor_format =
        options_.input_format.empty() ? std::string("FP32") : options_.input_format;
    const std::string tensor_format_up = to_upper(tensor_format);
    if (quant_like_pre && (tensor_format_up == "FP32" || tensor_format_up == "FLOAT32" ||
                           tensor_format_up == "EVXX_FLOAT32")) {
      // processcvu model-managed quant-like stages negotiate EVXX float tokens on sink caps.
      tensor_format = "EVXX_FLOAT32";
    }

    opt.payload_type = PayloadType::Tensor;
    opt.format = normalize_caps_format_for_media(resolve_input_media_type(opt), tensor_format);
    const std::string input_format_up = to_upper(opt.format.str());
    if (input_format_up.find("BF16") != std::string::npos ||
        input_format_up.find("BFLOAT16") != std::string::npos) {
      opt.buffer_name = "ifm0";
    }
    opt.max_width = options_.max_input_width;
    opt.max_height = options_.max_input_height;
    opt.max_depth = options_.max_input_depth;
    return opt;
  }

  std::string fmt = options_.input_format;
  if (fmt.empty())
    fmt = "RGB";
  const std::string fmt_upper = to_upper(fmt);
  const bool tensor_like_format =
      fmt_upper.find("FP") != std::string::npos || fmt_upper.find("FLOAT") != std::string::npos ||
      fmt_upper.find("BF16") != std::string::npos ||
      fmt_upper.find("BFLOAT16") != std::string::npos ||
      fmt_upper.find("INT8") != std::string::npos || fmt_upper.find("INT16") != std::string::npos ||
      fmt_upper.find("INT32") != std::string::npos ||
      fmt_upper.find("UINT8") != std::string::npos ||
      fmt_upper.find("UINT16") != std::string::npos ||
      fmt_upper.find("UINT32") != std::string::npos;
  if (tensor_like_format) {
    // Accuracy/debug pipelines may request video appsrc explicitly; keep caps
    // valid even if the model-level input_format currently carries tensor dtype.
    fmt = "RGB";
  }
  if (fmt == "GRAY")
    fmt = "GRAY8";

  opt.payload_type = PayloadType::Image;
  opt.format = fmt;
  opt.max_width = options_.max_input_width;
  opt.max_height = options_.max_input_height;
  opt.max_depth = options_.max_input_depth;
  return opt;
}

} // namespace simaai::neat::internal
