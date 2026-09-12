#include "model/internal/ModelPack.h"
#include "model/internal/ModelManagedPreprocEnvelope.h"
#include "pipeline/internal/sima/ProcessCvuFamily.h"
#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include "builder/NodeContractConfigurable.h"
#include "builder/CompiledChildStageProvider.h"
#include "builder/NodeContractProvider.h"
#include "gst/GstHelpers.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/internal/DmabufEligibility.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TempJsonFileUtil.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
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
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
  if (raw.find("tvm") != std::string::npos || raw.find("a65") != std::string::npos) {
    return ExecutionStageKind::HostTvm;
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
  case ExecutionStageKind::HostTvm:
    return "a65";
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

static std::string require_stage_factory(ExecutionStageKind kind,
                                         const bool physical_cvu_command = false) {
  const char* factory = nullptr;
  if (physical_cvu_command) {
    factory = "neatprocesscvu";
  } else
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
    case ExecutionStageKind::HostTvm:
      factory = "neatprocesstvm";
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

static bool append_model_paths_if_exists(json& json_data, const std::string& append_path) {
  if (json_data.contains("simaai__params") && json_data["simaai__params"].contains("model_path")) {
    std::string model_path = json_data["simaai__params"]["model_path"];
    json_data["simaai__params"]["model_path"] = append_path + "/share/" + model_path;
    return true;
  } else if (json_data.contains("model_info") && json_data["model_info"].contains("path")) {
    std::string model_info_path = json_data["model_info"]["path"];
    json_data["model_info"]["path"] = append_path + "/lib/" + model_info_path;
    return true;
  }
  return false;
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

static std::optional<MlaRuntimeProperties> read_mla_runtime_properties_from_mpk_contract(
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract,
    const pipeline_internal::sima::MpkPluginIoContract* exact_mla_stage = nullptr) {
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
  const auto* mla_stage = exact_mla_stage != nullptr
                              ? exact_mla_stage
                              : pipeline_internal::sima::get_mla_stage_io_contract(*mpk_contract);
  if (mla_stage && !mla_stage->executable.empty()) {
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

static pipeline_internal::DmabufPlanCompileResult
compile_dmabuf_plan_execution_plan(const pipeline_internal::sima::MpkContract& mpk_contract) {
  if (mpk_contract.mpk_json_path.empty()) {
    return pipeline_internal::try_compile_dmabuf_plan(
        std::filesystem::path{}, std::vector<pipeline_internal::MlaExecutableArtifact>{});
  }
  const auto stages = pipeline_internal::sima::get_mla_stage_io_contracts(mpk_contract);
  std::vector<pipeline_internal::MlaExecutableArtifact> artifacts;
  artifacts.reserve(stages.size());
  for (const auto* stage : stages) {
    const auto runtime = read_mla_runtime_properties_from_mpk_contract(mpk_contract, stage);
    if (!runtime.has_value() || runtime->model_path.empty()) {
      return pipeline_internal::try_compile_dmabuf_plan(
          mpk_contract.mpk_json_path, std::vector<pipeline_internal::MlaExecutableArtifact>{});
    }
    artifacts.push_back({stage->name, stage->executable, runtime->model_path});
  }
  std::vector<pipeline_internal::HostTvmExecutableArtifact> host_artifacts;
  fs::path package_root = fs::path(mpk_contract.mpk_json_path).parent_path();
  if (package_root.filename() == "etc") {
    package_root = package_root.parent_path();
  }
  for (const auto& stage : mpk_contract.plugins) {
    if (to_upper(stage.processor) != "A65") {
      continue;
    }
    fs::path resolved;
    const fs::path raw(stage.executable);
    const std::array<fs::path, 4> candidates = {
        raw.is_absolute() ? raw : package_root / "lib" / raw,
        raw.is_absolute() ? raw : package_root / raw,
        raw.is_absolute() ? raw : package_root / "share" / raw, raw};
    for (const auto& candidate : candidates) {
      std::error_code ec;
      if (!candidate.empty() && fs::is_regular_file(candidate, ec) && !ec) {
        resolved = candidate;
        break;
      }
    }
    host_artifacts.push_back(
        {stage.name, stage.executable, resolved.empty() ? candidates.front() : resolved});
  }
  return pipeline_internal::try_compile_dmabuf_plan(mpk_contract.mpk_json_path, artifacts,
                                                    host_artifacts);
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

// Model paths are baked into rewritten JSON, so the selected base must not depend on a later cwd.
static fs::path canonical_base(const fs::path& base) {
  std::error_code ec;
  const fs::path resolved = fs::weakly_canonical(base, ec);
  if (!ec && !resolved.empty())
    return resolved;
  ec.clear();
  const fs::path absolute = fs::absolute(base, ec);
  return ec ? base : absolute;
}

// True for a mount that exists to hold system files rather than data. Filling any of these is the
// failure automatic selection has to avoid, and none of them is a plausible model store.
static bool system_mount_point(const std::string& mount_point) {
  static constexpr const char* kSystemMounts[] = {"/",     "/boot", "/efi",  "/usr",
                                                  "/var",  "/etc",  "/opt",  "/home",
                                                  "/root", "/srv",  "/snap", "/recovery"};
  for (const char* system : kSystemMounts) {
    // Prefix match so /boot also excludes /boot/efi and /boot/firmware.
    const std::string base(system);
    if (mount_point == base || (base != "/" && mount_point.rfind(base + "/", 0) == 0))
      return true;
  }
  return mount_point == "/";
}

// Exact token match over a comma-separated /proc/mounts option list. Substring matching would
// confuse "ro" with "relatime" and is blind to any option that is not listed first.
static bool mount_has_option(const std::string& options, const std::string& option) {
  for (std::size_t pos = 0; pos <= options.size();) {
    const std::size_t end = options.find(',', pos);
    const std::size_t len = (end == std::string::npos ? options.size() : end) - pos;
    if (options.compare(pos, len, option) == 0)
      return true;
    if (end == std::string::npos)
      break;
    pos = end + 1;
  }
  return false;
}

// Defined outside this anonymous namespace so the mount filtering is reachable from its test.
static std::vector<std::string> nvme_model_bases() {
  std::ifstream mounts("/proc/mounts");
  return nvme_model_bases_from_mounts(mounts);
}

// Selects and pins the per-process extraction root. Capacity is deliberately not a criterion: an
// archive's inflated size cannot be derived from its compressed size, so any pre-inflation space
// requirement would be a guess. The loader enforces the real budget per chunk while inflating and
// again from the manifest before extracting. Selection therefore asks only whether a candidate is
// writable, which makes an eligible NVMe unconditional: a full one fails the load rather than
// silently moving hundreds of megabytes of writes onto eMMC.
static std::string modelpack_output_root(bool cleanup_enabled_request) {
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
    std::string reason;
    for (const auto& nvme_base : nvme_model_bases()) {
      reason.clear();
      if (is_writable_dir(nvme_base, &reason))
        return nvme_base;
      rejected.push_back(nvme_base + " (" + reason + ")");
    }

    const fs::path preferred(kDefaultBaseOutputDir);
    reason.clear();
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
    oss << "ModelPack: output_storage_unavailable: no writable extraction root. Tried:";
    for (const auto& item : rejected) {
      oss << " [" << item << "]";
    }
    oss << ". Set SIMA_MPK_EXTRACT_ROOT to a writable filesystem with enough space.";
    throw std::runtime_error(oss.str());
  };

  static std::mutex root_mu;
  static std::string root;
  std::lock_guard<std::mutex> lock(root_mu);
  if (root.empty()) {
    const fs::path base = canonical_base(choose_base());
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

// Stable across processes and library builds, unlike std::hash.
static std::string fnv1a_hex(const std::string& text) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const unsigned char c : text) {
    h ^= static_cast<std::uint64_t>(c);
    h *= 1099511628211ULL;
  }
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << h;
  return oss.str();
}

// Archive identity, not basename: the loader clears its package root before extracting, so two
// archives sharing a basename would otherwise evict each other inside the process root.
static std::string identity_dir_name(const std::string& cache_key) {
  return "pkg_" + fnv1a_hex(cache_key);
}

struct OrganizedModelPackage {
  std::string package_root;
  std::string logical_package_name;
};

static std::unordered_map<std::string, OrganizedModelPackage>& modelpack_extract_cache() {
  static std::unordered_map<std::string, OrganizedModelPackage> cache;
  return cache;
}

static std::mutex& modelpack_extract_cache_mutex() {
  static std::mutex mu;
  return mu;
}

static OrganizedModelPackage extract_and_organize(const std::string& tar_path,
                                                  bool cleanup_extracted_model_data) {
  {
    std::error_code ec;
    const fs::path direct_root(tar_path);
    if (fs::exists(direct_root, ec) && !ec && fs::is_directory(direct_root, ec) && !ec &&
        extracted_layout_ready(direct_root)) {
      return {.package_root = direct_root.string(),
              .logical_package_name = direct_root.filename().string()};
    }
  }

  const std::string cache_key = archive_cache_key(tar_path);
  const std::string identity_dir = identity_dir_name(cache_key);

  simaai::neat::internal::ModelArchiveLoaderOptions opt;
  // Runtime model packs may include auxiliary build/report artifacts.
  opt.reject_unsupported_file_types = false;
  opt.min_output_free_bytes = modelpack_extract_free_reserve_bytes();
  // Archive identity is already carried by the parent pkg_<hash> directory. Repeating the logical
  // archive name here needlessly lengthens every rewritten runtime artifact path and can exceed
  // downstream fixed-width transport fields. The manifest retains the logical package name.
  opt.physical_package_leaf = "p";
  try {
    std::lock_guard<std::mutex> lock(modelpack_extract_cache_mutex());
    // Applied on every path, including a cache hit, so the cleanup policy a caller asks for is
    // honoured whether or not this load is the one that extracts.
    const std::string process_root = modelpack_output_root(cleanup_extracted_model_data);

    auto& cache = modelpack_extract_cache();
    const auto found = cache.find(cache_key);
    if (found != cache.end() && extracted_layout_ready(fs::path(found->second.package_root))) {
      return found->second;
    }

    // One filesystem holds the inflated snapshot and the package, so the loader's free-space
    // budget covers both coexisting.
    opt.staging_base = process_root;
    const auto extracted = simaai::neat::internal::ModelArchiveLoader::extract(
        tar_path, (fs::path(process_root) / identity_dir).string(), opt);
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
        if (append_model_paths_if_exists(cfg, target_dir.string())) {
          write_json_file_atomic(entry.path(), cfg, "ModelPack");
        }
      }
    } catch (...) {
      std::error_code cleanup_ec;
      fs::remove_all(target_dir, cleanup_ec);
      throw;
    }

    OrganizedModelPackage organized{.package_root = target_dir.string(),
                                    .logical_package_name = extracted.manifest.package_name};
    cache[cache_key] = organized;
    return organized;
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
  const auto* mla_stage = pipeline_internal::sima::get_first_mla_stage_io_contract(contract);
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
mla_ranks_in_order(const pipeline_internal::sima::MpkContract& contract,
                   const std::vector<std::size_t>& ordered) {
  std::vector<std::size_t> result;
  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    const auto index = ordered[rank];
    if (index >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[index];
    if (to_upper(stage.processor) == "MLA" ||
        canonical_execution_stage_kind(stage.kernel) == ExecutionStageKind::Mla) {
      result.push_back(rank);
    }
  }
  return result;
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

static bool any_candidate_has_kind(const pipeline_internal::sima::MpkContract& contract,
                                   const std::vector<std::size_t>& candidates,
                                   ExecutionStageKind kind) {
  for (const std::size_t idx : candidates) {
    if (idx >= contract.plugins.size()) {
      continue;
    }
    const auto& stage = contract.plugins[idx];
    if (canonical_execution_stage_kind(!stage.kernel.empty() ? stage.kernel : stage.name) == kind) {
      return true;
    }
  }
  return false;
}

static bool
contract_graph_has_fused_detessdequant(const pipeline_internal::sima::MpkContract& contract) {
  return std::any_of(
      contract.graph.nodes.begin(), contract.graph.nodes.end(), [](const auto& node) {
        return node.kind == pipeline_internal::sima::MpkGraphNodeKind::FusedDetessDequant;
      });
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
  case ExecutionStageKind::HostTvm:
    return "processtvm";
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
  if (kind == ExecutionStageKind::HostTvm) {
    return "A65";
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
  case ExecutionStageKind::HostTvm:
    return "tvm";
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

static std::vector<ExecutionStage>
make_mla_stages_from_contract(const pipeline_internal::sima::MpkContract& contract,
                              std::size_t order_index) {
  std::vector<ExecutionStage> result;
  for (const auto* mla_stage : pipeline_internal::sima::get_mla_stage_io_contracts(contract)) {
    ExecutionStage stage;
    stage.order_index = order_index++;
    stage.mpk_plugin_index =
        pipeline_internal::sima::find_plugin_index_by_name_or_id(contract, mla_stage->name);
    stage.stage_name = !mla_stage->name.empty()
                           ? mla_stage->name
                           : std::string(default_stage_name(ExecutionStageKind::Mla));
    stage.factory_name = require_stage_factory(ExecutionStageKind::Mla);
    stage.plugin_id = plugin_id_for_stage_kind(ExecutionStageKind::Mla);
    stage.processor = processor_for_stage_kind(ExecutionStageKind::Mla);
    stage.kernel = kernel_for_stage_kind(ExecutionStageKind::Mla);
    stage.kind = ExecutionStageKind::Mla;
    result.push_back(std::move(stage));
  }
  return result;
}

static std::vector<ExecutionStage> make_post_stages_from_contract(
    const pipeline_internal::sima::MpkContract& contract, const std::vector<std::size_t>& ordered,
    std::optional<std::size_t> mla_rank,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& route_flags,
    const std::vector<ExecutionStageKind>& route_post_kinds, std::size_t order_index) {
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

  const auto append_stage = [&](ExecutionStageKind k, const std::vector<std::size_t>& cands,
                                std::size_t ord) {
    ExecutionStage stage;
    stage.order_index = ord;
    stage.mpk_plugin_index = first_index(cands);
    stage.stage_name = pick_stage_name(contract, cands, k);
    stage.factory_name = require_stage_factory(k);
    stage.plugin_id = plugin_id_for_stage_kind(k);
    stage.processor = processor_for_stage_kind(k);
    stage.kernel = kernel_for_stage_kind(k);
    stage.kind = k;
    stages.push_back(std::move(stage));
  };

  if (!route_post_kinds.empty()) {
    std::size_t ord = order_index;
    for (const auto kind : route_post_kinds) {
      switch (kind) {
      case ExecutionStageKind::Detess:
        append_stage(kind, detess_indices, ord++);
        break;
      case ExecutionStageKind::DetessCast:
        append_stage(kind, !detess_indices.empty() ? detess_indices : cast_indices, ord++);
        break;
      case ExecutionStageKind::DetessDequant:
        append_stage(kind, !dequant_indices.empty() ? dequant_indices : detess_indices, ord++);
        break;
      case ExecutionStageKind::Dequant:
        append_stage(kind, dequant_indices, ord++);
        break;
      case ExecutionStageKind::BoxDecode:
        append_stage(kind, boxdecode_indices, ord++);
        break;
      case ExecutionStageKind::Cast:
        append_stage(kind, cast_indices, ord++);
        break;
      case ExecutionStageKind::Unknown:
      case ExecutionStageKind::Preproc:
      case ExecutionStageKind::Quant:
      case ExecutionStageKind::Tess:
      case ExecutionStageKind::QuantTess:
      case ExecutionStageKind::CastTess:
      case ExecutionStageKind::Mla:
      case ExecutionStageKind::HostTvm:
        break;
      }
    }
    return stages;
  }

  const bool route_requests_boxdecode = route_flags.has_value() && route_flags->boxdecode_selected;
  if (!boxdecode_indices.empty() || route_requests_boxdecode) {
    append_stage(ExecutionStageKind::BoxDecode,
                 boxdecode_indices.empty() ? std::vector<std::size_t>{} : boxdecode_indices,
                 order_index);
  } else if (!detess_indices.empty() && !cast_indices.empty() && dequant_indices.empty()) {
    append_stage(ExecutionStageKind::DetessCast, detess_indices, order_index);
  } else if (!detess_indices.empty() && !dequant_indices.empty()) {
    if (any_candidate_has_kind(contract, dequant_indices, ExecutionStageKind::DetessDequant) ||
        contract_graph_has_fused_detessdequant(contract)) {
      // The fused MPK graph is the source of truth for whether detess/dequant operate on the same
      // logical outputs. Equal raw plugin counts alone are not enough: heterogeneous egress can
      // have one detess-only output and one dequant-only output with matching counts.
      append_stage(ExecutionStageKind::DetessDequant, dequant_indices, order_index);
    } else {
      // Heterogeneous egress (e.g. one MLA output published dense/native and one dequant-only):
      // the route planner does NOT fuse -- it produces separate Detess and Dequantize regions --
      // so emit matching separate Detess + Dequant stages. Each then carries its own canonical
      // compiled contract that its region materializes against (otherwise the Detess region's
      // require_model_managed_postprocess_contract(Detess) finds only a fused DetessDequant
      // contract and throws "requires a canonical compiled contract").
      append_stage(ExecutionStageKind::Detess, detess_indices, order_index);
      append_stage(ExecutionStageKind::Dequant, dequant_indices, order_index + 1U);
    }
  } else if (!detess_indices.empty()) {
    append_stage(ExecutionStageKind::Detess, detess_indices, order_index);
  } else if (!dequant_indices.empty()) {
    append_stage(ExecutionStageKind::Dequant, dequant_indices, order_index);
  } else if (!cast_indices.empty()) {
    append_stage(ExecutionStageKind::Cast, cast_indices, order_index);
  }
  return stages;
}

static ExecutionPlan build_execution_plan_from_mpk_contract(
    const pipeline_internal::sima::MpkContract& contract, PipelineType requested_pipeline_type,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& route_flags,
    const std::vector<ExecutionStageKind>& route_post_kinds) {
  ExecutionPlan plan;
  const auto ordered = ordered_plugin_indices(contract);
  const auto mla_ranks = mla_ranks_in_order(contract, ordered);
  const auto first_mla_rank = mla_ranks.empty() ? std::optional<std::size_t>{}
                                                : std::optional<std::size_t>{mla_ranks.front()};
  const auto last_mla_rank = mla_ranks.empty() ? std::optional<std::size_t>{}
                                               : std::optional<std::size_t>{mla_ranks.back()};
  std::size_t stage_order = 0U;
  if (const auto pre = make_pre_stage_from_contract(contract, ordered, first_mla_rank,
                                                    requested_pipeline_type, stage_order);
      pre.has_value()) {
    plan.pre.push_back(*pre);
    ++stage_order;
  }
  auto mla_stages = make_mla_stages_from_contract(contract, stage_order);
  stage_order += mla_stages.size();
  if (!mla_stages.empty()) {
    plan.infer.insert(plan.infer.end(), std::make_move_iterator(mla_stages.begin()),
                      std::make_move_iterator(mla_stages.end()));
  }
  const auto post = make_post_stages_from_contract(contract, ordered, last_mla_rank, route_flags,
                                                   route_post_kinds, stage_order);
  if (!post.empty()) {
    plan.post.insert(plan.post.end(), post.begin(), post.end());
  }
  return plan;
}

static ExecutionStageKind
model_plan_stage_kind(const pipeline_internal::sima::static_contract::OpSpec& op) {
  using pipeline_internal::sima::static_contract::OpKind;
  switch (op.kind) {
  case OpKind::Cast:
    return ExecutionStageKind::Cast;
  case OpKind::Quantize:
    return ExecutionStageKind::Quant;
  case OpKind::Tessellate:
    return ExecutionStageKind::Tess;
  case OpKind::Mla:
    return ExecutionStageKind::Mla;
  case OpKind::Detessellate:
    return ExecutionStageKind::Detess;
  case OpKind::Dequantize:
    return ExecutionStageKind::Dequant;
  case OpKind::HostTvm:
    return ExecutionStageKind::HostTvm;
  case OpKind::Pack:
    throw std::runtime_error("ModelPack: model operation '" + op.name +
                             "' requires a registered material Pack implementation");
  case OpKind::Unpack:
  case OpKind::Slice:
  case OpKind::Reshape:
  case OpKind::PassThrough:
    throw std::runtime_error("ModelPack: model plan contains non-executable operation '" + op.name +
                             "'; views/publication must be storage facts");
  }
  throw std::runtime_error("ModelPack: model operation has no exact stage implementation");
}

static ExecutionPlan build_execution_plan_from_model_plan(
    const pipeline_internal::sima::static_contract::ModelExecutionPlan& command_plan,
    const pipeline_internal::sima::static_contract::PhysicalExecutionPlan& physical_plan,
    const pipeline_internal::sima::static_contract::FrameSlotArenaPlan& arena_plan,
    const pipeline_internal::sima::MpkContract& packaging_contract) {
  using namespace pipeline_internal::sima::static_contract;
  struct RenderCohort {
    PhysicalCohortId id = 0U;
    PhysicalEngine engine = PhysicalEngine::Cvu;
    PhysicalCommandRole role = PhysicalCommandRole::NonCvu;
    std::string implementation_id;
    std::uint32_t graph_id = 0U;
    std::uint32_t batch_size = 0U;
    std::uint32_t maximum_members = 0U;
    std::uint64_t first_rank = 0U;
    std::vector<PhysicalCommandId> commands;
    std::vector<PhysicalCommandMember> members;
    std::set<PhysicalCohortId> predecessors;
    std::set<PhysicalCohortId> successors;
  };

  std::map<PhysicalCohortId, RenderCohort> cohorts;
  for (const auto& command : physical_plan.commands) {
    if (command.members.empty()) {
      throw std::runtime_error("ModelPack: physical command has invalid semantic membership");
    }
    auto [it, inserted] = cohorts.try_emplace(command.cohort_id);
    auto& cohort = it->second;
    if (inserted) {
      cohort.id = command.cohort_id;
      cohort.engine = command.engine;
      cohort.role = command.role;
      cohort.implementation_id = command.implementation_id;
      cohort.graph_id = command.graph_id;
      cohort.batch_size = command.batch_size;
      cohort.maximum_members = command.maximum_members;
      cohort.first_rank = command.topological_rank;
    } else if (cohort.engine != command.engine || cohort.role != command.role ||
               cohort.implementation_id != command.implementation_id ||
               cohort.graph_id != command.graph_id || cohort.batch_size != command.batch_size ||
               cohort.maximum_members != command.maximum_members) {
      throw std::runtime_error(
          "ModelPack: one render cohort contains incompatible physical commands");
    }
    cohort.first_rank = std::min(cohort.first_rank, command.topological_rank);
    cohort.commands.push_back(command.id);
    for (const auto& member : command.members) {
      if (member.semantic_chain.empty() ||
          std::any_of(
              member.semantic_chain.begin(), member.semantic_chain.end(), [&](const auto op_id) {
                return op_id >= command_plan.ops().size() || command_plan.ops()[op_id].id != op_id;
              })) {
        throw std::runtime_error("ModelPack: physical member references a missing semantic op");
      }
      cohort.members.push_back(member);
    }
  }
  for (const auto& command : physical_plan.commands) {
    auto& cohort = cohorts.at(command.cohort_id);
    for (const auto predecessor : command.predecessors) {
      if (predecessor >= physical_plan.commands.size()) {
        throw std::runtime_error("ModelPack: physical predecessor is out of range");
      }
      const auto predecessor_cohort = physical_plan.commands[predecessor].cohort_id;
      if (predecessor_cohort == cohort.id) {
        throw std::runtime_error(
            "ModelPack: capacity chunks in one render cohort depend on each other");
      }
      cohort.predecessors.emplace(predecessor_cohort);
      cohorts.at(predecessor_cohort).successors.emplace(cohort.id);
    }
  }

  using Ready = std::tuple<std::uint64_t, PhysicalCohortId>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<>> ready;
  std::map<PhysicalCohortId, std::size_t> indegree;
  for (const auto& [id, cohort] : cohorts) {
    indegree.emplace(id, cohort.predecessors.size());
    if (cohort.predecessors.empty()) {
      ready.emplace(cohort.first_rank, id);
    }
  }
  std::vector<PhysicalCohortId> ordered;
  ordered.reserve(cohorts.size());
  while (!ready.empty()) {
    const auto id = std::get<1>(ready.top());
    ready.pop();
    ordered.push_back(id);
    for (const auto successor : cohorts.at(id).successors) {
      auto& count = indegree.at(successor);
      if (--count == 0U) {
        ready.emplace(cohorts.at(successor).first_rank, successor);
      }
    }
  }
  if (ordered.size() != cohorts.size()) {
    throw std::runtime_error("ModelPack: physical render cohorts contain a dependency cycle");
  }

  // A frame-arena buffer is the live-value catalogue.  Prove once, at setup,
  // that the deterministic cohort order respects every dependency and that
  // every non-public value read by a later command remains addressable in the
  // retained arena.  This admits real branch/join DAGs without pretending they
  // are graph-theoretic chains and without a bounce/copy fallback.
  std::vector<bool> command_complete(physical_plan.commands.size(), false);
  std::vector<bool> value_ready(command_plan.values().size(), false);
  for (const auto value_id : command_plan.model_inputs()) {
    if (value_id >= value_ready.size()) {
      throw std::runtime_error("ModelPack: model input is out of range");
    }
    value_ready[value_id] = true;
  }
  const auto root_value = [&](pipeline_internal::sima::static_contract::ValueId value_id) {
    std::size_t remaining = command_plan.values().size() + 1U;
    while (remaining-- > 0U) {
      const auto* value = command_plan.value(value_id);
      if (!value) {
        throw std::runtime_error("ModelPack: physical command input is out of range");
      }
      if (!value->read_expression.has_value()) {
        return value_id;
      }
      value_id = value->read_expression->source_value_id;
    }
    throw std::runtime_error("ModelPack: model value view chain contains a cycle");
  };
  // A direct Pack publishes the carrier written by its component producers;
  // it has no physical command whose completion could mark the parent ready.
  // The admitted plan already proves their exact non-overlapping placement.
  std::vector<const OpSpec*> direct_pack_producers(command_plan.values().size(), nullptr);
  for (const auto& op : command_plan.ops()) {
    const auto* pack = std::get_if<PackOpConfig>(&op.config);
    if (op.kind == OpKind::Pack && pack && !pack->materializes) {
      direct_pack_producers.at(op.outputs.front()) = &op;
    }
  }
  const auto is_value_ready = [&](const auto& self, const ValueId value_id,
                                  const std::size_t remaining) -> bool {
    const auto root = root_value(value_id);
    if (value_ready[root]) {
      return true;
    }
    const auto* pack = direct_pack_producers[root];
    if (!pack) {
      return false;
    }
    if (remaining == 0U) {
      throw std::runtime_error("ModelPack: direct Pack value dependencies contain a cycle");
    }
    const bool ready = std::all_of(pack->inputs.begin(), pack->inputs.end(), [&](const auto input) {
      return self(self, input, remaining - 1U);
    });
    if (ready) {
      value_ready[root] = true;
    }
    return ready;
  };
  for (const auto cohort_id : ordered) {
    const auto& cohort = cohorts.at(cohort_id);
    for (const auto command_id : cohort.commands) {
      if (command_id >= physical_plan.commands.size()) {
        throw std::runtime_error("ModelPack: physical cohort command is out of range");
      }
      const auto& command = physical_plan.commands[command_id];
      for (const auto predecessor : command.predecessors) {
        if (predecessor >= command_complete.size() || !command_complete[predecessor]) {
          throw std::runtime_error(
              "ModelPack: deterministic physical schedule violates a command dependency");
        }
      }
      for (const auto input : command.inputs) {
        const auto root = root_value(input);
        if (!is_value_ready(is_value_ready, root, command_plan.values().size())) {
          throw std::runtime_error(
              "ModelPack: physical command reads a value before its producer completes");
        }
        const bool public_input =
            std::find(command_plan.model_inputs().begin(), command_plan.model_inputs().end(),
                      root) != command_plan.model_inputs().end();
        if (!public_input && arena_plan.region(root) == nullptr) {
          throw std::runtime_error(
              "ModelPack: live internal command input is not retained in the frame arena");
        }
      }
    }
    for (const auto command_id : cohort.commands) {
      const auto& command = physical_plan.commands[command_id];
      for (const auto output : command.outputs) {
        const auto root = root_value(output);
        const bool detached_terminal_mla_output =
            command.engine == pipeline_internal::sima::static_contract::PhysicalEngine::Mla &&
            arena_plan.is_detached_root(root);
        if (root >= value_ready.size() ||
            (arena_plan.region(root) == nullptr && !detached_terminal_mla_output)) {
          throw std::runtime_error(
              "ModelPack: physical command output has no retained frame-arena value");
        }
        value_ready[root] = true;
      }
      command_complete[command_id] = true;
    }
  }
  for (const auto& output : command_plan.model_outputs()) {
    const auto root = root_value(output.value_id);
    if (!is_value_ready(is_value_ready, root, command_plan.values().size())) {
      throw std::runtime_error("ModelPack: public output is not produced by the physical schedule");
    }
  }

  // ExecutionPlan is a view of the immutable physical schedule, not a second
  // execution authority. Physical lowering already assigns every CVU command
  // its semantic placement relative to MLA. Preserve that authority here;
  // names, op ids, and model-package age are deliberately irrelevant.

  ExecutionPlan result;
  result.pre.reserve(ordered.size());
  result.infer.reserve(ordered.size());
  result.post.reserve(ordered.size());
  for (std::size_t schedule_index = 0U; schedule_index < ordered.size(); ++schedule_index) {
    const auto cohort_id = ordered[schedule_index];
    auto& cohort = cohorts.at(cohort_id);
    std::sort(cohort.members.begin(), cohort.members.end(),
              [](const auto& left, const auto& right) { return left.ordinal < right.ordinal; });
    if (std::adjacent_find(cohort.members.begin(), cohort.members.end(),
                           [](const auto& left, const auto& right) {
                             return left.ordinal == right.ordinal;
                           }) != cohort.members.end()) {
      throw std::runtime_error("ModelPack: render cohort contains a duplicate member ordinal");
    }
    const auto& first_member = cohort.members.front();
    const auto first_op_id = first_member.semantic_chain.front();
    const auto& op = command_plan.ops()[first_op_id];
    SimaCvuCapabilityAbiRecord cvu_capability{};
    const bool has_cvu_capability =
        cohort.engine == PhysicalEngine::Cvu && cohort.graph_id != 0U &&
        sima_cvu_capability_abi_lookup(cohort.graph_id, &cvu_capability);
    const auto kind = has_cvu_capability
                          ? canonical_execution_stage_kind(cvu_capability.canonical_token)
                          : model_plan_stage_kind(op);
    if ((cohort.engine == PhysicalEngine::Cvu &&
         (!has_cvu_capability || kind == ExecutionStageKind::Unknown)) ||
        (cohort.engine != PhysicalEngine::Cvu &&
         (cohort.graph_id != 0U || first_member.semantic_chain.size() != 1U))) {
      throw std::runtime_error("ModelPack: physical command has no exact renderer identity");
    }
    for (const auto& member : cohort.members) {
      if ((has_cvu_capability &&
           member.semantic_chain.size() != cvu_capability.semantic_pattern_length) ||
          (!has_cvu_capability && member.semantic_chain.size() != 1U)) {
        throw std::runtime_error(
            "ModelPack: render cohort contains incompatible semantic operations");
      }
    }
    ExecutionStage stage;
    stage.order_index = schedule_index;
    stage.execution_op_id = first_op_id;
    stage.physical_cohort_id = cohort.id;
    stage.physical_command_ids = cohort.commands;
    for (const auto& member : cohort.members) {
      stage.execution_op_ids.insert(stage.execution_op_ids.end(), member.semantic_chain.begin(),
                                    member.semantic_chain.end());
    }
    if (cohort.members.size() == 1U && first_member.semantic_chain.size() == 1U) {
      stage.mpk_plugin_index =
          pipeline_internal::sima::find_plugin_index_by_name_or_id(packaging_contract, op.name);
      stage.stage_name = op.name;
    } else {
      stage.stage_name = "physical_cvu_cohort_" + std::to_string(cohort.id);
    }
    stage.kind = kind;
    stage.factory_name = require_stage_factory(stage.kind, cohort.engine == PhysicalEngine::Cvu);
    stage.plugin_id = plugin_id_for_stage_kind(stage.kind);
    stage.processor = processor_for_stage_kind(stage.kind);
    stage.kernel = cohort.engine == PhysicalEngine::Cvu
                       ? kernel_for_stage_kind(stage.kind)
                       : (!op.kernel.empty() ? op.kernel : op.implementation_id);
    if (cohort.engine != PhysicalEngine::Cvu || cohort.role == PhysicalCommandRole::Interstitial) {
      result.infer.push_back(std::move(stage));
    } else if (cohort.role == PhysicalCommandRole::Ingress) {
      result.pre.push_back(std::move(stage));
    } else if (cohort.role == PhysicalCommandRole::Egress) {
      result.post.push_back(std::move(stage));
    } else {
      throw std::runtime_error("ModelPack: CVU render cohort has no physical placement role");
    }
  }
  return result;
}

// A direct TVM CPU epoch is a property of the rendered, linear pipeline, not
// of the semantic DAG's in/out degree. A branch join can legitimately add an
// already-complete device predecessor to the later A65 command while the
// immediately preceding A65 element still owns the active mapping of the same
// Core arena. Keep that mapping only across adjacent HostTVM elements which
// provably use this one Core-owned CMA arena; the right-hand element must reuse
// an arena input, otherwise it introduces a new allocation boundary.
static bool can_continue_direct_tvm_cpu_epoch(
    const ExecutionStage& left, const ExecutionStage& right,
    const pipeline_internal::sima::static_contract::ModelExecutionPlan& plan,
    const pipeline_internal::sima::static_contract::FrameSlotArenaPlan& arena) {
  namespace sc = pipeline_internal::sima::static_contract;
  const auto& placement = arena.placement();
  if (left.kind != ExecutionStageKind::HostTvm || right.kind != ExecutionStageKind::HostTvm ||
      placement.domain != sc::ArenaStorageDomain::Cma ||
      placement.provenance != sc::ArenaAllocationProvenance::CoreAllocated ||
      !placement.requires_access(sc::ArenaDeviceAccess::CpuA65) ||
      !left.execution_op_id.has_value() || !right.execution_op_id.has_value() ||
      *left.execution_op_id >= plan.ops().size() || *right.execution_op_id >= plan.ops().size()) {
    return false;
  }

  const auto& left_op = plan.ops()[*left.execution_op_id];
  const auto& right_op = plan.ops()[*right.execution_op_id];
  if (left_op.kind != sc::OpKind::HostTvm || right_op.kind != sc::OpKind::HostTvm ||
      left_op.outputs.empty() || right_op.inputs.empty() || right_op.outputs.empty() ||
      std::any_of(left_op.outputs.begin(), left_op.outputs.end(),
                  [&](const auto value_id) { return arena.region(value_id) == nullptr; }) ||
      std::any_of(right_op.outputs.begin(), right_op.outputs.end(),
                  [&](const auto value_id) { return arena.region(value_id) == nullptr; })) {
    return false;
  }

  return std::any_of(right_op.inputs.begin(), right_op.inputs.end(),
                     [&](const auto value_id) { return arena.region(value_id) != nullptr; });
}

static std::string encode_direct_tvm_contract(
    const pipeline_internal::sima::static_contract::ModelExecutionPlan& plan,
    const pipeline_internal::sima::static_contract::FrameSlotArenaPlan& arena,
    const pipeline_internal::sima::static_contract::PhysicalExecutionPlan& physical,
    const pipeline_internal::sima::static_contract::OpSpec& op,
    const pipeline_internal::sima::MpkContract& packaging_contract, const bool cpu_epoch_start,
    const bool cpu_epoch_end) {
  using pipeline_internal::sima::static_contract::HostTvmOpConfig;
  const auto* host = std::get_if<HostTvmOpConfig>(&op.config);
  if (!host || host->input_names.size() != op.inputs.size() ||
      host->output_types.size() != op.outputs.size() ||
      std::any_of(host->output_alias_input.begin(), host->output_alias_input.end(),
                  [](const std::int32_t alias) { return alias >= 0; })) {
    throw std::runtime_error("ModelFragment: A65 target command '" + op.name +
                             "' is not an exact materializing boundary");
  }

  fs::path package_root = fs::path(packaging_contract.mpk_json_path).parent_path();
  if (package_root.filename() == "etc") {
    package_root = package_root.parent_path();
  }
  const fs::path raw(host->executable);
  const std::array<fs::path, 4> candidates = {
      raw.is_absolute() ? raw : package_root / "lib" / raw,
      raw.is_absolute() ? raw : package_root / raw,
      raw.is_absolute() ? raw : package_root / "share" / raw, raw};
  fs::path resolved;
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (!candidate.empty() && fs::is_regular_file(candidate, ec) && !ec) {
      resolved = candidate;
      break;
    }
  }
  if (resolved.empty()) {
    throw std::runtime_error("ModelFragment: A65 target artifact for '" + op.name +
                             "' is not present in the extracted package");
  }

  if (op.id >= physical.command_for_semantic_op.size() ||
      !physical.command_for_semantic_op[op.id].has_value()) {
    throw std::runtime_error("ModelFragment: A65 stage has no physical command authority");
  }
  const auto command_id = *physical.command_for_semantic_op[op.id];
  if (command_id >= physical.commands.size() ||
      physical.commands[command_id].engine !=
          pipeline_internal::sima::static_contract::PhysicalEngine::A65) {
    throw std::runtime_error("ModelFragment: A65 stage has a contradictory physical command");
  }
  const auto storage_domain = [&]() -> const char* {
    using Domain = pipeline_internal::sima::static_contract::ArenaStorageDomain;
    switch (arena.placement().domain) {
    case Domain::Cma:
      return "cma";
    case Domain::Dms:
      return "dms";
    case Domain::Unknown:
      break;
    }
    throw std::runtime_error("ModelFragment: A65 arena has no selected storage domain");
  }();
  const auto provenance = [&]() -> const char* {
    using Provenance = pipeline_internal::sima::static_contract::ArenaAllocationProvenance;
    switch (arena.placement().provenance) {
    case Provenance::CoreAllocated:
      return "core_allocated";
    case Provenance::ExternalAdopted:
      return "external_adopted";
    case Provenance::Unknown:
      break;
    }
    throw std::runtime_error("ModelFragment: A65 arena has no allocation provenance");
  }();
  const auto escape_policy =
      arena.placement().escape ==
              pipeline_internal::sima::static_contract::ArenaEscapePolicy::CpuMappablePublic
          ? "cpu_mappable_public"
          : "internal_only";

  nlohmann::json encoded{{"schema", "sima.neat.direct-tvm-lane"},
                         {"version", 2},
                         {"command_id", command_id},
                         {"command_name", op.name},
                         {"executable", resolved.string()},
                         {"executable_bytes", host->executable_bytes},
                         {"executable_sha256", host->executable_sha256},
                         {"arena_bytes", arena.allocation_bytes()},
                         {"storage_domain", storage_domain},
                         {"allocation_provenance", provenance},
                         {"required_device_access", arena.placement().required_device_access},
                         {"escape_policy", escape_policy},
                         {"cpu_epoch_start", cpu_epoch_start},
                         {"cpu_epoch_end", cpu_epoch_end},
                         {"inputs", nlohmann::json::array()},
                         {"outputs", nlohmann::json::array()}};
  bool reuses_arena = false;
  for (std::size_t index = 0; index < op.inputs.size(); ++index) {
    const auto* value = plan.value(op.inputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* region = value ? arena.region(value->id) : nullptr;
    if (!value || !binding || !value->logical_dtype || !value->logical_shape) {
      throw std::runtime_error("ModelFragment: A65 input has no exact target binding");
    }
    std::uint64_t offset = binding->byte_offset;
    if (region) {
      if (offset > std::numeric_limits<std::uint64_t>::max() - region->byte_offset) {
        throw std::runtime_error("ModelFragment: A65 input arena offset overflows");
      }
      offset += region->byte_offset;
      reuses_arena = true;
    }
    encoded["inputs"].push_back({{"parameter_name", host->input_names[index]},
                                 {"value_name", value->name},
                                 {"scalar", *value->logical_dtype},
                                 {"shape", *value->logical_shape},
                                 {"required_bytes", value->required_bytes},
                                 {"arena_bound", region != nullptr},
                                 {"arena_offset", offset}});
  }
  for (std::size_t index = 0; index < op.outputs.size(); ++index) {
    const auto* value = plan.value(op.outputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* region = value ? arena.region(value->id) : nullptr;
    if (!value || !binding || !region || !value->logical_dtype || !value->logical_shape ||
        binding->byte_offset > std::numeric_limits<std::uint64_t>::max() - region->byte_offset) {
      throw std::runtime_error("ModelFragment: A65 stage '" + op.name + "' output '" +
                               (value ? value->name : std::string("<missing>")) +
                               "' has no exact frame-arena binding");
    }
    encoded["outputs"].push_back({{"value_name", value->name},
                                  {"scalar", *value->logical_dtype},
                                  {"shape", *value->logical_shape},
                                  {"required_bytes", value->required_bytes},
                                  {"arena_offset", region->byte_offset + binding->byte_offset}});
  }
  encoded["reuse_input_arena"] = reuses_arena;
  const auto text = encoded.dump();
  gchar* base64 = g_base64_encode(reinterpret_cast<const guchar*>(text.data()), text.size());
  if (!base64) {
    throw std::runtime_error("ModelFragment: failed to encode prepared A65 lane contract");
  }
  std::string result(base64);
  g_free(base64);
  return result;
}

static bool direct_tvm_storage_is_dense_address_equivalent(
    const pipeline_internal::sima::static_contract::ValueSpec& value) noexcept {
  if (!value.storage_binding || !value.logical_shape || !value.logical_dtype ||
      value.logical_shape->empty()) {
    return false;
  }
  const std::uint64_t element_bytes =
      pipeline_internal::sima::specbuilders::dtype_size_bytes_from_token(*value.logical_dtype);
  if (element_bytes == 0U) {
    return false;
  }
  const auto& shape = *value.logical_shape;
  const auto& strides = value.storage_binding->stride_bytes;
  if (!strides.empty() && strides.size() != shape.size()) {
    return false;
  }

  std::uint64_t dense_stride = element_bytes;
  for (std::size_t reverse = shape.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    if (shape[axis] <= 0 ||
        (!strides.empty() &&
         (strides[axis] <= 0 ||
          (shape[axis] > 1 && static_cast<std::uint64_t>(strides[axis]) != dense_stride))) ||
        static_cast<std::uint64_t>(shape[axis]) >
            std::numeric_limits<std::uint64_t>::max() / dense_stride) {
      return false;
    }
    dense_stride *= static_cast<std::uint64_t>(shape[axis]);
  }
  return dense_stride == value.required_bytes;
}

static CompiledRuntimeContract build_target_tvm_runtime_contract(
    const pipeline_internal::sima::static_contract::ModelExecutionPlan& plan,
    const pipeline_internal::sima::static_contract::FrameSlotArenaPlan& arena,
    const pipeline_internal::sima::static_contract::OpSpec& op) {
  using pipeline_internal::sima::DeviceKind;
  using pipeline_internal::sima::FrameArenaRole;
  using pipeline_internal::sima::TensorMaterializationKind;
  using pipeline_internal::sima::specbuilders::build_input_binding_static_spec;
  using pipeline_internal::sima::specbuilders::build_logical_input_static_spec;
  using pipeline_internal::sima::specbuilders::build_logical_output_static_spec;
  using pipeline_internal::sima::specbuilders::build_output_route_static_spec;
  using pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec;

  CompiledRuntimeContract runtime;
  runtime.plugin_kind = "neatprocesstvm";
  runtime.frame_arena_size_bytes = arena.allocation_bytes();
  runtime.frame_arena_storage_domain = arena.placement().domain;
  runtime.frame_arena_provenance = arena.placement().provenance;
  runtime.frame_arena_required_device_access = arena.placement().required_device_access;
  runtime.frame_arena_escape_policy = arena.placement().escape;
  bool consumes_internal_carrier = false;

  for (std::size_t index = 0; index < op.inputs.size(); ++index) {
    const auto* value = plan.value(op.inputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* carrier = binding ? plan.carrier(binding->carrier_id) : nullptr;
    const auto* region = value ? arena.region(value->id) : nullptr;
    if (!value || !binding || !carrier || !value->logical_dtype || !value->logical_shape ||
        index > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !direct_tvm_storage_is_dense_address_equivalent(*value)) {
      throw std::runtime_error("ModelFragment: A65 input has no exact typed storage binding");
    }
    std::uint64_t parent_offset = binding->byte_offset;
    if (region) {
      if (parent_offset > std::numeric_limits<std::uint64_t>::max() - region->byte_offset) {
        throw std::runtime_error("ModelFragment: A65 input carrier offset overflows");
      }
      parent_offset += region->byte_offset;
      consumes_internal_carrier = true;
    }
    if (parent_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("ModelFragment: A65 input carrier offset is not representable");
    }
    const int local = static_cast<int>(index);
    runtime.physical_inputs.push_back(build_physical_buffer_static_spec(
        local, local, value->required_bytes, DeviceKind::Cpu, value->name, local,
        static_cast<std::int64_t>(parent_offset), carrier->required_alignment_bytes));
    runtime.logical_inputs.push_back(build_logical_input_static_spec(
        local, local, local, *value->logical_shape, *value->logical_dtype,
        value->logical_layout.value_or(""), value->name, value->name, value->name,
        /*byte_offset=*/0, value->required_bytes, TensorMaterializationKind::OffsetView));
    runtime.input_bindings.push_back(build_input_binding_static_spec(
        local, local, value->name, value->name, local, local, local, value->required_bytes,
        static_cast<std::int64_t>(parent_offset), true));
  }

  for (std::size_t index = 0; index < op.outputs.size(); ++index) {
    const auto* value = plan.value(op.outputs[index]);
    const auto* binding = value && value->storage_binding ? &*value->storage_binding : nullptr;
    const auto* carrier = binding ? plan.carrier(binding->carrier_id) : nullptr;
    const auto* region = value ? arena.region(value->id) : nullptr;
    const bool dense_equivalent = value && direct_tvm_storage_is_dense_address_equivalent(*value);
    if (!value || !binding || !carrier || !region || !value->logical_dtype ||
        !value->logical_shape || !dense_equivalent ||
        binding->byte_offset > std::numeric_limits<std::uint64_t>::max() - region->byte_offset ||
        index > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      std::string strides;
      if (binding) {
        for (const auto stride : binding->stride_bytes) {
          if (!strides.empty()) {
            strides += ',';
          }
          strides += std::to_string(stride);
        }
      }
      throw std::runtime_error(
          "ModelFragment: A65 stage '" + op.name + "' output '" +
          (value ? value->name : std::string("<missing>")) +
          "' has no exact dense frame-arena binding (binding=" + (binding ? "yes" : "no") +
          ", carrier=" + (carrier ? "yes" : "no") + ", region=" + (region ? "yes" : "no") +
          ", dtype=" + (value && value->logical_dtype ? *value->logical_dtype : "<missing>") +
          ", rank=" +
          std::to_string(value && value->logical_shape ? value->logical_shape->size() : 0U) +
          ", required_bytes=" + std::to_string(value ? value->required_bytes : 0U) +
          ", physical_span=" + std::to_string(binding ? binding->physical_span : 0U) +
          ", strides=[" + strides + "]" +
          ", dense_equivalent=" + (dense_equivalent ? "yes" : "no") + ")");
    }
    const std::uint64_t parent_offset = region->byte_offset + binding->byte_offset;
    if (parent_offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("ModelFragment: A65 output carrier offset is not representable");
    }
    const int local = static_cast<int>(index);
    runtime.physical_outputs.push_back(build_physical_buffer_static_spec(
        local, local, value->required_bytes, DeviceKind::Cpu, value->name, local,
        static_cast<std::int64_t>(parent_offset), carrier->required_alignment_bytes));
    runtime.logical_outputs.push_back(build_logical_output_static_spec(
        local, local, local, local, local, *value->logical_shape, *value->logical_dtype,
        value->logical_layout.value_or(""), value->name, value->name, value->name,
        /*byte_offset=*/0, value->required_bytes));
    runtime.output_order.push_back(
        build_output_route_static_spec(local, local, local, value->name, value->name));
  }
  runtime.frame_arena_role =
      consumes_internal_carrier ? FrameArenaRole::ReuseInput : FrameArenaRole::Allocate;
  runtime.consumer_keeps_distinct_physical_inputs = op.inputs.size() > 1U;
  return runtime;
}

static std::vector<ModelFragment::StageFacts> build_stage_facts_from_execution_plan(
    const std::vector<ExecutionStage>& stages,
    const std::optional<pipeline_internal::sima::MpkContract>& mpk_contract,
    const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>& model_managed_route_flags,
    const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract,
    ModelStage stage_context,
    const pipeline_internal::sima::static_contract::ModelExecutionPlan& dmabuf_plan_execution_plan,
    const pipeline_internal::sima::static_contract::FrameSlotArenaPlan& dmabuf_frame_arena_plan,
    const pipeline_internal::sima::static_contract::PhysicalExecutionPlan&
        dmabuf_physical_execution_plan) {
  if (!mpk_contract.has_value()) {
    throw std::runtime_error(
        "ModelFragment: strict MPK contract required for typed execution plan");
  }

  std::vector<ModelFragment::StageFacts> facts;
  facts.reserve(stages.size());
  std::optional<CompiledRuntimeContract> previous_runtime;
  if (upstream_handoff_contract.has_value()) {
    previous_runtime = upstream_handoff_contract->runtime_contract;
  }
  for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
    const auto& stage = stages[stage_index];
    if (std::getenv("NEAT_DUMP_POST_REGIONS") != nullptr) {
      std::fprintf(stderr, "[modelpack-stage] ctx=%d idx=%zu kind=%d name=%s uses_pcvu=%d\n",
                   static_cast<int>(stage_context), stage_index, static_cast<int>(stage.kind),
                   stage.stage_name.c_str(),
                   execution_stage_uses_processcvu_contract(stage.kind) ? 1 : 0);
    }
    ModelFragment::StageFacts entry;
    entry.stage_name = stage.stage_name;
    entry.stage_order = stage.order_index;
    const auto* mpk_stage = find_mpk_stage_for_execution_stage(mpk_contract, stage);

    if (execution_stage_uses_processcvu_contract(stage.kind)) {
      if (stage.physical_command_ids.empty()) {
        throw std::runtime_error("ModelFragment: ProcessCVU stage has no physical command");
      }
      std::string contract_error;
      entry.processcvu_contract =
          pipeline_internal::sima::static_contract::build_dmabuf_plan_processcvu_command_contract(
              dmabuf_plan_execution_plan, dmabuf_physical_execution_plan,
              std::span<const pipeline_internal::sima::static_contract::PhysicalCommandId>(
                  stage.physical_command_ids),
              dmabuf_frame_arena_plan, &contract_error);
      if (!entry.processcvu_contract.has_value()) {
        throw std::runtime_error(
            "ModelFragment: dmabuf-plan strict ProcessCVU command contract failed: " +
            contract_error);
      }
      const auto command_id = stage.physical_command_ids.front();
      if (command_id >= dmabuf_physical_execution_plan.commands.size()) {
        throw std::runtime_error("ModelFragment: strict ProcessCVU command role is out of range");
      }
      const auto command_role = dmabuf_physical_execution_plan.commands[command_id].role;
      if (command_role == pipeline_internal::sima::static_contract::PhysicalCommandRole::NonCvu ||
          std::any_of(stage.physical_command_ids.begin(), stage.physical_command_ids.end(),
                      [&](const auto id) {
                        return id >= dmabuf_physical_execution_plan.commands.size() ||
                               dmabuf_physical_execution_plan.commands[id].role != command_role;
                      })) {
        throw std::runtime_error(
            "ModelFragment: strict ProcessCVU cohort has no exact uniform placement role");
      }
      entry.processcvu_contract->physical_command_role = command_role;
    }

    if (stage.kind == ExecutionStageKind::Mla) {
      const auto* mla_stage = mpk_stage;
      if (!mla_stage) {
        throw std::runtime_error(
            "ModelFragment: strict MPK MLA contract missing for stage '" + stage.stage_name +
            "'. Ensure the MPK manifest includes an MLA plugin with"
            " input/output tensor contracts (processor='MLA' or kernel='infer').");
      }
      if (to_upper(mla_stage->processor) != "MLA") {
        throw std::runtime_error("ModelFragment: execution-stage identity '" + stage.stage_name +
                                 "' does not select an MLA MPK plugin");
      }
      const bool one_mla = dmabuf_plan_execution_plan.mla_stage_count() == 1U;
      const auto published_outputs =
          one_mla ? pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_contract)
                  : mla_stage->output_tensors;
      const auto logical_outputs =
          one_mla ? pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_contract)
                  : mla_stage->output_tensors;
      const auto boundary_inputs =
          one_mla
              ? pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(*mpk_contract)
              : mla_stage->input_tensors;
      const auto physical_outputs =
          one_mla
              ? pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(*mpk_contract)
              : mla_stage->output_tensors;
      auto mla_contract = pipeline_internal::sima::build_mla_static_contract_from_mpk_stage(
          *mla_stage,
          !published_outputs.empty()
              ? published_outputs
              : (logical_outputs.empty() ? mla_stage->output_tensors : logical_outputs),
          physical_outputs.empty() ? mla_stage->output_tensors : physical_outputs, stage.stage_name,
          boundary_inputs.empty() ? nullptr : &boundary_inputs);
      mla_contract.consumer_keeps_distinct_physical_inputs =
          one_mla
              ? pipeline_internal::sima::mla_consumer_keeps_distinct_physical_inputs(*mpk_contract)
              : mla_stage->input_tensors.size() > 1U;
      const auto mla_props = read_mla_runtime_properties_from_mpk_contract(mpk_contract, mla_stage);
      if (!mla_props.has_value() || mla_props->model_path.empty()) {
        throw std::runtime_error(
            "ModelFragment: strict MPK MLA runtime payload missing for stage '" + stage.stage_name +
            "' (expected 'model_path', 'batch_size' fields in the MPK MLA"
            " config or simaai__params section).");
      }
      apply_mla_runtime_properties_to_contract(*mla_props, &mla_contract);
      {
        std::string projection_error;
        const auto* exact_stage = dmabuf_plan_execution_plan.mla_stage_for_identity(
            mla_stage->name, mla_stage->executable);
        if (!exact_stage) {
          throw std::runtime_error(
              "ModelFragment: no exact execution-plan MLA stage matches MPK stage '" +
              mla_stage->name + "' and executable '" + mla_stage->executable + "'");
        }
        const std::size_t mla_stage_index = exact_stage->key.stage_index;
        std::span<const pipeline_internal::sima::LogicalTensorStaticSpec> upstream_outputs;
        if (previous_runtime.has_value()) {
          upstream_outputs = previous_runtime->logical_outputs;
        }
        auto input_sources =
            pipeline_internal::sima::static_contract::resolve_mla_input_physical_sources(
                dmabuf_plan_execution_plan, mla_stage_index, dmabuf_frame_arena_plan,
                upstream_outputs, &projection_error);
        if (!input_sources.has_value()) {
          throw std::runtime_error(
              "ModelFragment: dmabuf-plan strict MLA physical-input projection failed: " +
              projection_error);
        }
        if (!pipeline_internal::sima::static_contract::apply_dmabuf_plan_contract_projection(
                dmabuf_plan_execution_plan, mla_stage_index, dmabuf_frame_arena_plan, &mla_contract,
                *input_sources, &projection_error)) {
          throw std::runtime_error("ModelFragment: dmabuf-plan strict MLA projection failed: " +
                                   projection_error);
        }
      }
      entry.mla_compiled =
          pipeline_internal::sima::stagesemantics::build_mla_compiled_contract(mla_contract);
      entry.mla_compiled->payload.dmabuf_plan_contract = true;
      if (!stage.execution_op_id.has_value() ||
          *stage.execution_op_id >= dmabuf_plan_execution_plan.ops().size()) {
        throw std::runtime_error("ModelFragment: MLA stage has no exact execution-plan identity");
      }
      {
        using pipeline_internal::sima::static_contract::MlaOpConfig;
        const auto* exact = std::get_if<MlaOpConfig>(
            &dmabuf_plan_execution_plan.ops()[*stage.execution_op_id].config);
        if (!exact || exact->executable_bytes == 0U || exact->executable_sha256.size() != 64U) {
          throw std::runtime_error("ModelFragment: MLA artifact identity was not retained");
        }
        entry.mla_compiled->payload.executable_bytes = exact->executable_bytes;
        entry.mla_compiled->payload.executable_sha256 = exact->executable_sha256;
      }
    }

    if (stage.kind == ExecutionStageKind::HostTvm) {
      if (!stage.execution_op_id.has_value() ||
          *stage.execution_op_id >= dmabuf_plan_execution_plan.ops().size()) {
        throw std::runtime_error(
            "ModelFragment: target A65 stage is missing its immutable command/arena plan");
      }
      const auto& op = dmabuf_plan_execution_plan.ops()[*stage.execution_op_id];
      if (op.id != *stage.execution_op_id ||
          op.kind != pipeline_internal::sima::static_contract::OpKind::HostTvm) {
        throw std::runtime_error(
            "ModelFragment: target A65 stage does not select an exact HostTVM command");
      }
      auto runtime = build_target_tvm_runtime_contract(dmabuf_plan_execution_plan,
                                                       dmabuf_frame_arena_plan, op);
      entry.transport_compiled = build_model_managed_transport_contract(
          "neatprocesstvm", "direct-tvm", std::nullopt, std::move(runtime));
      const bool cpu_epoch_start =
          stage_index == 0U ||
          !can_continue_direct_tvm_cpu_epoch(stages[stage_index - 1U], stage,
                                             dmabuf_plan_execution_plan, dmabuf_frame_arena_plan);
      const bool cpu_epoch_end =
          stage_index + 1U == stages.size() ||
          !can_continue_direct_tvm_cpu_epoch(stage, stages[stage_index + 1U],
                                             dmabuf_plan_execution_plan, dmabuf_frame_arena_plan);
      entry.fragment_properties.emplace_back(
          "direct-contract-b64",
          encode_direct_tvm_contract(dmabuf_plan_execution_plan, dmabuf_frame_arena_plan,
                                     dmabuf_physical_execution_plan, op, *mpk_contract,
                                     cpu_epoch_start, cpu_epoch_end));
    }

    if (stage.kind == ExecutionStageKind::BoxDecode) {
      if (mpk_stage) {
        std::string route_flags_error;
        auto resolved_route_flags =
            pipeline_internal::sima::resolve_model_managed_boxdecode_route_flags_from_mpk(
                *mpk_contract, mpk_stage, &route_flags_error);
        if (!resolved_route_flags.has_value()) {
          throw std::runtime_error(
              "ModelFragment: strict model-managed boxdecode route facts missing for stage '" +
              stage.stage_name + "': " +
              (route_flags_error.empty() ? std::string("missing MPK/upstream route facts")
                                         : route_flags_error));
        }
        if (model_managed_route_flags.has_value()) {
          *resolved_route_flags = pipeline_internal::sima::reconcile_exact_boxdecode_route_flags(
              *model_managed_route_flags, *resolved_route_flags);
        }
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
        // An authored terminal owns the package-time compiled contract.  A
        // synthetic/customer-selected BoxDecode stage has no MPK terminal
        // pointer; leave its optional compiled contract empty so ModelAccess
        // can derive the external true-leaf contract after the route family is
        // known.  Never pass nullptr here: that would conflate the two
        // authorities and weaken terminal edge/binding validation.
        if (simaai::neat::pipeline_internal::sima::is_box_decode_type_specified(
                boxdecode_subset->decode_type)) {
          pipeline_internal::sima::stagesemantics::BoxDecodeCompiledContractOptions compile_options;
          compile_options.decode_type = boxdecode_subset->decode_type;
          if (boxdecode_subset->decode_type_option.has_value()) {
            compile_options.decode_type_option = boxdecode_subset->decode_type_option;
          }
          compile_options.score_activation = boxdecode_subset->score_activation;
          compile_options.model_owned_flags = true;
          compile_options.required_preprocess_meta_fields =
              default_preprocess_meta_required_fields();
          entry.boxdecode_compiled = pipeline_internal::sima::stagesemantics::
              build_boxdecode_compiled_contract_from_subset(*boxdecode_subset, compile_options);
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

    if (entry.processcvu_contract.has_value()) {
      previous_runtime = entry.processcvu_contract->runtime_contract;
    } else if (entry.mla_compiled.has_value()) {
      previous_runtime = entry.mla_compiled->runtime_contract;
    } else if (entry.boxdecode_compiled.has_value()) {
      previous_runtime = entry.boxdecode_compiled->runtime_contract;
    } else if (entry.dequant_compiled.has_value()) {
      previous_runtime = entry.dequant_compiled->runtime_contract;
    } else if (entry.transport_compiled.has_value()) {
      previous_runtime = entry.transport_compiled->runtime_contract;
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

      if (plugin.find("processtvm") != std::string::npos) {
        const ModelFragment::StageFacts* entry = resolve_stage_facts(element);
        if (!entry || !entry->transport_compiled.has_value()) {
          if (err) {
            *err = "ModelFragment contract compile: missing direct TVM transport contract for '" +
                   element_name + "'";
          }
          return {};
        }
        if (!pipeline_internal::sima::stagesemantics::build_transport_node_contract(
                kind_, element_name, stage_id, fragment_stage_definition("neatprocesstvm"),
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

  const auto* mla_stage =
      pipeline_internal::sima::get_stage_io_contract(*mpk_contract, terminal.stage_name);
  if (!mla_stage) {
    throw std::runtime_error("Terminal MLA stage missing from strict MPK contract after infer "
                             "trimming: '" +
                             terminal.stage_name + "'");
  }

  const bool single_mla =
      pipeline_internal::sima::get_mla_stage_io_contracts(*mpk_contract).size() == 1U;
  const auto published_outputs =
      single_mla ? pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_contract)
                 : mla_stage->output_tensors;
  const auto logical_outputs =
      single_mla ? pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_contract)
                 : mla_stage->output_tensors;
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
      msg << "Terminal MLA contract incomplete after infer trimming" << " stage='"
          << terminal.stage_name << "'" << " output_index=" << i << " resolved={w=" << dims.width
          << ",h=" << dims.height << ",d=" << dims.depth
          << ",dtype=" << (dtype.empty() ? "<empty>" : dtype) << "}"
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
    msg << "Inference terminal policy could not resolve last MLA stage" << " infer_stages=["
        << infer_stage_summary(infer_seq) << "]";
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

static ModelFragment build_fragment_linear(const std::vector<ExecutionStage>& stages,
                                           const std::string& initial_input_name,
                                           int num_buffers_cvu, int num_buffers_mla,
                                           const std::string& name_suffix,
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
    const auto* exact_stage_facts = find_stage_facts(stage.stage_name);

    if (i)
      pipelineStr << "! ";
    pipelineStr << plugin << " name=" << name << " ";
    // ProcessTVM's strict direct-contract is self-contained and digest-bound;
    // unlike ProcessMLA/ProcessCVU it does not look up a manifest stage by a
    // GObject property. Do not leak the generic manifest-context property onto
    // this element merely because the compatibility renderer is linear.
    if (plugin != "neatprocesstvm") {
      pipelineStr << "stage-id=" << name << " ";
    }
    if (plugin == "neatprocesscvu") {
      if (num_buffers_cvu > 0) {
        pipelineStr << " num-buffers=" << num_buffers_cvu << " ";
      }
    } else if (plugin == "neatprocessmla") {
      const bool force_single_pipe = env_truthy_local("SIMA_FORCE_MLA_SINGLE_PIPE");
      const bool use_multi_pipeline = !force_single_pipe && num_buffers_mla > 1;
      pipelineStr << "multi-pipeline=" << (use_multi_pipeline ? "true" : "false") << " ";
      if (num_buffers_mla > 0) {
        pipelineStr << " num-buffers=" << num_buffers_mla << " ";
      }
    } else if (plugin == "neatprocesstvm") {
      if (num_buffers_cvu > 0) {
        pipelineStr << " num-buffers=" << num_buffers_cvu << " ";
      }
    }
    if (exact_stage_facts != nullptr) {
      for (const auto& [key, value] : exact_stage_facts->fragment_properties) {
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

// Automatic selection accepts only a local NVMe data volume. Matching the block device keeps NFS
// and every other network filesystem out without maintaining an fstype allow-list, and a read-only
// or system mount is never a model store.
std::vector<std::string> nvme_model_bases_from_mounts(std::istream& mounts) {
  std::vector<std::string> out;
  std::string device, mount_point, fstype, options, rest;
  while (mounts >> device >> mount_point >> fstype >> options) {
    std::getline(mounts, rest);
    if (device.rfind("/dev/nvme", 0) != 0)
      continue;
    // Mount points are octal-escaped ("\040" for space); skip rather than decode, which falls
    // through to the next candidate instead of creating a directory with a literal backslash.
    if (mount_point.find('\\') != std::string::npos)
      continue;
    if (system_mount_point(mount_point))
      continue;
    // vfat/EFI partitions cannot hold a model package's permissions or sizes.
    if (fstype == "vfat" || fstype == "msdos" || fstype == "iso9660")
      continue;
    // Eligibility, not capacity: a read-only mount cannot hold the package, and a noexec mount
    // cannot load a .so extracted into lib/. Both fall through to the next candidate, unlike a
    // full NVMe, which fails the load.
    if (mount_has_option(options, "ro") || mount_has_option(options, "noexec"))
      continue;
    out.push_back((fs::path(mount_point) / "simaai/coprocessing/models").string());
  }
  return out;
}

// Longest matching mount point wins, which is how the kernel resolves the path. Modalix exposes
// its eMMC root filesystem as either /dev/mmcblk* or /dev/root.
std::string modelpack_storage_label(const std::string& path) {
  std::ifstream mounts("/proc/mounts");
  std::string device, mount_point, rest, best_device;
  std::size_t best_len = 0;
  while (mounts >> device >> mount_point) {
    std::getline(mounts, rest);
    if (mount_point.find('\\') != std::string::npos)
      continue;
    const std::string prefix = (mount_point == "/") ? "/" : mount_point + "/";
    if (path != mount_point && path.rfind(prefix, 0) != 0)
      continue;
    if (mount_point.size() >= best_len) {
      best_len = mount_point.size();
      best_device = device;
    }
  }
  if (best_device.rfind("/dev/nvme", 0) == 0)
    return "NVMe";
  if (best_device == "/dev/root" || best_device.rfind("/dev/mmcblk", 0) == 0)
    return "eMMC";
  return best_device.empty() ? "unknown" : best_device;
}

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
    std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags,
    std::vector<ExecutionStageKind> model_managed_post_kinds) {
  // Match develop: route selection records facts but never rewrites the compiler's
  // semantic, physical, or arena plans.  Unpack/Slice views remain immutable
  // ReadExpressions and terminal execution is cut at render time.
  model_managed_route_flags_ = std::move(model_managed_route_flags);
  model_managed_post_kinds_ = std::move(model_managed_post_kinds);
}

void ModelPack::init(const std::string& tar_gz) {
  Config cfg;
  init_from_config(tar_gz, std::move(cfg));
}

void ModelPack::init_from_config(const std::string& tar_gz, Config cfg) {
  options_ = std::move(cfg);
  execution_admission_ = {};
  execution_plan_digest_.clear();
  mpk_contract_.reset();
  dmabuf_plan_execution_plan_.reset();
  dmabuf_frame_arena_plan_.reset();
  dmabuf_physical_execution_plan_.reset();
  route_graph_.reset();
  model_managed_route_flags_.reset();
  model_managed_post_kinds_.clear();

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

  const OrganizedModelPackage organized =
      extract_and_organize(tar_gz, options_.cleanup_extracted_model_data);
  const std::string& extracted = organized.package_root;
  logical_package_name_ = organized.logical_package_name;
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
  // Loading a model establishes only its semantic MPK contract. The exact
  // MLA ELF topology and physical DMA-BUF plan are execution concerns and are
  // admitted lazily when an executable route is requested. This keeps model
  // metadata/route inspection independent of target artifacts while the
  // execution boundary remains fail-closed.
  if (mpk_contract_.has_value()) {
    const auto mla_stages =
        simaai::neat::pipeline_internal::sima::get_mla_stage_io_contracts(*mpk_contract_);
    if (const auto* mla = mla_stages.empty() ? nullptr : mla_stages.front();
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

void ModelPack::ensure_dmabuf_execution_plan() const {
  if (dmabuf_plan_execution_plan_.has_value()) {
    if (!dmabuf_frame_arena_plan_.has_value() || !dmabuf_physical_execution_plan_.has_value()) {
      throw std::runtime_error(
          "ModelPack: cached dmabuf-plan admission is missing its physical or arena plan");
    }
    return;
  }
  if (dmabuf_frame_arena_plan_.has_value() || dmabuf_physical_execution_plan_.has_value()) {
    throw std::runtime_error("ModelPack: partial dmabuf-plan admission state is invalid");
  }
  if (!mpk_contract_.has_value()) {
    throw std::runtime_error("ModelPack: dmabuf-plan requires an exact mpk.json manifest");
  }

  auto compiled = compile_dmabuf_plan_execution_plan(*mpk_contract_);
  execution_admission_ = compiled.report;
  execution_plan_digest_ = compiled.plan_digest;
  if (!compiled.eligible()) {
    throw std::runtime_error(std::string("ModelPack: dmabuf-plan admission failed [") +
                             pipeline_internal::dmabuf_eligibility_code_name(compiled.report.code) +
                             "] at " +
                             (compiled.report.location.empty() ? "$" : compiled.report.location) +
                             ": " + compiled.report.detail);
  }

  dmabuf_plan_execution_plan_ = std::move(compiled.plan);
  dmabuf_frame_arena_plan_ = std::move(compiled.arena_plan);
  dmabuf_physical_execution_plan_ = std::move(compiled.physical_plan);
}

void ModelPack::prepare_for_execution() const {
  ensure_dmabuf_execution_plan();
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

ExecutionPlan ModelPack::semantic_execution_plan() const {
  if (!mpk_contract_.has_value()) {
    throw std::runtime_error(
        "ModelPack: strict MPK contract required to derive the semantic execution plan");
  }
  return build_execution_plan_from_mpk_contract(
      *mpk_contract_, pipeline_type_, model_managed_route_flags_, model_managed_post_kinds_);
}

ExecutionPlan ModelPack::execution_plan() const {
  prepare_for_execution();
  return build_execution_plan_from_model_plan(*dmabuf_plan_execution_plan_,
                                              *dmabuf_physical_execution_plan_,
                                              *dmabuf_frame_arena_plan_, *mpk_contract_);
}

std::string ModelPack::infer_output_name() const {
  if (internal::has_terminal_policy(options_.terminal_policy)) {
    // Explicit selectors address executable stages, including physical CVU
    // cohorts. Resolve their identity without constructing GStreamer elements.
    const auto stages = execution_plan().infer;
    const auto index = resolve_terminal_index_or_throw(stages, options_.terminal_policy);
    return apply_name_suffix(stages[index].stage_name);
  }

  // Ordinary name queries remain descriptive: both MLA and A65 execute inside
  // infer, while boundary CVU operations remain in the pre/post fragments.
  if (!mpk_contract_.has_value()) {
    throw std::runtime_error("ModelPack: strict MPK contract required to derive inference naming");
  }
  const auto& contract = *mpk_contract_;
  const auto ordered = pipeline_internal::sima::plugins_in_execution_order(contract);
  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) {
    const auto& stage = contract.plugins[*it];
    const auto processor = to_upper(stage.processor);
    if (processor == "MLA" || processor == "A65") {
      return apply_name_suffix(stage.name);
    }
  }
  return {};
}

std::vector<ModelFragment::StageFacts> ModelPack::build_stage_facts(
    const std::vector<ExecutionStage>& stages,
    const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract,
    ModelStage stage_context) const {
  return build_stage_facts_from_execution_plan(
      stages, mpk_contract_, model_managed_route_flags_, upstream_handoff_contract, stage_context,
      *dmabuf_plan_execution_plan_, *dmabuf_frame_arena_plan_, *dmabuf_physical_execution_plan_);
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
    return build_stage_facts_from_execution_plan(
        plan.infer, mpk_contract_, model_managed_route_flags_, upstream_handoff_contract,
        ModelStage::MlaOnly, *dmabuf_plan_execution_plan_, *dmabuf_frame_arena_plan_,
        *dmabuf_physical_execution_plan_);
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

    auto infer_facts = build_stage_facts_from_execution_plan(
        plan.infer, mpk_contract_, model_managed_route_flags_, upstream_handoff_contract,
        ModelStage::MlaOnly, *dmabuf_plan_execution_plan_, *dmabuf_frame_arena_plan_,
        *dmabuf_physical_execution_plan_);
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
                               options_.name_suffix, std::move(stage_facts));
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

CompiledProcessCvuContract
ModelPack::project_model_managed_preproc_contract(const PreprocOptions& options) const {
  prepare_for_execution();
  if (!options.model_managed_contract || !dmabuf_plan_execution_plan_ ||
      !dmabuf_frame_arena_plan_ || !dmabuf_physical_execution_plan_) {
    throw std::runtime_error(
        "ModelPack: strict model-managed preproc projection requires the immutable "
        "execution/physical/arena plans");
  }
  const PreprocOptions envelope = model_managed_preproc_static_envelope_options(options);
  auto compiled =
      pipeline_internal::sima::stagesemantics::build_processcvu_compiled_contract_from_options(
          envelope);
  std::vector<pipeline_internal::sima::static_contract::PhysicalCommandId> absorbed;
  std::string projection_error;
  if (!pipeline_internal::sima::static_contract::project_model_managed_preproc_contract(
          *dmabuf_plan_execution_plan_, *dmabuf_physical_execution_plan_, *dmabuf_frame_arena_plan_,
          &compiled, &absorbed, &projection_error)) {
    throw std::runtime_error("ModelPack: model-managed graph-200 ingress projection failed: " +
                             projection_error);
  }
  if (absorbed.empty()) {
    throw std::runtime_error(
        "ModelPack: model-managed graph-200 ingress projection absorbed no physical command");
  }
  return compiled;
}

std::vector<std::shared_ptr<Node>>
ModelPack::infer_block(const std::string& upstream_name,
                       std::shared_ptr<const ModelLineageBinding> model_lineage,
                       const bool absorb_model_managed_preproc) const {
  if (absorb_model_managed_preproc) {
    prepare_for_execution();
  }
  const ExecutionPlan plan = execution_plan();
  if (plan.infer.empty()) {
    throw std::runtime_error("ModelPack::infer_block: pipeline has no infer stages");
  }
  std::vector<ExecutionStage> infer_seq = plan.infer;

  if (absorb_model_managed_preproc) {
    if (!dmabuf_plan_execution_plan_ || !dmabuf_physical_execution_plan_) {
      throw std::runtime_error(
          "ModelPack::infer_block: graph-200 absorption requires the strict physical plan");
    }
    std::string absorption_error;
    auto absorbed =
        pipeline_internal::sima::static_contract::resolve_model_managed_preproc_ingress_commands(
            *dmabuf_plan_execution_plan_, *dmabuf_physical_execution_plan_, &absorption_error);
    if (!absorbed || absorbed->empty()) {
      throw std::runtime_error("ModelPack::infer_block: graph-200 ingress absorption failed: " +
                               absorption_error);
    }
    std::unordered_set<pipeline_internal::sima::static_contract::PhysicalCommandId> remaining(
        absorbed->begin(), absorbed->end());
    for (const auto& stage : plan.pre) {
      for (const auto id : stage.physical_command_ids) {
        remaining.erase(id);
      }
    }
    if (!remaining.empty()) {
      throw std::runtime_error(
          "ModelPack::infer_block: absorbed graph-200 commands are not in the exact "
          "preprocess region");
    }
  }

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
  auto stage_facts = build_stage_facts_from_execution_plan(
      infer_seq, mpk_contract_, model_managed_route_flags_, upstream_handoff_contract,
      ModelStage::MlaOnly, *dmabuf_plan_execution_plan_, *dmabuf_frame_arena_plan_,
      *dmabuf_physical_execution_plan_);
  ModelFragment frag =
      build_fragment_linear(infer_seq, upstream, options_.num_buffers_cvu, options_.num_buffers_mla,
                            options_.name_suffix, std::move(stage_facts));
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
