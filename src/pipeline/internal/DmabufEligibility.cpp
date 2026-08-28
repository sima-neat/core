#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/AfeMpkV2Decoder.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "pipeline/internal/sima/static_contract/TvmHostModuleGraph.h"

#include <glib.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

using Json = nlohmann::json;
namespace sc = sima::static_contract;

template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

std::string sha256_text(const std::string_view text) {
  gchar* digest = g_compute_checksum_for_data(
      G_CHECKSUM_SHA256, reinterpret_cast<const guchar*>(text.data()), text.size());
  if (!digest) {
    return {};
  }
  std::string result(digest);
  g_free(digest);
  return result;
}

std::optional<std::string> sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  GChecksum* checksum = g_checksum_new(G_CHECKSUM_SHA256);
  if (!checksum) {
    return std::nullopt;
  }
  std::array<char, 64U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      g_checksum_update(checksum, reinterpret_cast<const guchar*>(buffer.data()),
                        static_cast<gsize>(count));
    }
  }
  if (!input.eof()) {
    g_checksum_free(checksum);
    return std::nullopt;
  }
  const gchar* digest = g_checksum_get_string(checksum);
  std::string result = digest ? digest : "";
  g_checksum_free(checksum);
  return result.empty() ? std::nullopt : std::optional<std::string>(std::move(result));
}

std::string basename_or_placeholder(const std::filesystem::path& path, const char* placeholder) {
  const auto name = path.filename().string();
  return name.empty() ? std::string(placeholder) : name;
}

std::string sanitize_detail(std::string detail, const std::filesystem::path& mpk_manifest,
                            const std::filesystem::path& mla_executable) {
  const auto replace_all = [&](const std::string& needle, const char* replacement) {
    if (needle.empty()) {
      return;
    }
    std::size_t cursor = 0U;
    while ((cursor = detail.find(needle, cursor)) != std::string::npos) {
      detail.replace(cursor, needle.size(), replacement);
      cursor += std::char_traits<char>::length(replacement);
    }
  };
  replace_all(mpk_manifest.string(), "<mpk-manifest>");
  replace_all(mla_executable.string(), "<mla-executable>");
  return detail;
}

std::string detached_root_proof(
    const std::span<const sc::ValueId> detached_roots) {
  std::string result = ";detached_count=" +
                       std::to_string(detached_roots.size()) +
                       ";detached_roots=[";
  for (std::size_t index = 0; index < detached_roots.size(); ++index) {
    if (index != 0U) {
      result.push_back(',');
    }
    result += std::to_string(detached_roots[index]);
  }
  result.push_back(']');
  return result;
}

const char* value_representation_name(const sc::ValueRepresentation value) noexcept {
  switch (value) {
  case sc::ValueRepresentation::Dense:
    return "dense";
  case sc::ValueRepresentation::Tessellated:
    return "tessellated";
  case sc::ValueRepresentation::Packed:
    return "packed";
  case sc::ValueRepresentation::BackendNative:
    return "backend-native";
  }
  return "unknown";
}

const char* op_kind_name(const sc::OpKind value) noexcept {
  switch (value) {
  case sc::OpKind::Cast:
    return "cast";
  case sc::OpKind::Quantize:
    return "quantize";
  case sc::OpKind::Tessellate:
    return "tessellate";
  case sc::OpKind::Pack:
    return "pack";
  case sc::OpKind::Mla:
    return "mla";
  case sc::OpKind::Unpack:
    return "unpack";
  case sc::OpKind::Slice:
    return "slice";
  case sc::OpKind::Reshape:
    return "reshape";
  case sc::OpKind::Detessellate:
    return "detessellate";
  case sc::OpKind::Dequantize:
    return "dequantize";
  case sc::OpKind::HostTvm:
    return "host-tvm";
  case sc::OpKind::PassThrough:
    return "pass-through";
  }
  return "unknown";
}

Json quantization_json(const std::vector<sc::QuantizationSpec>& values) {
  Json result = Json::array();
  for (const auto& value : values) {
    result.push_back({{"scale", value.scale}, {"zero_point", value.zero_point}});
  }
  return result;
}

Json op_config_json(const sc::OpConfig& config) {
  return std::visit(
      Overloaded{
          [](const sc::CastOpConfig& value) { return Json{{"output_dtype", value.output_dtype}}; },
          [](const sc::QuantizeOpConfig& value) {
            return Json{{"output_dtype", value.output_dtype},
                        {"num_bits", value.num_bits},
                        {"rounding", value.rounding},
                        {"channel_params", quantization_json(value.channel_params)}};
          },
          [](const sc::TessellateOpConfig& value) {
            return Json{{"slice_shape", value.slice_shape},
                        {"align_c16", value.align_c16},
                        {"cblock", value.cblock},
                        {"frame_type", value.frame_type}};
          },
          [](const sc::PackOpConfig& value) {
            Json components = Json::array();
            for (const auto& component : value.components) {
              components.push_back({{"value_id", component.value_id},
                                    {"parent_offset", component.parent_offset},
                                    {"stored_bytes", component.stored_bytes}});
            }
            Json spans = Json::array();
            for (const auto& span : value.spans) {
              spans.push_back({{"value_id", span.value_id},
                               {"batch_index", span.batch_index},
                               {"source_byte_offset", span.source_byte_offset},
                               {"parent_offset", span.parent_offset},
                               {"logical_bytes", span.logical_bytes},
                               {"stored_bytes", span.stored_bytes},
                               {"padding_policy", span.padding_policy}});
            }
            return Json{{"components", std::move(components)},
                        {"batch_count", value.batch_count},
                        {"parent_required_bytes", value.parent_required_bytes},
                        {"spans", std::move(spans)},
                        {"materializes", value.materializes}};
          },
          [](const sc::MlaOpConfig& value) {
            const auto types_json = [](const auto& types) {
              Json result = Json::array();
              for (const auto& type : types) {
                result.push_back({{"scalar", type.scalar}, {"shape", type.shape}});
              }
              return result;
            };
            return Json{{"executable", value.executable},
                        {"executable_bytes", value.executable_bytes},
                        {"executable_sha256", value.executable_sha256},
                        {"number_of_quads", value.number_of_quads},
                        {"input_types", types_json(value.input_types)},
                        {"output_types", types_json(value.output_types)}};
          },
          [](const sc::UnpackOpConfig& value) {
            return Json{{"tensor_types", value.tensor_types},
                        {"tensor_shapes", value.tensor_shapes}};
          },
          [](const sc::SliceOpConfig& value) {
            return Json{{"begin", value.begin},
                        {"end", value.end},
                        {"input_shape", value.input_shape},
                        {"output_shape", value.output_shape}};
          },
          [](const sc::ReshapeOpConfig& value) { return Json{{"new_shape", value.new_shape}}; },
          [](const sc::DetessellateOpConfig& value) {
            return Json{{"slice_shape", value.slice_shape},
                        {"frame_shape", value.frame_shape},
                        {"align_c16", value.align_c16},
                        {"cblock", value.cblock},
                        {"frame_type", value.frame_type}};
          },
          [](const sc::DequantizeOpConfig& value) {
            return Json{{"input_dtype", value.input_dtype},
                        {"channel_params", quantization_json(value.channel_params)}};
          },
          [](const sc::HostTvmOpConfig& value) {
            const auto types_json = [](const auto& types) {
              Json result = Json::array();
              for (const auto& type : types) {
                result.push_back({{"scalar", type.scalar}, {"shape", type.shape}});
              }
              return result;
            };
            return Json{{"executable", value.executable},
                        {"executable_bytes", value.executable_bytes},
                        {"executable_sha256", value.executable_sha256},
                        {"input_names", value.input_names},
                        {"input_types", types_json(value.input_types)},
                        {"output_types", types_json(value.output_types)},
                        {"output_alias_input", value.output_alias_input},
                        {"linked_parameter_names", value.linked_parameter_names}};
          },
          [](const sc::PassThroughOpConfig&) { return Json::object(); }},
      config);
}

DmabufEligibilityCode map_decode_code(const sc::AfeMpkV2DecodeErrorCode code) noexcept {
  switch (code) {
  case sc::AfeMpkV2DecodeErrorCode::InvalidJson:
    return DmabufEligibilityCode::InvalidJson;
  case sc::AfeMpkV2DecodeErrorCode::MissingRequiredField:
    return DmabufEligibilityCode::MissingRequiredField;
  case sc::AfeMpkV2DecodeErrorCode::InvalidField:
    return DmabufEligibilityCode::InvalidField;
  case sc::AfeMpkV2DecodeErrorCode::UnsupportedContractVersion:
    return DmabufEligibilityCode::UnsupportedContractVersion;
  case sc::AfeMpkV2DecodeErrorCode::UnsupportedKernel:
    return DmabufEligibilityCode::UnsupportedKernel;
  case sc::AfeMpkV2DecodeErrorCode::UnsupportedHostModule:
    return DmabufEligibilityCode::UnsupportedKernel;
  case sc::AfeMpkV2DecodeErrorCode::InvalidKernelArity:
    return DmabufEligibilityCode::InvalidKernelArity;
  case sc::AfeMpkV2DecodeErrorCode::DuplicateSequence:
    return DmabufEligibilityCode::DuplicateSequence;
  case sc::AfeMpkV2DecodeErrorCode::DuplicateProducer:
    return DmabufEligibilityCode::DuplicateProducer;
  case sc::AfeMpkV2DecodeErrorCode::MissingProducer:
    return DmabufEligibilityCode::MissingProducer;
  case sc::AfeMpkV2DecodeErrorCode::ValueSizeMismatch:
    return DmabufEligibilityCode::ValueSizeMismatch;
  case sc::AfeMpkV2DecodeErrorCode::ConfigurationMismatch:
    return DmabufEligibilityCode::ConfigurationMismatch;
  case sc::AfeMpkV2DecodeErrorCode::MissingMlaStage:
    return DmabufEligibilityCode::MissingMlaStage;
  case sc::AfeMpkV2DecodeErrorCode::MultipleMlaStages:
    return DmabufEligibilityCode::MultipleMlaStages;
  case sc::AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence:
    return DmabufEligibilityCode::MissingMlaExecutableEvidence;
  case sc::AfeMpkV2DecodeErrorCode::AmbiguousMlaExecutableEvidence:
    return DmabufEligibilityCode::AmbiguousMlaExecutableEvidence;
  case sc::AfeMpkV2DecodeErrorCode::UnexpectedMlaExecutableEvidence:
    return DmabufEligibilityCode::UnexpectedMlaExecutableEvidence;
  case sc::AfeMpkV2DecodeErrorCode::MissingPublicationStage:
    return DmabufEligibilityCode::MissingPublicationStage;
  case sc::AfeMpkV2DecodeErrorCode::InvalidPublicationStage:
    return DmabufEligibilityCode::InvalidPublicationStage;
  case sc::AfeMpkV2DecodeErrorCode::ElfTopologyInvalid:
    return DmabufEligibilityCode::ElfTopologyInvalid;
  case sc::AfeMpkV2DecodeErrorCode::ElfTopologyMismatch:
    return DmabufEligibilityCode::ElfTopologyMismatch;
  case sc::AfeMpkV2DecodeErrorCode::PlanValidationFailed:
    return DmabufEligibilityCode::PlanValidationFailed;
  case sc::AfeMpkV2DecodeErrorCode::IoError:
    return DmabufEligibilityCode::IoError;
  }
  return DmabufEligibilityCode::InternalError;
}

DmabufPlanCompileResult rejected(const DmabufEligibilityCode code, std::string source,
                                 std::string location, std::string detail) {
  DmabufPlanCompileResult result;
  result.report.code = code;
  result.report.source = std::move(source);
  result.report.location = std::move(location);
  result.report.detail = std::move(detail);
  return result;
}

} // namespace

const char* dmabuf_eligibility_code_name(const DmabufEligibilityCode code) noexcept {
  switch (code) {
  case DmabufEligibilityCode::NotEvaluated:
    return "not-evaluated";
  case DmabufEligibilityCode::Eligible:
    return "eligible";
  case DmabufEligibilityCode::MissingMpkManifest:
    return "missing-mpk-manifest";
  case DmabufEligibilityCode::MissingMlaExecutable:
    return "missing-mla-executable";
  case DmabufEligibilityCode::MissingMlaExecutableEvidence:
    return "missing-mla-executable-evidence";
  case DmabufEligibilityCode::AmbiguousMlaExecutableEvidence:
    return "ambiguous-mla-executable-evidence";
  case DmabufEligibilityCode::UnexpectedMlaExecutableEvidence:
    return "unexpected-mla-executable-evidence";
  case DmabufEligibilityCode::ElfTopologyUnreadable:
    return "elf-topology-unreadable";
  case DmabufEligibilityCode::InvalidJson:
    return "invalid-json";
  case DmabufEligibilityCode::MissingRequiredField:
    return "missing-required-field";
  case DmabufEligibilityCode::InvalidField:
    return "invalid-field";
  case DmabufEligibilityCode::UnsupportedContractVersion:
    return "unsupported-contract-version";
  case DmabufEligibilityCode::UnsupportedKernel:
    return "unsupported-kernel";
  case DmabufEligibilityCode::InvalidKernelArity:
    return "invalid-kernel-arity";
  case DmabufEligibilityCode::DuplicateSequence:
    return "duplicate-sequence";
  case DmabufEligibilityCode::DuplicateProducer:
    return "duplicate-producer";
  case DmabufEligibilityCode::MissingProducer:
    return "missing-producer";
  case DmabufEligibilityCode::ValueSizeMismatch:
    return "value-size-mismatch";
  case DmabufEligibilityCode::ConfigurationMismatch:
    return "configuration-mismatch";
  case DmabufEligibilityCode::MissingMlaStage:
    return "missing-mla-stage";
  case DmabufEligibilityCode::MultipleMlaStages:
    return "multiple-mla-stages";
  case DmabufEligibilityCode::MissingPublicationStage:
    return "missing-publication-stage";
  case DmabufEligibilityCode::InvalidPublicationStage:
    return "invalid-publication-stage";
  case DmabufEligibilityCode::ElfTopologyInvalid:
    return "elf-topology-invalid";
  case DmabufEligibilityCode::ElfTopologyMismatch:
    return "elf-topology-mismatch";
  case DmabufEligibilityCode::PlanValidationFailed:
    return "plan-validation-failed";
  case DmabufEligibilityCode::ArenaPlanInvalid:
    return "arena-plan-invalid";
  case DmabufEligibilityCode::IoError:
    return "io-error";
  case DmabufEligibilityCode::InternalError:
    return "internal-error";
  }
  return "internal-error";
}

std::string canonical_dmabuf_plan_json(const sc::ModelExecutionPlan& plan) {
  Json root;
  root["schema_version"] = 2;
  root["contract_version"] = plan.contract_version();
  root["model_inputs"] = plan.model_inputs();

  Json values = Json::array();
  for (const auto& value : plan.values()) {
    Json entry{{"id", value.id},
               {"name", value.name},
               {"required_bytes", value.required_bytes},
               {"representation", value_representation_name(value.representation)},
               {"quantization", quantization_json(value.quantization)}};
    entry["logical_dtype"] = value.logical_dtype ? Json(*value.logical_dtype) : Json(nullptr);
    entry["logical_shape"] = value.logical_shape ? Json(*value.logical_shape) : Json(nullptr);
    entry["logical_layout"] = value.logical_layout ? Json(*value.logical_layout) : Json(nullptr);
    if (value.read_expression) {
      entry["read_expression"] = {{"source_value_id", value.read_expression->source_value_id},
                                  {"byte_offset", value.read_expression->byte_offset},
                                  {"stride_bytes", value.read_expression->stride_bytes}};
    } else {
      entry["read_expression"] = nullptr;
    }
    if (value.storage_binding) {
      const auto& binding = *value.storage_binding;
      const char* kind = binding.kind == sc::StorageBindingKind::External
                             ? "external"
                             : (binding.kind == sc::StorageBindingKind::View ? "view" : "root");
      const char* access = binding.access == sc::StorageAccess::ReadOnly
                               ? "read"
                               : (binding.access == sc::StorageAccess::WriteOnly ? "write"
                                                                                : "read-write");
      entry["storage_binding"] = {
          {"kind", kind},
          {"carrier_id", binding.carrier_id},
          {"byte_offset", binding.byte_offset},
          {"physical_span", binding.physical_span},
          {"stride_bytes", binding.stride_bytes},
          {"access", access},
          {"source_value_id", binding.source_value_id ? Json(*binding.source_value_id)
                                                       : Json(nullptr)}};
    } else {
      entry["storage_binding"] = nullptr;
    }
    values.push_back(std::move(entry));
  }
  root["values"] = std::move(values);

  Json carriers = Json::array();
  for (const auto& carrier : plan.carriers()) {
    carriers.push_back({{"id", carrier.id},
                        {"required_bytes", carrier.required_bytes},
                        {"required_alignment_bytes", carrier.required_alignment_bytes},
                        {"representation", value_representation_name(carrier.representation)}});
  }
  root["carriers"] = std::move(carriers);

  Json ops = Json::array();
  for (const auto& op : plan.ops()) {
    ops.push_back({{"id", op.id},
                   {"sequence", op.sequence},
                   {"name", op.name},
                   {"kind", op_kind_name(op.kind)},
                   {"processor", op.processor},
                   {"kernel", op.kernel},
                   {"implementation_id", op.implementation_id},
                   {"implementation_abi_version", op.implementation_abi_version},
                   {"dependencies", op.dependencies},
                   {"inputs", op.inputs},
                   {"outputs", op.outputs},
                   {"input_shapes", op.input_shapes},
                   {"output_shapes", op.output_shapes},
                   {"config", op_config_json(op.config)}});
  }
  root["ops"] = std::move(ops);

  Json mla_stages = Json::array();
  for (std::size_t index = 0; index < plan.mla_stage_count(); ++index) {
    const auto* stage = plan.mla_stage(index);
    if (!stage) {
      continue;
    }
    mla_stages.push_back({{"stage_index", stage->key.stage_index},
                          {"op_id", stage->key.op_id},
                          {"logical_stage_id", stage->key.logical_stage_id},
                          {"executable", stage->key.executable}});
  }
  root["mla_stages"] = std::move(mla_stages);

  Json ports = Json::array();
  for (const auto& port : plan.backend_ports()) {
    ports.push_back(
        {{"stage_index", port.stage_index},
         {"direction", port.direction == sc::BackendPortDirection::Input ? "input" : "output"},
         {"port_index", port.port_index},
         {"elf_symbol", port.elf_symbol},
         {"value_id", port.value_id},
         {"required_bytes", port.physical_extent_bytes},
         {"required_alignment_bytes", port.required_alignment_bytes},
         {"alignment_authority",
          port.alignment_authority == sc::BackendPortAlignmentAuthority::Contract
              ? "contract"
              : "legacy-policy"},
         {"access", port.access == sc::BackendPortAccess::ReadOnly ? "read" : "write"}});
  }
  root["backend_ports"] = std::move(ports);

  Json outputs = Json::array();
  for (const auto& output : plan.model_outputs()) {
    outputs.push_back({{"public_index", output.public_index},
                       {"name", output.name},
                       {"value_id", output.value_id}});
  }
  root["model_outputs"] = std::move(outputs);
  return root.dump();
}

std::string dmabuf_plan_digest(const sc::ModelExecutionPlan& plan) {
  return sha256_text(canonical_dmabuf_plan_json(plan));
}

std::string canonical_dmabuf_execution_json(
    const sc::ModelExecutionPlan& plan, const sc::PhysicalExecutionPlan& physical,
    const sc::FrameSlotArenaPlan& arena) {
  Json root = Json::parse(canonical_dmabuf_plan_json(plan));
  root["schema_version"] = 3;

  Json commands = Json::array();
  for (const auto& command : physical.commands) {
    const char* engine = "unknown";
    switch (command.engine) {
    case sc::PhysicalEngine::Mla: engine = "mla"; break;
    case sc::PhysicalEngine::A65: engine = "a65"; break;
    case sc::PhysicalEngine::Cvu: engine = "cvu"; break;
    }
    commands.push_back({{"id", command.id},
                        {"engine", engine},
                        {"inputs", command.inputs},
                        {"outputs", command.outputs},
                        {"predecessors", command.predecessors},
                        {"successors", command.successors}});
  }
  root["physical_commands"] = std::move(commands);
  root["physical_digest_material"] = physical.deterministic_digest_material;

  const auto& placement = arena.placement();
  const char* domain = placement.domain == sc::ArenaStorageDomain::Cma
                           ? "cma"
                           : (placement.domain == sc::ArenaStorageDomain::Dms ? "dms"
                                                                              : "unknown");
  const char* provenance =
      placement.provenance == sc::ArenaAllocationProvenance::CoreAllocated
          ? "core-allocated"
          : (placement.provenance == sc::ArenaAllocationProvenance::ExternalAdopted
                 ? "external-adopted"
                 : "unknown");
  const char* escape = placement.escape == sc::ArenaEscapePolicy::CpuMappablePublic
                           ? "cpu-mappable-public"
                           : "internal-only";
  Json regions = Json::array();
  for (const auto& region : arena.regions()) {
    regions.push_back({{"carrier_id", region.carrier_id},
                       {"value_id", region.value_id},
                       {"byte_offset", region.byte_offset},
                       {"size_bytes", region.size_bytes},
                       {"required_alignment_bytes", region.required_alignment_bytes},
                       {"first_sequence", region.lifetime.first_sequence},
                       {"last_sequence", region.lifetime.last_sequence}});
  }
  Json detached_roots = Json::array();
  for (const auto root_id : arena.detached_roots()) {
    detached_roots.push_back(root_id);
  }
  root["frame_arena"] = {{"allocation_bytes", arena.allocation_bytes()},
                         {"allocation_alignment_bytes",
                          arena.allocation_alignment_bytes()},
                         {"storage_domain", domain},
                         {"allocation_provenance", provenance},
                         {"required_device_access",
                          placement.required_device_access},
                         {"escape_policy", escape},
                         {"detached_roots", std::move(detached_roots)},
                         {"regions", std::move(regions)}};
  return root.dump();
}

std::string dmabuf_execution_digest(const sc::ModelExecutionPlan& plan,
                                    const sc::PhysicalExecutionPlan& physical,
                                    const sc::FrameSlotArenaPlan& arena) {
  return sha256_text(canonical_dmabuf_execution_json(plan, physical, arena));
}

DmabufPlanCompileResult
try_compile_dmabuf_plan(const std::filesystem::path& mpk_manifest,
                        const std::filesystem::path& mla_executable) noexcept {
  const auto mpk_source = basename_or_placeholder(mpk_manifest, "<mpk-manifest>");
  try {
    std::error_code ec;
    if (mpk_manifest.empty() || !std::filesystem::is_regular_file(mpk_manifest, ec) || ec) {
      return rejected(DmabufEligibilityCode::MissingMpkManifest, mpk_source, "$",
                      "exact MPK manifest is missing or unreadable");
    }
    ec.clear();
    if (mla_executable.empty() || !std::filesystem::is_regular_file(mla_executable, ec) || ec) {
      return rejected(DmabufEligibilityCode::MissingMlaExecutable, mpk_source,
                      "$.plugins[processor=MLA].resources.executable",
                      "exact MLA executable is missing or unreadable");
    }

    const auto mpk_digest = sha256_file(mpk_manifest);
    const auto elf_digest = sha256_file(mla_executable);
    if (!mpk_digest || !elf_digest) {
      return rejected(DmabufEligibilityCode::IoError, mpk_source, "$",
                      "failed to hash an explicitly supplied model artifact");
    }

    sima::MlaElfIoTopology topology;
    if (!sima::read_mla_elf_io_topology(mla_executable, &topology)) {
      auto result =
          rejected(DmabufEligibilityCode::ElfTopologyUnreadable, mpk_source,
                   "$.plugins[processor=MLA].resources.executable",
                   sanitize_detail(topology.error.empty() ? "failed to read MLA ELF topology"
                                                          : topology.error,
                                   mpk_manifest, mla_executable));
      result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
      return result;
    }

    sc::AfeMpkV2Decoder decoder;
    auto decoded = decoder.decode_file(mpk_manifest, topology);
    if (!decoded || !decoded.plan) {
      const auto code = decoded.error ? map_decode_code(decoded.error->code)
                                      : DmabufEligibilityCode::InternalError;
      auto result = rejected(
          code, mpk_source, decoded.error ? decoded.error->json_path : "$",
          sanitize_detail(decoded.error ? decoded.error->detail : "strict decoder returned no plan",
                          mpk_manifest, mla_executable));
      result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
      return result;
    }

    std::string physical_error;
    auto physical = sc::PhysicalExecutionLowerer::lower(*decoded.plan, &physical_error);
    if (!physical) {
      auto result = rejected(DmabufEligibilityCode::PlanValidationFailed, mpk_source, "$.plugins",
                             std::move(physical_error));
      result.report.contract_version = decoded.plan->contract_version();
      result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
      return result;
    }

    std::string arena_error;
    const auto detached_roots = sc::detached_mla_output_roots(*decoded.plan, *physical);
    const auto arena = sc::FrameSlotArenaPlan::compile(
        *decoded.plan, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
        sc::kLegacyEvoCmaRegionAlignmentBytes, &arena_error, sc::kModalixProductionArenaDmsPolicy,
        detached_roots);
    if (!arena) {
      auto result = rejected(DmabufEligibilityCode::ArenaPlanInvalid, mpk_source, "$.plugins",
                             std::move(arena_error));
      result.report.contract_version = decoded.plan->contract_version();
      result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
      return result;
    }

    DmabufPlanCompileResult result;
    result.plan_digest = dmabuf_execution_digest(*decoded.plan, *physical, *arena);
    if (result.plan_digest.empty()) {
      return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                      "failed to compute canonical plan digest");
    }
    result.report.code = DmabufEligibilityCode::Eligible;
    result.report.source = mpk_source;
    result.report.location = "$";
    result.report.detail = "strict MPK, ELF topology, immutable plan, and frame arena accepted";
    result.report.contract_version = decoded.plan->contract_version();
    result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
    result.report.proof.reserve(decoded.proof.size() + 1U);
    for (auto& fact : decoded.proof) {
      result.report.proof.push_back({std::move(fact.subject), std::move(fact.evidence)});
    }
    result.report.proof.push_back(
        {"frame-slot-arena", "regions=" + std::to_string(arena->regions().size()) +
                                 ";used_bytes=" + std::to_string(arena->used_bytes()) +
                                 ";allocation_bytes=" + std::to_string(arena->allocation_bytes()) +
                                 detached_root_proof(arena->detached_roots())});
    result.report.proof.push_back(
        {"physical-execution-plan",
         "commands=" + std::to_string(physical->commands.size()) + ";aligned-cvu-member-capacity=" +
             std::to_string(sc::minimum_cvu_member_capacity(*physical).value_or(0U))});
    result.plan = std::move(decoded.plan);
    result.arena_plan = *arena;
    result.physical_plan = std::move(physical);
    return result;
  } catch (const std::exception& error) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                    sanitize_detail(error.what(), mpk_manifest, mla_executable));
  } catch (...) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                    "unknown strict-plan compiler failure");
  }
}

DmabufPlanCompileResult
try_compile_dmabuf_plan(const std::filesystem::path& mpk_manifest,
                        const std::vector<MlaExecutableArtifact>& mla_executables) noexcept {
  return try_compile_dmabuf_plan(mpk_manifest, mla_executables,
                                 std::vector<HostTvmExecutableArtifact>{});
}

DmabufPlanCompileResult
try_compile_dmabuf_plan(const std::filesystem::path& mpk_manifest,
                        const std::vector<MlaExecutableArtifact>& mla_executables,
                        const std::vector<HostTvmExecutableArtifact>& host_executables) noexcept {
  const auto mpk_source = basename_or_placeholder(mpk_manifest, "<mpk-manifest>");
  try {
    std::error_code ec;
    if (mpk_manifest.empty() || !std::filesystem::is_regular_file(mpk_manifest, ec) || ec) {
      return rejected(DmabufEligibilityCode::MissingMpkManifest, mpk_source, "$",
                      "exact MPK manifest is missing or unreadable");
    }
    if (mla_executables.empty()) {
      return rejected(DmabufEligibilityCode::MissingMlaExecutableEvidence, mpk_source,
                      "$.plugins[processor=MLA].resources.executable",
                      "no exact MLA executable evidence was supplied");
    }

    const auto mpk_digest = sha256_file(mpk_manifest);
    if (!mpk_digest) {
      return rejected(DmabufEligibilityCode::IoError, mpk_source, "$",
                      "failed to hash the explicitly supplied MPK manifest");
    }

    std::vector<sc::MlaStageExecutableEvidence> evidence;
    evidence.reserve(mla_executables.size());
    std::string artifact_identity = "mpk=" + *mpk_digest;
    for (std::size_t index = 0; index < mla_executables.size(); ++index) {
      const auto& artifact = mla_executables[index];
      if (artifact.logical_stage_id.empty() || artifact.manifest_executable.empty()) {
        return rejected(DmabufEligibilityCode::MissingMlaExecutableEvidence, mpk_source,
                        "$.plugins[processor=MLA].resources.executable",
                        "MLA evidence has an empty logical stage or manifest executable identity");
      }
      ec.clear();
      if (artifact.resolved_path.empty() ||
          !std::filesystem::is_regular_file(artifact.resolved_path, ec) || ec) {
        return rejected(DmabufEligibilityCode::MissingMlaExecutable, mpk_source,
                        "$.plugins[processor=MLA].resources.executable",
                        "exact MLA executable for stage '" + artifact.logical_stage_id +
                            "' is missing or unreadable");
      }
      const auto digest = sha256_file(artifact.resolved_path);
      if (!digest) {
        return rejected(DmabufEligibilityCode::IoError, mpk_source,
                        "$.plugins[processor=MLA].resources.executable",
                        "failed to hash exact MLA executable for stage '" +
                            artifact.logical_stage_id + "'");
      }
      sima::MlaElfIoTopology topology;
      if (!sima::read_mla_elf_io_topology(artifact.resolved_path, &topology)) {
        auto result =
            rejected(DmabufEligibilityCode::ElfTopologyUnreadable, mpk_source,
                     "$.plugins[processor=MLA].resources.executable",
                     sanitize_detail(topology.error.empty() ? "failed to read MLA ELF topology"
                                                            : topology.error,
                                     mpk_manifest, artifact.resolved_path));
        result.report.artifact_digest = sha256_text(artifact_identity + ";elf=" + *digest);
        return result;
      }
      artifact_identity += ";stage=" + artifact.logical_stage_id +
                           ";executable=" + artifact.manifest_executable + ";elf=" + *digest;
      evidence.push_back(
          {artifact.logical_stage_id, artifact.manifest_executable, std::move(topology),
           static_cast<std::uint64_t>(std::filesystem::file_size(artifact.resolved_path)),
           *digest});
    }
    std::vector<sc::HostTvmExecutableEvidence> host_evidence;
    host_evidence.reserve(host_executables.size());
    for (const auto& artifact : host_executables) {
      if (artifact.logical_stage_id.empty() || artifact.manifest_executable.empty()) {
        return rejected(DmabufEligibilityCode::UnsupportedKernel, mpk_source,
                        "$.plugins[processor=A65].resources.executable",
                        "A65 evidence has an empty logical stage or executable identity");
      }
      ec.clear();
      if (artifact.resolved_path.empty() ||
          !std::filesystem::is_regular_file(artifact.resolved_path, ec) || ec) {
        return rejected(DmabufEligibilityCode::UnsupportedKernel, mpk_source,
                        "$.plugins[processor=A65].resources.executable",
                        "exact A65 host module for stage '" + artifact.logical_stage_id +
                            "' is missing or unreadable");
      }
      const auto digest = sha256_file(artifact.resolved_path);
      if (!digest) {
        return rejected(DmabufEligibilityCode::IoError, mpk_source,
                        "$.plugins[processor=A65].resources.executable",
                        "failed to hash exact A65 host module for stage '" +
                            artifact.logical_stage_id + "'");
      }
      std::string graph_error;
      auto graph = sc::read_tvm_host_module_graph(artifact.resolved_path, &graph_error);
      if (!graph) {
        return rejected(DmabufEligibilityCode::UnsupportedKernel, mpk_source,
                        "$.plugins[processor=A65].resources.executable",
                        graph_error.empty() ? "A65 host module has no valid GraphExecutor contract"
                                            : graph_error);
      }
      artifact_identity += ";host-stage=" + artifact.logical_stage_id +
                           ";executable=" + artifact.manifest_executable + ";so=" + *digest;
      sc::HostTvmExecutableEvidence item{
          artifact.logical_stage_id,
          artifact.manifest_executable,
          graph->input_names,
          graph->input_types,
          std::move(graph->output_types),
          std::move(graph->output_alias_input),
          static_cast<std::uint64_t>(std::filesystem::file_size(artifact.resolved_path)),
          *digest};
      item.argument_names = std::move(graph->input_names);
      item.argument_types = std::move(graph->input_types);
      host_evidence.push_back(std::move(item));
    }
    const auto artifact_digest = sha256_text(artifact_identity);

    auto decoded = sc::AfeMpkV2Decoder{}.decode_file(mpk_manifest, evidence, host_evidence);
    if (!decoded || !decoded.plan) {
      const auto code = decoded.error ? map_decode_code(decoded.error->code)
                                      : DmabufEligibilityCode::InternalError;
      auto result =
          rejected(code, mpk_source, decoded.error ? decoded.error->json_path : "$",
                   decoded.error ? decoded.error->detail : "strict decoder returned no plan");
      result.report.artifact_digest = artifact_digest;
      return result;
    }

    std::string physical_error;
    auto physical = sc::PhysicalExecutionLowerer::lower(*decoded.plan, &physical_error);
    if (!physical) {
      auto result = rejected(DmabufEligibilityCode::PlanValidationFailed, mpk_source, "$.plugins",
                             std::move(physical_error));
      result.report.contract_version = decoded.plan->contract_version();
      result.report.artifact_digest = artifact_digest;
      return result;
    }

    std::string arena_error;
    const auto detached_roots = sc::detached_mla_output_roots(*decoded.plan, *physical);
    auto arena = sc::FrameSlotArenaPlan::compile(
        *decoded.plan, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
        sc::kLegacyEvoCmaRegionAlignmentBytes, &arena_error, sc::kModalixProductionArenaDmsPolicy,
        detached_roots);
    if (!arena) {
      auto result = rejected(DmabufEligibilityCode::ArenaPlanInvalid, mpk_source, "$.plugins",
                             std::move(arena_error));
      result.report.contract_version = decoded.plan->contract_version();
      result.report.artifact_digest = artifact_digest;
      return result;
    }

    DmabufPlanCompileResult result;
    result.plan_digest = dmabuf_execution_digest(*decoded.plan, *physical, *arena);
    if (result.plan_digest.empty()) {
      return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                      "failed to compute canonical plan digest");
    }
    result.report.code = DmabufEligibilityCode::Eligible;
    result.report.source = mpk_source;
    result.report.location = "$";
    result.report.detail =
        "strict MPK, exact ELF/TVM artifact structures, immutable plan, and graph arena accepted";
    result.report.contract_version = decoded.plan->contract_version();
    result.report.artifact_digest = artifact_digest;
    result.report.proof.reserve(decoded.proof.size() + 1U);
    for (auto& fact : decoded.proof) {
      result.report.proof.push_back({std::move(fact.subject), std::move(fact.evidence)});
    }
    result.report.proof.push_back(
        {"frame-slot-arena", "regions=" + std::to_string(arena->regions().size()) +
                                 ";used_bytes=" + std::to_string(arena->used_bytes()) +
                                 ";allocation_bytes=" + std::to_string(arena->allocation_bytes()) +
                                 detached_root_proof(arena->detached_roots())});
    result.report.proof.push_back(
        {"physical-execution-plan",
         "commands=" + std::to_string(physical->commands.size()) + ";aligned-cvu-member-capacity=" +
             std::to_string(sc::minimum_cvu_member_capacity(*physical).value_or(0U))});
    result.plan = std::move(decoded.plan);
    result.arena_plan = std::move(arena);
    result.physical_plan = std::move(physical);
    return result;
  } catch (const std::exception& error) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$", error.what());
  } catch (...) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                    "unknown multi-stage strict-plan compiler failure");
  }
}

std::string dmabuf_plan_audit_json(const DmabufPlanCompileResult& result,
                                   const std::filesystem::path& mpk_manifest,
                                   const std::filesystem::path& mla_executable, const bool pretty) {
  Json proof = Json::array();
  for (const auto& fact : result.report.proof) {
    proof.push_back({{"subject", fact.subject}, {"evidence", fact.evidence}});
  }
  Json root{{"schema_version", 1},
            {"eligible", result.eligible()},
            {"code", dmabuf_eligibility_code_name(result.report.code)},
            {"source", result.report.source},
            {"location", result.report.location},
            {"detail", result.report.detail},
            {"contract_version", result.report.contract_version},
            {"artifact_digest", result.report.artifact_digest},
            {"plan_digest", result.plan_digest},
            {"mpk_basename", basename_or_placeholder(mpk_manifest, "<mpk-manifest>")},
            {"elf_basename", basename_or_placeholder(mla_executable, "<mla-executable>")},
            {"proof", std::move(proof)}};
  return root.dump(pretty ? 2 : -1);
}

std::string dmabuf_plan_audit_json(const DmabufPlanCompileResult& result,
                                   const std::filesystem::path& mpk_manifest,
                                   const std::vector<MlaExecutableArtifact>& mla_executables,
                                   const std::vector<HostTvmExecutableArtifact>& host_executables,
                                   const bool pretty) {
  Json root = Json::parse(dmabuf_plan_audit_json(result, mpk_manifest, mla_executables, false));
  root["schema_version"] = 3;
  Json artifacts = Json::array();
  for (const auto& artifact : host_executables) {
    artifacts.push_back(
        {{"logical_stage_id", artifact.logical_stage_id},
         {"manifest_executable", artifact.manifest_executable},
         {"so_basename", basename_or_placeholder(artifact.resolved_path, "<host-module>")}});
  }
  root["host_artifacts"] = std::move(artifacts);
  return root.dump(pretty ? 2 : -1);
}

std::string dmabuf_plan_audit_json(const DmabufPlanCompileResult& result,
                                   const std::filesystem::path& mpk_manifest,
                                   const std::vector<MlaExecutableArtifact>& mla_executables,
                                   const bool pretty) {
  Json proof = Json::array();
  for (const auto& fact : result.report.proof) {
    proof.push_back({{"subject", fact.subject}, {"evidence", fact.evidence}});
  }
  Json artifacts = Json::array();
  for (const auto& artifact : mla_executables) {
    artifacts.push_back(
        {{"logical_stage_id", artifact.logical_stage_id},
         {"manifest_executable", artifact.manifest_executable},
         {"elf_basename", basename_or_placeholder(artifact.resolved_path, "<mla-executable>")}});
  }
  Json root{{"schema_version", 2},
            {"eligible", result.eligible()},
            {"code", dmabuf_eligibility_code_name(result.report.code)},
            {"source", result.report.source},
            {"location", result.report.location},
            {"detail", result.report.detail},
            {"contract_version", result.report.contract_version},
            {"artifact_digest", result.report.artifact_digest},
            {"plan_digest", result.plan_digest},
            {"mpk_basename", basename_or_placeholder(mpk_manifest, "<mpk-manifest>")},
            {"mla_artifacts", std::move(artifacts)},
            {"proof", std::move(proof)}};
  return root.dump(pretty ? 2 : -1);
}

} // namespace simaai::neat::pipeline_internal
