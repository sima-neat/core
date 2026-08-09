#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/LegacyAfeMpkDecoder.h"

#include <glib.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
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
  case sc::OpKind::Detessellate:
    return "detessellate";
  case sc::OpKind::Dequantize:
    return "dequantize";
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
            return Json{{"components", std::move(components)}};
          },
          [](const sc::MlaOpConfig& value) {
            return Json{{"executable", value.executable},
                        {"number_of_quads", value.number_of_quads}};
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
          [](const sc::PassThroughOpConfig&) { return Json::object(); }},
      config);
}

DmabufEligibilityCode map_decode_code(const sc::LegacyAfeDecodeErrorCode code) noexcept {
  switch (code) {
  case sc::LegacyAfeDecodeErrorCode::InvalidJson:
    return DmabufEligibilityCode::InvalidJson;
  case sc::LegacyAfeDecodeErrorCode::MissingRequiredField:
    return DmabufEligibilityCode::MissingRequiredField;
  case sc::LegacyAfeDecodeErrorCode::InvalidField:
    return DmabufEligibilityCode::InvalidField;
  case sc::LegacyAfeDecodeErrorCode::UnsupportedContractVersion:
    return DmabufEligibilityCode::UnsupportedContractVersion;
  case sc::LegacyAfeDecodeErrorCode::UnsupportedKernel:
    return DmabufEligibilityCode::UnsupportedKernel;
  case sc::LegacyAfeDecodeErrorCode::InvalidKernelArity:
    return DmabufEligibilityCode::InvalidKernelArity;
  case sc::LegacyAfeDecodeErrorCode::DuplicateSequence:
    return DmabufEligibilityCode::DuplicateSequence;
  case sc::LegacyAfeDecodeErrorCode::DuplicateProducer:
    return DmabufEligibilityCode::DuplicateProducer;
  case sc::LegacyAfeDecodeErrorCode::MissingProducer:
    return DmabufEligibilityCode::MissingProducer;
  case sc::LegacyAfeDecodeErrorCode::ValueSizeMismatch:
    return DmabufEligibilityCode::ValueSizeMismatch;
  case sc::LegacyAfeDecodeErrorCode::ConfigurationMismatch:
    return DmabufEligibilityCode::ConfigurationMismatch;
  case sc::LegacyAfeDecodeErrorCode::MissingMlaStage:
    return DmabufEligibilityCode::MissingMlaStage;
  case sc::LegacyAfeDecodeErrorCode::MultipleMlaStages:
    return DmabufEligibilityCode::MultipleMlaStages;
  case sc::LegacyAfeDecodeErrorCode::MissingPublicationStage:
    return DmabufEligibilityCode::MissingPublicationStage;
  case sc::LegacyAfeDecodeErrorCode::InvalidPublicationStage:
    return DmabufEligibilityCode::InvalidPublicationStage;
  case sc::LegacyAfeDecodeErrorCode::ElfTopologyInvalid:
    return DmabufEligibilityCode::ElfTopologyInvalid;
  case sc::LegacyAfeDecodeErrorCode::ElfTopologyMismatch:
    return DmabufEligibilityCode::ElfTopologyMismatch;
  case sc::LegacyAfeDecodeErrorCode::PlanValidationFailed:
    return DmabufEligibilityCode::PlanValidationFailed;
  case sc::LegacyAfeDecodeErrorCode::IoError:
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
  root["schema_version"] = 1;
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
    values.push_back(std::move(entry));
  }
  root["values"] = std::move(values);

  Json ops = Json::array();
  for (const auto& op : plan.ops()) {
    ops.push_back({{"id", op.id},
                   {"sequence", op.sequence},
                   {"name", op.name},
                   {"kind", op_kind_name(op.kind)},
                   {"processor", op.processor},
                   {"kernel", op.kernel},
                   {"inputs", op.inputs},
                   {"outputs", op.outputs},
                   {"input_shapes", op.input_shapes},
                   {"output_shapes", op.output_shapes},
                   {"config", op_config_json(op.config)}});
  }
  root["ops"] = std::move(ops);

  Json ports = Json::array();
  for (const auto& port : plan.backend_ports()) {
    ports.push_back(
        {{"stage_index", port.stage_index},
         {"direction", port.direction == sc::BackendPortDirection::Input ? "input" : "output"},
         {"port_index", port.port_index},
         {"elf_symbol", port.elf_symbol},
         {"value_id", port.value_id},
         {"required_bytes", port.required_bytes},
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

    sc::LegacyAfeMpkDecoder decoder;
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

    std::string arena_error;
    const auto arena =
        sc::FrameSlotArenaPlan::compile(*decoded.plan, sc::FrameSlotArenaReuse::DisjointLifetimes,
                                        sc::kLegacyEvoCmaRegionAlignmentBytes, &arena_error);
    if (!arena) {
      auto result = rejected(DmabufEligibilityCode::ArenaPlanInvalid, mpk_source, "$.plugins",
                             std::move(arena_error));
      result.report.contract_version = decoded.plan->contract_version();
      result.report.artifact_digest = sha256_text("mpk=" + *mpk_digest + ";elf=" + *elf_digest);
      return result;
    }

    DmabufPlanCompileResult result;
    result.plan_digest = dmabuf_plan_digest(*decoded.plan);
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
                                 ";allocation_bytes=" + std::to_string(arena->allocation_bytes())});
    result.plan = std::move(decoded.plan);
    return result;
  } catch (const std::exception& error) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                    sanitize_detail(error.what(), mpk_manifest, mla_executable));
  } catch (...) {
    return rejected(DmabufEligibilityCode::InternalError, mpk_source, "$",
                    "unknown strict-plan compiler failure");
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

} // namespace simaai::neat::pipeline_internal
