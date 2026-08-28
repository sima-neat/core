#define SIMA_NEAT_INTERNAL 1
#include "model/internal/ModelPack.h"
#include "pipeline/internal/DmabufEligibility.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"

#include <glib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

namespace {

int failures = 0;

#define CHECK(expr)                                                                                \
  do {                                                                                             \
    if (!(expr)) {                                                                                 \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #expr << '\n';                \
      ++failures;                                                                                  \
    }                                                                                              \
  } while (false)

sc::ValueSpec value(sc::ValueId id, std::string name, std::uint64_t bytes) {
  sc::ValueSpec result;
  result.id = id;
  result.name = std::move(name);
  result.required_bytes = bytes;
  return result;
}

sc::ModelExecutionPlan make_plan(std::string contract_version) {
  sc::ModelExecutionPlanData data;
  data.contract_version = std::move(contract_version);
  data.values = {value(0, "input", 64), value(1, "mla-output", 64)};
  data.model_inputs = {0};

  sc::OpSpec mla;
  mla.id = 0;
  mla.sequence = 1;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {0};
  mla.outputs = {1};
  mla.config = sc::MlaOpConfig{"model.elf", 4};

  data.ops = {std::move(mla)};
  data.backend_ports = {
      {0, sc::BackendPortDirection::Input, 0, "data.ifm.b0", 0, 64, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Output, 0, "data.ofm.b0", 1, 64, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0, "public-output", 1}};

  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!plan) {
    std::cerr << "fixture creation failed: " << error << '\n';
    std::abort();
  }
  return std::move(*plan);
}

void write_monolithic_topology_elf(const std::filesystem::path& path) {
  struct Elf64Header {
    std::uint8_t ident[16]{};
    std::uint16_t type = 0;
    std::uint16_t machine = 0;
    std::uint32_t version = 0;
    std::uint64_t entry = 0;
    std::uint64_t program_header_offset = 0;
    std::uint64_t section_header_offset = 0;
    std::uint32_t flags = 0;
    std::uint16_t header_size = 0;
    std::uint16_t program_header_size = 0;
    std::uint16_t program_header_count = 0;
    std::uint16_t section_header_size = 0;
    std::uint16_t section_header_count = 0;
    std::uint16_t section_name_table_index = 0;
  };
  struct Elf64SectionHeader {
    std::uint32_t name = 0;
    std::uint32_t type = 0;
    std::uint64_t flags = 0;
    std::uint64_t address = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint32_t link = 0;
    std::uint32_t info = 0;
    std::uint64_t alignment = 0;
    std::uint64_t entry_size = 0;
  };
  static_assert(sizeof(Elf64Header) == 64U);
  static_assert(sizeof(Elf64SectionHeader) == 64U);

  constexpr char names[] = "\0.shstrtab\0data.ifm.b0\0data.ofm.b0\0";
  constexpr std::uint32_t shstrtab_name = 1U;
  constexpr std::uint32_t ifm_name = 11U;
  constexpr std::uint32_t ofm_name = 23U;
  const std::uint64_t names_offset = sizeof(Elf64Header);
  const std::uint64_t ifm_offset = (names_offset + sizeof(names) + 7U) & ~std::uint64_t{7U};
  const std::uint64_t ofm_offset = ifm_offset + 16U;
  const std::uint64_t sections_offset = ofm_offset + 16U;

  Elf64Header header;
  header.ident[0] = 0x7fU;
  header.ident[1] = 'E';
  header.ident[2] = 'L';
  header.ident[3] = 'F';
  header.ident[4] = 2U;
  header.ident[5] = 1U;
  header.ident[6] = 1U;
  header.type = 1U;
  header.machine = 183U;
  header.version = 1U;
  header.section_header_offset = sections_offset;
  header.header_size = sizeof(Elf64Header);
  header.section_header_size = sizeof(Elf64SectionHeader);
  header.section_header_count = 4U;
  header.section_name_table_index = 1U;

  std::vector<Elf64SectionHeader> sections(4U);
  sections[1].name = shstrtab_name;
  sections[1].type = 3U;
  sections[1].offset = names_offset;
  sections[1].size = sizeof(names);
  sections[1].alignment = 1U;
  sections[2].name = ifm_name;
  sections[2].type = 0x71ba0002U;
  sections[2].offset = ifm_offset;
  sections[2].size = 16U;
  sections[2].alignment = 8U;
  sections[3].name = ofm_name;
  sections[3].type = 0x71ba0002U;
  sections[3].offset = ofm_offset;
  sections[3].size = 16U;
  sections[3].alignment = 8U;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  CHECK(output.is_open());
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(names, sizeof(names));
  const auto written = names_offset + sizeof(names);
  std::vector<char> padding(static_cast<std::size_t>(ifm_offset - written), 0);
  output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
  const std::array<std::uint64_t, 2U> ifm_header{64U, 0U};
  const std::array<std::uint64_t, 2U> ofm_header{64U, 0U};
  output.write(reinterpret_cast<const char*>(ifm_header.data()),
               static_cast<std::streamsize>(sizeof(ifm_header)));
  output.write(reinterpret_cast<const char*>(ofm_header.data()),
               static_cast<std::streamsize>(sizeof(ofm_header)));
  output.write(reinterpret_cast<const char*>(sections.data()),
               static_cast<std::streamsize>(sections.size() * sizeof(sections.front())));
  CHECK(output.good());
}

const char* two_mla_manifest() {
  return R"json({
    "name":"eligibility-two-mla","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":64}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":64}],
       "output_nodes":[{"name":"encoded","size":64}],
       "resources":{"executable":"encoder.so"}},
      {"name":"MLA_decoder","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"encoded","size":64}],
       "output_nodes":[{"name":"decoded","size":64}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":64}],
       "output_nodes":[{"name":"output","size":64}]}
    ]
  })json";
}

void test_canonical_digest() {
  const auto plan = make_plan("digest-v1");
  const auto canonical = simaai::neat::pipeline_internal::canonical_dmabuf_plan_json(plan);
  const auto first = simaai::neat::pipeline_internal::dmabuf_plan_digest(plan);
  const auto second = simaai::neat::pipeline_internal::dmabuf_plan_digest(plan);
  CHECK(!canonical.empty());
  CHECK(nlohmann::json::parse(canonical).at("schema_version") == 2);
  const auto parsed = nlohmann::json::parse(canonical);
  CHECK(parsed.at("mla_stages").size() == 1U);
  CHECK(parsed.at("mla_stages").at(0).at("logical_stage_id") == "mla");
  CHECK(parsed.at("mla_stages").at(0).at("executable") == "model.elf");
  CHECK(first.size() == 64U);
  CHECK(first == second);
  CHECK(first != simaai::neat::pipeline_internal::dmabuf_plan_digest(make_plan("digest-v2")));

  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
  std::string error;
  const auto physical = sc::PhysicalExecutionLowerer::lower(plan, &error);
  const auto detached_roots =
      physical ? sc::detached_mla_output_roots(plan, *physical) : std::vector<sc::ValueId>{};
  const auto arena = physical ? sc::FrameSlotArenaPlan::compile(
                                    plan, *physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
                                    sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
                                    sc::kModalixProductionArenaDmsPolicy, detached_roots)
                              : std::nullopt;
  CHECK(physical.has_value());
  CHECK(arena.has_value());
  if (!arena) {
    std::cerr << "canonical execution fixture arena failed: " << error << '\n';
  }
  if (physical && arena) {
    const auto execution =
        simaai::neat::pipeline_internal::canonical_dmabuf_execution_json(plan, *physical, *arena);
    const auto execution_json = nlohmann::json::parse(execution);
    CHECK(execution_json.at("schema_version") == 3);
    CHECK(detached_roots.empty());
    CHECK(execution_json.at("frame_arena").at("storage_domain") == "dms");
    CHECK(execution_json.at("frame_arena").at("detached_roots").empty());
    CHECK(execution_json.at("frame_arena").at("allocation_provenance") == "core-allocated");
    CHECK(execution_json.at("frame_arena").at("required_device_access").get<std::uint32_t>() != 0U);
    CHECK(execution_json.at("physical_commands").size() == physical->commands.size());
    CHECK(!execution_json.at("physical_digest_material").get<std::string>().empty());
    CHECK(
        simaai::neat::pipeline_internal::dmabuf_execution_digest(plan, *physical, *arena).size() ==
        64U);
  }
}

void test_structured_rejections_and_sanitized_json() {
  using namespace simaai::neat::pipeline_internal;
  const auto absent =
      try_compile_dmabuf_plan("/tmp/not-present/mpk.json", "/tmp/not-present/model.elf");
  CHECK(!absent.eligible());
  CHECK(absent.report.code == DmabufEligibilityCode::MissingMpkManifest);
  CHECK(absent.report.location == "$");

  const auto root = std::filesystem::temp_directory_path() / "neat-dmabuf-eligibility-unit";
  std::filesystem::create_directories(root);
  const auto mpk = root / "fixture_mpk.json";
  {
    std::ofstream output(mpk);
    output << "{}\n";
  }
  const auto missing_elf = root / "missing-model.elf";
  const auto result = try_compile_dmabuf_plan(mpk, missing_elf);
  CHECK(!result.eligible());
  CHECK(result.report.code == DmabufEligibilityCode::MissingMlaExecutable);
  const auto audit = dmabuf_plan_audit_json(result, mpk, missing_elf, false);
  const auto parsed = nlohmann::json::parse(audit);
  CHECK(parsed.at("schema_version") == 1);
  CHECK(parsed.at("eligible") == false);
  CHECK(parsed.at("code") == "missing-mla-executable");
  CHECK(parsed.at("mpk_basename") == "fixture_mpk.json");
  CHECK(audit.find(root.string()) == std::string::npos);
  std::filesystem::remove_all(root);
}

void test_exact_multi_stage_artifact_admission() {
  using namespace simaai::neat::pipeline_internal;
  const auto root = std::filesystem::temp_directory_path() / "neat-dmabuf-multi-eligibility-unit";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  CHECK(!ec);
  const auto mpk = root / "mpk.json";
  const auto encoder = root / "encoder.so";
  const auto decoder = root / "decoder.elf";
  {
    std::ofstream output(mpk);
    output << two_mla_manifest();
  }
  write_monolithic_topology_elf(encoder);
  write_monolithic_topology_elf(decoder);

  const std::vector<MlaExecutableArtifact> reversed_evidence{
      {"MLA_decoder", "decoder.elf", decoder},
      {"MLA_encoder", "encoder.so", encoder},
  };
  const auto accepted = try_compile_dmabuf_plan(mpk, reversed_evidence);
  if (!accepted.eligible()) {
    std::cerr << "multi-stage admission rejected: code="
              << dmabuf_eligibility_code_name(accepted.report.code)
              << " location=" << accepted.report.location << " detail=" << accepted.report.detail
              << '\n';
  }
  CHECK(accepted.eligible());
  CHECK(accepted.plan && accepted.plan->mla_stage_count() == 2U);
  CHECK(accepted.arena_plan.has_value());
  if (accepted.plan && accepted.arena_plan) {
    CHECK(accepted.arena_plan->placement().domain == sc::ArenaStorageDomain::Dms);
    CHECK(accepted.arena_plan->placement().requires_access(sc::ArenaDeviceAccess::Mla));
    CHECK(!accepted.arena_plan->placement().requires_access(sc::ArenaDeviceAccess::CpuA65));
    CHECK(!accepted.arena_plan->placement().requires_access(sc::ArenaDeviceAccess::Ev74));
    const auto encoder_outputs = accepted.plan->backend_ports(0U, sc::BackendPortDirection::Output);
    const auto decoder_outputs = accepted.plan->backend_ports(1U, sc::BackendPortDirection::Output);
    CHECK(encoder_outputs.size() == 1U && decoder_outputs.size() == 1U);
    const auto* encoded = accepted.arena_plan->region(encoder_outputs.front().value_id);
    const auto* decoded = accepted.arena_plan->region(decoder_outputs.front().value_id);
    CHECK(encoded != nullptr);
    CHECK(decoded == nullptr);
    CHECK(accepted.arena_plan->is_detached_root(decoder_outputs.front().value_id));
    CHECK(accepted.arena_plan->detached_roots().size() == 1U &&
          accepted.arena_plan->detached_roots().front() == decoder_outputs.front().value_id);
    const auto arena_proof =
        std::find_if(accepted.report.proof.begin(), accepted.report.proof.end(),
                     [](const auto& fact) { return fact.subject == "frame-slot-arena"; });
    CHECK(arena_proof != accepted.report.proof.end());
    CHECK(arena_proof != accepted.report.proof.end() &&
          arena_proof->evidence.find("detached_count=1;detached_roots=[" +
                                     std::to_string(decoder_outputs.front().value_id) + "]") !=
              std::string::npos);
    CHECK(accepted.physical_plan.has_value());
    std::string unfiltered_error;
    const auto unfiltered_arena =
        accepted.physical_plan
            ? sc::FrameSlotArenaPlan::compile(
                  *accepted.plan, *accepted.physical_plan,
                  sc::FrameSlotArenaReuse::DisjointLifetimes, sc::kLegacyEvoCmaRegionAlignmentBytes,
                  &unfiltered_error, sc::kModalixProductionArenaDmsPolicy)
            : std::nullopt;
    CHECK(unfiltered_arena.has_value());
    CHECK(unfiltered_arena && dmabuf_execution_digest(*accepted.plan, *accepted.physical_plan,
                                                      *unfiltered_arena) != accepted.plan_digest);
  }
  CHECK(accepted.plan &&
        accepted.plan->mla_stage_for_identity("MLA_encoder", "encoder.so") != nullptr);
  const auto audit =
      nlohmann::json::parse(dmabuf_plan_audit_json(accepted, mpk, reversed_evidence, false));
  CHECK(audit.at("schema_version") == 2);
  CHECK(audit.at("mla_artifacts").size() == 2U);
  CHECK(audit.at("mla_artifacts").at(0).at("logical_stage_id") == "MLA_decoder");
  CHECK(audit.at("mla_artifacts").at(1).at("manifest_executable") == "encoder.so");
  CHECK(audit.dump().find(root.string()) == std::string::npos);

  auto ambiguous = reversed_evidence;
  ambiguous[0] = ambiguous[1];
  const auto rejected = try_compile_dmabuf_plan(mpk, ambiguous);
  CHECK(!rejected.eligible());
  CHECK(rejected.report.code == DmabufEligibilityCode::AmbiguousMlaExecutableEvidence);
  std::filesystem::remove_all(root, ec);
}

} // namespace

int main(int argc, char** argv) {
  // Optional integration mode for compiler-generated route fixtures.  This
  // deliberately enters through ModelPack rather than calling the decoder
  // directly, so an accepted contract must also render a complete strict
  // execution plan.  The normal no-argument unit-test behavior is unchanged.
  if (argc == 3 && std::string(argv[1]) == "--admit-modelpack") {
    try {
      CHECK(::setenv("SIMA_NEAT_MEMORY_BACKEND", "dmabuf-plan", 1) == 0);
      simaai::neat::internal::ModelPack model(argv[2]);
      const auto& admission = model.memory_backend_decision().admission;
      const auto execution = model.execution_plan();
      const auto stage_count =
          execution.pre.size() + execution.infer.size() + execution.post.size();
      std::size_t grouped_stage_count = 0U;
      std::size_t grouped_semantic_members = 0U;
      std::size_t grouped_physical_commands = 0U;
      const auto inspect_stages = [&](const auto& stages) {
        for (const auto& stage : stages) {
          if (stage.execution_op_ids.size() <= 1U) {
            continue;
          }
          ++grouped_stage_count;
          grouped_semantic_members += stage.execution_op_ids.size();
          grouped_physical_commands += stage.physical_command_ids.size();
          const std::size_t expected_commands =
              (stage.execution_op_ids.size() + 31U) / 32U;
          if (stage.physical_command_ids.size() != expected_commands) {
            throw std::runtime_error(
                "grouped strict stage lost its deterministic 32-member capacity split");
          }
        }
      };
      inspect_stages(execution.pre);
      inspect_stages(execution.infer);
      inspect_stages(execution.post);
      const auto facts = model.stage_facts_for_model_stage(
          simaai::neat::internal::ModelStage::Full);
      for (std::size_t index = 1U; index < facts.size(); ++index) {
        const auto& upstream = facts[index - 1U].processcvu_contract;
        const auto& consumer = facts[index].processcvu_contract;
        if (!upstream || !consumer ||
            upstream->runtime_contract.logical_outputs.size() <= 1U ||
            consumer->runtime_contract.logical_inputs.size() !=
                upstream->runtime_contract.logical_outputs.size()) {
          continue;
        }
        std::set<int> source_logicals;
        for (const auto& binding : consumer->runtime_contract.input_bindings) {
          if (binding.sink_pad_index != 0) {
            throw std::runtime_error(
                "grouped compatibility consumer requires more than the one upstream "
                "TensorBuffer link");
          }
          source_logicals.emplace(binding.src_logical_output_index);
        }
        if (source_logicals.size() !=
            consumer->runtime_contract.logical_inputs.size()) {
          throw std::runtime_error(
              "grouped compatibility consumer lost an upstream logical member");
        }
      }
      const auto fragment = model.fragment(simaai::neat::internal::ModelStage::Full);
      if (stage_count != 0U && fragment.gst.empty()) {
        throw std::runtime_error("strict execution plan produced an empty full-model fragment");
      }
      std::cout << "route-modelpack-admission: "
                << simaai::neat::pipeline_internal::dmabuf_eligibility_code_name(
                       admission.code)
                << " contract=" << admission.contract_version
                << " plan=" << model.memory_backend_decision().plan_digest
                << " stages=" << stage_count
                << " grouped-stages=" << grouped_stage_count
                << " grouped-members=" << grouped_semantic_members
                << " physical-cvu-commands=" << grouped_physical_commands
                << " fragment-elements=" << fragment.elements.size()
                << " fragment-bytes=" << fragment.gst.size()
                << '\n';
      return admission.eligible() ? 0 : 1;
    } catch (const std::exception& error) {
      std::cerr << "route-modelpack-admission: rejected: " << error.what() << '\n';
      return 1;
    }
  }
  if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--admit-modelpack MODEL_MPK_TAR_GZ]\n";
    return 2;
  }
  test_canonical_digest();
  test_structured_rejections_and_sanitized_json();
  test_exact_multi_stage_artifact_admission();
  if (failures != 0) {
    std::cerr << failures << " DMA-BUF eligibility checks failed\n";
    return 1;
  }
  std::cout << "DMA-BUF eligibility checks passed\n";
  return 0;
}
