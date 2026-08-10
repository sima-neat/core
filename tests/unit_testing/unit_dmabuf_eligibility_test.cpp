#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  data.values = {value(0, "input", 64), value(1, "mla-output", 64), value(2, "public-output", 64)};
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

  sc::OpSpec publish;
  publish.id = 1;
  publish.sequence = 2;
  publish.name = "publish";
  publish.kind = sc::OpKind::PassThrough;
  publish.processor = "EV74";
  publish.kernel = "pass_through";
  publish.inputs = {1};
  publish.outputs = {2};
  publish.config = sc::PassThroughOpConfig{};
  data.ops = {std::move(mla), std::move(publish)};
  data.backend_ports = {
      {0, sc::BackendPortDirection::Input, 0, "data.ifm.b0", 0, 64, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Output, 0, "data.ofm.b0", 1, 64, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0, "public-output", 2}};

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
  const std::uint64_t sections_offset = (names_offset + sizeof(names) + 7U) & ~std::uint64_t{7U};

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
  sections[2].type = 1U;
  sections[3].name = ofm_name;
  sections[3].type = 1U;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  CHECK(output.is_open());
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(names, sizeof(names));
  const auto written = names_offset + sizeof(names);
  std::vector<char> padding(static_cast<std::size_t>(sections_offset - written), 0);
  output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
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
  CHECK(nlohmann::json::parse(canonical).at("schema_version") == 1);
  const auto parsed = nlohmann::json::parse(canonical);
  CHECK(parsed.at("mla_stages").size() == 1U);
  CHECK(parsed.at("mla_stages").at(0).at("logical_stage_id") == "mla");
  CHECK(parsed.at("mla_stages").at(0).at("executable") == "model.elf");
  CHECK(first.size() == 64U);
  CHECK(first == second);
  CHECK(first != simaai::neat::pipeline_internal::dmabuf_plan_digest(make_plan("digest-v2")));
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
  CHECK(accepted.eligible());
  CHECK(accepted.plan && accepted.plan->mla_stage_count() == 2U);
  CHECK(accepted.arena_plan.has_value());
  if (accepted.plan && accepted.arena_plan) {
    const auto encoder_outputs = accepted.plan->backend_ports(0U, sc::BackendPortDirection::Output);
    const auto decoder_outputs = accepted.plan->backend_ports(1U, sc::BackendPortDirection::Output);
    CHECK(encoder_outputs.size() == 1U && decoder_outputs.size() == 1U);
    const auto* encoded = accepted.arena_plan->region(encoder_outputs.front().value_id);
    const auto* decoded = accepted.arena_plan->region(decoder_outputs.front().value_id);
    CHECK(encoded != nullptr && decoded != nullptr);
    CHECK(encoded && decoded && encoded->byte_offset != decoded->byte_offset);
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

int main() {
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
