#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/DmabufEligibility.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

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

void test_canonical_digest() {
  const auto plan = make_plan("digest-v1");
  const auto canonical = simaai::neat::pipeline_internal::canonical_dmabuf_plan_json(plan);
  const auto first = simaai::neat::pipeline_internal::dmabuf_plan_digest(plan);
  const auto second = simaai::neat::pipeline_internal::dmabuf_plan_digest(plan);
  CHECK(!canonical.empty());
  CHECK(nlohmann::json::parse(canonical).at("schema_version") == 1);
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

} // namespace

int main() {
  test_canonical_digest();
  test_structured_rejections_and_sanitized_json();
  if (failures != 0) {
    std::cerr << failures << " DMA-BUF eligibility checks failed\n";
    return 1;
  }
  std::cout << "DMA-BUF eligibility checks passed\n";
  return 0;
}
