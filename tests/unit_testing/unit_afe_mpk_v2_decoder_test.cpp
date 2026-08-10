#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/KernelRegistry.h"
#include "pipeline/internal/sima/static_contract/AfeMpkV2Decoder.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::static_contract;

void check(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

MlaElfIoTopology monolithic_topology() {
  MlaElfIoTopology topology;
  topology.valid = true;
  topology.monolithic_ifm = true;
  topology.monolithic_ofm = true;
  topology.source_path = "synthetic.elf";
  return topology;
}

const std::string& valid_manifest() {
  static const std::string manifest = R"json({
    "name":"synthetic",
    "model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {
        "name":"cast0","sequence":1,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"cast_transform","params":{"out_dtype":"bfloat16",
            "input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
        "input_nodes":[{"name":"input","size":16}],
        "output_nodes":[{"name":"cast0","size":8}]
      },
      {
        "name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "number_of_quads_to_user":4},
        "input_nodes":[{"name":"cast0","size":8}],
        "output_nodes":[{"name":"mla0","size":8}],
        "resources":{"executable":"synthetic_mla.elf"}
      },
      {
        "name":"cast1","sequence":3,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"cast_transform","params":{"out_dtype":"float32",
            "input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
        "input_nodes":[{"name":"mla0","size":8}],
        "output_nodes":[{"name":"decorated/model/output:0","size":16}]
      },
      {
        "name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"decorated/model/output:0","size":16}],
        "output_nodes":[{"name":"pass_through_out_0","size":16}]
      }
    ]
  })json";
  return manifest;
}

const std::string& packed_read_manifest() {
  static const std::string manifest = R"json({
    "name":"packed-read-synthetic",
    "model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input0","size":16},{"name":"input1","size":16}],
    "plugins":[
      {
        "name":"pack","sequence":1,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pack_transform","params":{"input_shapes":[[1,16],[1,16]],
            "output_shapes":[[1,32]]}},
        "input_nodes":[{"name":"input0","size":16},{"name":"input1","size":16}],
        "output_nodes":[{"name":"packed_ifm","size":32}]
      },
      {
        "name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "number_of_quads_to_user":4},
        "input_nodes":[{"name":"packed_ifm","size":32}],
        "output_nodes":[{"name":"packed_ofm","size":32}],
        "resources":{"executable":"synthetic_mla.elf"}
      },
      {
        "name":"unpack","sequence":3,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"unpack_transform","params":{
            "tensor_types":["int8","int8"],
            "tensor_shapes":[[1,2,2,4],[1,2,2,4]],
            "input_shapes":[[1,32]],
            "output_shapes":[[1,2,2,4],[1,2,2,4]]}},
        "input_nodes":[{"name":"packed_ofm","size":32}],
        "output_nodes":[{"name":"unpacked0","size":16},{"name":"unpacked1","size":16}]
      },
      {
        "name":"slice0","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,0],"end":[1,2,2,1],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked0","size":16}],
        "output_nodes":[{"name":"slice0_out","size":4}]
      },
      {
        "name":"slice1","sequence":5,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,1],"end":[1,2,2,2],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked1","size":16}],
        "output_nodes":[{"name":"slice1_out","size":4}]
      },
      {
        "name":"publish","sequence":6,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"slice0_out","size":4},{"name":"slice1_out","size":4}],
        "output_nodes":[{"name":"pass_through_out_0","size":4},
          {"name":"pass_through_out_1","size":4}]
      }
    ]
  })json";
  return manifest;
}

const std::string& two_mla_manifest() {
  static const std::string manifest = R"json({
    "name":"two-mla-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"encoded","size":32}],
       "resources":{"executable":"encoder.so"}},
      {"name":"MLA_decoder","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"encoded","size":32}],
       "output_nodes":[{"name":"decoded","size":8}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":8}],
       "output_nodes":[{"name":"output","size":8}]}
    ]
  })json";
  return manifest;
}

const std::string& two_mla_with_a65_module_manifest() {
  static const std::string manifest = R"json({
    "name":"two-mla-a65-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"encoded","size":32}],
       "resources":{"executable":"encoder.elf"}},
      {"name":"APU_module","sequence":2,"processor":"A65","type":"sgpProcess",
       "config_params":{"input_names":["arm_3_i0"],
                        "input_types":[{"scalar":"float32","shape":[1,8]}],
                        "output_types":[{"scalar":"float32","shape":[1,8]}]},
       "input_nodes":[{"name":"encoded","size":32}],
       "output_nodes":[{"name":"transformed","size":32}],
       "resources":{"executable":"middle.so"}},
      {"name":"MLA_decoder","sequence":3,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"transformed","size":32}],
       "output_nodes":[{"name":"decoded","size":8}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":8}],
       "output_nodes":[{"name":"output","size":8}]}
    ]
  })json";
  return manifest;
}

std::string replace_once(std::string value, const std::string& before, const std::string& after) {
  const auto position = value.find(before);
  check(position != std::string::npos, "test mutation token exists");
  value.replace(position, before.size(), after);
  return value;
}

void expect_error(const std::string& manifest, const MlaElfIoTopology& topology,
                  const AfeMpkV2DecodeErrorCode expected, const char* message) {
  const auto result = AfeMpkV2Decoder{}.decode_json(manifest, topology, "synthetic.json");
  check(!result, message);
  check(result.error.has_value() && result.error->code == expected, message);
  check(!result.error->json_path.empty() && !result.error->detail.empty(),
        "failure has stable path and detail");
}

void test_exact_registry() {
  const auto legacy = lookup_exact_kernel("2.0.0", "EV74", "cast_transform");
  const auto modern_alias = lookup_exact_kernel("2.0.0", "EV74", "cast");
  check(legacy.has_value() && legacy->kind == OpKind::Cast, "legacy cast token has an exact entry");
  check(modern_alias.has_value() && modern_alias->kind == OpKind::Cast,
        "new cast spelling has its own exact entry");
  check(!lookup_exact_kernel("2.0.0", "EV74", "prefix_cast_transform").has_value(),
        "registry rejects substring matches");
  check(!lookup_exact_kernel("2.0.0", "ev74", "cast_transform").has_value(),
        "registry rejects processor case folding");
  check(!lookup_exact_kernel("2.0.1", "EV74", "cast_transform").has_value(),
        "registry rejects version fallback");
}

void test_success_and_immutable_contract() {
  const auto result =
      AfeMpkV2Decoder{}.decode_json(valid_manifest(), monolithic_topology(), "synthetic.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "valid manifest decodes");
  const auto& plan = *result.plan;
  check(plan.contract_version() == "2.0.0", "contract version preserved exactly");
  check(plan.model_inputs().size() == 1U && plan.model_outputs().size() == 1U,
        "public input/output arity preserved");
  check(plan.ops().size() == 4U && plan.values().size() == 5U,
        "ordered graph and value table materialized once");
  check(plan.model_outputs()[0].name == "pass_through_out_0",
        "terminal full output name preserved");
  check(plan.value(plan.model_outputs()[0].value_id)->logical_dtype == "float32",
        "typed inverse preserves published dtype");
  check(plan.backend_ports().size() == 2U, "monolithic ELF is exactly 1 IFM / 1 OFM");
  for (const auto& port : plan.backend_ports()) {
    check(port.required_alignment_bytes == kLegacyEvoCmaRegionAlignmentBytes,
          "legacy port uses documented conservative alignment");
    check(port.alignment_authority == BackendPortAlignmentAuthority::LegacyPolicy,
          "legacy alignment provenance is policy, not MPK/ELF");
  }
  check(plan.backend_ports()[0].elf_symbol == "data.ifm.b0" &&
            plan.backend_ports()[1].elf_symbol == "data.ofm.b0",
        "exact monolithic symbols retained");
  check(!result.proof.empty(), "deterministic proof report emitted");
}

void test_unpack_and_slice_are_read_expressions() {
  const auto result = AfeMpkV2Decoder{}.decode_json(packed_read_manifest(), monolithic_topology(),
                                                    "packed-read-synthetic.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "packed read-expression manifest decodes");
  const auto& plan = *result.plan;
  check(plan.backend_ports().size() == 2U, "packed route remains one IFM and one OFM");
  check(plan.model_outputs().size() == 2U, "both logical reads are published");
  const auto* pack = std::get_if<PackOpConfig>(&plan.ops().front().config);
  check(pack != nullptr && pack->components.size() == 2U &&
            pack->components[0].parent_offset == 0U && pack->components[0].stored_bytes == 16U &&
            pack->components[1].parent_offset == 16U && pack->components[1].stored_bytes == 16U,
        "Pack carries exact ordered parent placement");

  // Value order is: two public inputs, pack, MLA, two unpack reads, two
  // slice reads, two publication reads.  Every read remains rooted in the
  // single materialized MLA OFM; neither Unpack nor Slice schedules work.
  const auto require_read = [&](const ValueId id, const std::uint64_t offset) {
    const auto* value = plan.value(id);
    check(value != nullptr && value->read_expression.has_value(),
          "logical value carries a compiled read expression");
    check(value->read_expression->source_value_id == 3U,
          "logical read is composed to the physical MLA carrier");
    check(value->read_expression->byte_offset == offset,
          "logical read preserves its exact carrier-relative offset");
    check(value->read_expression->stride_bytes == std::vector<std::int64_t>({16, 8, 4, 1}),
          "logical read preserves the exact inherited byte strides");
  };
  require_read(4U, 0U);
  require_read(5U, 16U);
  require_read(6U, 0U);
  require_read(7U, 17U);
  require_read(8U, 0U);
  require_read(9U, 17U);

  std::size_t read_proofs = 0U;
  for (const auto& fact : result.proof) {
    if (fact.subject.rfind("read[", 0U) == 0U &&
        fact.evidence.find("no runtime operation is scheduled") != std::string::npos) {
      ++read_proofs;
    }
  }
  check(read_proofs == 4U, "decoder proves both unpack and both slice reads are not jobs");
}

void test_fail_closed_cases() {
  const auto topology = monolithic_topology();
  expect_error(replace_once(valid_manifest(), "2.0.0", "2.0.1"), topology,
               AfeMpkV2DecodeErrorCode::UnsupportedContractVersion,
               "unsupported version fails closed");
  expect_error(replace_once(valid_manifest(), "cast_transform", "cast_transform_suffix"), topology,
               AfeMpkV2DecodeErrorCode::UnsupportedKernel, "kernel substring is not an alias");
  expect_error(replace_once(valid_manifest(),
                            "\"name\":\"cast0\",\"size\":8}],\n        \"output_nodes\"",
                            "\"name\":\"missing\",\"size\":8}],\n        \"output_nodes\""),
               topology, AfeMpkV2DecodeErrorCode::MissingProducer,
               "missing full-name producer fails closed");
  expect_error(replace_once(valid_manifest(), "\"params\":{\"out_dtype\":\"bfloat16\"",
                            "\"params\":{\"ignored\":1,\"out_dtype\":\"bfloat16\""),
               topology, AfeMpkV2DecodeErrorCode::InvalidField,
               "untyped extra operation config is not ignored");

  auto two_ifm = topology;
  two_ifm.monolithic_ifm = false;
  two_ifm.ifm_symbol_names = {"data.ifm.persistent.qmla_ifm_0.b0",
                              "data.ifm.persistent.qmla_ifm_1.b0"};
  expect_error(valid_manifest(), two_ifm, AfeMpkV2DecodeErrorCode::ElfTopologyMismatch,
               "MPK/ELF port arity mismatch fails before plan creation");

  auto conflict = topology;
  conflict.ifm_layout_conflict = true;
  expect_error(valid_manifest(), conflict, AfeMpkV2DecodeErrorCode::ElfTopologyInvalid,
               "ambiguous ELF layout fails closed");
}

void test_exact_multi_mla_evidence() {
  const auto topology = monolithic_topology();
  const std::vector<MlaStageExecutableEvidence> evidence{
      {"MLA_decoder", "decoder.elf", topology},
      {"MLA_encoder", "encoder.so", topology},
  };
  const auto result = AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), evidence, "two-mla.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "two MLA stages join evidence by identity, not list order");
  const auto& plan = *result.plan;
  check(plan.mla_stage_count() == 2U, "two immutable MLA stage keys are indexed");
  check(plan.mla_stage(0)->key.logical_stage_id == "MLA_encoder" &&
            plan.mla_stage(0)->key.executable == "encoder.so" &&
            plan.mla_stage(1)->key.logical_stage_id == "MLA_decoder" &&
            plan.mla_stage(1)->key.executable == "decoder.elf",
        "stage keys preserve MPK graph order and exact executable tokens");
  check(plan.mla_stage_for_identity("MLA_encoder", "encoder.so") == plan.mla_stage(0) &&
            plan.mla_stage_for_identity("MLA_encoder", "decoder.elf") == nullptr,
        "stage identity lookup requires both the MPK logical id and executable token");
  check(plan.backend_ports(0, BackendPortDirection::Input).size() == 1U &&
            plan.backend_ports(0, BackendPortDirection::Output).front().required_bytes == 32U &&
            plan.backend_ports(1, BackendPortDirection::Input).front().required_bytes == 32U &&
            plan.backend_ports(1, BackendPortDirection::Output).front().required_bytes == 8U,
        "each stage retains an independent dense ordered port span");

  auto missing = evidence;
  missing.pop_back();
  const auto missing_result =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), missing, "two-mla.json");
  check(!missing_result &&
            missing_result.error->code == AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence,
        "missing exact stage evidence fails before plan creation");

  auto wrong = evidence;
  wrong[1].executable = "decoder.elf";
  const auto wrong_result =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), wrong, "two-mla.json");
  check(!wrong_result &&
            wrong_result.error->code == AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence,
        "swapped executable identity cannot bind by topology position");

  const auto ambiguous_single =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), topology, "two-mla.json");
  check(!ambiguous_single &&
            ambiguous_single.error->code == AfeMpkV2DecodeErrorCode::MultipleMlaStages,
        "single-topology compatibility API rejects a multi-stage manifest");

  const auto a65_result = AfeMpkV2Decoder{}.decode_json(two_mla_with_a65_module_manifest(),
                                                        evidence, "two-mla-a65.json");
  check(!a65_result && a65_result.error->code == AfeMpkV2DecodeErrorCode::UnsupportedHostModule &&
            a65_result.error->json_path.find("resources.executable") != std::string::npos,
        "an AFE A65/ProcessTVM .so is an explicit host module and fails closed until its "
        "typed direct module ABI exists; it is never interpreted as an MLA ELF by suffix");
}

int validate_explicit_pair(const char* manifest_path, const char* elf_path) {
  MlaElfIoTopology topology;
  if (!read_mla_elf_io_topology(elf_path, &topology)) {
    std::cerr << topology.error << "\n";
    return 2;
  }
  const auto result = AfeMpkV2Decoder{}.decode_file(manifest_path, topology);
  if (!result) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
    return 1;
  }
  std::size_t ifm_count = 0U;
  std::size_t ofm_count = 0U;
  for (const auto& port : result.plan->backend_ports()) {
    if (port.direction == BackendPortDirection::Input) {
      ++ifm_count;
    } else {
      ++ofm_count;
    }
  }
  std::cout << "inputs=" << result.plan->model_inputs().size()
            << " outputs=" << result.plan->model_outputs().size() << " ifm=" << ifm_count
            << " ofm=" << ofm_count << "\n";
  return 0;
}

} // namespace

int main(const int argc, char** argv) {
  if (argc == 3) {
    return validate_explicit_pair(argv[1], argv[2]);
  }
  check(argc == 1, "usage: unit_afe_mpk_v2_decoder_test [manifest elf]");
  test_exact_registry();
  test_success_and_immutable_contract();
  test_unpack_and_slice_are_read_expressions();
  test_exact_multi_mla_evidence();
  test_fail_closed_cases();
  std::cout << "unit_afe_mpk_v2_decoder_test: PASS\n";
  return 0;
}
