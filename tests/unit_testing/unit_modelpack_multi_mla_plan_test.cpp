#define SIMA_NEAT_INTERNAL 1
#include "model/Model.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/MemoryBackendPolicy.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "test_main.h"

#include <glib.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <string_view>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_topology_elf(const std::filesystem::path& path, const std::string& ifm_name,
                        const std::uint64_t ifm_extent, const std::string& ofm_name,
                        const std::uint64_t ofm_extent) {
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

  std::string names(1U, '\0');
  const auto append_name = [&](const std::string& name) {
    const auto offset = static_cast<std::uint32_t>(names.size());
    names += name;
    names.push_back('\0');
    return offset;
  };
  const auto shstrtab_name = append_name(".shstrtab");
  const auto ifm_name_offset = append_name(ifm_name);
  const auto ofm_name_offset = append_name(ofm_name);
  const std::uint64_t names_offset = sizeof(Elf64Header);
  const std::uint64_t sections_offset =
      (names_offset + names.size() + 7U) & ~std::uint64_t{7U};

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
  sections[1].size = names.size();
  sections[1].alignment = 1U;
  sections[2].name = ifm_name_offset;
  sections[2].type = 0x71ba0002U;
  sections[2].offset = sections_offset + sections.size() * sizeof(sections.front());
  sections[2].size = 16U;
  sections[3].name = ofm_name_offset;
  sections[3].type = 0x71ba0002U;
  sections[3].offset = sections[2].offset + 16U;
  sections[3].size = 16U;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to create synthetic MLA ELF");
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.write(names.data(), static_cast<std::streamsize>(names.size()));
  const auto written = names_offset + names.size();
  std::vector<char> padding(static_cast<std::size_t>(sections_offset - written), 0);
  output.write(padding.data(), static_cast<std::streamsize>(padding.size()));
  output.write(reinterpret_cast<const char*>(sections.data()),
               static_cast<std::streamsize>(sections.size() * sizeof(sections.front())));
  const std::array<std::uint64_t, 2U> ifm_header{ifm_extent, 1U};
  const std::array<std::uint64_t, 2U> ofm_header{ofm_extent, 1U};
  output.write(reinterpret_cast<const char*>(ifm_header.data()), 16);
  output.write(reinterpret_cast<const char*>(ofm_header.data()), 16);
  require(output.good(), "failed to write synthetic MLA ELF");
}

void write_monolithic_topology_elf(const std::filesystem::path& path) {
  write_topology_elf(path, "data.ifm.b0", 64U, "data.ofm.b0", 64U);
}

void write_host_module(const std::filesystem::path& path, const std::uint64_t rows,
                       const std::vector<std::string>& input_names) {
  require(!input_names.empty(), "synthetic A65 module needs one or more inputs");
  nlohmann::json nodes = nlohmann::json::array();
  nlohmann::json arguments = nlohmann::json::array();
  nlohmann::json call_inputs = nlohmann::json::array();
  nlohmann::json shapes = nlohmann::json::array();
  nlohmann::json dtypes = nlohmann::json::array();
  nlohmann::json storage = nlohmann::json::array();
  nlohmann::json row_ptr = nlohmann::json::array({0});
  for (std::size_t index = 0U; index < input_names.size(); ++index) {
    nodes.push_back({{"op", "null"}, {"name", input_names[index]},
                     {"inputs", nlohmann::json::array()}});
    arguments.push_back(index);
    call_inputs.push_back(nlohmann::json::array({index, 0, 0}));
    shapes.push_back(nlohmann::json::array({rows, 1225U}));
    dtypes.push_back("float32");
    storage.push_back(index);
    row_ptr.push_back(index + 1U);
  }
  const auto call_index = input_names.size();
  nodes.push_back({{"op", "tvm_op"},
                   {"name", "identity"},
                   {"attrs", {{"func_name", "fused_identity"},
                              {"num_inputs", std::to_string(input_names.size())},
                              {"num_outputs", "1"}}},
                   {"inputs", call_inputs}});
  shapes.push_back(nlohmann::json::array({rows, 1225U}));
  dtypes.push_back("float32");
  storage.push_back(call_index);
  row_ptr.push_back(call_index + 1U);
  const nlohmann::ordered_json graph_json{
      {"nodes", std::move(nodes)},
      {"arg_nodes", std::move(arguments)},
      {"heads", nlohmann::json::array({nlohmann::json::array({call_index, 0, 0})})},
      {"attrs", {{"shape", nlohmann::json::array({"list_shape", shapes})},
                 {"dltype", nlohmann::json::array({"list_str", dtypes})},
                 {"storage_id", nlohmann::json::array({"list_int", storage})}}},
      {"node_row_ptr", std::move(row_ptr)}};
  const std::string graph = graph_json.dump(2);
  require(graph.starts_with("{\n  \"nodes\": ["),
          "synthetic A65 graph lost the frozen TVM GraphExecutor marker");
  std::string prefix(64U, '\0');
  prefix[0] = static_cast<char>(0x7fU);
  prefix[1] = 'E';
  prefix[2] = 'L';
  prefix[3] = 'F';
  prefix[4] = 2;
  prefix[5] = 1;
  prefix[16] = 3;                        // ET_DYN
  prefix[18] = static_cast<char>(183U); // EM_AARCH64
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to create synthetic A65 module");
  output.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  output.write(graph.data(), static_cast<std::streamsize>(graph.size()));
  output.put('\0');
  require(output.good(), "failed to write synthetic A65 module");
}

void write_host_module(const std::filesystem::path& path, const std::uint64_t rows) {
  write_host_module(path, rows, {"host_input"});
}

void write_mla_a65_manifest(const std::filesystem::path& path, const std::uint64_t rows) {
  const std::uint64_t logical_bytes = rows * 1225U * 4U;
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic MLA-to-A65 MPK");
  output << R"json({
    "name":"modelpack-mla-a65-pitch","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"float32","shape":[1,4]}],
                        "output_types":[{"scalar":"float32","shape":[)json"
         << rows << R"json(,1225]}]},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"pitched","size":)json"
         << logical_bytes << R"json(}],
       "resources":{"executable":"model.elf"}},
      {"name":"APU_1","sequence":2,"processor":"A65","type":"sgpProcess",
       "config_params":{"input_names":["host_input"],
                        "input_types":[{"scalar":"float32","shape":[)json"
         << rows << R"json(,1225]}],
                        "output_types":[{"scalar":"float32","shape":[)json"
         << rows << R"json(,1225]}]},
       "input_nodes":[{"name":"pitched","size":)json"
         << logical_bytes << R"json(}],
       "output_nodes":[{"name":"output","size":)json"
         << logical_bytes << R"json(}],
       "resources":{"executable":"host.so"}}
    ]
  })json";
  require(output.good(), "failed to write synthetic MLA-to-A65 MPK");
}

void write_a65_join_device_boundary_manifest(const std::filesystem::path& path) {
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic A65 join MPK");
  output << R"json({
    "name":"modelpack-a65-join-device-boundary","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":16,"dtype":"float32","shape":[1,4]}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"float32","shape":[1,4]}],
                        "output_types":[{"scalar":"float32","shape":[1,1225]}]},
       "input_nodes":[{"name":"input","size":16,"dtype":"float32","shape":[1,4]}],
       "output_nodes":[{"name":"mla_root","size":4900,"dtype":"float32","shape":[1,1225]}],
       "resources":{"executable":"first.elf"}},
      {"name":"APU_1","sequence":2,"processor":"A65","type":"sgpProcess",
       "config_params":{"input_names":["root_input"],
                        "input_types":[{"scalar":"float32","shape":[1,1225]}],
                        "output_types":[{"scalar":"float32","shape":[1,1225]}]},
       "input_nodes":[{"name":"mla_root","size":4900,"dtype":"float32","shape":[1,1225]}],
       "output_nodes":[{"name":"host_mid","size":4900,"dtype":"float32","shape":[1,1225]}],
       "resources":{"executable":"host_one.so"}},
      {"name":"APU_2","sequence":3,"processor":"A65","type":"sgpProcess",
       "config_params":{"input_names":["root_input","chain_input"],
                        "input_types":[{"scalar":"float32","shape":[1,1225]},
                                       {"scalar":"float32","shape":[1,1225]}],
                        "output_types":[{"scalar":"float32","shape":[1,1225]}]},
       "input_nodes":[{"name":"mla_root","size":4900,"dtype":"float32","shape":[1,1225]},
                      {"name":"host_mid","size":4900,"dtype":"float32","shape":[1,1225]}],
       "output_nodes":[{"name":"host_join","size":4900,"dtype":"float32","shape":[1,1225]}],
       "resources":{"executable":"host_join.so"}},
      {"name":"cast_after_cpu","sequence":4,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast_transform",
                        "params":{"out_dtype":"bfloat16",
                                  "input_shapes":[[1,1225]],
                                  "output_shapes":[[1,1225]]}},
       "input_nodes":[{"name":"host_join","size":4900,"dtype":"float32","shape":[1,1225]}],
       "output_nodes":[{"name":"cast_out","size":2450,"dtype":"bfloat16","shape":[1,1225]}]},
      {"name":"MLA_4","sequence":5,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"bfloat16","shape":[1,1225]}],
                        "output_types":[{"scalar":"bfloat16","shape":[1,1225]}]},
       "input_nodes":[{"name":"cast_out","size":2450,"dtype":"bfloat16","shape":[1,1225]}],
       "output_nodes":[{"name":"output","size":2450,"dtype":"bfloat16","shape":[1,1225]}],
       "resources":{"executable":"last.elf"}}
    ]
  })json";
  require(output.good(), "failed to write synthetic A65 join MPK");
}

void write_manifest(const std::filesystem::path& path) {
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic multi-MLA MPK");
  output << R"json({
    "name":"modelpack-two-mla","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":64,"dtype":"int8","shape":[1,64]}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"int8","shape":[1,64]}],
                        "output_types":[{"scalar":"int8","shape":[1,64]}]},
       "input_nodes":[{"name":"input","size":64,"dtype":"int8","shape":[1,64]}],
       "output_nodes":[{"name":"encoded","size":64,"dtype":"int8","shape":[1,64]}],
       "resources":{"executable":"encoder.so"}},
      {"name":"MLA_decoder","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"int8","shape":[1,64]}],
                        "output_types":[{"scalar":"int8","shape":[1,64]}]},
       "input_nodes":[{"name":"encoded","size":64,"dtype":"int8","shape":[1,64]}],
       "output_nodes":[{"name":"decoded","size":64,"dtype":"int8","shape":[1,64]}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":64,"dtype":"int8","shape":[1,64]}],
       "output_nodes":[{"name":"output","size":64,"dtype":"int8","shape":[1,64]}]}
    ]
  })json";
  require(output.good(), "failed to write synthetic multi-MLA MPK");
}

void write_boxdecode_cast_tail_manifest(const std::filesystem::path& path,
                                        const std::string& mla_scalar,
                                        const std::string& cast_output_scalar,
                                        const std::uint64_t input_elements,
                                        const std::uint64_t output_elements) {
  const auto scalar_bytes = [](const std::string& scalar) -> std::uint64_t {
    if (scalar == "bfloat16" || scalar == "float16") return 2U;
    if (scalar == "float32") return 4U;
    return 1U;
  };
  const std::string model_input_scalar =
      mla_scalar == "bfloat16" ? "float32" : "bfloat16";
  const auto model_input_bytes =
      input_elements * scalar_bytes(model_input_scalar);
  const auto input_bytes = input_elements * scalar_bytes(mla_scalar);
  const auto output_bytes = output_elements * scalar_bytes(cast_output_scalar);
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic MLA-to-Cast MPK");
  output << R"json({
    "name":"modelpack-boxdecode-cast-tail","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":)json"
         << model_input_bytes << R"json(,"dtype":")json" << model_input_scalar
         << R"json(","shape":[1,)json" << input_elements << R"json(]}],
    "plugins":[
      {"name":"pre_cast","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast_transform",
                        "params":{"out_dtype":")json" << mla_scalar
         << R"json(","input_shapes":[[1,)json" << input_elements
         << R"json(]],"output_shapes":[[1,)json" << input_elements << R"json(]]}},
       "input_nodes":[{"name":"input","size":)json" << model_input_bytes
         << R"json(,"dtype":")json" << model_input_scalar
         << R"json(","shape":[1,)json" << input_elements << R"json(]}],
       "output_nodes":[{"name":"mla_input","size":)json" << input_bytes
         << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}]},
      {"name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":")json" << mla_scalar
         << R"json(","shape":[1,)json" << input_elements << R"json(]}],
                        "output_types":[{"scalar":")json" << mla_scalar
         << R"json(","shape":[1,)json" << input_elements << R"json(]}]},
       "input_nodes":[{"name":"mla_input","size":)json" << input_bytes
         << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "output_nodes":[{"name":"mla_head","size":)json" << input_bytes
         << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "resources":{"executable":"model.elf"}},
      {"name":"cast_head","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast_transform",
                        "params":{"out_dtype":")json" << cast_output_scalar
         << R"json(","input_shapes":[[1,)json" << input_elements
         << R"json(]],"output_shapes":[[1,)json" << output_elements << R"json(]]}},
       "input_nodes":[{"name":"mla_head","size":)json" << input_bytes
         << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "output_nodes":[{"name":"class_prob_0","size":)json" << output_bytes
         << R"json(,"dtype":")json" << cast_output_scalar
         << R"json(","shape":[1,)json" << output_elements << R"json(]}]}
    ]
  })json";
  require(output.good(), "failed to write synthetic MLA-to-Cast MPK");
}

void write_archive_manifest(const std::filesystem::path& path) {
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic archive MPK");
  output << R"json({
    "name":"modelpack-json-preservation","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":64,"dtype":"int8","shape":[1,64]}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"int8","shape":[1,64]}],
                        "output_types":[{"scalar":"int8","shape":[1,64]}]},
       "input_nodes":[{"name":"input","size":64,"dtype":"int8","shape":[1,64]}],
       "output_nodes":[{"name":"output","size":64,"dtype":"int8","shape":[1,64]}],
       "resources":{"executable":"model.elf"}}
    ]
  })json";
  require(output.good(), "failed to write synthetic archive MPK");
}

void write_text(const std::filesystem::path& path, const std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.is_open(), "failed to create synthetic archive JSON");
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  require(output.good(), "failed to write synthetic archive JSON");
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.is_open(), "failed to read extracted archive JSON");
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string sha256(const std::string_view bytes) {
  gchar* digest = g_compute_checksum_for_data(
      G_CHECKSUM_SHA256, reinterpret_cast<const guchar*>(bytes.data()), bytes.size());
  require(digest != nullptr, "failed to compute archive JSON SHA-256");
  std::string result(digest);
  g_free(digest);
  return result;
}

std::string shell_quote(const std::string& text) {
  std::string result("'");
  for (const char ch : text) {
    result += ch == '\'' ? "'\\''" : std::string(1U, ch);
  }
  result += '\'';
  return result;
}

std::vector<nlohmann::json> direct_tvm_contracts(const std::string& fragment) {
  constexpr std::string_view prefix = "direct-contract-b64=";
  std::vector<nlohmann::json> result;
  for (std::size_t cursor = 0U;;) {
    const auto begin = fragment.find(prefix, cursor);
    if (begin == std::string::npos) {
      break;
    }
    const auto value_begin = begin + prefix.size();
    const auto value_end = fragment.find_first_of(" \t\r\n", value_begin);
    const auto encoded = fragment.substr(
        value_begin, value_end == std::string::npos ? std::string::npos
                                                    : value_end - value_begin);
    gsize decoded_size = 0U;
    guchar* decoded = g_base64_decode(encoded.c_str(), &decoded_size);
    require(decoded != nullptr && decoded_size != 0U,
            "failed to decode direct TVM lane contract");
    result.push_back(nlohmann::json::parse(
        reinterpret_cast<const char*>(decoded),
        reinterpret_cast<const char*>(decoded) + decoded_size));
    g_free(decoded);
    cursor = value_end == std::string::npos ? fragment.size() : value_end;
  }
  return result;
}

} // namespace

RUN_TEST("unit_modelpack_multi_mla_plan_test", ([] {
           namespace fs = std::filesystem;
           using simaai::neat::internal::ExecutionStageKind;
           using simaai::neat::internal::ModelPack;
           using simaai::neat::internal::ModelStage;
           using simaai::neat::internal::PipelineType;
           using simaai::neat::pipeline_internal::MemoryBackendPolicy;
           using simaai::neat::pipeline_internal::sima::FrameArenaRole;
           namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

           require(::setenv("SIMA_NEAT_MEMORY_BACKEND", "dmabuf-plan", 1) == 0,
                   "failed to select strict DMA-BUF backend");
           const auto root = fs::temp_directory_path() / "neat-modelpack-two-mla-unit";
           std::error_code ec;
           fs::remove_all(root, ec);
           fs::create_directories(root / "etc", ec);
           fs::create_directories(root / "lib", ec);
           fs::create_directories(root / "share", ec);
           require(!ec, "failed to create synthetic package layout");
           write_manifest(root / "etc" / "modelpack_two_mla_mpk.json");
           write_monolithic_topology_elf(root / "share" / "encoder.so");
           write_monolithic_topology_elf(root / "share" / "decoder.elf");

           ModelPack model(root.string());
           require(model.memory_backend_decision().backend == MemoryBackendPolicy::DmaBufPlan &&
                       model.memory_backend_decision().admission.eligible(),
                   "synthetic package did not pass strict multi-stage admission");
           require(model.mpk_contract().has_value(), "synthetic package lost its MPK contract");
           const auto& mpk = *model.mpk_contract();
           const auto* first_mla =
               simaai::neat::pipeline_internal::sima::get_first_mla_stage_io_contract(mpk);
           const auto* last_mla =
               simaai::neat::pipeline_internal::sima::get_last_mla_stage_io_contract(mpk);
           require(first_mla && first_mla->name == "MLA_encoder" && last_mla &&
                       last_mla->name == "MLA_decoder" &&
                       simaai::neat::pipeline_internal::sima::get_mla_stage_io_contract(mpk) ==
                           nullptr,
                   "MLA boundary queries must be explicit and the singular query fail ambiguous");
           const auto& route_graph = model.route_graph();
           require(route_graph.mla_plugin_index >= 0 && route_graph.last_mla_plugin_index >= 0 &&
                       route_graph.mla_plugin_index != route_graph.last_mla_plugin_index,
                   "route graph must retain distinct first and terminal MLA boundaries");
           const auto plan = model.execution_plan();
           require(plan.pre.empty() && plan.post.empty() && plan.infer.size() == 2U,
                   "multi-MLA execution plan must contain exactly two inference stages");
           require(plan.infer[0].kind == ExecutionStageKind::Mla &&
                       plan.infer[0].stage_name == "MLA_encoder" &&
                       plan.infer[1].kind == ExecutionStageKind::Mla &&
                       plan.infer[1].stage_name == "MLA_decoder",
                   "ModelPack lost compiler-authored MLA stage order or identity");

           const auto facts = model.stage_facts_for_model_stage(ModelStage::MlaOnly);
           require(facts.size() == 2U && facts[0].mla_compiled.has_value() &&
                       facts[1].mla_compiled.has_value(),
                   "ModelPack did not render one typed ProcessMLA contract per MLA stage");
           const auto& first = facts[0].mla_compiled->runtime_contract;
           const auto& second = facts[1].mla_compiled->runtime_contract;
           require(first.frame_arena_role == FrameArenaRole::Allocate &&
                       second.frame_arena_role == FrameArenaRole::Allocate &&
                       second.frame_arena_storage_domain == sc::ArenaStorageDomain::Dms &&
                       second.frame_arena_escape_policy ==
                           sc::ArenaEscapePolicy::CpuMappablePublic,
                   "only the terminal public MLA must own a separate CPU-visible output pool");
           require(first.frame_arena_size_bytes != 0U &&
                       second.frame_arena_size_bytes != 0U,
                   "both MLA stages must retain a non-empty output allocation contract");
           require(first.physical_outputs.size() == 1U && second.physical_inputs.size() == 1U &&
                       second.physical_inputs[0].source_physical_index == 0 &&
                       second.physical_inputs[0].source_byte_offset == 0 &&
                       second.physical_outputs[0].source_byte_offset == 0,
                   "the later MLA must import its exact preceding TensorBuffer carrier without "
                   "copying it into the terminal output pool");
           require(facts[0].mla_compiled->payload.model_path.ends_with("/share/encoder.so") &&
                       facts[1].mla_compiled->payload.model_path.ends_with("/share/decoder.elf"),
                   "an MLA .so/ELF must stay bound to its exact MPK stage identity");

           const auto make_mla_a65_package = [&](const std::string& label,
                                                  const std::uint64_t rows) {
             const fs::path package = root.parent_path() / label;
             fs::remove_all(package, ec);
             fs::create_directories(package / "etc", ec);
             fs::create_directories(package / "lib", ec);
             fs::create_directories(package / "share", ec);
             require(!ec, "failed to create synthetic MLA-to-A65 package layout");
             write_mla_a65_manifest(package / "etc" / "model_mpk.json", rows);
             write_topology_elf(package / "share" / "model.elf",
                                "data.ifm.persistent.afe_direct_input_0.b0", 16U,
                                "data.ofm.persistent.afe_mla_output_0.b0", rows * 4912U);
             write_host_module(package / "lib" / "host.so", rows);
             return package;
           };

           const fs::path singleton_package =
               make_mla_a65_package("neat-modelpack-mla-a65-singleton-pitch-unit", 1U);
           ModelPack singleton_model(singleton_package.string());
           const auto singleton_facts =
               singleton_model.stage_facts_for_model_stage(ModelStage::MlaOnly);
           require(singleton_facts.size() == 2U &&
                       singleton_facts[0].mla_compiled.has_value() &&
                       singleton_facts[1].transport_compiled.has_value() &&
                       !singleton_facts[0].mla_compiled->runtime_contract.physical_outputs.empty() &&
                       !singleton_facts[1].transport_compiled->runtime_contract.logical_inputs
                            .empty() &&
                       singleton_facts[0].mla_compiled->runtime_contract.physical_outputs[0]
                               .size_bytes == 4912U &&
                       singleton_facts[1].transport_compiled->runtime_contract.logical_inputs[0]
                               .shape == std::vector<std::int64_t>({1, 1225}),
                   "[1,1225] QMLA pitch is dense-address-equivalent and must remain valid for A65");

           simaai::neat::internal::InferenceTerminalPolicy false_boxdecode_policy;
           false_boxdecode_policy.mla_only = true;
           ModelPack false_boxdecode_model(
               singleton_package.string(), "application/vnd.simaai.tensor", "FP32", 1,
               1, 1, 1, false, {}, {}, {}, PipelineType::Preproc, "decoder", 4,
               4, 0, -1, {}, {}, false_boxdecode_policy,
               /*cleanup_extracted_model_data=*/false);
           simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags false_flags;
           false_flags.boxdecode_selected = true;
           bool false_boxdecode_rejected = false;
           try {
             false_boxdecode_model.set_model_managed_stage_facts(
                 std::nullopt, false_flags, {ExecutionStageKind::BoxDecode});
           } catch (const std::runtime_error& failure) {
             false_boxdecode_rejected =
                 std::string(failure.what()).find(
                     "non-CVU materializing successor") != std::string::npos;
           }
           require(false_boxdecode_rejected,
                   "BoxDecode route selection must not absorb an MLA-to-A65 materializer");

           const auto make_cast_tail_package =
               [&](const std::string& label, const std::string& mla_scalar,
                   const std::string& cast_output_scalar,
                   const std::uint64_t input_elements,
                   const std::uint64_t output_elements) {
                 const auto scalar_bytes = [](const std::string& scalar) {
                   if (scalar == "float32") return std::uint64_t{4U};
                   if (scalar == "bfloat16" || scalar == "float16") {
                     return std::uint64_t{2U};
                   }
                   return std::uint64_t{1U};
                 };
                 const fs::path package = root.parent_path() / label;
                 fs::remove_all(package, ec);
                 fs::create_directories(package / "etc", ec);
                 fs::create_directories(package / "lib", ec);
                 fs::create_directories(package / "share", ec);
                 require(!ec,
                         "failed to create synthetic MLA-to-Cast package layout");
                 write_boxdecode_cast_tail_manifest(
                     package / "etc" / "model_mpk.json", mla_scalar,
                     cast_output_scalar, input_elements, output_elements);
                 write_topology_elf(
                     package / "share" / "model.elf",
                     "data.ifm.persistent.afe_direct_input_0.b0",
                     input_elements * scalar_bytes(mla_scalar),
                     "data.ofm.persistent.afe_mla_output_0.b0",
                     input_elements * scalar_bytes(mla_scalar));
                 return package;
               };
           const auto select_boxdecode_route =
               [&](const fs::path& package) {
                 simaai::neat::internal::InferenceTerminalPolicy policy;
                 policy.mla_only = true;
                 ModelPack candidate(
                     package.string(), "application/vnd.simaai.tensor", "BF16",
                     1, 32, 1, 1, false, {}, {}, {}, PipelineType::Preproc,
                     "decoder", 4, 4, 0, -1, {}, {}, policy,
                     /*cleanup_extracted_model_data=*/false);
                 simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags
                     route_flags;
                 route_flags.boxdecode_selected = true;
                 candidate.set_model_managed_stage_facts(
                     std::nullopt, route_flags,
                     {ExecutionStageKind::BoxDecode});
                 return candidate;
               };

           const auto bf16_cast_package = make_cast_tail_package(
               "neat-modelpack-boxdecode-bf16-cast-tail-unit", "bfloat16",
               "float32", 32U, 32U);
           auto bf16_cast_model = select_boxdecode_route(bf16_cast_package);
           require(bf16_cast_model.mpk_contract().has_value(),
                   "BF16 Cast-tail fixture lost its semantic MPK contract");
           const auto& bf16_cast_mpk = *bf16_cast_model.mpk_contract();
           const auto pre_mla_edges = std::count_if(
               bf16_cast_mpk.edges.begin(), bf16_cast_mpk.edges.end(),
               [](const auto& edge) {
                 return edge.src_plugin == "pre_cast" &&
                        edge.src_output_index == 0 &&
                        edge.dst_plugin == "MLA_0" &&
                        edge.dst_input_index == 0 &&
                        edge.tensor_name == "mla_input";
               });
           require(pre_mla_edges == 1,
                   "BF16 Cast-tail fixture lost its exact pre-Cast output-to-MLA "
                   "input edge");
           const auto bf16_cast_plan = bf16_cast_model.execution_plan();
           require(bf16_cast_plan.infer.size() == 2U &&
                       bf16_cast_plan.infer[0].kind ==
                           ExecutionStageKind::Cast &&
                       bf16_cast_plan.infer[1].kind ==
                           ExecutionStageKind::Mla,
                   "BoxDecode must absorb an exact BF16-to-FP32 Cast tail");
           const auto bf16_cast_facts =
               bf16_cast_model.stage_facts_for_model_stage(ModelStage::Full);
           require(bf16_cast_facts.size() == 2U &&
                       bf16_cast_facts[0].processcvu_contract.has_value() &&
                       bf16_cast_facts[1].mla_compiled.has_value(),
                   "absorbed Cast route did not retain pre-Cast followed by MLA");
           const auto& pre_runtime =
               bf16_cast_facts[0].processcvu_contract->runtime_contract;
           const auto& mla_runtime =
               bf16_cast_facts[1].mla_compiled->runtime_contract;
           require(pre_runtime.frame_arena_storage_domain ==
                       sc::ArenaStorageDomain::Cma,
                   "pre-Cast retained carrier is not CMA (actual_domain=" +
                       std::to_string(static_cast<int>(
                           pre_runtime.frame_arena_storage_domain)) +
                       ")");
           require(pre_runtime.physical_outputs.size() == 1U &&
                       mla_runtime.physical_inputs.size() == 1U &&
                       mla_runtime.input_bindings.size() == 1U,
                   "pre-Cast-to-MLA runtime edge cardinality changed (pre_outputs=" +
                       std::to_string(pre_runtime.physical_outputs.size()) +
                       ",mla_inputs=" +
                       std::to_string(mla_runtime.physical_inputs.size()) +
                       ",mla_bindings=" +
                       std::to_string(mla_runtime.input_bindings.size()) + ")");
           const auto& pre_physical = pre_runtime.physical_outputs[0];
           const auto& mla_physical = mla_runtime.physical_inputs[0];
           const auto& mla_binding = mla_runtime.input_bindings[0];
           require(mla_physical.source_physical_index ==
                           pre_physical.physical_index &&
                       mla_binding.src_physical_output_index ==
                           pre_physical.physical_index,
                   "pre-Cast-to-MLA physical source index changed (pre=" +
                       std::to_string(pre_physical.physical_index) +
                       ",input_source=" +
                       std::to_string(mla_physical.source_physical_index) +
                       ",binding_source=" +
                       std::to_string(mla_binding.src_physical_output_index) + ")");
           require(mla_physical.source_byte_offset ==
                           pre_physical.source_byte_offset &&
                       mla_binding.src_physical_byte_offset ==
                           pre_physical.source_byte_offset,
                   "pre-Cast-to-MLA absolute carrier offset changed (pre=" +
                       std::to_string(pre_physical.source_byte_offset) +
                       ",input=" +
                       std::to_string(mla_physical.source_byte_offset) +
                       ",binding=" +
                       std::to_string(mla_binding.src_physical_byte_offset) + ")");
           require(mla_physical.size_bytes == pre_physical.size_bytes &&
                       mla_binding.src_physical_size_bytes ==
                           pre_physical.size_bytes,
                   "pre-Cast-to-MLA carrier span changed (pre=" +
                       std::to_string(pre_physical.size_bytes) +
                       ",input=" + std::to_string(mla_physical.size_bytes) +
                       ",binding=" +
                       std::to_string(mla_binding.src_physical_size_bytes) + ")");
           require(mla_runtime.logical_outputs.size() == 1U,
                   "absorbed Cast route changed raw MLA publication cardinality");
           require(mla_runtime.logical_outputs[0].dtype == "bfloat16",
                   "absorbed Cast route changed raw MLA dtype (actual=" +
                       mla_runtime.logical_outputs[0].dtype + ")");
           require(mla_runtime.logical_outputs[0].size_bytes == 64U,
                   "absorbed Cast route changed raw MLA bytes (actual=" +
                       std::to_string(
                           mla_runtime.logical_outputs[0].size_bytes) +
                       ")");

           const auto expect_cast_tail_rejected =
               [&](const fs::path& package,
                   const std::initializer_list<std::string_view> expected) {
                 bool rejected = false;
                 try {
                   (void)select_boxdecode_route(package);
                 } catch (const std::runtime_error& failure) {
                   const std::string detail = failure.what();
                   rejected = std::any_of(
                       expected.begin(), expected.end(),
                       [&](const auto token) {
                         return detail.find(token) != std::string::npos;
                       });
                 }
                 require(rejected,
                         "malformed BoxDecode Cast tail was not rejected by its exact "
                         "typed predicate");
               };
           expect_cast_tail_rejected(
               make_cast_tail_package(
                   "neat-modelpack-boxdecode-cast-wrong-direction-unit",
                   "float32", "bfloat16", 32U, 32U),
               {"BF16-to-FP32"});
           expect_cast_tail_rejected(
               make_cast_tail_package(
                   "neat-modelpack-boxdecode-cast-shape-mismatch-unit",
                   "bfloat16", "float32", 32U, 16U),
               {"changes semantic shape",
                "shape-preserving transform 'cast_head' has contradictory exact "
                "endpoint shapes"});

           bool wrong_dtype_rejected = false;
           std::string wrong_dtype_detail;
           try {
             (void)select_boxdecode_route(make_cast_tail_package(
                 "neat-modelpack-boxdecode-cast-wrong-dtype-unit", "int8",
                 "float32", 32U, 32U));
           } catch (const std::runtime_error& failure) {
             const std::string detail = failure.what();
             wrong_dtype_detail = detail;
             wrong_dtype_rejected =
                 detail.find("legacy cast has no exact registered transition for this "
                             "dtype") != std::string::npos;
           }
           require(wrong_dtype_rejected,
                   "non-BF16 Cast input must fail closed before BoxDecode route "
                   "selection (actual=" + wrong_dtype_detail + ")");

           const fs::path multirow_package =
               make_mla_a65_package("neat-modelpack-mla-a65-multirow-pitch-unit", 2U);
           bool multirow_rejected = false;
           try {
             ModelPack multirow_model(multirow_package.string());
             (void)multirow_model.stage_facts_for_model_stage(ModelStage::MlaOnly);
           } catch (const std::runtime_error& failure) {
             multirow_rejected =
                 std::string(failure.what()).find(
                     "A65 input has no exact typed storage binding") != std::string::npos;
           }
           require(multirow_rejected,
                   "[2,1225] QMLA row pitch must fail closed before A65 drops its strides");

           // A CPU epoch follows the deterministic rendered schedule, not the
           // physical DAG's degree. APU_2 joins the retained MLA root with the
           // immediately preceding APU_1 output, so it has two physical
           // predecessors but remains in the same Core-owned CMA CPU interval.
           // The following EV74 Cast is a real device interruption and must
           // force the second A65 command to end that interval.
           const fs::path join_package =
               root.parent_path() / "neat-modelpack-a65-join-device-boundary-unit";
           fs::remove_all(join_package, ec);
           fs::create_directories(join_package / "etc", ec);
           fs::create_directories(join_package / "lib", ec);
           fs::create_directories(join_package / "share", ec);
           require(!ec, "failed to create synthetic A65 join package layout");
           write_a65_join_device_boundary_manifest(join_package / "etc" / "model_mpk.json");
           write_topology_elf(
               join_package / "share" / "first.elf",
               "data.ifm.persistent.afe_direct_input_0.b0", 16U,
               "data.ofm.persistent.afe_mla_output_0.b0", 4912U);
           write_topology_elf(
               join_package / "share" / "last.elf",
               "data.ifm.persistent.afe_direct_input_0.b0", 2450U,
               "data.ofm.persistent.afe_mla_output_0.b0", 2464U);
           write_host_module(join_package / "lib" / "host_one.so", 1U,
                             {"root_input"});
           write_host_module(join_package / "lib" / "host_join.so", 1U,
                             {"root_input", "chain_input"});
           ModelPack join_model(join_package.string());
           const auto join_plan = join_model.execution_plan();
           require(join_plan.infer.size() == 5U &&
                       join_plan.infer[0].kind == ExecutionStageKind::Mla &&
                       join_plan.infer[1].kind == ExecutionStageKind::HostTvm &&
                       join_plan.infer[2].kind == ExecutionStageKind::HostTvm &&
                       join_plan.infer[3].kind == ExecutionStageKind::Cast &&
                       join_plan.infer[4].kind == ExecutionStageKind::Mla,
                   "synthetic A65 join lost its rendered CPU/device order");
           const auto join_contracts =
               direct_tvm_contracts(join_model.backend_fragment(ModelStage::MlaOnly));
           require(join_contracts.size() == 2U &&
                       join_contracts[0].at("storage_domain") == "cma" &&
                       join_contracts[1].at("storage_domain") == "cma" &&
                       join_contracts[0].at("allocation_provenance") ==
                           "core_allocated" &&
                       join_contracts[1].at("allocation_provenance") ==
                           "core_allocated" &&
                       join_contracts[0].at("cpu_epoch_start").get<bool>() &&
                       !join_contracts[0].at("cpu_epoch_end").get<bool>() &&
                       !join_contracts[1].at("cpu_epoch_start").get<bool>() &&
                       join_contracts[1].at("cpu_epoch_end").get<bool>() &&
                       join_contracts[1].at("inputs").size() == 2U,
                   "A65 DAG join split one rendered CMA CPU epoch or crossed an EV74 "
                   "device interruption");
           fs::remove_all(join_package, ec);
           fs::remove_all(singleton_package, ec);
           fs::remove_all(multirow_package, ec);

           // ModelPack extraction must not canonicalize opaque compiler JSON. The strict
           // publication ledger binds the compiler-authored MPK byte stream, so a no-op path
           // scan must retain both its bytes and therefore its SHA-256. A JSON which actually
           // carries a model-relative path still follows the established absolute-path rewrite.
           constexpr std::string_view no_op_json =
               "{\n"
               "    \"zeta\": [ 3, 2, 1 ],\n"
               "    \"alpha\": { \"opaque\": true }\n"
               "}\n";
           constexpr std::string_view relative_path_json =
               "{ \"model_info\" : { \"path\" : \"host_module.so\" }, "
               "\"keep\" : [ 2, 1 ] }\n";
           const fs::path archive_source =
               root.parent_path() / "neat-modelpack-json-preservation-unit";
           fs::remove_all(archive_source, ec);
           fs::create_directories(archive_source / "etc", ec);
           fs::create_directories(archive_source / "lib", ec);
           fs::create_directories(archive_source / "share", ec);
           require(!ec, "failed to create synthetic archive package layout");
           write_archive_manifest(archive_source / "etc" /
                                  "modelpack_json_preservation_mpk.json");
           write_monolithic_topology_elf(archive_source / "share" / "model.elf");
           write_text(archive_source / "etc" / "opaque_publication.json", no_op_json);
           write_text(archive_source / "etc" / "relative_model_path.json", relative_path_json);
           const std::string source_mpk = read_text(
               archive_source / "etc" / "modelpack_json_preservation_mpk.json");
           const std::string source_mpk_sha256 = sha256(source_mpk);
           require(sha256(no_op_json) ==
                       "f295c1cf18c3bfbb1f16dd581badec2393f6a85a5598b178cd4535c9f7e1e60e",
                   "no-op JSON fixture SHA-256 changed unexpectedly");

           const fs::path archive = root.parent_path() / "neat-modelpack-json-preservation.tar.gz";
           const std::string archive_command =
               "tar -czf " + shell_quote(archive.string()) + " -C " +
               shell_quote(archive_source.string()) + " .";
           require(std::system(archive_command.c_str()) == 0,
                   "failed to create synthetic ModelPack archive");
           {
             ModelPack archived_model(archive.string());
             const fs::path extracted_etc(archived_model.etc_dir());
             const std::string extracted_mpk = read_text(
                 extracted_etc / "modelpack_json_preservation_mpk.json");
             require(extracted_mpk == source_mpk && sha256(extracted_mpk) == source_mpk_sha256,
                     "a no-op ModelPack MPK scan changed its exact bytes or publication digest");
             const std::string preserved = read_text(extracted_etc / "opaque_publication.json");
             require(preserved == no_op_json,
                     "a no-op ModelPack JSON scan changed compiler-authored bytes");
             require(sha256(preserved) ==
                         "f295c1cf18c3bfbb1f16dd581badec2393f6a85a5598b178cd4535c9f7e1e60e",
                     "a no-op ModelPack JSON scan changed the publication digest");

             const fs::path rewritten_path = extracted_etc / "relative_model_path.json";
             const std::string rewritten = read_text(rewritten_path);
             require(rewritten != relative_path_json,
                     "a model-relative path was not materialized during archive extraction");
             const auto rewritten_json = nlohmann::json::parse(rewritten);
             require(rewritten_json.at("model_info").at("path").get<std::string>() ==
                         (extracted_etc.parent_path() / "lib" / "host_module.so").string(),
                     "archive path rewrite did not anchor the model under extracted lib/");
             require(rewritten_json.at("keep") == nlohmann::json::array({2, 1}),
                     "archive path rewrite changed unrelated JSON content");
           }
           fs::remove(archive, ec);
           fs::remove_all(archive_source, ec);
           fs::remove_all(root, ec);

           // Optional exact AFE 2.1 package gate for qualification jobs. The
           // fixture remains external because it is a compiler artifact, not
           // test source. It covers RF-DETR's real branch/join route:
           // MLA -> A65 -> A65 -> A65 -> EV74 Cast -> 161-IFM MLA.
           if (const char* real_package = std::getenv("SIMANEAT_AFE21_MULTI_MLA_PACKAGE");
               real_package != nullptr && *real_package != '\0') {
             simaai::neat::Model::Options public_options;
             public_options.preprocess.kind = simaai::neat::InputKind::Tensor;
             public_options.preprocess.enable = simaai::neat::AutoFlag::Off;
             public_options.cleanup_extracted_model_data = false;
             simaai::neat::Model public_model(real_package, public_options);
             require(public_model.resolved_preprocess_plan().warnings.empty(),
                     "RF-DETR route diagnostics were misclassified as user-facing planner "
                     "warnings");

             ModelPack real_model(
                 real_package, "application/vnd.simaai.tensor", "FP32", /*depth=*/1,
                 /*max_width=*/640, /*max_height=*/480, /*max_depth=*/1,
                 /*normalize=*/false, {}, {}, /*preproc_next_cpu=*/{}, PipelineType::QuantTess);
             require(real_model.memory_backend_decision().admission.eligible(),
                     "exact AFE 2.1 multi-MLA/A65 package failed ModelPack admission");
             const auto real_plan = real_model.execution_plan();
             require(real_plan.pre.empty() && real_plan.post.empty() &&
                         !real_plan.infer.empty(),
                     "exact AFE 2.1 package was not rendered as one model-owned schedule");
             std::vector<std::size_t> mla_indices;
             std::vector<std::size_t> host_indices;
             std::vector<std::size_t> cast_indices;
             for (std::size_t index = 0; index < real_plan.infer.size(); ++index) {
               const auto& stage = real_plan.infer[index];
               require(!stage.physical_command_ids.empty(),
                       "exact AFE 2.1 package lost physical command identity");
               switch (stage.kind) {
               case ExecutionStageKind::Mla:
                 mla_indices.push_back(index);
                 break;
               case ExecutionStageKind::HostTvm:
                 host_indices.push_back(index);
                 break;
               case ExecutionStageKind::Cast:
                 cast_indices.push_back(index);
                 break;
               default:
                 require(false,
                         "exact RF-DETR schedule contains work outside MLA, direct A65, or "
                         "registered CVU Cast commands");
               }
             }
             require(mla_indices.size() == 2U && host_indices.size() == 3U &&
                         !cast_indices.empty() &&
                         real_plan.infer[mla_indices[0]].stage_name == "MLA_0" &&
                         real_plan.infer[mla_indices[1]].stage_name == "MLA_171" &&
                         real_plan.infer[host_indices[0]].stage_name == "APU_3" &&
                         real_plan.infer[host_indices[1]].stage_name == "APU_5" &&
                         real_plan.infer[host_indices[2]].stage_name == "APU_10" &&
                         mla_indices[0] < host_indices[0] &&
                         host_indices[0] < host_indices[1] &&
                         host_indices[1] < host_indices[2] &&
                         host_indices[2] < mla_indices[1],
                     "exact RF-DETR schedule lost its two MLA boundaries or three ordered "
                     "direct A65 commands");

             real_model.set_model_managed_stage_facts(
                 /*processcvu_preproc_single_output_handoff=*/true, std::nullopt, {});
             const auto real_facts =
                 real_model.stage_facts_for_model_stage(ModelStage::MlaOnly);
             require(real_facts.size() == real_plan.infer.size(),
                     "exact RF-DETR route lost a typed stage fact");
             for (std::size_t index = 0; index < real_facts.size(); ++index) {
               const auto& facts = real_facts[index];
               switch (real_plan.infer[index].kind) {
               case ExecutionStageKind::Mla:
                 require(facts.mla_compiled.has_value(),
                         "exact RF-DETR MLA command lost its typed contract");
                 break;
               case ExecutionStageKind::HostTvm:
                 require(facts.transport_compiled.has_value(),
                         "exact RF-DETR A65 command lost its direct TVM contract");
                 break;
               case ExecutionStageKind::Cast:
                 require(facts.processcvu_contract.has_value(),
                         "exact RF-DETR CVU Cast command lost its typed contract");
                 break;
               default:
                 require(false, "unreachable RF-DETR execution-stage kind");
               }
             }
             require(real_facts[mla_indices[0]].mla_compiled.has_value() &&
                         real_facts[mla_indices[1]].mla_compiled.has_value(),
                     "exact AFE 2.1 route did not retain MLA/A65/CVU typed stage contracts");

             const auto& initial_mla =
                 real_facts[mla_indices[0]].mla_compiled->runtime_contract;
             const auto align16 = [](const std::uint64_t value) {
               return (value + 15U) & ~std::uint64_t{15U};
             };
             require(initial_mla.physical_outputs.size() > 1U &&
                         initial_mla.logical_outputs.size() > 1U,
                     "RF-DETR initial MLA lost its second physical/logical OFM");
             const auto& initial_feature = initial_mla.logical_outputs[1];
             require(initial_feature.size_bytes % 1225U == 0U,
                     "RF-DETR initial feature has a non-integral scalar byte extent");
             const auto initial_scalar_bytes = initial_feature.size_bytes / 1225U;
             require(initial_feature.shape == std::vector<std::int64_t>({1, 1225}) &&
                         (initial_scalar_bytes == 2U || initial_scalar_bytes == 4U) &&
                         initial_mla.physical_outputs[1].size_bytes ==
                             align16(initial_feature.size_bytes),
                     "RF-DETR initial MLA lost exact OFM1 physical/logical extents");

             const auto& terminal_mla =
                 real_facts[mla_indices[1]].mla_compiled->runtime_contract;
             require(terminal_mla.frame_arena_role == FrameArenaRole::Allocate &&
                         terminal_mla.frame_arena_storage_domain ==
                             sc::ArenaStorageDomain::Dms &&
                         terminal_mla.frame_arena_escape_policy ==
                             sc::ArenaEscapePolicy::CpuMappablePublic &&
                         (terminal_mla.frame_arena_required_device_access &
                          static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Mla)) != 0U &&
                         (terminal_mla.frame_arena_required_device_access &
                          static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65)) != 0U &&
                         (terminal_mla.frame_arena_required_device_access &
                          static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74)) == 0U &&
                         terminal_mla.physical_inputs.size() == 161U,
                     "RF-DETR terminal MLA must import its exact 161 IFMs while owning a "
                     "distinct CPU-visible DMS output carrier");
             require(terminal_mla.physical_outputs.size() > 1U &&
                         terminal_mla.logical_outputs.size() > 1U,
                     "RF-DETR terminal MLA lost one of its two physical/logical OFMs");
             // Exact package 9d2032... is the all-BF16 artifact. Its QMLA
             // SHT_DATA extents are 4,800 and 57,600 bytes; the second port is
             // the BF16 [300,91] tensor with a 192-byte physical row pitch.
             // 118,784 bytes / 110,400-byte OFM1 belongs to the stale FP32
             // synthetic expectation and must not override package authority.
             require(terminal_mla.frame_arena_size_bytes == 69632U &&
                         terminal_mla.physical_outputs[0].size_bytes == 4800U &&
                         terminal_mla.physical_outputs[0].source_byte_offset == 0 &&
                         terminal_mla.physical_outputs[1].size_bytes == 57600U &&
                         terminal_mla.physical_outputs[1].source_byte_offset == 8192,
                     "RF-DETR all-BF16 terminal DMS arena changed its exact QMLA layout");
             const auto& terminal_boxes = terminal_mla.logical_outputs[0];
             const auto& terminal_logits = terminal_mla.logical_outputs[1];
             require(terminal_logits.size_bytes % (300U * 91U) == 0U,
                     "RF-DETR terminal logits have a non-integral scalar byte extent");
             const auto terminal_scalar_bytes = terminal_logits.size_bytes / (300U * 91U);
             const auto terminal_row_pitch = align16(91U * terminal_scalar_bytes);
             require((terminal_scalar_bytes == 2U || terminal_scalar_bytes == 4U) &&
                         terminal_boxes.shape == std::vector<std::int64_t>({1, 300, 4}) &&
                         terminal_boxes.size_bytes == 300U * 4U * terminal_scalar_bytes &&
                         terminal_logits.shape == std::vector<std::int64_t>({1, 300, 91}) &&
                         terminal_mla.physical_outputs[1].size_bytes ==
                             300U * terminal_row_pitch &&
                         terminal_logits.stride_bytes ==
                             std::vector<std::int64_t>(
                                 {static_cast<std::int64_t>(300U * terminal_row_pitch),
                                  static_cast<std::int64_t>(terminal_row_pitch),
                                  static_cast<std::int64_t>(terminal_scalar_bytes)}),
                     "RF-DETR terminal logits lost exact physical carrier/logical row pitch");
             // MLA_171 is a DAG join, not a linear handoff. Its first IFM is
             // produced by cast_3 while the other 160 are still-live MLA_0
             // outputs retained in the same frame arena. The immediately
             // preceding rendered stage therefore cannot own the complete
             // input catalogue. Reconcile every terminal port against the
             // unique typed publication with the same decoded value identity,
             // parent-relative arena range and physical extent across all
             // prior stages instead.
             struct PriorPhysicalPublication {
               std::size_t stage_index = 0U;
               std::size_t logical_index = 0U;
               const simaai::neat::pipeline_internal::sima::LogicalTensorStaticSpec* logical =
                   nullptr;
               const simaai::neat::pipeline_internal::sima::PhysicalBufferStaticSpec* physical =
                   nullptr;
             };
             std::vector<PriorPhysicalPublication> prior_publications;
             for (std::size_t prior_index = 0U; prior_index < mla_indices[1]; ++prior_index) {
               const auto& prior = real_facts[prior_index];
               const simaai::neat::CompiledRuntimeContract* runtime = nullptr;
               if (prior.processcvu_contract.has_value()) {
                 runtime = &prior.processcvu_contract->runtime_contract;
               } else if (prior.transport_compiled.has_value()) {
                 runtime = &prior.transport_compiled->runtime_contract;
               } else if (prior.mla_compiled.has_value()) {
                 runtime = &prior.mla_compiled->runtime_contract;
               }
               if (runtime == nullptr) {
                 continue;
               }
               for (std::size_t logical_index = 0U;
                    logical_index < runtime->logical_outputs.size(); ++logical_index) {
                 const auto& logical = runtime->logical_outputs[logical_index];
                 const auto physical = std::find_if(
                     runtime->physical_outputs.begin(), runtime->physical_outputs.end(),
                     [&](const auto& candidate) {
                       return candidate.physical_index == logical.physical_index;
                     });
                 if (physical != runtime->physical_outputs.end() &&
                     !logical.backend_name.empty()) {
                   prior_publications.push_back(PriorPhysicalPublication{
                       prior_index, logical_index, &logical, &*physical});
                 }
               }
             }
             require(terminal_mla.input_bindings.size() ==
                         terminal_mla.physical_inputs.size(),
                     "RF-DETR terminal MLA lost one of its exact 161 input bindings");
             std::vector<bool> publication_used(prior_publications.size(), false);
             std::vector<bool> producer_stage_used(mla_indices[1], false);
             bool saw_nonzero_upstream_parent_offset = false;
             for (std::size_t index = 0; index < terminal_mla.physical_inputs.size(); ++index) {
               const auto& input = terminal_mla.physical_inputs[index];
               const auto& binding = terminal_mla.input_bindings[index];
               const auto publication = std::find_if(
                   prior_publications.begin(), prior_publications.end(),
                   [&](const auto& candidate) {
                     const auto publication_index = static_cast<std::size_t>(
                         &candidate - prior_publications.data());
                     return !publication_used[publication_index] && candidate.logical != nullptr &&
                            candidate.physical != nullptr &&
                            candidate.logical->backend_name == input.segment_name &&
                            candidate.physical->size_bytes == input.size_bytes &&
                            candidate.physical->source_byte_offset == input.source_byte_offset &&
                            candidate.physical->required_alignment_bytes ==
                                input.required_alignment_bytes;
                   });
               require(publication != prior_publications.end(),
                       "RF-DETR terminal MLA IFM has no unique typed producer publication");
               const auto publication_index = static_cast<std::size_t>(
                   &*publication - prior_publications.data());
               publication_used[publication_index] = true;
               producer_stage_used[publication->stage_index] = true;
               require(input.physical_index == static_cast<int>(index) &&
                           input.source_physical_index == 0 &&
                           input.source_byte_offset >= 0 &&
                           binding.local_logical_input_index == static_cast<int>(index) &&
                           binding.src_physical_output_index == 0 &&
                           binding.src_physical_size_bytes == input.size_bytes &&
                           binding.src_physical_byte_offset == input.source_byte_offset &&
                           binding.cm_input_name == input.segment_name &&
                           binding.source_segment_name == input.segment_name,
                       "RF-DETR terminal MLA IFM lost its exact upstream shared-parent "
                       "region");
               saw_nonzero_upstream_parent_offset |= input.source_byte_offset != 0;
             }
             require(std::count(producer_stage_used.begin(), producer_stage_used.end(), true) >
                         1,
                     "RF-DETR terminal MLA DAG join collapsed to one producer stage");
             require(saw_nonzero_upstream_parent_offset,
                     "RF-DETR split-carrier proof must include nonzero offsets in the shared "
                     "upstream parent");

             const std::string fragment = real_model.backend_fragment(ModelStage::MlaOnly);
             const auto occurrences = [&](const std::string& token) {
               std::size_t count = 0U;
               for (std::size_t pos = 0U; (pos = fragment.find(token, pos)) != std::string::npos;
                    pos += token.size()) {
                 ++count;
               }
               return count;
             };
             require(occurrences("neatprocessmla ") == mla_indices.size() &&
                         occurrences("neatprocesscvu ") == cast_indices.size() &&
                         occurrences("neatprocesstvm ") == host_indices.size() &&
                         occurrences("direct-contract-b64=") == host_indices.size() &&
                         occurrences("stage-id=") == mla_indices.size() + cast_indices.size(),
                     "RF-DETR fragment did not render its exact MLA/CVU stages and three "
                     "self-contained direct A65 commands");
             require(occurrences("model-path=") == 0U &&
                         occurrences("batch-size=") == 0U &&
                         occurrences("batch-sz-model=") == 0U,
                     "strict DMA-BUF MLA fragment leaked deprecated property-owned runtime "
                     "configuration instead of using its typed manifest");
             const auto a65_contracts = direct_tvm_contracts(fragment);
             require(a65_contracts.size() == 3U,
                     "RF-DETR fragment lost one prepared direct A65 contract");
             for (std::size_t index = 0; index < a65_contracts.size(); ++index) {
               const auto& contract = a65_contracts[index];
               require(contract.at("schema") == "sima.neat.direct-tvm-lane" &&
                           contract.at("version") == 2 &&
                           contract.at("storage_domain") == "cma" &&
                           contract.at("allocation_provenance") == "core_allocated" &&
                           (contract.at("required_device_access").get<std::uint32_t>() & 1U) !=
                               0U &&
                           contract.at("cpu_epoch_start").get<bool>() == (index == 0U) &&
                           contract.at("cpu_epoch_end").get<bool>() ==
                               (index + 1U == a65_contracts.size()),
                       "RF-DETR direct A65 commands lost one maximal shared CPU epoch or its "
                       "Core-authored arena placement");
             }
           }

           // Optional exact-artifact production-path gate. Qualification jobs
           // point this at the YOLOv8 package identified by 029ddb60.... It
           // exercises ModelPack's post-planner setter rather than the pure
           // projection helper: the admitted full graph226->MLA->graph227
           // snapshot must be replaced by graph226->MLA with the six affine
           // BoxDecode views as its immutable public boundary.
           if (const char* yolo_package =
                   std::getenv("SIMANEAT_EXACT_YOLOV8_029DDB60_PACKAGE");
               yolo_package != nullptr && *yolo_package != '\0') {
             simaai::neat::internal::InferenceTerminalPolicy terminal_policy;
             terminal_policy.mla_only = true;
             ModelPack yolo_model(
                 yolo_package, "application/vnd.simaai.tensor", "FP32", 3,
                 640, 640, 3, false, {}, {}, {}, PipelineType::QuantTess,
                 "decoder", 4, 4, 0, -1, {}, {}, terminal_policy,
                 /*cleanup_extracted_model_data=*/false);
             const auto full_digest = yolo_model.memory_backend_decision().plan_digest;
             const auto full_plan = yolo_model.execution_plan();
             require(std::any_of(full_plan.infer.begin(), full_plan.infer.end(),
                                 [](const auto& stage) {
                                   return stage.kind ==
                                          ExecutionStageKind::DetessDequant;
                                 }),
                     "exact YOLO full admission must retain graph227 before route selection");

             simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags flags;
             flags.boxdecode_selected = true;
             yolo_model.set_model_managed_stage_facts(
                 /*processcvu_preproc_single_output_handoff=*/true, flags,
                 {ExecutionStageKind::BoxDecode});

             const auto selected_plan = yolo_model.execution_plan();
             require(std::none_of(selected_plan.infer.begin(), selected_plan.infer.end(),
                                  [](const auto& stage) {
                                    return stage.kind ==
                                           ExecutionStageKind::DetessDequant;
                                  }),
                     "selected YOLO execution snapshot must not render graph227");
             const auto& selected_decision = yolo_model.memory_backend_decision();
             require(selected_decision.plan_digest != full_digest &&
                         selected_decision.admission.detail.find(
                             "terminal MLA-to-BoxDecode route snapshot") !=
                             std::string::npos,
                     "selected YOLO route must replace the full execution digest/provenance");

             const auto selected_facts =
                 yolo_model.stage_facts_for_model_stage(ModelStage::Full);
             const simaai::neat::CompiledRuntimeContract* graph226 = nullptr;
             const simaai::neat::CompiledRuntimeContract* terminal_mla = nullptr;
             bool saw_graph227 = false;
             for (const auto& fact : selected_facts) {
               if (fact.processcvu_contract.has_value()) {
                 const auto graph = fact.processcvu_contract->payload.graph_id;
                 if (graph == 226) graph226 = &fact.processcvu_contract->runtime_contract;
                 saw_graph227 |= graph == 227;
               }
               if (fact.mla_compiled.has_value()) {
                 terminal_mla = &fact.mla_compiled->runtime_contract;
               }
             }
             require(graph226 != nullptr && terminal_mla != nullptr && !saw_graph227,
                     "exact selected route must render graph226 and terminal MLA only");
             require(graph226->frame_arena_size_bytes == 1228800U &&
                         graph226->frame_arena_storage_domain ==
                             sc::ArenaStorageDomain::Cma &&
                         terminal_mla->frame_arena_size_bytes == 1212416U &&
                         terminal_mla->frame_arena_storage_domain ==
                             sc::ArenaStorageDomain::Dms &&
                         terminal_mla->frame_arena_role == FrameArenaRole::Allocate,
                     "exact selected route must retain 1,228,800-byte CMA ingress and own a "
                     "1,212,416-byte terminal DMS carrier");
             const std::array<std::int64_t, 6U> expected_offsets{
                 0, 409600, 512000, 537600, 1049600, 1177600};
             const std::array<std::uint64_t, 6U> expected_spans{
                 409600U, 102400U, 25600U, 512000U, 128000U, 32000U};
             require(terminal_mla->logical_outputs.size() == expected_offsets.size(),
                     "exact selected route must publish six ordered BoxDecode views");
             require(terminal_mla->physical_outputs.size() == 1U &&
                         terminal_mla->physical_outputs.front().size_bytes == 1209600U &&
                         terminal_mla->physical_outputs.front().source_byte_offset == 0 &&
                         terminal_mla->physical_outputs.front().segment_name == "MLA_0",
                     "exact selected route must retain one 1,209,600-byte MLA root output");
             for (std::size_t index = 0U; index < expected_offsets.size(); ++index) {
               const auto& logical = terminal_mla->logical_outputs[index];
               require(logical.byte_offset ==
                               expected_offsets[index] &&
                           logical.size_bytes == expected_spans[index] &&
                           logical.physical_index == 0 &&
                           logical.segment_name == "MLA_0" &&
                           logical.logical_name ==
                               "MLA_0_ofm_unpack_transform_" +
                                   std::to_string(index) &&
                           logical.logical_name.find("pass_through_out") ==
                               std::string::npos,
                       "exact selected route changed an ordered affine BoxDecode view");
             }
             const auto route_fact = std::find_if(
                 selected_decision.admission.proof.begin(),
                 selected_decision.admission.proof.end(), [](const auto& fact) {
                   return fact.subject == "route-publication-boundary";
                 });
             require(route_fact != selected_decision.admission.proof.end() &&
                         route_fact->evidence.find("logical_views=6") !=
                             std::string::npos &&
                         route_fact->evidence.find(selected_decision.plan_digest) !=
                             std::string::npos,
                     "exact selected route proof must bind six views to the selected digest");
           }

           // Optional six-real-OFM production-path gate.  Unlike the packed
           // YOLOv8s ELF above, this MLATess ELF exposes six independent MLArt
           // output ports.  A strict terminal BoxDecode route still owns one
           // DMS arena, but its manifest must retain all six physical port
           // bindings and their exact arena offsets.  Collapsing them into a
           // legacy packed-parent physical output makes ProcessMLA reject the
           // Core/backend OFM arity before the graph can start.
           if (const char* six_ofm_package =
                   std::getenv("SIMANEAT_EXACT_YOLOV8N_SIX_REAL_OFM_PACKAGE");
               six_ofm_package != nullptr && *six_ofm_package != '\0') {
             simaai::neat::internal::InferenceTerminalPolicy terminal_policy;
             terminal_policy.mla_only = true;
             ModelPack six_ofm_model(
                 six_ofm_package, "application/vnd.simaai.tensor", "FP32", 3,
                 640, 640, 3, false, {}, {}, {}, PipelineType::Quant,
                 "decoder", 4, 4, 0, -1, {}, {}, terminal_policy,
                 /*cleanup_extracted_model_data=*/false);

             simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags flags;
             flags.boxdecode_selected = true;
             six_ofm_model.set_model_managed_stage_facts(
                 /*processcvu_preproc_single_output_handoff=*/true, flags,
                 {ExecutionStageKind::BoxDecode});

             const auto selected_facts =
                 six_ofm_model.stage_facts_for_model_stage(ModelStage::Full);
             const simaai::neat::CompiledMlaContract* terminal_mla = nullptr;
             for (const auto& fact : selected_facts) {
               if (fact.mla_compiled.has_value()) {
                 terminal_mla = &*fact.mla_compiled;
               }
             }
             require(terminal_mla != nullptr,
                     "six-real-OFM selected route must retain terminal MLA");

             const auto& runtime = terminal_mla->runtime_contract;
             const std::array<std::int64_t, 6U> expected_offsets{
                 0, 409600, 512000, 540672, 1052672, 1183744};
             const std::array<std::uint64_t, 6U> expected_extents{
                 409600U, 102400U, 25600U, 512000U, 128000U, 32000U};
             require(runtime.frame_arena_storage_domain ==
                         sc::ArenaStorageDomain::Dms &&
                         runtime.frame_arena_role == FrameArenaRole::Allocate &&
                         runtime.frame_arena_size_bytes == 1216512U,
                     "six-real-OFM terminal MLA must own one aligned 1,216,512-byte DMS arena "
                     "(actual_bytes=" + std::to_string(runtime.frame_arena_size_bytes) + ")");
             require(runtime.physical_outputs.size() == expected_offsets.size() &&
                         terminal_mla->dispatcher_physical_outputs.size() ==
                             expected_offsets.size() &&
                         terminal_mla->payload.dispatcher_output_sizes.size() ==
                             expected_offsets.size() &&
                         runtime.elf_ofm_symbol_names.size() == expected_offsets.size() &&
                         runtime.logical_outputs.size() == expected_offsets.size(),
                     "six-real-OFM Core manifest must remain arity-compatible with ProcessMLA");
             for (std::size_t index = 0U; index < expected_offsets.size(); ++index) {
               const auto& physical = runtime.physical_outputs[index];
               const auto& dispatcher = terminal_mla->dispatcher_physical_outputs[index];
               const auto& logical = runtime.logical_outputs[index];
               require(physical.physical_index == static_cast<int>(index) &&
                           physical.source_physical_index == static_cast<int>(index) &&
                           physical.source_byte_offset == expected_offsets[index] &&
                           physical.size_bytes == expected_extents[index] &&
                           dispatcher.size_bytes == expected_extents[index] &&
                           terminal_mla->payload.dispatcher_output_sizes[index] ==
                               expected_extents[index] &&
                           logical.backend_output_index == static_cast<int>(index) &&
                           logical.physical_index == static_cast<int>(index) &&
                           logical.byte_offset == 0 &&
                           logical.size_bytes == expected_extents[index],
                       "six-real-OFM port/view geometry changed at index " +
                           std::to_string(index));
             }
           }
         }));
