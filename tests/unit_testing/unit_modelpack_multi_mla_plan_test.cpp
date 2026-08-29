#define SIMA_NEAT_INTERNAL 1
#include "model/Model.h"
#include "model/internal/ModelPack.h"
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
                        "params":{"out_dtype":")json"
         << mla_scalar << R"json(","input_shapes":[[1,)json" << input_elements
         << R"json(]],"output_shapes":[[1,)json" << input_elements << R"json(]]}},
       "input_nodes":[{"name":"input","size":)json"
         << model_input_bytes << R"json(,"dtype":")json" << model_input_scalar
         << R"json(","shape":[1,)json" << input_elements << R"json(]}],
       "output_nodes":[{"name":"mla_input","size":)json"
         << input_bytes << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}]},
      {"name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":")json"
         << mla_scalar << R"json(","shape":[1,)json" << input_elements << R"json(]}],
                        "output_types":[{"scalar":")json"
         << mla_scalar << R"json(","shape":[1,)json" << input_elements << R"json(]}]},
       "input_nodes":[{"name":"mla_input","size":)json"
         << input_bytes << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "output_nodes":[{"name":"mla_head","size":)json"
         << input_bytes << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "resources":{"executable":"model.elf"}},
      {"name":"cast_head","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast_transform",
                        "params":{"out_dtype":")json"
         << cast_output_scalar << R"json(","input_shapes":[[1,)json" << input_elements
         << R"json(]],"output_shapes":[[1,)json" << output_elements << R"json(]]}},
       "input_nodes":[{"name":"mla_head","size":)json"
         << input_bytes << R"json(,"dtype":")json" << mla_scalar << R"json(","shape":[1,)json"
         << input_elements << R"json(]}],
       "output_nodes":[{"name":"class_prob_0","size":)json"
         << output_bytes << R"json(,"dtype":")json" << cast_output_scalar
         << R"json(","shape":[1,)json" << output_elements << R"json(]}]}
    ]
  })json";
  require(output.good(), "failed to write synthetic MLA-to-Cast MPK");
}

void write_boxdecode_relation_tail_manifest(const std::filesystem::path& path,
                                            const bool interleave_materializers,
                                            const bool add_unused_relation = false) {
  using json = nlohmann::ordered_json;
  const auto node = [](const std::string& name, const std::uint64_t size) {
    return json{{"name", name}, {"type", "buffer"}, {"size", size}};
  };
  json plugins = json::array();
  plugins.push_back(
      {{"name", "MLA_0"},
       {"sequence", 1},
       {"processor", "MLA"},
       {"type", "sgpProcess"},
       {"config_params",
        {{"desired_batch_size", 1},
         {"actual_batch_size", 1},
         {"number_of_quads_to_user", 4},
         {"input_types", json::array({{{"scalar", "float32"}, {"shape", {1, 4}}}})},
         {"output_types", json::array({{{"scalar", "int8"}, {"shape", {1, 2016}}}})}}},
       {"input_nodes", json::array({node("input", 16U)})},
       {"output_nodes", json::array({node("mla_root", 2016U)})},
       {"resources", {{"executable", "model.elf"}}}});
  plugins.push_back(
      {{"name", "unpack"},
       {"sequence", 2},
       {"processor", "EV74"},
       {"type", "sgpProcess"},
       {"config_params",
        {{"desired_batch_size", 1},
         {"actual_batch_size", 1},
         {"kernel", "unpack_transform"},
         {"params",
          {{"tensor_types", {"int8", "int8", "int8", "int8", "int8", "int8"}},
           {"tensor_shapes", json::array({{1, 4, 4, 16},
                                          {1, 2, 2, 16},
                                          {1, 1, 1, 16},
                                          {1, 4, 4, 80},
                                          {1, 2, 2, 80},
                                          {1, 1, 1, 80}})},
           {"input_shapes", json::array({{1, 2016}})},
           {"output_shapes", json::array({{1, 4, 4, 16},
                                          {1, 2, 2, 16},
                                          {1, 1, 1, 16},
                                          {1, 4, 4, 80},
                                          {1, 2, 2, 80},
                                          {1, 1, 1, 80}})}}}}},
       {"input_nodes", json::array({node("mla_root", 2016U)})},
       {"output_nodes", json::array({node("bbox_parent_0", 256U), node("bbox_parent_1", 64U),
                                     node("bbox_parent_2", 16U), node("class_raw_0", 1280U),
                                     node("class_raw_1", 320U), node("class_raw_2", 80U)})},
       {"resources", {{"executable", "kernel_name_tbd"}}}});

  const std::array<std::array<std::int64_t, 4U>, 3U> bbox_shapes{
      std::array<std::int64_t, 4U>{1, 4, 4, 16}, std::array<std::int64_t, 4U>{1, 2, 2, 16},
      std::array<std::int64_t, 4U>{1, 1, 1, 16}};
  const std::array<std::uint64_t, 3U> bbox_parent_bytes{256U, 64U, 16U};
  const std::array<std::uint64_t, 3U> bbox_raw_bytes{64U, 16U, 4U};
  const std::array<std::uint64_t, 3U> bbox_fp32_bytes{256U, 64U, 16U};
  std::array<json, 3U> slices;
  std::array<json, 3U> bbox_dequants;
  for (std::size_t head = 0U; head < 3U; ++head) {
    auto output_shape = bbox_shapes[head];
    output_shape[3] = 4;
    slices[head] = {{"name", "slice_" + std::to_string(head)},
                    {"processor", "EV74"},
                    {"type", "sgpProcess"},
                    {"config_params",
                     {{"desired_batch_size", 1},
                      {"actual_batch_size", 1},
                      {"kernel", "slice_transform"},
                      {"params",
                       {{"begin", {0, 0, 0, 0}},
                        {"end", {output_shape[0], output_shape[1], output_shape[2], 4}},
                        {"input_shape", bbox_shapes[head]},
                        {"output_shape", output_shape},
                        {"input_shapes", json::array({bbox_shapes[head]})},
                        {"output_shapes", json::array({output_shape})}}}}},
                    {"input_nodes", json::array({node("bbox_parent_" + std::to_string(head),
                                                      bbox_parent_bytes[head])})},
                    {"output_nodes",
                     json::array({node("bbox_raw_" + std::to_string(head), bbox_raw_bytes[head])})},
                    {"resources", {{"executable", "kernel_name_tbd"}}}};
    bbox_dequants[head] = {
        {"name", "dequant_bbox_" + std::to_string(head)},
        {"processor", "EV74"},
        {"type", "sgpProcess"},
        {"config_params",
         {{"desired_batch_size", 1},
          {"actual_batch_size", 1},
          {"kernel", "dequantization_transform"},
          {"params",
           {{"channel_params", json::array({json::array({32.0, -128})})},
            {"input_data_type", "int8"},
            {"input_shapes", json::array({output_shape})},
            {"output_shapes", json::array({output_shape})}}}}},
        {"input_nodes",
         json::array({node("bbox_raw_" + std::to_string(head), bbox_raw_bytes[head])})},
        {"output_nodes",
         json::array({node("bbox_" + std::to_string(head), bbox_fp32_bytes[head])})},
        {"resources", {{"executable", "kernel_name_tbd"}}}};
  }
  int sequence = 3;
  if (interleave_materializers) {
    for (std::size_t head = 0U; head < 3U; ++head) {
      slices[head]["sequence"] = sequence++;
      plugins.push_back(slices[head]);
      bbox_dequants[head]["sequence"] = sequence++;
      plugins.push_back(bbox_dequants[head]);
    }
  } else {
    for (auto& slice : slices) {
      slice["sequence"] = sequence++;
      plugins.push_back(slice);
    }
    for (auto& dequant : bbox_dequants) {
      dequant["sequence"] = sequence++;
      plugins.push_back(dequant);
    }
  }

  const std::array<std::array<std::int64_t, 4U>, 3U> class_shapes{
      std::array<std::int64_t, 4U>{1, 4, 4, 80}, std::array<std::int64_t, 4U>{1, 2, 2, 80},
      std::array<std::int64_t, 4U>{1, 1, 1, 80}};
  const std::array<std::uint64_t, 3U> class_raw_bytes{1280U, 320U, 80U};
  const std::array<std::uint64_t, 3U> class_fp32_bytes{5120U, 1280U, 320U};
  for (std::size_t head = 0U; head < 3U; ++head) {
    plugins.push_back(
        {{"name", "dequant_class_" + std::to_string(head)},
         {"sequence", sequence++},
         {"processor", "EV74"},
         {"type", "sgpProcess"},
         {"config_params",
          {{"desired_batch_size", 1},
           {"actual_batch_size", 1},
           {"kernel", "dequantization_transform"},
           {"params",
            {{"channel_params", json::array({json::array({8.0, -128})})},
             {"input_data_type", "int8"},
             {"input_shapes", json::array({class_shapes[head]})},
             {"output_shapes", json::array({class_shapes[head]})}}}}},
         {"input_nodes",
          json::array({node("class_raw_" + std::to_string(head), class_raw_bytes[head])})},
         {"output_nodes",
          json::array({node("class_prob_" + std::to_string(head), class_fp32_bytes[head])})},
         {"resources", {{"executable", "kernel_name_tbd"}}}});
  }
  if (add_unused_relation) {
    plugins.push_back({{"name", "unused_slice"},
                       {"sequence", sequence++},
                       {"processor", "EV74"},
                       {"type", "sgpProcess"},
                       {"config_params",
                        {{"desired_batch_size", 1},
                         {"actual_batch_size", 1},
                         {"kernel", "slice_transform"},
                         {"params",
                          {{"begin", {0, 0, 0, 0}},
                           {"end", {1, 4, 4, 4}},
                           {"input_shape", {1, 4, 4, 16}},
                           {"output_shape", {1, 4, 4, 4}},
                           {"input_shapes", json::array({{1, 4, 4, 16}})},
                           {"output_shapes", json::array({{1, 4, 4, 4}})}}}}},
                       {"input_nodes", json::array({node("bbox_parent_0", 256U)})},
                       {"output_nodes", json::array({node("unused_raw", 64U)})},
                       {"resources", {{"executable", "kernel_name_tbd"}}}});
  }

  json pass_inputs = json::array();
  json pass_outputs = json::array();
  for (std::size_t head = 0U; head < 3U; ++head) {
    pass_inputs.push_back(node("bbox_" + std::to_string(head), bbox_fp32_bytes[head]));
    pass_outputs.push_back(node("pass_bbox_" + std::to_string(head), bbox_fp32_bytes[head]));
  }
  for (std::size_t head = 0U; head < 3U; ++head) {
    pass_inputs.push_back(node("class_prob_" + std::to_string(head), class_fp32_bytes[head]));
    pass_outputs.push_back(node("pass_class_" + std::to_string(head), class_fp32_bytes[head]));
  }
  plugins.push_back({{"name", "PassThrough"},
                     {"sequence", sequence},
                     {"processor", "EV74"},
                     {"type", "sgpProcess"},
                     {"config_params",
                      {{"desired_batch_size", 1},
                       {"actual_batch_size", 1},
                       {"kernel", "pass_through"},
                       {"params", json::object()}}},
                     {"input_nodes", std::move(pass_inputs)},
                     {"output_nodes", std::move(pass_outputs)},
                     {"resources", nullptr}});

  json manifest{{"name", interleave_materializers ? "modelpack-interleaved-terminal-relations"
                                                  : "modelpack-prefix-terminal-relations"},
                {"model_sdk_version", "2.1.0"},
                {"input_nodes", json::array({node("input", 16U)})},
                {"plugins", std::move(plugins)}};
  std::ofstream output(path);
  require(output.is_open(), "failed to create synthetic relation-tail MPK");
  output << manifest.dump(2);
  require(output.good(), "failed to write synthetic relation-tail MPK");
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
           using simaai::neat::pipeline_internal::sima::FrameArenaRole;
           namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

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
           require(model.dmabuf_plan_admission().eligible(),
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
           singleton_model.prepare_for_execution();
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

           // A terminal consumer changes only the rendered execution boundary.
           // The compiler-authored semantic/physical/arena plan and its audit
           // proof remain immutable, including interleaved materializing tail
           // operations. Unpack/Slice are read views over the MLA carrier.
           const auto make_relation_tail_package = [&](const std::string& label,
                                                       const bool interleaved,
                                                       const bool add_unused_relation = false) {
             const fs::path package = root.parent_path() / label;
             fs::remove_all(package, ec);
             fs::create_directories(package / "etc", ec);
             fs::create_directories(package / "lib", ec);
             fs::create_directories(package / "share", ec);
             require(!ec, "failed to create synthetic relation-tail package layout");
             write_boxdecode_relation_tail_manifest(package / "etc" / "model_mpk.json", interleaved,
                                                    add_unused_relation);
             write_topology_elf(package / "share" / "model.elf",
                                "data.ifm.persistent.afe_direct_input_0.b0", 16U,
                                "data.ofm.persistent.afe_mla_output_0.b0", 2016U);
             return package;
           };
           const auto select_relation_tail_route = [&](const fs::path& package) {
             simaai::neat::internal::InferenceTerminalPolicy policy;
             policy.mla_only = true;
             ModelPack candidate(package.string(), "application/vnd.simaai.tensor", "FP32", 1, 4, 1,
                                 1, false, {}, {}, {}, PipelineType::Preproc, "decoder", 4, 4, 0,
                                 -1, {}, {}, policy,
                                 /*cleanup_extracted_model_data=*/false);
             const auto full_proof = candidate.dmabuf_plan_admission().proof;
             const auto full_digest = candidate.dmabuf_plan_digest();
             simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags route_flags;
             route_flags.boxdecode_selected = true;
             route_flags.terminal_consumer_owns_tensor_tail = true;
             candidate.set_model_managed_stage_facts(std::nullopt, route_flags,
                                                     {ExecutionStageKind::BoxDecode});
             return std::tuple{std::move(candidate), full_proof, full_digest};
           };
           const auto require_proof_unchanged = [](const auto& before, const auto& after) {
             require(before.size() == after.size(),
                     "terminal selection changed immutable admission proof cardinality");
             for (std::size_t index = 0U; index < before.size(); ++index) {
               require(before[index].subject == after[index].subject &&
                           before[index].evidence == after[index].evidence,
                       "terminal selection rewrote immutable admission proof");
             }
           };

           const auto interleaved_package =
               make_relation_tail_package("neat-modelpack-boxdecode-interleaved-tail-unit", true);
           auto [interleaved_model, interleaved_full_proof, interleaved_full_digest] =
               select_relation_tail_route(interleaved_package);
           require(interleaved_model.dmabuf_plan_digest() == interleaved_full_digest,
                   "terminal selection changed the immutable full-plan digest");
           require_proof_unchanged(interleaved_full_proof,
                                   interleaved_model.dmabuf_plan_admission().proof);
           const auto interleaved_plan = interleaved_model.execution_plan();
           require(interleaved_plan.pre.empty() && interleaved_plan.post.empty(),
                   "physical plan was split across competing route authorities");
           require(std::any_of(interleaved_plan.infer.begin(), interleaved_plan.infer.end(),
                               [](const auto& stage) {
                                 return stage.kind == ExecutionStageKind::Dequant;
                               }),
                   "terminal selection removed compiler-authored semantic tail operations");
           require(interleaved_model.infer_block().size() == 1U,
                   "terminal rendering did not stop execution at the MLA");

           const auto interleaved_facts =
               interleaved_model.stage_facts_for_model_stage(ModelStage::Full);
           const simaai::neat::CompiledRuntimeContract* interleaved_mla = nullptr;
           for (const auto& fact : interleaved_facts) {
             if (fact.mla_compiled.has_value()) {
               interleaved_mla = &fact.mla_compiled->runtime_contract;
             }
           }
           const std::array<std::string, 6U> expected_interleaved_names{
               "bbox_raw_0",  "bbox_raw_1",  "bbox_raw_2",
               "class_raw_0", "class_raw_1", "class_raw_2"};
           require(interleaved_mla != nullptr &&
                       interleaved_mla->logical_outputs.size() == expected_interleaved_names.size(),
                   "full-plan projection lost six raw terminal read views");
           for (std::size_t index = 0U; index < expected_interleaved_names.size(); ++index) {
             require(interleaved_mla->logical_outputs[index].logical_name ==
                         expected_interleaved_names[index],
                     "full-plan projection changed raw terminal-view order");
           }

           const auto prefix_package =
               make_relation_tail_package("neat-modelpack-boxdecode-prefix-tail-unit", false);
           auto [prefix_model, prefix_full_proof, prefix_full_digest] =
               select_relation_tail_route(prefix_package);
           require(prefix_model.dmabuf_plan_digest() == prefix_full_digest,
                   "dense-prefix terminal selection changed the full-plan digest");
           require_proof_unchanged(prefix_full_proof,
                                   prefix_model.dmabuf_plan_admission().proof);

           // Optional exact-artifact gate for the App48 YOLO26 package whose
           // compiler order interleaves Slice and Dequant operations.
           if (const char* yolo26_package =
                   std::getenv("SIMANEAT_EXACT_YOLO26_INTERLEAVED_PACKAGE");
               yolo26_package != nullptr && *yolo26_package != '\0') {
             simaai::neat::internal::InferenceTerminalPolicy policy;
             policy.mla_only = true;
             ModelPack yolo26_model(yolo26_package, "application/vnd.simaai.tensor", "FP32", 3, 640,
                                    640, 3, false, {}, {}, {}, PipelineType::QuantTess, "decoder",
                                    4, 4, 0, -1, {}, {}, policy,
                                    /*cleanup_extracted_model_data=*/false);
             const auto full_proof = yolo26_model.dmabuf_plan_admission().proof;
             const auto full_digest = yolo26_model.dmabuf_plan_digest();
             simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags route_flags;
             route_flags.boxdecode_selected = true;
             route_flags.terminal_consumer_owns_tensor_tail = true;
             yolo26_model.set_model_managed_stage_facts(std::nullopt, route_flags,
                                                        {ExecutionStageKind::BoxDecode});
             require(yolo26_model.dmabuf_plan_digest() == full_digest,
                     "exact YOLO26 terminal selection changed the full-plan digest");
             require_proof_unchanged(full_proof,
                                     yolo26_model.dmabuf_plan_admission().proof);
             require(yolo26_model.infer_block().size() == 1U,
                     "exact YOLO26 terminal rendering did not stop at MLA");
             const auto selected_facts = yolo26_model.stage_facts_for_model_stage(ModelStage::Full);
             const simaai::neat::CompiledRuntimeContract* terminal_mla = nullptr;
             for (const auto& fact : selected_facts) {
               if (fact.mla_compiled.has_value()) {
                 terminal_mla = &fact.mla_compiled->runtime_contract;
               }
             }
             const std::array<std::string, 6U> expected_names{
                 "slice_MLA_0/tuple_get_item_0_slice_transform",
                 "slice_MLA_0/tuple_get_item_1_slice_transform",
                 "slice_MLA_0/tuple_get_item_2_slice_transform",
                 "MLA_0_ofm_unpack_transform_3",
                 "MLA_0_ofm_unpack_transform_4",
                 "MLA_0_ofm_unpack_transform_5"};
             require(terminal_mla != nullptr &&
                         terminal_mla->logical_outputs.size() == expected_names.size(),
                     "exact YOLO26 full-plan projection lost six raw views");
             for (std::size_t index = 0U; index < expected_names.size(); ++index) {
               require(terminal_mla->logical_outputs[index].logical_name == expected_names[index],
                       "exact YOLO26 full-plan projection changed view order");
             }
           }

           const fs::path multirow_package =
               make_mla_a65_package("neat-modelpack-mla-a65-multirow-pitch-unit", 2U);
           bool multirow_rejected = false;
           try {
             ModelPack multirow_model(multirow_package.string());
             multirow_model.prepare_for_execution();
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
           join_model.prepare_for_execution();
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
             require(real_model.dmabuf_plan_admission().eligible(),
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

           // Optional exact YOLOv8 regression: selecting the terminal consumer
           // must preserve the admitted full plan while its region view exposes
           // one immutable schedule. The MLA contract still publishes six raw
           // views, while terminal rendering stops after graph226 plus MLA.
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
             const auto full_digest = yolo_model.dmabuf_plan_digest();
             const auto full_proof = yolo_model.dmabuf_plan_admission().proof;

             simaai::neat::pipeline_internal::sima::ModelManagedRouteFlags flags;
             flags.boxdecode_selected = true;
             flags.terminal_consumer_owns_tensor_tail = true;
             yolo_model.set_model_managed_stage_facts(
                 /*processcvu_preproc_single_output_handoff=*/true, flags,
                 {ExecutionStageKind::BoxDecode});

             const auto selected_plan = yolo_model.execution_plan();
             require(selected_plan.pre.empty() && selected_plan.post.empty(),
                     "exact YOLO physical plan was split across route stages");
             require(std::any_of(selected_plan.infer.begin(), selected_plan.infer.end(),
                                 [](const auto& stage) {
                                   return stage.kind == ExecutionStageKind::DetessDequant;
                                 }),
                     "terminal selection removed graph227 from the immutable full plan");
             require(yolo_model.dmabuf_plan_digest() == full_digest,
                     "terminal selection changed the exact YOLO full-plan digest");
             require_proof_unchanged(full_proof,
                                     yolo_model.dmabuf_plan_admission().proof);
             require(yolo_model.infer_block().size() == 2U,
                     "exact YOLO terminal renderer did not stop after graph226 and MLA");

             const auto selected_facts =
                 yolo_model.stage_facts_for_model_stage(ModelStage::Full);
             const simaai::neat::CompiledRuntimeContract* terminal_mla = nullptr;
             for (const auto& fact : selected_facts) {
               if (fact.mla_compiled.has_value()) {
                 terminal_mla = &fact.mla_compiled->runtime_contract;
               }
             }
             const std::array<std::int64_t, 6U> expected_offsets{
                 0, 409600, 512000, 537600, 1049600, 1177600};
             const std::array<std::uint64_t, 6U> expected_spans{
                 409600U, 102400U, 25600U, 512000U, 128000U, 32000U};
             require(terminal_mla != nullptr &&
                         terminal_mla->logical_outputs.size() == expected_offsets.size(),
                     "exact YOLO full-plan projection lost six BoxDecode views");
             for (std::size_t index = 0U; index < expected_offsets.size(); ++index) {
               const auto& logical = terminal_mla->logical_outputs[index];
               require(logical.byte_offset == expected_offsets[index] &&
                           logical.size_bytes == expected_spans[index] &&
                           logical.physical_index == 0,
                       "exact YOLO full-plan projection changed an affine view");
             }
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
             flags.terminal_consumer_owns_tensor_tail = true;
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
