#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/TvmHostModuleGraph.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat::pipeline_internal::sima::static_contract;

void check(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

std::filesystem::path write_fixture(const std::string& name, const std::string& graph) {
  const auto path =
      std::filesystem::temp_directory_path() / ("neat-tvm-host-graph-" + name + ".bin");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  std::string prefix(64U, '\0');
  prefix[0] = static_cast<char>(0x7fU);
  prefix[1] = 'E';
  prefix[2] = 'L';
  prefix[3] = 'F';
  prefix[4] = 2;                        // ELFCLASS64
  prefix[5] = 1;                        // ELFDATA2LSB
  prefix[16] = 3;                       // ET_DYN
  prefix[18] = static_cast<char>(183U); // EM_AARCH64
  output.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  output.write(graph.data(), static_cast<std::streamsize>(graph.size()));
  output.put('\0');
  check(static_cast<bool>(output), "synthetic host module can be written");
  return path;
}

std::string graph(const std::string& function, const int output_storage) {
  return R"json({
  "nodes": [
    {"op":"null","name":"arm_2_i0","inputs":[]},
    {"op":"tvm_op","name":"reshape_nop","attrs":{"func_name":")json" +
         function +
         R"json(","num_inputs":"1","num_outputs":"1"},"inputs":[[0,0,0]]}
  ],
  "arg_nodes":[0],
  "heads":[[1,0,0]],
  "attrs":{
    "shape":["list_shape",[[1,60,80],[1,1,60,80]]],
    "dltype":["list_str",["float32","float32"]],
    "storage_id":["list_int",[0,)json" +
         std::to_string(output_storage) +
         R"json(]]
  },
  "node_row_ptr":[0,1,2]
})json";
}

} // namespace

int main() {
  std::string error;

  const auto alias_path = write_fixture("alias", graph("__nop", 0));
  const auto alias = read_tvm_host_module_graph(alias_path, &error);
  check(alias.has_value(), "exact embedded GraphExecutor JSON is decoded");
  check(error.empty(), "successful structural inspection clears its error");
  check(alias->input_names == std::vector<std::string>{"arm_2_i0"},
        "compiler-authored input name is preserved");
  check(alias->input_types.size() == 1U && alias->input_types[0].scalar == "float32" &&
            alias->input_types[0].shape == TensorShape({1, 60, 80}),
        "input scalar and shape are exact");
  check(alias->output_types.size() == 1U &&
            alias->output_types[0].shape == TensorShape({1, 1, 60, 80}),
        "output reshape is preserved semantically");
  check(alias->output_alias_input == std::vector<std::int32_t>{0},
        "only exact __nop plus shared storage lowers to an address alias");

  const auto materialized_path = write_fixture("materialized", graph("fused_add", 1));
  const auto materialized = read_tvm_host_module_graph(materialized_path, &error);
  check(materialized.has_value(), "non-nop host graph remains admissible structurally");
  check(materialized->output_alias_input == std::vector<std::int32_t>{-1},
        "a real host graph is never mislabeled as an address alias");

  const auto malformed_path = write_fixture("malformed", "not-json");
  const auto malformed = read_tvm_host_module_graph(malformed_path, &error);
  check(!malformed.has_value() && !error.empty(),
        "module without embedded GraphExecutor metadata fails closed");

  const auto wrong_arch_path = write_fixture("wrong-arch", graph("__nop", 0));
  {
    std::fstream wrong_arch(wrong_arch_path, std::ios::binary | std::ios::in | std::ios::out);
    wrong_arch.seekp(18);
    wrong_arch.put(static_cast<char>(62U)); // EM_X86_64
  }
  const auto wrong_arch = read_tvm_host_module_graph(wrong_arch_path, &error);
  check(!wrong_arch.has_value() && error.find("AArch64 ET_DYN") != std::string::npos,
        "host module target architecture is proved without loading code");

  std::filesystem::remove(alias_path);
  std::filesystem::remove(materialized_path);
  std::filesystem::remove(malformed_path);
  std::filesystem::remove(wrong_arch_path);
  std::cout << "unit_tvm_host_module_graph_test: PASS\n";
  return 0;
}
