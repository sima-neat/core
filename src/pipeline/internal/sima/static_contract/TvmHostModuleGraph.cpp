#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/TvmHostModuleGraph.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {
using Json = nlohmann::json;

bool fail(std::string* error, std::string detail) {
  if (error) {
    *error = std::move(detail);
  }
  return false;
}

std::optional<HostTensorTypeSpec> tensor_type(const Json& shapes, const Json& dtypes,
                                              const std::size_t entry, std::string* error) {
  if (!shapes.is_array() || !dtypes.is_array() || entry >= shapes.size() ||
      entry >= dtypes.size() || !shapes[entry].is_array() || !dtypes[entry].is_string()) {
    fail(error, "TVM graph tensor attribute cardinality/type is invalid");
    return std::nullopt;
  }
  if (shapes[entry].empty() || shapes[entry].size() > 32U) {
    fail(error, "TVM graph tensor rank is empty or exceeds the structural-audit bound");
    return std::nullopt;
  }
  HostTensorTypeSpec result;
  result.scalar = dtypes[entry].get<std::string>();
  if (result.scalar != "bool" && result.scalar != "int8" && result.scalar != "uint8" && result.scalar != "int16" &&
      result.scalar != "uint16" && result.scalar != "int32" && result.scalar != "uint32" &&
      result.scalar != "int64" && result.scalar != "uint64" && result.scalar != "float16" &&
      result.scalar != "bfloat16" && result.scalar != "float32" && result.scalar != "float64") {
    fail(error, "TVM graph tensor dtype is not registered");
    return std::nullopt;
  }
  for (const auto& dim : shapes[entry]) {
    if (!dim.is_number_integer()) {
      fail(error, "TVM graph tensor shape contains a non-integer");
      return std::nullopt;
    }
    const auto value = dim.get<std::int64_t>();
    if (value <= 0) {
      fail(error, "TVM graph tensor shape contains a non-positive dimension");
      return std::nullopt;
    }
    result.shape.push_back(value);
  }
  return result;
}

const Json* attribute_values(const Json& attrs, const char* name, const char* expected_tag,
                             std::string* error) {
  if (!attrs.is_object() || !attrs.contains(name)) {
    fail(error, std::string("TVM graph lacks '") + name + "' attributes");
    return nullptr;
  }
  const auto& value = attrs.at(name);
  if (!value.is_array() || value.size() != 2U || !value[0].is_string() ||
      value[0].get<std::string>() != expected_tag || !value[1].is_array()) {
    fail(error, std::string("TVM graph '") + name + "' attribute encoding is invalid");
    return nullptr;
  }
  return &value[1];
}

std::optional<std::size_t> entry_id(const Json& row_ptr, const Json& tuple, std::string* error) {
  if (!row_ptr.is_array() || !tuple.is_array() || tuple.size() < 2U ||
      (!tuple[0].is_number_unsigned() && !tuple[0].is_number_integer()) ||
      (!tuple[1].is_number_unsigned() && !tuple[1].is_number_integer())) {
    fail(error, "TVM graph node-entry tuple is invalid");
    return std::nullopt;
  }
  const auto node = tuple[0].get<std::int64_t>();
  const auto output = tuple[1].get<std::int64_t>();
  if (node < 0 || output < 0 || static_cast<std::size_t>(node + 1) >= row_ptr.size() ||
      !row_ptr[static_cast<std::size_t>(node)].is_number_integer() ||
      !row_ptr[static_cast<std::size_t>(node + 1)].is_number_integer()) {
    fail(error, "TVM graph node-entry index is out of range");
    return std::nullopt;
  }
  const auto first = row_ptr[static_cast<std::size_t>(node)].get<std::int64_t>();
  const auto last = row_ptr[static_cast<std::size_t>(node + 1)].get<std::int64_t>();
  if (first < 0 || last < first || output >= last - first) {
    fail(error, "TVM graph node output index is out of range");
    return std::nullopt;
  }
  return static_cast<std::size_t>(first + output);
}

std::optional<Json> find_embedded_graph(const std::vector<char>& bytes, std::string* error) {
  constexpr std::string_view marker = "{\n  \"nodes\": [";
  auto cursor = bytes.begin();
  while (cursor != bytes.end()) {
    cursor = std::search(cursor, bytes.end(), marker.begin(), marker.end());
    if (cursor == bytes.end()) {
      break;
    }
    const auto begin = static_cast<std::size_t>(cursor - bytes.begin());
    const auto end_it = std::find(cursor, bytes.end(), '\0');
    if (end_it != bytes.end()) {
      const auto end = static_cast<std::size_t>(end_it - bytes.begin());
      Json candidate = Json::parse(bytes.data() + begin, bytes.data() + end, nullptr, false);
      if (!candidate.is_discarded() && candidate.is_object() && candidate.contains("nodes") &&
          candidate.contains("arg_nodes") && candidate.contains("heads") &&
          candidate.contains("attrs") && candidate.contains("node_row_ptr")) {
        return candidate;
      }
    }
    ++cursor;
  }
  fail(error, "A65 object contains no valid embedded TVM GraphExecutor JSON");
  return std::nullopt;
}

bool is_aarch64_shared_object(const std::vector<char>& bytes, std::string* error) {
  // ELF64 little-endian e_type/e_machine are fixed-width fields at offsets
  // 16 and 18. Read them as bytes so admission never executes or maps code.
  if (bytes.size() < 20U || static_cast<unsigned char>(bytes[0]) != 0x7fU || bytes[1] != 'E' ||
      bytes[2] != 'L' || bytes[3] != 'F' || static_cast<unsigned char>(bytes[4]) != 2U ||
      static_cast<unsigned char>(bytes[5]) != 1U) {
    return fail(error, "A65 host module is not a little-endian ELF64 object");
  }
  const auto u8 = [&bytes](const std::size_t offset) {
    return static_cast<std::uint16_t>(static_cast<unsigned char>(bytes[offset]));
  };
  const std::uint16_t type = u8(16U) | static_cast<std::uint16_t>(u8(17U) << 8U);
  const std::uint16_t machine = u8(18U) | static_cast<std::uint16_t>(u8(19U) << 8U);
  constexpr std::uint16_t kElfDynamicObject = 3U;
  constexpr std::uint16_t kElfMachineAarch64 = 183U;
  if (type != kElfDynamicObject || machine != kElfMachineAarch64) {
    return fail(error, "A65 host module must be an AArch64 ET_DYN object");
  }
  return true;
}

} // namespace

std::optional<TvmHostModuleGraph> read_tvm_host_module_graph(const std::filesystem::path& module,
                                                             std::string* error) noexcept {
  try {
    std::ifstream input(module, std::ios::binary | std::ios::ate);
    if (!input) {
      fail(error, "cannot open A65 host module");
      return std::nullopt;
    }
    const auto length = input.tellg();
    constexpr std::streamoff kMaximumModuleBytes = 256LL * 1024LL * 1024LL;
    if (length <= 0 || length > kMaximumModuleBytes) {
      fail(error, "A65 host module size is invalid or exceeds the structural-audit limit");
      return std::nullopt;
    }
    input.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(length));
    input.read(bytes.data(), length);
    if (!input) {
      fail(error, "cannot read A65 host module");
      return std::nullopt;
    }
    if (!is_aarch64_shared_object(bytes, error)) {
      return std::nullopt;
    }
    const auto graph = find_embedded_graph(bytes, error);
    if (!graph) {
      return std::nullopt;
    }
    const auto& nodes = graph->at("nodes");
    const auto& arg_nodes = graph->at("arg_nodes");
    const auto& heads = graph->at("heads");
    const auto& row_ptr = graph->at("node_row_ptr");
    if (!nodes.is_array() || nodes.empty() || !arg_nodes.is_array() || arg_nodes.empty() ||
        !heads.is_array() || heads.empty() || !row_ptr.is_array() ||
        row_ptr.size() != nodes.size() + 1U) {
      fail(error, "TVM graph topology arrays are empty or inconsistent");
      return std::nullopt;
    }
    constexpr std::size_t kMaximumNodes = 65536U;
    constexpr std::size_t kMaximumEntries = 1048576U;
    constexpr std::size_t kMaximumPorts = 4096U;
    if (nodes.size() > kMaximumNodes || arg_nodes.size() > kMaximumPorts ||
        heads.size() > kMaximumPorts) {
      fail(error, "TVM graph topology exceeds the structural-audit bounds");
      return std::nullopt;
    }
    const auto* shapes = attribute_values(graph->at("attrs"), "shape", "list_shape", error);
    const auto* dtypes = attribute_values(graph->at("attrs"), "dltype", "list_str", error);
    const auto* storage = attribute_values(graph->at("attrs"), "storage_id", "list_int", error);
    if (!shapes || !dtypes || !storage || shapes->size() != dtypes->size() ||
        shapes->size() != storage->size()) {
      fail(error, "TVM graph tensor attributes disagree in cardinality");
      return std::nullopt;
    }
    if (shapes->size() > kMaximumEntries || !row_ptr.front().is_number_integer() ||
        row_ptr.front().get<std::int64_t>() != 0) {
      fail(error, "TVM graph entry table exceeds bounds or does not start at zero");
      return std::nullopt;
    }
    std::int64_t previous_entry = 0;
    for (const auto& entry : row_ptr) {
      if (!entry.is_number_integer()) {
        fail(error, "TVM graph node_row_ptr contains a non-integer");
        return std::nullopt;
      }
      const auto value = entry.get<std::int64_t>();
      if (value < previous_entry) {
        fail(error, "TVM graph node_row_ptr is not monotonic");
        return std::nullopt;
      }
      previous_entry = value;
    }
    if (previous_entry < 0 || static_cast<std::size_t>(previous_entry) != shapes->size()) {
      fail(error, "TVM graph node entries disagree with tensor attributes");
      return std::nullopt;
    }
    for (const auto& storage_id : *storage) {
      if (!storage_id.is_number_integer() || storage_id.get<std::int64_t>() < 0) {
        fail(error, "TVM graph storage_id contains an invalid entry");
        return std::nullopt;
      }
    }
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const auto& node = nodes[node_index];
      if (!node.is_object() || !node.contains("inputs") || !node["inputs"].is_array()) {
        fail(error, "TVM graph node is not a complete object");
        return std::nullopt;
      }
      const auto op = node.value("op", std::string{});
      if (op != "null" && op != "tvm_op") {
        fail(error, "TVM graph node has an unregistered operation class");
        return std::nullopt;
      }
      for (const auto& input_entry : node["inputs"]) {
        if (!entry_id(row_ptr, input_entry, error)) {
          return std::nullopt;
        }
      }
    }

    TvmHostModuleGraph result;
    std::vector<std::size_t> input_entries;
    std::unordered_set<std::int64_t> unique_arguments;
    for (std::size_t index = 0; index < arg_nodes.size(); ++index) {
      if (!arg_nodes[index].is_number_integer()) {
        fail(error, "TVM graph arg_nodes contains a non-integer");
        return std::nullopt;
      }
      const auto node_index = arg_nodes[index].get<std::int64_t>();
      if (node_index < 0 || static_cast<std::size_t>(node_index) >= nodes.size() ||
          !unique_arguments.emplace(node_index).second ||
          !nodes[static_cast<std::size_t>(node_index)].is_object() ||
          nodes[static_cast<std::size_t>(node_index)].value("op", std::string{}) != "null") {
        fail(error, "TVM graph argument does not identify a null input node");
        return std::nullopt;
      }
      Json tuple = Json::array({node_index, 0, 0});
      const auto entry = entry_id(row_ptr, tuple, error);
      const auto type = entry ? tensor_type(*shapes, *dtypes, *entry, error) : std::nullopt;
      if (!entry || !type) {
        return std::nullopt;
      }
      const auto name = nodes[static_cast<std::size_t>(node_index)].value("name", std::string{});
      if (name.empty()) {
        fail(error, "TVM graph input name is empty");
        return std::nullopt;
      }
      result.input_names.push_back(name);
      result.input_types.push_back(*type);
      input_entries.push_back(*entry);
    }

    for (const auto& head : heads) {
      const auto entry = entry_id(row_ptr, head, error);
      const auto type = entry ? tensor_type(*shapes, *dtypes, *entry, error) : std::nullopt;
      if (!entry || !type) {
        return std::nullopt;
      }
      result.output_types.push_back(*type);
      result.output_alias_input.push_back(-1);
    }

    // Accept an address view only for the exact GraphExecutor shape-noop form
    // authored by TVM: one null argument, one __nop node consuming it, one head,
    // and a shared storage id. Anything else remains a real direct host job.
    if (nodes.size() == 2U && result.input_types.size() == 1U && result.output_types.size() == 1U &&
        nodes[1].is_object() && nodes[1].value("op", std::string{}) == "tvm_op" &&
        nodes[1].contains("attrs") && nodes[1]["attrs"].is_object() &&
        nodes[1]["attrs"].value("func_name", std::string{}) == "__nop" &&
        nodes[1].contains("inputs") && nodes[1]["inputs"].is_array() &&
        nodes[1]["inputs"].size() == 1U) {
      const auto nop_input = entry_id(row_ptr, nodes[1]["inputs"][0], error);
      const auto head_entry = entry_id(row_ptr, heads[0], error);
      if (!nop_input || !head_entry) {
        return std::nullopt;
      }
      if (*nop_input == input_entries[0] && (*storage)[*head_entry].is_number_integer() &&
          (*storage)[input_entries[0]].is_number_integer() &&
          (*storage)[*head_entry].get<std::int64_t>() ==
              (*storage)[input_entries[0]].get<std::int64_t>()) {
        result.output_alias_input[0] = 0;
      }
    }
    if (error) {
      error->clear();
    }
    return result;
  } catch (const std::exception& ex) {
    fail(error, std::string("A65 host-module structural audit failed: ") + ex.what());
    return std::nullopt;
  } catch (...) {
    fail(error, "A65 host-module structural audit failed with an unknown exception");
    return std::nullopt;
  }
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
