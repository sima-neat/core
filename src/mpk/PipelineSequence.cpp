#include "mpk/PipelineSequence.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::mpk {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct StageRecord {
  SequenceEntry entry;
  std::vector<std::string> deps;
  std::size_t original_index = 0;
};

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool is_supported_kernel(const std::string& kernel) {
  static const std::unordered_set<std::string> kSupported = {
      "preproc", "quanttess", "infer", "mla", "detessdequant", "boxdecode",
  };
  return kSupported.find(lower_copy(kernel)) != kSupported.end();
}

void ensure_safe_relative_path(const std::string& path) {
  if (path.empty()) {
    throw std::runtime_error("schema_error: empty configPath is not allowed");
  }

  std::string normalized = path;
  for (char& c : normalized) {
    if (c == '\\')
      c = '/';
  }

  if (normalized.front() == '/') {
    throw std::runtime_error("schema_error: absolute configPath is not allowed: " + path);
  }
  if (normalized.size() >= 2 && std::isalpha(static_cast<unsigned char>(normalized[0])) &&
      normalized[1] == ':') {
    throw std::runtime_error("schema_error: drive-prefixed configPath is not allowed: " + path);
  }

  std::stringstream ss(normalized);
  std::string seg;
  while (std::getline(ss, seg, '/')) {
    if (seg.empty() || seg == ".")
      continue;
    if (seg == "..") {
      throw std::runtime_error("schema_error: traversal segment in configPath is not allowed: " +
                               path);
    }
  }
}

int read_int_required(const json& obj, const char* key) {
  if (!obj.contains(key) || !obj[key].is_number_integer()) {
    throw std::runtime_error(std::string("schema_error: missing integer field '") + key + "'");
  }
  try {
    return obj[key].get<int>();
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("schema_error: invalid integer field '") + key +
                             "': " + e.what());
  }
}

std::string read_string_required(const json& obj, const char* key) {
  if (!obj.contains(key) || !obj[key].is_string()) {
    throw std::runtime_error(std::string("schema_error: missing string field '") + key + "'");
  }
  const std::string value = obj[key].get<std::string>();
  if (value.empty()) {
    throw std::runtime_error(std::string("schema_error: empty string field '") + key + "'");
  }
  return value;
}

std::vector<std::string> read_input_dependencies(const json& obj) {
  std::vector<std::string> deps;
  if (!obj.contains("input")) {
    return deps;
  }

  const auto& in = obj["input"];
  if (in.is_string()) {
    deps.push_back(in.get<std::string>());
    return deps;
  }

  if (!in.is_array()) {
    throw std::runtime_error("schema_error: 'input' must be string or array of strings");
  }

  for (const auto& item : in) {
    if (!item.is_string()) {
      throw std::runtime_error("schema_error: 'input' array must contain only strings");
    }
    deps.push_back(item.get<std::string>());
  }
  return deps;
}

void validate_sorted_records(const std::vector<StageRecord>& records) {
  std::unordered_set<std::string> names;
  names.reserve(records.size());

  for (const auto& rec : records) {
    if (!names.insert(rec.entry.name).second) {
      throw std::runtime_error("schema_error: duplicate stage name: " + rec.entry.name);
    }

    if (!is_supported_kernel(rec.entry.kernel)) {
      throw std::runtime_error("schema_error: unsupported kernel: " + rec.entry.kernel);
    }

    for (const auto& dep : rec.deps) {
      if (dep.empty()) {
        throw std::runtime_error("schema_error: dependency name cannot be empty");
      }
      if (dep == "decoder") {
        continue;
      }
      if (names.find(dep) == names.end()) {
        throw std::runtime_error("schema_error: invalid dependency: '" + dep +
                                 "' is not available before stage '" + rec.entry.name + "'");
      }
    }
  }
}

} // namespace

bool is_pre_adapter_kernel(const std::string& kernel) {
  const std::string k = lower_copy(kernel);
  return k == "preproc" || k == "quanttess";
}

bool is_post_adapter_kernel(const std::string& kernel) {
  const std::string k = lower_copy(kernel);
  return k == "detessdequant" || k == "boxdecode";
}

std::vector<SequenceEntry> load_pipeline_sequence(const std::string& etc_dir) {
  const std::string pipeline_path = (fs::path(etc_dir) / "pipeline_sequence.json").string();
  std::ifstream in_file(pipeline_path);
  if (!in_file.is_open()) {
    throw std::runtime_error("schema_error: failed to open " + pipeline_path);
  }

  json pipeline_json;
  try {
    in_file >> pipeline_json;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("schema_error: pipeline_sequence parse error: ") +
                             e.what());
  }

  if (!pipeline_json.contains("pipelines") || !pipeline_json["pipelines"].is_array() ||
      pipeline_json["pipelines"].empty()) {
    throw std::runtime_error("schema_error: invalid pipeline format");
  }

  const auto& pipeline = pipeline_json["pipelines"][0];
  if (!pipeline.is_object() || !pipeline.contains("sequence") || !pipeline["sequence"].is_array()) {
    throw std::runtime_error("schema_error: missing sequence array");
  }

  const auto& sequence = pipeline["sequence"];
  if (sequence.empty()) {
    throw std::runtime_error("schema_error: sequence array must not be empty");
  }

  std::vector<StageRecord> records;
  records.reserve(sequence.size());

  for (std::size_t i = 0; i < sequence.size(); ++i) {
    const auto& elem = sequence[i];
    if (!elem.is_object()) {
      throw std::runtime_error("schema_error: sequence entry must be an object");
    }

    StageRecord rec;
    rec.original_index = i;
    rec.entry.sequence_id = read_int_required(elem, "sequence_id");
    rec.entry.name = read_string_required(elem, "name");
    rec.entry.plugin_id = read_string_required(elem, "pluginId");
    rec.entry.config_path = read_string_required(elem, "configPath");
    rec.entry.processor = read_string_required(elem, "processor");
    rec.entry.kernel = read_string_required(elem, "kernel");
    rec.deps = read_input_dependencies(elem);

    if (rec.entry.sequence_id <= 0) {
      throw std::runtime_error("schema_error: sequence_id must be positive");
    }

    ensure_safe_relative_path(rec.entry.config_path);

    records.push_back(std::move(rec));
  }

  std::stable_sort(records.begin(), records.end(), [](const StageRecord& a, const StageRecord& b) {
    if (a.entry.sequence_id != b.entry.sequence_id) {
      return a.entry.sequence_id < b.entry.sequence_id;
    }
    return a.original_index < b.original_index;
  });

  validate_sorted_records(records);

  std::vector<SequenceEntry> ordered;
  ordered.reserve(records.size());
  for (const auto& rec : records) {
    ordered.push_back(rec.entry);
  }
  return ordered;
}

SequenceSplit split_sequence_for_infer(const std::vector<SequenceEntry>& seq) {
  SequenceSplit out;
  if (seq.empty())
    return out;

  std::size_t begin = 0;
  while (begin < seq.size() && is_pre_adapter_kernel(seq[begin].kernel)) {
    out.pre.push_back(seq[begin]);
    ++begin;
  }

  std::size_t end = seq.size();
  while (end > begin && is_post_adapter_kernel(seq[end - 1].kernel)) {
    out.post.insert(out.post.begin(), seq[end - 1]);
    --end;
  }

  out.infer.insert(out.infer.end(), seq.begin() + static_cast<long>(begin),
                   seq.begin() + static_cast<long>(end));
  return out;
}

} // namespace simaai::neat::mpk
