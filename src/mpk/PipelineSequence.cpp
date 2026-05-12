#include "mpk/PipelineSequence.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::mpk {
namespace fs = std::filesystem;
using json = nlohmann::json;

bool is_supported_kernel(const std::string& kernel);

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

std::string canonical_kernel(std::string kernel) {
  kernel = lower_copy(std::move(kernel));
  if (kernel == "tessellate")
    return "tess";
  return kernel;
}

bool ends_with_ci(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size())
    return false;
  const std::string v = lower_copy(value);
  const std::string s = lower_copy(suffix);
  return std::equal(s.rbegin(), s.rend(), v.rbegin());
}

bool is_mla_stage(const SequenceEntry& entry) {
  const std::string k = canonical_kernel(entry.kernel);
  if (k == "mla") {
    return true;
  }

  const std::string processor = lower_copy(entry.processor);
  if (processor == "mla") {
    return true;
  }

  const std::string plugin = lower_copy(entry.plugin_id);
  return plugin.find("mla") != std::string::npos;
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

std::string infer_processor_from_context(const json& obj, const std::string& plugin_id,
                                         const std::string& kernel) {
  const std::string plugin = lower_copy(plugin_id);
  const std::string k = lower_copy(kernel);

  if (plugin.find("mla") != std::string::npos || k == "mla") {
    return "MLA";
  }
  if (plugin.find("cvu") != std::string::npos || k == "preproc" || k == "quant" || k == "tess" ||
      k == "quanttess" || k == "detessdequant" || k == "dequantize") {
    return "CVU";
  }
  if (plugin.find("boxdecode") != std::string::npos || k == "boxdecode") {
    return "APU";
  }

  if (obj.contains("processor") && obj["processor"].is_number()) {
    // Legacy packs sometimes encode processor as an integer enum.
    // Keep deterministic mapping for known values and default to APU.
    int v = 0;
    try {
      v = obj["processor"].get<int>();
    } catch (const std::exception&) {
      v = -1;
    }
    if (v == 1)
      return "CVU";
    if (v == 2)
      return "MLA";
  }
  return "APU";
}

std::string read_processor_flexible(const json& obj, const std::string& plugin_id,
                                    const std::string& kernel) {
  if (obj.contains("processor")) {
    if (obj["processor"].is_string()) {
      const std::string v = obj["processor"].get<std::string>();
      if (!v.empty()) {
        return v;
      }
      throw std::runtime_error("schema_error: empty string field 'processor'");
    }
    if (obj["processor"].is_number()) {
      return infer_processor_from_context(obj, plugin_id, kernel);
    }
    throw std::runtime_error("schema_error: invalid field 'processor'");
  }
  return infer_processor_from_context(obj, plugin_id, kernel);
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
      throw std::runtime_error("schema_error: unsupported kernel: '" + rec.entry.kernel +
                               "' in stage '" + rec.entry.name +
                               "'. Supported kernels: preproc, quant, tess, tessellate, quanttess,"
                               " cast, infer, mla, detessdequant, detessellate, dequantize,"
                               " boxdecode, buffer_concat.");
    }

    for (const auto& dep : rec.deps) {
      if (dep.empty()) {
        throw std::runtime_error("schema_error: dependency name cannot be empty");
      }
    }
  }

  for (const auto& rec : records) {
    for (const auto& dep : rec.deps) {
      if (dep == "decoder") {
        continue;
      }
      if (names.find(dep) == names.end()) {
        std::string available;
        for (const auto& n : names) {
          if (!available.empty())
            available += ", ";
          available += "'" + n + "'";
        }
        throw std::runtime_error(
            "schema_error: dependency '" + dep + "' in stage '" + rec.entry.name +
            "' references unknown stage. Available stages: [" + available + "].");
      }
    }
  }
}

SequenceEntry make_entry(int sequence_id, std::string name, std::string plugin_id,
                         std::string processor, std::string kernel, std::string config_path) {
  SequenceEntry out;
  out.sequence_id = sequence_id;
  out.name = std::move(name);
  out.plugin_id = std::move(plugin_id);
  out.processor = std::move(processor);
  out.kernel = canonical_kernel(std::move(kernel));
  out.config_path = std::move(config_path);
  return out;
}

std::optional<std::string> find_config_by_suffix(const fs::path& etc_dir,
                                                 std::initializer_list<const char*> suffixes) {
  std::vector<std::string> matches;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(etc_dir, ec)) {
    if (ec)
      break;
    if (!entry.is_regular_file())
      continue;
    const auto path = entry.path();
    if (path.extension() != ".json")
      continue;
    const std::string name = path.filename().string();
    for (const char* suffix : suffixes) {
      if (!suffix || !*suffix)
        continue;
      if (ends_with_ci(name, suffix)) {
        matches.push_back(name);
        break;
      }
    }
  }
  if (matches.empty()) {
    return std::nullopt;
  }
  std::sort(matches.begin(), matches.end());
  matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
  return matches.front();
}

std::vector<SequenceEntry> synthesize_mla_only_sequence(const std::string& etc_dir) {
  const fs::path dir(etc_dir);
  std::error_code ec;
  if (!fs::exists(dir, ec) || ec || !fs::is_directory(dir, ec)) {
    return {};
  }

  const auto mla_cfg = find_config_by_suffix(dir, {"_process_mla.json"});
  if (!mla_cfg.has_value()) {
    return {};
  }

  std::vector<SequenceEntry> out;
  out.reserve(1);
  SequenceEntry mla = make_entry(1, "simaaiprocessmla_1", "processmla", "MLA", "mla", *mla_cfg);
  ensure_safe_relative_path(mla.config_path);
  out.push_back(std::move(mla));
  return out;
}

std::vector<SequenceEntry> load_pipeline_sequence_strict(const std::string& etc_dir) {
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
    rec.entry.kernel = canonical_kernel(read_string_required(elem, "kernel"));
    rec.entry.processor = read_processor_flexible(elem, rec.entry.plugin_id, rec.entry.kernel);
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

} // namespace

bool is_pre_adapter_kernel(const std::string& kernel) {
  const std::string k = canonical_kernel(kernel);
  return k == "preproc" || k == "quant" || k == "tess" || k == "quanttess";
}

bool is_post_adapter_kernel(const std::string& kernel) {
  const std::string k = lower_copy(kernel);
  return k == "detessdequant" || k == "dequantize" || k == "boxdecode";
}

SequenceLoadResult load_pipeline_sequence_with_source(const std::string& etc_dir) {
  SequenceLoadResult out;
  const fs::path pipeline_path = fs::path(etc_dir) / "pipeline_sequence.json";
  std::error_code ec;
  const bool has_pipeline_sequence = fs::exists(pipeline_path, ec) && !ec;
  if (!has_pipeline_sequence) {
    const std::vector<SequenceEntry> mla_only = synthesize_mla_only_sequence(etc_dir);
    if (!mla_only.empty()) {
      static std::mutex warn_mu;
      static std::unordered_set<std::string> warned;
      {
        std::lock_guard<std::mutex> lock(warn_mu);
        if (warned.insert(etc_dir).second) {
          std::cerr << "[WARN] pipeline_sequence missing: assuming MLA-only sequence for etc_dir="
                    << etc_dir << "\n";
        }
      }
      out.sequence = mla_only;
      out.source = SequenceLoadSource::MissingSequenceAssumeMlaOnly;
      out.strict_error = "schema_error: required pipeline_sequence.json is missing";
      return out;
    }
    throw std::runtime_error("schema_error: required pipeline_sequence.json is missing");
  }

  try {
    out.sequence = load_pipeline_sequence_strict(etc_dir);
    out.source = SequenceLoadSource::Strict;
    out.strict_error.clear();
    return out;
  } catch (const std::exception& strict_err) {
    const std::vector<SequenceEntry> mla_only = synthesize_mla_only_sequence(etc_dir);
    if (!mla_only.empty()) {
      static std::mutex warn_mu;
      static std::unordered_set<std::string> warned;
      const std::string key = etc_dir + "|" + strict_err.what();
      {
        std::lock_guard<std::mutex> lock(warn_mu);
        if (warned.insert(key).second) {
          std::cerr << "[WARN] pipeline_sequence fallback: " << strict_err.what()
                    << " ; using MLA-only fallback for etc_dir=" << etc_dir << "\n";
        }
      }
      out.sequence = mla_only;
      out.source = SequenceLoadSource::MissingSequenceAssumeMlaOnly;
      out.strict_error = strict_err.what();
      return out;
    }
    throw;
  }
}

std::vector<SequenceEntry> load_pipeline_sequence(const std::string& etc_dir) {
  return load_pipeline_sequence_with_source(etc_dir).sequence;
}

SequenceSplit split_sequence_for_infer(const std::vector<SequenceEntry>& seq) {
  SequenceSplit out;
  if (seq.empty())
    return out;

  std::size_t first_mla = seq.size();
  std::size_t last_mla = seq.size();
  for (std::size_t i = 0; i < seq.size(); ++i) {
    if (!is_mla_stage(seq[i])) {
      continue;
    }
    if (first_mla == seq.size()) {
      first_mla = i;
    }
    last_mla = i;
  }

  if (first_mla != seq.size()) {
    out.pre.insert(out.pre.end(), seq.begin(), seq.begin() + static_cast<long>(first_mla));
    out.infer.insert(out.infer.end(), seq.begin() + static_cast<long>(first_mla),
                     seq.begin() + static_cast<long>(last_mla + 1U));
    out.post.insert(out.post.end(), seq.begin() + static_cast<long>(last_mla + 1U), seq.end());
    return out;
  }

  // No MLA marker in sequence: treat the entire sequence as infer-only topology.
  out.infer = seq;
  return out;
}

} // namespace simaai::neat::mpk
