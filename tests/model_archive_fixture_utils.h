#pragma once

#include "asset_utils.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sima_test {

struct ModelArchiveFixture {
  std::string root_dir;
  std::string tar_path;
};

inline std::string model_archive_shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

inline std::string make_fixture_temp_dir(const std::string& tag) {
  fs::path base = fs::temp_directory_path() / "sima_neat_model_archive_fixtures";
  std::error_code ec;
  fs::create_directories(base, ec);

  std::string templ = (base / (tag + "_XXXXXX")).string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');
  char* out = ::mkdtemp(buf.data());
  if (!out) {
    throw std::runtime_error("model_archive_fixture_utils: mkdtemp failed");
  }
  return std::string(out);
}

inline void write_text_file(const fs::path& path, const std::string& text) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("model_archive_fixture_utils: failed to open file for write: " +
                             path.string());
  }
  out << text;
}

inline void write_binary_file(const fs::path& path, const std::vector<unsigned char>& bytes) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("model_archive_fixture_utils: failed to open file for write: " +
                             path.string());
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

inline ModelArchiveFixture
make_model_archive_fixture(const std::string& tag,
                           const std::vector<std::pair<std::string, std::string>>& text_files,
                           bool include_placeholder_elf = true) {
  const std::string root = make_fixture_temp_dir(tag);
  const fs::path root_path(root);

  for (const auto& entry : text_files) {
    write_text_file(root_path / entry.first, entry.second);
  }

  if (include_placeholder_elf) {
    write_binary_file(root_path / "share" / "placeholder.elf",
                      std::vector<unsigned char>{0x7f, 'E', 'L', 'F', 0x02, 0x01, 0x01});
  }

  const fs::path tar_path = root_path.parent_path() / (tag + ".tar.gz");
  const std::string cmd = "tar -czf " + model_archive_shell_quote(tar_path.string()) + " -C " +
                          model_archive_shell_quote(root_path.string()) + " .";
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("model_archive_fixture_utils: failed to create tar fixture");
  }

  return ModelArchiveFixture{.root_dir = root_path.string(), .tar_path = tar_path.string()};
}

inline std::string read_first_mpk_json_from_tar(const std::string& tar_path) {
  const std::string list_cmd = "tar -tzf " + model_archive_shell_quote(tar_path);
  FILE* list_pipe = ::popen(list_cmd.c_str(), "r");
  if (!list_pipe) {
    throw std::runtime_error("model_archive_fixture_utils: failed to list modelzoo tar: " +
                             tar_path);
  }

  std::string selected;
  char line[4096];
  while (std::fgets(line, sizeof(line), list_pipe)) {
    std::string entry(line);
    while (!entry.empty() && (entry.back() == '\n' || entry.back() == '\r')) {
      entry.pop_back();
    }
    if (selected.empty() && entry.size() >= 9 && entry.rfind("_mpk.json") == entry.size() - 9) {
      selected = entry;
    }
  }
  const int list_rc = ::pclose(list_pipe);
  if (list_rc != 0) {
    throw std::runtime_error("model_archive_fixture_utils: failed to list modelzoo tar: " +
                             tar_path);
  }
  if (selected.empty()) {
    throw std::runtime_error("model_archive_fixture_utils: no *_mpk.json found in modelzoo tar: " +
                             tar_path);
  }

  const std::string read_cmd = "tar -xOzf " + model_archive_shell_quote(tar_path) + " -- " +
                               model_archive_shell_quote(selected);
  FILE* read_pipe = ::popen(read_cmd.c_str(), "r");
  if (!read_pipe) {
    throw std::runtime_error("model_archive_fixture_utils: failed to read " + selected +
                             " from modelzoo tar: " + tar_path);
  }

  std::ostringstream out;
  while (std::fgets(line, sizeof(line), read_pipe)) {
    out << line;
  }
  const int read_rc = ::pclose(read_pipe);
  if (read_rc != 0) {
    throw std::runtime_error("model_archive_fixture_utils: failed to read " + selected +
                             " from modelzoo tar: " + tar_path);
  }
  return out.str();
}

inline fs::path repo_root_for_modelzoo() {
  std::error_code ec;
  fs::path cur = fs::current_path(ec);
  if (ec) {
    return fs::current_path();
  }
  while (!cur.empty()) {
    if (fs::exists(cur / "tests", ec) && fs::exists(cur / "CMakeLists.txt", ec) && !ec) {
      return cur;
    }
    const fs::path parent = cur.parent_path();
    if (parent == cur) {
      break;
    }
    cur = parent;
  }
  return fs::current_path();
}

inline std::string fixture_file_sha256(const fs::path& path) {
  const std::string command = "sha256sum -- " + model_archive_shell_quote(path.string());
  FILE* pipe = ::popen(command.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("model_archive_fixture_utils: failed to run sha256sum for " +
                             path.string());
  }
  char line[256]{};
  const bool read_ok = std::fgets(line, sizeof(line), pipe) != nullptr;
  const int rc = ::pclose(pipe);
  std::string digest;
  if (read_ok) {
    for (std::size_t i = 0; i < 64U && std::isxdigit(static_cast<unsigned char>(line[i])); ++i) {
      digest.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(line[i]))));
    }
  }
  if (rc != 0 || digest.size() != 64U) {
    throw std::runtime_error("model_archive_fixture_utils: sha256sum failed for " +
                             path.string());
  }
  return digest;
}

inline const std::array<const char*, 10>& exact_yolo_v9c_seg_leaf_names() {
  static const std::array<const char*, 10> names = {
      "dequantize_2/bbox_0",       "dequantize_3/bbox_1",
      "dequantize_4/bbox_2",       "dequantize_5/class_prob_0",
      "dequantize_6/class_prob_1", "dequantize_7/class_prob_2",
      "dequantize_8/mask_coeff_0", "dequantize_9/mask_coeff_1",
      "dequantize_10/mask_coeff_2", "dequantize_11/mask",
  };
  return names;
}

inline void validate_exact_yolo_v9c_seg_seed_json(const std::string& text,
                                                   const fs::path& source_path) {
  const auto doc = nlohmann::json::parse(text);
  if (doc.value("name", std::string{}) != "yolo_v9c_seg" ||
      !doc.contains("plugins") || !doc["plugins"].is_array()) {
    throw std::runtime_error(
        "model_archive_fixture_utils: strict yolo_v9c_seg seed has wrong model identity: " +
        source_path.string());
  }

  const nlohmann::json* unpack = nullptr;
  const nlohmann::json* pass_through = nullptr;
  std::vector<std::string> semantic_leaves;
  for (const auto& plugin : doc["plugins"]) {
    const std::string name = plugin.value("name", std::string{});
    if (name == "MLA_0_ofm_unpack_transform") {
      unpack = &plugin;
    } else if (name == "PassThrough") {
      pass_through = &plugin;
    }
    if (!plugin.contains("output_nodes") || !plugin["output_nodes"].is_array()) {
      continue;
    }
    for (const auto& output : plugin["output_nodes"]) {
      const std::string output_name = output.value("name", std::string{});
      if (output_name.find("/bbox_") != std::string::npos ||
          output_name.find("/class_prob_") != std::string::npos ||
          output_name.find("/mask_coeff_") != std::string::npos ||
          output_name == "dequantize_11/mask") {
        semantic_leaves.push_back(output_name);
      }
    }
  }
  if (!unpack || !unpack->contains("output_nodes") ||
      !(*unpack)["output_nodes"].is_array() || (*unpack)["output_nodes"].size() != 10U) {
    throw std::runtime_error(
        "model_archive_fixture_utils: strict yolo_v9c_seg seed must have one exact 10-way "
        "MLA unpack");
  }
  for (std::size_t i = 0; i < 10U; ++i) {
    const std::string expected = "MLA_0_ofm_unpack_transform_" + std::to_string(i);
    if ((*unpack)["output_nodes"][i].value("name", std::string{}) != expected) {
      throw std::runtime_error(
          "model_archive_fixture_utils: strict yolo_v9c_seg unpack order is not exact");
    }
  }

  const auto& expected_leaves = exact_yolo_v9c_seg_leaf_names();
  if (semantic_leaves.size() != expected_leaves.size()) {
    throw std::runtime_error(
        "model_archive_fixture_utils: strict yolo_v9c_seg seed must have exactly ten semantic "
        "leaves");
  }
  for (std::size_t i = 0; i < expected_leaves.size(); ++i) {
    if (semantic_leaves[i] != expected_leaves[i]) {
      throw std::runtime_error(
          "model_archive_fixture_utils: strict yolo_v9c_seg semantic leaf order is not exact");
    }
  }
  if (!pass_through || !pass_through->contains("input_nodes") ||
      !(*pass_through)["input_nodes"].is_array() ||
      (*pass_through)["input_nodes"].size() != expected_leaves.size() ||
      !pass_through->contains("output_nodes") ||
      !(*pass_through)["output_nodes"].is_array() ||
      (*pass_through)["output_nodes"].size() != expected_leaves.size()) {
    throw std::runtime_error(
        "model_archive_fixture_utils: strict yolo_v9c_seg seed must end in one exact 10-way "
        "PassThrough");
  }
  for (std::size_t i = 0; i < expected_leaves.size(); ++i) {
    if ((*pass_through)["input_nodes"][i].value("name", std::string{}) != expected_leaves[i]) {
      throw std::runtime_error(
          "model_archive_fixture_utils: strict yolo_v9c_seg PassThrough binding order is not "
          "exact");
    }
  }
}

inline std::pair<std::string, std::string>
strict_contract_json_entry_from_modelzoo(const std::string& model_name = "yolo_v9c_seg") {
  static std::mutex cache_mutex;
  static std::map<std::string, std::pair<std::string, std::string>> cache;
  std::lock_guard<std::mutex> guard(cache_mutex);
  if (const auto it = cache.find(model_name); it != cache.end()) {
    return it->second;
  }
  if (model_name != "yolo_v9c_seg") {
    throw std::runtime_error("model_archive_fixture_utils: no isolated strict seed for model '" +
                             model_name + "'");
  }

  const fs::path seed_path = test_model_archive_fixture_root_path() / "strict-seeds" /
                             "yolo_v9c_seg_mpk.json";
  constexpr std::uintmax_t expected_size = 47224U;
  constexpr const char* expected_sha256 =
      "bf3c96dd5863446349be7f675c078cfebd8032b658a4ee3355393c5590575f74";
  std::error_code ec;
  if (!fs::is_regular_file(seed_path, ec) || ec || fs::file_size(seed_path, ec) != expected_size ||
      ec) {
    throw std::runtime_error(
        "model_archive_fixture_utils: exact yolo_v9c_seg strict seed is missing or has the "
        "wrong size: " +
        seed_path.string());
  }
  if (fixture_file_sha256(seed_path) != expected_sha256) {
    throw std::runtime_error(
        "model_archive_fixture_utils: exact yolo_v9c_seg strict seed checksum mismatch: " +
        seed_path.string());
  }
  std::ifstream input(seed_path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  validate_exact_yolo_v9c_seg_seed_json(text, seed_path);
  auto inserted = cache.emplace(
      model_name, std::make_pair(std::string("etc/strict_seed_mpk.json"), std::move(text)));
  return inserted.first->second;
}

inline void require_exact_yolo_v9c_seg_parsed_contract(
    const simaai::neat::internal::ModelPack& pack) {
  const auto& parsed = pack.mpk_contract();
  if (!parsed.has_value() || parsed->model_name != "yolo_v9c_seg") {
    throw std::runtime_error(
        "model_archive_fixture_utils: parsed fixture is not the exact yolo_v9c_seg seed");
  }
  const auto logical_outputs =
      simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract(*parsed);
  if (logical_outputs.size() != 10U) {
    throw std::runtime_error(
        "model_archive_fixture_utils: parsed yolo_v9c_seg fixture must expose ten logical "
        "outputs");
  }
  const auto* unpack =
      simaai::neat::pipeline_internal::sima::get_mla_unpack_stage_io_contract(*parsed);
  if (!unpack || unpack->output_tensors.size() != 10U) {
    throw std::runtime_error(
        "model_archive_fixture_utils: parsed yolo_v9c_seg fixture must preserve its 10-way "
        "unpack");
  }
}

inline ModelArchiveFixture make_strict_model_archive_fixture(
    const std::string& tag, const std::vector<std::pair<std::string, std::string>>& text_files,
    bool include_placeholder_elf = true, const std::string& model_name = "yolo_v9c_seg") {
  std::vector<std::pair<std::string, std::string>> files = text_files;
  const auto strict_contract = strict_contract_json_entry_from_modelzoo(model_name);
  files.push_back(strict_contract);
  return make_model_archive_fixture(tag, files, include_placeholder_elf);
}

inline ModelArchiveFixture make_malformed_model_archive_fixture(const std::string& tag) {
  return make_model_archive_fixture(
      tag,
      {
          {"etc/bad_mpk.json", "{ \"node_name\": \"bad\", \"input_buffers\": [ "},
      },
      true);
}

} // namespace sima_test
