#pragma once

#include "asset_utils.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

inline std::pair<std::string, std::string> strict_contract_json_entry_from_modelzoo() {
  static std::pair<std::string, std::string> cached = []() {
    const fs::path root = repo_root_for_modelzoo();
    const std::string tar = sima_test::resolve_modelzoo_tar("yolo_v9c_seg", root);
    if (tar.empty() || !fs::exists(tar)) {
      throw std::runtime_error(
          "model_archive_fixture_utils: failed to resolve modelzoo tar for 'yolo_v9c_seg'");
    }
    return std::make_pair(std::string("etc/strict_seed_mpk.json"),
                          read_first_mpk_json_from_tar(tar));
  }();
  return cached;
}

inline ModelArchiveFixture make_strict_model_archive_fixture(
    const std::string& tag, const std::vector<std::pair<std::string, std::string>>& text_files,
    bool include_placeholder_elf = true, const std::string& model_name = "yolo_v9c_seg") {
  std::vector<std::pair<std::string, std::string>> files = text_files;
  (void)model_name;
  const auto strict_contract = strict_contract_json_entry_from_modelzoo();
  files.push_back(strict_contract);
  return make_model_archive_fixture(tag, files, include_placeholder_elf);
}

inline ModelArchiveFixture make_malformed_model_archive_fixture(const std::string& tag) {
  return make_model_archive_fixture(
      tag,
      {
          {"etc/bad.json", "{ \"node_name\": \"bad\", \"input_buffers\": [ "},
      },
      true);
}

} // namespace sima_test
