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

struct MpkFixture {
  std::string root_dir;
  std::string tar_path;
};

inline std::string mpk_shell_quote(const std::string& s) {
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
  fs::path base = fs::temp_directory_path() / "sima_neat_mpk_fixtures";
  std::error_code ec;
  fs::create_directories(base, ec);

  std::string templ = (base / (tag + "_XXXXXX")).string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');
  char* out = ::mkdtemp(buf.data());
  if (!out) {
    throw std::runtime_error("mpk_fixture_utils: mkdtemp failed");
  }
  return std::string(out);
}

inline void write_text_file(const fs::path& path, const std::string& text) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("mpk_fixture_utils: failed to open file for write: " + path.string());
  }
  out << text;
}

inline void write_binary_file(const fs::path& path, const std::vector<unsigned char>& bytes) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("mpk_fixture_utils: failed to open file for write: " + path.string());
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
}

inline MpkFixture
make_mpk_tar_fixture(const std::string& tag,
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
  const std::string cmd = "tar -czf " + mpk_shell_quote(tar_path.string()) + " -C " +
                          mpk_shell_quote(root_path.string()) + " .";
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("mpk_fixture_utils: failed to create tar fixture");
  }

  return MpkFixture{.root_dir = root_path.string(), .tar_path = tar_path.string()};
}

inline std::string read_text_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("mpk_fixture_utils: failed to open file for read: " + path.string());
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
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

inline fs::path resolve_real_mpk_json_from_modelzoo(const fs::path& root_in = {},
                                                    const std::string& model_name = "yolo_v9c_seg") {
  const fs::path root = root_in.empty() ? repo_root_for_modelzoo() : root_in;

  const std::string tar = sima_test::resolve_modelzoo_tar(model_name, root);
  if (tar.empty() || !fs::exists(tar)) {
    throw std::runtime_error("mpk_fixture_utils: failed to resolve modelzoo tar for '" +
                             model_name + "'");
  }

  const std::string extract_root = make_fixture_temp_dir("real_mpk_seed");
  const fs::path extract_path(extract_root);
  const std::string cmd = "tar -xzf " + mpk_shell_quote(tar) + " -C " + mpk_shell_quote(extract_root);
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("mpk_fixture_utils: failed to extract modelzoo tar: " + tar);
  }

  std::error_code ec;
  fs::recursive_directory_iterator it(extract_path, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    if (!it->is_regular_file())
      continue;
    const std::string name = it->path().filename().string();
    if (name.size() >= 9 && name.rfind("_mpk.json") == name.size() - 9) {
      return it->path();
    }
  }

  throw std::runtime_error("mpk_fixture_utils: no *_mpk.json found in modelzoo tar: " + tar);
}

inline std::pair<std::string, std::string> strict_mpk_json_entry_from_modelzoo() {
  static std::pair<std::string, std::string> cached = []() {
    const fs::path mpk_path =
        resolve_real_mpk_json_from_modelzoo(repo_root_for_modelzoo(), "yolo_v9c_seg");
    return std::make_pair(std::string("etc/strict_seed_mpk.json"), read_text_file(mpk_path));
  }();
  return cached;
}

inline MpkFixture
make_strict_mpk_tar_fixture(const std::string& tag,
                            const std::vector<std::pair<std::string, std::string>>& text_files,
                            bool include_placeholder_elf = true,
                            const std::string& model_name = "yolo_v9c_seg") {
  std::vector<std::pair<std::string, std::string>> files = text_files;
  (void)model_name;
  const auto strict_mpk = strict_mpk_json_entry_from_modelzoo();
  files.push_back(strict_mpk);
  return make_mpk_tar_fixture(tag, files, include_placeholder_elf);
}

inline MpkFixture make_malformed_mpk_tar_fixture(const std::string& tag) {
  return make_mpk_tar_fixture(
      tag,
      {
          {"etc/bad.json", "{ \"node_name\": \"bad\", \"input_buffers\": [ "},
      },
      true);
}

} // namespace sima_test
