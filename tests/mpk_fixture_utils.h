#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sima_test {

namespace fs = std::filesystem;

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

inline std::string make_temp_dir(const std::string& tag) {
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
  const std::string root = make_temp_dir(tag);
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

inline MpkFixture make_malformed_mpk_tar_fixture(const std::string& tag) {
  return make_mpk_tar_fixture(
      tag,
      {
          {"etc/bad.json", "{ \"node_name\": \"bad\", \"input_buffers\": [ "},
      },
      true);
}

} // namespace sima_test
