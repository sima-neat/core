#pragma once

#include "asset_utils.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdlib>
#include <unistd.h>

namespace sima_test {

inline fs::path repo_root_from_tests() {
  return sima_test::test_source_root();
}

inline fs::path mpk_fixture_root() {
  return sima_test::test_mpk_fixture_root_path();
}

inline fs::path mpk_fixture_path(const std::string& rel) {
  return mpk_fixture_root() / rel;
}

inline std::string make_temp_dir(const std::string& tag) {
  fs::path base = fs::temp_directory_path() / "sima_neat_mpk_tests";
  std::error_code ec;
  fs::create_directories(base, ec);

  std::string templ = (base / (tag + "_XXXXXX")).string();
  std::vector<char> buf(templ.begin(), templ.end());
  buf.push_back('\0');

  char* out = ::mkdtemp(buf.data());
  if (!out) {
    throw std::runtime_error("mpk_test_utils: mkdtemp failed");
  }
  return std::string(out);
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const std::string& value) : key_(key) {
    const char* prev = std::getenv(key_);
    if (prev)
      prev_ = std::string(prev);
    ::setenv(key_, value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (prev_.has_value()) {
      ::setenv(key_, prev_->c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  std::optional<std::string> prev_;
};

} // namespace sima_test
