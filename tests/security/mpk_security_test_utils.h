#pragma once

#include "mpk/MpKLoader.h"
#include "mpk_test_utils.h"
#include "test_utils.h"

#include <filesystem>
#include <functional>
#include <fstream>
#include <string>

namespace sima_test::mpk_security {

namespace fs = std::filesystem;
using simaai::neat::mpk::error_class_name;
using simaai::neat::mpk::ErrorClass;
using simaai::neat::mpk::MpKError;
using simaai::neat::mpk::MpKLoader;
using simaai::neat::mpk::MpKLoaderOptions;

inline void require_mpk_error(const std::function<void()>& fn, ErrorClass expected,
                              const std::string& context) {
  try {
    fn();
  } catch (const MpKError& e) {
    require(e.code() == expected, context + ": expected " +
                                      std::string(error_class_name(expected)) + " got " +
                                      error_class_name(e.code()));
    require_contains(e.what(), error_class_name(expected),
                     context + ": error message should include taxonomy token");
    return;
  }
  throw std::runtime_error(context + ": expected MpKError");
}

inline void require_fixture_root_exists() {
  const fs::path fixtures = sima_test::mpk_fixture_root();
  require(fs::exists(fixtures),
          "missing MPK fixtures; expected tests/assets/mpk (run tests/tools/make_mpk_fixtures.py)");
}

inline void require_fixture_inspect_and_extract_error(const std::string& rel_fixture_path,
                                                      ErrorClass expected,
                                                      const MpKLoaderOptions& opt = {}) {
  const fs::path fixture = sima_test::mpk_fixture_path(rel_fixture_path);
  const std::string context = "fixture(" + rel_fixture_path + ")";

  require_mpk_error([&]() { (void)MpKLoader::inspect(fixture.string(), opt); }, expected,
                    context + " inspect");

  const std::string extract_root = sima_test::make_temp_dir("mpk_security_extract");
  const fs::path sentinel = fs::path(extract_root).parent_path() / "mpk_security_sentinel.txt";
  {
    std::ofstream out(sentinel, std::ios::binary);
    out << "sentinel";
  }

  require_mpk_error([&]() { (void)MpKLoader::extract(fixture.string(), extract_root, opt); },
                    expected, context + " extract");

  require(fs::exists(sentinel), context + ": sentinel should still exist after failed extraction");

  std::size_t extracted_dirs = 0;
  for (const auto& it : fs::directory_iterator(extract_root)) {
    if (it.is_directory())
      ++extracted_dirs;
  }
  require(extracted_dirs == 0,
          context + ": failed extraction should not leave extracted directories");
}

} // namespace sima_test::mpk_security
