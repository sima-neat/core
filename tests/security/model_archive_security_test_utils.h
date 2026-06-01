#pragma once

#include "model/internal/ModelArchiveLoader.h"
#include "model_archive_test_utils.h"
#include "test_utils.h"

#include <filesystem>
#include <functional>
#include <fstream>
#include <string>

namespace sima_test::model_archive_security {

namespace fs = std::filesystem;
using simaai::neat::internal::model_archive_error_class_name;
using simaai::neat::internal::ModelArchiveError;
using simaai::neat::internal::ModelArchiveErrorClass;
using simaai::neat::internal::ModelArchiveLoader;
using simaai::neat::internal::ModelArchiveLoaderOptions;

inline void require_model_archive_error(const std::function<void()>& fn,
                                        ModelArchiveErrorClass expected,
                                        const std::string& context) {
  try {
    fn();
  } catch (const ModelArchiveError& e) {
    require(e.code() == expected, context + ": expected " +
                                      std::string(model_archive_error_class_name(expected)) +
                                      " got " + model_archive_error_class_name(e.code()));
    require_contains(e.what(), model_archive_error_class_name(expected),
                     context + ": error message should include taxonomy token");
    return;
  }
  throw std::runtime_error(context + ": expected ModelArchiveError");
}

inline void require_fixture_root_exists() {
  const fs::path fixtures = sima_test::model_archive_fixture_root();
  require(fs::exists(fixtures),
          "missing model archive fixtures; run tests/tools/make_model_archive_fixtures.py");
}

inline void require_fixture_inspect_and_extract_error(const std::string& rel_fixture_path,
                                                      ModelArchiveErrorClass expected,
                                                      const ModelArchiveLoaderOptions& opt = {}) {
  const fs::path fixture = sima_test::model_archive_fixture_path(rel_fixture_path);
  const std::string context = "fixture(" + rel_fixture_path + ")";

  require_model_archive_error([&]() { (void)ModelArchiveLoader::inspect(fixture.string(), opt); },
                              expected, context + " inspect");

  const std::string extract_root = sima_test::make_temp_dir("model_archive_security_extract");
  const fs::path sentinel =
      fs::path(extract_root).parent_path() / "model_archive_security_sentinel.txt";
  {
    std::ofstream out(sentinel, std::ios::binary);
    out << "sentinel";
  }

  require_model_archive_error(
      [&]() { (void)ModelArchiveLoader::extract(fixture.string(), extract_root, opt); }, expected,
      context + " extract");

  require(fs::exists(sentinel), context + ": sentinel should still exist after failed extraction");

  std::size_t extracted_dirs = 0;
  for (const auto& it : fs::directory_iterator(extract_root)) {
    if (it.is_directory())
      ++extracted_dirs;
  }
  require(extracted_dirs == 0,
          context + ": failed extraction should not leave extracted directories");
}

} // namespace sima_test::model_archive_security
