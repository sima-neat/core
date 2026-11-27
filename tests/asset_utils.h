#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace sima_test {

namespace fs = std::filesystem;

inline std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

inline bool move_to_tmp(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  ec.clear();
  fs::rename(src, dst, ec);
  if (!ec)
    return true;

  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec)
    return false;
  fs::remove(src, ec);
  return true;
}

inline bool is_usable_regular_file(const fs::path& path) {
  if (path.empty())
    return false;

  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec)
    return false;

  ec.clear();
  const auto size = fs::file_size(path, ec);
  if (ec || size == 0)
    return false;

  std::ifstream stream(path, std::ios::binary);
  return stream.good();
}

inline bool has_tar_gz_suffix(const fs::path& path) {
  const std::string name = path.filename().string();
  constexpr const char* suffix = ".tar.gz";
  constexpr std::size_t suffix_len = 7;
  return name.size() > suffix_len &&
         name.compare(name.size() - suffix_len, suffix_len, suffix) == 0;
}

// Defensive: returns false when `tar -tzf` cannot list the archive (file is
// truncated, gzip CRC is wrong, content is HTML from a failed download, etc.).
// It rejects a corrupt cached archive that `is_usable_regular_file` (size > 0
// only) lets through, which would otherwise surface as ModelPack "tar listing
// failed".
inline bool is_listable_tar_gz(const fs::path& tar_path) {
  if (!is_usable_regular_file(tar_path) || !has_tar_gz_suffix(tar_path))
    return false;
  const std::string cmd = "tar -tzf " + shell_quote(tar_path.string()) + " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

inline bool tar_contains_strict_mpk_json(const fs::path& tar_path) {
  if (!is_usable_regular_file(tar_path) || !has_tar_gz_suffix(tar_path))
    return false;

  const std::string cmd = "tar -tzf " + shell_quote(tar_path.string()) +
                          " 2>/dev/null | grep -E '(^|/)[^/]+_mpk\\.json$' >/dev/null";
  return std::system(cmd.c_str()) == 0;
}

inline bool is_strict_mpk_tar_gz(const fs::path& tar_path) {
  return tar_contains_strict_mpk_json(tar_path);
}

struct TestRuntimePaths {
  fs::path manifest_path;
  fs::path build_root;
  fs::path source_root;
  fs::path shared_test_asset_root;
  fs::path repo_tmp_root;
  fs::path model_archive_fixture_root;
  fs::path model_archive_fixture_manifest;
  fs::path decoder_fixture;
  fs::path codec_perf_h264_fixture;
  fs::path codec_perf_h265_fixture;
};

inline void append_unique_path(std::vector<fs::path>& out, const fs::path& path) {
  if (path.empty())
    return;
  for (const auto& existing : out) {
    if (existing == path)
      return;
  }
  out.push_back(path);
}

inline std::vector<fs::path> runtime_search_roots() {
  std::vector<fs::path> roots;

  std::error_code ec;
  append_unique_path(roots, fs::current_path(ec));

  ec.clear();
  const fs::path exe_path = fs::read_symlink("/proc/self/exe", ec);
  if (!ec && !exe_path.empty()) {
    append_unique_path(roots, exe_path.parent_path());
  }

  append_unique_path(roots, fs::path(__FILE__).parent_path().parent_path());
  return roots;
}

inline fs::path find_runtime_manifest_path() {
  std::error_code ec;
  for (const auto& seed : runtime_search_roots()) {
    fs::path cur = seed;
    while (!cur.empty()) {
      const fs::path candidate = cur / "test-fixtures" / "runtime_manifest.json";
      if (is_usable_regular_file(candidate))
        return candidate;
      const fs::path parent = cur.parent_path();
      if (parent == cur)
        break;
      cur = parent;
    }
  }
  return {};
}

inline std::optional<std::string> json_string_field(const std::string& text,
                                                    const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos)
    return std::nullopt;

  const std::size_t colon_pos = text.find(':', key_pos + needle.size());
  if (colon_pos == std::string::npos)
    return std::nullopt;

  const std::size_t quote_pos = text.find('"', colon_pos + 1);
  if (quote_pos == std::string::npos)
    return std::nullopt;

  std::string value;
  bool escaped = false;
  for (std::size_t i = quote_pos + 1; i < text.size(); ++i) {
    const char c = text[i];
    if (escaped) {
      value.push_back(c);
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"')
      return value;
    value.push_back(c);
  }
  return std::nullopt;
}

inline fs::path resolve_manifest_relative_path(const std::string& text, const std::string& key,
                                               const fs::path& build_root) {
  const std::optional<std::string> rel = json_string_field(text, key);
  if (!rel.has_value())
    return {};
  if (rel->empty())
    return build_root.lexically_normal();
  const fs::path raw(*rel);
  if (raw.is_absolute())
    return raw;
  return (build_root / raw).lexically_normal();
}

inline fs::path discover_source_root_from_runtime() {
  std::error_code ec;
  for (const auto& seed : runtime_search_roots()) {
    fs::path cur = seed;
    while (!cur.empty()) {
      if (fs::exists(cur / "CMakeLists.txt", ec) && !ec && fs::exists(cur / "src", ec) && !ec &&
          fs::exists(cur / "tests", ec) && !ec) {
        return cur;
      }
      ec.clear();
      const fs::path parent = cur.parent_path();
      if (parent == cur)
        break;
      cur = parent;
    }
  }
  return fs::path(__FILE__).parent_path().parent_path();
}

inline const TestRuntimePaths& test_runtime_paths() {
  static const TestRuntimePaths paths = [] {
    TestRuntimePaths value;
    value.manifest_path = find_runtime_manifest_path();
    if (!value.manifest_path.empty()) {
      value.build_root = value.manifest_path.parent_path().parent_path();
      std::ifstream in(value.manifest_path, std::ios::binary);
      if (in.is_open()) {
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        value.source_root =
            resolve_manifest_relative_path(text, "source_root_rel", value.build_root);
        value.shared_test_asset_root =
            resolve_manifest_relative_path(text, "shared_test_asset_root_rel", value.build_root);
        value.repo_tmp_root =
            resolve_manifest_relative_path(text, "repo_tmp_root_rel", value.build_root);
        value.model_archive_fixture_root = resolve_manifest_relative_path(
            text, "model_archive_fixture_root_rel", value.build_root);
        value.model_archive_fixture_manifest = resolve_manifest_relative_path(
            text, "model_archive_fixture_manifest_rel", value.build_root);
        value.decoder_fixture =
            resolve_manifest_relative_path(text, "decoder_fixture_rel", value.build_root);
        value.codec_perf_h264_fixture =
            resolve_manifest_relative_path(text, "codec_perf_h264_fixture_rel", value.build_root);
        value.codec_perf_h265_fixture =
            resolve_manifest_relative_path(text, "codec_perf_h265_fixture_rel", value.build_root);
      }
    }

    if (value.source_root.empty())
      value.source_root = discover_source_root_from_runtime();
    if (value.shared_test_asset_root.empty() && !value.build_root.empty()) {
      std::error_code ec;
      const fs::path installed_share_assets =
          (value.build_root / ".." / ".." / "share" / "sima-neat" / "test-assets")
              .lexically_normal();
      if (fs::is_directory(installed_share_assets, ec) && !ec)
        value.shared_test_asset_root = installed_share_assets;
    }
    if (value.shared_test_asset_root.empty())
      value.shared_test_asset_root = value.source_root;
    if (value.repo_tmp_root.empty())
      value.repo_tmp_root = value.source_root / "tmp";
    if (value.model_archive_fixture_root.empty())
      value.model_archive_fixture_root =
          value.source_root / "build" / "test-assets" / "model-archive";
    if (value.model_archive_fixture_manifest.empty())
      value.model_archive_fixture_manifest =
          value.model_archive_fixture_root / "fixtures_manifest.json";
    if (value.decoder_fixture.empty())
      value.decoder_fixture =
          value.source_root / "tests" / "assets" / "decoder" / "dynamic_caps.h264";
    if (value.codec_perf_h264_fixture.empty())
      value.codec_perf_h264_fixture = value.source_root / "build" / "test-assets" / "codec-perf" /
                                      "h264_1280x720_30fps_no_sei.h264";
    if (value.codec_perf_h265_fixture.empty())
      value.codec_perf_h265_fixture = value.source_root / "build" / "test-assets" / "codec-perf" /
                                      "h265_1280x720_30fps_no_sei.h265";

    return value;
  }();
  return paths;
}

inline fs::path test_source_root() {
  return test_runtime_paths().source_root;
}

inline fs::path test_shared_asset_root() {
  return test_runtime_paths().shared_test_asset_root;
}

inline fs::path test_shared_asset_path(const fs::path& rel) {
  if (rel.empty())
    return test_shared_asset_root();
  if (rel.is_absolute())
    return rel.lexically_normal();
  return (test_shared_asset_root() / rel).lexically_normal();
}

inline fs::path test_image_fixture_path() {
  return test_shared_asset_path("test.jpg");
}

inline fs::path test_tmp_root() {
  return test_runtime_paths().repo_tmp_root;
}

inline fs::path test_model_archive_fixture_root_path() {
  return test_runtime_paths().model_archive_fixture_root;
}

inline fs::path test_model_archive_fixture_manifest_path() {
  return test_runtime_paths().model_archive_fixture_manifest;
}

inline fs::path test_decoder_fixture_path() {
  return test_runtime_paths().decoder_fixture;
}

inline fs::path test_codec_perf_h264_fixture_path() {
  return test_runtime_paths().codec_perf_h264_fixture;
}

inline fs::path test_codec_perf_h265_fixture_path() {
  return test_runtime_paths().codec_perf_h265_fixture;
}

inline fs::path default_asset_root(const fs::path& root_in = {}) {
  return root_in.empty() ? test_source_root() : root_in;
}

inline bool ensure_writable_directory(const fs::path& dir) {
  if (dir.empty())
    return false;

  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec)
    return false;

  const fs::path probe =
      dir / (".sima-neat-write-test-" + std::to_string(static_cast<long long>(::getpid())));
  {
    std::ofstream out(probe, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
      return false;
    out << "ok\n";
  }
  fs::remove(probe, ec);
  return true;
}

inline bool download_file(const std::string& url, const fs::path& out_path) {
  if (is_usable_regular_file(out_path))
    return true;

  std::error_code ec;
  fs::create_directories(out_path.parent_path(), ec);
  if (ec) {
    return false;
  }

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());

  // Public URLs only. curl and wget carry no Developer Portal session.
  std::string cmd = "curl -L --fail --silent --show-error -o " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0)
    return true;

  cmd = "wget -O " + qout + " " + qurl;
  if (std::system(cmd.c_str()) == 0)
    return true;

  std::error_code rm_ec;
  fs::remove(out_path, rm_ec);
  return false;
}

inline fs::path default_goldfish_path() {
  try {
    return fs::temp_directory_path() / "sima_imagenet_goldfish.jpg";
  } catch (...) {
    return fs::path("tmp") / "imagenet_goldfish.jpg";
  }
}

inline std::string env_existing_model_tar_path(const char* specific_env) {
  if (specific_env && *specific_env) {
    const char* specific = std::getenv(specific_env);
    if (specific && *specific && is_usable_regular_file(specific)) {
      return std::string(specific);
    }
  }
  const char* generic = std::getenv("SIMA_MODEL_TAR");
  if (generic && *generic && is_usable_regular_file(generic)) {
    return std::string(generic);
  }
  return {};
}

inline std::string resolve_resnet50_tar_local_only(const fs::path& root_in = {}) {
  const fs::path root = default_asset_root(root_in);
  const std::string env_tar = env_existing_model_tar_path("SIMA_RESNET50_TAR");
  if (!env_tar.empty()) {
    return env_tar;
  }

  const fs::path local = root / "tmp" / "resnet_50_mpk.tar.gz";
  if (is_usable_regular_file(local))
    return local.string();

  const std::vector<fs::path> candidates = {
      root / "resnet_50_mpk.tar.gz",
      root / "resnet-50_mpk.tar.gz",
      fs::current_path() / "resnet_50_mpk.tar.gz",
      fs::current_path() / "resnet-50_mpk.tar.gz",
      fs::temp_directory_path() / "resnet_50_mpk.tar.gz",
      fs::temp_directory_path() / "resnet-50_mpk.tar.gz",
  };
  for (const auto& candidate : candidates) {
    if (is_usable_regular_file(candidate) && move_to_tmp(candidate, local)) {
      return local.string();
    }
  }

  return "";
}

// CI prepares the Model Browser resnet_50 w8a8 MPK and names it in SIMA_RESNET50_TAR. An empty
// result means the variable is unset or names something unusable; the reason goes to stderr.
inline std::string resolve_resnet50_tar() {
  for (const char* name : {"SIMA_RESNET50_TAR", "SIMA_MODEL_TAR"}) {
    const char* value = std::getenv(name);
    if (!value || !*value)
      continue;
    if (is_usable_regular_file(value))
      return value;
    std::cerr << name << "=" << value << " is not a non-empty file\n";
    return "";
  }
  std::cerr << "SIMA_RESNET50_TAR is not set. Set it, or SIMA_MODEL_TAR, to the "
               "resnet_50_w8a8_mpk.tar.gz from 'sima-cli models download --id resnet_50 "
               "--variant w8a8'.\n";
  return "";
}

// For the stress test in an installed tree: the variables, then the one fixed cache path.
// No folder search and no download.
inline std::string resolve_yolov8s_tar_local_only(const fs::path& root_in = {}) {
  const std::string env_tar = env_existing_model_tar_path("SIMA_YOLO_TAR");
  if (!env_tar.empty())
    return env_tar;
  const fs::path local = default_asset_root(root_in) / "tmp" / "yolo_v8s_mpk.tar.gz";
  if (is_usable_regular_file(local) && is_listable_tar_gz(local))
    return local.string();
  return "";
}

inline std::string resolve_yolov8s_tar(const fs::path& = {}) {
  for (const char* name : {"SIMA_YOLO_TAR", "SIMA_MODEL_TAR"}) {
    const char* value = std::getenv(name);
    if (!value || !*value)
      continue;
    if (is_usable_regular_file(value))
      return value;
    std::cerr << name << "=" << value << " is not a non-empty file\n";
    return "";
  }
  std::cerr << "SIMA_YOLO_TAR is not set. Set it, or SIMA_MODEL_TAR, to the "
               "yolov8s_w8a8_mpk.tar.gz from 'sima-cli models download --id yolov8s "
               "--variant w8a8'.\n";
  return "";
}

// The same variables as resolve_yolov8s_tar(), and the archive must contain a *_mpk.json.
inline std::string resolve_yolov8s_strict_mpk_tar(const fs::path& root = {}) {
  const std::string tar = resolve_yolov8s_tar(root);
  if (tar.empty() || is_strict_mpk_tar_gz(tar))
    return tar;
  std::cerr << tar << " is not a .tar.gz MPK with a *_mpk.json\n";
  return "";
}

inline fs::path ensure_coco_sample(const fs::path& root_in = {}) {
  const fs::path root = default_asset_root(root_in);
  const char* url_env = std::getenv("SIMA_COCO_URL");
  const std::string url =
      (url_env && *url_env)
          ? std::string(url_env)
          : "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";

  // Primary destination matches the historical layout (caller looks here
  // first via resolve_people_image_path). Falls back to writable scratch
  // locations when the primary parent can't be created — this happens on
  // CI when `root` is a build-host path that doesn't exist on the test
  // runner, so curl/wget bail with "Failure writing output to destination"
  // even though the URL resolves cleanly and the test would otherwise pass.
  const fs::path primary = root / "tmp" / "coco_sample.jpg";
  if (ensure_writable_directory(primary.parent_path()) && download_file(url, primary))
    return primary;

  std::error_code ec;
  fs::path temp_root;
  try {
    temp_root = fs::temp_directory_path();
  } catch (...) {
    temp_root = fs::path("/tmp");
  }
  const std::vector<fs::path> fallbacks = {
      temp_root / "sima-coco-sample" / "coco_sample.jpg",
      fs::path("/tmp") / "sima-coco-sample" / "coco_sample.jpg",
  };
  for (const auto& candidate : fallbacks) {
    ec.clear();
    fs::create_directories(candidate.parent_path(), ec);
    if (download_file(url, candidate))
      return candidate;
  }
  return {};
}

} // namespace sima_test
