#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
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

class ScopedFileLock {
public:
  explicit ScopedFileLock(const fs::path& path) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("failed to create lock directory " + path.parent_path().string() +
                               ": " + ec.message());
    }
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd_ < 0) {
      throw std::runtime_error("failed to open lock file " + path.string());
    }
    while (::flock(fd_, LOCK_EX) != 0) {
      if (errno != EINTR) {
        const int err = errno;
        (void)::close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to lock " + path.string() + ": " + std::to_string(err));
      }
    }
  }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

  ~ScopedFileLock() {
    if (fd_ >= 0) {
      (void)::flock(fd_, LOCK_UN);
      (void)::close(fd_);
    }
  }

private:
  int fd_ = -1;
};

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
// Callers of resolve_yolov8s_tar_*() rely on this to reject corrupt cached
// fixtures that previously slipped through `is_usable_regular_file` (which
// only checks size>0) and surfaced as ModelPack "tar listing failed".
inline bool is_listable_tar_gz(const fs::path& tar_path) {
  if (!is_usable_regular_file(tar_path) || !has_tar_gz_suffix(tar_path))
    return false;
  const std::string cmd = "tar -tzf " + shell_quote(tar_path.string()) + " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

// Remove a corrupt cached tar so the next resolve attempt re-downloads.
inline void purge_unlistable_tar_gz(const fs::path& tar_path) {
  if (tar_path.empty())
    return;
  std::error_code ec;
  if (!fs::is_regular_file(tar_path, ec) || ec)
    return;
  if (is_listable_tar_gz(tar_path))
    return;
  fs::remove(tar_path, ec);
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
      value.codec_perf_h264_fixture =
          value.source_root / "build" / "test-assets" / "codec-perf" / "h264_1280x720_30fps.h264";
    if (value.codec_perf_h265_fixture.empty())
      value.codec_perf_h265_fixture =
          value.source_root / "build" / "test-assets" / "codec-perf" / "h265_1280x720_30fps.h265";

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

inline fs::path writable_asset_tmp_dir(const fs::path& root, const std::string& fallback_name) {
  const fs::path primary = root / "tmp";
  if (ensure_writable_directory(primary))
    return primary;

  fs::path temp_root;
  try {
    temp_root = fs::temp_directory_path();
  } catch (...) {
    temp_root = fs::path("/tmp");
  }

  const std::vector<fs::path> fallbacks = {
      temp_root / fallback_name,
      fs::path("/tmp") / fallback_name,
  };
  for (const auto& candidate : fallbacks) {
    if (ensure_writable_directory(candidate))
      return candidate;
  }

  // Let the original path surface the concrete create/open error to callers.
  return primary;
}

inline int run_modelzoo_get_noninteractive(const std::string& model_name) {
  const std::string cmd =
      "SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli modelzoo get " + shell_quote(model_name);
  return std::system(cmd.c_str());
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

  // Prefer sima-cli when present — it transparently handles the Developer
  // Portal OAuth flow that docs.sima.ai/pkg_downloads/... requires (test
  // fixtures live alongside SDK model downloads behind that auth wall).
  // Falls back to bare curl/wget for public URLs or hosts where sima-cli
  // isn't available.
  if (std::system("command -v sima-cli >/dev/null 2>&1") == 0) {
    const std::string qdest = shell_quote(out_path.parent_path().string());
    const std::string sima_cmd = "SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli download --dest " + qdest +
                                 " " + qurl + " >/dev/null 2>&1";
    if (std::system(sima_cmd.c_str()) == 0 && is_usable_regular_file(out_path)) {
      return true;
    }
  }

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

inline std::string resolve_resnet50_tar(const fs::path& root_in = {}) {
  const fs::path root = default_asset_root(root_in);
  const std::string env_tar = env_existing_model_tar_path("SIMA_RESNET50_TAR");
  if (!env_tar.empty()) {
    return env_tar;
  }

  const fs::path local = root / "tmp" / "resnet_50_mpk.tar.gz";
  const ScopedFileLock lock(local.string() + ".lock");
  if (is_usable_regular_file(local))
    return local.string();

  const int rc = run_modelzoo_get_noninteractive("resnet_50");
  if (rc != 0)
    return "";

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

inline std::string resolve_yolov8s_tar_local_first(const fs::path& root_in, bool skip_download) {
  const fs::path root = default_asset_root(root_in);
  const fs::path root_tmp = root / "tmp";
  const fs::path tmp_dir = writable_asset_tmp_dir(root, "sima-yolov8s-cache");
  const fs::path tmp_tar = tmp_dir / "yolo_v8s_mpk.tar.gz";
  const ScopedFileLock lock(tmp_tar.string() + ".lock");

  // Drop corrupt cached tars up front so the subsequent search/download
  // path can refresh them. is_usable_regular_file only checks size>0;
  // a partial download or HTML error page satisfies that but fails
  // `tar -tzf` and surfaces deep in ModelPack as "tar listing failed".
  //
  // NB: we intentionally do NOT apply is_listable_tar_gz to the broader
  // search-loop candidates below — unit_asset_utils_readable_model_tar_test
  // pre-stages dummy non-tar payloads to validate move_to_tmp semantics,
  // and downstream callers handle real corruption via the cached-tar
  // validation they perform on the path we return.
  purge_unlistable_tar_gz(tmp_tar);
  purge_unlistable_tar_gz(root / "yolo_v8s_mpk.tar.gz");
  purge_unlistable_tar_gz(root_tmp / "yolo_v8s_mpk.tar.gz");

  const std::string env_tar = env_existing_model_tar_path("SIMA_YOLO_TAR");
  if (!env_tar.empty()) {
    return env_tar;
  }

  const fs::path direct_tar = root / "yolo_v8s_mpk.tar.gz";
  if (is_usable_regular_file(direct_tar))
    return direct_tar.string();

  const fs::path root_tmp_tar = root_tmp / "yolo_v8s_mpk.tar.gz";
  if (root_tmp_tar != tmp_tar && is_usable_regular_file(root_tmp_tar)) {
    if (move_to_tmp(root_tmp_tar, tmp_tar))
      return tmp_tar.string();
    return root_tmp_tar.string();
  }

  if (is_usable_regular_file(tmp_tar))
    return tmp_tar.string();

  auto stage_candidate = [&](const fs::path& candidate) -> std::string {
    if (!is_usable_regular_file(candidate))
      return "";
    if (candidate == tmp_tar)
      return tmp_tar.string();
    if (move_to_tmp(candidate, tmp_tar))
      return tmp_tar.string();
    return candidate.string();
  };

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  fs::path temp_root;
  try {
    temp_root = fs::temp_directory_path();
  } catch (...) {
    temp_root = fs::path("/tmp");
  }
  const std::vector<fs::path> search_dirs = {
      root,
      fs::current_path(),
      root_tmp,
      tmp_dir,
      temp_root,
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  const std::vector<std::string> names = {
      "yolo_v8s_mpk.tar.gz",
      "yolo-v8s_mpk.tar.gz",
      "yolov8s_mpk.tar.gz",
      "yolov8_s_mpk.tar.gz",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      const std::string staged = stage_candidate(dir / name);
      if (!staged.empty())
        return staged;
    }
  }

  if (!skip_download) {
    const int rc = run_modelzoo_get_noninteractive("yolo_v8s");
    if (rc == 0 && is_usable_regular_file(tmp_tar)) {
      // After a real download we *do* validate — the modelzoo path is
      // the one that has historically produced partial/corrupt tarballs
      // that fail `tar -tzf` and surface deep in ModelPack.
      if (is_listable_tar_gz(tmp_tar))
        return tmp_tar.string();
      purge_unlistable_tar_gz(tmp_tar);
    }
  }

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      fs::path candidate = dir / name;
      if (is_usable_regular_file(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  // First, try the explicit sima-cli internal-mode layout. When sima-cli
  // detects board mode (Environment: board (modalix)) it routes through
  // _download_model_internal, which lands the tarball at
  //   <getcwd()>/<model_name>/<filename>
  // (see /usr/local/lib/python3.11/dist-packages/sima_cli/model_zoo/model.py
  //  around `dest_dir = os.path.join(os.getcwd(), selected_model.split("/")[-1])`).
  // None of the directories in the broader search loop above check this
  // per-model subdir form, so the CI runner — which detects board mode
  // because /etc/build advertises SIMA_BUILD_VERSION+MACHINE — trips the
  // SKIP path after a perfectly successful sima-cli download. Add the exact
  // subdir as an explicit candidate before falling back to recursive walks.
  const std::vector<fs::path> board_mode_subdir_roots = {
      root,
      fs::current_path(),
  };
  for (const auto& base : board_mode_subdir_roots) {
    if (base.empty())
      continue;
    const fs::path subdir = base / "yolo_v8s";
    for (const auto& name : names) {
      const fs::path candidate = subdir / name;
      const std::string staged = stage_candidate(candidate);
      if (!staged.empty())
        return staged;
    }
  }

  // Defensive recursive fallback: sima-cli reports "File already exists and is
  // complete: yolo_v8s_mpk.tar.gz" when its own cache has the file, but the
  // exact cache layout (subdir-per-model, environment-specific paths) is
  // not directly observable from the resolver — it can't see sima-cli's
  // stdout. The top-level + per-known-cache-dir scan above misses these
  // sub-cached copies and the test then trips with
  // "Failed to locate yolo_v8s model archive" even though sima-cli succeeded.
  // Walk a bounded set of cache roots one level deep so we recover the file
  // wherever sima-cli's modelzoo layout put it, then move it into tmp_tar so
  // subsequent calls hit the fast path. Skip if a candidate cache root is
  // missing — we don't want this fallback to slow down the common path with
  // nonexistent traversals.
  const std::vector<fs::path> recursive_roots = {
      root,
      fs::current_path(),
      home_path.empty() ? fs::path{} : home_path / ".simaai",
      home_path.empty() ? fs::path{} : home_path / ".sima",
      "/data/simaai",
  };
  for (const auto& base : recursive_roots) {
    if (base.empty())
      continue;
    std::error_code rec_ec;
    if (!fs::is_directory(base, rec_ec) || rec_ec)
      continue;
    fs::recursive_directory_iterator it(base, fs::directory_options::skip_permission_denied,
                                        rec_ec);
    if (rec_ec)
      continue;
    for (const auto& entry : it) {
      if (!entry.is_regular_file(rec_ec) || rec_ec)
        continue;
      const std::string fname = entry.path().filename().string();
      for (const auto& name : names) {
        if (fname != name)
          continue;
        const fs::path& found = entry.path();
        if (!is_usable_regular_file(found))
          continue;
        // Try to stage into tmp_tar so the fast path picks it up next time,
        // but if that fails (e.g. `root` is a synthetic path on the test
        // runner so `root/tmp/` can't be created), fall back to the located
        // path. Either way the caller gets a usable tarball — silently
        // skipping here was the bug behind the "Failed to locate" CI flake
        // when sima-cli's board-mode cache layout puts the file under a
        // per-model subdir (e.g. `<cwd>/yolo_v8s/yolo_v8s_mpk.tar.gz`).
        const std::string staged = stage_candidate(found);
        if (!staged.empty())
          return staged;
      }
    }
  }

  return "";
}

inline std::string resolve_yolov8s_tar(const fs::path& root = {}) {
  return resolve_yolov8s_tar_local_first(root, false);
}

inline std::string resolve_modelzoo_tar(const std::string& model_name,
                                        const fs::path& root_in = {}) {
  const fs::path root = default_asset_root(root_in);
  const fs::path tmp_dir = root / "tmp";
  std::error_code ec;
  fs::create_directories(tmp_dir, ec);

  auto append_unique = [](std::vector<std::string>& out, const std::string& v) {
    if (v.empty())
      return;
    for (const auto& existing : out) {
      if (existing == v)
        return;
    }
    out.push_back(v);
  };

  const std::string base = [&]() {
    const std::size_t pos = model_name.find_last_of('/');
    return (pos == std::string::npos) ? model_name : model_name.substr(pos + 1);
  }();

  std::string flat = model_name;
  for (char& c : flat) {
    if (c == '/')
      c = '_';
  }
  std::string flat_dash = model_name;
  for (char& c : flat_dash) {
    if (c == '/')
      c = '-';
  }

  std::vector<std::string> stems;
  append_unique(stems, base);
  append_unique(stems, flat);
  append_unique(stems, flat_dash);

  const std::size_t initial = stems.size();
  for (std::size_t i = 0; i < initial; ++i) {
    std::string alt = stems[i];
    bool changed = false;
    for (char& c : alt) {
      if (c == '_') {
        c = '-';
        changed = true;
      }
    }
    if (changed)
      append_unique(stems, alt);
  }

  std::vector<std::string> names;
  names.reserve(stems.size() * 3);
  for (const auto& stem : stems) {
    append_unique(names, stem + "_mpk.tar.gz");
    append_unique(names, stem + "-mpk.tar.gz");
    append_unique(names, stem + ".tar.gz");
  }

  const fs::path local = tmp_dir / (base + "_mpk.tar.gz");
  const ScopedFileLock lock(local.string() + ".lock");
  // Only validate listability for the *cached* tmp tar (the path that
  // gets handed back to callers and consumed by ModelPack). Candidates
  // discovered via the search loop below may be unit-test fixtures
  // containing dummy payloads — see unit_asset_utils_readable_model_tar_test.
  purge_unlistable_tar_gz(local);
  if (is_usable_regular_file(local))
    return local.string();

  auto try_candidate = [&](const fs::path& candidate) -> std::string {
    if (!is_usable_regular_file(candidate))
      return "";
    if (candidate == local)
      return local.string();
    if (move_to_tmp(candidate, local))
      return local.string();
    return "";
  };

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      tmp_dir,
      root,
      fs::current_path(),
      fs::temp_directory_path(),
      home_path / ".simaai",
      home_path / ".simaai" / "modelzoo",
      home_path / ".sima" / "modelzoo",
      "/data/simaai/modelzoo",
  };

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      const std::string found = try_candidate(dir / name);
      if (!found.empty())
        return found;
    }
  }

  if (run_modelzoo_get_noninteractive(model_name) != 0)
    return "";

  // Post-download validation: the modelzoo path historically produces
  // partial/corrupt tarballs that pass is_usable_regular_file (size > 0)
  // but fail `tar -tzf`. Purge those so a future retry isn't poisoned.
  if (is_usable_regular_file(local)) {
    if (is_listable_tar_gz(local))
      return local.string();
    purge_unlistable_tar_gz(local);
  }

  for (const auto& dir : search_dirs) {
    if (dir.empty())
      continue;
    for (const auto& name : names) {
      const std::string found = try_candidate(dir / name);
      if (!found.empty())
        return found;
    }
  }

  return "";
}

inline std::string resolve_yolov8s_strict_mpk_tar(const fs::path& root = {}) {
  const std::string local_first = resolve_yolov8s_tar_local_first(root, false);
  if (!local_first.empty() && is_strict_mpk_tar_gz(local_first))
    return local_first;

  const fs::path tmp_tar = default_asset_root(root) / "tmp" / "yolo_v8s_mpk.tar.gz";
  if (is_usable_regular_file(tmp_tar) && !is_strict_mpk_tar_gz(tmp_tar)) {
    std::error_code ec;
    fs::remove(tmp_tar, ec);
  }

  const std::string modelzoo = resolve_modelzoo_tar("yolo_v8s", root);
  if (!modelzoo.empty() && is_strict_mpk_tar_gz(modelzoo))
    return modelzoo;

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
