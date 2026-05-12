#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

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

struct TestRuntimePaths {
  fs::path manifest_path;
  fs::path build_root;
  fs::path source_root;
  fs::path repo_tmp_root;
  fs::path mpk_fixture_root;
  fs::path mpk_fixture_manifest;
  fs::path decoder_fixture;
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
  if (!rel.has_value() || rel->empty())
    return {};
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
        value.repo_tmp_root =
            resolve_manifest_relative_path(text, "repo_tmp_root_rel", value.build_root);
        value.mpk_fixture_root =
            resolve_manifest_relative_path(text, "mpk_fixture_root_rel", value.build_root);
        value.mpk_fixture_manifest =
            resolve_manifest_relative_path(text, "mpk_fixture_manifest_rel", value.build_root);
        value.decoder_fixture =
            resolve_manifest_relative_path(text, "decoder_fixture_rel", value.build_root);
      }
    }

    if (value.source_root.empty())
      value.source_root = discover_source_root_from_runtime();
    if (value.repo_tmp_root.empty())
      value.repo_tmp_root = value.source_root / "tmp";
    if (value.mpk_fixture_root.empty())
      value.mpk_fixture_root = value.source_root / "tests" / "assets" / "mpk";
    if (value.mpk_fixture_manifest.empty())
      value.mpk_fixture_manifest = value.mpk_fixture_root / "fixtures_manifest.json";
    if (value.decoder_fixture.empty())
      value.decoder_fixture =
          value.source_root / "tests" / "assets" / "decoder" / "dynamic_caps.h264";

    return value;
  }();
  return paths;
}

inline fs::path test_source_root() {
  return test_runtime_paths().source_root;
}

inline fs::path test_tmp_root() {
  return test_runtime_paths().repo_tmp_root;
}

inline fs::path test_mpk_fixture_root_path() {
  return test_runtime_paths().mpk_fixture_root;
}

inline fs::path test_mpk_fixture_manifest_path() {
  return test_runtime_paths().mpk_fixture_manifest;
}

inline fs::path test_decoder_fixture_path() {
  return test_runtime_paths().decoder_fixture;
}

inline fs::path default_asset_root(const fs::path& root_in = {}) {
  return root_in.empty() ? test_source_root() : root_in;
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

  const std::string qurl = shell_quote(url);
  const std::string qout = shell_quote(out_path.string());

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
  const fs::path tmp_tar = root / "tmp" / "yolo_v8s_mpk.tar.gz";

  const std::string env_tar = env_existing_model_tar_path("SIMA_YOLO_TAR");
  if (!env_tar.empty()) {
    return env_tar;
  }

  const fs::path direct_tar = root / "yolo_v8s_mpk.tar.gz";
  if (is_usable_regular_file(direct_tar))
    return direct_tar.string();

  if (is_usable_regular_file(tmp_tar))
    return tmp_tar.string();

  const char* home = std::getenv("HOME");
  const fs::path home_path = home ? fs::path(home) : fs::path();
  const std::vector<fs::path> search_dirs = {
      root,
      fs::current_path(),
      root / "tmp",
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
      fs::path candidate = dir / name;
      if (is_usable_regular_file(candidate) && move_to_tmp(candidate, tmp_tar)) {
        return tmp_tar.string();
      }
    }
  }

  if (!skip_download) {
    const int rc = run_modelzoo_get_noninteractive("yolo_v8s");
    if (rc == 0 && is_usable_regular_file(tmp_tar))
      return tmp_tar.string();
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

  if (is_usable_regular_file(local))
    return local.string();

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

inline fs::path ensure_coco_sample(const fs::path& root_in = {}) {
  const fs::path root = default_asset_root(root_in);
  const char* url_env = std::getenv("SIMA_COCO_URL");
  const std::string url =
      (url_env && *url_env)
          ? std::string(url_env)
          : "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";
  const fs::path out_path = root / "tmp" / "coco_sample.jpg";
  if (!download_file(url, out_path))
    return {};
  return out_path;
}

} // namespace sima_test
