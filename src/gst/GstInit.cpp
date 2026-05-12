// src/gst/GstInit.cpp
#include "gst/GstInit.h"

#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/UxLogging.h"

#include <gst/gst.h>
#include <glib.h>

#include <mutex>
#include <cstring>
#include <cstdlib>
#include <string>
#include <string_view>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>
#include <limits.h>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace simaai::neat {
using pipeline_internal::env_bool;
using pipeline_internal::env_truthy;

namespace {

bool should_emit_gst_init_detail() {
  return pipeline_internal::ux::should_emit_gstreamer_for_current_context() ||
         env_truthy("GST_DEBUG");
}

pipeline_internal::ux::VerboseTopic classify_message_topic(const std::string& line);

void* g_allocator_handle = nullptr;

std::string resolve_allocator_symbol_path();
bool is_neat_allocator_path(const std::string& path);

void json_log_suppressor(const gchar* domain, GLogLevelFlags level, const gchar* message,
                         gpointer user_data) {
  if (message && (std::strstr(message, "json_object_get_array_member") ||
                  std::strstr(message, "json_object_get_string_member"))) {
    return;
  }
  g_log_default_handler(domain, level, message, user_data);
}

void gobject_log_suppressor(const gchar* domain, GLogLevelFlags level, const gchar* message,
                            gpointer user_data) {
  if (message && (std::strstr(message, "g_pointer_type_register_static") ||
                  std::strstr(message, "g_type_set_qdata"))) {
    return;
  }
  g_log_default_handler(domain, level, message, user_data);
}

void gstreamer_segment_log_suppressor(const gchar* domain, GLogLevelFlags level,
                                      const gchar* message, gpointer user_data) {
  if (message && (std::strstr(message, "Got data flow before segment event") ||
                  std::strstr(message, "gst_segment_to_running_time") ||
                  std::strstr(message, "segment->format == format"))) {
    return;
  }
  g_log_default_handler(domain, level, message, user_data);
}

void glib_message_suppressor(const gchar* domain, GLogLevelFlags level, const gchar* message,
                             gpointer user_data) {
  if (message && !pipeline_internal::ux::should_emit_topic_for_current_context(
                     classify_message_topic(message))) {
    return;
  }
  g_log_default_handler(domain, level, message, user_data);
}

gboolean sima_custom_meta_transform(GstBuffer* transbuf, GstCustomMeta* meta, GstBuffer* /*buffer*/,
                                    GQuark type, gpointer /*data*/, gpointer user_data) {
  if (!GST_META_TRANSFORM_IS_COPY(type) || !transbuf || !meta)
    return TRUE;
  const char* name = static_cast<const char*>(user_data);
  if (!name || !*name)
    return TRUE;
  GstCustomMeta* out = gst_buffer_add_custom_meta(transbuf, name);
  if (!out)
    return FALSE;
  GstStructure* src = gst_custom_meta_get_structure(meta);
  GstStructure* dst = gst_custom_meta_get_structure(out);
  if (src && dst) {
    gst_structure_remove_all_fields(dst);
    if (!gst_structure_has_name(dst, gst_structure_get_name(src))) {
      gst_structure_set_name(dst, gst_structure_get_name(src));
    }
    gst_structure_foreach(
        src,
        [](GQuark field, const GValue* value, gpointer user_data_inner) -> gboolean {
          auto* dst_inner = static_cast<GstStructure*>(user_data_inner);
          if (!dst_inner || !value)
            return TRUE;
          gst_structure_id_set_value(dst_inner, field, value);
          return TRUE;
        },
        dst);
  }
  return TRUE;
}

bool should_suppress_device_line(const std::string& line) {
  if (line.find("opening /dev/rpmsg_ctrl0") != std::string::npos)
    return true;
  if (line.find("opened /dev/rpmsg") != std::string::npos)
    return true;
  if (line.find("/dev/rpmsg") != std::string::npos)
    return true;
  if (line.find("Nodes:") != std::string::npos)
    return true;
  if ((line.find("Loading model ") != std::string::npos ||
       line.find("Done loading ") != std::string::npos) &&
      line.find("_mla.elf") != std::string::npos) {
    return true;
  }
  return false;
}

bool starts_with(const std::string& line, std::string_view prefix) {
  return line.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), line.begin());
}

pipeline_internal::ux::VerboseTopic classify_message_topic(const std::string& line) {
  if (line.find("FPSSINK: Calling plugin Init Function") != std::string::npos) {
    return pipeline_internal::ux::VerboseTopic::GStreamer;
  }
  if (line.find("CvNormalizeQuantize Config") != std::string::npos ||
      line.find("Out_dims:") != std::string::npos ||
      line.find("Size of buffer allocated:") != std::string::npos ||
      line.find("TOPK Config") != std::string::npos ||
      line.find("TRANSFORM_ANCHORS Config") != std::string::npos ||
      line.find("MaskFiltering:") != std::string::npos ||
      line.find("Decoder destroyed asynchronously") != std::string::npos ||
      line.find("perf: ") != std::string::npos) {
    return pipeline_internal::ux::VerboseTopic::Plugins;
  }
  return pipeline_internal::ux::VerboseTopic::GStreamer;
}

bool should_suppress_known_glib_warning(const std::string& line) {
  return line.find("Failed to build typed processcvu CM config from manifest stage") !=
             std::string::npos ||
         line.find("processcvu_tensor_descs_missing") != std::string::npos;
}

bool should_suppress_verbosity_line(const std::string& line) {
  if (line.empty()) {
    return false;
  }

  const auto opt = pipeline_internal::ux::current_effective_verbose_options();
  const bool any_detail = pipeline_internal::ux::should_emit_any_details(opt);
  const auto allow = [&](pipeline_internal::ux::VerboseTopic topic) {
    return pipeline_internal::ux::should_emit_topic(opt, topic);
  };

  if (starts_with(line, "[GST]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::GStreamer);
  }

  if (starts_with(line, "** Message:")) {
    return !allow(classify_message_topic(line));
  }

  if (line.find("FPSSINK: Calling plugin Init Function") != std::string::npos) {
    return !allow(pipeline_internal::ux::VerboseTopic::GStreamer);
  }

  if (line.find("WARNING **:") != std::string::npos && should_suppress_known_glib_warning(line) &&
      !allow(pipeline_internal::ux::VerboseTopic::GStreamer) &&
      !allow(pipeline_internal::ux::VerboseTopic::Plugins)) {
    return true;
  }

  if (starts_with(line, "[route-debug]") || starts_with(line, "[typed-adapter]") ||
      starts_with(line, "[model-info-shadow]") || starts_with(line, "[mla-contract]") ||
      starts_with(line, "[mpk-contract]") || starts_with(line, "[dequant-compare]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Planner);
  }

  if (starts_with(line, "[GRAPH]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Graph);
  }

  if (starts_with(line, "[STOP]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Graph) &&
           !allow(pipeline_internal::ux::VerboseTopic::Pipeline);
  }

  if (starts_with(line, "[PIPELINE]") || starts_with(line, "[PIPELINE:") ||
      starts_with(line, "[FLOW:") || starts_with(line, "[rtsp]") ||
      starts_with(line, "[sync-cache]") || starts_with(line, "[prepared-runtime-build]") ||
      starts_with(line, "[prepared-runtime-graph]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Pipeline);
  }

  if (starts_with(line, "[INPUTSTREAM]") || starts_with(line, "[APPSINK") ||
      starts_with(line, "[APPSINK_DROP]") || starts_with(line, "[APPSINK_LAST]") ||
      starts_with(line, "[PUSH_REF]") || starts_with(line, "[WEAKREF]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::InputStream);
  }

  if (starts_with(line, "[HOLDER]") || starts_with(line, "[SAMPLE]") ||
      starts_with(line, "[NEAT_CAPS]") || starts_with(line, "[GRAPH_OUTPUT]") ||
      starts_with(line, "[tensorbuffer-") || starts_with(line, "[tensor-set][debug]") ||
      starts_with(line, "[GstDataAdapter]")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Tensor) &&
           !allow(pipeline_internal::ux::VerboseTopic::InputStream);
  }

  if (starts_with(line, "[stage]") || starts_with(line, "[manifest-stage-debug]") ||
      starts_with(line, "[processcvu-stage-debug]") || starts_with(line, "[processcvu-compare]") ||
      starts_with(line, "[preproc-mla-compare]") || starts_with(line, "[preproc-tess-debug]") ||
      starts_with(line, "[preproc-tensor]") || starts_with(line, "[detess-") ||
      starts_with(line, "[mla-") || starts_with(line, "[model-ingress-debug]") ||
      starts_with(line, "[neatdec]") || starts_with(line, "input_track[") ||
      starts_with(line, "neatdecoder init:")) {
    return !allow(pipeline_internal::ux::VerboseTopic::Plugins);
  }

  if (starts_with(line, "[DBG]") || starts_with(line, "[TRACE]") || starts_with(line, "[DIAG]")) {
    return !any_detail;
  }

  return false;
}

void start_stdio_filter(int target_fd, std::vector<std::unique_ptr<std::thread>>& threads) {
  int fds[2];
  if (pipe(fds) != 0)
    return;
  int orig_fd = dup(target_fd);
  if (orig_fd < 0) {
    close(fds[0]);
    close(fds[1]);
    return;
  }
  if (dup2(fds[1], target_fd) < 0) {
    close(orig_fd);
    close(fds[0]);
    close(fds[1]);
    return;
  }
  close(fds[1]);

  threads.emplace_back(std::make_unique<std::thread>([read_fd = fds[0], orig_fd]() {
    std::string buffer;
    buffer.reserve(1024);
    char chunk[512];
    auto flush = [&](bool force_newline) {
      if (buffer.empty())
        return;
      std::string_view raw(buffer);
      std::string_view trimmed = raw;
      while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == '\r')) {
        trimmed.remove_suffix(1);
      }
      std::string line(trimmed);
      if (!line.empty() &&
          (should_suppress_device_line(line) || should_suppress_verbosity_line(line))) {
        buffer.clear();
        return;
      }
      const ssize_t wrote = write(orig_fd, raw.data(), raw.size());
      const bool wrote_output = (wrote > 0);
      if (force_newline && wrote_output && raw.back() != '\n') {
        (void)write(orig_fd, "\n", 1);
      }
      buffer.clear();
    };

    for (;;) {
      const ssize_t n = read(read_fd, chunk, sizeof(chunk));
      if (n <= 0)
        break;
      for (ssize_t i = 0; i < n; ++i) {
        buffer.push_back(chunk[i]);
        if (chunk[i] == '\n') {
          flush(false);
        } else if (buffer.size() >= 8192) {
          flush(true);
        }
      }
    }
    flush(false);
    close(read_fd);
    close(orig_fd);
  }));
  threads.back()->detach();
}

void install_stdio_filters() {
  static std::once_flag once;
  static std::vector<std::unique_ptr<std::thread>> threads;
  std::call_once(once, [&]() {
    start_stdio_filter(STDOUT_FILENO, threads);
    start_stdio_filter(STDERR_FILENO, threads);
  });
}

std::string resolve_gst_plugin_scanner_path() {
  const gchar* env_scanner = g_getenv("GST_PLUGIN_SCANNER");
  if (env_scanner && *env_scanner)
    return env_scanner;

  const char* candidates[] = {
      "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner",
      "/usr/lib/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner",
      "/usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner",
      "/usr/lib/gstreamer-1.0/gst-plugin-scanner",
  };
  for (const char* path : candidates) {
    if (!path)
      continue;
    if (g_file_test(path, G_FILE_TEST_IS_EXECUTABLE)) {
      return path;
    }
  }
  return {};
}

std::string ensure_scanner_wrapper(const std::string& third_party_dir) {
  if (third_party_dir.empty())
    return {};
  const std::string scanner = resolve_gst_plugin_scanner_path();
  if (scanner.empty())
    return {};
  const std::string wrapper = "/tmp/sima_gst_plugin_scanner_" + std::to_string(getpid());

  const std::string script =
      "#!/bin/sh\n"
      "export LD_LIBRARY_PATH=\"" +
      third_party_dir +
      ":${LD_LIBRARY_PATH}\"\n"
      "if [ \"${SIMA_GST_SUPPRESS_GOBJECT_ASSERTS:-1}\" != \"0\" ]; then\n"
      "  fifo=\"/tmp/sima_gst_scanner_fifo_$$\"\n"
      "  mkfifo \"$fifo\" || exit 1\n"
      "  (grep -v -E \"g_pointer_type_register_static|g_type_set_qdata\" < \"$fifo\" >&2) &\n"
      "  filter_pid=$!\n"
      "  \"" +
      scanner +
      "\" \"$@\" 2> \"$fifo\"\n"
      "  status=$?\n"
      "  wait \"$filter_pid\"\n"
      "  rm -f \"$fifo\"\n"
      "  exit \"$status\"\n"
      "fi\n"
      "exec \"" +
      scanner + "\" \"$@\"\n";

  const int fd = ::open(wrapper.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0755);
  if (fd < 0)
    return {};
  const ssize_t written = ::write(fd, script.data(), script.size());
  ::close(fd);
  if (written < 0 || static_cast<size_t>(written) != script.size()) {
    return {};
  }
  ::chmod(wrapper.c_str(), 0755);
  return wrapper;
}

std::string default_plugin_path() {
  const gchar* env_override = g_getenv("SIMA_GST_PLUGIN_DIR");
  if (env_override && *env_override && g_file_test(env_override, G_FILE_TEST_IS_DIR)) {
    gchar* canon = g_canonicalize_filename(env_override, nullptr);
    if (canon) {
      std::string out(canon);
      g_free(canon);
      return out;
    }
    return env_override;
  }

  const std::string loaded_allocator = resolve_allocator_symbol_path();
  if (is_neat_allocator_path(loaded_allocator)) {
    gchar* dir = g_path_get_dirname(loaded_allocator.c_str());
    if (dir && *dir && g_file_test(dir, G_FILE_TEST_IS_DIR)) {
      gchar* canon = g_canonicalize_filename(dir, nullptr);
      std::string out = canon ? std::string(canon) : std::string(dir);
      g_free(canon);
      g_free(dir);
      return out;
    }
    g_free(dir);
  }

  const char* installed_candidates[] = {
      "/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/usr/lib/neat/gst-plugins",
      "/lib/neat/gst-plugins",
  };
  for (const char* dir : installed_candidates) {
    if (!dir || !*dir)
      continue;
    if (!g_file_test(dir, G_FILE_TEST_IS_DIR))
      continue;
    gchar* canon = g_canonicalize_filename(dir, nullptr);
    if (canon) {
      std::string out(canon);
      g_free(canon);
      return out;
    }
    return dir;
  }

  return {};
}

std::vector<std::string> default_system_plugin_dirs() {
  std::vector<std::string> dirs;
  const char* candidates[] = {
      "/lib/aarch64-linux-gnu/gstreamer-1.0",
      "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
      "/usr/lib/gstreamer-1.0",
  };
  for (const char* path : candidates) {
    if (!path || !*path)
      continue;
    if (g_file_test(path, G_FILE_TEST_IS_DIR)) {
      dirs.emplace_back(path);
    }
  }
  return dirs;
}

std::unordered_set<std::string> list_plugin_basenames(const std::string& dir) {
  std::unordered_set<std::string> out;
  if (dir.empty())
    return out;
  DIR* dp = opendir(dir.c_str());
  if (!dp)
    return out;
  while (dirent* ent = readdir(dp)) {
    const std::string name(ent->d_name);
    if (name.rfind("libgst", 0) != 0)
      continue;
    if (name.find(".so") == std::string::npos)
      continue;
    out.emplace(name);
  }
  closedir(dp);
  return out;
}

bool is_regular_file(const std::string& path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0)
    return false;
  return S_ISREG(st.st_mode);
}

std::string third_party_allocator_path(const std::string& third_party_dir) {
  if (third_party_dir.empty())
    return {};
  const std::string neat_allocator = third_party_dir + "/libgstneatallocator.so";
  if (is_regular_file(neat_allocator))
    return neat_allocator;
  throw std::runtime_error("Required NEAT allocator is missing: " + neat_allocator +
                           ". Ensure libgstneatallocator.so is installed in the third-party"
                           " plugin directory (searched: " +
                           third_party_dir + ").");
}

std::string canonicalize_path(const std::string& path) {
  if (path.empty())
    return {};
  char resolved[PATH_MAX];
  if (realpath(path.c_str(), resolved)) {
    return std::string(resolved);
  }
  gchar* canon = g_canonicalize_filename(path.c_str(), nullptr);
  if (!canon)
    return path;
  std::string out(canon);
  g_free(canon);
  return out;
}

bool path_has_prefix(const std::string& path, const std::string& dir) {
  if (path.empty() || dir.empty())
    return false;
  if (path == dir)
    return true;
  const std::string prefix = dir + "/";
  return path.rfind(prefix, 0) == 0;
}

bool same_file_identity(const std::string& lhs, const std::string& rhs) {
  if (lhs.empty() || rhs.empty())
    return false;
  struct stat l {};
  struct stat r {};
  if (stat(lhs.c_str(), &l) != 0 || stat(rhs.c_str(), &r) != 0)
    return false;
  return l.st_dev == r.st_dev && l.st_ino == r.st_ino;
}

bool is_neat_allocator_path(const std::string& path) {
  return path.find("libgstneatallocator.so") != std::string::npos;
}

bool is_legacy_allocator_basename(const std::string& name) {
  return name.rfind("libgstsimaallocator.so", 0) == 0 ||
         name.rfind("libgstsimaaibufferpool.so", 0) == 0;
}

std::string resolve_allocator_symbol_path() {
  void* sym = dlsym(RTLD_DEFAULT, "gst_neat_memory_init_once");
  if (!sym)
    return {};
  Dl_info info{};
  if (dladdr(sym, &info) == 0 || !info.dli_fname || !*info.dli_fname)
    return {};
  return canonicalize_path(info.dli_fname);
}

void preload_third_party_allocator(const std::string& third_party_dir) {
  const std::string allocator_path = canonicalize_path(third_party_allocator_path(third_party_dir));
  if (allocator_path.empty()) {
    throw std::runtime_error(
        "Required NEAT allocator path resolved empty"
        " (third_party_dir='" +
        third_party_dir +
        "'). Check that the directory exists and contains libgstneatallocator.so.");
  }
  const std::string expected_dir = canonicalize_path(third_party_dir);
  const std::string loaded_before = resolve_allocator_symbol_path();
  if (!loaded_before.empty()) {
    if (!is_neat_allocator_path(loaded_before)) {
      throw std::runtime_error(
          "Legacy allocator already loaded before NEAT init: " + loaded_before +
          ". Remove the legacy allocator plugin from GST_PLUGIN_PATH or ensure"
          " NEAT init runs before any GStreamer pipeline creation.");
    }
    if (same_file_identity(loaded_before, allocator_path) ||
        path_has_prefix(loaded_before, expected_dir)) {
      if (should_emit_gst_init_detail()) {
        std::fprintf(stderr, "[GST] allocator already loaded: %s\n", loaded_before.c_str());
      }
      return;
    }
    throw std::runtime_error("Allocator symbol already loaded from unexpected path: " +
                             loaded_before + " (expected under " + expected_dir + ")");
  }

  void* handle = dlopen(allocator_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    const char* err = dlerror();
    std::string msg = "Failed to preload third-party allocator: " + allocator_path;
    if (err && *err) {
      msg += " (";
      msg += err;
      msg += ")";
    }
    throw std::runtime_error(msg);
  }

  g_allocator_handle = handle;

  if (should_emit_gst_init_detail()) {
    std::fprintf(stderr, "[GST] preloaded allocator: %s\n", allocator_path.c_str());
  }
}

void enforce_third_party_allocator(const std::string& third_party_dir) {
  const std::string allocator_path = canonicalize_path(third_party_allocator_path(third_party_dir));
  if (allocator_path.empty()) {
    throw std::runtime_error(
        "Required NEAT allocator path resolved empty"
        " (third_party_dir='" +
        third_party_dir +
        "'). Check that the directory exists and contains libgstneatallocator.so.");
  }

  void* sym = nullptr;
  if (g_allocator_handle) {
    sym = dlsym(g_allocator_handle, "gst_neat_memory_init_once");
  }
  if (!sym) {
    sym = dlsym(RTLD_DEFAULT, "gst_neat_memory_init_once");
  }
  if (!sym) {
    throw std::runtime_error(
        "Failed to locate gst_neat_memory_init_once after preloading third-party allocator.");
  }

  Dl_info info{};
  if (dladdr(sym, &info) == 0 || !info.dli_fname || !*info.dli_fname) {
    throw std::runtime_error("Unable to resolve allocator library path (dladdr failed).");
  }

  const std::string loaded_path = canonicalize_path(info.dli_fname);
  const std::string expected_dir = canonicalize_path(third_party_dir);

  if (!is_neat_allocator_path(loaded_path)) {
    std::string msg = "Legacy allocator loaded: ";
    msg += loaded_path.empty() ? std::string(info.dli_fname) : loaded_path;
    msg += ". NEAT allocator is required.";
    throw std::runtime_error(msg);
  }

  if (!same_file_identity(loaded_path, allocator_path) &&
      !path_has_prefix(loaded_path, expected_dir)) {
    std::string msg = "Unexpected allocator loaded: ";
    msg += loaded_path.empty() ? std::string(info.dli_fname) : loaded_path;
    msg += " (expected under ";
    msg += expected_dir.empty() ? third_party_dir : expected_dir;
    msg += ").";
    throw std::runtime_error(msg);
  }
}

std::string build_filtered_system_dir(const std::vector<std::string>& sys_dirs,
                                      const std::string& third_party_dir,
                                      std::vector<std::string>* skipped, bool strict_neat_only) {
  if (sys_dirs.empty() || third_party_dir.empty())
    return {};

  const auto dup_names = list_plugin_basenames(third_party_dir);
  if (dup_names.empty())
    return {};

  char templ[] = "/tmp/sima_gst_system_filtered_XXXXXX";
  char* tmpdir = mkdtemp(templ);
  if (!tmpdir)
    return {};

  std::string out_dir(tmpdir);
  for (const auto& sys_dir : sys_dirs) {
    DIR* dp = opendir(sys_dir.c_str());
    if (!dp)
      continue;
    while (dirent* ent = readdir(dp)) {
      const std::string name(ent->d_name);
      const bool is_gst = (name.rfind("libgst", 0) == 0);
      const bool is_simaai = (name.rfind("libsimaai", 0) == 0);
      if (!is_gst && !is_simaai)
        continue;
      if (name.find(".so") == std::string::npos)
        continue;
      if (is_legacy_allocator_basename(name)) {
        if (skipped)
          skipped->push_back(name);
        continue;
      }
      if (strict_neat_only && name.find("simaai") != std::string::npos) {
        if (skipped)
          skipped->push_back(name);
        continue;
      }
      if (dup_names.find(name) != dup_names.end()) {
        if (skipped)
          skipped->push_back(name);
        continue;
      }
      const std::string src = sys_dir + "/" + name;
      if (!is_regular_file(src))
        continue;
      const std::string dst = out_dir + "/" + name;
      if (symlink(src.c_str(), dst.c_str()) != 0) {
        continue;
      }
    }
    closedir(dp);
  }

  return out_dir;
}

void prepend_env_list(const char* key, const std::string& value) {
  if (!key || !*key || value.empty())
    return;
  const gchar* cur = g_getenv(key);
  if (!cur || !*cur) {
    g_setenv(key, value.c_str(), FALSE);
    return;
  }
  std::string cur_str(cur);
  std::string_view remaining(cur_str);
  while (!remaining.empty()) {
    const size_t pos = remaining.find(':');
    std::string_view token = (pos == std::string_view::npos) ? remaining : remaining.substr(0, pos);
    if (token == value)
      return;
    if (pos == std::string_view::npos)
      break;
    remaining.remove_prefix(pos + 1);
  }
  std::string merged = value + ":" + cur_str;
  g_setenv(key, merged.c_str(), TRUE);
}

void maybe_enable_dispatcher_watchdog() {
  const gchar* env = g_getenv("SIMA_DISPATCHER_WATCHDOG");
  if (env && *env)
    return;

  const char* helper_candidates[] = {
      "/usr/libexec/sima-neat/dispatcher_watchdog",
      "/usr/libexec/neat/dispatcher_watchdog",
  };

  std::string watchdog_path;
  for (const char* path : helper_candidates) {
    if (!path || !*path)
      continue;
    if (g_file_test(path, G_FILE_TEST_IS_EXECUTABLE)) {
      watchdog_path = path;
      break;
    }
  }
  if (watchdog_path.empty()) {
    std::string plugin_path = default_plugin_path();
    if (!plugin_path.empty()) {
      gchar* watchdog = g_build_filename(plugin_path.c_str(), "dispatcher_watchdog", nullptr);
      if (watchdog && g_file_test(watchdog, G_FILE_TEST_IS_EXECUTABLE)) {
        watchdog_path = watchdog;
      }
      if (watchdog)
        g_free(watchdog);
    }
  }
  if (watchdog_path.empty()) {
    return;
  }

  g_setenv("SIMA_DISPATCHER_WATCHDOG", "1", FALSE);
  const gchar* path_env = g_getenv("SIMA_DISPATCHER_WATCHDOG_PATH");
  if (!path_env || !*path_env) {
    g_setenv("SIMA_DISPATCHER_WATCHDOG_PATH", watchdog_path.c_str(), FALSE);
  }
}

void validate_neat_factory_loaded(const char* factory, const char* plugin_dir) {
  GstElementFactory* f = gst_element_factory_find(factory);
  if (!f) {
    throw std::runtime_error(std::string("Required NEAT factory is missing: ") + factory);
  }

  std::string plugin_name = "<unknown>";
  std::string plugin_path = "<unknown>";
  GstPlugin* plugin = gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(f));
  if (plugin) {
    if (const gchar* name = gst_plugin_get_name(plugin); name && *name) {
      plugin_name = name;
    }
    if (const gchar* filename = gst_plugin_get_filename(plugin); filename && *filename) {
      plugin_path = filename;
    }
    gst_object_unref(plugin);
  }
  gst_object_unref(f);

  if (plugin_path.find("/third_party/") != std::string::npos) {
    throw std::runtime_error("NEAT-only mode violation: factory '" + std::string(factory) +
                             "' loaded from repo-local path: " + plugin_path);
  }
  if (plugin_dir && *plugin_dir) {
    const std::string plugin_dir_canon = canonicalize_path(plugin_dir);
    const std::string plugin_path_canon = canonicalize_path(plugin_path);
    if (!plugin_dir_canon.empty() && !plugin_path_canon.empty() &&
        !path_has_prefix(plugin_path_canon, plugin_dir_canon)) {
      const std::string gst_dir = canonicalize_path("/usr/lib/aarch64-linux-gnu/gstreamer-1.0");
      if (gst_dir.empty() || !path_has_prefix(plugin_path_canon, gst_dir)) {
        throw std::runtime_error("NEAT-only mode violation: factory '" + std::string(factory) +
                                 "' loaded from unexpected path: " + plugin_path_canon);
      }
    }
  }

  if (should_emit_gst_init_detail()) {
    std::fprintf(stderr, "[GST] required factory=%s plugin=%s path=%s\n", factory,
                 plugin_name.c_str(), plugin_path.c_str());
  }
}

void validate_neat_startup_contract(const std::string& plugin_dir) {
  // Contract: PipelineSession must resolve and instantiate NEAT factories.
  // Legacy SIMAAI factories may still be discoverable in the process.
  const char* required[] = {
      "neatprocesscvu", "neatprocessmla", "neatboxdecode", "neatdequant", "neatdetess",
  };
  for (const char* factory : required) {
    validate_neat_factory_loaded(factory, plugin_dir.c_str());
  }
}

} // namespace

void gst_init_once() {
  static std::once_flag once;
  std::call_once(once, []() {
    int argc = 0;
    char** argv = nullptr;

    const bool allow_manual = env_bool("SIMA_ALLOW_GST_INIT", false);
    if (gst_is_initialized() && !allow_manual) {
      std::string plugin_path = default_plugin_path();
      if (plugin_path.empty())
        plugin_path = "<unknown>";
      std::string msg =
          "GStreamer was already initialized before simaai::neat::gst_init_once(). "
          "This bypasses plugin path setup and can load the wrong plugins.\n"
          "Fix: remove manual gst_init() and use simaai::neat::gst_init_once() or Session.\n"
          "If you must initialize manually, set GST_PLUGIN_PATH/GST_PLUGIN_PATH_1_0 to include " +
          plugin_path +
          " or source scripts/use_tensordecoder.sh.\n"
          "Override: set SIMA_ALLOW_GST_INIT=1 to bypass this check.";
      throw std::runtime_error(msg);
    }

    if (env_bool("SIMA_GST_SUPPRESS_DEVICE_LOGS", true)) {
      install_stdio_filters();
    }

    const gchar* plugin_env_1_0 = g_getenv("GST_PLUGIN_PATH_1_0");
    const gchar* registry_env_1_0 = g_getenv("GST_REGISTRY_1_0");
    const gchar* plugin_env = g_getenv("GST_PLUGIN_PATH");

    const bool skip_third_party = false;
    const bool strict_neat_only = env_bool("SIMA_GST_NEAT_ONLY", true);
    const gchar* plugin_dir_override = g_getenv("SIMA_GST_PLUGIN_DIR");
    const std::string third_party = default_plugin_path();
    if (third_party.empty()) {
      throw std::runtime_error(
          "Failed to locate NEAT system plugin directory. Expected "
          "/usr/lib/aarch64-linux-gnu/neat/gst-plugins (or override SIMA_GST_PLUGIN_DIR).");
    }
    if (!third_party.empty() && !skip_third_party) {
      if (strict_neat_only) {
        g_setenv("GST_PLUGIN_PATH_1_0", third_party.c_str(), TRUE);
        g_unsetenv("GST_PLUGIN_PATH");
      } else {
        prepend_env_list("GST_PLUGIN_PATH_1_0", third_party);
        prepend_env_list("GST_PLUGIN_PATH", third_party);
      }
      prepend_env_list("LD_LIBRARY_PATH", third_party);
      if (should_emit_gst_init_detail()) {
        std::fprintf(stderr, "[GST] plugin_dir selected=%s override=%s strict_neat_only=%d\n",
                     third_party.c_str(),
                     (plugin_dir_override && *plugin_dir_override) ? plugin_dir_override
                                                                   : "<unset>",
                     strict_neat_only ? 1 : 0);
      }
    }

    const bool allow_system_plugins = env_bool("SIMA_GST_ALLOW_SYSTEM_PLUGINS", false);
    const gchar* sys_env = g_getenv("GST_PLUGIN_SYSTEM_PATH_1_0");
    if (sys_env && *sys_env && !allow_system_plugins) {
      std::string msg =
          "GST_PLUGIN_SYSTEM_PATH_1_0 is set. This bypasses Sima's system-plugin filtering\n"
          "and can cause duplicate GStreamer plugin registrations (e.g., gst-plugin-scanner\n"
          "GObject warnings) when third-party plugins provide the same elements.\n"
          "If you intentionally want system plugins, set SIMA_GST_ALLOW_SYSTEM_PLUGINS=1\n"
          "to disable this guard. Otherwise, unset GST_PLUGIN_SYSTEM_PATH_1_0.";
      throw std::runtime_error(msg);
    }

    if (!allow_system_plugins) {
      std::vector<std::string> skipped;
      const std::vector<std::string> sys_dirs = default_system_plugin_dirs();
      const std::string filtered =
          build_filtered_system_dir(sys_dirs, third_party, &skipped, strict_neat_only);
      if (!filtered.empty()) {
        g_setenv("GST_PLUGIN_SYSTEM_PATH_1_0", filtered.c_str(), TRUE);
        if (should_emit_gst_init_detail()) {
          std::fprintf(stderr, "[GST] filtered system plugins: %s (skipped %zu)\n",
                       filtered.c_str(), skipped.size());
        }
      }
      if (should_emit_gst_init_detail() && !skipped.empty()) {
        std::fprintf(stderr, "[GST] skipped duplicate plugins:\n");
        for (const auto& name : skipped) {
          std::fprintf(stderr, "  - %s\n", name.c_str());
        }
      }
    }

    if ((!plugin_env_1_0 || !*plugin_env_1_0) && (!registry_env_1_0 || !*registry_env_1_0) &&
        (!plugin_env || !*plugin_env)) {
      if (!third_party.empty() && !skip_third_party) {
        g_setenv("GST_PLUGIN_PATH_1_0", third_party.c_str(), FALSE);
        if (strict_neat_only) {
          g_unsetenv("GST_PLUGIN_PATH");
        } else {
          g_setenv("GST_PLUGIN_PATH", third_party.c_str(), FALSE);
        }
        g_setenv("LD_LIBRARY_PATH", third_party.c_str(), FALSE);
      }
      std::string registry = "/tmp/sima_gst_registry_" + std::to_string(getpid()) + ".bin";
      g_setenv("GST_REGISTRY_1_0", registry.c_str(), FALSE);
    }

    if (!skip_third_party) {
      preload_third_party_allocator(third_party);
      enforce_third_party_allocator(third_party);
    }

    maybe_enable_dispatcher_watchdog();

    if (!third_party.empty() && env_bool("SIMA_GST_WRAP_SCANNER", true)) {
      const gchar* scanner_env = g_getenv("GST_PLUGIN_SCANNER");
      if (!scanner_env || !*scanner_env) {
        const std::string wrapper = ensure_scanner_wrapper(third_party);
        if (!wrapper.empty()) {
          g_setenv("GST_PLUGIN_SCANNER", wrapper.c_str(), TRUE);
          if (should_emit_gst_init_detail()) {
            std::fprintf(stderr, "[GST] using plugin scanner wrapper: %s\n", wrapper.c_str());
          }
        }
      }
    }

    g_log_set_handler(nullptr, G_LOG_LEVEL_MESSAGE, glib_message_suppressor, nullptr);
    gst_init(&argc, &argv);
    if ((!g_getenv("GST_DEBUG") || !*g_getenv("GST_DEBUG")) &&
        pipeline_internal::ux::should_emit_gstreamer_for_current_context()) {
      gst_debug_set_default_threshold(GST_LEVEL_INFO);
    }
    // Register custom metadata after gst_init to avoid GLib type init issues,
    // but before loading plugins to ensure transforms are wired.
    if (!gst_meta_get_info("GstSimaMeta")) {
      static const gchar* sima_meta_tags[] = {GST_META_TAG_MEMORY_STR, nullptr};
      gst_meta_register_custom("GstSimaMeta", sima_meta_tags, sima_custom_meta_transform,
                               const_cast<char*>("GstSimaMeta"), nullptr);
    }
    if (!gst_meta_get_info("GstSimaSampleMeta")) {
      static const gchar* sima_sample_tags[] = {GST_META_TAG_MEMORY_STR, nullptr};
      gst_meta_register_custom("GstSimaSampleMeta", sima_sample_tags, sima_custom_meta_transform,
                               const_cast<char*>("GstSimaSampleMeta"), nullptr);
    }
    if (!gst_meta_get_info(SIMA_TENSOR_SET_META_NAME)) {
      static const gchar* sima_tensor_set_tags[] = {GST_META_TAG_MEMORY_STR, nullptr};
      gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, sima_tensor_set_tags,
                               sima_custom_meta_transform,
                               const_cast<char*>(SIMA_TENSOR_SET_META_NAME), nullptr);
    }
    if (env_bool("SIMA_GST_SUPPRESS_JSON_WARNINGS", true)) {
      g_log_set_handler("Json",
                        static_cast<GLogLevelFlags>(G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_WARNING),
                        json_log_suppressor, nullptr);
    }
    if (env_bool("SIMA_GST_SUPPRESS_GOBJECT_ASSERTS", true)) {
      g_log_set_handler("GLib-GObject", static_cast<GLogLevelFlags>(G_LOG_LEVEL_CRITICAL),
                        gobject_log_suppressor, nullptr);
    }
    if (env_bool("SIMA_GST_SUPPRESS_SEGMENT_WARNINGS", true)) {
      g_log_set_handler("GStreamer",
                        static_cast<GLogLevelFlags>(G_LOG_LEVEL_WARNING | G_LOG_LEVEL_CRITICAL),
                        gstreamer_segment_log_suppressor, nullptr);
    }
    validate_neat_startup_contract(third_party);
  });
}

} // namespace simaai::neat
