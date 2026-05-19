#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

namespace simaai::neat::pipeline_internal {

struct DebugProfileRule {
  const char* key;
  const char* component;
  int min_level;
};

inline bool parse_bool_literal(const char* v, bool* out) {
  if (!out || !v || !*v)
    return false;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") || !std::strcmp(v, "on") ||
      !std::strcmp(v, "ON")) {
    *out = true;
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") || !std::strcmp(v, "off") ||
      !std::strcmp(v, "OFF")) {
    *out = false;
    return true;
  }
  return false;
}

inline bool token_sep(char c) {
  return c == ',' || c == ';' || c == '|' || std::isspace(static_cast<unsigned char>(c));
}

inline bool token_equals_ci(const char* token, std::size_t token_len, const char* probe) {
  if (!token || !probe)
    return false;
  std::size_t probe_len = std::strlen(probe);
  if (token_len != probe_len)
    return false;
  for (std::size_t i = 0; i < token_len; ++i) {
    const unsigned char a = static_cast<unsigned char>(token[i]);
    const unsigned char b = static_cast<unsigned char>(probe[i]);
    if (std::tolower(a) != std::tolower(b))
      return false;
  }
  return true;
}

inline bool token_list_contains_ci(const char* list, const char* token) {
  if (!list || !*list || !token || !*token)
    return false;
  const char* cur = list;
  while (*cur) {
    while (*cur && token_sep(*cur)) {
      ++cur;
    }
    const char* start = cur;
    while (*cur && !token_sep(*cur)) {
      ++cur;
    }
    const std::size_t len = static_cast<std::size_t>(cur - start);
    if (len > 0 && token_equals_ci(start, len, token))
      return true;
  }
  return false;
}

inline int debug_profile_level() {
  const char* v = std::getenv("SIMA_DEBUG_LEVEL");
  if (!v || !*v)
    return 1;
  char* end = nullptr;
  long raw = std::strtol(v, &end, 10);
  if (!end || *end != '\0')
    return 1;
  if (raw < 0)
    return 0;
  if (raw > 3)
    return 3;
  return static_cast<int>(raw);
}

inline bool debug_profile_component_enabled(const char* component) {
  if (!component || !*component)
    return false;
  const char* profile = std::getenv("SIMA_DEBUG_PROFILE");
  if (!profile || !*profile)
    return false;
  return token_list_contains_ci(profile, "all") || token_list_contains_ci(profile, "*") ||
         token_list_contains_ci(profile, component);
}

inline int framework_verbose_level() {
  const char* topics = std::getenv("SIMA_NEAT_VERBOSE_TOPICS");
  const char* v = std::getenv("SIMA_NEAT_VERBOSE_LEVEL");
  if ((!v || !*v) && topics && *topics) {
    return 2;
  }
  if (!v || !*v) {
    return 0;
  }
  char* end = nullptr;
  long raw = std::strtol(v, &end, 10);
  if (!end || *end != '\0')
    return 0;
  if (raw < 0)
    return 0;
  if (raw > 2)
    return 2;
  return static_cast<int>(raw);
}

inline bool framework_verbose_component_enabled(const char* component) {
  if (!component || !*component)
    return false;
  const char* topics = std::getenv("SIMA_NEAT_VERBOSE_TOPICS");
  if (!topics || !*topics)
    return false;
  return token_list_contains_ci(topics, "all") || token_list_contains_ci(topics, "*") ||
         token_list_contains_ci(topics, component);
}

inline const DebugProfileRule* debug_profile_rule(const char* key) {
  if (!key || !*key)
    return nullptr;
  static constexpr DebugProfileRule rules[] = {
      {"SIMA_ASYNC_TPUT_DIAG", "pipeline", 1},
      {"SIMA_APPSINK_CAPS_DEBUG", "appsink", 1},
      {"SIMA_APPSINK_CB_DEBUG", "appsink", 1},
      {"SIMA_APPSINK_DROP_LAST_DEBUG", "appsink", 1},
      {"SIMA_APPSINK_LAST_SAMPLE_DEBUG", "appsink", 1},
      {"SIMA_APPSINK_PULL_DEBUG", "appsink", 1},
      {"SIMA_BUILD_MODE_DEBUG", "pipeline", 1},
      {"SIMA_DEBUG_INPUT_POOL", "inputstream", 1},
      {"SIMA_DISPATCHER_TRACE", "pipeline", 1},
      {"SIMA_GRAPH_DEBUG", "graph", 1},
      {"SIMA_GRAPH_DIAG_ON_STOP", "graph", 1},
      {"SIMA_GRAPH_PUSH_FAIL_DEBUG", "graph", 1},
      {"SIMA_GRAPH_SCHED_DEBUG", "graph", 1},
      {"SIMA_GRAPH_SCHED_LOG_FIRST_STREAM", "graph", 1},
      {"SIMA_GRAPH_TEARDOWN_DEBUG", "graph", 1},
      {"SIMA_GST_ALL_BUFFER_DEBUG", "gst", 2},
      {"SIMA_GST_APPSINK_BUFFER_DEBUG", "gst", 1},
      {"SIMA_GST_BOUNDARY_PROBES", "gst", 1},
      {"SIMA_GST_BOUNDARY_BUFFER_DEBUG", "gst", 2},
      {"SIMA_GST_BOXDECODE_BUFFER_DEBUG", "gst", 2},
      {"SIMA_GST_DETESS_INPUT_DEBUG", "gst", 2},
      {"SIMA_GST_DETESS_OUTPUT_DEBUG", "gst", 2},
      {"SIMA_GST_DETESS_POOL_DEBUG", "gst", 2},
      {"SIMA_GST_ELEMENT_TIMINGS", "gst", 1},
      {"SIMA_GST_FLOW_DEBUG", "gst", 1},
      {"SIMA_GST_OPTIONS_DEBUG", "gst", 1},
      {"SIMA_GST_PAD_LINK_DEBUG", "gst", 2},
      {"SIMA_GST_STAGE_TIMINGS", "gst", 1},
      {"SIMA_INPUTSTREAM_ALLOC_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_DOT_ON_TIMEOUT", "inputstream", 2},
      {"SIMA_INPUTSTREAM_EOS_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_HOLDER_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_META_DEBUG", "inputstream", 1},
      {"SIMA_NEAT_CAPS_TRACE", "inputstream", 2},
      {"SIMA_INPUTSTREAM_POOL_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_PUSH_FAIL_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_PUSH_FAIL_DETAIL", "inputstream", 2},
      {"SIMA_INPUTSTREAM_PUSH_REF_DEBUG", "inputstream", 2},
      {"SIMA_INPUTSTREAM_PUSH_TIMING", "inputstream", 2},
      {"SIMA_INPUTSTREAM_USE_APPSINK_CALLBACKS", "inputstream", 2},
      {"SIMA_INPUTSTREAM_WARN", "inputstream", 1},
      {"SIMA_INPUTSTREAM_WEAKREF_DEBUG", "inputstream", 2},
      {"SIMA_PIPELINE_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_STATE_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_STRING_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_TEARDOWN_DEBUG", "pipeline", 1},
      {"SIMA_PULL_TIMEOUT_DIAG", "pipeline", 1},
      {"SIMA_PULL_TIMEOUT_POOL_DIAG", "pipeline", 2},
      {"SIMA_RTSP_DEBUG", "pipeline", 1},
      {"SIMA_RTSP_STATS_DEBUG", "pipeline", 1},
      {"SIMA_SAMPLE_BYTES", "inputstream", 1},
      {"SIMA_SAMPLE_DEBUG", "inputstream", 1},
      {"SIMA_SAMPLE_FORCE_BUNDLE", "inputstream", 2},
      {"SIMA_STAGE_DEBUG", "pipeline", 1},
      {"SIMA_STOP_TRACE", "pipeline", 1},
      {"SIMA_TENSOR_MAPFAIL_DEBUG", "inputstream", 1},
  };
  for (const auto& rule : rules) {
    if (!std::strcmp(key, rule.key))
      return &rule;
  }
  return nullptr;
}

inline const DebugProfileRule* framework_verbose_rule(const char* key) {
  if (!key || !*key)
    return nullptr;
  static constexpr DebugProfileRule rules[] = {
      {"SIMA_ASYNC_TPUT_DIAG", "pipeline", 1},
      {"SIMA_APPSINK_CAPS_DEBUG", "inputstream", 1},
      {"SIMA_APPSINK_CB_DEBUG", "inputstream", 1},
      {"SIMA_APPSINK_DROP_LAST_DEBUG", "inputstream", 1},
      {"SIMA_APPSINK_LAST_SAMPLE_DEBUG", "inputstream", 1},
      {"SIMA_APPSINK_PULL_DEBUG", "inputstream", 1},
      {"SIMA_BOXDECODE_DEBUG", "planner", 1},
      {"SIMA_BUILD_MODE_DEBUG", "pipeline", 1},
      {"SIMA_DEBUG_OUTPUTSPEC_LOG", "planner", 1},
      {"SIMA_DETESS_DISPATCH_DEBUG", "plugins", 1},
      {"SIMA_DETESS_LAYOUT_DEBUG", "planner", 1},
      {"SIMA_DETESS_LIFECYCLE_DEBUG", "plugins", 1},
      {"SIMA_DETESS_OVERRIDE_DEBUG", "plugins", 1},
      {"SIMA_DEBUG_INPUT_POOL", "inputstream", 1},
      {"SIMA_DISPATCHER_TRACE", "pipeline", 1},
      {"SIMA_GRAPH_DEBUG", "graph", 1},
      {"SIMA_GRAPH_DIAG_ON_STOP", "graph", 1},
      {"SIMA_GRAPH_GDB_ON_PUSH_FAIL", "graph", 1},
      {"SIMA_GRAPH_OUTPUT_COPY_DEBUG", "tensor", 1},
      {"SIMA_GRAPH_PIPELINE_DIAG_SUMMARY", "graph", 1},
      {"SIMA_GRAPH_PUSH_FAIL_DEBUG", "graph", 1},
      {"SIMA_GRAPH_SCHED_DEBUG", "graph", 1},
      {"SIMA_GRAPH_SCHED_LOG_FIRST_STREAM", "graph", 1},
      {"SIMA_GRAPH_SERIAL_PIPELINE_BUILD", "graph", 1},
      {"SIMA_GRAPH_TEARDOWN_DEBUG", "graph", 1},
      {"SIMA_GRAPH_ZERO_COPY_DEBUG", "graph", 1},
      {"SIMA_GST_ALL_BUFFER_DEBUG", "gstreamer", 2},
      {"SIMA_GST_APPSINK_BUFFER_DEBUG", "gstreamer", 1},
      {"SIMA_GST_BUFFER_MEMFLAGS_DEBUG", "gstreamer", 2},
      {"SIMA_GST_BOUNDARY_PROBES", "gstreamer", 1},
      {"SIMA_GST_BOUNDARY_BUFFER_DEBUG", "gstreamer", 2},
      {"SIMA_GST_BOXDECODE_BUFFER_DEBUG", "gstreamer", 2},
      {"SIMA_GST_DATA_ADAPTER_DEBUG", "tensor", 1},
      {"SIMA_GST_DETESS_INPUT_DEBUG", "gstreamer", 2},
      {"SIMA_GST_DETESS_OUTPUT_DEBUG", "gstreamer", 2},
      {"SIMA_GST_DETESS_POOL_DEBUG", "gstreamer", 2},
      {"SIMA_GST_ELEMENT_TIMINGS", "gstreamer", 1},
      {"SIMA_GST_FLOW_DEBUG", "gstreamer", 1},
      {"SIMA_GST_OPTIONS_DEBUG", "gstreamer", 1},
      {"SIMA_GST_PAD_LINK_DEBUG", "gstreamer", 2},
      {"SIMA_GST_PLUGIN_PATH_DEBUG", "gstreamer", 1},
      {"SIMA_GST_STAGE_TIMINGS", "gstreamer", 1},
      {"SIMA_INPUTSTREAM_ALLOC_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_EOS_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_HOLDER_DEBUG", "tensor", 1},
      {"SIMA_INPUTSTREAM_META_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_POOL_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_PUSH_FAIL_DEBUG", "inputstream", 1},
      {"SIMA_INPUTSTREAM_PUSH_FAIL_DETAIL", "inputstream", 2},
      {"SIMA_INPUTSTREAM_PUSH_REF_DEBUG", "inputstream", 2},
      {"SIMA_INPUTSTREAM_PUSH_TIMING", "inputstream", 2},
      {"SIMA_INPUTSTREAM_WARN", "inputstream", 1},
      {"SIMA_INPUTSTREAM_WEAKREF_DEBUG", "inputstream", 2},
      {"SIMA_MANIFEST_DEBUG", "planner", 1},
      {"SIMA_MLA_CONTRACT_DEBUG", "planner", 1},
      {"SIMA_MLA_STAGE_STORAGE_DEBUG", "planner", 1},
      {"SIMA_MLA_NUM_BUFFERS_DEBUG", "pipeline", 1},
      {"SIMA_MANIFEST_CONTEXT_DEBUG", "planner", 1},
      {"SIMA_MODEL_INFO_SHADOW_DEBUG", "planner", 1},
      {"SIMA_MODEL_RUNNER_DEBUG", "planner", 1},
      {"SIMA_MPK_CONTRACT_DEBUG", "planner", 1},
      {"SIMA_NEAT_CAPS_TRACE", "tensor", 1},
      {"SIMA_PIPELINE_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_PUSH_RETURN_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_STATE_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_STRING_DEBUG", "pipeline", 1},
      {"SIMA_PIPELINE_TEARDOWN_DEBUG", "pipeline", 1},
      {"SIMA_PREPROC_DEBUG_CONFIG", "planner", 1},
      {"SIMA_PREPARED_RUNTIME_BUILD_DEBUG", "pipeline", 1},
      {"SIMA_PULL_TIMEOUT_DIAG", "pipeline", 1},
      {"SIMA_PULL_TIMEOUT_POOL_DIAG", "pipeline", 1},
      {"SIMA_RENDER_STAGE_DEBUG", "planner", 1},
      {"SIMA_ROUTE_DEBUG", "planner", 1},
      {"SIMA_RTSP_DEBUG", "pipeline", 1},
      {"SIMA_RTSP_STATS_DEBUG", "pipeline", 1},
      {"SIMA_SAMPLE_BYTES", "tensor", 1},
      {"SIMA_SAMPLE_DEBUG", "tensor", 1},
      {"SIMA_NEAT_BUNDLED_DEBUG", "tensor", 1},
      {"SIMA_SESSION_SYNC_CACHE_DEBUG", "pipeline", 1},
      {"SIMA_STAGE_DEBUG", "plugins", 1},
      {"SIMA_STAGE_TENSOR_STATS", "tensor", 2},
      {"SIMA_TESS_SEGMENT_DEBUG", "planner", 1},
      {"SIMA_STOP_TRACE", "pipeline", 1},
      {"SIMA_TENSOR_MAPFAIL_DEBUG", "tensor", 1},
      {"SIMA_TENSOR_MAP_DEBUG", "tensor", 1},
      {"SIMA_TENSOR_SET_DEBUG", "tensor", 1},
      {"SIMA_TRANSPORT_CONTRACT_DEBUG", "planner", 1},
      {"SIMA_TYPED_ADAPTER_DEBUG", "planner", 1},
  };
  for (const auto& rule : rules) {
    if (!std::strcmp(key, rule.key))
      return &rule;
  }
  return nullptr;
}

inline bool debug_profile_enabled_for_key(const char* key) {
  const DebugProfileRule* rule = debug_profile_rule(key);
  if (!rule)
    return false;
  if (!debug_profile_component_enabled(rule->component))
    return false;
  return debug_profile_level() >= rule->min_level;
}

inline bool framework_verbose_enabled_for_key(const char* key) {
  const DebugProfileRule* rule = framework_verbose_rule(key);
  if (!rule)
    return false;
  if (!framework_verbose_component_enabled(rule->component))
    return false;
  return framework_verbose_level() >= rule->min_level;
}

inline bool debug_profile_int_default(const char* key, int def_val, int* out) {
  if (!out || !key || !*key)
    return false;
  const int level = debug_profile_level();
  if (level <= 0)
    return false;

  if (!std::strcmp(key, "SIMA_GST_BUFFER_DEBUG_LIMIT")) {
    if (!debug_profile_component_enabled("gst") && !debug_profile_component_enabled("appsink"))
      return false;
    *out = std::max(def_val, (level >= 2) ? 10 : 5);
    return true;
  }
  if (!std::strcmp(key, "SIMA_GRAPH_SCHED_LOG_EVERY")) {
    if (!debug_profile_component_enabled("graph"))
      return false;
    *out = std::max(def_val, (level >= 2) ? 100 : 25);
    return true;
  }
  if (!std::strcmp(key, "SIMA_INPUTSTREAM_POOL_WAIT_LOG_MS")) {
    if (!debug_profile_component_enabled("inputstream"))
      return false;
    *out = std::max(def_val, (level >= 2) ? 5 : def_val);
    return true;
  }
  return false;
}

inline bool debug_profile_string_default(const char* key, std::string* out) {
  if (!out || !key || !*key)
    return false;
  const int level = debug_profile_level();
  if (level <= 0)
    return false;
  if (!std::strcmp(key, "SIMA_GST_ELEMENT_BUFFER_DEBUG")) {
    if (!debug_profile_component_enabled("gst"))
      return false;
    if (level < 2)
      return false;
    *out = "all";
    return true;
  }
  if (!std::strcmp(key, "SIMA_GST_ELEMENT_BUFFER_DEBUG_DIR")) {
    if (!debug_profile_component_enabled("gst"))
      return false;
    *out = "both";
    return true;
  }
  return false;
}

inline bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  bool parsed = false;
  if (parse_bool_literal(v, &parsed))
    return parsed;
  if (v && *v)
    return def_val;
  if (framework_verbose_enabled_for_key(key))
    return true;
  if (debug_profile_enabled_for_key(key))
    return true;
  return def_val;
}

inline bool env_truthy(const char* key) {
  const char* value = std::getenv(key);
  if (!value || !*value) {
    return false;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "FALSE") == 0) {
    return false;
  }
  return true;
}

inline bool multi_io_bundled_appsrc_enabled() {
  return env_truthy("SIMA_NEAT_MULTI_IO_BUNDLED_APPSRC");
}

inline bool multi_io_bundled_debug_enabled() {
  return env_truthy("SIMA_NEAT_BUNDLED_DEBUG");
}

inline bool mpk_no_json_path_enabled() {
  return true;
}

inline bool mpk_strict_processcvu_abi_enabled() {
  return true;
}

inline bool mpk_shadow_processcvu_geometry_only_enabled() {
  return true;
}

inline bool manifest_route_debug_enabled_default_on() {
  return env_bool("SIMA_MANIFEST_ROUTE_DEBUG", true);
}

inline int env_int(const char* key, int def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    int profile_default = def_val;
    if (debug_profile_int_default(key, def_val, &profile_default))
      return profile_default;
    return def_val;
  }
  return std::atoi(v);
}

inline double env_double(const char* key, double def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def_val;
  return std::strtod(v, nullptr);
}

inline std::string env_str(const char* key, const std::string& def_val = "") {
  const char* v = std::getenv(key);
  if (!v) {
    std::string profile_default;
    if (debug_profile_string_default(key, &profile_default))
      return profile_default;
    return def_val;
  }
  return std::string(v);
}

inline bool env_int(const char* key, int* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (!end || *end != '\0')
    return false;
  *out = static_cast<int>(val);
  return true;
}

inline bool env_double(const char* key, double* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  double val = std::strtod(v, &end);
  if (!end || *end != '\0')
    return false;
  *out = val;
  return true;
}

inline bool env_string(const char* key, std::string* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  *out = v;
  return true;
}

} // namespace simaai::neat::pipeline_internal
