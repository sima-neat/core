#include "pipeline/internal/UxLogging.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace simaai::neat::pipeline_internal::ux {
namespace {

thread_local const VerboseOptions* g_verbose_context = nullptr;
std::mutex g_runtime_verbose_mu;
std::uint64_t g_runtime_verbose_next_id = 1;
std::unordered_map<std::uint64_t, VerboseOptions> g_runtime_verbose_leases;
bool g_runtime_saved_env = false;
std::optional<std::string> g_saved_verbose_level_env;
std::optional<std::string> g_saved_verbose_topics_env;

const VerboseOptions& quiet_verbose_options() {
  static const VerboseOptions opt{
      .level = VerbosityLevel::Quiet,
      .progress = false,
      .progress_force = false,
      .gstreamer = false,
      .planner = false,
      .graph = false,
      .pipeline = false,
      .inputstream = false,
      .tensor = false,
      .plugins = false,
  };
  return opt;
}

std::string step_prefix(int current, int total) {
  std::ostringstream oss;
  oss << "[" << current << "/" << total << "] ";
  return oss.str();
}

long long elapsed_ms_since(const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool parse_bool_literal(const char* raw, bool* out) {
  if (!out || !raw || !*raw) {
    return false;
  }
  if (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
      std::strcmp(raw, "TRUE") == 0 || std::strcmp(raw, "yes") == 0 ||
      std::strcmp(raw, "YES") == 0 || std::strcmp(raw, "on") == 0 ||
      std::strcmp(raw, "ON") == 0) {
    *out = true;
    return true;
  }
  if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
      std::strcmp(raw, "FALSE") == 0 || std::strcmp(raw, "no") == 0 ||
      std::strcmp(raw, "NO") == 0 || std::strcmp(raw, "off") == 0 ||
      std::strcmp(raw, "OFF") == 0) {
    *out = false;
    return true;
  }
  return false;
}

bool env_truthy(const char* key) {
  if (!key || !*key) {
    return false;
  }
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return false;
  }
  bool parsed = false;
  if (parse_bool_literal(raw, &parsed)) {
    return parsed;
  }
  return std::strcmp(raw, "0") != 0;
}

bool token_sep(char c) {
  return c == ',' || c == ';' || c == '|' ||
         std::isspace(static_cast<unsigned char>(c));
}

bool token_equals_ci(const char* token, std::size_t token_len,
                     std::string_view probe) {
  if (!token || token_len != probe.size()) {
    return false;
  }
  for (std::size_t i = 0; i < token_len; ++i) {
    const unsigned char a = static_cast<unsigned char>(token[i]);
    const unsigned char b = static_cast<unsigned char>(probe[i]);
    if (std::tolower(a) != std::tolower(b)) {
      return false;
    }
  }
  return true;
}

bool token_list_contains_ci(const char* list, std::string_view token) {
  if (!list || !*list || token.empty()) {
    return false;
  }
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
    if (len > 0 && token_equals_ci(start, len, token)) {
      return true;
    }
  }
  return false;
}

int parse_verbose_level(const char* raw, int fallback = 0) {
  if (!raw || !*raw) {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (!end || *end != '\0') {
    return fallback;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 2) {
    return 2;
  }
  return static_cast<int>(value);
}

bool topic_enabled_from_env(const char* raw, std::string_view topic) {
  if (!raw || !*raw) {
    return false;
  }
  return token_list_contains_ci(raw, "all") || token_list_contains_ci(raw, "*") ||
         token_list_contains_ci(raw, topic);
}

VerboseOptions manual_verbose_options() {
  const char* saved_topics =
      (g_saved_verbose_topics_env && !g_saved_verbose_topics_env->empty())
          ? g_saved_verbose_topics_env->c_str()
          : nullptr;
  int level = 0;
  if (g_saved_verbose_level_env && !g_saved_verbose_level_env->empty()) {
    level = parse_verbose_level(g_saved_verbose_level_env->c_str(), 0);
  } else if (saved_topics) {
    level = 2;
  }

  VerboseOptions opt = quiet_verbose_options();
  opt.level = (level <= 0) ? VerbosityLevel::Quiet
                           : (level == 1 ? VerbosityLevel::Production
                                         : VerbosityLevel::Verbose);
  opt.gstreamer = topic_enabled_from_env(saved_topics, "gstreamer");
  opt.planner = topic_enabled_from_env(saved_topics, "planner");
  opt.graph = topic_enabled_from_env(saved_topics, "graph");
  opt.pipeline = topic_enabled_from_env(saved_topics, "pipeline");
  opt.inputstream = topic_enabled_from_env(saved_topics, "inputstream");
  opt.tensor = topic_enabled_from_env(saved_topics, "tensor");
  opt.plugins = topic_enabled_from_env(saved_topics, "plugins");
  return opt;
}

int verbosity_level_value(VerbosityLevel level) {
  switch (level) {
  case VerbosityLevel::Quiet:
    return 0;
  case VerbosityLevel::Production:
    return 1;
  case VerbosityLevel::Verbose:
    return 2;
  }
  return 0;
}

bool topic_enabled_direct(const VerboseOptions& opt, VerboseTopic topic) {
  switch (topic) {
  case VerboseTopic::Planner:
    return opt.planner;
  case VerboseTopic::Graph:
    return opt.graph;
  case VerboseTopic::Pipeline:
    return opt.pipeline;
  case VerboseTopic::InputStream:
    return opt.inputstream;
  case VerboseTopic::Tensor:
    return opt.tensor;
  case VerboseTopic::Plugins:
    return opt.plugins;
  case VerboseTopic::GStreamer:
    return opt.gstreamer;
  }
  return false;
}

VerboseOptions merge_verbose_options(const VerboseOptions& lhs,
                                     const VerboseOptions& rhs) {
  VerboseOptions out = lhs;
  if (verbosity_level_value(rhs.level) > verbosity_level_value(out.level)) {
    out.level = rhs.level;
  }
  out.progress = out.progress || rhs.progress;
  out.progress_force = out.progress_force || rhs.progress_force;
  out.gstreamer = out.gstreamer || rhs.gstreamer;
  out.planner = out.planner || rhs.planner;
  out.graph = out.graph || rhs.graph;
  out.pipeline = out.pipeline || rhs.pipeline;
  out.inputstream = out.inputstream || rhs.inputstream;
  out.tensor = out.tensor || rhs.tensor;
  out.plugins = out.plugins || rhs.plugins;
  return out;
}

std::vector<std::string> verbose_topics(const VerboseOptions& opt) {
  std::vector<std::string> topics;
  auto push_if = [&](std::string_view name, bool enabled) {
    if (enabled) {
      topics.emplace_back(name);
    }
  };
  push_if("planner", should_emit_topic(opt, VerboseTopic::Planner));
  push_if("graph", should_emit_topic(opt, VerboseTopic::Graph));
  push_if("pipeline", should_emit_topic(opt, VerboseTopic::Pipeline));
  push_if("inputstream", should_emit_topic(opt, VerboseTopic::InputStream));
  push_if("tensor", should_emit_topic(opt, VerboseTopic::Tensor));
  push_if("plugins", should_emit_topic(opt, VerboseTopic::Plugins));
  push_if("gstreamer", should_emit_topic(opt, VerboseTopic::GStreamer));
  return topics;
}

std::string join_topics_csv(const std::vector<std::string>& topics) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < topics.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << topics[i];
  }
  return oss.str();
}

int runtime_env_level(const VerboseOptions& opt) {
  int level = verbosity_level_value(opt.level);
  if (level <= 0 && !verbose_topics(opt).empty()) {
    level = 1;
  }
  return level;
}

VerboseOptions aggregate_runtime_verbose_options_locked() {
  VerboseOptions merged = manual_verbose_options();
  for (const auto& [_, opt] : g_runtime_verbose_leases) {
    merged = merge_verbose_options(merged, opt);
  }
  return merged;
}

void restore_saved_runtime_env_locked() {
  if (!g_runtime_saved_env) {
    return;
  }
  if (g_saved_verbose_level_env.has_value()) {
    ::setenv("SIMA_NEAT_VERBOSE_LEVEL", g_saved_verbose_level_env->c_str(), 1);
  } else {
    ::unsetenv("SIMA_NEAT_VERBOSE_LEVEL");
  }
  if (g_saved_verbose_topics_env.has_value()) {
    ::setenv("SIMA_NEAT_VERBOSE_TOPICS", g_saved_verbose_topics_env->c_str(), 1);
  } else {
    ::unsetenv("SIMA_NEAT_VERBOSE_TOPICS");
  }
}

void sync_runtime_env_locked() {
  if (g_runtime_verbose_leases.empty()) {
    restore_saved_runtime_env_locked();
    return;
  }

  const VerboseOptions merged = aggregate_runtime_verbose_options_locked();
  const int level = runtime_env_level(merged);
  const std::string level_text = std::to_string(level);
  ::setenv("SIMA_NEAT_VERBOSE_LEVEL", level_text.c_str(), 1);

  const std::string topics_csv = join_topics_csv(verbose_topics(merged));
  if (topics_csv.empty()) {
    ::unsetenv("SIMA_NEAT_VERBOSE_TOPICS");
  } else {
    ::setenv("SIMA_NEAT_VERBOSE_TOPICS", topics_csv.c_str(), 1);
  }
}

class RuntimeVerboseLease {
public:
  explicit RuntimeVerboseLease(std::uint64_t id) : id_(id) {}
  ~RuntimeVerboseLease() {
    std::lock_guard<std::mutex> lock(g_runtime_verbose_mu);
    g_runtime_verbose_leases.erase(id_);
    sync_runtime_env_locked();
    if (g_runtime_verbose_leases.empty()) {
      g_runtime_saved_env = false;
      g_saved_verbose_level_env.reset();
      g_saved_verbose_topics_env.reset();
    }
  }

private:
  std::uint64_t id_ = 0;
};

} // namespace

ScopedVerboseContext::ScopedVerboseContext(const VerboseOptions& opt) : prev_(g_verbose_context) {
  g_verbose_context = &opt;
}

ScopedVerboseContext::~ScopedVerboseContext() {
  g_verbose_context = prev_;
}

std::shared_ptr<void> acquire_runtime_verbosity(const VerboseOptions& opt) {
  std::lock_guard<std::mutex> lock(g_runtime_verbose_mu);
  if (!g_runtime_saved_env) {
    if (const char* raw = std::getenv("SIMA_NEAT_VERBOSE_LEVEL"); raw && *raw) {
      g_saved_verbose_level_env = std::string(raw);
    } else {
      g_saved_verbose_level_env.reset();
    }
    if (const char* raw = std::getenv("SIMA_NEAT_VERBOSE_TOPICS"); raw && *raw) {
      g_saved_verbose_topics_env = std::string(raw);
    } else {
      g_saved_verbose_topics_env.reset();
    }
    g_runtime_saved_env = true;
  }
  const std::uint64_t id = g_runtime_verbose_next_id++;
  g_runtime_verbose_leases.emplace(id, opt);
  sync_runtime_env_locked();
  return std::static_pointer_cast<void>(std::make_shared<RuntimeVerboseLease>(id));
}

const VerboseOptions& current_verbose_options() {
  return g_verbose_context ? *g_verbose_context : quiet_verbose_options();
}

VerboseOptions current_effective_verbose_options() {
  if (g_verbose_context) {
    return *g_verbose_context;
  }
  std::lock_guard<std::mutex> lock(g_runtime_verbose_mu);
  return aggregate_runtime_verbose_options_locked();
}

bool should_emit_progress(const VerboseOptions& opt) {
  if (!opt.progress) {
    return false;
  }
  if (opt.progress_force) {
    return true;
  }
  if (::isatty(STDERR_FILENO) == 1) {
    return true;
  }
  if (env_truthy("CI")) {
    return false;
  }
  return true;
}

bool should_emit_verbose_details(const VerboseOptions& opt) {
  return opt.level == VerbosityLevel::Verbose;
}

bool should_emit_any_details(const VerboseOptions& opt) {
  return should_emit_topic(opt, VerboseTopic::Planner) ||
         should_emit_topic(opt, VerboseTopic::Graph) ||
         should_emit_topic(opt, VerboseTopic::Pipeline) ||
         should_emit_topic(opt, VerboseTopic::InputStream) ||
         should_emit_topic(opt, VerboseTopic::Tensor) ||
         should_emit_topic(opt, VerboseTopic::Plugins) ||
         should_emit_topic(opt, VerboseTopic::GStreamer);
}

bool should_emit_topic(const VerboseOptions& opt, VerboseTopic topic) {
  if (opt.level == VerbosityLevel::Verbose) {
    return true;
  }
  return topic_enabled_direct(opt, topic);
}

bool should_emit_topic_for_current_context(VerboseTopic topic) {
  return should_emit_topic(current_effective_verbose_options(), topic);
}

bool should_emit_gstreamer(const VerboseOptions& opt) {
  return should_emit_topic(opt, VerboseTopic::GStreamer);
}

bool should_emit_gstreamer_for_current_context() {
  return should_emit_gstreamer(current_effective_verbose_options());
}

void emit_line(const VerboseOptions& opt, const std::string& message) {
  if (!should_emit_progress(opt)) {
    return;
  }
  std::fprintf(stderr, "%s\n", message.c_str());
}

ProgressReporter::ProgressReporter(const VerboseOptions& opt, int total_steps)
    : opt_(opt), total_steps_(std::max(1, total_steps)), start_(std::chrono::steady_clock::now()) {}

void ProgressReporter::step(const std::string& message) {
  if (!should_emit_progress(opt_)) {
    return;
  }
  current_step_ = std::min(total_steps_, current_step_ + 1);
  std::fprintf(stderr, "%s%s\n", step_prefix(current_step_, total_steps_).c_str(),
               message.c_str());
}

void ProgressReporter::detail(const std::string& message) {
  if (!should_emit_verbose_details(opt_)) {
    return;
  }
  std::fprintf(stderr, "      %s\n", message.c_str());
}

void ProgressReporter::done(const std::string& message) {
  if (!should_emit_progress(opt_)) {
    return;
  }
  current_step_ = total_steps_;
  std::fprintf(stderr, "%s%s (%lld ms)\n", step_prefix(current_step_, total_steps_).c_str(),
               message.c_str(), elapsed_ms_since(start_));
}

} // namespace simaai::neat::pipeline_internal::ux
