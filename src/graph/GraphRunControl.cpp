#include "internal/GraphRunState.h"

#include <stdexcept>

namespace simaai::neat::graph {
bool graph_debug_enabled() {
  return env_bool("SIMA_GRAPH_DEBUG", false);
}

bool graph_push_fail_debug_enabled() {
  return env_bool("SIMA_GRAPH_PUSH_FAIL_DEBUG", false);
}

bool stop_trace_enabled() {
  return env_bool("SIMA_STOP_TRACE", false);
}

int graph_output_rate_ms() {
  static int cached = -1;
  if (cached < 0) {
    cached = std::max(0, env_int("SIMA_GRAPH_OUTPUT_RATE_MS", 0));
  }
  return cached;
}

bool graph_output_rate_enabled() {
  return graph_output_rate_ms() > 0;
}

struct GraphOutputRateEntry {
  int64_t count = 0;
  int64_t total_count = 0;
  int64_t total_window_count = 0;
  int64_t total_window_ms = 0;
  int64_t total_reports = 0;
  std::chrono::time_point<std::chrono::steady_clock> last;
  std::chrono::time_point<std::chrono::steady_clock> first;
  std::chrono::time_point<std::chrono::steady_clock> last_sample;
  bool initialized = false;
};

struct GraphOutputRateState {
  std::mutex mu;
  std::unordered_map<NodeId, GraphOutputRateEntry> nodes;
};

GraphOutputRateState& graph_output_rate_state() {
  static GraphOutputRateState state;
  return state;
}

bool graph_sched_debug_enabled() {
  return env_bool("SIMA_GRAPH_SCHED_DEBUG", false);
}

int graph_sched_log_every() {
  static int cached = -1;
  if (cached < 0) {
    cached = std::max(0, env_int("SIMA_GRAPH_SCHED_LOG_EVERY", 0));
  }
  return cached;
}

bool graph_sched_log_first_stream() {
  static int cached = -1;
  if (cached < 0) {
    cached = env_bool("SIMA_GRAPH_SCHED_LOG_FIRST_STREAM", true) ? 1 : 0;
  }
  return cached == 1;
}

bool graph_is_sched_label(const std::string& label) {
  if (label.empty())
    return false;
  std::string lower = label;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower.find("sched") != std::string::npos;
}

struct GraphSchedStats {
  int64_t total = 0;
  std::unordered_map<std::string, int64_t> per_stream;
};

struct GraphSchedState {
  std::mutex mu;
  std::unordered_map<NodeId, GraphSchedStats> nodes;
};

GraphSchedState& graph_sched_state() {
  static GraphSchedState state;
  return state;
}

void graph_sched_record(NodeId node_id, const std::string& label, const Sample& sample) {
  if (!graph_sched_debug_enabled())
    return;
  if (!graph_is_sched_label(label))
    return;
  auto& state = graph_sched_state();
  std::lock_guard<std::mutex> lock(state.mu);
  auto& stats = state.nodes[node_id];
  stats.total++;
  if (!sample.stream_id.empty()) {
    auto& count = stats.per_stream[sample.stream_id];
    count++;
    if (graph_sched_log_first_stream() && count == 1) {
      std::fprintf(stderr, "[GRAPH] sched_assign node=%zu label=%s stream=%s total=%lld\n",
                   static_cast<std::size_t>(node_id), label.c_str(), sample.stream_id.c_str(),
                   static_cast<long long>(stats.total));
    }
  }
  const int every = graph_sched_log_every();
  if (every > 0 && stats.total % every == 0) {
    std::fprintf(stderr, "[GRAPH] sched_count node=%zu label=%s total=%lld\n",
                 static_cast<std::size_t>(node_id), label.c_str(),
                 static_cast<long long>(stats.total));
  }
}

void graph_sched_summary(const std::vector<std::string>* labels) {
  if (!graph_sched_debug_enabled())
    return;
  auto& state = graph_sched_state();
  std::lock_guard<std::mutex> lock(state.mu);
  std::fprintf(stderr, "[GRAPH] sched_summary begin\n");
  for (const auto& [node_id, stats] : state.nodes) {
    std::string label = "<unknown>";
    if (labels && node_id < labels->size() && !(*labels)[node_id].empty()) {
      label = (*labels)[node_id];
    }
    std::vector<std::pair<std::string, int64_t>> items;
    items.reserve(stats.per_stream.size());
    for (const auto& kv : stats.per_stream) {
      items.push_back(kv);
    }
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::fprintf(stderr, "[GRAPH] sched_summary node=%zu label=%s total=%lld streams=%zu\n",
                 static_cast<std::size_t>(node_id), label.c_str(),
                 static_cast<long long>(stats.total), stats.per_stream.size());
    const std::size_t top_n = std::min<std::size_t>(3, items.size());
    for (std::size_t i = 0; i < top_n; ++i) {
      std::fprintf(stderr, "  [GRAPH] sched_top node=%zu label=%s stream=%s count=%lld\n",
                   static_cast<std::size_t>(node_id), label.c_str(), items[i].first.c_str(),
                   static_cast<long long>(items[i].second));
    }
  }
  std::fprintf(stderr, "[GRAPH] sched_summary end\n");
}

void graph_log_output_rate(NodeId node_id, const Sample& sample, const char* label) {
  if (!graph_output_rate_enabled())
    return;
  const auto now = std::chrono::steady_clock::now();
  auto& rate_state = graph_output_rate_state();
  std::lock_guard<std::mutex> lock(rate_state.mu);
  auto& entry = rate_state.nodes[node_id];
  if (!entry.initialized) {
    entry.initialized = true;
    entry.first = now;
    entry.last = now;
    entry.last_sample = now;
    entry.count = 0;
    entry.total_count = 0;
  }
  entry.count++;
  entry.total_count++;
  entry.last_sample = now;
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - entry.last).count();
  const int interval_ms = graph_output_rate_ms();
  if (interval_ms <= 0 || elapsed_ms < interval_ms)
    return;
  const double rate =
      (elapsed_ms > 0)
          ? (static_cast<double>(entry.count) * 1000.0 / static_cast<double>(elapsed_ms))
          : 0.0;
  entry.total_reports++;
  entry.total_window_count += entry.count;
  entry.total_window_ms += static_cast<int64_t>(elapsed_ms);
  const char* safe_label = (label && *label) ? label : "<unknown>";
  std::fprintf(stderr,
               "[GRAPH] output_rate node=%zu label=%s rate=%.1f/s count=%lld window_ms=%lld "
               "stream_id=%s input_seq=%lld\n",
               static_cast<std::size_t>(node_id), safe_label, rate,
               static_cast<long long>(entry.count), static_cast<long long>(elapsed_ms),
               sample.stream_id.c_str(), static_cast<long long>(sample.input_seq));
  entry.count = 0;
  entry.last = now;
}

void graph_output_rate_summary(
    const std::vector<std::string>* labels,
    const std::unordered_map<NodeId, std::shared_ptr<BlockingQueueSample>>* sinks) {
  if (!graph_output_rate_enabled())
    return;
  auto& rate_state = graph_output_rate_state();
  std::lock_guard<std::mutex> lock(rate_state.mu);
  const auto now = std::chrono::steady_clock::now();
  std::fprintf(stderr, "[GRAPH] output_rate_summary begin\n");
  const std::size_t max_nodes = labels ? labels->size() : 0;
  for (std::size_t node_id = 0; node_id < max_nodes; ++node_id) {
    auto it = rate_state.nodes.find(node_id);
    if (it == rate_state.nodes.end())
      continue;
    const auto& entry = it->second;
    if (!entry.initialized)
      continue;
    const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(entry.last_sample - entry.first)
            .count();
    const auto last_age_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - entry.last_sample).count();
    const double overall_rate =
        (total_ms > 0)
            ? (static_cast<double>(entry.total_count) * 1000.0 / static_cast<double>(total_ms))
            : 0.0;
    const double window_rate = (entry.total_window_ms > 0)
                                   ? (static_cast<double>(entry.total_window_count) * 1000.0 /
                                      static_cast<double>(entry.total_window_ms))
                                   : 0.0;
    const auto& label = labels ? (*labels)[node_id] : std::string();
    int queue_depth = -1;
    if (sinks) {
      auto it_sink = sinks->find(node_id);
      if (it_sink != sinks->end() && it_sink->second) {
        queue_depth = static_cast<int>(it_sink->second->size());
      }
    }
    std::fprintf(stderr,
                 "[GRAPH] output_rate_summary node=%zu label=%s total=%lld overall_rate=%.2f/s "
                 "window_rate=%.2f/s windows=%lld total_window_ms=%lld last_output_age_ms=%lld "
                 "queue_depth=%d\n",
                 node_id, label.empty() ? "<unknown>" : label.c_str(),
                 static_cast<long long>(entry.total_count), overall_rate, window_rate,
                 static_cast<long long>(entry.total_reports),
                 static_cast<long long>(entry.total_window_ms), static_cast<long long>(last_age_ms),
                 queue_depth);
  }
  std::fprintf(stderr, "[GRAPH] output_rate_summary end\n");
}

const std::unordered_set<std::string>& first_frame_streams() {
  static std::unordered_set<std::string> streams = [] {
    std::unordered_set<std::string> out;
    const char* env = std::getenv("SIMA_FIRST_FRAME_STREAMS");
    if (!env || !*env)
      return out;
    std::string raw(env);
    std::string token;
    for (char c : raw) {
      if (c == ',') {
        if (!token.empty())
          out.insert(token);
        token.clear();
        continue;
      }
      if (!std::isspace(static_cast<unsigned char>(c)))
        token.push_back(c);
    }
    if (!token.empty())
      out.insert(token);
    return out;
  }();
  return streams;
}

bool should_log_first_frame(const std::string& stream_id) {
  const auto& streams = first_frame_streams();
  return !streams.empty() && !stream_id.empty() && streams.find(stream_id) != streams.end();
}

void log_first_decoded_once(const Sample& sample, std::size_t segment_id) {
  if (sample.media_type != "video/x-raw")
    return;
  if (!should_log_first_frame(sample.stream_id))
    return;
  static std::mutex mu;
  static std::unordered_set<std::string> seen;
  std::lock_guard<std::mutex> lock(mu);
  if (!seen.insert(sample.stream_id).second)
    return;
  const char* label =
      !sample.stream_label.empty() ? sample.stream_label.c_str() : sample.port_name.c_str();
  std::fprintf(stderr, "[GRAPH] first_decoded stream=%s frame_id=%lld seg=%zu port=%s\n",
               sample.stream_id.c_str(), static_cast<long long>(sample.frame_id), segment_id,
               label);
}

void graph_debug_sample(const char* tag, const Sample& sample) {
  if (!graph_debug_enabled())
    return;
  std::string kind;
  switch (sample.kind) {
  case SampleKind::TensorSet:
    kind = "TensorSet";
    break;
  case SampleKind::Bundle:
    kind = "Bundle";
    break;
  default:
    kind = "Unknown";
    break;
  }
  const char* label =
      !sample.stream_label.empty() ? sample.stream_label.c_str() : sample.port_name.c_str();
  std::fprintf(stderr,
               "[GRAPH] %s kind=%s frame_id=%lld input_seq=%lld orig_input_seq=%lld stream_id=%s "
               "port=%s pts_ns=%lld dts_ns=%lld duration_ns=%lld media=%s format=%s tag=%s\n",
               tag, kind.c_str(), static_cast<long long>(sample.frame_id),
               static_cast<long long>(sample.input_seq),
               static_cast<long long>(sample.orig_input_seq), sample.stream_id.c_str(), label,
               static_cast<long long>(sample.pts_ns), static_cast<long long>(sample.dts_ns),
               static_cast<long long>(sample.duration_ns), sample.media_type.c_str(),
               sample.format.c_str(), sample.payload_tag.c_str());
}

int pull_step_timeout_ms(bool has_timeout, std::chrono::steady_clock::time_point deadline,
                         std::size_t output_count) {
  if (!has_timeout)
    return 50;
  const auto now = std::chrono::steady_clock::now();
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
  if (remaining <= 0)
    return -1;
  return static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(50, remaining / static_cast<int64_t>(output_count) + 1)));
}

std::optional<Sample> pull_round_robin_once(const std::vector<GraphRun::Output>& outputs,
                                            std::size_t start, int step_ms, GraphRunStats* stats,
                                            NodeId* out_node) {
  const std::size_t count = outputs.size();
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t idx = (start + i) % count;
    auto sample = outputs[idx].pull(step_ms, stats);
    if (!sample.has_value())
      continue;
    if (out_node)
      *out_node = outputs[idx].node_id();
    return sample;
  }
  return std::nullopt;
}

void validate_pull_until_args(const std::vector<GraphRun::Output>& outputs,
                              const GraphRunPullOptions& opt) {
  if (outputs.empty()) {
    throw std::runtime_error("GraphRun::pull_until requires at least one output");
  }
  if (opt.per_stream_target <= 0) {
    throw std::runtime_error("GraphRun::pull_until requires per_stream_target > 0");
  }
}

void validate_collect_args(bool has_state, const std::vector<GraphRun::Output>& outputs,
                           const GraphRunPullOptions& opt, GraphRunStats* stats) {
  if (!has_state || outputs.empty()) {
    throw std::runtime_error("GraphRun::collect requires at least one output");
  }
  if (opt.per_stream_target <= 0) {
    throw std::runtime_error("GraphRun::collect requires per_stream_target > 0");
  }
  if (!stats) {
    throw std::runtime_error("GraphRun::collect requires stats");
  }
}

std::string join_stream_id_set(const std::unordered_set<std::string>& ids) {
  std::string out;
  for (const auto& sid : ids) {
    if (!out.empty())
      out += ", ";
    out += sid;
  }
  return out;
}

GraphRun::GraphRun(std::shared_ptr<State> state) : state_(std::move(state)) {}

GraphRun::GraphRun(GraphRun&& other) noexcept : state_(std::move(other.state_)) {}

GraphRun& GraphRun::operator=(GraphRun&& other) noexcept {
  if (this != &other) {
    stop();
    state_ = std::move(other.state_);
  }
  return *this;
}

GraphRun::~GraphRun() {
  if (graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] GraphRun::~GraphRun\n");
  }
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::~GraphRun begin\n");
  }
  stop();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] GraphRun::~GraphRun end\n");
  }
}

GraphRun::operator bool() const noexcept {
  return static_cast<bool>(state_);
}

bool GraphRun::running() const {
  return state_ && !state_->stop_requested();
}

bool GraphRun::push_state(const std::shared_ptr<State>& state, NodeId node_id, PortId port,
                          bool has_port, const Sample& sample) {
  if (!state) {
    std::fprintf(stderr, "[GRAPH] GraphRun::push failed: run state is null (graph not started or "
                         "already stopped)\n");
    return false;
  }
  return state->core->graph_push(node_id, port, has_port, sample,
                                 state->core->graph_options.router_options());
}

bool GraphRun::push(NodeId node_id, const Sample& samples) {
  if (samples.empty()) {
    throw std::runtime_error("GraphRun::push: empty sample list");
  }
  for (const auto& sample : samples) {
    if (!push_state(state_, node_id, kInvalidPort, false, sample)) {
      return false;
    }
  }
  return true;
}

bool GraphRun::push(NodeId node_id, PortId port, const Sample& samples) {
  if (samples.empty()) {
    throw std::runtime_error("GraphRun::push: empty sample list");
  }
  for (const auto& sample : samples) {
    if (!push_state(state_, node_id, port, true, sample)) {
      return false;
    }
  }
  return true;
}

std::optional<Sample> GraphRun::pull_state(const std::shared_ptr<State>& state, NodeId node_id,
                                           int timeout_ms) {
  if (!state) {
    std::fprintf(stderr, "[GRAPH] GraphRun::pull failed: run state is null (graph not started or "
                         "already stopped)\n");
    return std::nullopt;
  }
  return state->core->graph_pull(node_id, timeout_ms);
}

std::optional<Sample> GraphRun::pull(NodeId node_id, int timeout_ms) {
  return pull_state(state_, node_id, timeout_ms);
}

GraphRun::Input GraphRun::input(NodeId node_id) {
  return Input(state_, node_id, kInvalidPort, false);
}

GraphRun::Input GraphRun::input(NodeId node_id, PortId port) {
  return Input(state_, node_id, port, true);
}

GraphRun::Output GraphRun::output(NodeId node_id) {
  return Output(state_, node_id);
}

GraphRunStats& GraphRun::enable_stats() {
  if (!state_ || !state_->core) {
    static GraphRunStats dummy;
    return dummy;
  }
  if (!state_->core->graph_stats) {
    state_->core->graph_stats = std::make_shared<GraphRunStats>();
  }
  return *state_->core->graph_stats;
}

const GraphRunStats* GraphRun::stats() const {
  if (!state_ || !state_->core)
    return nullptr;
  return state_->core->graph_stats.get();
}

bool GraphRun::Input::push(const Sample& samples) const {
  auto state = state_.lock();
  if (!state) {
    std::fprintf(stderr, "[GRAPH] Input::push failed: GraphRun handle is null\n");
    return false;
  }
  if (samples.empty()) {
    throw std::runtime_error("GraphRun::Input::push: empty sample list");
  }
  for (const auto& sample : samples) {
    if (!GraphRun::push_state(state, node_, port_, has_port_, sample)) {
      return false;
    }
  }
  return true;
}

std::optional<Sample> GraphRun::Output::pull(int timeout_ms, GraphRunStats* stats) const {
  auto state = state_.lock();
  if (!state)
    return std::nullopt;
  GraphRunStats* record_stats = stats;
  if (!record_stats && state->core && state->core->graph_stats) {
    record_stats = state->core->graph_stats.get();
  }
  auto result = GraphRun::pull_state(state, node_, timeout_ms);
  if (result.has_value() && graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] output_pull node=%zu\n", static_cast<std::size_t>(node_));
    graph_debug_sample("output_pull", *result);
  }
  if (result.has_value()) {
    const char* label = nullptr;
    if (node_ < state->execution().node_labels.size()) {
      label = state->execution().node_labels[node_].c_str();
    }
    graph_log_output_rate(node_, *result, label);
    if (record_stats) {
      record_stats->record(node_, *result);
    }
  }
  return result;
}

Sample GraphRun::Output::pull_or_throw(int timeout_ms, GraphRunStats* stats) const {
  auto result = pull(timeout_ms, stats);
  if (result.has_value())
    return *result;
  if (auto state = state_.lock()) {
    const std::string err = state->core ? state->core->last_error() : std::string{};
    if (!err.empty())
      throw std::runtime_error(err);
  }
  throw std::runtime_error("GraphRun::pull timed out (timeout_ms=" + std::to_string(timeout_ms) +
                           ", node=" + std::to_string(static_cast<std::size_t>(node_)) + ")");
}

void GraphRunStats::record(NodeId node_id, const Sample& sample) {
  const auto now = std::chrono::steady_clock::now();
  const std::string sid = sample.stream_id.empty() ? "<empty>" : sample.stream_id;
  std::lock_guard<std::mutex> lock(mu_);
  auto& node = nodes_[node_id];
  node.total++;
  if (!node.initialized) {
    node.initialized = true;
    node.first = now;
  }
  node.last = now;
  auto& stream = node.streams[sid];
  stream.count++;
  if (!stream.initialized) {
    stream.initialized = true;
    stream.first = now;
  }
  stream.last = now;
}

std::vector<GraphRunStats::Snapshot> GraphRunStats::snapshot() const {
  std::vector<Snapshot> out;
  std::lock_guard<std::mutex> lock(mu_);
  out.reserve(nodes_.size());
  for (const auto& it : nodes_) {
    Snapshot snap;
    snap.node_id = it.first;
    snap.total = it.second.total;
    snap.first = it.second.first;
    snap.last = it.second.last;
    for (const auto& kv : it.second.streams) {
      snap.counts.emplace(kv.first, kv.second.count);
      snap.last_seen.emplace(kv.first, kv.second.last);
    }
    out.push_back(std::move(snap));
  }
  return out;
}

bool GraphRunStats::empty() const {
  std::lock_guard<std::mutex> lock(mu_);
  return nodes_.empty();
}

std::unordered_map<std::string, int64_t>
GraphRunStats::stream_counts(const std::vector<NodeId>& nodes) const {
  std::unordered_map<std::string, int64_t> counts;
  std::lock_guard<std::mutex> lock(mu_);
  if (nodes.empty()) {
    for (const auto& node_it : nodes_) {
      for (const auto& stream_it : node_it.second.streams) {
        counts[stream_it.first] += stream_it.second.count;
      }
    }
    return counts;
  }
  for (const auto node_id : nodes) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end())
      continue;
    for (const auto& stream_it : it->second.streams) {
      counts[stream_it.first] += stream_it.second.count;
    }
  }
  return counts;
}

bool GraphRunStats::has_missing_streams(const std::unordered_set<std::string>& expected,
                                        const std::vector<NodeId>& nodes) const {
  if (expected.empty())
    return false;
  const auto counts = stream_counts(nodes);
  for (const auto& sid : expected) {
    auto it = counts.find(sid);
    if (it == counts.end() || it->second <= 0)
      return true;
  }
  return false;
}

std::string GraphRunStats::missing_streams_list(const std::unordered_set<std::string>& expected,
                                                const std::vector<NodeId>& nodes) const {
  if (expected.empty())
    return {};
  const auto counts = stream_counts(nodes);
  std::string missing;
  for (const auto& sid : expected) {
    auto it = counts.find(sid);
    if (it == counts.end() || it->second <= 0) {
      if (!missing.empty())
        missing += ", ";
      missing += sid;
    }
  }
  return missing;
}

GraphRun::StallGuard::StallGuard(std::vector<NodeId> nodes, std::vector<std::string> streams,
                                 int per_stream_target, int stall_ms)
    : nodes_(std::move(nodes)), streams_(std::move(streams)), per_stream_target_(per_stream_target),
      stall_ms_(stall_ms) {}

bool GraphRun::StallGuard::update(const GraphRunStats& stats) {
  if (per_stream_target_ <= 0) {
    std::fprintf(stderr, "[GRAPH] StallGuard::update: invalid per_stream_target=%d (must be > 0)\n",
                 per_stream_target_);
    return false;
  }
  std::unordered_map<std::string, int64_t> counts;
  const auto snaps = stats.snapshot();
  for (const auto& snap : snaps) {
    if (!nodes_.empty() && std::find(nodes_.begin(), nodes_.end(), snap.node_id) == nodes_.end()) {
      continue;
    }
    for (const auto& kv : snap.counts) {
      counts[kv.first] += kv.second;
    }
  }
  if (streams_.empty() && !counts.empty()) {
    streams_.reserve(counts.size());
    for (const auto& kv : counts) {
      streams_.push_back(kv.first);
    }
  }
  if (streams_.empty())
    return false;

  int64_t progress = 0;
  bool done = true;
  for (const auto& sid : streams_) {
    const int64_t count = counts[sid];
    progress += std::min<int64_t>(count, per_stream_target_);
    if (count < per_stream_target_)
      done = false;
  }
  const auto now = std::chrono::steady_clock::now();
  if (!initialized_) {
    initialized_ = true;
    target_progress_ = progress;
    last_progress = now;
  }
  if (progress > target_progress_) {
    target_progress_ = progress;
    last_progress = now;
  }
  done_ = done;
  stalled_ =
      (!done_ && stall_ms_ > 0 && now - last_progress > std::chrono::milliseconds(stall_ms_));
  return stalled_;
}

std::optional<Sample> GraphRun::pull_any_state(const std::shared_ptr<State>& state,
                                               const std::vector<Output>& outputs, int timeout_ms,
                                               GraphRunStats* stats, NodeId* out_node) {
  if (!state || outputs.empty())
    return std::nullopt;
  const bool has_timeout = (timeout_ms >= 0);
  const auto deadline =
      has_timeout ? (std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms))
                  : std::chrono::steady_clock::time_point{};
  const std::size_t count = outputs.size();
  const std::size_t start = state->execution().output_rr.fetch_add(1);
  while (true) {
    if (has_timeout && std::chrono::steady_clock::now() >= deadline) {
      return std::nullopt;
    }
    const int step_ms = pull_step_timeout_ms(has_timeout, deadline, count);
    if (step_ms < 0)
      return std::nullopt;
    auto sample = pull_round_robin_once(outputs, start, step_ms, stats, out_node);
    if (sample.has_value())
      return sample;
  }
}

std::optional<Sample> GraphRun::pull_any(const std::vector<Output>& outputs, int timeout_ms,
                                         GraphRunStats* stats, NodeId* out_node) {
  return pull_any_state(state_, outputs, timeout_ms, stats, out_node);
}

GraphRun::StallGuard GraphRun::stall_guard(const std::vector<Output>& outputs,
                                           int per_stream_target, int stall_ms,
                                           std::vector<std::string> stream_ids) {
  std::vector<NodeId> nodes;
  nodes.reserve(outputs.size());
  for (const auto& out : outputs) {
    nodes.push_back(out.node_);
  }
  return StallGuard(std::move(nodes), std::move(stream_ids), per_stream_target, stall_ms);
}

void GraphRun::pull_until_state(const std::shared_ptr<State>& state,
                                const std::vector<Output>& outputs, GraphRunStats& stats,
                                const GraphRunPullOptions& opt,
                                const std::function<void(const Sample&, NodeId)>& on_sample) {
  if (!state) {
    throw std::runtime_error("GraphRun::pull_until requires a live run");
  }
  validate_pull_until_args(outputs, opt);
  std::vector<NodeId> nodes;
  nodes.reserve(outputs.size());
  for (const auto& out : outputs) {
    nodes.push_back(out.node_);
  }
  StallGuard guard(std::move(nodes), opt.stream_ids, opt.per_stream_target, opt.stall_ms);
  const auto t0 = std::chrono::steady_clock::now();
  while (true) {
    NodeId out_node = kInvalidNode;
    auto sample = pull_any_state(state, outputs, opt.timeout_ms, &stats, &out_node);
    if (sample.has_value()) {
      if (on_sample) {
        on_sample(*sample, out_node);
      }
    } else {
      const std::string err = state->core ? state->core->last_error() : std::string{};
      if (!err.empty()) {
        throw std::runtime_error(err);
      }
    }
    guard.update(stats);
    if (guard.done())
      return;
    if (guard.stalled()) {
      throw std::runtime_error(
          "GraphRun: stalled waiting for per-stream outputs (per_stream_target=" +
          std::to_string(opt.per_stream_target) + ", stall_ms=" + std::to_string(opt.stall_ms) +
          ", streams=" + std::to_string(opt.stream_ids.size()) + ")");
    }
    if (opt.max_runtime_ms > 0 &&
        std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(opt.max_runtime_ms)) {
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
      throw std::runtime_error(
          "GraphRun: max runtime exceeded (max_runtime_ms=" + std::to_string(opt.max_runtime_ms) +
          ", elapsed_ms=" + std::to_string(elapsed_ms) + ")");
    }
  }
}

void GraphRun::pull_until(const std::vector<Output>& outputs, GraphRunStats& stats,
                          const GraphRunPullOptions& opt,
                          const std::function<void(const Sample&, NodeId)>& on_sample) {
  pull_until_state(state_, outputs, stats, opt, on_sample);
}

GraphRun::PullLoop::PullLoop(std::weak_ptr<State> state, std::vector<Output> outputs,
                             std::vector<NodeId> output_nodes, GraphRunStats* stats)
    : state_(std::move(state)), outputs_(std::move(outputs)),
      output_nodes_(std::move(output_nodes)), stats_(stats) {
  if (auto state = state_.lock()) {
    opt_.timeout_ms = state->core->graph_options.pull_timeout_ms;
  }
}

GraphRun::PullLoop& GraphRun::PullLoop::per_stream_target(int n) {
  opt_.per_stream_target = n;
  return *this;
}

GraphRun::PullLoop& GraphRun::PullLoop::stall_after_ms(int ms) {
  opt_.stall_ms = ms;
  return *this;
}

GraphRun::PullLoop& GraphRun::PullLoop::timeout_ms(int ms) {
  opt_.timeout_ms = ms;
  return *this;
}

GraphRun::PullLoop& GraphRun::PullLoop::max_runtime_ms(int ms) {
  opt_.max_runtime_ms = ms;
  return *this;
}

GraphRun::PullLoop& GraphRun::PullLoop::expect_streams(std::vector<std::string> ids) {
  expected_.clear();
  expected_.insert(ids.begin(), ids.end());
  opt_.stream_ids = std::move(ids);
  return *this;
}

GraphRun::PullLoop& GraphRun::PullLoop::on_sample(std::function<void(const Sample&, NodeId)> cb) {
  on_sample_ = std::move(cb);
  return *this;
}

GraphRunStats& GraphRun::PullLoop::stats() {
  if (!stats_) {
    throw std::runtime_error("GraphRun::PullLoop missing stats");
  }
  return *stats_;
}

void GraphRun::PullLoop::run() {
  auto state = state_.lock();
  validate_collect_args(static_cast<bool>(state), outputs_, opt_, stats_);
  if (!expected_.empty() && opt_.stream_ids.empty()) {
    opt_.stream_ids.assign(expected_.begin(), expected_.end());
  }

  GraphRun::pull_until_state(
      state, outputs_, *stats_, opt_, [&](const Sample& sample, NodeId out_node) {
        if (sample.stream_id.empty()) {
          saw_empty_stream_id_ = true;
        } else if (!expected_.empty() && expected_.find(sample.stream_id) == expected_.end()) {
          unknown_.insert(sample.stream_id);
        }
        if (on_sample_)
          on_sample_(sample, out_node);
      });

  if (!expected_.empty()) {
    if (saw_empty_stream_id_) {
      throw std::runtime_error("GraphRun: output missing stream_id");
    }
    if (!unknown_.empty()) {
      throw std::runtime_error("GraphRun: unexpected stream_id(s): " +
                               join_stream_id_set(unknown_));
    }
    const std::string missing = stats_->missing_streams_list(expected_, output_nodes_);
    if (!missing.empty()) {
      throw std::runtime_error("GraphRun: missing outputs for streams: " + missing);
    }
  }
}

GraphRun::PullLoop GraphRun::collect(const std::vector<Output>& outputs, GraphRunStats* stats) {
  GraphRunStats* use_stats = stats;
  if (!use_stats) {
    use_stats = &enable_stats();
  }
  std::vector<NodeId> nodes;
  nodes.reserve(outputs.size());
  for (const auto& out : outputs) {
    nodes.push_back(out.node_);
  }
  return PullLoop(state_, outputs, std::move(nodes), use_stats);
}

std::string GraphRun::last_error() const {
  if (!state_)
    return {};
  return state_->core ? state_->core->last_error() : std::string{};
}

void GraphRun::last_error_or_throw() const {
  const std::string err = last_error();
  if (!err.empty()) {
    throw std::runtime_error(err);
  }
}
} // namespace simaai::neat::graph
