#include "internal/GraphRunState.h"

namespace simaai::neat::graph {
bool graph_debug_enabled() {
  return env_bool("SIMA_GRAPH_DEBUG", false);
}

bool graph_push_fail_debug_enabled() {
  return env_bool("SIMA_GRAPH_PUSH_FAIL_DEBUG", false);
}

bool graph_serial_pipeline_build_enabled() {
  return env_bool("SIMA_GRAPH_SERIAL_PIPELINE_BUILD", false);
}

std::mutex& graph_pipeline_build_mu() {
  static std::mutex mu;
  return mu;
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

void log_first_decoded_once(const Sample& sample, const CompiledPipelineSegment& seg) {
  if (sample.media_type != "video/x-raw")
    return;
  if (!should_log_first_frame(sample.stream_id))
    return;
  static std::mutex mu;
  static std::unordered_set<std::string> seen;
  std::lock_guard<std::mutex> lock(mu);
  if (!seen.insert(sample.stream_id).second)
    return;
  std::fprintf(stderr, "[GRAPH] first_decoded stream=%s frame_id=%lld seg=%zu port=%s\n",
               sample.stream_id.c_str(), static_cast<long long>(sample.frame_id),
               static_cast<std::size_t>(seg.id), sample.port_name.c_str());
}

void graph_debug_sample(const char* tag, const Sample& sample) {
  if (!graph_debug_enabled())
    return;
  std::string kind;
  switch (sample.kind) {
  case SampleKind::Tensor:
    kind = "Tensor";
    break;
  case SampleKind::Bundle:
    kind = "Bundle";
    break;
  default:
    kind = "Unknown";
    break;
  }
  std::fprintf(stderr,
               "[GRAPH] %s kind=%s frame_id=%lld input_seq=%lld orig_input_seq=%lld stream_id=%s "
               "port=%s media=%s format=%s tag=%s\n",
               tag, kind.c_str(), static_cast<long long>(sample.frame_id),
               static_cast<long long>(sample.input_seq),
               static_cast<long long>(sample.orig_input_seq), sample.stream_id.c_str(),
               sample.port_name.c_str(), sample.media_type.c_str(), sample.format.c_str(),
               sample.payload_tag.c_str());
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

void validate_collect_args(GraphRun* run, const std::vector<GraphRun::Output>* outputs,
                           const GraphRunPullOptions& opt, GraphRunStats* stats) {
  if (!run || !outputs || outputs->empty()) {
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
  return state_ && !state_->stop.load();
}

bool GraphRun::push(NodeId node_id, const Sample& sample) {
  if (!state_)
    return false;
  auto it_pipe = state_->node_to_pipeline.find(node_id);
  if (it_pipe != state_->node_to_pipeline.end()) {
    auto& pipe = *state_->pipelines[it_pipe->second];
    if (!pipe.input_queue)
      return false;
    std::string build_err;
    if (!state_->ensure_pipeline_built(it_pipe->second, sample, &build_err)) {
      state_->request_stop(build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
      return false;
    }
    Sample copy = sample;
    if (pipe.input_queue) {
      const std::size_t qsize = pipe.input_queue->size();
      maybe_force_copy_for_backpressure(copy, qsize, "graph_push", pipe.seg.id);
    }
    const bool ok = pipe.input_queue->push(std::move(copy), state_->opt.push_timeout_ms);
    if (!ok && !state_->stop.load(std::memory_order_relaxed)) {
      std::ostringstream msg;
      msg << "GraphRun::push timed out waiting for pipeline input queue (seg="
          << static_cast<std::size_t>(pipe.seg.id) << ", edge_queue=" << state_->opt.edge_queue
          << ", push_timeout_ms=" << state_->opt.push_timeout_ms
          << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
      state_->request_stop(msg.str());
    }
    return ok;
  }
  auto it_stage = state_->node_to_stage_group.find(node_id);
  if (it_stage != state_->node_to_stage_group.end()) {
    PortId in_port = kInvalidPort;
    const auto& group = state_->stage_groups[it_stage->second];
    if (!group.instances.empty()) {
      const auto& st = *state_->stages[group.instances.front()];
      if (st.input_ports.size() == 1) {
        in_port = st.input_ports[0];
      }
    }
    Sample copy = sample;
    return state_->dispatch_to_stage_group(it_stage->second, in_port, std::move(copy));
  }
  return false;
}

bool GraphRun::push(NodeId node_id, PortId port, const Sample& sample) {
  if (!state_)
    return false;
  auto it_pipe = state_->node_to_pipeline.find(node_id);
  if (it_pipe != state_->node_to_pipeline.end()) {
    auto& pipe = *state_->pipelines[it_pipe->second];
    if (!pipe.input_queue)
      return false;
    std::string build_err;
    if (!state_->ensure_pipeline_built(it_pipe->second, sample, &build_err)) {
      state_->request_stop(build_err.empty() ? "GraphRun: pipeline build failed" : build_err);
      return false;
    }
    Sample copy = sample;
    if (pipe.input_queue) {
      const std::size_t qsize = pipe.input_queue->size();
      maybe_force_copy_for_backpressure(copy, qsize, "graph_push", pipe.seg.id);
    }
    const bool ok = pipe.input_queue->push(std::move(copy), state_->opt.push_timeout_ms);
    if (!ok && !state_->stop.load(std::memory_order_relaxed)) {
      std::ostringstream msg;
      msg << "GraphRun::push timed out waiting for pipeline input queue (seg="
          << static_cast<std::size_t>(pipe.seg.id) << ", edge_queue=" << state_->opt.edge_queue
          << ", push_timeout_ms=" << state_->opt.push_timeout_ms
          << "). Increase GraphRunOptions.edge_queue or pull outputs concurrently.";
      state_->request_stop(msg.str());
    }
    return ok;
  }
  auto it_stage = state_->node_to_stage_group.find(node_id);
  if (it_stage != state_->node_to_stage_group.end()) {
    Sample copy = sample;
    return state_->dispatch_to_stage_group(it_stage->second, port, std::move(copy));
  }
  return false;
}

std::optional<Sample> GraphRun::pull(NodeId node_id, int timeout_ms) {
  if (!state_)
    return std::nullopt;
  const bool has_timeout = (timeout_ms >= 0);
  auto it_pipe = state_->node_to_pipeline.find(node_id);
  if (it_pipe != state_->node_to_pipeline.end()) {
    auto& pipe = *state_->pipelines[it_pipe->second];
    if (!pipe.built.load(std::memory_order_acquire)) {
      if (graph_debug_enabled()) {
        std::fprintf(stderr, "[GRAPH] pull_wait_for_build seg=%zu\n",
                     static_cast<std::size_t>(pipe.seg.id));
      }
      const int build_timeout_ms =
          has_timeout ? std::max(timeout_ms, env_int("SIMA_GRAPH_BUILD_TIMEOUT_MS", 5000))
                      : env_int("SIMA_GRAPH_BUILD_TIMEOUT_MS", -1);
      std::unique_lock<std::mutex> lock(pipe.mu);
      if (has_timeout && build_timeout_ms >= 0) {
        pipe.cv.wait_until(
            lock, std::chrono::steady_clock::now() + std::chrono::milliseconds(build_timeout_ms),
            [&] { return pipe.built.load(std::memory_order_acquire) || state_->stop.load(); });
      } else {
        pipe.cv.wait(lock, [&] {
          return pipe.built.load(std::memory_order_acquire) || state_->stop.load();
        });
      }
      if (!pipe.built.load(std::memory_order_acquire)) {
        if (graph_debug_enabled()) {
          std::fprintf(stderr, "[GRAPH] pull_wait_for_build_timeout seg=%zu\n",
                       static_cast<std::size_t>(pipe.seg.id));
        }
        return std::nullopt;
      }
    }
  }
  auto it = state_->sinks.find(node_id);
  if (it == state_->sinks.end())
    return std::nullopt;
  Sample out;
  const int wait_ms = has_timeout ? timeout_ms : -1;
  if (!it->second->pop(out, wait_ms))
    return std::nullopt;
  return out;
}

GraphRun::Input GraphRun::input(NodeId node_id) {
  return Input(this, node_id, kInvalidPort, false);
}

GraphRun::Input GraphRun::input(NodeId node_id, PortId port) {
  return Input(this, node_id, port, true);
}

GraphRun::Output GraphRun::output(NodeId node_id) {
  return Output(this, node_id);
}

GraphRunStats& GraphRun::enable_stats() {
  if (!state_) {
    static GraphRunStats dummy;
    return dummy;
  }
  if (!state_->stats) {
    state_->stats = std::make_shared<GraphRunStats>();
  }
  return *state_->stats;
}

const GraphRunStats* GraphRun::stats() const {
  if (!state_)
    return nullptr;
  return state_->stats.get();
}

bool GraphRun::Input::push(const Sample& sample) const {
  if (!run_)
    return false;
  if (has_port_)
    return run_->push(node_, port_, sample);
  return run_->push(node_, sample);
}

std::optional<Sample> GraphRun::Output::pull(int timeout_ms, GraphRunStats* stats) const {
  if (!run_)
    return std::nullopt;
  GraphRunStats* record_stats = stats;
  if (!record_stats && run_->state_ && run_->state_->stats) {
    record_stats = run_->state_->stats.get();
  }
  auto result = run_->pull(node_, timeout_ms);
  if (result.has_value() && graph_debug_enabled()) {
    std::fprintf(stderr, "[GRAPH] output_pull node=%zu\n", static_cast<std::size_t>(node_));
    graph_debug_sample("output_pull", *result);
  }
  if (result.has_value()) {
    const char* label = nullptr;
    if (run_ && run_->state_ && node_ < run_->state_->node_labels.size()) {
      label = run_->state_->node_labels[node_].c_str();
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
  if (run_) {
    const std::string err = run_->last_error();
    if (!err.empty())
      throw std::runtime_error(err);
  }
  throw std::runtime_error("GraphRun::pull timed out");
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
  if (per_stream_target_ <= 0)
    return false;
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

std::optional<Sample> GraphRun::pull_any(const std::vector<Output>& outputs, int timeout_ms,
                                         GraphRunStats* stats, NodeId* out_node) {
  if (!state_ || outputs.empty())
    return std::nullopt;
  const bool has_timeout = (timeout_ms >= 0);
  const auto deadline =
      has_timeout ? (std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms))
                  : std::chrono::steady_clock::time_point{};
  const std::size_t count = outputs.size();
  const std::size_t start = state_->output_rr.fetch_add(1);
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

bool GraphRun::warmup(const std::vector<Output>& outputs, int warmup_count, int timeout_ms) {
  if (!state_ || outputs.empty())
    return false;
  if (warmup_count <= 0)
    return true;
  for (int i = 0; i < warmup_count; ++i) {
    auto sample = pull_any(outputs, timeout_ms);
    if (sample.has_value())
      continue;
    if (!last_error().empty())
      return false;
    return false;
  }
  return true;
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

void GraphRun::pull_until(const std::vector<Output>& outputs, GraphRunStats& stats,
                          const GraphRunPullOptions& opt,
                          const std::function<void(const Sample&, NodeId)>& on_sample) {
  validate_pull_until_args(outputs, opt);
  StallGuard guard = stall_guard(outputs, opt.per_stream_target, opt.stall_ms, opt.stream_ids);
  const auto t0 = std::chrono::steady_clock::now();
  while (true) {
    NodeId out_node = kInvalidNode;
    auto sample = pull_any(outputs, opt.timeout_ms, &stats, &out_node);
    if (sample.has_value()) {
      if (on_sample) {
        on_sample(*sample, out_node);
      }
    } else {
      last_error_or_throw();
    }
    guard.update(stats);
    if (guard.done())
      return;
    if (guard.stalled()) {
      throw std::runtime_error("GraphRun: stalled waiting for per-stream outputs");
    }
    if (opt.max_runtime_ms > 0 &&
        std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(opt.max_runtime_ms)) {
      throw std::runtime_error("GraphRun: max runtime exceeded");
    }
  }
}

GraphRun::PullSession::PullSession(GraphRun* run, const std::vector<Output>* outputs,
                                   std::vector<NodeId> output_nodes, GraphRunStats* stats)
    : run_(run), outputs_(outputs), output_nodes_(std::move(output_nodes)), stats_(stats) {
  if (run_ && run_->state_) {
    opt_.timeout_ms = run_->state_->opt.pull_timeout_ms;
  }
}

GraphRun::PullSession& GraphRun::PullSession::per_stream_target(int n) {
  opt_.per_stream_target = n;
  return *this;
}

GraphRun::PullSession& GraphRun::PullSession::stall_after_ms(int ms) {
  opt_.stall_ms = ms;
  return *this;
}

GraphRun::PullSession& GraphRun::PullSession::timeout_ms(int ms) {
  opt_.timeout_ms = ms;
  return *this;
}

GraphRun::PullSession& GraphRun::PullSession::max_runtime_ms(int ms) {
  opt_.max_runtime_ms = ms;
  return *this;
}

GraphRun::PullSession& GraphRun::PullSession::expect_streams(std::vector<std::string> ids) {
  expected_.clear();
  expected_.insert(ids.begin(), ids.end());
  opt_.stream_ids = std::move(ids);
  return *this;
}

GraphRun::PullSession&
GraphRun::PullSession::on_sample(std::function<void(const Sample&, NodeId)> cb) {
  on_sample_ = std::move(cb);
  return *this;
}

GraphRunStats& GraphRun::PullSession::stats() {
  if (!stats_) {
    throw std::runtime_error("GraphRun::PullSession missing stats");
  }
  return *stats_;
}

void GraphRun::PullSession::run() {
  validate_collect_args(run_, outputs_, opt_, stats_);
  if (!expected_.empty() && opt_.stream_ids.empty()) {
    opt_.stream_ids.assign(expected_.begin(), expected_.end());
  }

  run_->pull_until(*outputs_, *stats_, opt_, [&](const Sample& sample, NodeId out_node) {
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

GraphRun::PullSession GraphRun::collect(const std::vector<Output>& outputs, GraphRunStats* stats) {
  GraphRunStats* use_stats = stats;
  if (!use_stats) {
    use_stats = &enable_stats();
  }
  std::vector<NodeId> nodes;
  nodes.reserve(outputs.size());
  for (const auto& out : outputs) {
    nodes.push_back(out.node_);
  }
  return PullSession(this, &outputs, std::move(nodes), use_stats);
}

std::string GraphRun::last_error() const {
  if (!state_)
    return {};
  std::lock_guard<std::mutex> lock(state_->error_mu);
  return state_->error;
}

void GraphRun::last_error_or_throw() const {
  const std::string err = last_error();
  if (!err.empty()) {
    throw std::runtime_error(err);
  }
}
} // namespace simaai::neat::graph
