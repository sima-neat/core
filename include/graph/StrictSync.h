/**
 * @file
 * @ingroup graph
 * @brief Strict multi-stream synchronization helpers (pending frame store, token store, release
 * pacer).
 */
#pragma once

#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph::strict_sync {

class PendingVideoStore {
public:
  struct PendingFrame {
    simaai::neat::Sample sample;
    int64_t cap_ms = -1;
    size_t bytes = 0;
  };

  struct StreamStats {
    int64_t enqueued = 0;
    int64_t matched = 0;
    int64_t miss = 0;
    size_t pending_depth = 0;
    size_t pending_bytes = 0;
    size_t max_pending_depth = 0;
    size_t max_pending_bytes = 0;
  };

  explicit PendingVideoStore(size_t streams);

  bool enqueue(size_t idx, int64_t frame_id, simaai::neat::Sample&& sample, int64_t cap_ms,
               size_t bytes);

  std::optional<PendingFrame> take(size_t idx, int64_t frame_id);

  StreamStats stats(size_t idx) const;

private:
  struct StreamState {
    mutable std::mutex mu;
    std::unordered_map<int64_t, PendingFrame> pending;
    std::deque<int64_t> order;
    size_t bytes_total = 0;
    StreamStats stats;
  };

  std::vector<StreamState> states_;
};

class YoloTokenStore {
public:
  struct Token {
    int64_t frame_id = -1;
    int64_t enqueued_ms = 0;
  };

  struct OrderedToken {
    size_t stream_idx = 0;
    int64_t frame_id = -1;
    int64_t enqueued_ms = 0;
  };

  struct Stats {
    int64_t enqueued = 0;
    int64_t dequeued = 0;
    int64_t miss = 0;
    size_t depth = 0;
    size_t max_depth = 0;
  };

  explicit YoloTokenStore(size_t streams);

  void enqueue(size_t idx, int64_t frame_id);
  std::optional<OrderedToken> take_ordered();
  std::optional<Token> take(size_t idx);
  Stats stats(size_t idx) const;

private:
  struct State {
    mutable std::mutex mu;
    std::deque<Token> q;
    Stats stats;
  };

  static int64_t now_ms_i64();

  mutable std::mutex order_mu_;
  std::deque<OrderedToken> order_q_;
  std::vector<State> states_;
};

class ReleasePacer {
public:
  struct Stats {
    int64_t enqueued = 0;
    int64_t sent = 0;
    int64_t dropped = 0;
    int64_t send_fail = 0;
    int64_t max_queue_depth = 0;
  };

  using OnSendResult = std::function<void(size_t /*stream_idx*/, bool /*ok*/)>;
  using OnDrop = std::function<void(size_t /*stream_idx*/, int64_t /*dropped_count*/)>;

  ReleasePacer(const std::vector<std::shared_ptr<simaai::neat::Run>>& runs, int target_fps,
               size_t max_queue, OnSendResult on_send_result = {}, OnDrop on_drop = {});
  ~ReleasePacer();

  bool enabled() const {
    return interval_ms_ > 0;
  }
  int64_t interval_ms() const {
    return interval_ms_;
  }
  size_t max_queue() const {
    return max_queue_;
  }

  bool enqueue(size_t idx, simaai::neat::Sample&& sample);
  void stop();
  Stats stats(size_t idx) const;

private:
  struct State {
    mutable std::mutex mu;
    std::condition_variable cv;
    std::deque<simaai::neat::Sample> queue;
    std::thread worker;
    bool stop = false;

    int64_t next_release_ms = -1;
    int64_t enqueued = 0;
    int64_t sent = 0;
    int64_t dropped = 0;
    int64_t send_fail = 0;
    int64_t max_queue_depth = 0;
  };

  static int64_t now_ms_i64();
  void worker_loop(size_t idx);

  std::vector<std::shared_ptr<simaai::neat::Run>> runs_;
  std::vector<std::unique_ptr<State>> states_;
  OnSendResult on_send_result_;
  OnDrop on_drop_;

  int64_t interval_ms_ = 0;
  size_t max_queue_ = 0;
  std::atomic<bool> stopped_{false};
};

} // namespace simaai::neat::graph::strict_sync
