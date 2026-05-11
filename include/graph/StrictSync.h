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

/**
 * @brief Per-stream store of video frames awaiting their matching detection metadata.
 *
 * Used by strict-sync detection pipelines: video frames are enqueued by `frame_id`, then
 * `take()` pulls the frame back when matching metadata arrives so it can be paired and
 * forwarded together. Tracks per-stream queue depth, byte usage, and hit/miss counts.
 *
 * @see YoloTokenStore
 * @see ReleasePacer
 * @ingroup graph
 */
class PendingVideoStore {
public:
  /// Frame held in the store while awaiting a matching metadata token.
  struct PendingFrame {
    simaai::neat::Sample sample; ///< Held sample.
    int64_t cap_ms = -1;         ///< Per-frame timeout in ms; -1 disables.
    size_t bytes = 0;            ///< Sample size in bytes (for memory accounting).
  };

  /// Per-stream counters tracking pending depth, hits, and misses.
  struct StreamStats {
    int64_t enqueued = 0;          ///< Total frames enqueued.
    int64_t matched = 0;           ///< Frames successfully taken by a matching token.
    int64_t miss = 0;              ///< `take()` calls that found no matching frame.
    size_t pending_depth = 0;      ///< Current number of pending frames.
    size_t pending_bytes = 0;      ///< Current bytes held in the store.
    size_t max_pending_depth = 0;  ///< High-watermark of `pending_depth`.
    size_t max_pending_bytes = 0;  ///< High-watermark of `pending_bytes`.
  };

  /// Create a store sized for `streams` distinct stream slots.
  explicit PendingVideoStore(size_t streams);

  /// Enqueue a frame under `(idx, frame_id)`. Returns false if the slot is full or duplicate.
  bool enqueue(size_t idx, int64_t frame_id, simaai::neat::Sample&& sample, int64_t cap_ms,
               size_t bytes);

  /// Take the frame stored under `(idx, frame_id)` if present.
  std::optional<PendingFrame> take(size_t idx, int64_t frame_id);

  /// Snapshot of the per-stream stats.
  StreamStats stats(size_t idx) const;

private:
  /// Per-stream internal state: pending frames keyed by frame_id plus accounting.
  struct StreamState {
    mutable std::mutex mu;                              ///< Guards the per-stream state.
    std::unordered_map<int64_t, PendingFrame> pending;  ///< Pending frames keyed by frame_id.
    std::deque<int64_t> order;                          ///< Frame ids in arrival order.
    size_t bytes_total = 0;                             ///< Current bytes held in the store.
    StreamStats stats;                                  ///< Per-stream counters.
  };

  std::vector<StreamState> states_;
};

/**
 * @brief Per-stream queue of yolo (or other detection) tokens awaiting a matching frame.
 *
 * Counterpart of `PendingVideoStore`: detection tokens land here and are paired with the
 * corresponding video frame. Also exposes a global ordered queue (`take_ordered`) so a
 * downstream consumer can drain tokens in arrival order across all streams.
 *
 * @see PendingVideoStore
 * @ingroup graph
 */
class YoloTokenStore {
public:
  /// Per-stream token: a `frame_id` plus the time it was enqueued.
  struct Token {
    int64_t frame_id = -1;     ///< Frame id this token refers to.
    int64_t enqueued_ms = 0;   ///< Wall-clock time (ms) the token was enqueued.
  };

  /// Globally-ordered token: same as `Token` plus the originating stream index.
  struct OrderedToken {
    size_t stream_idx = 0;     ///< Index of the originating stream.
    int64_t frame_id = -1;     ///< Frame id this token refers to.
    int64_t enqueued_ms = 0;   ///< Wall-clock time (ms) the token was enqueued.
  };

  /// Per-stream counters tracking depth, hits, and misses.
  struct Stats {
    int64_t enqueued = 0;     ///< Total tokens enqueued.
    int64_t dequeued = 0;     ///< Total tokens taken successfully.
    int64_t miss = 0;         ///< `take()` calls with nothing to return.
    size_t depth = 0;         ///< Current queue depth.
    size_t max_depth = 0;     ///< High-watermark of `depth`.
  };

  /// Create a store sized for `streams` distinct stream slots.
  explicit YoloTokenStore(size_t streams);

  /// Enqueue a token for stream `idx` with the given `frame_id`.
  void enqueue(size_t idx, int64_t frame_id);
  /// Pop the oldest token across all streams in arrival order.
  std::optional<OrderedToken> take_ordered();
  /// Pop the oldest token from a specific stream.
  std::optional<Token> take(size_t idx);
  /// Snapshot of the per-stream stats.
  Stats stats(size_t idx) const;

private:
  /// Per-stream internal state: token queue plus counters.
  struct State {
    mutable std::mutex mu;     ///< Guards the per-stream state.
    std::deque<Token> q;       ///< FIFO of tokens awaiting a matching frame.
    Stats stats;               ///< Per-stream counters.
  };

  static int64_t now_ms_i64();

  mutable std::mutex order_mu_;
  std::deque<OrderedToken> order_q_;
  std::vector<State> states_;
};

/**
 * @brief Paces downstream releases of paired samples to a target frame rate.
 *
 * Drains paired (frame + metadata) samples and forwards them to per-stream `Run` instances
 * at a steady cadence derived from `target_fps`. Provides backpressure via a bounded queue
 * and per-stream stats; optional callbacks fire on send completion and on drop events.
 *
 * @see PendingVideoStore
 * @see YoloTokenStore
 * @ingroup graph
 */
class ReleasePacer {
public:
  /// Per-stream counters tracking sent/dropped/fail counts and queue depth.
  struct Stats {
    int64_t enqueued = 0;          ///< Total samples enqueued.
    int64_t sent = 0;              ///< Total samples successfully sent downstream.
    int64_t dropped = 0;           ///< Total samples dropped due to overflow.
    int64_t send_fail = 0;         ///< Total send attempts that failed.
    int64_t max_queue_depth = 0;   ///< High-watermark of queue depth.
  };

  /// Callback fired after each downstream send attempt; `ok` indicates success.
  using OnSendResult = std::function<void(size_t /*stream_idx*/, bool /*ok*/)>;
  /// Callback fired when one or more samples are dropped due to overflow.
  using OnDrop = std::function<void(size_t /*stream_idx*/, int64_t /*dropped_count*/)>;

  /// Construct the pacer over a vector of per-stream `Run`s with the target fps and queue cap.
  ReleasePacer(const std::vector<std::shared_ptr<simaai::neat::Run>>& runs, int target_fps,
               size_t max_queue, OnSendResult on_send_result = {}, OnDrop on_drop = {});
  /// Stops the pacer and joins worker threads.
  ~ReleasePacer();

  /// True iff a positive `target_fps` was supplied (pacing active).
  bool enabled() const {
    return interval_ms_ > 0;
  }
  /// Per-frame release interval in milliseconds.
  int64_t interval_ms() const {
    return interval_ms_;
  }
  /// Per-stream maximum queue depth.
  size_t max_queue() const {
    return max_queue_;
  }

  /// Enqueue a paired sample for stream `idx`. Drops per `max_queue` policy on overflow.
  bool enqueue(size_t idx, simaai::neat::Sample&& sample);
  /// Stop the pacer; idempotent.
  void stop();
  /// Snapshot of the per-stream stats.
  Stats stats(size_t idx) const;

private:
  /// Per-stream internal state: queue, worker thread, and live counters.
  struct State {
    mutable std::mutex mu;                  ///< Guards the per-stream state.
    std::condition_variable cv;             ///< Condition variable used to signal queue state changes.
    std::deque<simaai::neat::Sample> queue; ///< Pending samples awaiting paced release.
    std::thread worker;                     ///< Per-stream pacing worker thread.
    bool stop = false;                      ///< Worker stop flag.

    int64_t next_release_ms = -1;           ///< Wall-clock deadline (ms) for the next release.
    int64_t enqueued = 0;                   ///< Total samples enqueued.
    int64_t sent = 0;                       ///< Total samples successfully sent downstream.
    int64_t dropped = 0;                    ///< Total samples dropped due to overflow.
    int64_t send_fail = 0;                  ///< Total send attempts that failed.
    int64_t max_queue_depth = 0;            ///< High-watermark of queue depth.
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
