#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace simaai::neat {

// Returns pooled parent buffers to their pool only after every downstream
// reference is gone. A root is re-checked when a reference reaches the
// releaser (enqueue) or when a holder outside the releaser reports a release
// (wake). Releases that report nothing, such as a GStreamer release that
// finalizes parent metadata before the shared memory, are found by a timed
// re-check that starts at 1 ms and backs off to kMaxRecheckInterval.
class DeferredPoolBufferReleaser {
public:
  static constexpr std::chrono::milliseconds kMaxRecheckInterval{4};

  DeferredPoolBufferReleaser() : worker_([this] { run(); }) {}

  ~DeferredPoolBufferReleaser() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_one();
    worker_.join();

    // pending_ is a non-owning worklist. tracked_ owns the one retained
    // reference for every root that has not been returned to its pool yet.
    for (GstBuffer* buffer : tracked_) {
      gst_buffer_unref(buffer);
    }
  }

  DeferredPoolBufferReleaser(const DeferredPoolBufferReleaser&) = delete;
  DeferredPoolBufferReleaser& operator=(const DeferredPoolBufferReleaser&) = delete;

  void enqueue(std::vector<GstBuffer*> buffers) {
    if (buffers.empty()) {
      return;
    }
    std::size_t redundant_count = buffers.size();
    bool wake_worker = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!stopping_) {
        auto redundant_end = buffers.begin();
        for (GstBuffer* buffer : buffers) {
          if (tracked_.insert(buffer).second) {
            // tracked_ owns this reference. pending_ only schedules checks.
            pending_.push_back(buffer);
          } else {
            // Every enqueue transfers an owned reference. One retained root is
            // sufficient to keep the pooled buffer alive, so consume extras.
            *redundant_end++ = buffer;
          }
        }
        redundant_count = static_cast<std::size_t>(redundant_end - buffers.begin());
        wake_worker = true;
      }
    }

    // GstBuffer finalizers can run qdata callbacks that enqueue more roots.
    // Never unref while holding mu_, or that re-entry can deadlock.
    for (std::size_t i = 0; i < redundant_count; ++i) {
      gst_buffer_unref(buffers[i]);
    }
    if (wake_worker) {
      // Signal after the extra references are gone so the check can see it.
      wake();
    }
  }

  // Re-check pending roots now: a holder outside the releaser dropped a
  // reference, for example GstAppSink replacing the sample it retains.
  void wake() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      // The flag also reaches a check already in progress, whose roots are
      // out of pending_; the next pass clears it.
      woken_ = true;
      if (pending_.empty()) {
        return;
      }
    }
    cv_.notify_one();
  }

  // Number of passes over the pending roots so far.
  std::uint64_t checks() const {
    return checks_.load(std::memory_order_relaxed);
  }

private:
  void run() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stopping_) {
      cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
      auto recheck_interval = std::chrono::milliseconds(1);
      while (!stopping_ && !pending_.empty()) {
        std::vector<GstBuffer*> buffers;
        buffers.swap(pending_);
        woken_ = false;
        checks_.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();

        const auto releasable_end =
            std::partition(buffers.begin(), buffers.end(), [](GstBuffer* buffer) {
              // Some supported GStreamer releases finalize parent metadata
              // before releasing shared memory. Return pooled parents only
              // after every downstream memory reference is gone.
              return gst_buffer_is_writable(buffer) && gst_buffer_is_all_memory_writable(buffer);
            });

        lock.lock();
        pending_.insert(pending_.end(), releasable_end, buffers.end());

        // Stop tracking before the final unref can synchronously return a
        // buffer to its pool and make the pointer available for reuse.
        for (auto it = buffers.begin(); it != releasable_end; ++it) {
          tracked_.erase(*it);
        }

        lock.unlock();
        for (auto it = buffers.begin(); it != releasable_end; ++it) {
          gst_buffer_unref(*it);
        }
        lock.lock();

        if (!pending_.empty()) {
          if (cv_.wait_for(lock, recheck_interval, [this] { return stopping_ || woken_; })) {
            recheck_interval = std::chrono::milliseconds(1);
          } else {
            recheck_interval = std::min(recheck_interval * 2, kMaxRecheckInterval);
          }
        }
      }
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<GstBuffer*> pending_;
  std::unordered_set<GstBuffer*> tracked_;
  bool stopping_ = false;
  bool woken_ = false;
  std::atomic<std::uint64_t> checks_{0};
  std::thread worker_;
};

} // namespace simaai::neat
