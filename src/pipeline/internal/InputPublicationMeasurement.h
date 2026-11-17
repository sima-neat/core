#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

// Metadata only. Never owns, maps or changes the submitted buffer or its payload.
struct InputPublicationIdentity {
  std::string stream_id;
  std::int64_t original_sequence = -1;
  std::int64_t input_sequence = -1;
  std::int64_t frame_id = -1;
};

struct InputPublicationSnapshot {
  // Recorded before handoff so inline output is already correlated. Entries
  // still pending at stop are attempts, never latency evidence without a span.
  std::vector<InputPublicationIdentity> entries;
  bool failed = false;
};

// An optimized linear Run has no graph router entry events. Observe its actual
// appsrc admission instead, using the final transport metadata before handoff.
class InputPublicationMeasurement {
public:
  class Reservation {
  public:
    Reservation() = default;
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&& other) noexcept;
    ~Reservation() {
      if (owner_)
        owner_->cancel(epoch_, index_);
    }
    void finish(GstFlowReturn result) noexcept {
      if (owner_ && result != GST_FLOW_OK)
        owner_->cancel(epoch_, index_);
      owner_ = nullptr;
    }

  private:
    friend class InputPublicationMeasurement;
    Reservation(InputPublicationMeasurement* owner, std::uint64_t epoch, std::size_t index)
        : owner_(owner), epoch_(epoch), index_(index) {}
    InputPublicationMeasurement* owner_ = nullptr;
    std::uint64_t epoch_ = 0;
    std::size_t index_ = 0;
  };

  void start(std::uint64_t epoch);
  InputPublicationSnapshot stop(std::uint64_t epoch);
  Reservation observe(GstBuffer* buffer) noexcept {
    // No metadata reads, locks, clocks or allocations on the disabled path.
    const auto epoch = epoch_.load(std::memory_order_acquire);
    return epoch == 0 ? Reservation{} : observe_active(buffer, epoch);
  }
  void discard(std::uint64_t epoch) noexcept;

private:
  struct Entry {
    InputPublicationIdentity identity;
    bool cancelled = false;
  };
  Reservation observe_active(GstBuffer* buffer, std::uint64_t epoch) noexcept;
  void cancel(std::uint64_t epoch, std::size_t index) noexcept;
  std::atomic<std::uint64_t> epoch_{0};
  bool failed_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
};

} // namespace simaai::neat::pipeline_internal
