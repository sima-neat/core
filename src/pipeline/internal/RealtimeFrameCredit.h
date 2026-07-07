#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/internal/HolderLoanGate.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {

struct RealtimeFrameCredit {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
  bool graph_private = false;
};

using RealtimeFrameCreditWakeFn = std::function<void()>;

struct RealtimeFrameCreditLane {
  explicit RealtimeFrameCreditLane(int credit_limit,
                                   RealtimeFrameCreditWakeFn wake_fn = RealtimeFrameCreditWakeFn{})
      : gate(std::make_shared<HolderLoanGate>(credit_limit)), wake(std::move(wake_fn)) {}

  HolderLoanGatePtr gate;
  RealtimeFrameCreditWakeFn wake;
  std::atomic<std::uint64_t> registered{0};
  std::atomic<std::uint64_t> released_by_output{0};
  std::atomic<std::uint64_t> released_without_output{0};
  std::atomic<std::uint64_t> missing_key{0};
};

using RealtimeFrameCreditLanePtr = std::shared_ptr<RealtimeFrameCreditLane>;

RealtimeFrameCreditLanePtr
make_realtime_frame_credit_lane(int credit_limit,
                                RealtimeFrameCreditWakeFn wake_fn = RealtimeFrameCreditWakeFn{});
std::uint64_t next_realtime_frame_credit_namespace();
bool register_realtime_frame_credit(
    std::uint64_t namespace_id, const std::string& stream_id, std::int64_t frame_id,
    const RealtimeFrameCreditLanePtr& lane,
    const std::vector<RealtimeFrameCreditLanePtr>& companion_lanes = {});
bool alias_registered_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                             const Sample& sample, const char* mode);
bool release_registered_realtime_frame_credit(const RealtimeFrameCredit& credit, const char* mode,
                                              bool by_output);
void release_all_registered_realtime_frame_credits(std::uint64_t namespace_id, const char* mode);

bool sample_has_attached_realtime_frame_credit(const Sample& sample);
void attach_realtime_frame_credit_to_sample(const Sample& sample,
                                            const RealtimeFrameCredit& credit);
std::vector<RealtimeFrameCredit> attached_realtime_frame_credits_from_sample(const Sample& sample);

std::vector<RealtimeFrameCredit> realtime_frame_credits_for_sample(const Sample& sample);
void release_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                    const char* mode);
void release_realtime_frame_credits_without_output(const std::vector<RealtimeFrameCredit>& credits,
                                                   const char* mode);
void release_realtime_frame_credits_for_sample(const Sample& sample, const char* mode);

} // namespace simaai::neat::pipeline_internal
