#pragma once

#include "HostPcieTensorPayload.h"
#include "HostPcieOutputLayout.h"
#include "PcieModelFactsReader.h"
#include "RuntimeModelAccess.h"
#include "simaai/neat/pcie/Model.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <gst/gst.h>
namespace simaai::scheduler::pcie {
class Channel;
}

namespace simaai::neat::pcie::internal {

class HostPcieChannel {
public:
  HostPcieChannel();
  ~HostPcieChannel();

  HostPcieChannel(const HostPcieChannel&) = delete;
  HostPcieChannel& operator=(const HostPcieChannel&) = delete;

  void configure(const PcieModelFacts& facts, int queue, int card_id, int max_inflight,
                 bool expects_bbox_output, std::size_t output_buffer_bytes = 0);
  void request_stop();
  void stop();
  bool is_running() const;

  bool push(const TensorList& tensors);
  bool try_push(std::int32_t request_id, const TensorList& tensors);
  std::optional<TensorList> pull(int timeout_ms);
  std::optional<RuntimeInferenceResult> pull_result(int timeout_ms);

  static std::string tensor_set_caps();
  static std::string caps_for_tensors(const TensorList& tensors);
  static std::size_t required_transport_buffer_size(std::size_t packed_input_bytes,
                                                    std::size_t packed_output_bytes,
                                                    std::size_t submitted_payload_bytes,
                                                    std::size_t output_buffer_bytes = 0);
  static void validate_output_payload_size(std::size_t received_bytes, std::size_t expected_bytes);
  static void attach_request_id(GstBuffer* buffer, std::int32_t request_id);
  static std::optional<std::int32_t> request_id_from_buffer(GstBuffer* buffer);

private:
  bool push_payload(std::int32_t, const TensorList&, bool wait);
  std::unique_ptr<simaai::scheduler::pcie::Channel> transport_;
  std::atomic<std::int32_t> next_request_id_{0};
  std::mutex receive_mutex_;
  PcieModelFacts facts_;
  std::optional<HostPcieOutputLayout> output_layout_;
  bool expects_bbox_output_ = false;
};
} // namespace simaai::neat::pcie::internal
