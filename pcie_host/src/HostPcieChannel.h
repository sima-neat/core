#pragma once

#include "HostPcieTensorPayload.h"
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

namespace simaai::neat::pcie::internal {

class HostPcieChannel {
public:
  HostPcieChannel();
  ~HostPcieChannel();

  HostPcieChannel(const HostPcieChannel&) = delete;
  HostPcieChannel& operator=(const HostPcieChannel&) = delete;

  void configure(const PcieModelFacts& facts, int queue, int card_id, int max_inflight,
                 bool expects_bbox_output);
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
                                                    std::size_t submitted_payload_bytes);
  static void validate_output_payload_size(std::size_t received_bytes, std::size_t expected_bytes);
  static void attach_request_id(GstBuffer* buffer, std::int32_t request_id);
  static std::optional<std::int32_t> request_id_from_buffer(GstBuffer* buffer);

private:
  static GstFlowReturn on_new_sample_static(GstElement* sink, gpointer user_data);
  GstFlowReturn on_new_sample(GstElement* sink);
  void monitor_bus(GstBus* bus);

  void start_with_caps(const std::string& caps, std::size_t submitted_payload_bytes);
  void stop_locked();
  bool reserve_inflight();
  bool wait_and_reserve_inflight();
  void release_inflight();
  bool push_prepared_payload(std::int32_t request_id, PreparedPayload&& payload);

  GstElement* pipeline_ = nullptr;
  GstElement* appsrc_ = nullptr;
  GstElement* queue_element_ = nullptr;
  GstElement* pciehost_ = nullptr;
  GstElement* appsink_ = nullptr;
  GstPad* pciehost_sink_pad_ = nullptr;
  GstPad* pciehost_src_pad_ = nullptr;
  std::thread bus_thread_;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};

  mutable std::mutex send_mutex_;
  std::mutex inflight_mutex_;
  std::condition_variable inflight_cv_;
  int inflight_ = 0;
  mutable std::mutex receive_mutex_;
  std::condition_variable receive_cv_;
  std::deque<RuntimeInferenceResult> received_results_;
  std::optional<std::string> receive_error_;
  std::atomic<std::int32_t> next_request_id_{0};

  PcieModelFacts facts_;
  int pcie_queue_ = 0;
  int card_id_ = 0;
  int max_inflight_ = 0;
  bool expects_bbox_output_ = false;
  bool configured_ = false;
  std::size_t transport_buffer_size_ = 0;
  std::string caps_;
};

} // namespace simaai::neat::pcie::internal
