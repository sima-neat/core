#pragma once

#include "HostPcieTensorPayload.h"
#include "PcieModelFactsReader.h"
#include "simaai/neat/pcie/Model.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

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
  void stop();
  bool is_running() const;

  bool push(const TensorList& tensors);
  std::optional<TensorList> pull(int timeout_ms);

  static std::string tensor_set_caps();
  static std::string caps_for_tensors(const TensorList& tensors);

private:
  static GstFlowReturn on_new_sample_static(GstElement* sink, gpointer user_data);
  GstFlowReturn on_new_sample(GstElement* sink);

  void start_with_caps(const std::string& caps);
  bool push_prepared_payload(PreparedPayload&& payload);

  GstElement* pipeline_ = nullptr;
  GstElement* appsrc_ = nullptr;
  GstElement* queue_element_ = nullptr;
  GstElement* pciehost_ = nullptr;
  GstElement* appsink_ = nullptr;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};

  mutable std::mutex receive_mutex_;
  std::condition_variable receive_cv_;
  std::deque<TensorList> received_results_;
  std::optional<std::string> receive_error_;

  PcieModelFacts facts_;
  int pcie_queue_ = 0;
  int card_id_ = 0;
  int max_inflight_ = 0;
  bool expects_bbox_output_ = false;
  std::string caps_;
};

} // namespace simaai::neat::pcie::internal
