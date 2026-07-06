#pragma once

#include "PcieModelFactsReader.h"
#include "simaai/neat/pcie/SimaPCIeHost.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <gst/gst.h>

namespace simaai::neat::pcie::internal {

class HostPcieChannel {
public:
  HostPcieChannel();
  ~HostPcieChannel();

  HostPcieChannel(const HostPcieChannel&) = delete;
  HostPcieChannel& operator=(const HostPcieChannel&) = delete;

  void configure(const PcieModelFacts& facts, int queue, int card_id, int max_inflight);
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
  bool push_bytes(const std::vector<std::uint8_t>& payload, const TensorList& tensors);
  std::uint64_t next_sequence();

  GstElement* pipeline_ = nullptr;
  GstElement* appsrc_ = nullptr;
  GstElement* queue_element_ = nullptr;
  GstElement* pciehost_ = nullptr;
  GstElement* appsink_ = nullptr;

  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<std::uint64_t> sequence_{0};
  std::mutex pending_mutex_;
  std::deque<std::uint64_t> pending_sequences_;

  mutable std::mutex receive_mutex_;
  std::condition_variable receive_cv_;
  std::deque<TensorList> received_results_;

  PcieModelFacts facts_;
  int pcie_queue_ = 0;
  int card_id_ = 0;
  int max_inflight_ = 0;
  std::string caps_;
};

} // namespace simaai::neat::pcie::internal
