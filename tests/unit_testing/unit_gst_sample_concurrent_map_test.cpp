// Sample copies share their GstSample-backed storage, so graph fan-out branches map one payload
// at the same time (for example two VideoSenders copying the same encoded frame). A second
// read mapping must succeed while the first is held, and each must unmap independently. Two
// branches mapping a multi-memory buffer at once must not both merge its memories.
#include "gst/GstInit.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using simaai::neat::MapMode;
using simaai::neat::Mapping;
using simaai::neat::Tensor;

const std::vector<std::uint8_t> kPayload = {0xff, 0xd8, 0x01, 0x02, 0x03, 0xff, 0xd9};

GstBuffer* make_payload_buffer() {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, kPayload.size(), nullptr);
  require(buffer != nullptr, "failed to allocate payload buffer");
  gst_buffer_fill(buffer, 0, kPayload.data(), kPayload.size());
  return buffer;
}

void require_payload(const Mapping& mapping, const std::string& where) {
  require(mapping.data != nullptr, where + ": mapping failed");
  require(mapping.size_bytes == kPayload.size(), where + ": mapped size mismatch");
  require(std::memcmp(mapping.data, kPayload.data(), kPayload.size()) == 0,
          where + ": mapped bytes mismatch");
}

void require_concurrent_read_maps(const Tensor& tensor, const std::string& label) {
  const Tensor branch_copy = tensor; // what EdgeRouter fan-out hands each branch
  {
    Mapping first = tensor.map(MapMode::Read);
    require_payload(first, label + " first map");
    Mapping second = branch_copy.map(MapMode::Read);
    require_payload(second, label + " concurrent map");
  }
  Mapping again = branch_copy.map(MapMode::Read);
  require_payload(again, label + " map after both released");
}

// The payload split across two memories, as a depayloader or adapter can deliver it.
GstBuffer* make_two_memory_buffer() {
  GstBuffer* buffer = gst_buffer_new();
  const std::size_t half = kPayload.size() / 2;
  gst_buffer_append_memory(buffer,
                           gst_memory_new_wrapped(GST_MEMORY_FLAG_READONLY,
                                                  const_cast<std::uint8_t*>(kPayload.data()), half,
                                                  0, half, nullptr, nullptr));
  gst_buffer_append_memory(
      buffer, gst_memory_new_wrapped(
                  GST_MEMORY_FLAG_READONLY, const_cast<std::uint8_t*>(kPayload.data() + half),
                  kPayload.size() - half, 0, kPayload.size() - half, nullptr, nullptr));
  return buffer;
}

void require_racing_read_maps(const std::function<Tensor(GstBuffer*)>& make_tensor,
                              const std::string& label) {
  for (int round = 0; round < 2000; ++round) {
    GstBuffer* buffer = make_two_memory_buffer();
    const Tensor tensor = make_tensor(buffer); // the tensor's sample holds the only buffer ref
    gst_buffer_unref(buffer);
    const Tensor branch_copy = tensor;
    std::atomic<int> ready{0};
    bool ok[2] = {false, false};
    auto map_branch = [&](const Tensor& t, bool* result) {
      ready.fetch_add(1);
      while (ready.load() < 2) {
      }
      Mapping mapping = t.map(MapMode::Read);
      *result = mapping.data && mapping.size_bytes == kPayload.size() &&
                std::memcmp(mapping.data, kPayload.data(), kPayload.size()) == 0;
    };
    std::thread other(map_branch, std::cref(branch_copy), &ok[1]);
    map_branch(tensor, &ok[0]);
    other.join();
    require(ok[0] && ok[1], label + ": racing maps of a two-memory buffer");
  }
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    GstCaps* caps = gst_caps_from_string("image/jpeg,width=(int)16,height=(int)16");
    require(caps != nullptr, "failed to parse caps");

    GstBuffer* buffer = make_payload_buffer();
    GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
    gst_buffer_unref(buffer);
    const Tensor tensor = simaai::neat::from_gst_sample(sample);
    gst_sample_unref(sample);
    require_concurrent_read_maps(tensor, "GstSample tensor");

    GstBuffer* tap = make_payload_buffer();
    const simaai::neat::Sample encoded =
        simaai::neat::session_test::make_fused_encoded_output_sample_for_test(tap, caps, "stream0");
    gst_buffer_unref(tap);
    require(encoded.tensors.size() == 1U, "fused encoded Output should carry one tensor");
    require_concurrent_read_maps(encoded.tensors.front(), "fused encoded Output");

    g_log_set_always_fatal(static_cast<GLogLevelFlags>(G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_ERROR));
    require_racing_read_maps(
        [caps](GstBuffer* b) {
          GstSample* s = gst_sample_new(b, caps, nullptr, nullptr);
          Tensor t = simaai::neat::from_gst_sample(s);
          gst_sample_unref(s);
          return t;
        },
        "GstSample tensor");
    require_racing_read_maps(
        [caps](GstBuffer* b) {
          return simaai::neat::session_test::make_fused_encoded_output_sample_for_test(b, caps,
                                                                                       "stream0")
              .tensors.front();
        },
        "fused encoded Output");
    gst_caps_unref(caps);
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_gst_sample_concurrent_map_test\n";
  return 0;
}
