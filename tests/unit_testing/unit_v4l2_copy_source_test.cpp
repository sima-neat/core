#include "gst/NeatV4L2CopySource.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "gstsimaaitensorbuffer.h"
#include "v4l2_copy_mock.h"

#include <gst/app/gstappsink.h>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
// This test links libgstapp already. Instantiate its type directly so CI's
// isolated plugin registry need not discover app/coreelements plugins.
GstElement* make_app_sink() {
  return GST_ELEMENT(g_object_new(GST_TYPE_APP_SINK, nullptr));
}

void verify(GstSample* sample, unsigned char expected, unsigned sequence, unsigned bytes = 48) {
  copy_check(sample != nullptr, "missing copied sample");
  auto decoded = simaai::neat::sample_from_gst_envelope(sample, "v4l2-copy-test", false, nullptr);
  copy_check(!decoded.empty() && decoded.caps_string.find("row-stride") != std::string::npos,
             "public Sample lost the wire contract");
  auto* buffer = gst_sample_get_buffer(sample);
  copy_check(gst_buffer_get_size(buffer) == bytes, "payload span lost");
  copy_check(GST_BUFFER_OFFSET(buffer) == sequence, "sequence lost");
  GstMapInfo map = GST_MAP_INFO_INIT;
  copy_check(gst_buffer_map(buffer, &map, GST_MAP_READ), "map failed");
  const bool matches = map.size == bytes && std::all_of(map.data, map.data + map.size,
                                                        [=](auto b) { return b == expected; });
  gst_buffer_unmap(buffer, &map);
  copy_check(matches, "retained tensor changed after requeue/stop");
  auto* caps = gst_caps_get_structure(gst_sample_get_caps(sample), 0);
  guint stride = 0, size = 0;
  copy_check(gst_structure_get_uint(caps, "row-stride", &stride) && stride == 10 &&
                 gst_structure_get_uint(caps, "sizeimage", &size) && size == 48,
             "negotiated wire contract missing");
  char* error = nullptr;
  auto* handle = simaai::gst::sima_tensor_buffer_create_view_handle_from_sample(sample, &error);
  const std::string message = error ? error : "no tensor descriptor";
  g_free(error);
  copy_check(handle != nullptr, message.c_str());
  SimaTensorDescriptorV2 descriptor{};
  const bool valid =
      simaai::gst::sima_tensor_buffer_handle_tensor_descriptor(handle, 0, &descriptor) &&
      descriptor.rank == 1 && descriptor.shape[0] == bytes && descriptor.stride_bytes[0] == 1 &&
      descriptor.size_bytes == bytes && descriptor.dtype == SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  simaai::gst::sima_tensor_buffer_handle_unref(handle);
  copy_check(valid, "owned flat tensor descriptor invalid");
}
} // namespace
int main() {
  try {
    // All exercised types are linked or provided by the fixture. Keep this
    // hardware-isolated test independent of installed plugin discovery.
    g_setenv("GST_PLUGIN_SYSTEM_PATH_1_0", "", TRUE);
    g_setenv("GST_PLUGIN_PATH_1_0", "", TRUE);
    gst_init(nullptr, nullptr);
    // Graph startup registers these ABIs. This isolated source test deliberately
    // avoids the broader Neat plugin discovery (and all real camera devices).
    const gchar* tags[] = {nullptr};
    for (const auto* name : {"GstSimaMeta", "GstSimaSampleMeta", SIMA_TENSOR_SET_META_NAME}) {
      if (!gst_meta_get_info(name))
        gst_meta_register_custom(name, tags, nullptr, nullptr, nullptr);
    }
    auto mock = std::make_shared<CopyCameraMock>();
    mock->short_second_frame = true;
    auto* source = simaai::neat::make_v4l2_copy_source_for_test(mock);
    g_object_set(source, "device", "/dev/mock", "width", 8U, "height", 4U, "fourcc", "BA81",
                 "capture-buffer-count", 4U, "output-buffer-count", 2U, nullptr);
    mock->before_streamon = [source] {
      auto* pad = gst_element_get_static_pad(source, "src");
      auto* caps = gst_pad_get_current_caps(pad);
      const bool ready =
          caps && gst_structure_has_field(gst_caps_get_structure(caps, 0), "sizeimage");
      if (caps)
        gst_caps_unref(caps);
      gst_object_unref(pad);
      copy_check(ready, "STREAMON preceded output caps negotiation");
    };
    auto* pipeline = gst_pipeline_new(nullptr);
    auto* sink = make_app_sink();
    copy_check(sink != nullptr, "appsink unavailable");
    g_object_set(sink, "sync", FALSE, "max-buffers", 1U, "enable-last-sample", FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), source, sink, nullptr);
    copy_check(gst_element_link(source, sink), "source/appsink link failed");
    copy_check(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
               "start failed");
    GstSample* first = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    GstSample* second = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    verify(first, 1, 0);
    verify(second, 5, 1, 38);
    auto retained =
        simaai::neat::sample_from_gst_envelope(second, "retained-copy-test", false, nullptr);
    copy_check(retained.tensors.size() == 1 &&
                   retained.tensors[0].shape == std::vector<std::int64_t>{38},
               "public flat tensor shape changed");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    copy_check(mock->dequeues.load() == 2, "pool did not bound retained outputs");
    gst_sample_unref(first);
    GstSample* third = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    verify(third, 6, 2);
    verify(second, 5, 1, 38);
    // Both pooled buffers remain held: NULL must interrupt a blocked acquire.
    const auto before = std::chrono::steady_clock::now();
    copy_check(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
               "stop failed");
    copy_check(std::chrono::steady_clock::now() - before < std::chrono::seconds(2),
               "pool cancellation hung");
    copy_check(mock->close_count == 1 && mock->unmaps == 4, "capture cleanup failed");
    // Restart the same element while tensors from the old pool remain held.
    copy_check(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
               "restart failed");
    auto* restarted = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 2 * GST_SECOND);
    copy_check(restarted != nullptr, "restart did not produce output");
    gst_sample_unref(restarted);
    copy_check(gst_element_set_state(pipeline, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
               "second stop failed");
    copy_check(mock->close_count == 2 && mock->releases == 2, "restart leaked capture queue");
    gst_object_unref(pipeline);
    verify(second, 5, 1, 38);
    verify(third, 6, 2);
    gst_sample_unref(second);
    gst_sample_unref(third);
    const auto mapping = retained.tensors[0].map_read();
    copy_check(mapping.data && mapping.size_bytes >= 38 &&
                   static_cast<const unsigned char*>(mapping.data)[37] == 5,
               "public Tensor did not retain the independent copy after source teardown");
    // Exercise the GstBaseSrc unlock/stop protocol while create() is inside
    // poll(), rather than only while waiting for a free output-pool buffer.
    for (bool failed_stop : {false, true}) {
      auto waiting = std::make_shared<CopyCameraMock>();
      waiting->wait_for_cancel.store(true);
      waiting->stop_fail = failed_stop;
      auto* input = simaai::neat::make_v4l2_copy_source_for_test(waiting);
      g_object_set(input, "device", "/dev/mock", "width", 8U, "height", 4U, "fourcc", "BA81",
                   "capture-buffer-count", 4U, nullptr);
      auto* graph = gst_pipeline_new(nullptr);
      auto* output = make_app_sink();
      copy_check(output != nullptr, "appsink unavailable");
      g_object_set(output, "sync", FALSE, "enable-last-sample", FALSE, nullptr);
      gst_bin_add_many(GST_BIN(graph), input, output, nullptr);
      copy_check(gst_element_link(input, output), "waiting source link failed");
      copy_check(gst_element_set_state(graph, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                 "waiting source start failed");
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (waiting->waits.load() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      copy_check(waiting->waits.load() != 0, "capture never entered poll");
      std::atomic<bool> retirement_error{false};
      auto* bus = gst_element_get_bus(graph);
      gst_bus_set_sync_handler(
          bus,
          [](GstBus*, GstMessage* message, gpointer data) {
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR)
              static_cast<std::atomic<bool>*>(data)->store(true);
            return GST_BUS_PASS;
          },
          &retirement_error, nullptr);
      const auto stopping = std::chrono::steady_clock::now();
      // GstBaseSrc may report a stop failure on the bus rather than propagate
      // it through set_state(). Restart and ownership are the safety contract.
      gst_element_set_state(graph, GST_STATE_NULL);
      copy_check(std::chrono::steady_clock::now() - stopping < std::chrono::seconds(2),
                 "frame-wait cancellation hung");
      copy_check(waiting->dequeues.load() == 0, "cancelled wait dequeued a frame");
      if (failed_stop) {
        copy_check(retirement_error.load(), "retirement failure was not reported on the bus");
        copy_check(gst_element_set_state(graph, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE,
                   "parked source restarted");
        gst_element_set_state(graph, GST_STATE_NULL);
        copy_check(waiting->unmaps == 0 && waiting->close_count == 0 && waiting->releases == 0,
                   "failed stop released camera ownership");
      } else {
        copy_check(waiting->unmaps == 4 && waiting->close_count == 1 && waiting->releases == 1,
                   "waiting stop leaked capture resources");
      }
      gst_bus_set_sync_handler(bus, nullptr, nullptr, nullptr);
      gst_object_unref(bus);
      gst_object_unref(graph);
    }
    // Downstream rejection must fail preparation without ever queueing DMA.
    {
      auto rejected = std::make_shared<CopyCameraMock>();
      auto* input = simaai::neat::make_v4l2_copy_source_for_test(rejected);
      g_object_set(input, "device", "/dev/mock", "width", 8U, "height", 4U, "fourcc", "BA81",
                   "capture-buffer-count", 4U, nullptr);
      auto* graph = gst_pipeline_new(nullptr);
      auto* output = make_app_sink();
      copy_check(output != nullptr, "appsink unavailable");
      auto* incompatible = gst_caps_from_string("application/vnd.simaai.tensor,width=(int)9");
      g_object_set(output, "caps", incompatible, "sync", FALSE, nullptr);
      gst_caps_unref(incompatible);
      gst_bin_add_many(GST_BIN(graph), input, output, nullptr);
      copy_check(gst_element_link(input, output), "caps test link failed");
      copy_check(gst_element_set_state(graph, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE,
                 "incompatible output caps accepted");
      gst_element_set_state(graph, GST_STATE_NULL);
      copy_check(rejected->queues == 0 && rejected->unmaps == 4 && rejected->releases == 1 &&
                     rejected->close_count == 1,
                 "caps failure reached DMA or leaked preparation");
      gst_object_unref(graph);
    }
    std::cout << "PASS V4L2 copy source: real tensor metadata, bounded pool, retention and stop\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
