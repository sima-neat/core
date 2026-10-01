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
    auto* pipeline = gst_pipeline_new(nullptr);
    auto* sink = gst_element_factory_make("appsink", nullptr);
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
    gst_object_unref(pipeline);
    verify(second, 5, 1, 38);
    verify(third, 6, 2);
    gst_sample_unref(second);
    gst_sample_unref(third);
    const auto mapping = retained.tensors[0].map_read();
    copy_check(mapping.data && mapping.size_bytes >= 38 &&
                   static_cast<const unsigned char*>(mapping.data)[37] == 5,
               "public Tensor did not retain the independent copy after source teardown");
    std::cout << "PASS V4L2 copy source: real tensor metadata, bounded pool, retention and stop\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
