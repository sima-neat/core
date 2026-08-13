#include "gst/NeatCameraMemoryBridge.h"
#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void prove_dmabuf_pool_proposal() {
  require(simaai::neat::register_neat_camera_memory_bridge(),
          "failed to register neatcamerabridge");
  GstElement* bridge = gst_element_factory_make("neatcamerabridge", "bridge");
  require(bridge != nullptr, "failed to create neatcamerabridge");

  GstPad* sink = gst_element_get_static_pad(bridge, "sink");
  GstCaps* caps =
      gst_caps_from_string("video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1");
  GstQuery* query = gst_query_new_allocation(caps, TRUE);
  GstStructure* alignment = gst_structure_new(
      "video-meta", "padding-top", G_TYPE_UINT, 0U, "padding-bottom", G_TYPE_UINT, 0U,
      "padding-left", G_TYPE_UINT, 0U, "padding-right", G_TYPE_UINT, 64U, nullptr);
  gst_query_add_allocation_meta(query, GST_VIDEO_META_API_TYPE, alignment);
  gst_structure_free(alignment);
  require(gst_pad_query(sink, query), "neatcamerabridge rejected ALLOCATION query");
  require(gst_query_find_allocation_meta(query, GST_VIDEO_META_API_TYPE, nullptr),
          "neatcamerabridge did not request GstVideoMeta");

  bool found = false;
  const guint pools = gst_query_get_n_allocation_pools(query);
  for (guint i = 0; i < pools; ++i) {
    GstBufferPool* pool = nullptr;
    guint size = 0;
    guint min_buffers = 0;
    guint max_buffers = 0;
    gst_query_parse_nth_allocation_pool(query, i, &pool, &size, &min_buffers, &max_buffers);
    if (!pool)
      continue;

    GstStructure* config = gst_buffer_pool_get_config(pool);
    const bool compatible =
        size >= 1984U * 1080U * 3U / 2U && min_buffers == 4 && max_buffers == 4 &&
        gst_buffer_pool_config_has_option(config, GST_BUFFER_POOL_OPTION_VIDEO_META);
    gst_structure_free(config);
    if (!compatible) {
      gst_object_unref(pool);
      continue;
    }

    require(gst_buffer_pool_set_active(pool, TRUE), "failed to activate proposed DMA-BUF pool");
    GstBuffer* buffer = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool, &buffer, nullptr) == GST_FLOW_OK && buffer,
            "failed to acquire a proposed DMA-BUF buffer");
    require(gst_buffer_n_memory(buffer) == 2,
            "NV12 DMA-BUF pool did not return one GstMemory per plane");
    for (guint plane = 0; plane < gst_buffer_n_memory(buffer); ++plane) {
      GstMemory* memory = gst_buffer_peek_memory(buffer, plane);
      require(gst_is_dmabuf_memory(memory),
              "proposed capture buffer contains non-DMA-BUF GstMemory");
      require(gst_dmabuf_memory_get_fd(memory) >= 0,
              "proposed capture DMA-BUF has an invalid file descriptor");
    }
    GstVideoMeta* video_meta = gst_buffer_get_video_meta(buffer);
    require(video_meta && video_meta->format == GST_VIDEO_FORMAT_NV12 &&
                video_meta->width == 1920 && video_meta->height == 1080 &&
                video_meta->n_planes == 2 && video_meta->stride[0] == 1984 &&
                video_meta->stride[1] == 1984,
            "proposed capture buffer has incompatible GstVideoMeta");
    GstParentBufferMeta* parent_meta = gst_buffer_get_parent_buffer_meta(buffer);
    GstMemory* parent_memory =
        parent_meta && parent_meta->buffer && gst_buffer_n_memory(parent_meta->buffer) == 1
            ? gst_buffer_peek_memory(parent_meta->buffer, 0)
            : nullptr;
    require(parent_memory && parent_memory->allocator && parent_memory->allocator->mem_type &&
                (std::strcmp(parent_memory->allocator->mem_type, "NeatSimaaiSegmentMemory") == 0 ||
                 std::strcmp(parent_memory->allocator->mem_type, "SimaaiSegmentMemory") == 0) &&
                GST_MEMORY_FLAG_IS_SET(parent_memory, GST_SIMAAI_MEMORY_TARGET_EV74),
            "Core DMA-BUF wrapper did not retain its EV74 SiMa backing buffer");

    gst_buffer_unref(buffer);
    require(gst_buffer_pool_set_active(pool, FALSE), "failed to deactivate proposed DMA-BUF pool");
    gst_object_unref(pool);
    found = true;
    break;
  }
  require(found, "neatcamerabridge did not propose a usable standard DMA-BUF pool");

  guint64 proposals = 0;
  g_object_get(bridge, "allocation-proposal-count", &proposals, nullptr);
  require(proposals == 1, "neatcamerabridge proposal counter is incorrect");

  gst_query_unref(query);
  gst_caps_unref(caps);
  gst_object_unref(sink);
  gst_object_unref(bridge);
}

void print_bus_error(GstMessage* message) {
  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  std::cerr << "GStreamer error: " << (error ? error->message : "unknown") << " ("
            << (debug ? debug : "no details") << ")\n";
  g_clear_error(&error);
  g_free(debug);
}

void prove_live_camera_negotiation(bool prove_backpressure) {
  const char* configured_camera = std::getenv("SIMA_TEST_CAMERA_NAME");
  const char* camera =
      configured_camera && *configured_camera ? configured_camera : "og05c10 5-0036";
  const std::string description =
      "libcamerasrc name=camera-src camera-name=\"" + std::string(camera) +
      "\" "
      "buffer-count=32 simaai-zero-copy-required=true ! "
      "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
      "neatcamerabridge name=bridge copy-allowed=false num-buffers=32 ! "
      "appsink name=sink sync=false emit-signals=false max-buffers=32 drop=false";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
  if (!pipeline) {
    const std::string message = error ? error->message : "unknown parse error";
    g_clear_error(&error);
    throw std::runtime_error("failed to build live camera pipeline: " + message);
  }

  GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "camera-src");
  GstElement* bridge = gst_bin_get_by_name(GST_BIN(pipeline), "bridge");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  require(source && bridge && sink, "live camera pipeline elements are missing");
  require(g_object_class_find_property(G_OBJECT_GET_CLASS(source), "simaai-zero-copy-negotiated") !=
              nullptr,
          "libcamerasrc lacks negotiated zero-copy observability");

  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "live camera pipeline failed to enter PLAYING");
  GstBus* bus = gst_element_get_bus(pipeline);
  unsigned int samples = 0;
  std::vector<GstSample*> held_samples;
  if (prove_backpressure) {
    for (unsigned int i = 0; i < 32; ++i) {
      GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), GST_SECOND);
      require(sample != nullptr, "live pipeline stalled before all 32 buffers were held");
      GstBuffer* buffer = gst_sample_get_buffer(sample);
      GstVideoMeta* video_meta = buffer ? gst_buffer_get_video_meta(buffer) : nullptr;
      require(video_meta && video_meta->format == GST_VIDEO_FORMAT_NV12 &&
                  video_meta->width == 1920 && video_meta->height == 1080 &&
                  video_meta->n_planes == 2,
              "live zero-copy output lost GstVideoMeta");
      held_samples.push_back(sample);
      ++samples;
    }

    GstSample* unexpected = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 750 * GST_MSECOND);
    if (unexpected)
      gst_sample_unref(unexpected);
    require(unexpected == nullptr,
            "a 33rd frame arrived while all 32 application buffers were held");
    std::cout << "BACKPRESSURE_METRICS held=32 unexpected_frames=0\n";

    for (GstSample* sample : held_samples)
      gst_sample_unref(sample);
    held_samples.clear();
  }
  for (unsigned int attempt = 0; attempt < 300 && samples < 96; ++attempt) {
    if (GstMessage* message =
            gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR))) {
      print_bus_error(message);
      gst_message_unref(message);
      break;
    }
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 100 * GST_MSECOND);
    if (sample) {
      ++samples;
      gst_sample_unref(sample);
    }
  }

  gboolean active = FALSE;
  gboolean negotiated = FALSE;
  guint64 frames = 0;
  guint64 proposals = 0;
  guint64 passthrough = 0;
  guint64 copies = 0;
  guint64 rejected = 0;
  g_object_get(source, "simaai-zero-copy-active", &active, "simaai-zero-copy-negotiated",
               &negotiated, "simaai-zero-copy-frames", &frames, nullptr);
  g_object_get(bridge, "allocation-proposal-count", &proposals, "passthrough-count", &passthrough,
               "copy-count", &copies, "rejected-count", &rejected, nullptr);

  std::cout << "LIVE_METRICS camera=\"" << camera << "\" samples=" << samples
            << " active=" << static_cast<bool>(active)
            << " negotiated=" << static_cast<bool>(negotiated) << " zero_copy_frames=" << frames
            << " proposals=" << proposals << " passthrough=" << passthrough << " copies=" << copies
            << " rejected=" << rejected << '\n';

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(sink);
  gst_object_unref(bridge);
  gst_object_unref(source);
  gst_object_unref(pipeline);

  require(samples == 96, "live pipeline did not deliver 96 camera frames");
  require(active && negotiated,
          "libcamerasrc did not activate the downstream-negotiated DMA-BUF pool");
  require(frames >= samples, "libcamerasrc zero-copy frame counter did not advance");
  require(proposals >= 1, "bridge did not answer the live ALLOCATION query");
  require(passthrough >= samples, "bridge did not pass every observed camera frame through");
  require(copies == 0 && rejected == 0, "strict live pipeline copied or rejected a camera frame");
}

} // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  if (!gst_meta_get_info("GstSimaMeta")) {
    static const gchar* tags[] = {GST_META_TAG_MEMORY_STR, nullptr};
    gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
  }
  try {
    prove_dmabuf_pool_proposal();
    if (std::getenv("SIMA_TEST_CAMERA_ALLOCATION_QUERY")) {
      for (unsigned int cycle = 0; cycle < 3; ++cycle)
        prove_live_camera_negotiation(cycle == 0);
    }
    std::cout << "PASS: standard DMA-BUF pool proposal";
    if (std::getenv("SIMA_TEST_CAMERA_ALLOCATION_QUERY"))
      std::cout << " and three live 32-buffer zero-copy restart cycles";
    std::cout << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
