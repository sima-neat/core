#include "gst/NeatCameraMemoryBridge.h"
#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool is_neat_allocator(const GstAllocator* allocator) {
  return allocator && allocator->mem_type &&
         (std::strcmp(allocator->mem_type, "NeatSimaaiSegmentMemory") == 0 ||
          std::strcmp(allocator->mem_type, "SimaaiSegmentMemory") == 0);
}

void prove_allocator_proposal() {
  require(simaai::neat::register_neat_camera_memory_bridge(),
          "failed to register neatcamerabridge");
  GstElement* bridge = gst_element_factory_make("neatcamerabridge", "bridge");
  require(bridge != nullptr, "failed to create neatcamerabridge");

  GstPad* sink = gst_element_get_static_pad(bridge, "sink");
  GstCaps* caps =
      gst_caps_from_string("video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1");
  GstQuery* query = gst_query_new_allocation(caps, TRUE);
  require(gst_pad_query(sink, query), "neatcamerabridge rejected ALLOCATION query");
  require(gst_query_find_allocation_meta(query, GST_VIDEO_META_API_TYPE, nullptr),
          "neatcamerabridge did not request GstVideoMeta");

  bool found = false;
  const guint params = gst_query_get_n_allocation_params(query);
  for (guint i = 0; i < params; ++i) {
    GstAllocator* allocator = nullptr;
    GstAllocationParams allocation_params;
    gst_query_parse_nth_allocation_param(query, i, &allocator, &allocation_params);
    if (!is_neat_allocator(allocator))
      continue;
    found = true;
    require((allocation_params.flags & GST_SIMAAI_MEMORY_TARGET_EV74) != 0,
            "Neat allocator proposal did not request EV74 memory");
    require((allocation_params.flags & GST_SIMAAI_MEMORY_FLAG_CACHED) != 0,
            "Neat allocator proposal did not request cached memory");
  }
  require(found, "neatcamerabridge did not propose the Neat segment allocator");

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

void prove_live_camera_negotiation() {
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
          "libcamerasrc did not activate the downstream-negotiated allocator");
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
    prove_allocator_proposal();
    if (std::getenv("SIMA_TEST_CAMERA_ALLOCATION_QUERY"))
      prove_live_camera_negotiation();
    std::cout << "PASS: Neat allocator proposal";
    if (std::getenv("SIMA_TEST_CAMERA_ALLOCATION_QUERY"))
      std::cout << " and live 32-buffer zero-copy negotiation";
    std::cout << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
