#include "gst/NeatCameraMemoryBridge.h"
#include "pipeline/internal/SimaaiGstCompat.h"

#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool camera_memory_provider_available() {
  using ApiGet = const GstNeatCameraMemoryApiV1* (*)(guint32);
  auto* get_api = reinterpret_cast<ApiGet>(dlsym(RTLD_DEFAULT, "gst_neat_camera_memory_api_get"));
  return get_api && get_api(GST_NEAT_CAMERA_MEMORY_API_VERSION_1);
}

bool prove_dmabuf_pool_proposal() {
  const bool provider_available = camera_memory_provider_available();
  require(simaai::neat::register_neat_camera_memory_bridge(),
          "failed to register neatcamerabridge");
  GstElement* bridge = gst_element_factory_make("neatcamerabridge", "bridge");
  require(bridge != nullptr, "failed to create neatcamerabridge");

  GstPad* sink = gst_element_get_static_pad(bridge, "sink");
  GstCaps* caps =
      gst_caps_from_string("video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1");
  GstQuery* query = gst_query_new_allocation(caps, TRUE);
  GstStructure* alignment = gst_structure_new(
      "video-meta", "padding-top", G_TYPE_UINT, 2U, "padding-bottom", G_TYPE_UINT, 2U,
      "padding-left", G_TYPE_UINT, 16U, "padding-right", G_TYPE_UINT, 64U, "stride-align0",
      G_TYPE_UINT, 127U, "stride-align1", G_TYPE_UINT, 255U, nullptr);
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
        size >= 2048U * 1080U * 3U / 2U && min_buffers == 0 && max_buffers == 0 &&
        gst_buffer_pool_config_has_option(config, GST_BUFFER_POOL_OPTION_VIDEO_META);
    gst_structure_free(config);
    if (!compatible) {
      gst_object_unref(pool);
      continue;
    }

    require(gst_buffer_pool_set_active(pool, TRUE), "failed to activate proposed DMA-BUF pool");

    GstBuffer* recycled = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool, &recycled, nullptr) == GST_FLOW_OK && recycled,
            "failed to acquire DMA-BUF metadata reset probe");
    require(gst_buffer_add_custom_meta(recycled, "GstSimaMeta") != nullptr,
            "failed to attach transient metadata reset probe");
    gst_buffer_unref(recycled);
    recycled = nullptr;
    require(gst_buffer_pool_acquire_buffer(pool, &recycled, nullptr) == GST_FLOW_OK && recycled,
            "failed to reacquire DMA-BUF metadata reset probe");
    require(gst_buffer_get_custom_meta(recycled, "GstSimaMeta") == nullptr,
            "capture pool retained non-pooled metadata across buffer reuse");
    require(gst_buffer_get_video_meta(recycled) != nullptr &&
                gst_buffer_get_parent_buffer_meta(recycled) != nullptr,
            "capture pool reset removed pooled layout or backing metadata");
    gst_buffer_unref(recycled);

    std::vector<GstBuffer*> buffers;
    for (unsigned int index = 0; index < 8; ++index) {
      GstBuffer* buffer = nullptr;
      require(gst_buffer_pool_acquire_buffer(pool, &buffer, nullptr) == GST_FLOW_OK && buffer,
              "failed to grow the proposed DMA-BUF pool to the camera default");
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
                  video_meta->n_planes == 2 && (video_meta->stride[0] & 127) == 0 &&
                  (video_meta->stride[1] & 255) == 0 && video_meta->offset[0] > 0 &&
                  gst_buffer_get_size(buffer) == size,
              "proposed capture buffer has incompatible GstVideoMeta");
      GstParentBufferMeta* parent_meta = gst_buffer_get_parent_buffer_meta(buffer);
      GstMemory* parent_memory =
          parent_meta && parent_meta->buffer && gst_buffer_n_memory(parent_meta->buffer) == 1
              ? gst_buffer_peek_memory(parent_meta->buffer, 0)
              : nullptr;
      require(
          parent_memory && parent_memory->allocator && parent_memory->allocator->mem_type &&
              (std::strcmp(parent_memory->allocator->mem_type, "NeatSimaaiSegmentMemory") == 0 ||
               std::strcmp(parent_memory->allocator->mem_type, "SimaaiSegmentMemory") == 0) &&
              GST_MEMORY_FLAG_IS_SET(parent_memory, GST_SIMAAI_MEMORY_TARGET_EV74),
          "Core DMA-BUF wrapper did not retain its EV74 SiMa backing buffer");
      buffers.push_back(buffer);
    }

    for (GstBuffer* buffer : buffers)
      gst_buffer_unref(buffer);
    require(gst_buffer_pool_set_active(pool, FALSE), "failed to deactivate proposed DMA-BUF pool");
    gst_object_unref(pool);
    found = true;
    break;
  }
  guint64 proposals = 0;
  g_object_get(bridge, "allocation-proposal-count", &proposals, nullptr);
  if (provider_available) {
    require(found, "neatcamerabridge did not propose a usable standard DMA-BUF pool");
    require(proposals == 1, "neatcamerabridge proposal counter is incorrect");
  } else {
    require(!found && proposals == 0,
            "neatcamerabridge proposed a pool without the optional camera-memory provider");
  }

  gst_query_unref(query);
  gst_caps_unref(caps);
  gst_object_unref(sink);
  gst_object_unref(bridge);
  return found;
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

void require_live_sample_layout(GstSample* sample) {
  GstBuffer* buffer = sample ? gst_sample_get_buffer(sample) : nullptr;
  GstVideoMeta* video_meta = buffer ? gst_buffer_get_video_meta(buffer) : nullptr;
  require(video_meta && video_meta->format == GST_VIDEO_FORMAT_NV12 && video_meta->width == 1920 &&
              video_meta->height == 1080 && video_meta->n_planes == 2,
          "live zero-copy output lost GstVideoMeta");
  require(gst_buffer_n_memory(buffer) == 1,
          "live zero-copy output did not restore one packed memory");
  GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
  require(memory && gst_simaai_memory_get_segment(memory, "camera"),
          "live zero-copy output did not publish the configured packed segment");
}

void prove_fallback_pool_does_not_block_leaky_queue() {
  constexpr unsigned int kInputFrames = 24;
  const std::string description =
      "appsrc name=src is-live=false format=time block=true max-buffers=1 "
      "caps=video/x-raw,format=NV12,width=64,height=32,framerate=30/1 ! "
      "neatcamerabridge name=bridge copy-allowed=true num-buffers=2 ! "
      "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
      "identity sleep-time=100000 ! fakesink name=sink sync=false signal-handoffs=true";

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(description.c_str(), &error);
  if (!pipeline) {
    const std::string message = error ? error->message : "unknown parse error";
    g_clear_error(&error);
    throw std::runtime_error("failed to build fallback queue pipeline: " + message);
  }

  GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "src");
  GstElement* bridge = gst_bin_get_by_name(GST_BIN(pipeline), "bridge");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  require(source && bridge && sink, "fallback queue pipeline elements are missing");

  std::atomic<unsigned int> output_frames{0};
  g_signal_connect(
      sink, "handoff", G_CALLBACK(+[](GstElement*, GstBuffer*, GstPad*, gpointer data) {
        static_cast<std::atomic<unsigned int>*>(data)->fetch_add(1, std::memory_order_relaxed);
      }),
      &output_frames);

  GstBus* bus = gst_element_get_bus(pipeline);
  auto cleanup = [&]() {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(sink);
    gst_object_unref(bridge);
    gst_object_unref(source);
    gst_object_unref(pipeline);
  };

  guint64 copies = 0;
  try {
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "fallback queue pipeline failed to enter PLAYING");
    for (unsigned int frame = 0; frame < kInputFrames; ++frame) {
      GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 64U * 32U * 3U / 2U, nullptr);
      require(buffer != nullptr, "failed to allocate fallback test input");
      GST_BUFFER_PTS(buffer) = frame * GST_SECOND / 30U;
      GST_BUFFER_DURATION(buffer) = GST_SECOND / 30U;
      require(gst_app_src_push_buffer(GST_APP_SRC(source), buffer) == GST_FLOW_OK,
              "failed to push fallback test input");
    }
    require(gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK,
            "failed to finish fallback test input");

    GstMessage* message = gst_bus_timed_pop_filtered(
        bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    require(message != nullptr, "fallback queue pipeline timed out");
    const bool eos = GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS;
    if (!eos)
      print_bus_error(message);
    gst_message_unref(message);
    require(eos, "fallback queue pipeline failed before EOS");
    g_object_get(bridge, "copy-count", &copies, nullptr);
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();
  const unsigned int delivered = output_frames.load(std::memory_order_relaxed);
  require(copies == kInputFrames, "fallback bridge did not process every input frame");
  require(delivered < kInputFrames,
          "fallback copy pool blocked before the leaky queue could drop stale frames");
  std::cout << "FALLBACK_QUEUE_METRICS copied=" << copies << " delivered=" << delivered << '\n';
}

void prove_live_camera_negotiation(bool prove_backpressure) {
  const char* configured_camera = std::getenv("SIMA_TEST_CAMERA_NAME");
  const char* camera =
      configured_camera && *configured_camera ? configured_camera : "og05c10 5-0036";
  const std::string description =
      "libcamerasrc name=camera-src camera-name=\"" + std::string(camera) +
      "\" "
      "buffer-count=32 external-buffer-mode=required ! "
      "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
      "neatcamerabridge name=bridge copy-allowed=false num-buffers=2 "
      "capture-min-buffers=32 ! "
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
  require(g_object_class_find_property(G_OBJECT_GET_CLASS(source), "external-buffer-mode") !=
              nullptr,
          "libcamerasrc lacks the standard external-buffer policy");

  unsigned int samples = 0;
  guint64 proposals = 0;
  guint64 passthrough = 0;
  guint64 copies = 0;
  guint64 rejected = 0;
  GstBus* bus = nullptr;
  std::vector<GstSample*> held_samples;
  auto cleanup = [&]() {
    for (GstSample* sample : held_samples)
      gst_sample_unref(sample);
    held_samples.clear();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    if (bus)
      gst_object_unref(bus);
    gst_object_unref(sink);
    gst_object_unref(bridge);
    gst_object_unref(source);
    gst_object_unref(pipeline);
  };

  try {
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "live camera pipeline failed to enter PLAYING");
    const GstStateChangeReturn ready =
        gst_element_get_state(pipeline, nullptr, nullptr, 10 * GST_SECOND);
    require(ready == GST_STATE_CHANGE_SUCCESS || ready == GST_STATE_CHANGE_NO_PREROLL,
            "live camera pipeline did not finish entering PLAYING");
    bus = gst_element_get_bus(pipeline);
    require(bus != nullptr, "live camera pipeline has no bus");

    if (prove_backpressure) {
      for (unsigned int i = 0; i < 32; ++i) {
        GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), GST_SECOND);
        if (!sample) {
          if (GstMessage* message =
                  gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR))) {
            print_bus_error(message);
            gst_message_unref(message);
          }
          throw std::runtime_error("live pipeline stalled after holding " + std::to_string(i) +
                                   " of 32 buffers");
        }
        require_live_sample_layout(sample);
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
        require_live_sample_layout(sample);
        ++samples;
        gst_sample_unref(sample);
      }
    }

    g_object_get(bridge, "allocation-proposal-count", &proposals, "passthrough-count", &passthrough,
                 "copy-count", &copies, "rejected-count", &rejected, nullptr);

    std::cout << "LIVE_METRICS camera=\"" << camera << "\" samples=" << samples
              << " proposals=" << proposals << " passthrough=" << passthrough
              << " copies=" << copies << " rejected=" << rejected << '\n';
  } catch (...) {
    cleanup();
    throw;
  }

  cleanup();

  require(samples == 96, "live pipeline did not deliver 96 camera frames");
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
    const bool provider_available = prove_dmabuf_pool_proposal();
    if (provider_available)
      prove_fallback_pool_does_not_block_leaky_queue();
    const bool run_live_camera = std::getenv("SIMA_TEST_CAMERA_ALLOCATION_QUERY");
    if (run_live_camera) {
      require(provider_available,
              "live camera validation requires the optional Neat camera-memory provider");
      for (unsigned int cycle = 0; cycle < 3; ++cycle)
        prove_live_camera_negotiation(cycle == 0);
    }
    std::cout << "PASS: ";
    if (provider_available)
      std::cout << "standard DMA-BUF pool proposal";
    else
      std::cout << "graceful compatibility path without the optional camera-memory provider";
    if (run_live_camera)
      std::cout << " and three live 32-buffer zero-copy restart cycles";
    std::cout << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
