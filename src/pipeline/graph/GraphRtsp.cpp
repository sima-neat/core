/**
 * @file SessionRtsp.cpp
 * @brief RTSP server support for Graph.
 *
 * This is a mechanical split from the original monolithic Graph.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Graph.h"
#include "GraphDetail.h"
#include "internal/GraphBuildInternal.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/BackendSetting.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/GraphReport.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <simaai/scheduler/client/dma_memory.h>

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

// =====================================================================================
// RtspServer internal impl (behaviorally similar)
// =====================================================================================

struct RtspServerImpl {
  std::string mount_path;
  std::string appsrc_name;
  int rtp_port_base = -1;
  int rtp_port_count = 0;

  std::unique_ptr<GMainContext, decltype(&g_main_context_unref)> context{g_main_context_new(),
                                                                         g_main_context_unref};
  std::unique_ptr<GSource, decltype(&g_source_unref)> listener{nullptr, g_source_unref};
  std::thread th;
  std::future<void> stopped;
  std::atomic<unsigned> live_resources{0};
  std::atomic<bool> running{false};
  std::atomic<bool> stop_requested{false};

  int enc_w = 0;
  int enc_h = 0;
  int fps = 30;
  std::shared_ptr<std::vector<uint8_t>> nv12_enc;
  std::shared_ptr<void> guard;
};

struct PushCtx {
  GstElement* appsrc = nullptr;
  std::mutex appsrc_mu;
  GSource* timer = nullptr;

  int w = 0, h = 0, fps = 30;
  guint64 frame_count = 0;
  guint64 frame_duration_ns = 0;

  std::shared_ptr<std::vector<uint8_t>> nv12;
  GstVideoInfo source_info{};
  GstVideoInfo surface_info{};
  GstVideoConverter* converter = nullptr;
  GstBufferPool* surface_pool = nullptr;
  GstBuffer* source_buffer = nullptr;

  ~PushCtx() {
    if (timer)
      g_source_unref(timer);
    gst_clear_buffer(&source_buffer);
    if (converter)
      gst_video_converter_free(converter);
    if (surface_pool) {
      gst_buffer_pool_set_flushing(surface_pool, TRUE);
      gst_buffer_pool_set_active(surface_pool, FALSE);
      gst_object_unref(surface_pool);
    }
  }
  std::atomic<bool> stopped{false};
  std::atomic<bool> need_data{true};
  std::atomic<int> refs{1};
};

using RtspImplPtr = std::shared_ptr<RtspServerImpl>;

// A DESCRIBE request owns its preparing media before the client has a session.
static void track_rtsp_resource(GObject* object, const RtspImplPtr& impl) {
  ++impl->live_resources;
  g_object_set_data_full(object, "neat-rtsp-retirement", new RtspImplPtr(impl), [](gpointer data) {
    std::unique_ptr<RtspImplPtr> owner(static_cast<RtspImplPtr*>(data));
    if (--(*owner)->live_resources == 0)
      g_main_context_wakeup((*owner)->context.get());
  });
}

static RtspImplPtr* rtsp_impl_holder(void* impl) {
  return reinterpret_cast<RtspImplPtr*>(impl);
}

static RtspImplPtr rtsp_impl_shared(void* impl) {
  return impl ? *rtsp_impl_holder(impl) : RtspImplPtr{};
}

static bool rtsp_debug_enabled() {
  return simaai::neat::pipeline_internal::env_bool("SIMA_RTSP_DEBUG", false);
}

static bool stop_trace_enabled() {
  return simaai::neat::pipeline_internal::env_bool("SIMA_STOP_TRACE", false);
}

static std::string ensure_mount_path(const std::string& mount) {
  if (mount.empty())
    return "/image";
  if (mount[0] == '/')
    return mount;
  return "/" + mount;
}

static std::string make_rtsp_url(int port, const std::string& mount) {
  return "rtsp://127.0.0.1:" + std::to_string(port) + ensure_mount_path(mount);
}

static void push_ctx_ref(PushCtx* pc) {
  if (!pc)
    return;
  pc->refs.fetch_add(1, std::memory_order_relaxed);
}

static void push_ctx_unref(PushCtx* pc) {
  if (!pc)
    return;
  if (pc->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete pc;
  }
}

static GstElement* push_ctx_ref_appsrc(PushCtx* pc) {
  if (!pc)
    return nullptr;
  std::lock_guard<std::mutex> lock(pc->appsrc_mu);
  if (!pc->appsrc)
    return nullptr;
  return GST_ELEMENT(gst_object_ref(pc->appsrc));
}

static GstElement* push_ctx_take_appsrc(PushCtx* pc) {
  if (!pc)
    return nullptr;
  std::lock_guard<std::mutex> lock(pc->appsrc_mu);
  GstElement* src = pc->appsrc;
  pc->appsrc = nullptr;
  return src;
}

static void appsrc_need_data(GstAppSrc* /*src*/, guint /*length*/, gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc)
    return;
  push_ctx_ref(pc);
  pc->need_data.store(true);
  push_ctx_unref(pc);
}

static void appsrc_enough_data(GstAppSrc* /*src*/, gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc)
    return;
  push_ctx_ref(pc);
  pc->need_data.store(false);
  push_ctx_unref(pc);
}

static gboolean push_frame_cb(gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc)
    return G_SOURCE_REMOVE;
  push_ctx_ref(pc);
  if (pc->stopped.load()) {
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }
  if (!pc->need_data.load()) {
    push_ctx_unref(pc);
    return G_SOURCE_CONTINUE;
  }
  if (!pc->nv12 || pc->nv12->empty()) {
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  const size_t y_sz = (size_t)pc->w * (size_t)pc->h;
  const size_t uv_sz = y_sz / 2;
  const size_t total = y_sz + uv_sz;
  if (pc->nv12->size() != total) {
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  GstElement* appsrc = push_ctx_ref_appsrc(pc);
  if (!appsrc) {
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  // This GLib callback must never block teardown waiting for a codec slot.
  // If every surface is retained, retry this same still-image frame next tick.
  GstBufferPoolAcquireParams acquire{};
  acquire.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
  GstBuffer* buf = nullptr;
  const GstFlowReturn acquired = gst_buffer_pool_acquire_buffer(pc->surface_pool, &buf, &acquire);
  if (acquired == GST_FLOW_EOS) {
    gst_object_unref(appsrc);
    push_ctx_unref(pc);
    return G_SOURCE_CONTINUE;
  }
  ::simaai::scheduler::client::dma::Error error;
  if (acquired != GST_FLOW_OK || !buf ||
      !::simaai::scheduler::client::dma::convertEncoderInput(
          pc->converter, pc->source_info, pc->source_buffer, pc->surface_info, buf, &error)) {
    if (acquired != GST_FLOW_FLUSHING) {
      GST_ELEMENT_ERROR(appsrc, RESOURCE, WRITE, ("Cannot prepare RTSP encoder DMA input"),
                        ("%s", error.message().c_str()));
    }
    gst_clear_buffer(&buf);
    gst_object_unref(appsrc);
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }
  const guint64 pts = pc->frame_count * pc->frame_duration_ns;
  GST_BUFFER_PTS(buf) = pts;
  GST_BUFFER_DTS(buf) = pts;
  GST_BUFFER_DURATION(buf) = pc->frame_duration_ns;

  GstFlowReturn fr = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);
  gst_object_unref(appsrc);
  if (fr == GST_FLOW_OK) {
    pc->frame_count++;
    push_ctx_unref(pc);
    return G_SOURCE_CONTINUE;
  }
  if (fr == GST_FLOW_FLUSHING || fr == GST_FLOW_EOS) {
    if (rtsp_debug_enabled()) {
      std::fprintf(stderr, "[rtsp] push backpressure ret=%s (frame=%" G_GUINT64_FORMAT ")\n",
                   gst_flow_get_name(fr), pc->frame_count);
    }
    push_ctx_unref(pc);
    return G_SOURCE_CONTINUE;
  }
  if (rtsp_debug_enabled()) {
    std::fprintf(stderr, "[rtsp] push failed ret=%s (frame=%" G_GUINT64_FORMAT ")\n",
                 gst_flow_get_name(fr), pc->frame_count);
  }
  pc->stopped.store(true);
  push_ctx_unref(pc);
  return G_SOURCE_REMOVE;
}

static void stop_media_push(PushCtx* pc) {
  pc->stopped.store(true);
  if (pc->timer)
    g_source_destroy(pc->timer);

  if (GstElement* src = push_ctx_take_appsrc(pc)) {
    g_signal_handlers_disconnect_by_data(src, pc);
    gst_object_unref(src);
  }
}

// RtspServerHandle implementation
RtspServerHandle::~RtspServerHandle() {
  stop();
}

RtspServerHandle::RtspServerHandle(RtspServerHandle&& o) noexcept {
  url_ = std::move(o.url_);
  impl_ = o.impl_;
  guard_ = std::move(o.guard_);
  o.impl_ = nullptr;
}

RtspServerHandle& RtspServerHandle::operator=(RtspServerHandle&& o) noexcept {
  if (this != &o) {
    stop();
    url_ = std::move(o.url_);
    impl_ = o.impl_;
    guard_ = std::move(o.guard_);
    o.impl_ = nullptr;
  }
  return *this;
}

bool RtspServerHandle::running() const {
  auto impl = rtsp_impl_shared(impl_);
  return impl && impl->running.load();
}

void RtspServerHandle::stop() {
  auto* holder = rtsp_impl_holder(impl_);
  if (!holder)
    return;
  RtspImplPtr impl = *holder;
  if (!impl)
    return;

  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] RtspServerHandle::stop begin\n");
  }
  if (rtsp_debug_enabled()) {
    std::fprintf(stderr, "[rtsp] stop: running=%d\n", impl->running.load() ? 1 : 0);
  }
  if (impl->listener) {
    g_source_destroy(impl->listener.get());
    impl->listener.reset();
  }
  impl->stop_requested.store(true);
  g_main_context_wakeup(impl->context.get());

  if (impl->th.joinable()) {
    const auto timeout = std::chrono::milliseconds(
        std::max(0, pipeline_internal::env_int("SIMA_RTSP_STOP_TIMEOUT_MS", 2000)));
    if (impl->stopped.wait_for(timeout) == std::future_status::ready)
      impl->th.join();
    else
      impl->th.detach(); // The worker owns the context and device guard until retirement.
  }

  delete holder;
  impl_ = nullptr;

  guard_.reset();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] RtspServerHandle::stop end\n");
  }
}

RtspServerHandle Graph::run_rtsp(const RtspServerOptions& opt) {
  pipeline_internal::require_gstreamer_backend(
      "Graph::run_rtsp", {error_codes::kPipelineShape, "this graph's RTSP server",
                          "it streams the graph's H.264 output over RTSP"});
  if (opt.port < 0 || opt.port > 65535)
    pipeline_internal::error_util::throw_session_error(
        error_codes::kOptionOutOfRange, "RtspServerOptions.port must be in [0, 65535].");
  gst_init_once();

  const auto nodes = linear_nodes_snapshot("Graph::run_rtsp");
  enforce_caps_behavior(nodes, "Graph::run_rtsp");

  require_element("appsrc", "Graph::run_rtsp");
  require_element("rtph264pay", "Graph::run_rtsp");
  require_element("h264parse", "Graph::run_rtsp");

  const StillImageInput* src_img = find_rtsp_still_image(nodes, "Graph::run_rtsp");
  if (!src_img) {
    throw std::runtime_error("Graph::run_rtsp: missing StillImageInput node");
  }

  const NameTransform name_transform = make_name_transform(opt_);
  std::vector<LaunchNameOrigin> name_origins;
  std::ostringstream ss;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i)
      ss << " ! ";
    NodeFragment frag = make_node_fragment(nodes[i], static_cast<int>(i), name_transform);
    for (const auto& name : frag.element_names) {
      name_origins.push_back(LaunchNameOrigin{
          .kind = LaunchNameOrigin::Kind::NodeFragment,
          .name = name,
          .node_index = static_cast<int>(i),
          .node_kind = nodes[i] ? nodes[i]->kind() : "",
          .role = "RTSP Node fragment",
      });
    }
    ss << frag.fragment;
  }
  std::string launch = ss.str();
  const std::string pay_token = "rtph264pay";
  const size_t pay_pos = launch.find(pay_token);
  if (pay_pos != std::string::npos) {
    const std::string capsfix_name = apply_name_transform(name_transform, "rtsp_h264_capsfix");
    name_origins.push_back(LaunchNameOrigin{
        .kind = LaunchNameOrigin::Kind::RtspHelper,
        .name = capsfix_name,
        .role = "injected H.264 caps filter",
    });
    const std::string caps = "capsfilter name=" + capsfix_name +
                             " caps=\"video/x-h264,parsed=true,"
                             "stream-format=(string)byte-stream,alignment=(string)au,"
                             "width=(int)" +
                             std::to_string(src_img->enc_w()) + ",height=(int)" +
                             std::to_string(src_img->enc_h()) + ",framerate=(fraction)" +
                             std::to_string(src_img->fps()) + "/1\"";
    launch.insert(pay_pos, caps + " ! ");
  }
  last_pipeline_ = "( " + launch + " )";
  session_build_validate_explicit_launch_names_or_throw(last_pipeline_, "Graph::run_rtsp",
                                                        name_origins);

  auto guard = guard_;

  auto impl_sp = std::make_shared<RtspServerImpl>();
  auto* impl = impl_sp.get();
  impl->rtp_port_base = opt.rtp_port_base;
  impl->rtp_port_count = opt.rtp_port_count;
  impl->mount_path = ensure_mount_path(opt.mount);
  impl->appsrc_name = apply_name_transform(name_transform, "mysrc");
  impl->enc_w = src_img->enc_w();
  impl->enc_h = src_img->enc_h();
  impl->fps = src_img->fps();
  impl->nv12_enc = src_img->nv12_enc();
  impl->guard = guard;

  g_autoptr(GstRTSPServer) server = gst_rtsp_server_new();
  gst_rtsp_server_set_service(server, std::to_string(opt.port).c_str());
  GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
  GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_launch(factory, last_pipeline_.c_str());
  gst_rtsp_media_factory_set_shared(factory, FALSE);

  if (impl->rtp_port_base > 0 && impl->rtp_port_count > 0) {
    const int min_port = impl->rtp_port_base;
    const int max_port = impl->rtp_port_base + impl->rtp_port_count - 1;
    if (min_port > 0 && max_port <= 65535 && min_port <= max_port) {
      GstRTSPAddressPool* pool = gst_rtsp_address_pool_new();
      const gboolean added = gst_rtsp_address_pool_add_range(pool, "0.0.0.0", "0.0.0.0",
                                                             static_cast<guint16>(min_port),
                                                             static_cast<guint16>(max_port), 0);
      if (added) {
        gst_rtsp_media_factory_set_address_pool(factory, pool);
        if (rtsp_debug_enabled()) {
          std::fprintf(stderr, "[rtsp] rtp-port-range=%d-%d\n", min_port, max_port);
        }
      } else if (rtsp_debug_enabled()) {
        std::fprintf(stderr, "[rtsp] failed to set rtp-port-range=%d-%d\n", min_port, max_port);
      }
      g_object_unref(pool);
    } else if (rtsp_debug_enabled()) {
      std::fprintf(stderr, "[rtsp] invalid rtp-port-range=%d+%d\n", impl->rtp_port_base,
                   impl->rtp_port_count);
    }
  }

  g_signal_connect_data(
      server, "client-connected",
      G_CALLBACK(+[](GstRTSPServer*, GstRTSPClient* client, gpointer data) {
        track_rtsp_resource(G_OBJECT(client), *static_cast<RtspImplPtr*>(data));
      }),
      new RtspImplPtr(impl_sp),
      [](gpointer data, GClosure*) { delete static_cast<RtspImplPtr*>(data); }, G_CONNECT_DEFAULT);
  g_signal_connect_data(
      factory, "media-configure",
      G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer user_data) {
        const auto& impl = *static_cast<RtspImplPtr*>(user_data);
        track_rtsp_resource(G_OBJECT(media), impl);

        GstElement* top = gst_rtsp_media_get_element(media);
        if (!top)
          return;

        GstElement* src = gst_bin_get_by_name_recurse_up(GST_BIN(top), impl->appsrc_name.c_str());
        if (!src) {
          gst_object_unref(top);
          return;
        }

        GstCaps* caps =
            gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT,
                                impl->enc_w, "height", G_TYPE_INT, impl->enc_h, "framerate",
                                GST_TYPE_FRACTION, impl->fps, 1, nullptr);
        gst_app_src_set_caps(GST_APP_SRC(src), caps);
        gst_caps_unref(caps);

        g_object_set(G_OBJECT(src), "is-live", TRUE, "format", GST_FORMAT_TIME, "do-timestamp",
                     FALSE, "block", FALSE, nullptr);

        auto* pc = new PushCtx();
        pc->appsrc = (GstElement*)gst_object_ref(src);
        pc->w = impl->enc_w;
        pc->h = impl->enc_h;
        pc->fps = impl->fps;
        pc->nv12 = impl->nv12_enc;
        pc->frame_duration_ns = gst_util_uint64_scale_int(GST_SECOND, 1, pc->fps);
        ::simaai::scheduler::client::dma::Error error;
        gst_video_info_init(&pc->source_info);
        const bool valid_surface =
            gst_video_info_set_format(&pc->source_info, GST_VIDEO_FORMAT_NV12, pc->w, pc->h) &&
            ::simaai::scheduler::client::dma::encoderInputInfo(pc->source_info, &pc->surface_info,
                                                               &error);
        if (valid_surface) {
          // StillImageInput owns tight NV12, independent of GstVideoInfo's
          // generic four-byte row rounding for arbitrary even widths.
          pc->source_info.stride[0] = pc->w;
          pc->source_info.stride[1] = pc->w;
          pc->source_info.offset[1] = static_cast<gsize>(pc->w) * pc->h;
          pc->source_info.size = pc->source_info.offset[1] * 3 / 2;
          pc->converter = gst_video_converter_new(&pc->source_info, &pc->surface_info, nullptr);
          pc->surface_pool = ::simaai::scheduler::client::dma::createEncoderInputPool(
              pc->surface_info, 2, 30, &error);
          if (pc->nv12) {
            // The shared image vector owns these immutable source bytes.
            // Only final DMA surfaces are ever submitted to appsrc.
            pc->source_buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY,
                                                            pc->nv12->data(), pc->nv12->size(), 0,
                                                            pc->nv12->size(), nullptr, nullptr);
          }
        }
        if (!pc->converter || !pc->surface_pool || !pc->source_buffer) {
          GST_ELEMENT_ERROR(src, RESOURCE, OPEN_WRITE,
                            ("Cannot initialize RTSP encoder DMA surfaces"),
                            ("%s", error.message().c_str()));
          gst_object_unref(push_ctx_take_appsrc(pc));
          delete pc;
          gst_object_unref(src);
          gst_object_unref(top);
          return;
        }

        const auto release_push_ctx =
            +[](gpointer data, GClosure*) { push_ctx_unref(static_cast<PushCtx*>(data)); };
        push_ctx_ref(pc);
        g_signal_connect_data(src, "need-data", G_CALLBACK(appsrc_need_data), pc, release_push_ctx,
                              G_CONNECT_DEFAULT);
        push_ctx_ref(pc);
        g_signal_connect_data(src, "enough-data", G_CALLBACK(appsrc_enough_data), pc,
                              release_push_ctx, G_CONNECT_DEFAULT);

        const int period_ms = std::max(1, 1000 / pc->fps);
        pc->timer = g_timeout_source_new(period_ms);
        push_ctx_ref(pc);
        g_source_set_callback(pc->timer, push_frame_cb, pc,
                              [](gpointer data) { push_ctx_unref(static_cast<PushCtx*>(data)); });
        g_source_attach(pc->timer, impl->context.get());
        g_signal_connect_data(
            media, "unprepared", G_CALLBACK(+[](GstRTSPMedia*, gpointer data) {
              stop_media_push(static_cast<PushCtx*>(data));
            }),
            pc,
            [](gpointer data, GClosure*) {
              auto* ctx = static_cast<PushCtx*>(data);
              stop_media_push(ctx);
              push_ctx_unref(ctx);
            },
            G_CONNECT_DEFAULT);

        gst_object_unref(src);
        gst_object_unref(top);
      }),
      new RtspImplPtr(impl_sp),
      [](gpointer data, GClosure*) { delete static_cast<RtspImplPtr*>(data); }, G_CONNECT_DEFAULT);

  gst_rtsp_mount_points_add_factory(mounts, impl->mount_path.c_str(), factory);
  g_object_unref(mounts);

  g_autoptr(GError) bind_error = nullptr;
  std::unique_ptr<GSource, decltype(&g_source_unref)> listener{
      gst_rtsp_server_create_source(server, nullptr, &bind_error), g_source_unref};
  if (!listener || !g_source_attach(listener.get(), impl->context.get())) {
    std::string message = "Could not bind the RTSP server to port " + std::to_string(opt.port);
    if (bind_error)
      message += ": " + std::string(bind_error->message);
    pipeline_internal::error_util::throw_session_error(
        error_codes::kIoOpen, message, {},
        "Check whether the port is occupied or requires additional privileges.");
  }

  RtspServerHandle handle;
  handle.impl_ = new RtspImplPtr(impl_sp);
  handle.guard_ = std::move(guard);
  const int port = gst_rtsp_server_get_bound_port(server);
  handle.url_ = make_rtsp_url(port, opt.mount);
  if (rtsp_debug_enabled())
    std::fprintf(stderr, "[rtsp] starting server on port %d\n", port);
  impl->running.store(true);
  impl->listener = std::move(listener);
  std::promise<void> stopped;
  impl->stopped = stopped.get_future();
  impl->th = std::thread([impl_sp, stopped = std::move(stopped),
                          server = std::unique_ptr<GstRTSPServer, decltype(&g_object_unref)>(
                              g_steal_pointer(&server), g_object_unref)]() mutable {
    while (!impl_sp->stop_requested.load())
      g_main_context_iteration(impl_sp->context.get(), TRUE);
    impl_sp->running.store(false);
    // Request references under the server lock and close the clients after it is released.
    GList* clients = gst_rtsp_server_client_filter(server.get(), nullptr, nullptr);
    for (GList* item = clients; item; item = item->next)
      gst_rtsp_client_close(GST_RTSP_CLIENT(item->data));
    g_list_free_full(clients, g_object_unref);
    g_autoptr(GstRTSPSessionPool) sessions = gst_rtsp_server_get_session_pool(server.get());
    gst_rtsp_session_pool_filter(
        sessions,
        +[](GstRTSPSessionPool*, GstRTSPSession*, gpointer) { return GST_RTSP_FILTER_REMOVE; },
        nullptr);
    // Keep dispatching frame timers while an accepted client is finishing preroll.
    while (impl_sp->live_resources.load() != 0)
      g_main_context_iteration(impl_sp->context.get(), TRUE);
    stopped.set_value_at_thread_exit();
    if (rtsp_debug_enabled())
      std::fprintf(stderr, "[rtsp] thread exit\n");
  });
  return handle;
}

} // namespace simaai::neat
