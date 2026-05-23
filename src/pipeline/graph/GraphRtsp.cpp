/**
 * @file SessionRtsp.cpp
 * @brief RTSP server support for Graph.
 *
 * This is a mechanical split from the original monolithic Graph.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Graph.h"
#include "GraphDetail.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/DispatcherRecovery.h"
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
  std::string url;
  std::string mount_path;
  std::string appsrc_name;
  int port = 8554;
  int rtp_port_base = -1;
  int rtp_port_count = 0;

  std::thread th;
  std::atomic<bool> running{false};
  std::atomic<bool> stop_requested{false};
  std::atomic<GMainLoop*> loop{nullptr};
  std::atomic<bool> thread_done{false};

  int enc_w = 0;
  int enc_h = 0;
  int fps = 30;
  std::shared_ptr<std::vector<uint8_t>> nv12_enc;
  std::shared_ptr<void> guard;
};

struct PushCtx {
  GstElement* appsrc = nullptr;
  std::mutex appsrc_mu;
  guint timer_id = 0;

  int w = 0, h = 0, fps = 30;
  guint64 frame_count = 0;
  guint64 frame_duration_ns = 0;

  std::shared_ptr<std::vector<uint8_t>> nv12;
  std::atomic<bool> stopped{false};
  std::atomic<bool> need_data{true};
  std::atomic<int> refs{1};
  std::atomic<bool> initial_released{false};
};

using RtspImplPtr = std::shared_ptr<RtspServerImpl>;

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

static void push_ctx_release_initial(PushCtx* pc) {
  if (!pc)
    return;
  if (!pc->initial_released.exchange(true)) {
    push_ctx_unref(pc);
  }
}

static void push_ctx_unref_with_initial_release(PushCtx* pc) {
  if (!pc)
    return;
  const bool release_initial = !pc->initial_released.exchange(true);
  int drops = release_initial ? 2 : 1;
  while (drops-- > 0) {
    if (pc->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete pc;
      return;
    }
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
    pc->timer_id = 0;
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }
  if (!pc->need_data.load()) {
    push_ctx_unref(pc);
    return G_SOURCE_CONTINUE;
  }
  if (!pc->nv12 || pc->nv12->empty()) {
    pc->timer_id = 0;
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  const size_t y_sz = (size_t)pc->w * (size_t)pc->h;
  const size_t uv_sz = y_sz / 2;
  const size_t total = y_sz + uv_sz;
  if (pc->nv12->size() != total) {
    pc->timer_id = 0;
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  GstElement* appsrc = push_ctx_ref_appsrc(pc);
  if (!appsrc) {
    pc->timer_id = 0;
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  GstBuffer* buf = gst_buffer_new_allocate(nullptr, total, nullptr);
  if (!buf) {
    pc->timer_id = 0;
    gst_object_unref(appsrc);
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  const guint64 pts = pc->frame_count * pc->frame_duration_ns;
  GST_BUFFER_PTS(buf) = pts;
  GST_BUFFER_DTS(buf) = pts;
  GST_BUFFER_DURATION(buf) = pc->frame_duration_ns;

  GstMapInfo map{};
  if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    pc->timer_id = 0;
    gst_object_unref(appsrc);
    push_ctx_unref(pc);
    return G_SOURCE_REMOVE;
  }

  std::memcpy(map.data, pc->nv12->data(), total);
  gst_buffer_unmap(buf, &map);

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
  pc->timer_id = 0;
  push_ctx_unref(pc);
  return G_SOURCE_REMOVE;
}

static void media_unprepared_cb(GstRTSPMedia*, gpointer user_data) {
  auto* pc = reinterpret_cast<PushCtx*>(user_data);
  if (!pc)
    return;

  push_ctx_ref(pc);
  pc->stopped.store(true);
  if (pc->timer_id) {
    g_source_remove(pc->timer_id);
    pc->timer_id = 0;
  }

  if (GstElement* src = push_ctx_take_appsrc(pc)) {
    g_signal_handlers_disconnect_by_data(src, pc);
    gst_object_unref(src);
  }

  push_ctx_unref_with_initial_release(pc);
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
    std::fprintf(stderr, "[rtsp] stop: running=%d loop=%p\n", impl->running.load() ? 1 : 0,
                 static_cast<void*>(impl->loop.load()));
  }
  impl->stop_requested.store(true);
  if (auto* loop = impl->loop.load(std::memory_order_acquire)) {
    g_main_loop_quit(loop);
  }
  g_main_context_wakeup(g_main_context_default());

  const int stop_timeout_ms =
      std::max(0, simaai::neat::pipeline_internal::env_int("SIMA_RTSP_STOP_TIMEOUT_MS", 2000));
  if (impl->th.joinable()) {
    if (stop_timeout_ms > 0) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(stop_timeout_ms);
      while (!impl->thread_done.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= deadline)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (impl->thread_done.load(std::memory_order_relaxed)) {
      impl->th.join();
    } else {
      if (rtsp_debug_enabled()) {
        std::fprintf(stderr, "[rtsp] stop: thread did not exit within %dms; detaching\n",
                     stop_timeout_ms);
      }
      impl->th.detach();
    }
  }

  delete holder;
  impl_ = nullptr;

  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  guard_.reset();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] RtspServerHandle::stop end\n");
  }
}

RtspServerHandle Graph::run_rtsp(const RtspServerOptions& opt) {
  gst_init_once();

  const auto nodes = linear_nodes_snapshot("Graph::run_rtsp");
  enforce_caps_behavior(nodes, "Graph::run_rtsp");

  require_element("appsrc", "Graph::run_rtsp");
  require_element("rtph264pay", "Graph::run_rtsp");
  require_element("h264parse", "Graph::run_rtsp");

  StillImageInput* src_img = nullptr;
  for (auto& n : nodes) {
    if (!src_img) {
      src_img = dynamic_cast<StillImageInput*>(n.get());
    }
  }
  if (!src_img) {
    throw std::runtime_error("Graph::run_rtsp: missing StillImageInput node");
  }

  const NameTransform name_transform = make_name_transform(opt_);
  std::ostringstream ss;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i)
      ss << " ! ";
    NodeFragment frag = make_node_fragment(nodes[i], static_cast<int>(i), name_transform);
    ss << frag.fragment;
  }
  std::string launch = ss.str();
  const std::string pay_token = "rtph264pay";
  const size_t pay_pos = launch.find(pay_token);
  if (pay_pos != std::string::npos) {
    const std::string caps = "capsfilter name=rtsp_h264_capsfix caps=\"video/x-h264,parsed=true,"
                             "stream-format=(string)byte-stream,alignment=(string)au,"
                             "width=(int)" +
                             std::to_string(src_img->enc_w()) + ",height=(int)" +
                             std::to_string(src_img->enc_h()) + ",framerate=(fraction)" +
                             std::to_string(src_img->fps()) + "/1\"";
    launch.insert(pay_pos, caps + " ! ");
  }
  last_pipeline_ = "( " + launch + " )";

  auto guard = guard_;

  auto impl_sp = std::make_shared<RtspServerImpl>();
  auto* impl = impl_sp.get();
  impl->port = opt.port;
  impl->rtp_port_base = opt.rtp_port_base;
  impl->rtp_port_count = opt.rtp_port_count;
  impl->mount_path = ensure_mount_path(opt.mount);
  impl->url = make_rtsp_url(opt.port, opt.mount);
  impl->appsrc_name = apply_name_transform(name_transform, "mysrc");
  impl->enc_w = src_img->enc_w();
  impl->enc_h = src_img->enc_h();
  impl->fps = src_img->fps();
  impl->nv12_enc = src_img->nv12_enc();
  impl->guard = guard;

  RtspServerHandle handle;
  handle.url_ = impl->url;
  handle.impl_ = new RtspImplPtr(impl_sp);
  handle.guard_ = std::move(guard);

  impl->th = std::thread([impl_sp, launch = last_pipeline_]() {
    auto* impl = impl_sp.get();
    struct DoneGuard {
      std::atomic<bool>& flag;
      ~DoneGuard() {
        flag.store(true);
      }
    } done_guard{impl->thread_done};
    if (rtsp_debug_enabled()) {
      std::fprintf(stderr, "[rtsp] thread start (port=%d)\n", impl->port);
    }
    GstRTSPServer* server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, std::to_string(impl->port).c_str());

    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, launch.c_str());
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

    g_signal_connect(factory, "media-configure",
                     G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer user_data) {
                       auto* impl = reinterpret_cast<RtspServerImpl*>(user_data);
                       if (!impl)
                         return;

                       GstElement* top = gst_rtsp_media_get_element(media);
                       if (!top)
                         return;

                       GstElement* src =
                           gst_bin_get_by_name_recurse_up(GST_BIN(top), impl->appsrc_name.c_str());
                       if (!src) {
                         gst_object_unref(top);
                         return;
                       }

                       GstCaps* caps = gst_caps_new_simple(
                           "video/x-raw", "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT,
                           impl->enc_w, "height", G_TYPE_INT, impl->enc_h, "framerate",
                           GST_TYPE_FRACTION, impl->fps, 1, nullptr);
                       gst_app_src_set_caps(GST_APP_SRC(src), caps);
                       gst_caps_unref(caps);

                       g_object_set(G_OBJECT(src), "is-live", TRUE, "format", GST_FORMAT_TIME,
                                    "do-timestamp", FALSE, "block", FALSE, nullptr);

                       auto* pc = new PushCtx();
                       pc->appsrc = (GstElement*)gst_object_ref(src);
                       pc->w = impl->enc_w;
                       pc->h = impl->enc_h;
                       pc->fps = impl->fps;
                       pc->nv12 = impl->nv12_enc;
                       pc->frame_duration_ns = gst_util_uint64_scale_int(GST_SECOND, 1, pc->fps);

                       g_signal_connect(src, "need-data", G_CALLBACK(appsrc_need_data), pc);
                       g_signal_connect(src, "enough-data", G_CALLBACK(appsrc_enough_data), pc);

                       const int period_ms = std::max(1, 1000 / pc->fps);
                       g_signal_connect(media, "unprepared", G_CALLBACK(media_unprepared_cb), pc);
                       pc->timer_id = g_timeout_add(period_ms, push_frame_cb, pc);

                       gst_object_unref(src);
                       gst_object_unref(top);
                     }),
                     impl);

    gst_rtsp_mount_points_add_factory(mounts, impl->mount_path.c_str(), factory);
    g_object_unref(mounts);

    if (gst_rtsp_server_attach(server, nullptr) == 0) {
      if (rtsp_debug_enabled()) {
        std::fprintf(stderr, "[rtsp] attach failed\n");
      }
      g_object_unref(server);
      return;
    }

    if (impl->stop_requested.load()) {
      if (rtsp_debug_enabled()) {
        std::fprintf(stderr, "[rtsp] stop requested before loop\n");
      }
      g_object_unref(server);
      return;
    }

    GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
    impl->loop.store(loop, std::memory_order_release);
    if (impl->stop_requested.load()) {
      if (rtsp_debug_enabled()) {
        std::fprintf(stderr, "[rtsp] stop requested after loop init\n");
      }
      g_main_loop_unref(loop);
      impl->loop.store(nullptr, std::memory_order_release);
      g_object_unref(server);
      return;
    }
    impl->running.store(true);
    g_main_loop_run(loop);

    impl->running.store(false);
    g_main_loop_unref(loop);
    impl->loop.store(nullptr, std::memory_order_release);

    if (rtsp_debug_enabled()) {
      std::fprintf(stderr, "[rtsp] thread exit\n");
    }
    g_object_unref(server);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  return handle;
}

} // namespace simaai::neat
