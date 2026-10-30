#include "neat.h"
#include "gst/GstInit.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "test_utils.h"
#include "runtime_test_utils.h"
#include "pipeline/ErrorCodes.h"

#include <gst/rtsp-server/rtsp-server.h>

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using namespace simaai::neat;
using namespace std::chrono_literals;

class Publisher {
public:
  Publisher()
      : port(sima_test::allocate_local_rtsp_port()), context(g_main_context_new()),
        loop(g_main_loop_new(context, FALSE)), worker([this] { g_main_loop_run(loop); }) {}
  ~Publisher() {
    stop();
    g_main_loop_quit(loop);
    worker.join();
    for (auto* gate : gates)
      gst_object_unref(gate);
    g_main_loop_unref(loop);
    g_main_context_unref(context);
  }

  void start() {
    server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, std::to_string(port).c_str());
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(
        factory,
        "( videotestsrc is-live=true ! video/x-raw,format=I420,width=160,height=120,framerate=20/1 "
        "! valve name=gate ! jpegenc ! rtpjpegpay name=pay0 pt=26 )");
    g_signal_connect(factory, "media-configure",
                     G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer data) {
                       auto& self = *static_cast<Publisher*>(data);
                       GstElement* pipeline = gst_rtsp_media_get_element(media);
                       GstElement* gate = gst_bin_get_by_name(GST_BIN(pipeline), "gate");
                       gst_object_unref(pipeline);
                       std::lock_guard<std::mutex> lock(self.mutex);
                       self.gates.push_back(gate);
                     }),
                     this);
    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mounts, "/camera", factory);
    g_object_unref(mounts);
    source_id = gst_rtsp_server_attach(server, context);
    require(source_id != 0, "RTSP publisher failed to bind");
  }

  void stop() {
    if (!server)
      return;
    if (auto* source = g_main_context_find_source_by_id(context, source_id))
      g_source_destroy(source);
    gst_rtsp_server_client_filter(
        server,
        +[](GstRTSPServer*, GstRTSPClient* client, gpointer) -> GstRTSPFilterResult {
          gst_rtsp_client_close(client);
          return GST_RTSP_FILTER_REMOVE;
        },
        nullptr);
    g_object_unref(server);
    server = nullptr;
  }

  void silence() {
    std::lock_guard<std::mutex> lock(mutex);
    require(!gates.empty(), "publisher has no active media");
    g_object_set(gates.back(), "drop", TRUE, nullptr);
  }

  std::size_t connections() {
    std::lock_guard<std::mutex> lock(mutex);
    return gates.size();
  }

  std::string url() const {
    return "rtsp://127.0.0.1:" + std::to_string(port) + "/camera";
  }

private:
  int port;
  GMainContext* context;
  GMainLoop* loop;
  std::thread worker;
  GstRTSPServer* server = nullptr;
  guint source_id = 0;
  std::mutex mutex;
  std::vector<GstElement*> gates;
};

Graph source(const Publisher& publisher, int timeout = 600, int attempts = -1) {
  nodes::groups::RtspEncodedInputOptions options;
  options.url = publisher.url();
  options.codec = nodes::groups::RtspCodec::MJPEG;
  options.source_fps = 20;
  options.reconnect_timeout_ms = timeout;
  options.reconnect_attempts = attempts;
  return nodes::groups::RtspEncodedInput(options);
}

void frame(Run& run, const std::string& name, int timeout = 6000) {
  Sample sample;
  PullError error;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = run.pull(name, 100, sample, &error);
    if (status == PullStatus::Ok) {
      require(!sample.tensors.empty(), "RTSP output is empty");
      return;
    }
    require(status == PullStatus::Timeout, "RTSP output failed: " + error.message);
  }
  throw std::runtime_error("RTSP source did not produce a frame: " + name);
}

void closed(Run& run, const std::string& name) {
  Sample sample;
  PullError error;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = run.pull(name, 100, sample, &error);
    if (status == PullStatus::Closed) {
      require(error.code == error_codes::kRtspConnectionFailed,
              "exhaustion lost connection-failure detail: " + error.code + " " + error.message);
      return;
    }
    require(status != PullStatus::Error,
            "retry exhaustion stopped the whole Run: " + error.message);
  }
  throw std::runtime_error("retry budget did not close the source");
}
} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    Publisher healthy, failing;
    healthy.start();
    failing.start();
    Graph graph;
    graph.connect(source(healthy), nodes::Output("healthy", OutputOptions::Latest()));
    graph.connect(source(failing, 600, 12), nodes::Output("failing", OutputOptions::Latest()));
    Run run = graph.build();
    std::cout << "phase=initial\n" << std::flush;
    frame(run, "healthy");
    frame(run, "failing");
    for (int cycle = 0; cycle < 2; ++cycle) {
      std::cout << "phase=restart cycle=" << cycle << "\n" << std::flush;
      failing.stop();
      const auto end = std::chrono::steady_clock::now() + 5s;
      while (std::chrono::steady_clock::now() < end)
        frame(run, "healthy", 1500);
      failing.start();
      frame(run, "failing");
    }
    std::cout << "phase=silence\n" << std::flush;
    const auto before_silence = failing.connections();
    failing.silence();
    const auto silent_deadline = std::chrono::steady_clock::now() + 8s;
    while (failing.connections() == before_silence &&
           std::chrono::steady_clock::now() < silent_deadline)
      frame(run, "healthy", 1500);
    require(failing.connections() > before_silence, "silent RTSP connection was not restarted");
    frame(run, "failing", 8000);
    run.close();

    std::cout << "phase=exhaustion\n" << std::flush;
    // Exhausting one offline producer must leave the healthy peer running.
    failing.stop();
    Graph offline;
    offline.connect(source(healthy), nodes::Output("healthy", OutputOptions::Latest()));
    offline.connect(source(failing, 600, 2), nodes::Output("failing", OutputOptions::Latest()));
    Run exhausted = offline.build();
    frame(exhausted, "healthy");
    closed(exhausted, "failing");
    frame(exhausted, "healthy");
    exhausted.close();

    std::cout << "phase=reconnect-disabled\n" << std::flush;
    failing.start();
    Graph disabled;
    disabled.add(source(failing, 0));
    disabled.add(nodes::Output("disabled", OutputOptions::Latest()));
    Run unchanged = disabled.build();
    frame(unchanged, "disabled");
    failing.stop();
    Sample ignored;
    PullError close_error;
    PullStatus final_status = PullStatus::Ok;
    const auto disabled_deadline = std::chrono::steady_clock::now() + 8s;
    while ((final_status == PullStatus::Ok || final_status == PullStatus::Timeout) &&
           std::chrono::steady_clock::now() < disabled_deadline)
      final_status = unchanged.pull("disabled", 100, ignored, &close_error);
    require(final_status == PullStatus::Closed || final_status == PullStatus::Error,
            "disabled reconnect changed source failure behavior");
    unchanged.close();

    std::cout << "phase=offline-startup\n" << std::flush;
    Graph waiting;
    waiting.add(source(failing));
    waiting.add(nodes::Output("waiting"));
    Run startup = waiting.build();
    const auto offline_deadline = std::chrono::steady_clock::now() + 1200ms;
    while (std::chrono::steady_clock::now() < offline_deadline) {
      require(startup.pull("waiting", 100, ignored, &close_error) == PullStatus::Timeout,
              "initially offline source must remain recoverable: " + close_error.message);
    }
    failing.start();
    std::cout << "phase=offline-publisher-started\n" << std::flush;
    frame(startup, "waiting");
    std::cout << "phase=offline-first-frame\n" << std::flush;
    failing.stop();
    std::this_thread::sleep_for(800ms);
    std::cout << "phase=shutdown\n" << std::flush;
    const auto before_close = std::chrono::steady_clock::now();
    startup.close();
    require(std::chrono::steady_clock::now() - before_close < 3s,
            "shutdown waited for the retry budget");
    std::cout << "[OK] RTSP reconnect, offline startup, exhaustion, isolation and shutdown\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
