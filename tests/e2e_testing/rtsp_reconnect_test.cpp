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

Graph source(const Publisher& publisher, int timeout = 1, int attempts = -1) {
  nodes::groups::RtspEncodedInputOptions options;
  options.url = publisher.url();
  options.codec = nodes::groups::RtspCodec::MJPEG;
  options.source_fps = 20;
  options.reconnect_interval_sec = timeout;
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

void closed(Run& run, const std::string& name, const std::string& source_name) {
  Sample sample;
  PullError error;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    const auto status = run.pull(name, 100, sample, &error);
    if (status == PullStatus::Closed) {
      require(error.code == error_codes::kRtspConnectionFailed,
              "exhaustion lost connection-failure detail: " + error.code + " " + error.message);
      require(error.source == source_name, "lost source name: " + error.source);
      return;
    }
    require(status != PullStatus::Error,
            "retry exhaustion stopped the whole Run: " + error.message);
  }
  throw std::runtime_error("retry budget did not close the source");
}
void transition(Run& run, SourceStatus expected) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto event = run.poll_source_event()) {
      require(event->source == "src7" && event->status == expected,
              "unexpected source transition for " + event->source);
      return;
    }
    std::this_thread::sleep_for(10ms);
  }
  throw std::runtime_error("missing source transition");
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
    graph.connect(source(failing, 1, 12).set_name("src7"),
                  nodes::Output("failing", OutputOptions::Latest()));
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
      transition(run, SourceStatus::Reconnecting);
      failing.start();
      frame(run, "failing");
      transition(run, SourceStatus::Recovered);
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
    offline.connect(source(failing, 1, 2).set_name("src7"),
                    nodes::Output("failing", OutputOptions::Latest()));
    Run exhausted = offline.build();
    frame(exhausted, "healthy");
    closed(exhausted, "failing", "src7");
    transition(exhausted, SourceStatus::Reconnecting);
    transition(exhausted, SourceStatus::Disconnected);
    frame(exhausted, "healthy");
    exhausted.close();

    std::cout << "phase=reconnect-disabled\n" << std::flush;
    failing.start();
    Graph disabled;
    // Seven camera outputs share a Run with the lost eighth source.
    for (int i = 0; i < 7; ++i) {
      disabled.connect(source(healthy, 0).set_name("src" + std::to_string(i)),
                       nodes::Output("frame" + std::to_string(i), OutputOptions::Latest()));
    }
    disabled.connect(source(failing, 0).set_name("src7"), nodes::Output("frame7"));
    Run unchanged = disabled.build();
    frame(unchanged, "frame7");
    failing.stop();
    closed(unchanged, "frame7", "src7");
    transition(unchanged, SourceStatus::Disconnected);
    for (int i = 0; i < 7; ++i)
      frame(unchanged, "frame" + std::to_string(i));
    Sample ignored;
    PullError close_error;
    unchanged.close();

    std::cout << "phase=status-only-timeout\n" << std::flush;
    failing.start();
    Graph monitor;
    monitor.connect(source(failing, 1, 0).set_name("src7"), nodes::Output("frame7"));
    Run watch = monitor.build();
    frame(watch, "frame7");
    const auto sessions = failing.connections();
    failing.silence();
    closed(watch, "frame7", "src7");
    transition(watch, SourceStatus::Disconnected);
    require(failing.connections() == sessions, "monitor-only source attempted reconnect");
    watch.close();
    failing.stop();

    std::cout << "phase=downstream-error\n" << std::flush;
    Graph broken;
    broken.add(source(healthy));
    broken.add(nodes::Custom("identity error-after=15"));
    broken.add(nodes::Output("broken"));
    Run faulty = broken.build();
    PullStatus status = PullStatus::Ok;
    const auto error_deadline = std::chrono::steady_clock::now() + 5s;
    while ((status == PullStatus::Ok || status == PullStatus::Timeout) &&
           std::chrono::steady_clock::now() < error_deadline)
      status = faulty.pull("broken", 100, ignored, &close_error);
    require(status == PullStatus::Error && close_error.code != error_codes::kRtspConnectionFailed,
            "a downstream failure was classified as a camera loss");
    require(!faulty.poll_source_event(), "downstream failure triggered an RTSP retry");
    faulty.close();

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
