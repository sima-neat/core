#include "asset_utils.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/sima/SimaEncode.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/Graph.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

namespace {

bool runtime_missing_rtsp_elements(const std::string& msg) {
  return msg.find("missing element") != std::string::npos ||
         msg.find("No such element") != std::string::npos ||
         msg.find("not found") != std::string::npos;
}

void check_active_client_shutdown(simaai::neat::RtspServerHandle& h) {
  const std::string launch = "rtspsrc protocols=tcp latency=0 location=" + h.url() +
                             " ! rtph264depay ! appsink name=frames sync=false";
  g_autoptr(GstElement) client = gst_parse_launch(launch.c_str(), nullptr);
  require(client != nullptr, "could not create the loopback client");
  g_autoptr(GstElement) sink = gst_bin_get_by_name(GST_BIN(client), "frames");
  gst_element_set_state(client, GST_STATE_PLAYING);
  g_autoptr(GstSample) frame = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 10 * GST_SECOND);
  require(frame != nullptr, "client must receive media before stopping the server");
  h.stop();
  g_autoptr(GstBus) bus = gst_element_get_bus(client);
  g_autoptr(GstMessage) terminal = gst_bus_timed_pop_filtered(
      bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  gst_element_set_state(client, GST_STATE_NULL);
  require(terminal != nullptr, "active client did not observe server shutdown");
}

} // namespace

RUN_TEST("graph_rtsp_lifecycle_regression_test", ([] {
           using namespace simaai::neat;
           namespace fs = std::filesystem;

           // Default handle lifecycle should be safe and idempotent.
           RtspServerHandle handle;
           require(!handle.running(), "default RTSP handle should not be running");
           handle.stop();
           handle.stop();

           Graph graph;
           graph.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
           graph.custom("fakesink sync=false");

           try {
             RtspServerOptions opt;
             opt.mount = "stream";
             opt.port = sima_test::allocate_local_rtsp_port();
             (void)graph.run_rtsp(opt);
             throw std::runtime_error(
                 "Graph::run_rtsp unexpectedly succeeded without StillImageInput");
           } catch (const std::exception& e) {
             const std::string msg = e.what();
             if (runtime_missing_rtsp_elements(msg)) {
               throw std::runtime_error(std::string("RTSP runtime elements missing: ") + msg);
             }
             require_contains(msg, "missing StillImageInput node",
                              "Graph::run_rtsp should reject non-StillImageInput pipelines");
           }

           const fs::path image = sima_test::test_image_fixture_path();
           if (!fs::exists(image)) {
             throw std::runtime_error("graph_rtsp_lifecycle_regression_test: missing test.jpg at " +
                                      image.string());
           }

           auto make_rtsp_graph = [&]() {
             Graph s;
             s.add(nodes::StillImageInput(image.string(), 640, 480, 640, 480, 10));
             SimaEncodeOptions encode;
             encode.fps = 10;
             s.add(nodes::SimaEncode(encode));
             s.add(nodes::H264Parse(1));
             s.add(nodes::H264Packetize(96, 1));
             return s;
           };

           // Start/stop lifecycle with explicit idempotent stop check.
           {
             Graph rtsp_graph = make_rtsp_graph();
             RtspServerOptions opt;
             opt.mount = "lifecycle";
             opt.port = sima_test::allocate_local_rtsp_port();

             RtspServerHandle h;
             try {
               h = rtsp_graph.run_rtsp(opt);
             } catch (const std::exception& e) {
               if (runtime_missing_rtsp_elements(e.what())) {
                 throw std::runtime_error(std::string("RTSP runtime unavailable: ") + e.what());
               }
               throw;
             }

             std::this_thread::sleep_for(std::chrono::milliseconds(350));
             require(h.running(), "Graph::run_rtsp lifecycle: server did not enter running state");

             check_active_client_shutdown(h);
             h.stop();
             require(!h.running(), "Graph::run_rtsp lifecycle: handle should be stopped");
           }

           // Port conflict path: second server on same port must not become running.
           {
             const int conflict_port = sima_test::allocate_local_rtsp_port();
             Graph s1 = make_rtsp_graph();
             Graph s2 = make_rtsp_graph();

             RtspServerOptions o1;
             o1.mount = "conflict_a";
             o1.port = conflict_port;

             RtspServerOptions o2;
             o2.mount = "conflict_b";
             o2.port = conflict_port;

             auto h1 = s1.run_rtsp(o1);
             require(h1.running(), "the listening server must be ready on return");
             require_neat_error([&] { (void)s2.run_rtsp(o2); }, error_codes::kIoOpen,
                                std::to_string(conflict_port));
             o2.port = 65536;
             require_neat_error([&] { (void)s2.run_rtsp(o2); }, error_codes::kOptionOutOfRange,
                                "port");
             o2.port = 0;
             auto ephemeral = s2.run_rtsp(o2);
             require(ephemeral.running() && ephemeral.url().find(":0/") == std::string::npos,
                     "an ephemeral RTSP server must publish the bound port");
             auto third = s2.run_rtsp(o2);
             h1.stop();
             auto rebound = s1.run_rtsp(o1);
             require(rebound.running(), "an idle worker must release its own listener on stop");
             third.stop();
             ephemeral.stop();
           }
         }));
