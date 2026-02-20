#include "nodes/io/StillImageInput.h"
#include "pipeline/Session.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

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

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

} // namespace

RUN_TEST(
    "session_rtsp_lifecycle_regression_test", ([] {
      using namespace simaai::neat;
      namespace fs = std::filesystem;

      // Default handle lifecycle should be safe and idempotent.
      RtspServerHandle handle;
      require(!handle.running(), "default RTSP handle should not be running");
      handle.stop();
      handle.stop();

      Session session;
      session.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
      session.custom("fakesink sync=false");

      try {
        RtspServerOptions opt;
        opt.mount = "stream";
        opt.port = sima_test::allocate_local_rtsp_port();
        (void)session.run_rtsp(opt);
        throw std::runtime_error(
            "Session::run_rtsp unexpectedly succeeded without StillImageInput");
      } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (runtime_missing_rtsp_elements(msg)) {
          throw std::runtime_error(std::string("RTSP runtime elements missing: ") + msg);
        }
        require_contains(msg, "missing StillImageInput node",
                         "Session::run_rtsp should reject non-StillImageInput pipelines");
      }

      const fs::path image = find_repo_root() / "test.jpg";
      if (!fs::exists(image)) {
        throw std::runtime_error("session_rtsp_lifecycle_regression_test: missing test.jpg");
      }

      auto make_rtsp_session = [&]() {
        Session s;
        s.add(nodes::StillImageInput(image.string(), 640, 480, 640, 480, 10));
        return s;
      };

      // Start/stop lifecycle with explicit idempotent stop check.
      {
        Session rtsp_session = make_rtsp_session();
        RtspServerOptions opt;
        opt.mount = "lifecycle";
        opt.port = sima_test::allocate_local_rtsp_port();

        RtspServerHandle h;
        try {
          h = rtsp_session.run_rtsp(opt);
        } catch (const std::exception& e) {
          if (runtime_missing_rtsp_elements(e.what())) {
            throw std::runtime_error(std::string("RTSP runtime unavailable: ") + e.what());
          }
          throw;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        require(h.running(), "Session::run_rtsp lifecycle: server did not enter running state");

        h.stop();
        h.stop();
        require(!h.running(), "Session::run_rtsp lifecycle: handle should be stopped");
      }

      // Port conflict path: second server on same port must not become running.
      {
        const int conflict_port = sima_test::allocate_local_rtsp_port();
        Session s1 = make_rtsp_session();
        Session s2 = make_rtsp_session();

        RtspServerOptions o1;
        o1.mount = "conflict_a";
        o1.port = conflict_port;

        RtspServerOptions o2;
        o2.mount = "conflict_b";
        o2.port = conflict_port;

        RtspServerHandle h1;
        RtspServerHandle h2;

        try {
          h1 = s1.run_rtsp(o1);
          h2 = s2.run_rtsp(o2);
        } catch (const std::exception& e) {
          if (runtime_missing_rtsp_elements(e.what())) {
            throw std::runtime_error(std::string("RTSP runtime unavailable: ") + e.what());
          }
          throw;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(450));
        if (!h1.running()) {
          h1.stop();
          h2.stop();
          throw std::runtime_error(
              "RTSP server failed to start on primary port; cannot validate conflict path");
        }
        require(!h2.running(),
                "Session::run_rtsp conflict: second server on occupied port should not be running");

        h2.stop();
        h1.stop();
      }
    }));
