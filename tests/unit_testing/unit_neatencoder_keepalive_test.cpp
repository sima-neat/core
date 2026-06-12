#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264EncodeSima.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main() {
  try {
    setenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0", 1);

    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatencoder")) {
      return fail_test("neatencoder element missing");
    }
    if (!simaai::neat::element_exists("appsrc") || !simaai::neat::element_exists("appsink") ||
        !simaai::neat::element_exists("identity")) {
      return fail_test("required appsrc/appsink/identity elements missing");
    }

    const int width = 640;
    const int height = 360;
    const int fps = 30;

    simaai::neat::InputOptions src_opt;
    src_opt.payload_type = simaai::neat::PayloadType::Image;
    src_opt.format = simaai::neat::FormatTag::NV12;
    src_opt.width = width;
    src_opt.height = height;
    src_opt.fps_n = fps;
    src_opt.fps_d = 1;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = true;
    src_opt.stream_type = 0;
    src_opt.use_simaai_pool = false;

    simaai::neat::Graph p;
    p.add(simaai::neat::nodes::Input(src_opt));
    p.add(simaai::neat::nodes::H264EncodeSima(width, height, fps, 400, "baseline", "4.0"));
    p.add(simaai::neat::nodes::Output());

    simaai::neat::Tensor sample = make_nv12_tensor(width, height, 0x12);

    simaai::neat::RunOptions opt;
    opt.queue_depth = 1;

    simaai::neat::Run run = p.build(simaai::neat::TensorList{sample}, opt);
    require(run.running(), "Pipeline did not enter running state");

    int sleep_ms = 35000;
    if (const char* env = std::getenv("SIMA_NEATENCODER_KEEPALIVE_SLEEP_MS")) {
      if (*env) {
        const int val = std::atoi(env);
        if (val > 0)
          sleep_ms = val;
      }
    }
    if (sleep_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    require(run.running(), "Pipeline stopped during idle keepalive sleep");
    const std::string err = run.last_error();
    require(err.empty(), "Pipeline error during idle keepalive: " + err);

    run.close();
    std::cout << "[OK] unit_tensorencoder_keepalive_test passed\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
