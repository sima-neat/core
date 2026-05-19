#include "pipeline/Session.h"
#include "pipeline/EncodedSampleUtil.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264Parse.h"

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
    if (!simaai::neat::element_exists("neatdecoder")) {
      return fail_test("neatdecoder element missing");
    }
    if (!simaai::neat::element_exists("appsrc") || !simaai::neat::element_exists("appsink") ||
        !simaai::neat::element_exists("identity")) {
      return fail_test("required appsrc/appsink/identity elements missing");
    }
    if (!simaai::neat::element_exists("h264parse")) {
      return fail_test("required h264parse element missing");
    }

    const int width = 640;
    const int height = 360;
    const int fps = 30;
    const std::string caps = "video/x-h264,parsed=true,stream-format=(string)byte-stream,"
                             "alignment=(string)au,width=640,height=360,framerate=30/1";

    simaai::neat::InputOptions src_opt;
    src_opt.media_type = "video/x-h264";
    src_opt.format = simaai::neat::FormatTag::H264;
    src_opt.caps_override = caps;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = true;
    src_opt.stream_type = 0;
    src_opt.use_simaai_pool = false;

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::Input(src_opt));
    p.add(simaai::neat::nodes::H264ParseAu(/*config_interval=*/1));
    p.add(simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));
    p.add(simaai::neat::nodes::Output());

    simaai::neat::Sample sample =
        simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x00, 0x00, 0x00, 0x01}, caps);

    simaai::neat::RunOptions opt;
    opt.queue_depth = 1;

    simaai::neat::Run run =
        p.build(simaai::neat::SampleList{sample}, simaai::neat::RunMode::Async, opt);
    require(run.running(), "Pipeline did not enter running state");

    int sleep_ms = 35000;
    if (const char* env = std::getenv("SIMA_NEATDECODER_KEEPALIVE_SLEEP_MS")) {
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
    std::cout << "[OK] unit_tensordecoder_keepalive_test passed\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
