#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/io/CameraInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "gst/GstHelpers.h"

#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static const char* decoder_element_name() {
  return "neatdecoder";
}

int main() {
  try {
    auto caps = simaai::neat::nodes::CapsNV12SysMem(128, 64, 30);
    require_contains(caps->backend_fragment(3), "capsfilter name=n3_caps",
                     "Caps fragment missing name");

    auto fs = simaai::neat::nodes::FileInput("path.jpg");
    require_contains(fs->backend_fragment(1), "filesrc name=n1_filesrc", "FileInput name mismatch");

    auto jd = simaai::neat::nodes::JpegDecode();
    require_contains(jd->backend_fragment(2), "jpegdec name=n2_jpegdec",
                     "JpegDecode name mismatch");

    auto q = simaai::neat::nodes::Queue();
    require_contains(q->backend_fragment(5), "queue name=n5_queue", "Queue name mismatch");

    auto vc = simaai::neat::nodes::VideoConvert();
    require_contains(vc->backend_fragment(6), "videoconvert name=n6_videoconvert",
                     "VideoConvert name mismatch");

    auto vs = simaai::neat::nodes::VideoScale();
    require_contains(vs->backend_fragment(7), "videoscale name=n7_videoscale",
                     "VideoScale name mismatch");

    auto vr = simaai::neat::nodes::VideoRate();
    require_contains(vr->backend_fragment(8), "videorate name=n8_videorate",
                     "VideoRate name mismatch");

    auto demux = simaai::neat::nodes::VideoTrackSelect(0);
    require_contains(demux->backend_fragment(4), "qtdemux name=n4_demux", "QtDemux name mismatch");

    auto rtsp = simaai::neat::nodes::RTSPInput("rtsp://example", 200, true);
    require_contains(rtsp->backend_fragment(1), "rtspsrc name=n1_rtspsrc",
                     "RTSPInput name mismatch");

    simaai::neat::CameraInputOptions cam_opt;
    require(cam_opt.allow_cpu_fallback,
            "CameraInput must default to the adaptive bridge on supported package sets");
    auto cam = simaai::neat::nodes::CameraInput(cam_opt);
    require_contains(cam->backend_fragment(0), "libcamerasrc name=n0_camera_src",
                     "CameraInput source missing");
    require_contains(cam->backend_fragment(0), "neatcamerabridge name=n0_camera_bridge",
                     "CameraInput fallback bridge missing");
    require(cam->backend_fragment(0).find(" buffer-size=") == std::string::npos,
            "CameraInput bridge should derive buffer size from actual camera buffers");
    if (!simaai::neat::element_exists("neatcamerabridge")) {
      throw std::runtime_error("private neatcamerabridge factory not registered");
    }

    auto depay = simaai::neat::nodes::H264Depacketize(97);
    require_contains(depay->backend_fragment(2), "rtph264depay name=n2_depay",
                     "Depay fragment mismatch");
    require_contains(depay->backend_fragment(2), "payload=97", "Depay payload mismatch");
    require(depay->backend_fragment(2).find("width=(int)[") == std::string::npos,
            "Depay without explicit caps should not emit open-ended width ranges");
    auto depay_partial = simaai::neat::nodes::H264Depacketize(
        97, /*h264_parse_config_interval=*/1, /*h264_fps=*/30,
        /*h264_width=*/1280, /*h264_height=*/-1, /*enforce_h264_caps=*/true);
    bool depay_partial_threw = false;
    try {
      (void)depay_partial->backend_fragment(2);
    } catch (const std::exception& e) {
      depay_partial_threw =
          std::string(e.what()).find("require width, height, and fps") != std::string::npos;
    }
    require(depay_partial_threw, "Partial H264 caps should throw an actionable error");

    auto dec = simaai::neat::nodes::H264Decode(2, "NV12");
    const std::string dec_expect = std::string(decoder_element_name()) + " name=n1_decoder";
    require_contains(dec->backend_fragment(1), dec_expect, "Decode fragment mismatch");

    auto enc = simaai::neat::nodes::H264EncodeSima(64, 64, 30, 400, "baseline", "4.0");
    require_contains(enc->backend_fragment(0), "neatencoder name=n0_encoder",
                     "Encode fragment mismatch");

    simaai::neat::H264ParseOptions opt;
    opt.enforce_caps = true;
    opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
    auto parse = simaai::neat::nodes::H264Parse(opt);
    require_contains(parse->backend_fragment(3), "h264parse name=n3_h264parse",
                     "Parse fragment mismatch");
    require_contains(parse->backend_fragment(3), "capsfilter name=n3_h264_caps",
                     "Parse capsfilter missing");

    auto pay = simaai::neat::nodes::H264Packetize(96, 1);
    require_contains(pay->backend_fragment(9), "rtph264pay name=pay0", "Pay fragment mismatch");

    auto sink = simaai::neat::nodes::Output();
    require_contains(sink->backend_fragment(0), "appsink name=mysink", "Output fragment mismatch");

    auto idec = simaai::neat::nodes::ImageDecode();
    require_contains(idec->backend_fragment(2), "decodebin name=n2_decodebin",
                     "ImageDecode fragment mismatch");

    bool has_sw = simaai::neat::element_exists("x264enc") ||
                  simaai::neat::element_exists("openh264enc") ||
                  simaai::neat::element_exists("avenc_h264");
    if (has_sw) {
      auto sw = simaai::neat::nodes::H264EncodeSW(400);
      require_contains(sw->backend_fragment(1), "name=n1_swenc", "H264EncodeSW name mismatch");
    }

    std::cout << "[OK] unit_nodes_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
