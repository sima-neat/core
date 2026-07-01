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
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/SimaDecode.h"
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

    auto depay = simaai::neat::nodes::H264Depacketize(97);
    require_contains(depay->backend_fragment(2), "rtph264depay name=n2_depay",
                     "Depay fragment mismatch");
    require_contains(depay->backend_fragment(2), "payload=97", "Depay payload mismatch");

    auto dec = simaai::neat::nodes::H264Decode(2, "NV12");
    const std::string dec_expect = std::string(decoder_element_name()) + " name=n1_decoder";
    require_contains(dec->backend_fragment(1), dec_expect, "Decode fragment mismatch");

    simaai::neat::SimaDecodeOptions native_dec_opt;
    native_dec_opt.type = simaai::neat::SimaDecodeType::H264;
    native_dec_opt.raw_output = true;
    native_dec_opt.dec_width = 1280;
    native_dec_opt.dec_height = 720;
    native_dec_opt.dec_fps = 30;
    auto native_dec = simaai::neat::nodes::SimaDecode(native_dec_opt);
    require_contains(native_dec->backend_fragment(3), "neatdecoder name=n3_decoder",
                     "SimaDecode fragment missing decoder");
    require_contains(native_dec->backend_fragment(3), "dec-type=h264",
                     "SimaDecode H264 type mismatch");
    require_contains(native_dec->backend_fragment(3), "dec-width=1280",
                     "SimaDecode width override missing");
    require_contains(native_dec->backend_fragment(3), "dec-height=720",
                     "SimaDecode height override missing");
    require_contains(native_dec->backend_fragment(3), "dec-fps=30",
                     "SimaDecode fps override missing");

    const auto* native_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(native_dec.get());
    require(native_provider != nullptr, "SimaDecode should provide output spec");
    simaai::neat::OutputSpec encoded_input;
    encoded_input.media_type = "video/x-h264";
    const simaai::neat::OutputSpec native_spec = native_provider->output_spec(encoded_input);
    require(native_spec.media_type == "video/x-raw", "SimaDecode media type mismatch");
    require(native_spec.format == "NV12", "SimaDecode output format mismatch");
    require(native_spec.width == 1280 && native_spec.height == 720,
            "SimaDecode output shape mismatch");
    require(native_spec.fps_num == 30 && native_spec.fps_den == 1,
            "SimaDecode output fps mismatch");
    require(native_spec.memory == "SimaAI", "SimaDecode raw output should report SimaAI memory");

    simaai::neat::SimaDecodeOptions jpeg_dec_opt;
    jpeg_dec_opt.type = simaai::neat::SimaDecodeType::JPEG;
    jpeg_dec_opt.raw_output = false;
    auto jpeg_native_dec = simaai::neat::nodes::SimaDecode(jpeg_dec_opt);
    require_contains(jpeg_native_dec->backend_fragment(4), "dec-type=jpeg",
                     "SimaDecode JPEG type mismatch");
    require_contains(jpeg_native_dec->backend_fragment(4), "videoconvert name=n4_videoconvert",
                     "SimaDecode non-raw output should insert videoconvert");
    const auto* jpeg_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(jpeg_native_dec.get());
    require(jpeg_provider != nullptr, "JPEG SimaDecode should provide output spec");
    require(jpeg_provider->output_spec({}).memory == "SystemMemory",
            "SimaDecode non-raw output should report SystemMemory");

    simaai::neat::SimaDecodeOptions mjpeg_dec_opt;
    mjpeg_dec_opt.type = simaai::neat::SimaDecodeType::MJPEG;
    auto mjpeg_native_dec = simaai::neat::nodes::SimaDecode(mjpeg_dec_opt);
    require_contains(mjpeg_native_dec->backend_fragment(5), "dec-type=mjpeg",
                     "SimaDecode MJPEG type mismatch");

    simaai::neat::SimaDecodeOptions i420_dec_opt;
    i420_dec_opt.out_format = simaai::neat::FormatTag::I420;
    auto i420_native_dec = simaai::neat::nodes::SimaDecode(i420_dec_opt);
    require_contains(i420_native_dec->backend_fragment(6), "dec-fmt=YUV420P",
                     "SimaDecode I420 should map to neatdecoder YUV420P");
    const auto* i420_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(i420_native_dec.get());
    require(i420_provider != nullptr, "I420 SimaDecode should provide output spec");
    require(i420_provider->output_spec({}).format == "I420",
            "SimaDecode should expose public I420 caps");

    simaai::neat::SimaDecodeOptions unsupported_raw_dec_opt;
    unsupported_raw_dec_opt.out_format = simaai::neat::FormatTag::RGB;
    auto unsupported_raw_dec = simaai::neat::nodes::SimaDecode(unsupported_raw_dec_opt);
    bool unsupported_raw_threw = false;
    try {
      (void)unsupported_raw_dec->backend_fragment(7);
    } catch (const std::invalid_argument&) {
      unsupported_raw_threw = true;
    }
    require(unsupported_raw_threw, "SimaDecode raw_output should reject non-native formats");

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
