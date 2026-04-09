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
#include "nodes/io/V4L2Input.h"
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

    if (simaai::neat::element_exists("v4l2src")) {
      simaai::neat::V4L2InputOptions mipi_opt;
      mipi_opt.device = "/dev/video0out";
      mipi_opt.media_type = "video/x-raw";
      mipi_opt.format = "RGB";
      mipi_opt.width = 1920;
      mipi_opt.height = 1080;
      mipi_opt.fps_n = 30;

      auto mipi = simaai::neat::nodes::V4L2Input(mipi_opt);
      require(mipi->kind() == "V4L2Input", "V4L2Input kind mismatch");
      require(mipi->user_label() == "/dev/video0out", "V4L2Input user label mismatch");
      require(mipi->input_role() == simaai::neat::InputRole::Source,
              "V4L2Input input role mismatch");
      require(mipi->caps_behavior() == simaai::neat::NodeCapsBehavior::Dynamic,
              "V4L2Input caps behavior mismatch");

      const std::string mipi_fragment = mipi->backend_fragment(0);
      require_contains(mipi_fragment, "v4l2src name=n0_v4l2src device=/dev/video0out",
                       "V4L2Input fragment device mismatch");
      require_contains(mipi_fragment, "capsfilter name=n0_v4l2src_caps",
                       "V4L2Input capsfilter missing");
      require_contains(mipi_fragment,
                       "caps=\"video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1\"",
                       "V4L2Input RGB caps mismatch");

      const auto mipi_names = mipi->element_names(0);
      require(mipi_names.size() == 2, "V4L2Input element_names with caps size mismatch");
      require(mipi_names[0] == "n0_v4l2src", "V4L2Input primary element name mismatch");
      require(mipi_names[1] == "n0_v4l2src_caps", "V4L2Input caps element name mismatch");

      auto* mipi_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(mipi.get());
      require(mipi_provider != nullptr, "V4L2Input OutputSpecProvider missing");
      const simaai::neat::OutputSpec rgb_spec = mipi_provider->output_spec({});
      require(rgb_spec.media_type == "video/x-raw", "V4L2Input RGB media_type mismatch");
      require(rgb_spec.format == "RGB", "V4L2Input RGB format mismatch");
      require(rgb_spec.width == 1920 && rgb_spec.height == 1080,
              "V4L2Input RGB dimensions mismatch");
      require(rgb_spec.fps_num == 30 && rgb_spec.fps_den == 1, "V4L2Input RGB fps mismatch");
      require(rgb_spec.dtype == "UInt8", "V4L2Input RGB dtype mismatch");
      require(rgb_spec.layout == "HWC", "V4L2Input RGB layout mismatch");
      require(rgb_spec.depth == 3, "V4L2Input RGB depth mismatch");
      require(rgb_spec.memory == "SystemMemory", "V4L2Input RGB memory mismatch");
      require(rgb_spec.certainty == simaai::neat::SpecCertainty::Hint,
              "V4L2Input RGB certainty mismatch");

      simaai::neat::V4L2InputOptions mjpeg_opt;
      mjpeg_opt.media_type = "image/jpeg";
      mjpeg_opt.width = 1280;
      mjpeg_opt.height = 720;
      auto mjpeg = simaai::neat::nodes::V4L2Input(mjpeg_opt);
      const std::string mjpeg_fragment = mjpeg->backend_fragment(2);
      require_contains(mjpeg_fragment, "v4l2src name=n2_v4l2src device=/dev/video0",
                       "V4L2Input MJPEG default device mismatch");
      require_contains(mjpeg_fragment, "caps=\"image/jpeg,width=1280,height=720\"",
                       "V4L2Input MJPEG caps mismatch");
      require(mjpeg_fragment.find("format=") == std::string::npos,
              "V4L2Input MJPEG should omit format");
      auto* mjpeg_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(mjpeg.get());
      require(mjpeg_provider != nullptr, "V4L2Input MJPEG OutputSpecProvider missing");
      const simaai::neat::OutputSpec mjpeg_spec = mjpeg_provider->output_spec({});
      require(mjpeg_spec.media_type == "image/jpeg", "V4L2Input MJPEG media_type mismatch");
      require(mjpeg_spec.width == 1280 && mjpeg_spec.height == 720,
              "V4L2Input MJPEG dimensions mismatch");
      require(mjpeg_spec.depth == -1, "V4L2Input MJPEG depth mismatch");

      simaai::neat::V4L2InputOptions bayer_opt;
      bayer_opt.media_type = "video/x-bayer";
      bayer_opt.format = "rggb12le";
      bayer_opt.width = 640;
      bayer_opt.height = 480;
      auto bayer = simaai::neat::nodes::V4L2Input(bayer_opt);
      const std::string bayer_fragment = bayer->backend_fragment(3);
      require_contains(bayer_fragment,
                       "caps=\"video/x-bayer,format=rggb12le,width=640,height=480\"",
                       "V4L2Input bayer caps mismatch");

      simaai::neat::V4L2InputOptions io_mode_opt = mipi_opt;
      io_mode_opt.io_mode = "dmabuf";
      io_mode_opt.num_buffers = 4;
      auto io_mode = simaai::neat::nodes::V4L2Input(io_mode_opt);
      const std::string io_mode_fragment = io_mode->backend_fragment(4);
      require_contains(io_mode_fragment, "io-mode=dmabuf", "V4L2Input io-mode mismatch");
      require_contains(io_mode_fragment, "num-buffers=4", "V4L2Input num-buffers mismatch");

      simaai::neat::V4L2InputOptions no_caps_opt;
      auto no_caps = simaai::neat::nodes::V4L2Input(no_caps_opt);
      const std::string no_caps_fragment = no_caps->backend_fragment(1);
      require_contains(no_caps_fragment, "v4l2src name=n1_v4l2src device=/dev/video0",
                       "V4L2Input no-caps fragment mismatch");
      require(no_caps_fragment.find("capsfilter") == std::string::npos,
              "V4L2Input no-caps fragment should omit capsfilter");
      require(no_caps_fragment.find("num-buffers=") == std::string::npos,
              "V4L2Input default num-buffers should be omitted");

      const auto no_caps_names = no_caps->element_names(1);
      require(no_caps_names.size() == 1, "V4L2Input no-caps element_names size mismatch");
      require(no_caps_names[0] == "n1_v4l2src", "V4L2Input no-caps element name mismatch");

      auto* no_caps_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(no_caps.get());
      require(no_caps_provider != nullptr, "V4L2Input no-caps OutputSpecProvider missing");
      const simaai::neat::OutputSpec no_caps_spec = no_caps_provider->output_spec({});
      require(no_caps_spec.certainty == simaai::neat::SpecCertainty::Unknown,
              "V4L2Input no-caps certainty mismatch");

      simaai::neat::V4L2InputOptions nv12_opt;
      nv12_opt.media_type = "video/x-raw";
      nv12_opt.format = "NV12";
      nv12_opt.width = 1280;
      nv12_opt.height = 720;
      auto nv12 = simaai::neat::nodes::V4L2Input(nv12_opt);
      auto* nv12_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(nv12.get());
      require(nv12_provider != nullptr, "V4L2Input NV12 OutputSpecProvider missing");
      const simaai::neat::OutputSpec nv12_spec = nv12_provider->output_spec({});
      require(nv12_spec.layout == "Planar", "V4L2Input NV12 layout mismatch");
      require(nv12_spec.depth == 3, "V4L2Input NV12 depth mismatch");

      // Bayer output_spec: rggb12le -> UInt16, layout HW, depth 1
      auto* bayer_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(bayer.get());
      require(bayer_provider != nullptr, "V4L2Input bayer OutputSpecProvider missing");
      const simaai::neat::OutputSpec bayer_spec = bayer_provider->output_spec({});
      require(bayer_spec.media_type == "video/x-bayer", "V4L2Input bayer media_type mismatch");
      require(bayer_spec.layout == "HW", "V4L2Input bayer layout mismatch");
      require(bayer_spec.depth == 1, "V4L2Input bayer depth mismatch");
      require(bayer_spec.dtype == "UInt16", "V4L2Input bayer 12-bit dtype mismatch");

      // BGR output_spec
      simaai::neat::V4L2InputOptions bgr_opt;
      bgr_opt.media_type = "video/x-raw";
      bgr_opt.format = "BGR";
      bgr_opt.width = 640;
      bgr_opt.height = 480;
      auto bgr = simaai::neat::nodes::V4L2Input(bgr_opt);
      auto* bgr_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(bgr.get());
      require(bgr_provider != nullptr, "V4L2Input BGR OutputSpecProvider missing");
      const simaai::neat::OutputSpec bgr_spec = bgr_provider->output_spec({});
      require(bgr_spec.dtype == "UInt8", "V4L2Input BGR dtype mismatch");
      require(bgr_spec.layout == "HWC", "V4L2Input BGR layout mismatch");
      require(bgr_spec.depth == 3, "V4L2Input BGR depth mismatch");

      // GRAY output_spec
      simaai::neat::V4L2InputOptions gray_opt;
      gray_opt.media_type = "video/x-raw";
      gray_opt.format = "GRAY8";
      gray_opt.width = 320;
      gray_opt.height = 240;
      auto gray = simaai::neat::nodes::V4L2Input(gray_opt);
      auto* gray_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(gray.get());
      require(gray_provider != nullptr, "V4L2Input GRAY OutputSpecProvider missing");
      const simaai::neat::OutputSpec gray_spec = gray_provider->output_spec({});
      require(gray_spec.dtype == "UInt8", "V4L2Input GRAY dtype mismatch");
      require(gray_spec.layout == "HW", "V4L2Input GRAY layout mismatch");
      require(gray_spec.depth == 1, "V4L2Input GRAY depth mismatch");

      // I420 output_spec
      simaai::neat::V4L2InputOptions i420_opt;
      i420_opt.media_type = "video/x-raw";
      i420_opt.format = "I420";
      i420_opt.width = 640;
      i420_opt.height = 480;
      auto i420 = simaai::neat::nodes::V4L2Input(i420_opt);
      auto* i420_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(i420.get());
      require(i420_provider != nullptr, "V4L2Input I420 OutputSpecProvider missing");
      const simaai::neat::OutputSpec i420_spec = i420_provider->output_spec({});
      require(i420_spec.layout == "Planar", "V4L2Input I420 layout mismatch");
      require(i420_spec.depth == 3, "V4L2Input I420 depth mismatch");

      // Unrecognized format output_spec
      simaai::neat::V4L2InputOptions unk_opt;
      unk_opt.media_type = "video/x-raw";
      unk_opt.format = "UYVY";
      unk_opt.width = 640;
      unk_opt.height = 480;
      auto unk = simaai::neat::nodes::V4L2Input(unk_opt);
      auto* unk_provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(unk.get());
      require(unk_provider != nullptr, "V4L2Input unrecognized OutputSpecProvider missing");
      const simaai::neat::OutputSpec unk_spec = unk_provider->output_spec({});
      require_contains(unk_spec.note, "unrecognized format",
                       "V4L2Input unrecognized format note missing");

      // Partial caps: width+height set but media_type empty -> no capsfilter
      simaai::neat::V4L2InputOptions partial_opt;
      partial_opt.width = 1920;
      partial_opt.height = 1080;
      auto partial = simaai::neat::nodes::V4L2Input(partial_opt);
      const std::string partial_fragment = partial->backend_fragment(5);
      require(partial_fragment.find("capsfilter") == std::string::npos,
              "V4L2Input partial caps should not produce capsfilter");

      // Empty device should throw
      bool threw = false;
      try {
        simaai::neat::V4L2InputOptions empty_dev;
        empty_dev.device = "";
        auto bad = simaai::neat::nodes::V4L2Input(empty_dev);
        (void)bad;
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "V4L2Input empty device should throw invalid_argument");
    }

    auto depay = simaai::neat::nodes::H264Depacketize(97);
    require_contains(depay->backend_fragment(2), "rtph264depay name=n2_depay",
                     "Depay fragment mismatch");
    require_contains(depay->backend_fragment(2), "payload=97", "Depay payload mismatch");

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
