#include "model/Model.h"
#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/runtime.h"
#include "nodes/groups/ImageInputGroup.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"
#include "test_main.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

RUN_TEST(
    "unit_header_neat_compile_surface_test", ([] {
      static_assert(std::is_constructible_v<simaai::neat::Graph>);
      static_assert(std::is_constructible_v<simaai::neat::Graph, simaai::neat::GraphOptions>);
      static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().validate()),
                                   simaai::neat::GraphReport>);
      static_assert(std::is_base_of_v<std::runtime_error, simaai::neat::NeatError>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()),
                                   simaai::neat::Graph>);
      static_assert(
          std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().preprocess()),
                         simaai::neat::Graph>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().inference()),
                                   simaai::neat::Graph>);
      static_assert(
          std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().postprocess()),
                         simaai::neat::Graph>);
      static_assert(std::is_same_v<decltype(std::declval<const simaai::neat::Model&>().graph()),
                                   simaai::neat::Graph>);
      simaai::neat::Graph graph_pipeline;
      simaai::neat::GraphOptions graph_opt;
      simaai::neat::GraphReport graph_report;
      simaai::neat::RunOptions run_opt;
      simaai::neat::Sample sample;
      simaai::neat::Model::Options model_opt;
      simaai::neat::Model::RouteOptions model_route_opt;
      simaai::neat::VerboseOptions quiet = simaai::neat::VerboseOptions::quiet();
      simaai::neat::VerboseOptions production = simaai::neat::VerboseOptions::production();
      simaai::neat::VerboseOptions debug_plugins = simaai::neat::VerboseOptions::debug_plugins();
      simaai::neat::VerboseOptions debug_all = simaai::neat::VerboseOptions::debug_all();
      simaai::neat::stages::BoxDecodeOptions decode_opt(simaai::neat::BoxDecodeType::YoloV8);
      auto in = simaai::neat::nodes::Input();
      auto out = simaai::neat::nodes::Output();
      simaai::neat::HttpSourceOptions http_opt;
      http_opt.location = "http://example.local/mjpeg";
      auto http = simaai::neat::nodes::HttpSource(http_opt);
      auto encoded_caps_fixup = simaai::neat::nodes::EncodedCapsFixup({"image/jpeg", 30});
      auto multipart_demux = simaai::neat::nodes::MultipartJpegDemux();
      auto jpeg_parse = simaai::neat::nodes::JpegParse();
      auto rtp_jpeg_depay = simaai::neat::nodes::RTPJpegDepacketize();
      simaai::neat::SimaDecodeOptions sima_decode_opt;
      sima_decode_opt.type = simaai::neat::SimaDecodeType::JPEG;
      auto sima_decode = simaai::neat::nodes::SimaDecode(sima_decode_opt);
      simaai::neat::nodes::groups::RtspEncodedInputOptions rtsp_encoded_opt;
      rtsp_encoded_opt.url = "rtsp://example.local/mjpeg";
      rtsp_encoded_opt.codec = simaai::neat::nodes::groups::RtspCodec::MJPEG;
      rtsp_encoded_opt.source_fps = 120;
      auto rtsp_encoded_group = simaai::neat::nodes::groups::RtspEncodedInput(rtsp_encoded_opt);
      auto rtsp_encoded_spec =
          simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(rtsp_encoded_opt);
      simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_decoded_opt;
      rtsp_decoded_opt.url = "rtsp://example.local/h264";
      rtsp_decoded_opt.codec = simaai::neat::nodes::groups::RtspCodec::H264;
      rtsp_decoded_opt.source_fps = 30;
      rtsp_decoded_opt.use_videorate = true;
      rtsp_decoded_opt.video_rate_fps = 15;
      auto rtsp_decoded_group = simaai::neat::nodes::groups::RtspDecodedInput(rtsp_decoded_opt);
      simaai::neat::nodes::groups::RtspDecodedInputOptions legacy_rtsp_decoded_opt{
          "rtsp://example.local/h264",
          200,
          true,
          96,
      };
      simaai::neat::nodes::groups::RtspDecodedInputOptions legacy_rtsp_videorate_opt{
          "rtsp://example.local/h264",
          200,
          true,
          96,
          -1,
          30,
          1920,
          1080,
          true,
          false,
          true,
          -1,
          -1,
          -1,
          2,
          simaai::neat::FormatTag::NV12,
          "decoder",
          true,
          "CVU",
          false,
          false,
          {},
          "",
          simaai::neat::nodes::groups::RtspCodec::H264,
          false,
          "",
          26,
          -1,
          -1,
          -1,
          -1,
          30,
          true,
      };
      require(legacy_rtsp_videorate_opt.use_videorate,
              "RtspDecodedInputOptions aggregate field order changed before use_videorate");
      require(legacy_rtsp_videorate_opt.decoder_input_buffers == -1,
              "RtspDecodedInputOptions new decoder fields must remain appended");
      simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions http_mjpeg_opt;
      http_mjpeg_opt.url = "http://example.local/mjpeg";
      http_mjpeg_opt.source_fps = 25;
      http_mjpeg_opt.use_videorate = true;
      http_mjpeg_opt.video_rate_fps = 10;
      auto http_mjpeg_group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(http_mjpeg_opt);
      simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions legacy_http_mjpeg_opt{
          "http://example.local/mjpeg", 15, 3, true, true, "Neat", "frame"};
      require(legacy_http_mjpeg_opt.multipart_boundary == "frame",
              "HttpMjpegDecodedInputOptions aggregate field order changed");
      require(legacy_http_mjpeg_opt.ssl_strict,
              "HttpMjpegDecodedInputOptions ssl_strict should default to true");
      simaai::neat::nodes::groups::ImageInputGroupOptions image_opt;
      image_opt.path = "test.jpg";
      auto group = simaai::neat::nodes::groups::ImageInputGroup(image_opt);

      (void)graph_pipeline;
      (void)graph_opt;
      (void)graph_report;
      (void)run_opt;
      (void)sample;
      (void)model_opt;
      (void)model_route_opt;
      (void)quiet;
      (void)production;
      (void)debug_plugins;
      (void)debug_all;
      (void)decode_opt;
      (void)in;
      (void)out;
      (void)http;
      (void)encoded_caps_fixup;
      (void)multipart_demux;
      (void)jpeg_parse;
      (void)rtp_jpeg_depay;
      (void)sima_decode;
      (void)rtsp_encoded_group;
      (void)rtsp_encoded_spec;
      (void)rtsp_decoded_group;
      (void)legacy_rtsp_decoded_opt;
      (void)http_mjpeg_group;
      (void)legacy_http_mjpeg_opt;
      (void)group;
    }));
