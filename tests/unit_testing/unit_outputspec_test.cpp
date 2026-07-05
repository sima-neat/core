#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "nodes/common/Caps.h"
#include "nodes/io/Input.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

namespace simaai::neat::graph {
simaai::neat::InputOptions input_opts_from_spec(const simaai::neat::OutputSpec& spec,
                                                bool complete);
}

int main() {
  try {
    simaai::neat::OutputSpec nv12;
    nv12.media_type = "video/x-raw";
    nv12.format = "NV12";
    nv12.width = 4;
    nv12.height = 2;
    const std::size_t nv12_bytes = simaai::neat::expected_byte_size(nv12);
    require(nv12_bytes == 12, "NV12 byte size mismatch");

    simaai::neat::InputOptions appsrc_opt;
    appsrc_opt.payload_type = simaai::neat::PayloadType::Image;
    appsrc_opt.format = simaai::neat::FormatTag::RGB;
    appsrc_opt.width = 10;
    appsrc_opt.height = 8;
    appsrc_opt.fps_n = 24;
    appsrc_opt.fps_d = 1;
    appsrc_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

    auto appsrc = simaai::neat::nodes::Input(appsrc_opt);
    const auto* appsrc_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(appsrc.get());
    require(appsrc_provider != nullptr, "Input should provide output spec");
    const simaai::neat::OutputSpec input_spec = appsrc_provider->output_spec({});
    require(input_spec.fps_num == 24 && input_spec.fps_den == 1,
            "Input output_spec should preserve fps options");
    require(input_spec.memory == "SystemMemory", "SystemMemory policy should report SystemMemory");

    simaai::neat::InputOptions ev74_opt = appsrc_opt;
    ev74_opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
    auto ev74_input = simaai::neat::nodes::Input(ev74_opt);
    const auto* ev74_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(ev74_input.get());
    require(ev74_provider != nullptr, "EV74 Input should provide output spec");
    require(ev74_provider->output_spec({}).memory == "SimaAI",
            "Ev74 memory policy should report SimaAI memory");

    simaai::neat::InputOptions system_opt = appsrc_opt;
    system_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
    auto system_input = simaai::neat::nodes::Input(system_opt);
    const auto* system_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(system_input.get());
    require(system_provider != nullptr, "SystemMemory Input should provide output spec");
    require(system_provider->output_spec({}).memory == "SystemMemory",
            "SystemMemory policy should report SystemMemory");

    simaai::neat::InputOptions legacy_pool_opt = appsrc_opt;
    legacy_pool_opt.memory_policy = simaai::neat::InputMemoryPolicy::Auto;
    legacy_pool_opt.use_simaai_pool = false;
    auto legacy_pool_input = simaai::neat::nodes::Input(legacy_pool_opt);
    const auto* legacy_pool_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(legacy_pool_input.get());
    require(legacy_pool_provider != nullptr, "legacy pool Input should provide output spec");
    require(legacy_pool_provider->output_spec({}).memory == "SystemMemory",
            "deprecated use_simaai_pool=false should map Auto to SystemMemory");

    require(simaai::neat::payload_type_from_media_type("video/x-h265") ==
                simaai::neat::PayloadType::Encoded,
            "H265 media type should map to Encoded payload_type");
    require(simaai::neat::payload_type_from_media_type("image/jpeg") ==
                simaai::neat::PayloadType::Encoded,
            "JPEG media type should map to Encoded payload_type");

    simaai::neat::InputOptions h264_caps_opt;
    h264_caps_opt.payload_type = simaai::neat::PayloadType::Auto;
    h264_caps_opt.caps_override =
        "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
        "parsed=(boolean)true";
    auto h264_caps_input = simaai::neat::nodes::Input(h264_caps_opt);
    const auto* h264_caps_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(h264_caps_input.get());
    require(h264_caps_provider != nullptr, "H264 caps Input should provide output spec");
    const simaai::neat::OutputSpec h264_spec = h264_caps_provider->output_spec({});
    require(h264_spec.payload_type == simaai::neat::PayloadType::Encoded,
            "encoded caps_override should infer Encoded payload_type");
    require(h264_spec.media_type == "video/x-h264",
            "encoded caps_override should infer H264 media_type");

    simaai::neat::InputOptions jpeg_caps_opt;
    jpeg_caps_opt.payload_type = simaai::neat::PayloadType::Auto;
    jpeg_caps_opt.caps_override = "image/jpeg,width=(int)16,height=(int)16";
    auto jpeg_caps_input = simaai::neat::nodes::Input(jpeg_caps_opt);
    const auto* jpeg_caps_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(jpeg_caps_input.get());
    require(jpeg_caps_provider != nullptr, "JPEG caps Input should provide output spec");
    const simaai::neat::OutputSpec jpeg_spec = jpeg_caps_provider->output_spec({});
    require(jpeg_spec.payload_type == simaai::neat::PayloadType::Encoded,
            "JPEG caps_override should infer Encoded payload_type");
    require(jpeg_spec.media_type == "image/jpeg",
            "JPEG caps_override should infer image/jpeg media_type");

    simaai::neat::InputOptions raw_feature_caps_opt;
    raw_feature_caps_opt.payload_type = simaai::neat::PayloadType::Auto;
    raw_feature_caps_opt.caps_override =
        "video/x-raw(memory:SystemMemory),format=(string)NV12,width=(int)16,height=(int)8";
    auto raw_feature_caps_input = simaai::neat::nodes::Input(raw_feature_caps_opt);
    const auto* raw_feature_caps_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(raw_feature_caps_input.get());
    require(raw_feature_caps_provider != nullptr,
            "raw feature caps Input should provide output spec");
    const simaai::neat::OutputSpec raw_feature_spec = raw_feature_caps_provider->output_spec({});
    require(raw_feature_spec.payload_type == simaai::neat::PayloadType::Image,
            "raw caps features should infer Image payload_type");
    require(raw_feature_spec.media_type == "video/x-raw",
            "raw caps features should infer canonical video/x-raw media_type");

    simaai::neat::InputOptions tensor_feature_caps_opt;
    tensor_feature_caps_opt.payload_type = simaai::neat::PayloadType::Auto;
    tensor_feature_caps_opt.caps_override =
        "application/vnd.simaai.tensor(memory:SystemMemory),rank=(int)3,dim0=(int)8,"
        "dim1=(int)16,dim2=(int)3";
    auto tensor_feature_caps_input = simaai::neat::nodes::Input(tensor_feature_caps_opt);
    const auto* tensor_feature_caps_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(tensor_feature_caps_input.get());
    require(tensor_feature_caps_provider != nullptr,
            "tensor feature caps Input should provide output spec");
    const simaai::neat::OutputSpec tensor_feature_spec =
        tensor_feature_caps_provider->output_spec({});
    require(tensor_feature_spec.payload_type == simaai::neat::PayloadType::Tensor,
            "tensor caps features should infer Tensor payload_type");
    require(tensor_feature_spec.media_type == "application/vnd.simaai.tensor",
            "tensor caps features should infer canonical tensor media_type");

    auto caps =
        simaai::neat::nodes::CapsRaw("GRAY8", 10, 8, 30, simaai::neat::CapsMemory::SystemMemory);
    std::vector<std::shared_ptr<simaai::neat::Node>> nodes{appsrc, caps};

    simaai::neat::OutputSpec derived = simaai::neat::derive_output_spec(nodes);
    std::cerr << "[DBG] derived media=" << derived.media_type << " format=" << derived.format
              << " w=" << derived.width << " h=" << derived.height << " d=" << derived.depth
              << " layout=" << derived.layout << " dtype=" << derived.dtype
              << " mem=" << derived.memory << " bytes=" << derived.byte_size << "\n";
    require(derived.media_type == "video/x-raw", "derived media_type mismatch");
    require(derived.format == "GRAY8", "derived format mismatch");
    require(derived.width == 10 && derived.height == 8, "derived shape mismatch");
    require(derived.layout == "HW", "derived layout mismatch");
    require(derived.byte_size == 80, "derived byte_size mismatch");
    require(derived.fps_num == 30 && derived.fps_den == 1, "derived fps mismatch");

    simaai::neat::InputOptions boundary_opt =
        simaai::neat::graph::input_opts_from_spec(derived, true);
    require(boundary_opt.fps_n == 30 && boundary_opt.fps_d == 1,
            "boundary input options should preserve derived fps");

    simaai::neat::nodes::groups::ImageInputGroupOptions img_opt;
    img_opt.output_caps.enable = true;
    img_opt.output_caps.format = simaai::neat::FormatTag::NV12;
    img_opt.output_caps.width = 32;
    img_opt.output_caps.height = 16;
    img_opt.output_caps.fps = 15;
    simaai::neat::OutputSpec img_spec =
        simaai::neat::nodes::groups::ImageInputGroupOutputSpec(img_opt);
    require(img_spec.format == "NV12", "image group spec format mismatch");
    require(img_spec.width == 32 && img_spec.height == 16, "image group spec shape mismatch");
    require(img_spec.byte_size == 32 * 16 * 3 / 2, "image group spec byte size mismatch");

    simaai::neat::nodes::groups::RtspEncodedInputOptions rtsp_encoded;
    rtsp_encoded.codec = simaai::neat::nodes::groups::RtspCodec::MJPEG;
    rtsp_encoded.source_fps = 120;
    simaai::neat::OutputSpec rtsp_encoded_spec =
        simaai::neat::nodes::groups::RtspEncodedInputOutputSpec(rtsp_encoded);
    require(rtsp_encoded_spec.fps_num == 120,
            "RTSP encoded source_fps should advertise encoded FPS");

    simaai::neat::nodes::groups::RtspDecodedInputOptions rtsp_decoded;
    rtsp_decoded.codec = simaai::neat::nodes::groups::RtspCodec::MJPEG;
    rtsp_decoded.dec_width = 1280;
    rtsp_decoded.dec_height = 720;
    rtsp_decoded.source_fps = 120;
    simaai::neat::OutputSpec rtsp_decoded_spec =
        simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(rtsp_decoded);
    require(rtsp_decoded_spec.fps_num == 120,
            "RTSP decoded source_fps should advertise decoder FPS");

    rtsp_decoded.use_videorate = true;
    rtsp_decoded.video_rate_fps = 30;
    simaai::neat::OutputSpec rtsp_rate_spec =
        simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(rtsp_decoded);
    require(rtsp_rate_spec.fps_num == 30,
            "RTSP decoded video_rate_fps should advertise rate-limited FPS");
    require(rtsp_rate_spec.width == 1280 && rtsp_rate_spec.height == 720,
            "RTSP decoded videorate-only spec should preserve decoder shape");
    require(rtsp_rate_spec.format == "NV12" && rtsp_rate_spec.memory == "SimaAI",
            "RTSP decoded videorate-only spec should preserve decoder format and memory");
    require(rtsp_rate_spec.layout == "Planar" && rtsp_rate_spec.byte_size == 1280 * 720 * 3 / 2,
            "RTSP decoded videorate-only spec should preserve decoder layout and byte size");

    auto rtsp_tail_caps = rtsp_decoded;
    rtsp_tail_caps.output_caps.enable = true;
    rtsp_tail_caps.output_caps.width = 640;
    rtsp_tail_caps.output_caps.height = 360;
    rtsp_tail_caps.output_caps.fps = 30;
    simaai::neat::OutputSpec rtsp_tail_spec =
        simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(rtsp_tail_caps);
    require(rtsp_tail_spec.width == 640 && rtsp_tail_spec.height == 360,
            "RTSP decoded output_caps should apply explicit shape");
    require(rtsp_tail_spec.memory == "SimaAI",
            "RTSP decoded output_caps should preserve decoder memory by default");
    rtsp_tail_caps.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;
    rtsp_tail_spec = simaai::neat::nodes::groups::RtspDecodedInputOutputSpec(rtsp_tail_caps);
    require(rtsp_tail_spec.memory == "SystemMemory",
            "RTSP decoded output_caps should force system memory when requested");

    simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions http_mjpeg;
    http_mjpeg.dec_width = 1280;
    http_mjpeg.dec_height = 720;
    http_mjpeg.source_fps = 25;
    simaai::neat::OutputSpec http_mjpeg_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(http_mjpeg);
    require(http_mjpeg_spec.fps_num == 25, "HTTP MJPEG source_fps should advertise decoder FPS");

    http_mjpeg.use_videorate = true;
    http_mjpeg.video_rate_fps = 10;
    simaai::neat::OutputSpec http_rate_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(http_mjpeg);
    require(http_rate_spec.fps_num == 10,
            "HTTP MJPEG video_rate_fps should advertise rate-limited FPS");
    require(http_rate_spec.width == 1280 && http_rate_spec.height == 720,
            "HTTP MJPEG videorate-only spec should preserve decoder shape");
    require(http_rate_spec.format == "NV12" && http_rate_spec.memory == "SimaAI",
            "HTTP MJPEG videorate-only spec should preserve decoder format and memory");
    require(http_rate_spec.layout == "Planar" && http_rate_spec.byte_size == 1280 * 720 * 3 / 2,
            "HTTP MJPEG videorate-only spec should preserve decoder layout and byte size");

    auto http_tail_caps = http_mjpeg;
    http_tail_caps.output_caps.enable = true;
    http_tail_caps.output_caps.width = 640;
    http_tail_caps.output_caps.height = 360;
    http_tail_caps.output_caps.fps = 10;
    simaai::neat::OutputSpec http_tail_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(http_tail_caps);
    require(http_tail_spec.width == 640 && http_tail_spec.height == 360,
            "HTTP MJPEG output_caps should apply explicit shape");
    require(http_tail_spec.memory == "SimaAI",
            "HTTP MJPEG output_caps should preserve decoder memory by default");
    http_tail_caps.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;
    http_tail_spec = simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(http_tail_caps);
    require(http_tail_spec.memory == "SystemMemory",
            "HTTP MJPEG output_caps should force system memory when requested");

    std::cout << "[OK] unit_outputspec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
