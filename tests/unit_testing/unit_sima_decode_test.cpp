#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/SimaDecode.h"

#include "test_utils.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (unexpected: " + needle + ")");
  }
}

void require_throws_with(const std::function<void()>& fn, const std::string& needle,
                         const std::string& label) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), needle, label);
    return;
  }
  throw std::runtime_error(label + " did not throw");
}

simaai::neat::OutputSpec encoded_h264_spec() {
  simaai::neat::OutputSpec spec;
  spec.media_type = "video/x-h264";
  spec.format = "H264";
  spec.width = 1280;
  spec.height = 720;
  spec.fps_num = 30;
  spec.fps_den = 1;
  return spec;
}

void require_element_names(const std::vector<std::string>& actual,
                           const std::vector<std::string>& expected, const std::string& label) {
  require(actual.size() == expected.size(), label + " element count mismatch");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require(actual[i] == expected[i],
            label + " element name mismatch at index " + std::to_string(i));
  }
}

void check_default_h264_raw_output() {
  simaai::neat::SimaDecode decode;
  const std::string fragment = decode.backend_fragment(3);
  require_contains(fragment, "neatdecoder name=n3_decoder", "SimaDecode should emit neatdecoder");
  require_contains(fragment, "sima-allocator-type=2", "SimaDecode should pass allocator option");
  require_contains(fragment, "dec-type=h264", "SimaDecode should default to H.264");
  require_contains(fragment, "dec-fmt=NV12", "SimaDecode should default to NV12");
  require_not_contains(fragment, "videoconvert", "raw_output=true should not insert videoconvert");
  require_not_contains(fragment, "capsfilter", "raw_output=true should not insert capsfilter");
  require_element_names(decode.element_names(3), {"n3_decoder"}, "default H.264 raw output");
  require(decode.buffer_name_hint(3) == "n3_decoder",
          "SimaDecode should expose default decoder buffer name");
  require(decode.memory_contract() == simaai::neat::MemoryContract::PreferDeviceZeroCopy,
          "raw SimaDecode should advertise device zero-copy output");

  const simaai::neat::OutputSpec out = decode.output_spec(encoded_h264_spec());
  require(out.media_type == "video/x-raw", "SimaDecode output media type mismatch");
  require(out.format == "NV12", "SimaDecode output format mismatch");
  require(out.width == 1280 && out.height == 720, "SimaDecode should preserve input shape");
  require(out.fps_num == 30 && out.fps_den == 1, "SimaDecode should preserve input fps");
  require(out.memory == "SimaAI", "raw_output=true should advertise SimaAI memory");
  require(out.layout == "Planar", "NV12 output should advertise planar layout");
  require(out.dtype == "UInt8", "SimaDecode dtype mismatch");
  require(out.byte_size == 1280 * 720 * 3 / 2, "NV12 byte size mismatch");
}

void check_jpeg_raw_output_options() {
  simaai::neat::SimaDecodeOptions opt;
  opt.type = simaai::neat::SimaDecodeType::JPEG;
  opt.sima_allocator_type = 3;
  opt.out_format = simaai::neat::FormatTag::I420;
  opt.decoder_name = "jpeg_decoder";
  opt.next_element = "CVU";
  opt.dec_width = 640;
  opt.dec_height = 480;
  opt.dec_fps = 30;
  opt.num_buffers = 8;

  simaai::neat::SimaDecode decode(opt);
  const std::string fragment = decode.backend_fragment(4);
  require_contains(fragment, "neatdecoder name=jpeg_decoder",
                   "SimaDecode should use custom decoder name");
  require_contains(fragment, "sima-allocator-type=3",
                   "SimaDecode should pass custom allocator option");
  require_contains(fragment, "dec-type=jpeg", "SimaDecode JPEG type mismatch");
  require_contains(fragment, "op-buff-name=jpeg_decoder",
                   "SimaDecode should use decoder name as output buffer name");
  require_contains(fragment, "dec-fmt=YUV420P",
                   "SimaDecode should map public I420 to decoder YUV420P");
  require_contains(fragment, "next-element=CVU", "SimaDecode next-element missing");
  require_contains(fragment, "dec-width=640", "SimaDecode width override missing");
  require_contains(fragment, "dec-height=480", "SimaDecode height override missing");
  require_contains(fragment, "dec-fps=30", "SimaDecode fps override missing");
  require_contains(fragment, "num-buffers=8", "SimaDecode pool override missing");
  require_element_names(decode.element_names(4), {"jpeg_decoder"}, "JPEG raw output");
  require(decode.buffer_name_hint(4) == "jpeg_decoder",
          "SimaDecode should expose explicit decoder buffer name");

  const simaai::neat::OutputSpec out = decode.output_spec({});
  require(out.format == "I420", "SimaDecode should expose public I420 caps");
  require(out.width == 640 && out.height == 480, "SimaDecode explicit shape mismatch");
  require(out.fps_num == 30 && out.fps_den == 1, "SimaDecode explicit fps mismatch");
  require(out.memory == "SimaAI", "JPEG raw output should advertise SimaAI memory");
  require(out.byte_size == 640 * 480 * 3 / 2, "I420 byte size mismatch");
}

void check_mjpeg_system_memory_output() {
  simaai::neat::SimaDecodeOptions opt;
  opt.type = simaai::neat::SimaDecodeType::MJPEG;
  opt.raw_output = false;
  opt.dec_width = 800;
  opt.dec_height = 600;
  opt.dec_fps = 25;

  simaai::neat::SimaDecode decode(opt);
  const std::string fragment = decode.backend_fragment(5);
  require_contains(fragment, "dec-type=mjpeg", "SimaDecode MJPEG type mismatch");
  require_contains(fragment, "videoconvert name=n5_videoconvert",
                   "raw_output=false should insert videoconvert");
  require_contains(fragment, "capsfilter name=n5_raw_caps",
                   "raw_output=false should insert raw capsfilter");
  require_contains(fragment, "video/x-raw(memory:SystemMemory),format=NV12",
                   "raw_output=false should request SystemMemory raw caps");
  require_element_names(decode.element_names(5), {"n5_decoder", "n5_videoconvert", "n5_raw_caps"},
                        "MJPEG SystemMemory output");
  require(decode.memory_contract() == simaai::neat::MemoryContract::AllowEitherButReport,
          "converted SimaDecode should not advertise device zero-copy output");

  const simaai::neat::OutputSpec out = decode.output_spec({});
  require(out.width == 800 && out.height == 600, "MJPEG explicit shape mismatch");
  require(out.fps_num == 25 && out.fps_den == 1, "MJPEG explicit fps mismatch");
  require(out.memory == "SystemMemory", "raw_output=false should advertise SystemMemory output");
}

void check_invalid_options() {
  simaai::neat::SimaDecodeOptions invalid_codec;
  invalid_codec.type = static_cast<simaai::neat::SimaDecodeType>(999);
  require_throws_with(
      [&] {
        simaai::neat::SimaDecode decode(invalid_codec);
        (void)decode.backend_fragment(1);
      },
      "unsupported decode type", "invalid SimaDecode type");

  simaai::neat::SimaDecodeOptions unsupported_raw_format;
  unsupported_raw_format.out_format = simaai::neat::FormatTag::RGB;
  require_throws_with(
      [&] {
        simaai::neat::SimaDecode decode(unsupported_raw_format);
        (void)decode.backend_fragment(2);
      },
      "raw_output supports only NV12 or I420", "unsupported raw SimaDecode format");

  unsupported_raw_format.raw_output = false;
  simaai::neat::SimaDecode converted(unsupported_raw_format);
  const std::string fragment = converted.backend_fragment(2);
  require_contains(fragment, "video/x-raw(memory:SystemMemory),format=RGB",
                   "raw_output=false should allow RGB through videoconvert");
  require(converted.output_spec({}).memory == "SystemMemory",
          "converted RGB output should advertise SystemMemory");
}

void check_h264_decode_wrapper_compatibility() {
  simaai::neat::H264Decode decode(2, "NV12", "legacy_decoder", true, "CVU", 1280, 720, 30, 6);
  const std::string fragment = decode.backend_fragment(6);
  require_contains(fragment, "neatdecoder name=legacy_decoder",
                   "H264Decode should still emit neatdecoder");
  require_contains(fragment, "sima-allocator-type=2", "H264Decode allocator option mismatch");
  require_contains(fragment, "op-buff-name=legacy_decoder",
                   "H264Decode buffer name option mismatch");
  require_contains(fragment, "dec-fmt=NV12", "H264Decode format option mismatch");
  require_contains(fragment, "next-element=CVU", "H264Decode next-element option mismatch");
  require_contains(fragment, "dec-width=1280", "H264Decode width option mismatch");
  require_contains(fragment, "dec-height=720", "H264Decode height option mismatch");
  require_contains(fragment, "dec-fps=30", "H264Decode fps option mismatch");
  require_contains(fragment, "num-buffers=6", "H264Decode pool option mismatch");
  require_element_names(decode.element_names(6), {"legacy_decoder"}, "H264Decode raw output");

  const simaai::neat::OutputSpec out = decode.output_spec(encoded_h264_spec());
  require(out.media_type == "video/x-raw", "H264Decode output media type mismatch");
  require(out.format == "NV12", "H264Decode output format mismatch");
  require(out.width == 1280 && out.height == 720,
          "H264Decode should preserve input shape in output spec");
  require(out.memory == "SimaAI", "H264Decode raw output should advertise SimaAI memory");
}

} // namespace

int main() {
  try {
    check_default_h264_raw_output();
    check_jpeg_raw_output_options();
    check_mjpeg_system_memory_output();
    check_invalid_options();
    check_h264_decode_wrapper_compatibility();

    std::cout << "[OK] unit_sima_decode_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
