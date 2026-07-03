#include "nodes/common/EncodedCapsFixup.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/io/HttpSource.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>

int main() {
  try {
    simaai::neat::HttpSourceOptions http_opt;
    http_opt.location = "http://example.local/mjpeg";
    http_opt.timeout_seconds = 7;
    http_opt.retries = -1;
    http_opt.is_live = true;
    http_opt.do_timestamp = true;
    http_opt.user_agent = "NeatTest";
    http_opt.ssl_strict = false;
    auto http = simaai::neat::nodes::HttpSource(http_opt);
    require_contains(http->backend_fragment(9), "souphttpsrc name=n9_souphttpsrc",
                     "HttpSource name mismatch");
    require_contains(http->backend_fragment(9), "location=\"http://example.local/mjpeg\"",
                     "HttpSource location mismatch");
    require_contains(http->backend_fragment(9), "timeout=7", "HttpSource timeout mismatch");
    require_contains(http->backend_fragment(9), "retries=-1", "HttpSource retries mismatch");
    require_contains(http->backend_fragment(9), "is-live=true", "HttpSource live flag mismatch");
    require_contains(http->backend_fragment(9), "do-timestamp=true",
                     "HttpSource timestamp flag mismatch");
    require_contains(http->backend_fragment(9), "user-agent=\"NeatTest\"",
                     "HttpSource user-agent mismatch");
    require_contains(http->backend_fragment(9), "ssl-strict=false",
                     "HttpSource ssl-strict mismatch");
    require(http->element_names(9).size() == 1, "HttpSource element_names size mismatch");
    require(http->element_names(9).front() == "n9_souphttpsrc", "HttpSource element name mismatch");

    bool empty_http_threw = false;
    try {
      (void)simaai::neat::nodes::HttpSource(simaai::neat::HttpSourceOptions{});
    } catch (const std::invalid_argument&) {
      empty_http_threw = true;
    }
    require(empty_http_threw, "HttpSource should reject empty location");

    simaai::neat::MultipartJpegDemuxOptions demux_opt;
    demux_opt.boundary = "frame";
    demux_opt.single_stream = true;
    auto multipart_demux = simaai::neat::nodes::MultipartJpegDemux(demux_opt);
    require_contains(multipart_demux->backend_fragment(10),
                     "multipartdemux name=n10_multipartdemux", "MultipartJpegDemux name mismatch");
    require_contains(multipart_demux->backend_fragment(10), "boundary=\"frame\"",
                     "MultipartJpegDemux boundary mismatch");
    require_contains(multipart_demux->backend_fragment(10), "single-stream=true",
                     "MultipartJpegDemux single-stream mismatch");
    require(multipart_demux->element_names(10).size() == 1,
            "MultipartJpegDemux element_names size mismatch");
    require(multipart_demux->element_names(10).front() == "n10_multipartdemux",
            "MultipartJpegDemux element name mismatch");

    simaai::neat::JpegParseOptions jpeg_parse_opt;
    jpeg_parse_opt.disable_passthrough = false;
    auto jpeg_parse = simaai::neat::nodes::JpegParse(jpeg_parse_opt);
    require_contains(jpeg_parse->backend_fragment(11), "jpegparse name=n11_jpegparse",
                     "JpegParse name mismatch");
    require_contains(jpeg_parse->backend_fragment(11), "disable-passthrough=false",
                     "JpegParse passthrough option mismatch");
    require(jpeg_parse->element_names(11).size() == 1, "JpegParse element_names size mismatch");
    require(jpeg_parse->element_names(11).front() == "n11_jpegparse",
            "JpegParse element name mismatch");

    const auto* jpeg_parse_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(jpeg_parse.get());
    require(jpeg_parse_provider != nullptr, "JpegParse should provide output spec");
    simaai::neat::OutputSpec jpeg_parse_input;
    jpeg_parse_input.width = 640;
    jpeg_parse_input.height = 480;
    jpeg_parse_input.fps_num = 30;
    jpeg_parse_input.fps_den = 1;
    jpeg_parse_input.memory = "SystemMemory";
    jpeg_parse_input.byte_size = 99;
    const simaai::neat::OutputSpec jpeg_parse_spec =
        jpeg_parse_provider->output_spec(jpeg_parse_input);
    require(jpeg_parse_spec.payload_type == simaai::neat::PayloadType::Encoded,
            "JpegParse output should be encoded");
    require(jpeg_parse_spec.media_type == "image/jpeg", "JpegParse media type mismatch");
    require(jpeg_parse_spec.format == "JPEG", "JpegParse format mismatch");
    require(jpeg_parse_spec.width == 640 && jpeg_parse_spec.height == 480,
            "JpegParse should preserve known dimensions");
    require(jpeg_parse_spec.fps_num == 30 && jpeg_parse_spec.fps_den == 1,
            "JpegParse should preserve known framerate");
    require(jpeg_parse_spec.memory.empty() && jpeg_parse_spec.byte_size == 0,
            "JpegParse should not preserve raw buffer fields");

    simaai::neat::EncodedCapsFixupOptions fixup_opt;
    fixup_opt.media_type = "image/jpeg";
    fixup_opt.fallback_fps = 30;
    auto fixup = simaai::neat::nodes::EncodedCapsFixup(fixup_opt);
    require(fixup->kind() == "EncodedCapsFixup", "EncodedCapsFixup kind mismatch");
    require_contains(fixup->backend_fragment(12), "identity name=n12_encoded_capsfix",
                     "EncodedCapsFixup fragment mismatch");
    require(fixup->element_names(12).size() == 1, "EncodedCapsFixup element_names size mismatch");
    require(fixup->element_names(12).front() == "n12_encoded_capsfix",
            "EncodedCapsFixup element name mismatch");
    const auto* raw_fixup = dynamic_cast<simaai::neat::EncodedCapsFixup*>(fixup.get());
    require(raw_fixup != nullptr, "EncodedCapsFixup dynamic_cast failed");
    require(raw_fixup->options().media_type == "image/jpeg",
            "EncodedCapsFixup media_type option mismatch");
    require(raw_fixup->options().fallback_fps == 30,
            "EncodedCapsFixup fallback_fps option mismatch");
    require(!raw_fixup->options().use_rtsp_sdp_fps,
            "EncodedCapsFixup should default to fallback-only FPS repair");

    std::cout << "[OK] unit_jpeg_framing_nodes_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
