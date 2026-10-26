#include "nodes/groups/MjpegDecode.h"

#include "nodes/sima/ImageTransform.h"

#include <stdexcept>
#include <utility>

namespace simaai::neat::nodes::groups {

simaai::neat::Graph MjpegDecode(SimaDecodeOptions decode, FormatTag source_format) {
  if (decode.type != SimaDecodeType::JPEG && decode.type != SimaDecodeType::MJPEG) {
    throw std::invalid_argument("MjpegDecode: decode.type must be JPEG or MJPEG");
  }
  if (source_format != FormatTag::NV12 && source_format != FormatTag::NV16) {
    throw std::invalid_argument("MjpegDecode: source_format must be NV12 (4:2:0) or NV16 (4:2:2)");
  }
  if (source_format == FormatTag::NV16 && !decode.raw_output) {
    throw std::invalid_argument("MjpegDecode: 4:2:2 (NV16) decoding requires raw_output");
  }
  decode.out_format = source_format;
  simaai::neat::Graph graph;
  graph.add(nodes::SimaDecode(std::move(decode)));
  if (source_format == FormatTag::NV16) {
    // Zero-copy decode writes 4:2:2 JPEGs only as NV16; convert on the EV74, in the
    // encoder's NV12 surface so a downstream VideoSender can take it without a copy.
    ColorConvertOptions color;
    color.output_format = ImageFormat::NV12;
    color.surface = ImageSurfaceLayout::EncoderNv12;
    graph.add(nodes::ColorConvert(color));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
