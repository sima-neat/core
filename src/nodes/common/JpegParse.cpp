#include "nodes/common/JpegParse.h"

#include "pipeline/PayloadType.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

JpegParse::JpegParse(JpegParseOptions opt) : opt_(std::move(opt)) {}

std::string JpegParse::backend_fragment(int node_index) const {
  std::string out = "jpegparse name=n" + std::to_string(node_index) + "_jpegparse";
  out += " disable-passthrough=";
  out += opt_.disable_passthrough ? "true" : "false";
  return out;
}

std::vector<std::string> JpegParse::element_names(int node_index) const {
  return {"n" + std::to_string(node_index) + "_jpegparse"};
}

OutputSpec JpegParse::output_spec(const OutputSpec& input) const {
  OutputSpec out;
  out.payload_type = PayloadType::Encoded;
  out.media_type = "image/jpeg";
  out.format = "JPEG";
  out.width = input.width;
  out.height = input.height;
  out.fps_num = input.fps_num;
  out.fps_den = input.fps_den;
  out.certainty = SpecCertainty::Hint;
  out.note = "JpegParse output";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> JpegParse(JpegParseOptions opt) {
  return std::make_shared<simaai::neat::JpegParse>(std::move(opt));
}

} // namespace simaai::neat::nodes
