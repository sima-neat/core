#include "nodes/sima/H264Parse.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

std::string alignment_to_caps(H264ParseOptions::Alignment a) {
  switch (a) {
  case H264ParseOptions::Alignment::AU:
    return "alignment=(string)au";
  case H264ParseOptions::Alignment::NAL:
    return "alignment=(string)nal";
  case H264ParseOptions::Alignment::Auto:
    return "";
  }
  return "";
}

std::string stream_format_to_caps(H264ParseOptions::StreamFormat f) {
  switch (f) {
  case H264ParseOptions::StreamFormat::AVC:
    return "stream-format=(string)avc";
  case H264ParseOptions::StreamFormat::ByteStream:
    return "stream-format=(string)byte-stream";
  case H264ParseOptions::StreamFormat::Auto:
    return "";
  }
  return "";
}

} // namespace

H264Parse::H264Parse(H264ParseOptions opt) : opt_(std::move(opt)) {}

std::string H264Parse::backend_fragment(int node_index) const {
  const std::string p = "n" + std::to_string(node_index) + "_h264parse";
  std::ostringstream ss;
  ss << "h264parse name=" << p << " disable-passthrough=true "
     << "config-interval=" << opt_.config_interval;

  if (!opt_.enforce_caps) {
    return ss.str();
  }

  const std::string c = "n" + std::to_string(node_index) + "_h264_caps";

  std::ostringstream caps;
  caps << "video/x-h264";

  const std::string align = alignment_to_caps(opt_.alignment);
  const std::string sf = stream_format_to_caps(opt_.stream_format);

  if (!sf.empty())
    caps << "," << sf;
  if (!align.empty())
    caps << "," << align;

  ss << " ! capsfilter name=" << c << " caps=\"" << caps.str() << "\"";
  return ss.str();
}

std::vector<std::string> H264Parse::element_names(int node_index) const {
  std::vector<std::string> names;
  names.push_back("n" + std::to_string(node_index) + "_h264parse");
  if (opt_.enforce_caps) {
    names.push_back("n" + std::to_string(node_index) + "_h264_caps");
  }
  return names;
}

OutputSpec H264Parse::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-h264";
  out.format = "H264";
  out.certainty = SpecCertainty::Hint;
  out.note = "H264 parse output";
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> H264Parse(simaai::neat::H264ParseOptions opt) {
  return std::make_shared<simaai::neat::H264Parse>(std::move(opt));
}

} // namespace simaai::neat::nodes
