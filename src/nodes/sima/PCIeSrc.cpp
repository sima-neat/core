#include "nodes/sima/PCIeSrc.h"

#include "gst/GstHelpers.h"

#include <cctype>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

std::string upper_copy(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

bool has_fixed_caps(const PCIeSrcOptions& opt) {
  return !opt.format.empty() && opt.width > 0 && opt.height > 0;
}

} // namespace

PCIeSrc::PCIeSrc(PCIeSrcOptions opt) : opt_(std::move(opt)) {}

std::string PCIeSrc::backend_fragment(int node_index) const {
  require_element("neatpciesrc", "PCIeSrc::backend_fragment");

  std::ostringstream ss;
  ss << "neatpciesrc name=n" << node_index << "_pciesrc"
     << " buffer-size=" << opt_.buffer_size;

  if (has_fixed_caps(opt_)) {
    ss << " ! capsfilter name=n" << node_index << "_pciesrc_caps"
       << " caps=\"video/x-raw,format=" << opt_.format << ",width=" << opt_.width
       << ",height=" << opt_.height;
    if (opt_.fps_n > 0 && opt_.fps_d > 0) {
      ss << ",framerate=" << opt_.fps_n << "/" << opt_.fps_d;
    }
    ss << "\"";
  }

  return ss.str();
}

std::vector<std::string> PCIeSrc::element_names(int node_index) const {
  std::vector<std::string> names;
  names.push_back("n" + std::to_string(node_index) + "_pciesrc");
  if (has_fixed_caps(opt_)) {
    names.push_back("n" + std::to_string(node_index) + "_pciesrc_caps");
  }
  return names;
}

OutputSpec PCIeSrc::output_spec(const OutputSpec& /*input*/) const {
  OutputSpec out;
  out.media_type = "video/x-raw";
  out.format = opt_.format;
  out.width = opt_.width;
  out.height = opt_.height;
  out.fps_num = (opt_.fps_n > 0) ? opt_.fps_n : 0;
  out.fps_den = (opt_.fps_d > 0) ? opt_.fps_d : 1;
  out.memory = "System";
  out.certainty = SpecCertainty::Hint;
  out.note = "neatpciesrc output";

  const std::string fmt = upper_copy(opt_.format);
  if (fmt == "RGB" || fmt == "BGR") {
    out.dtype = "UInt8";
    out.layout = "HWC";
    out.depth = 3;
  } else if (fmt == "GRAY" || fmt == "GRAY8") {
    out.dtype = "UInt8";
    out.layout = "HW";
    out.depth = 1;
  } else if (fmt == "NV12" || fmt == "I420") {
    out.dtype = "UInt8";
    out.layout = "Planar";
    out.depth = 3;
  } else if (!opt_.format.empty()) {
    out.note = "unrecognized format: " + opt_.format;
  }

  out.byte_size = expected_byte_size(out);
  return out;
}

} // namespace simaai::neat

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> PCIeSrc(PCIeSrcOptions opt) {
  return std::make_shared<simaai::neat::PCIeSrc>(std::move(opt));
}

} // namespace simaai::neat::nodes
