#include "pipeline/internal/CapsStringUtil.h"

#include "gst/GstInit.h"
#include "pipeline/EncodedSampleUtil.h"

#include <gst/gst.h>

#include <memory>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {
namespace {

struct GstCapsUnref {
  void operator()(GstCaps* caps) const {
    if (caps) {
      gst_caps_unref(caps);
    }
  }
};

using GstCapsPtr = std::unique_ptr<GstCaps, GstCapsUnref>;

std::string trim_left(std::string_view value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) {
    return {};
  }
  return std::string(value.substr(start));
}

} // namespace

std::string caps_media_type(std::string_view caps) {
  std::string caps_string = trim_left(caps);
  if (caps_string.empty()) {
    return {};
  }

  simaai::neat::gst_init_once();
  GstCapsPtr parsed(gst_caps_from_string(caps_string.c_str()));
  if (!parsed || gst_caps_is_empty(parsed.get()) || gst_caps_get_size(parsed.get()) == 0) {
    return {};
  }

  const GstStructure* structure = gst_caps_get_structure(parsed.get(), 0);
  const char* media = structure ? gst_structure_get_name(structure) : nullptr;
  return media ? std::string(media) : std::string();
}

PayloadType payload_type_from_caps_string(std::string_view caps) {
  const std::string media = caps_media_type(caps);
  const PayloadType from_media = payload_type_from_media_type(media);
  if (from_media != PayloadType::Auto) {
    return from_media;
  }
  return caps_to_codec(std::string(caps)) != EncodedSpec::Codec::UNKNOWN ? PayloadType::Encoded
                                                                         : PayloadType::Auto;
}

} // namespace simaai::neat::pipeline_internal
