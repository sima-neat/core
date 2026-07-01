#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/PayloadType.h"

#include <cctype>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

inline std::string caps_media_type(std::string_view caps) {
  const std::size_t start = caps.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) {
    return {};
  }

  std::size_t end = start;
  while (end < caps.size()) {
    const unsigned char c = static_cast<unsigned char>(caps[end]);
    if (caps[end] == ',' || std::isspace(c) != 0) {
      break;
    }
    ++end;
  }
  return std::string(caps.substr(start, end - start));
}

inline PayloadType payload_type_from_caps_string(std::string_view caps) {
  const std::string media = caps_media_type(caps);
  const PayloadType from_media = payload_type_from_media_type(media);
  if (from_media != PayloadType::Auto) {
    return from_media;
  }
  return caps_to_codec(std::string(caps)) != EncodedSpec::Codec::UNKNOWN ? PayloadType::Encoded
                                                                         : PayloadType::Auto;
}

} // namespace simaai::neat::pipeline_internal
