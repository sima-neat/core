#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/PreprocessPlan.h"
#include "pipeline/TensorCore.h"

#include <string>

namespace simaai::neat::internal {

inline std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::NV16:
    return "NV16";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

inline std::string preprocess_color_format_name(PreprocessColorFormat fmt) {
  switch (fmt) {
  case PreprocessColorFormat::RGB:
    return "RGB";
  case PreprocessColorFormat::BGR:
    return "BGR";
  case PreprocessColorFormat::GRAY8:
    return "GRAY8";
  case PreprocessColorFormat::NV12:
    return "NV12";
  case PreprocessColorFormat::I420:
    return "I420";
  case PreprocessColorFormat::NV16:
    return "NV16";
  case PreprocessColorFormat::Auto:
    break;
  }
  return "";
}

} // namespace simaai::neat::internal
