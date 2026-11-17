#include "gst/GstNeatMultipartJpegDemux.h"
#include <simaai/runtime/gstreamer/elements.h>

namespace simaai::neat {
bool register_neat_multipart_jpeg_demux() {
  return simaai::runtime::gstreamer::register_neat_multipart_jpeg_demux();
}
} // namespace simaai::neat
