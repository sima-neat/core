#pragma once

namespace simaai::neat {

/// Register the private `neatmultipartjpegdemux` element.
///
/// The element replaces stock `multipartdemux` (and `jpegparse`) only on the
/// header-capture-enabled path. It parses part boundaries and part headers in one state
/// machine so selected headers stay attached to the JPEG bytes they arrived with.
bool register_neat_multipart_jpeg_demux();

} // namespace simaai::neat
