/**
 * @file
 * @ingroup nodes_groups
 * @brief Zero-copy MJPEG decode Graph fragment that always outputs NV12.
 */
#pragma once

#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"

namespace simaai::neat::nodes::groups {

/**
 * @brief Hardware MJPEG decode whose output is NV12 for 4:2:0 and 4:2:2 cameras.
 *
 * `source_format` is the stream's native decoded format, from
 * `jpeg_output_format(first_frame)`. NV12 (4:2:0) decodes straight to NV12. NV16 (4:2:2)
 * decodes zero-copy to NV16, then an EV74 ColorConvert writes NV12, because the decoder
 * writes only the stream's own chroma layout in zero-copy mode. `decode.out_format` is
 * replaced by `source_format`.
 *
 * @throws std::invalid_argument if `decode.type` is not JPEG/MJPEG or `source_format` is
 * neither NV12 nor NV16.
 */
simaai::neat::Graph MjpegDecode(SimaDecodeOptions decode, FormatTag source_format);

} // namespace simaai::neat::nodes::groups
