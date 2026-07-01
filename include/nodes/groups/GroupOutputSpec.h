/**
 * @file
 * @ingroup nodes_groups
 * @brief `OutputSpec` projections for input Graph fragment factories.
 *
 * Each input Graph fragment factory (image, HTTP MJPEG, RTSP, video file) advertises its
 * negotiated downstream caps via an `OutputSpec`. These free functions translate a fragment's
 * options struct into the corresponding `OutputSpec` so callers can discover stream geometry
 * without instantiating the Graph fragment itself.
 *
 * @see HttpMjpegDecodedInput
 * @see ImageInputGroup
 * @see RtspEncodedInput
 * @see RtspDecodedInput
 * @see VideoInputGroup
 */
#pragma once

#include "builder/OutputSpec.h"
#include "nodes/groups/HttpMjpegDecodedInput.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/groups/VideoInputGroup.h"

namespace simaai::neat::nodes::groups {

/**
 * @brief Project a `HttpMjpegDecodedInputOptions` into the `OutputSpec` the fragment will
 * advertise.
 * @param opt HTTP MJPEG group options (url, framing, decoder, output caps).
 * @return Negotiated downstream caps for this configuration.
 * @ingroup nodes_groups
 */
OutputSpec HttpMjpegDecodedInputOutputSpec(const HttpMjpegDecodedInputOptions& opt);

/**
 * @brief Project an `ImageInputGroupOptions` into the `OutputSpec` the fragment will advertise.
 * @param opt Image-input group options (path, decoder, output caps).
 * @return Negotiated downstream caps for this configuration.
 * @ingroup nodes_groups
 */
OutputSpec ImageInputGroupOutputSpec(const ImageInputGroupOptions& opt);

/**
 * @brief Project a `RtspEncodedInputOptions` into the `OutputSpec` the fragment will advertise.
 * @param opt RTSP encoded-input group options (url, latency, codec framing).
 * @return Encoded caps for this configuration.
 * @ingroup nodes_groups
 */
OutputSpec RtspEncodedInputOutputSpec(const RtspEncodedInputOptions& opt);

/**
 * @brief Project an `RtspDecodedInputOptions` into the `OutputSpec` the fragment will advertise.
 * @param opt RTSP-input group options (url, latency, decoder, output caps).
 * @return Negotiated downstream caps for this configuration.
 * @ingroup nodes_groups
 */
OutputSpec RtspDecodedInputOutputSpec(const RtspDecodedInputOptions& opt);

/**
 * @brief Project a `VideoInputGroupOptions` into the `OutputSpec` the fragment will advertise.
 * @param opt Video-file-input group options (path, parse config, output caps).
 * @return Negotiated downstream caps for this configuration.
 * @ingroup nodes_groups
 */
OutputSpec VideoInputGroupOutputSpec(const VideoInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
