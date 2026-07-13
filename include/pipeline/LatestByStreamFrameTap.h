// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pipeline/GraphOptions.h"

#include <functional>

namespace simaai::neat {

using LatestByStreamEncodedFrameCallback = std::function<void(Sample)>;

/**
 * @brief Install the process-wide callback for encoded frames in a fused live graph.
 *
 * The callback receives owned CPU copies of AU-aligned H.264 buffers immediately
 * after RTSP depacketization and before the hardware decoder. This lets an
 * application publish the already-encoded stream without allocating a second
 * decoder-frame queue or a hardware encoder.
 *
 * There is one subscriber per process. Install it before `Graph::build()`; only
 * fused pipelines built while a callback is installed receive a tap. Calls can
 * arrive concurrently from multiple RTSP streaming threads, so the callback must
 * be non-blocking. Do not call this function or the clear function recursively
 * from the callback.
 */
void set_latest_by_stream_encoded_frame_callback(LatestByStreamEncodedFrameCallback callback);

/**
 * @brief Disable new encoded-frame callbacks and wait for callbacks in progress.
 *
 * Do not call this function from the callback itself.
 */
void clear_latest_by_stream_encoded_frame_callback();

} // namespace simaai::neat
