/**
 * @file
 * @brief Umbrella include for SiMa NEAT's atomic Node types.
 *
 * Pulls in every built-in Node the framework ships, organized by subdirectory:
 * common GStreamer-backed nodes (FileInput, ImageDecode/Freeze, JpegDecode, JpegParse,
 * MultipartJpegDemux, Output, Queue, VideoConvert/Rate/Scale, VideoTrackSelect, Caps),
 * I/O nodes (HttpSource, Input, MetadataSender, RTSPInput, StillImageInput, UdpOutput), RTP helpers
 * (H264CapsFixup, H264Depacketize, RTPJpegDepacketize), and the SiMa-specific MLA-bearing nodes
 * (Cast, CastTess, Dequant, Detess, DetessCast, DetessDequant, H264DecodeSima,
 * FeatureHistogram, GriderFast, H264EncodeSima, H264Packetize, H264Parse, SimaDecode,
 * PCIeSink/Src, Preproc, QuantTess, TrackDescriptor, TrackKLT, SimaArgMax,
 * SimaBoxDecode, SimaRender).
 *
 * Include this instead of cherry-picking individual node headers.
 */
#pragma once

#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/JpegDecode.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/io/HttpSource.h"
#include "nodes/io/Input.h"
#include "nodes/io/MetadataSender.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/io/StillImageInput.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/rtp/RTPJpegDepacketize.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/Dequant.h"
#include "nodes/sima/Detess.h"
#include "nodes/sima/DetessCast.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/FeatureHistogram.h"
#include "nodes/sima/GriderFast.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/PCIeSink.h"
#include "nodes/sima/PCIeSrc.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/TrackDescriptor.h"
#include "nodes/sima/TrackKLT.h"
#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/SimaDecode.h"
#include "nodes/sima/SimaRender.h"
