/**
 * @file
 * @brief Umbrella include for SiMa NEAT's NodeGroup factories.
 *
 * Pulls in every built-in NodeGroup the framework ships: image and video input groups,
 * RTSP input, still-image input, model groups, MPK-compat group, OptiView output,
 * UDP/H264 output groups, image-to-H264-RTSP, and the GroupOutputSpec metadata type.
 * NodeGroups are the high-level "snap-in" units users assemble into Sessions.
 *
 * Include this instead of cherry-picking individual `nodes/groups` subheaders.
 */
#pragma once

#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/ImageToH264RtspGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/MpKCompatGroup.h"
#include "nodes/groups/OptiViewOutputGroup.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/groups/UdpOutputGroupG.h"
#include "nodes/groups/VideoInputGroup.h"
