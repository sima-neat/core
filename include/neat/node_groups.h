/**
 * @file
 * @brief Compatibility umbrella include for SiMa NEAT's reusable Graph fragment factories.
 *
 * Pulls in every built-in reusable Graph fragment factory the framework ships: image and
 * video input fragments, RTSP input, still-image input, model fragments, metadata/video sender
 * fragments, UDP/H264 output fragments, image-to-H264-RTSP, and the GroupOutputSpec metadata type.
 * Reusable fragments are Graphs; add them with `Graph::add(fragment)`.
 *
 * Include this instead of cherry-picking individual `nodes/groups` subheaders. The header
 * name remains for source compatibility during the Graph migration.
 */
#pragma once

#include "nodes/groups/GroupOutputSpec.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/ImageToH264RtspGroup.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/groups/UdpOutputGroupG.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/groups/VideoSender.h"
#include "graphs/Fragments.h"
