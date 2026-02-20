/**
 * @file
 * @ingroup nodes_groups
 * @brief RTSP input group options and builder.
 */
#pragma once

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct RtspDecodedInputOptions {
  std::string url;
  int latency_ms = 200;
  bool tcp = true;
  int payload_type = 96;
  int h264_parse_config_interval = -1;
  int h264_fps = -1;
  int h264_width = -1;
  int h264_height = -1;
  bool insert_queue = true;
  bool sync_mode = false;
  bool auto_caps_from_stream = true;
  int fallback_h264_fps = -1;
  int fallback_h264_width = -1;
  int fallback_h264_height = -1;

  int sima_allocator_type = 2;
  std::string out_format = "NV12";
  std::string decoder_name;
  bool decoder_raw_output = true;
  std::string decoder_next_element;

  bool use_videoconvert = false;
  bool use_videoscale = false;

  struct OutputCaps {
    bool enable = false;
    std::string format = "NV12";
    int width = -1;
    int height = -1;
    int fps = -1;
    simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::SystemMemory;
  } output_caps;

  std::string extra_fragment;
};

simaai::neat::NodeGroup RtspDecodedInput(const RtspDecodedInputOptions& opt);

} // namespace simaai::neat::nodes::groups
