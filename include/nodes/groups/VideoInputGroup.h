/**
 * @file
 * @ingroup nodes_groups
 * @brief Video input group options and builder.
 */
#pragma once

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"
#include "nodes/sima/H264Parse.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct VideoInputGroupOptions {
  std::string path;
  int demux_video_pad_index = 0;
  bool insert_queue = true;
  bool sync_mode = false;

  int parse_config_interval = 1;
  bool parse_enforce_au = true;

  int sima_allocator_type = 2;
  std::string out_format = "NV12";

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

simaai::neat::NodeGroup VideoInputGroup(const VideoInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
