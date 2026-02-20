/**
 * @file
 * @ingroup nodes_groups
 * @brief Image input group options and builder.
 */
#pragma once

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"

#include <string>

namespace simaai::neat::nodes::groups {

struct ImageInputGroupOptions {
  std::string path;
  // If sima_decoder is enabled and this is >0, it may be clamped to a minimum
  // to allow decoder startup. Set SIMA_IMAGEFREEZE_MIN_BUFFERS=0 to disable.
  int imagefreeze_num_buffers = -1;
  int fps = 30;
  bool sync_mode = false;

  bool use_videorate = false;
  bool use_videoconvert = true;
  bool use_videoscale = false;

  struct OutputCaps {
    bool enable = true;
    std::string format = "NV12";
    int width = -1;
    int height = -1;
    int fps = -1;
    simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::SystemMemory;
  } output_caps;

  enum class Decoder {
    Auto = 0, // decodebin (jpg/png auto)
    ForceJpeg,
    ForcePng,
    Custom,
  };

  Decoder decoder = Decoder::Auto;
  std::string custom_decoder_fragment;

  struct SimaDecoder {
    bool enable = false;
    int sima_allocator_type = 2;
    std::string decoder_name = "decoder";
    bool raw_output = false;
    // Optional: select output buffer target ("CVU" or "APU") for neatdecoder.
    std::string next_element;
    // Use software H264 encoder (x264/openh264) before sima decoder.
    bool use_sw_encoder = false;
    int sw_bitrate_kbps = 4000;
  } sima_decoder;

  // Optional raw fragment inserted before ImageFreeze (advanced use)
  std::string extra_fragment;
};

simaai::neat::NodeGroup ImageInputGroup(const ImageInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
