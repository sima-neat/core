#include "nodes/groups/VideoInputGroup.h"

#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/internal/SyncBuild.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

bool caps_enabled(const VideoInputGroupOptions::OutputCaps& c) {
  return c.enable || c.width > 0 || c.height > 0 || c.fps > 0;
}

} // namespace

simaai::neat::Graph VideoInputGroup(const VideoInputGroupOptions& opt) {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;

  nodes.push_back(nodes::FileInput(opt.path));
  nodes.push_back(nodes::VideoTrackSelect(opt.demux_video_pad_index));

  const bool force_sync = simaai::neat::pipeline_internal::sync_build_mode();
  if (force_sync && opt.insert_queue && !opt.sync_mode) {
    simaai::neat::pipeline_internal::warn_sync_override("VideoInputGroup");
  }
  const bool insert_queue = opt.insert_queue && !(opt.sync_mode || force_sync);
  if (insert_queue)
    nodes.push_back(nodes::Queue());

  if (opt.parse_enforce_au) {
    nodes.push_back(nodes::H264ParseAu(opt.parse_config_interval));
  } else {
    nodes.push_back(nodes::H264Parse(opt.parse_config_interval));
  }

  if (insert_queue)
    nodes.push_back(nodes::Queue());

  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.raw_output = false;
  nodes.push_back(nodes::SimaDecode(dec));

  if (opt.use_videoconvert)
    nodes.push_back(nodes::VideoConvert());
  if (opt.use_videoscale)
    nodes.push_back(nodes::VideoScale());

  if (caps_enabled(opt.output_caps)) {
    const auto& c = opt.output_caps;
    nodes.push_back(nodes::CapsRaw(c.format, c.width, c.height, c.fps, c.memory));
  }

  if (!opt.extra_fragment.empty()) {
    nodes.push_back(nodes::Custom(opt.extra_fragment));
  }

  simaai::neat::Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

} // namespace simaai::neat::nodes::groups
