#include "nodes/groups/UdpOutputGroup.h"

#include "nodes/groups/UdpH264OutputGroup.h"
#include "nodes/io/Input.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Session.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

simaai::neat::Sample make_dummy_encoded_sample(size_t bytes, const std::string& caps) {
  if (bytes == 0)
    bytes = 1;
  return simaai::neat::make_encoded_sample(std::vector<uint8_t>(bytes, 0), caps);
}

} // namespace

bool UdpOutputNodeGroup::init(const UdpOutputNodeGroupOptions& opt, size_t streams,
                              std::string* err) {
  runs_.clear();
  opt_ = opt;
  if (streams == 0) {
    if (err)
      *err = "UdpOutputNodeGroup: streams must be > 0";
    return false;
  }
  if (opt_.h264_caps.empty()) {
    if (err)
      *err = "UdpOutputNodeGroup: missing h264_caps";
    return false;
  }

  runs_.reserve(streams);
  try {
    for (size_t i = 0; i < streams; ++i) {
      simaai::neat::Session forward;
      simaai::neat::InputOptions src_opt;
      src_opt.use_simaai_pool = false;
      src_opt.block = true;
      src_opt.caps_override = opt_.h264_caps;
      forward.add(simaai::neat::nodes::Input(src_opt));

      simaai::neat::nodes::groups::UdpH264OutputGroupOptions udp_opt;
      udp_opt.h264_caps = opt_.h264_caps;
      udp_opt.payload_type = opt_.payload_type;
      udp_opt.config_interval = opt_.config_interval;
      udp_opt.udp_host = opt_.host;
      udp_opt.udp_port = opt_.video_port_base + static_cast<int>(i);
      udp_opt.udp_sync = opt_.udp_sync;
      udp_opt.udp_async = opt_.udp_async;
      forward.add(simaai::neat::nodes::groups::UdpH264OutputGroup(udp_opt));

      simaai::neat::RunOptions run_opt;
      run_opt.enable_metrics = opt_.enable_timings;

      auto dummy = make_dummy_encoded_sample(1, opt_.h264_caps);
      auto run = std::make_shared<simaai::neat::Run>(
          forward.build(dummy, simaai::neat::RunMode::Async, run_opt));
      runs_.push_back(std::move(run));
    }
  } catch (const std::exception& ex) {
    stop();
    if (err)
      *err = ex.what();
    return false;
  }
  return true;
}

bool UdpOutputNodeGroup::push_video(size_t idx, const simaai::neat::Sample& sample) const {
  if (idx >= runs_.size() || !runs_[idx])
    return false;
  return runs_[idx]->push(sample);
}

bool UdpOutputNodeGroup::try_push_video(size_t idx, const simaai::neat::Sample& sample) const {
  if (idx >= runs_.size() || !runs_[idx])
    return false;
  return runs_[idx]->try_push(sample);
}

void UdpOutputNodeGroup::stop() {
  for (auto& run : runs_) {
    if (run)
      run->stop();
  }
}

} // namespace simaai::neat::nodes::groups
