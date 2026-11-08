#include "pipeline/internal/SampleProjection.h"

#include <utility>

namespace simaai::neat::pipeline_internal {

StreamNames::StreamNames(std::string default_name) : default_name_(std::move(default_name)) {
  names_.emplace_back(); // id 0: unassigned
  if (!default_name_.empty()) {
    default_id_ = 1;
    names_.push_back(default_name_);
    ids_.emplace(default_name_, default_id_);
  }
}

std::uint32_t StreamNames::intern(std::string_view name) {
  if (name.empty() || name == default_name_) {
    return default_id_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  const std::string key(name);
  if (const auto it = ids_.find(key); it != ids_.end()) {
    return it->second;
  }
  const auto id = static_cast<std::uint32_t>(names_.size());
  names_.push_back(key);
  ids_.emplace(key, id);
  return id;
}

std::string StreamNames::name(std::uint32_t id) const {
  if (id == 0) {
    return {};
  }
  if (id == default_id_) {
    return default_name_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  return id < names_.size() ? names_[id] : std::string{};
}

ProjectedStamp project_push_identity(const Sample& sample, StreamNames& streams) {
  ProjectedStamp out;
  out.stream = streams.intern(sample.stream_id);
  out.frame_id = sample.frame_id >= 0 ? sample.frame_id : -1;
  if (sample.pts_ns >= 0) {
    out.source_time_ns = sample.pts_ns;
  }
  return out;
}

void apply_pulled_identity(const ProjectedStamp& stamp, const StreamNames& streams, Sample& out) {
  out.frame_id = stamp.frame_id >= 0 ? stamp.frame_id : -1;
  out.stream_id = streams.name(stamp.stream);
  out.pts_ns = stamp.source_time_ns.value_or(-1);
  out.input_seq = stamp.seq >= 1 ? static_cast<std::int64_t>(stamp.seq - 1) : -1;
  out.orig_input_seq = out.frame_id >= 0 ? out.frame_id : out.input_seq;
}

const char* pixel_format_token(ImageSpec::PixelFormat format) noexcept {
  switch (format) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::NV16:
    return "NV16";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

} // namespace simaai::neat::pipeline_internal
