#include "nodes/groups/MetadataReceiverOutputGroup.h"

#include "pipeline/DetectionTypes.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

std::string upper_ascii_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

std::string bbox_format_hint(const simaai::neat::Sample& sample) {
  if (!sample.payload_tag.empty())
    return sample.payload_tag;
  if (!sample.format.empty())
    return sample.format;
  if (sample.tensor.has_value() && sample.tensor->semantic.tess.has_value()) {
    return sample.tensor->semantic.tess->format;
  }
  return {};
}

bool decode_bbox_sample_recursive(const simaai::neat::Sample& sample, int frame_w, int frame_h,
                                  int topk, std::vector<simaai::neat::Box>* out, std::string* err) {
  if (!out)
    return false;
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    bool saw_candidate = false;
    std::string last_err;
    for (const auto& field : sample.fields) {
      std::vector<simaai::neat::Box> parsed;
      std::string field_err;
      if (decode_bbox_sample_recursive(field, frame_w, frame_h, topk, &parsed, &field_err)) {
        *out = std::move(parsed);
        return true;
      }
      if (!field_err.empty()) {
        saw_candidate = true;
        last_err = std::move(field_err);
      }
    }
    if (err) {
      if (saw_candidate) {
        *err = std::move(last_err);
      } else {
        *err = "bundle missing BBOX field";
      }
    }
    return false;
  }

  if (sample.kind != simaai::neat::SampleKind::Tensor || !sample.tensor.has_value()) {
    return false;
  }
  const std::string fmt = upper_ascii_copy(bbox_format_hint(sample));
  if (!fmt.empty() && fmt != "BBOX") {
    return false;
  }

  try {
    *out =
        simaai::neat::decode_bbox_tensor(sample.tensor.value(), frame_w, frame_h, topk, true).boxes;
    return true;
  } catch (const std::exception& ex) {
    if (err)
      *err = std::string("bbox parse failed: ") + ex.what();
    return false;
  }
}

int64_t now_ms_i64() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

} // namespace

bool MetadataReceiverOutputNodeGroup::init(const MetadataReceiverOutputNodeGroupOptions& opt,
                                           size_t streams, std::string* err) {
  opt_ = opt;
  senders_.clear();
  if (opt_.labels.empty()) {
    opt_.labels = simaai::neat::MetadataReceiverDefaultLabels();
  }

  std::string udp_err;
  if (!udp_.init(opt_.udp, streams, &udp_err)) {
    if (err)
      *err = "MetadataReceiverOutputNodeGroup: UDP init failed: " + udp_err;
    return false;
  }

  if (!opt_.send_metadata)
    return true;

  try {
    senders_.reserve(streams);
    for (size_t i = 0; i < streams; ++i) {
      simaai::neat::MetadataReceiverChannelOptions sender_opt;
      sender_opt.host = opt_.udp.host;
      sender_opt.channel = static_cast<int>(i);
      sender_opt.metadata_port_base = opt_.metadata_port_base;

      std::string sender_err;
      auto sender = std::make_unique<simaai::neat::MetadataReceiverOutput>(sender_opt, &sender_err);
      if (!sender->ok()) {
        if (err)
          *err = "MetadataReceiverOutputNodeGroup: sender init failed: " + sender_err;
        stop();
        return false;
      }
      senders_.push_back(std::move(sender));
    }
  } catch (const std::exception& ex) {
    stop();
    if (err)
      *err = ex.what();
    return false;
  }
  return true;
}

bool MetadataReceiverOutputNodeGroup::push_video(size_t idx,
                                                 const simaai::neat::Sample& sample) const {
  return udp_.push_video(idx, sample);
}

bool MetadataReceiverOutputNodeGroup::try_push_video(size_t idx,
                                                     const simaai::neat::Sample& sample) const {
  return udp_.try_push_video(idx, sample);
}

bool MetadataReceiverOutputNodeGroup::send_json(size_t stream_idx, const std::string& payload,
                                                std::string* err) const {
  if (!opt_.send_metadata)
    return true;
  if (stream_idx >= senders_.size() || !senders_[stream_idx]) {
    if (err)
      *err = "invalid stream index for metadata sender";
    return false;
  }
  return senders_[stream_idx]->send_json(payload, err);
}

bool MetadataReceiverOutputNodeGroup::send_metadata(
    size_t stream_idx, const simaai::neat::MetadataReceiverPayload& payload,
    std::string* err) const {
  if (!opt_.send_metadata)
    return true;
  if (stream_idx >= senders_.size() || !senders_[stream_idx]) {
    if (err)
      *err = "invalid stream index for metadata sender";
    return false;
  }
  return senders_[stream_idx]->send_metadata(payload, err);
}

bool MetadataReceiverOutputNodeGroup::emit_object_detection(
    const MetadataReceiverObjectDetectionInput& in,
    MetadataReceiverObjectDetectionResult* out) const {
  MetadataReceiverObjectDetectionResult local;
  if (!out)
    out = &local;
  *out = MetadataReceiverObjectDetectionResult{};

  if (!opt_.send_metadata) {
    out->ok = true;
    return true;
  }
  if (in.stream_idx >= senders_.size() || !senders_[in.stream_idx]) {
    out->error = "invalid stream index for metadata sender";
    return false;
  }
  if (!in.yolo_sample) {
    out->error = "missing yolo sample";
    return false;
  }
  if (opt_.frame_w <= 0 || opt_.frame_h <= 0) {
    out->error = "invalid frame size for box parsing";
    return false;
  }

  std::vector<simaai::neat::Box> boxes;
  std::string parse_err;
  if (!decode_bbox_sample_recursive(*in.yolo_sample, opt_.frame_w, opt_.frame_h, opt_.topk, &boxes,
                                    &parse_err)) {
    out->error = parse_err.empty() ? "bbox tensor not found in yolo sample" : std::move(parse_err);
    return false;
  }

  (void)opt_.parse_debug;

  std::vector<simaai::neat::MetadataReceiverObject> objects;
  objects.reserve(boxes.size());
  for (const auto& b : boxes) {
    int x1 = static_cast<int>(b.x1);
    int y1 = static_cast<int>(b.y1);
    int w = static_cast<int>(b.x2 - b.x1);
    int h = static_cast<int>(b.y2 - b.y1);
    if (x1 < 0)
      x1 = 0;
    if (y1 < 0)
      y1 = 0;
    if (w < 0)
      w = 0;
    if (h < 0)
      h = 0;
    if (x1 + w > opt_.frame_w)
      w = opt_.frame_w - x1;
    if (y1 + h > opt_.frame_h)
      h = opt_.frame_h - y1;
    if (w < 0)
      w = 0;
    if (h < 0)
      h = 0;

    simaai::neat::MetadataReceiverObject obj;
    obj.x = x1;
    obj.y = y1;
    obj.w = w;
    obj.h = h;
    obj.score = b.score;
    obj.class_id = b.class_id;
    objects.push_back(obj);
  }

  const int64_t ts_ms = pick_timestamp_ms_(in);
  const std::string frame_id =
      (in.output_frame_id >= 0) ? std::to_string(in.output_frame_id) : std::to_string(in.frame_id);

  const std::string payload_json =
      simaai::neat::MetadataReceiverMakeObjectDetectionJson(ts_ms, frame_id, objects, opt_.labels);

  int metadata_delay_ms = opt_.metadata_delay_ms;
  if (metadata_delay_ms <= 0 && opt_.video_delay_ms > 0) {
    metadata_delay_ms = opt_.video_delay_ms;
  }
  if (metadata_delay_ms > 0 && ts_ms >= 0) {
    const int64_t now_ms = now_ms_i64();
    const int64_t target_ms = ts_ms + metadata_delay_ms;
    if (now_ms < target_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(target_ms - now_ms));
    }
  }

  std::string send_err;
  if (!senders_[in.stream_idx]->send_json(payload_json, &send_err)) {
    out->error = "metadata send failed: " + send_err;
    return false;
  }

  out->ok = true;
  out->boxes = static_cast<int>(objects.size());
  out->nonempty = !objects.empty();
  return true;
}

void MetadataReceiverOutputNodeGroup::stop() {
  udp_.stop();
  senders_.clear();
}

int64_t MetadataReceiverOutputNodeGroup::pick_timestamp_ms_(
    const MetadataReceiverObjectDetectionInput& in) const {
  if (in.capture_ms >= 0)
    return in.capture_ms;
  if (in.decoded_sample && in.decoded_sample->pts_ns >= 0) {
    return in.decoded_sample->pts_ns / 1000000;
  }
  if (in.yolo_sample && in.yolo_sample->pts_ns >= 0) {
    return in.yolo_sample->pts_ns / 1000000;
  }
  if (in.yolo_ms >= 0)
    return in.yolo_ms;
  return now_ms_i64();
}

} // namespace simaai::neat::nodes::groups
