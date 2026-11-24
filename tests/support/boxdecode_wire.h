#pragma once

// The BoxDecode wire of a decode family, for the tests that compare a model route's pulled wire
// between the two backends: which task a Model's options decode, how long the wire is, what its
// unused capacity holds, and what the public parser of the task makes of it. Public API only.
//
//   detection      int32 count; top_k records of 24 bytes (int32 x, y, w, h; float32 score;
//                  int32 class); records at and above count zero.
//   pose           the detection wire, then top_k blocks of 17 keypoints of 12 bytes (uint32 x, y;
//                  float32 visibility); blocks at and above count zero.
//   segmentation   the detection wire, then top_k masks of 160x160 uint8; masks at and above count
//                  zero.
//   oriented       OrientedBoxesHeaderV1, then capacity records of OrientedBoxRecordV1; records at
//                  and above count zero.

#include "model/Model.h"
#include "pipeline/DetectionTypes.h"
#include "qualification/qualification_common.h"
#include "support/backend_selection.h"
#include "test_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sima_test {

enum class DecodeTask { Detection, Pose, Segmentation, Oriented };

inline DecodeTask decode_task(simaai::neat::BoxDecodeType type) {
  using simaai::neat::BoxDecodeType;
  switch (type) {
  case BoxDecodeType::YoloV8Pose:
  case BoxDecodeType::YoloV26Pose:
    return DecodeTask::Pose;
  case BoxDecodeType::YoloV5Seg:
  case BoxDecodeType::YoloV7Seg:
  case BoxDecodeType::YoloV8Seg:
  case BoxDecodeType::YoloV9Seg:
  case BoxDecodeType::YoloV10Seg:
  case BoxDecodeType::YoloV26Seg:
    return DecodeTask::Segmentation;
  case BoxDecodeType::YoloV8Obb:
  case BoxDecodeType::YoloV26Obb:
    return DecodeTask::Oriented;
  default:
    return DecodeTask::Detection;
  }
}

inline const char* decode_task_name(DecodeTask task) {
  return task == DecodeTask::Pose           ? "pose"
         : task == DecodeTask::Segmentation ? "segmentation"
         : task == DecodeTask::Oriented     ? "oriented"
                                            : "detection";
}

// The options of the test's Model: the catalogue's option set of the package (--opts=<file>, the
// harness's format: qualification_common.h) or, without one, `fallback`. `frame_format` is the
// format the pushed frames say they are.
struct DecodeOptions {
  simaai::neat::Model::Options options;
  simaai::neat::ImageSpec::PixelFormat frame_format = simaai::neat::ImageSpec::PixelFormat::BGR;
  simaai::neat::FormatTag input_format = simaai::neat::FormatTag::BGR;
  bool from_file = false;
  DecodeTask task = DecodeTask::Detection;
};

inline DecodeOptions decode_options(int argc, char** argv,
                                    const simaai::neat::Model::Options& fallback) {
  DecodeOptions out;
  out.options = fallback;
  const std::string path = arg_value(argc, argv, "opts");
  if (!path.empty()) {
    const auto set = simaai::neat::qualification::read_option_set(path);
    require(set.given && set.image && set.not_applied.empty(),
            "--opts=" + path + ": an image model's option set, every line applied");
    require(set.frame_format == "RGB" || set.frame_format == "BGR",
            "--opts=" + path + ": RGB or BGR frames");
    out.options = set.options;
    out.from_file = true;
    if (set.frame_format == "RGB") {
      out.frame_format = simaai::neat::ImageSpec::PixelFormat::RGB;
      out.input_format = simaai::neat::FormatTag::RGB;
    }
  }
  out.task = decode_task(out.options.decode_type);
  return out;
}

inline std::size_t wire_bytes(DecodeTask task, int top_k) {
  const std::size_t k = static_cast<std::size_t>(top_k);
  switch (task) {
  case DecodeTask::Pose:
    return 4U + k * (24U + 17U * 12U);
  case DecodeTask::Segmentation:
    return 4U + k * (24U + 160U * 160U);
  case DecodeTask::Oriented:
    return sizeof(simaai::neat::OrientedBoxesHeaderV1) +
           k * sizeof(simaai::neat::OrientedBoxRecordV1);
  default:
    return 4U + 24U * k;
  }
}

inline const char* wire_format(DecodeTask task) {
  return task == DecodeTask::Oriented ? simaai::neat::kDetectionFormatBboxObbV1
                                      : simaai::neat::kDetectionFormatBbox;
}

// The wire's count of detections; -1 when the wire is too short to say.
inline std::int32_t wire_count(DecodeTask task, const std::vector<std::uint8_t>& wire) {
  if (task == DecodeTask::Oriented) {
    simaai::neat::OrientedBoxesHeaderV1 header;
    if (wire.size() < sizeof header) {
      return -1;
    }
    std::memcpy(&header, wire.data(), sizeof header);
    return static_cast<std::int32_t>(header.count);
  }
  std::int32_t count = -1;
  if (wire.size() >= sizeof count) {
    std::memcpy(&count, wire.data(), sizeof count);
  }
  return count;
}

// The wire has the task's size for `top_k`, a count of 0..top_k, and zeros in all its unused
// capacity.
inline void require_wire(DecodeTask task, const std::vector<std::uint8_t>& wire, int top_k,
                         const std::string& at) {
  require(wire.size() == wire_bytes(task, top_k),
          at + "the " + decode_task_name(task) + " wire of top_k " + std::to_string(top_k) +
              " is " + std::to_string(wire_bytes(task, top_k)) + " bytes (" +
              std::to_string(wire.size()) + ")");
  const std::int32_t count = wire_count(task, wire);
  require(count >= 0 && count <= top_k, at + "the wire's count is 0..top_k");
  const std::size_t n = static_cast<std::size_t>(count), k = static_cast<std::size_t>(top_k);
  const auto zero = [&](std::size_t from, std::size_t to, const char* what) {
    for (std::size_t i = from; i < to; ++i) {
      require(wire[i] == 0U, at + "the wire is zero behind its last " + what);
    }
  };
  if (task == DecodeTask::Oriented) {
    simaai::neat::OrientedBoxesHeaderV1 header;
    std::memcpy(&header, wire.data(), sizeof header);
    const simaai::neat::OrientedBoxesHeaderV1 expected;
    require(header.magic == expected.magic && header.version == expected.version &&
                header.header_bytes == 32U && header.total_bytes == wire.size() &&
                header.capacity == k && header.records_offset == 32U && header.record_stride == 32U,
            at + "the oriented wire's header states its layout");
    zero(32U + 32U * n, wire.size(), "box");
    return;
  }
  zero(4U + 24U * n, 4U + 24U * k, "box");
  if (task == DecodeTask::Pose) {
    zero(4U + 24U * k + 204U * n, wire.size(), "keypoint block");
  } else if (task == DecodeTask::Segmentation) {
    zero(4U + 24U * k + 25600U * n, wire.size(), "mask");
  }
}

// The wire's detections as text: its records as they lie in the wire.
inline std::string wire_text(DecodeTask task, const std::vector<std::uint8_t>& wire) {
  std::string out;
  const std::int32_t count = wire_count(task, wire);
  char text[160];
  for (std::int32_t i = 0; i < count; ++i) {
    if (task == DecodeTask::Oriented) {
      simaai::neat::OrientedBoxRecordV1 r;
      std::memcpy(&r, wire.data() + 32 + 32 * i, sizeof r);
      std::snprintf(text, sizeof text,
                    " [cx=%.3f cy=%.3f w=%.3f h=%.3f angle=%.5f score=%.6f cls=%d]",
                    static_cast<double>(r.center_x), static_cast<double>(r.center_y),
                    static_cast<double>(r.width), static_cast<double>(r.height),
                    static_cast<double>(r.angle_radians), static_cast<double>(r.score), r.class_id);
    } else {
      std::int32_t box[4];
      float score = 0;
      std::int32_t cls = 0;
      const std::uint8_t* row = wire.data() + 4 + 24 * i;
      std::memcpy(box, row, sizeof box);
      std::memcpy(&score, row + 16, sizeof score);
      std::memcpy(&cls, row + 20, sizeof cls);
      std::snprintf(text, sizeof text, " [x=%d y=%d w=%d h=%d score=%.6f cls=%d]", box[0], box[1],
                    box[2], box[3], static_cast<double>(score), cls);
    }
    out += text;
  }
  return out;
}

// What the public parser of the task makes of a pulled wire tensor: each parsed tensor's shape
// and a digest of its bytes. A parser that refuses the tensor fails the test.
inline std::string parsed_text(DecodeTask task, const simaai::neat::Tensor& wire, int top_k) {
  const auto one = [](const char* name, const simaai::neat::Tensor& t) {
    std::string shape;
    for (const std::int64_t d : t.shape) {
      shape += (shape.empty() ? "" : "x") + std::to_string(d);
    }
    return std::string(" ") + name + "[" + shape + "]=" + bytes_digest(t.copy_dense_bytes_tight());
  };
  switch (task) {
  case DecodeTask::Pose: {
    const auto parsed = simaai::neat::decode_pose_tensor(wire, 0, 0, top_k);
    return "decode_pose:" + one("boxes", parsed.boxes) + one("keypoints", parsed.keypoints);
  }
  case DecodeTask::Segmentation: {
    const auto parsed = simaai::neat::decode_segmentation_tensor(wire, 0, 0, top_k);
    return "decode_segmentation:" + one("boxes", parsed.boxes) + one("masks", parsed.masks);
  }
  case DecodeTask::Oriented: {
    const auto parsed = simaai::neat::decode_obb_tensor(wire);
    return "decode_obb:" + one("geometry", parsed.geometry) + one("scores", parsed.scores) +
           one("class_ids", parsed.class_ids);
  }
  default: {
    const simaai::neat::TensorList parsed = simaai::neat::decode_bbox({wire}, 0, 0, top_k);
    require(parsed.size() == 1U, "decode_bbox gives one tensor a wire");
    return "decode_bbox:" + one("boxes", parsed.front());
  }
  }
}

// Whether every detection of the wire lies in a `width` x `height` frame, to within `slack`
// pixels: an axis-aligned box whole, an oriented box by its centre, a pose's visible keypoints.
inline bool wire_in_frame(DecodeTask task, const std::vector<std::uint8_t>& wire, int top_k,
                          int width, int height, int slack = 4) {
  const std::int32_t count = wire_count(task, wire);
  for (std::int32_t i = 0; i < count; ++i) {
    if (task == DecodeTask::Oriented) {
      simaai::neat::OrientedBoxRecordV1 r;
      std::memcpy(&r, wire.data() + 32 + 32 * i, sizeof r);
      if (!(r.width > 0 && r.height > 0 && r.center_x >= -slack && r.center_y >= -slack &&
            r.center_x <= width + slack && r.center_y <= height + slack)) {
        return false;
      }
      continue;
    }
    std::int32_t box[4];
    std::memcpy(box, wire.data() + 4 + 24 * i, sizeof box);
    if (!(box[2] > 0 && box[3] > 0 && box[0] >= -slack && box[1] >= -slack &&
          box[0] + box[2] <= width + slack && box[1] + box[3] <= height + slack)) {
      return false;
    }
    if (task == DecodeTask::Pose) {
      const std::uint8_t* block = wire.data() + 4 + 24 * top_k + 204 * i;
      for (int k = 0; k < 17; ++k) {
        std::uint32_t xy[2];
        float visibility = 0;
        std::memcpy(xy, block + 12 * k, sizeof xy);
        std::memcpy(&visibility, block + 12 * k + 8, sizeof visibility);
        if (!std::isfinite(visibility) || xy[0] > static_cast<std::uint32_t>(width + slack) ||
            xy[1] > static_cast<std::uint32_t>(height + slack)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace sima_test
