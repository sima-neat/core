#pragma once
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
/**
 * Shared fixture for the yolox-seg-pose payload decode benchmarks.
 *
 * Each retained detection carries a 160x160 mask plane and a 204-byte keypoint
 * record, so a frame moves roughly top_k * 25.8 KB through decode_segmentation_pose.
 * That copy is the per-frame Core cost this measures, and it needs no model: the
 * payload is synthesized, so the workload is exact and repeatable.
 *
 * One scenario per binary - the perf matrix runner parses a single JSON document
 * from each executable's stdout.
 *
 * Baseline ceilings under tests/perf/baselines/v2/modalix_default were set from repeated
 * x86 container runs (representative p50 ~0.021 ms, heavy p50 ~0.119 ms, both stable to
 * within a few percent across runs) and then given roughly 8x headroom for slower memory
 * bandwidth, because they could not be measured on Modalix from a development host. They
 * are deliberately loose: tighten them from a real Modalix run before relying on this to
 * catch a small regression.
 */
#include "perf_metrics_common.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/TensorCore.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

struct RawBox {
  int32_t x = 0;
  int32_t y = 0;
  int32_t w = 0;
  int32_t h = 0;
  float score = 0.0f;
  int32_t cls = 0;
};

struct RawPosePoint {
  uint32_t x = 0;
  uint32_t y = 0;
  float visible = 0.0f;
};

struct RawPoseOut {
  RawPosePoint points[static_cast<std::size_t>(kDecodedPoseKeypoints)];
};

static_assert(sizeof(RawBox) == 24);
static_assert(sizeof(RawPoseOut) == 204);

const std::size_t kMaskBytes =
    static_cast<std::size_t>(kDecodedMaskWidth) * static_cast<std::size_t>(kDecodedMaskHeight);

std::vector<uint8_t> make_payload(uint32_t count, std::size_t capacity) {
  std::vector<uint8_t> out(sizeof(uint32_t) +
                           capacity * (sizeof(RawBox) + kMaskBytes + sizeof(RawPoseOut)));
  std::memcpy(out.data(), &count, sizeof(count));

  std::vector<RawBox> boxes(capacity);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    boxes[slot] = RawBox{.x = static_cast<int32_t>(10 + slot),
                         .y = static_cast<int32_t>(20 + slot),
                         .w = 30,
                         .h = 40,
                         .score = 0.5f,
                         .cls = static_cast<int32_t>(slot % 36)};
  }
  std::memcpy(out.data() + sizeof(uint32_t), boxes.data(), capacity * sizeof(RawBox));

  const std::size_t mask_base = sizeof(uint32_t) + capacity * sizeof(RawBox);
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    std::memset(out.data() + mask_base + slot * kMaskBytes, static_cast<int>(slot % 251 + 1),
                kMaskBytes);
  }

  const std::size_t pose_base = mask_base + capacity * kMaskBytes;
  for (std::size_t slot = 0; slot < capacity; ++slot) {
    RawPoseOut pose{};
    for (std::size_t point = 0; point < static_cast<std::size_t>(kDecodedPoseKeypoints); ++point) {
      pose.points[point].x = static_cast<uint32_t>(1000 * (slot + 1) + point);
      pose.points[point].y = static_cast<uint32_t>(2000 * (slot + 1) + point);
      pose.points[point].visible = 0.01f * static_cast<float>(point + 1);
    }
    std::memcpy(out.data() + pose_base + slot * sizeof(RawPoseOut), &pose, sizeof(pose));
  }
  return out;
}

Tensor make_tensor(const std::vector<uint8_t>& payload) {
  auto storage = make_cpu_owned_storage(payload.size());
  Mapping dst = storage->map(MapMode::Write);
  std::memcpy(dst.data, payload.data(), payload.size());
  Tensor t;
  t.storage = std::move(storage);
  t.device = {DeviceType::CPU, 0};
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::Unknown;
  t.shape = {static_cast<int64_t>(payload.size())};
  tag_detection_format(t, kDetectionFormatBboxSegmentationPose);
  return t;
}

struct Scenario {
  const char* scenario_id;
  uint32_t detections;
  std::size_t capacity;
};

inline void run_scenario(const Scenario& scenario, int warmup, int iterations) {
  const Tensor tensor = make_tensor(make_payload(scenario.detections, scenario.capacity));

  // Warm up so first-touch page faults on the output allocations are not counted.
  for (int i = 0; i < warmup; ++i) {
    const auto result = decode_segmentation_pose(TensorList{tensor}).front();
    if (result.boxes.shape[0] != static_cast<int64_t>(scenario.detections)) {
      throw std::runtime_error(std::string(scenario.scenario_id) + ": warmup decoded " +
                               std::to_string(result.boxes.shape[0]) + " boxes, expected " +
                               std::to_string(scenario.detections));
    }
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  const auto wall_start = sima_perf::Clock::now();
  for (int i = 0; i < iterations; ++i) {
    const auto iter_start = sima_perf::Clock::now();
    const auto result = decode_segmentation_pose(TensorList{tensor}).front();
    const auto iter_end = sima_perf::Clock::now();
    samples.push_back(sima_perf::elapsed_ms(iter_start, iter_end));
    // Touch the result so the copies cannot be optimized away.
    if (result.masks.shape[0] != static_cast<int64_t>(scenario.detections)) {
      throw std::runtime_error(std::string(scenario.scenario_id) + ": mask row count drifted");
    }
  }
  const auto wall_end = sima_perf::Clock::now();

  sima_perf::PerfMetrics metrics;
  const double wall_seconds = sima_perf::elapsed_seconds(wall_start, wall_end);
  metrics.throughput = wall_seconds > 0.0 ? static_cast<double>(iterations) / wall_seconds : 0.0;
  metrics.p50 = sima_perf::percentile(samples, 50.0);
  metrics.p95 = sima_perf::percentile(samples, 95.0);
  metrics.startup = 0.0;
  // getrusage high-water mark over the whole run, so a leak in the decode path shows as
  // growth rather than a single sample.
  metrics.rss_peak_kb = sima_perf::rss_peak_kb();
  sima_perf::emit_metrics_json(scenario.scenario_id, iterations, metrics, "sync");
}

inline int run_payload_scenario_main(const Scenario& scenario) {
  try {
    const int warmup = sima_perf::env_int("SIMA_SEG_POSE_PERF_WARMUP", 50);
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 500);
    run_scenario(scenario, warmup, iterations);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << scenario.scenario_id << " failed: " << error.what() << "\n";
    return 1;
  }
}

} // namespace
