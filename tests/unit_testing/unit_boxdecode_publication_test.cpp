#include "pipeline/internal/sima/BoxDecodePublication.h"
#include "pipeline/FeatureTypes.h"
#include "pipeline/DetectionTypes.h"
#include <simaai/scheduler/testing/probe.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace k = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace core = simaai::neat::pipeline_internal::sima;
using F = core::BoxDecodeWireFormat;
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T read(const std::byte* p) {
  T result;
  std::memcpy(&result, p, sizeof(T));
  return result;
}
template <class T> void write(std::byte* p, T value) {
  std::memcpy(p, &value, sizeof(T));
}
struct Fixture {
  core::BoxDecodePublication plan;
  std::vector<std::vector<std::byte>> storage;
  std::vector<k::ConstBytes> input;
  std::vector<std::byte> output;
  explicit Fixture(F format, std::span<const k::TensorSpec> specs)
      : plan(core::prepare_boxdecode_publication(format, specs)),
        output(plan.bytes + 16, std::byte{0x5a}) {
    for (auto r :
         std::get<k::KernelInspection>(k::inspect_kernel(plan.executable)).input_requirements())
      storage.emplace_back(r.storage_bytes, std::byte{0});
    for (const auto& bytes : storage)
      input.push_back({bytes.data(), bytes.size()});
  }
  k::Status run() {
    const std::array<k::MutableBytes, 1> out{{{output.data(), plan.bytes}}};
    return probe::execute_host(plan.executable, input, out);
  }
  void check_guard() const {
    require(std::all_of(output.begin() + plan.bytes, output.end(),
                        [](auto b) { return b == std::byte{0x5a}; }),
            "wire output guard changed");
  }
};
void boxes(F format, std::int64_t capacity) {
  std::vector<k::TensorSpec> specs{{{capacity, 4}, k::DType::Int32},
                                   {{capacity}, k::DType::Float32},
                                   {{capacity}, k::DType::Int32},
                                   {{1}, k::DType::Int32}};
  if (format == F::Pose)
    specs.push_back({{capacity, 17, 3}, k::DType::Float32});
  if (format == F::Masks)
    specs.push_back({{capacity, 3, 5}, k::DType::UInt8});
  Fixture fixture(format, specs);
  for (std::int64_t count : {capacity, std::int64_t{1}, std::int64_t{0}}) {
    write(fixture.storage[3].data(), static_cast<std::int32_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
      for (int q = 0; q < 4; ++q)
        write(fixture.storage[0].data() + i * 16 + q * 4, std::int32_t(10 * i + q));
      write(fixture.storage[1].data() + i * 4, .75f);
      write(fixture.storage[2].data() + i * 4, std::int32_t(i + 2));
      if (format == F::Pose)
        for (int point = 0; point < 17; ++point) {
          auto* p = fixture.storage[4].data() + i * 204 + point * 12;
          write(p, float(point + 1));
          write(p + 4, float(point + 2));
          write(p + 8, .5f);
        }
      if (format == F::Masks)
        std::memset(fixture.storage[4].data() + i * 15, 0xa5, 15);
    }
    require(!fixture.run(), "valid box publication rejected");
    require(read<std::int32_t>(fixture.output.data()) == count, "wire count mismatch");
    for (std::int64_t i = 0; i < count; ++i) {
      auto* row = fixture.output.data() + 4 + i * 24;
      for (int q = 0; q < 4; ++q)
        require(read<std::int32_t>(row + q * 4) == 10 * i + q, "bbox coordinate mismatch");
      require(read<float>(row + 16) == .75f && read<std::int32_t>(row + 20) == i + 2,
              "bbox score/class mismatch");
      auto* extra = fixture.output.data() + 4 + capacity * 24;
      if (format == F::Pose)
        for (int point = 0; point < 17; ++point) {
          const auto* p = extra + i * 204 + point * 12;
          require(read<std::uint32_t>(p) == unsigned(point + 1) &&
                      read<std::uint32_t>(p + 4) == unsigned(point + 2) &&
                      read<float>(p + 8) == .5f,
                  "pose wire encoding mismatch");
        }
      if (format == F::Masks)
        require(std::all_of(extra + i * 15, extra + (i + 1) * 15,
                            [](auto b) { return b == std::byte{0xa5}; }),
                "mask plane mismatch");
    }
    require(std::all_of(fixture.output.begin() + 4 + count * 24,
                        fixture.output.begin() + 4 + capacity * 24,
                        [](auto b) { return b == std::byte{}; }),
            "stale box tail");
    const auto extra_row = format == F::Pose ? 204 : format == F::Masks ? 15 : 0;
    require(std::all_of(fixture.output.begin() + 4 + capacity * 24 + count * extra_row,
                        fixture.output.begin() + fixture.plan.bytes,
                        [](auto b) { return b == std::byte{}; }),
            "stale auxiliary tail");
    fixture.check_guard();
  }
  const auto before = fixture.output;
  write(fixture.storage[3].data(), std::int32_t(capacity + 1));
  require(bool(fixture.run()) && fixture.output == before, "invalid count modified wire");
  write(fixture.storage[3].data(), std::int32_t{0});
  fixture.input[0].size_bytes--;
  require(bool(fixture.run()) && fixture.output == before, "short input modified wire");
}
void masks_pose(int points) {
  constexpr std::int64_t capacity = 3;
  constexpr std::size_t mask_bytes = 160 * 160;
  const std::array<k::TensorSpec, 6> specs{{{{capacity, 4}, k::DType::Int32},
                                            {{capacity}, k::DType::Float32},
                                            {{capacity}, k::DType::Int32},
                                            {{1}, k::DType::Int32},
                                            {{capacity, 160, 160}, k::DType::UInt8},
                                            {{capacity, points, 3}, k::DType::Float32}}};
  Fixture f(F::MasksPose, specs);
  require(f.plan.bytes == 4 + capacity * (24 + mask_bytes + 204), "combined wire extent");
  for (const int count : {3, 1, 0}) {
    write(f.storage[3].data(), std::int32_t(count));
    for (int row = 0; row < count; ++row) {
      write(f.storage[0].data() + row * 16, std::int32_t(10 + row));
      write(f.storage[1].data() + row * 4, .25f * (row + 1));
      write(f.storage[2].data() + row * 4, std::int32_t(7 - row));
      std::memset(f.storage[4].data() + row * mask_bytes, 0x10 + row, mask_bytes);
      for (int point = 0; point < points; ++point) {
        auto* to = f.storage[5].data() + (row * points + point) * 12;
        // Simulate a class-gated zero row. Publisher must not filter/reorder it.
        write(to, row == 1 ? 0.f : float(100 * row + point));
        write(to + 4, row == 1 ? 0.f : float(200 * row + point));
        write(to + 8, row == 1 ? 0.f : .75f);
      }
    }
    require(!f.run(), "combined publication rejected");
    const auto* masks = f.output.data() + 4 + capacity * 24;
    const auto* poses = masks + capacity * mask_bytes;
    require(read<std::int32_t>(f.output.data()) == count, "combined count changed");
    for (int row = 0; row < capacity; ++row) {
      const auto* box = f.output.data() + 4 + row * 24;
      if (row < count) {
        require(read<std::int32_t>(box) == 10 + row && read<std::int32_t>(box + 20) == 7 - row,
                "combined survivor row association changed");
        require(std::all_of(masks + row * mask_bytes, masks + (row + 1) * mask_bytes,
                            [row](auto byte) { return byte == std::byte(0x10 + row); }),
                "combined mask survivor changed");
        for (int point = 0; point < points; ++point) {
          const auto* from = poses + row * 204 + point * 12;
          require(read<std::uint32_t>(from) == unsigned(row == 1 ? 0 : 100 * row + point) &&
                      read<std::uint32_t>(from + 4) == unsigned(row == 1 ? 0 : 200 * row + point) &&
                      read<float>(from + 8) == (row == 1 ? 0.f : .75f),
                  "combined pose survivor changed");
        }
      } else {
        require(std::all_of(box, box + 24, [](auto byte) { return byte == std::byte{}; }),
                "combined box tail stale");
        require(std::all_of(masks + row * mask_bytes, masks + (row + 1) * mask_bytes,
                            [](auto byte) { return byte == std::byte{}; }),
                "combined mask tail stale");
      }
      const auto* zero_begin = poses + row * 204 + (row < count ? points * 12 : 0);
      require(std::all_of(zero_begin, poses + (row + 1) * 204,
                          [](auto byte) { return byte == std::byte{}; }),
              "combined pose padding or tail stale");
    }
    f.check_guard();
  }
  const auto before = f.output;
  write(f.storage[3].data(), std::int32_t{1});
  for (float bad : {-1.f, std::numeric_limits<float>::infinity(), 4294967296.f}) {
    write(f.storage[5].data(), bad);
    require(bool(f.run()) && f.output == before, "invalid combined coordinate modified wire");
  }
  write(f.storage[3].data(), std::int32_t{0});
  f.input[5] = {f.output.data(), f.storage[5].size()};
  require(bool(f.run()) && f.output == before, "combined alias modified wire");
  auto malformed = specs;
  malformed[5] = {{capacity, 18, 3}, k::DType::Float32};
  bool rejected = false;
  try {
    (void)core::prepare_boxdecode_publication(F::MasksPose, malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "oversized combined point count admitted");
}
void rcnn() {
  const std::array<k::TensorSpec, 5> specs{{{{8, 4}, k::DType::Float32},
                                            {{8}, k::DType::Float32},
                                            {{8}, k::DType::Int32},
                                            {{1}, k::DType::Int32},
                                            {{8}, k::DType::Int32}}};
  Fixture f(F::RcnnProposals, specs);
  require(f.plan.bytes == 4 * (4 + 2 * 12), "RCNN four-group wire extent");
  write(f.storage[3].data(), std::int32_t{4});
  for (int i = 0; i < 4; ++i) {
    const std::array<float, 4> xywh{1.5f, 2.25f, 4.f, 8.f};
    std::memcpy(f.storage[0].data() + i * 16, xywh.data(), 16);
    write(f.storage[1].data() + i * 4, .75f);
    write(f.storage[4].data() + i * 4, std::int32_t(3 - i));
  }
  require(!f.run(), "RCNN publication rejected");
  // Independent IEEE-half bit patterns: score,.0,x1,y1,x2,y2.
  const std::array<std::uint16_t, 6> expected{0x3a00, 0, 0x3e00, 0x4080, 0x4580, 0x4920};
  for (int level = 0; level < 4; ++level) {
    const auto* group = f.output.data() + level * 28;
    require(read<std::int32_t>(group) == 1, "RCNN group count");
    for (int field = 0; field < 6; ++field)
      require(read<std::uint16_t>(group + 4 + field * 2) == expected[field], "RCNN half wire");
    for (int byte = 16; byte < 28; ++byte)
      require(group[byte] == std::byte{}, "RCNN unused proposal tail");
  }
  f.check_guard();
  const auto before = f.output;
  write(f.storage[4].data(), std::int32_t{4});
  require(bool(f.run()) && before == f.output, "invalid RCNN level changed output");
  write(f.storage[3].data(), std::int32_t{0});
  require(!f.run(), "empty RCNN output rejected");
  require(std::all_of(f.output.begin(), f.output.begin() + f.plan.bytes,
                      [](auto x) { return x == std::byte{}; }),
          "stale RCNN output");
}
void features(F format, k::DType dtype, std::int64_t dim) {
  constexpr std::int64_t capacity = 3;
  const std::array<k::TensorSpec, 4> specs{{{{capacity, 2}, k::DType::Float32},
                                            {{capacity}, k::DType::Float32},
                                            {{capacity, dim}, dtype},
                                            {{1}, k::DType::Int32}}};
  Fixture f(format, specs);
  const auto scalar = dtype == k::DType::Float32 ? 4 : dtype == k::DType::BFloat16 ? 2 : 1;
  for (int count : {3, 1, 0}) {
    write(f.storage[3].data(), std::int32_t(count));
    for (int i = 0; i < count; ++i) {
      write(f.storage[0].data() + i * 8, float(i + 5));
      write(f.storage[0].data() + i * 8 + 4, float(i + 8));
      write(f.storage[1].data() + i * 4, .125f);
      std::memset(f.storage[2].data() + i * dim * scalar, 0x42, dim * scalar);
    }
    require(!f.run(), "feature publication rejected");
    if (format == F::FeaturePointsV1) {
      const auto h = read<simaai::neat::FeaturePointsHeaderV1>(f.output.data());
      require(h.magic == simaai::neat::kFeaturePointsMagicV1 && h.version == 1 &&
                  h.header_bytes == 68 && h.count == unsigned(count) && h.capacity == capacity &&
                  h.total_bytes == f.plan.bytes && h.descriptor_dim == dim,
              "feature header mismatch");
      require(h.keypoints_offset == 80 && h.scores_offset == 112 && h.descriptors_offset == 128,
              "feature aligned offsets changed");
      for (int i = 0; i < count; ++i) {
        require(read<float>(f.output.data() + h.keypoints_offset + i * 8) == i + 5 &&
                    read<float>(f.output.data() + h.scores_offset + i * 4) == .125f,
                "feature scalar mismatch");
        auto* row = f.output.data() + h.descriptors_offset + i * h.descriptor_stride;
        require(std::all_of(row, row + dim * scalar, [](auto b) { return b == std::byte{0x42}; }),
                "descriptor bits changed");
        require(std::all_of(row + dim * scalar, row + h.descriptor_stride,
                            [](auto b) { return b == std::byte{}; }),
                "descriptor row padding stale");
      }
      require(std::all_of(f.output.begin() + h.descriptors_offset + count * h.descriptor_stride,
                          f.output.begin() + f.plan.bytes, [](auto b) { return b == std::byte{}; }),
              "feature tail stale");
    } else {
      require(read<std::int32_t>(f.output.data()) == count, "legacy count mismatch");
      for (int i = 0; i < count; ++i) {
        auto* row = f.output.data() + 4 + i * 264;
        require(read<std::uint16_t>(row) == i + 5 && read<std::uint16_t>(row + 2) == i + 8 &&
                    read<float>(row + 4) == .125f,
                "legacy scalar mismatch");
        require(std::all_of(row + 8, row + 264, [](auto b) { return b == std::byte{0x42}; }),
                "legacy descriptor changed");
      }
      require(std::all_of(f.output.begin() + 4 + count * 264, f.output.begin() + f.plan.bytes,
                          [](auto b) { return b == std::byte{}; }),
              "legacy tail stale");
    }
    f.check_guard();
  }
}
void oriented() {
  const std::array<k::TensorSpec, 4> specs{{{{3, 5}, k::DType::Float32},
                                            {{3}, k::DType::Float32},
                                            {{3}, k::DType::Int32},
                                            {{1}, k::DType::Int32}}};
  Fixture f(F::OrientedBoxesV1, specs);
  require(f.plan.bytes == 128, "OBB wire extent must be header32 + capacity*32");
  for (int count : {3, 1, 0}) {
    write(f.storage[3].data(), std::int32_t(count));
    for (int i = 0; i < count; ++i) {
      const std::array<float, 5> values{-4.f + i, 12.f, -32.f, 48.f, .4f};
      std::memcpy(f.storage[0].data() + i * 20, values.data(), 20);
      write(f.storage[1].data() + i * 4, .5f);
      write(f.storage[2].data() + i * 4, std::int32_t(9 + i));
    }
    require(!f.run(), "valid OBB publication rejected");
    require(read<std::uint32_t>(f.output.data()) == 0x3156424f &&
                read<std::uint16_t>(f.output.data() + 4) == 1 &&
                read<std::uint16_t>(f.output.data() + 6) == 32 &&
                read<std::uint32_t>(f.output.data() + 8) == 128 &&
                read<std::uint32_t>(f.output.data() + 12) == unsigned(count) &&
                read<std::uint32_t>(f.output.data() + 16) == 3 &&
                read<std::uint32_t>(f.output.data() + 20) == 32 &&
                read<std::uint32_t>(f.output.data() + 24) == 32 &&
                read<std::uint32_t>(f.output.data() + 28) == 0,
            "OBB independent header byte offsets changed");
    for (int i = 0; i < count; ++i) {
      const auto* row = f.output.data() + 32 + i * 32;
      require(read<float>(row) == -4.f + i && read<float>(row + 8) == -32.f &&
                  read<float>(row + 16) == .4f && read<float>(row + 20) == .5f &&
                  read<std::int32_t>(row + 24) == 9 + i && read<std::uint32_t>(row + 28) == 0,
              "OBB signed geometry, angle, score or integer class altered");
    }
    require(std::all_of(f.output.begin() + 32 + count * 32, f.output.begin() + f.plan.bytes,
                        [](auto b) { return b == std::byte{}; }),
            "OBB tails not cleared");
    f.check_guard();
  }
  for (int port : {0, 1, 2, 3}) {
    write(f.storage[3].data(), std::int32_t(1));
    write(f.storage[0].data(), 1.f);
    write(f.storage[1].data(), .5f);
    write(f.storage[2].data(), std::int32_t(1));
    if (port < 2)
      write(f.storage[port].data(), std::numeric_limits<float>::quiet_NaN());
    else
      write(f.storage[port].data(), std::int32_t(-1));
    const auto before = f.output;
    require(bool(f.run()), "malformed OBB result accepted");
    require(f.output == before, "failed OBB publication changed caller output");
  }
}
} // namespace
int main() {
  try {
    oriented();
    rcnn();
    for (int points : {1, 13, 17})
      masks_pose(points);
    for (auto format : {F::Boxes, F::Pose, F::Masks})
      for (auto capacity : {1, 3, 31})
        boxes(format, capacity);
    for (auto dtype : {k::DType::Int8, k::DType::BFloat16, k::DType::Float32})
      for (auto dim : {1, 17, 256})
        features(F::FeaturePointsV1, dtype, dim);
    features(F::LegacyFeaturePointsV0, k::DType::Int8, 256);
    std::cout << "BoxDecode publication: formats, padding, survivor association and negative "
                 "controls passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
