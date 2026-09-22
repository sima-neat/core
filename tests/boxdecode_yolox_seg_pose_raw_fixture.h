#pragma once

#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <unistd.h>

namespace yolox_test {

// Two detections with known masks; only class zero carries keypoints.
class RawHeads {
public:
  static constexpr int capacity = 20;
  static constexpr int mask_bytes = 160 * 160;
  static constexpr int pose_bytes = 17 * 12;
  static constexpr int output_bytes = 4 + capacity * (24 + mask_bytes + pose_bytes);

  RawHeads(float sx = 2.0f, float sy = 1.125f, float ox = 0.0f, float oy = 0.0f,
           bool threaded = false)
      : sx_(sx), sy_(sy), ox_(ox), oy_(oy), output_(output_bytes) {
    library_ = dlopen("libsimaai_genboxdecode.so", RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      throw std::runtime_error(std::string("load installed BoxDecode backend: ") + dlerror());
    }
    try {
      configure_ = symbol<Configure>("configure");
      run_ = symbol<Run>("simaai_boxdecode_run_v1");
      destroy_ = symbol<Destroy>("simaai_boxdecode_destroy_v1");
      Dl_info info{};
      require(dladdr(reinterpret_cast<void*>(run_), &info) != 0, "backend path must resolve");
      std::cerr << "BoxDecode runtime: " << info.dli_fname << '\n';
      std::array<std::size_t, 13> offsets{};
      for (int t = 0; t < 13; ++t) {
        offsets[t] = input_.size();
        input_.resize(input_.size() + widths_[t] * widths_[t] * stored_[t], 0.0f);
        if (t >= 3 && t < 6) {
          std::fill(input_.begin() + offsets[t], input_.end(), -8.0f);
        }
      }
      auto value = [&](int tensor, int cell, int channel) -> float& {
        return input_[offsets[tensor] + (cell * widths_[tensor] + cell) * stored_[tensor] +
                      channel];
      };
      for (int cls = 0; cls < 2; ++cls) {
        const int cell = cls == 0 ? 20 : 40;
        value(0, cell, 2) = std::log(2.0f);
        value(0, cell, 3) = std::log(2.0f);
        value(3, cell, 0) = 8.0f;
        value(3, cell, 1 + cls) = 8.0f;
        for (int ch = 0; ch < 32; ++ch)
          value(6, cell, ch) = 1.0f;
        for (int point = 0; point < 13; ++point)
          value(9, cell, point * 3 + 2) = 8.0f;
      }
      std::fill(input_.begin() + offsets[12], input_.end(), 1.0f);
      std::ostringstream json;
      json << R"({"decode_type":"yolox-seg-pose","num_in_tensor":13,"num_classes":36,)"
           << R"("topk":20,"detection_threshold":0.5,"nms_iou_threshold":0.5,)"
           << R"("model_width":640,"model_height":640,"original_width":1280,"original_height":720,)"
           << R"("pose_classes":[0],"multicore_acceleration":)" << (threaded ? "true" : "false")
           << R"(,"preproc_affine":{"valid":true,"scale_x":)" << sx << R"(,"scale_y":)" << sy
           << R"(,"offset_x":)" << ox << R"(,"offset_y":)" << oy << '}';
      array(json, "input_width", widths_);
      array(json, "input_height", widths_);
      array(json, "input_depth", stored_);
      array(json, "slice_width", widths_);
      array(json, "slice_height", widths_);
      array(json, "slice_depth", logical_);
      const std::array<int, 13> ones{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      array(json, "dq_scale", ones);
      array(json, "dq_zp", std::array<int, 13>{});
      array(json, "tensor_storage_kind", ones);
      json << R"(,"data_type":[)";
      for (int t = 0; t < 13; ++t)
        json << (t ? "," : "") << "\"FP32\"";
      json << "]}";
      char path[] = "/tmp/neat-yolox-raw-XXXXXX";
      const int fd = mkstemp(path);
      require(fd >= 0, "create temporary decoder config");
      close(fd);
      {
        std::ofstream config(path);
        config << json.str();
      }
      const int rc = configure_(1, path);
      std::filesystem::remove(path);
      require(rc >= 0, "configure raw-head BoxDecode fixture");
    } catch (...) {
      if (destroy_)
        destroy_(1);
      dlclose(library_);
      throw;
    }
  }
  RawHeads(const RawHeads&) = delete;
  RawHeads& operator=(const RawHeads&) = delete;
  ~RawHeads() {
    destroy_(1);
    dlclose(library_);
  }

  const std::vector<std::uint8_t>& decode() {
    std::fill(output_.begin(), output_.end(), 0);
    require(run_(1, input_.data(), static_cast<int>(input_.size() * sizeof(float)), output_.data(),
                 output_bytes) == 0,
            "raw-head decoding must succeed");
    return output_;
  }

  void verify_reference() const {
    require(read<int>(0) == 2, "exactly two raw-head detections");
    const float confidence = std::pow(1.0f / (1.0f + std::exp(-8.0f)), 2);
    std::array<bool, 2> seen{};
    for (int row = 0; row < 2; ++row) {
      const int box = 4 + row * 24;
      const int cls = read<int>(box + 20);
      require(cls >= 0 && cls < 2 && !seen[cls], "each expected class occurs once");
      seen[cls] = true;
      const int cell = cls == 0 ? 20 : 40;
      require(read<int>(box) == std::lround((cell * 8 - 8) * sx_ + ox_), "reference box x");
      require(read<int>(box + 4) == std::lround((cell * 8 - 8) * sy_ + oy_), "reference box y");
      require(read<int>(box + 8) == std::lround(16 * sx_), "reference box width");
      require(read<int>(box + 12) == std::lround(16 * sy_), "reference box height");
      require(std::abs(read<float>(box + 16) - confidence) < 1e-5f, "reference score");
      const int mask = 4 + capacity * 24 + row * mask_bytes;
      for (int y = 0; y < 160; ++y) {
        for (int x = 0; x < 160; ++x) {
          const bool inside =
              x >= cell * 2 - 2 && x < cell * 2 + 2 && y >= cell * 2 - 2 && y < cell * 2 + 2;
          require(output_[mask + y * 160 + x] == (inside ? 255 : 0), "reference mask pixel");
        }
      }
      const int pose = 4 + capacity * (24 + mask_bytes) + row * pose_bytes;
      for (int point = 0; point < 17; ++point) {
        const bool valid = cls == 0 && point < 13;
        require(read<unsigned>(pose + point * 12) ==
                    (valid ? std::lround(cell * 8 * sx_ + ox_) : 0),
                "reference keypoint x");
        require(read<unsigned>(pose + point * 12 + 4) ==
                    (valid ? std::lround(cell * 8 * sy_ + oy_) : 0),
                "reference keypoint y");
        const float expected = valid ? 1.0f / (1.0f + std::exp(-8.0f)) : 0.0f;
        require(std::abs(read<float>(pose + point * 12 + 8) - expected) < 1e-5f,
                "reference keypoint visibility and class gating");
      }
    }
  }

private:
  using Configure = int (*)(int, const char*);
  using Run = int (*)(int, void*, int, void*, int);
  using Destroy = int (*)(int);
  template <class T> T symbol(const char* name) {
    void* address = dlsym(library_, name);
    require(address != nullptr, std::string("missing backend symbol: ") + name);
    return reinterpret_cast<T>(address);
  }
  template <class T> T read(std::size_t offset) const {
    T value;
    std::memcpy(&value, output_.data() + offset, sizeof(value));
    return value;
  }
  static void array(std::ostream& out, const char* name, const std::array<int, 13>& values) {
    out << ",\"" << name << "\":[";
    for (int t = 0; t < 13; ++t)
      out << (t ? "," : "") << values[t];
    out << ']';
  }
  const std::array<int, 13> widths_{80, 40, 20, 80, 40, 20, 80, 40, 20, 80, 40, 20, 160};
  const std::array<int, 13> stored_{16, 16, 16, 48, 48, 48, 32, 32, 32, 48, 48, 48, 32};
  const std::array<int, 13> logical_{4, 4, 4, 37, 37, 37, 32, 32, 32, 39, 39, 39, 32};
  float sx_, sy_, ox_, oy_;
  void* library_ = nullptr;
  Configure configure_ = nullptr;
  Run run_ = nullptr;
  Destroy destroy_ = nullptr;
  std::vector<float> input_;
  std::vector<std::uint8_t> output_;
};

} // namespace yolox_test
