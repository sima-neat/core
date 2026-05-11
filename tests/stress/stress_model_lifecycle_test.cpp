#include "asset_utils.h"
#include "model/Model.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(10, std::min(value, 5000));
}

std::string resolve_model_pack_for_stress() {
  const char* env_tar = std::getenv("SIMA_STRESS_MODEL_TAR");
  if (env_tar && *env_tar && std::filesystem::exists(env_tar)) {
    return std::string(env_tar);
  }
  const char* generic = std::getenv("SIMA_MODEL_TAR");
  if (generic && *generic && std::filesystem::exists(generic)) {
    return std::string(generic);
  }

  const std::filesystem::path root = std::filesystem::current_path();
  std::string tar = sima_test::resolve_resnet50_tar_local_only(root);
  if (!tar.empty()) {
    return tar;
  }
  tar = sima_test::resolve_yolov8s_tar_local_first(root, true);
  return tar;
}

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

simaai::neat::Tensor make_tensor_for_model(const simaai::neat::Model& model) {
  const simaai::neat::InputOptions in_opt = model.input_appsrc_options(true);
  const int width = in_opt.width > 0 ? in_opt.width : in_opt.max_width;
  const int height = in_opt.height > 0 ? in_opt.height : in_opt.max_height;
  int depth = in_opt.depth > 0 ? in_opt.depth : in_opt.max_depth;
  if (depth <= 0)
    depth = 3;

  if (width <= 0 || height <= 0 || depth <= 0) {
    throw std::runtime_error("model input dimensions are not valid");
  }

  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                            static_cast<std::size_t>(depth) * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0, map.size_bytes);
  }

  simaai::neat::Tensor input;
  input.storage = storage;
  input.dtype = simaai::neat::TensorDType::Float32;
  input.layout = simaai::neat::TensorLayout::HWC;
  input.shape = {height, width, depth};
  input.strides_bytes = contiguous_strides_bytes(input.shape, static_cast<int64_t>(sizeof(float)));
  input.device = {simaai::neat::DeviceType::CPU, 0};
  input.read_only = true;
  return input;
}

bool runtime_is_unavailable(const std::string& msg) {
  if (is_dispatcher_unavailable(msg)) {
    return true;
  }
  return msg.find("dispatcher unavailable") != std::string::npos ||
         msg.find("Failed to create") != std::string::npos ||
         msg.find("No such element") != std::string::npos ||
         msg.find("could not load") != std::string::npos;
}

} // namespace

RUN_TEST("stress_model_lifecycle_test", [] {
  const std::string tar = resolve_model_pack_for_stress();
  if (tar.empty()) {
    skip_long_test_exception(
        "no local model pack found (set SIMA_MODEL_TAR or SIMA_STRESS_MODEL_TAR to run model "
        "stress)");
  }

  const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 100));
  const bool run_build = env_flag("SIMA_STRESS_MODEL_BUILD", false);

  try {
    for (int i = 0; i < iters; ++i) {
      simaai::neat::Model model(tar);
      (void)model.input_spec();
      (void)model.output_spec();
      (void)model.metadata();
      (void)model.session();
      (void)model.fragment(simaai::neat::Model::Stage::Inference);

      if (run_build) {
        simaai::neat::Tensor input = make_tensor_for_model(model);
        auto runner = model.build(simaai::neat::TensorList{input});
        runner.close();
      }
    }
  } catch (const std::exception& e) {
    if (runtime_is_unavailable(e.what())) {
      skip_long_test_exception(std::string("runtime unavailable for model stress: ") + e.what());
    }
    throw;
  }
});
