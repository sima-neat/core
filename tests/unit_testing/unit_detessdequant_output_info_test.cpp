#include "pipeline/internal/StageConfig.h"
#include "nodes/sima/DetessDequant.h"

#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>

int main() {
  try {
    nlohmann::json cfg;
    cfg["simaai__params"] = {
        {"num_in_tensor", 2},     {"batch_size", 2},       {"input_width", {4, 2}},
        {"input_height", {3, 5}}, {"input_depth", {6, 7}}, {"output_format", "NCHW"},
        {"fp16_out_en", 1},
    };

    simaai::neat::DetessDequantOptions opt;
    opt.num_buffers_model = 4;
    opt.num_buffers = 4;
    opt.num_buffers_locked = true;
    opt.config_json = cfg;

    auto node = simaai::neat::nodes::DetessDequant(opt);
    simaai::neat::NodeGroup group({node});

    const auto info = simaai::neat::stages::read_detessdequant_output_info(group, false);
    require(info.outputs.size() == 2, "detess outputs size mismatch");
    require(info.dtype == simaai::neat::TensorDType::BFloat16, "detess dtype mismatch");
    require(info.layout == simaai::neat::TensorLayout::CHW, "detess layout mismatch");
    require(info.outputs[0].shape == std::vector<int64_t>({6, 3, 4}), "detess shape0 mismatch");
    require(info.outputs[0].byte_offset == 0, "detess offset0 mismatch");
    require(info.outputs[1].shape == std::vector<int64_t>({7, 5, 2}), "detess shape1 mismatch");
    require(info.outputs[1].byte_offset == 144, "detess offset1 mismatch");

    const auto info_batch = simaai::neat::stages::read_detessdequant_output_info(group, true);
    require(info_batch.outputs.size() == 2, "detess batch outputs size mismatch");
    require(info_batch.outputs[0].shape == std::vector<int64_t>({2, 6, 3, 4}),
            "detess batch shape0 mismatch");
    require(info_batch.outputs[0].byte_offset == 0, "detess batch offset0 mismatch");
    require(info_batch.outputs[1].shape == std::vector<int64_t>({2, 7, 5, 2}),
            "detess batch shape1 mismatch");
    require(info_batch.outputs[1].byte_offset == 288, "detess batch offset1 mismatch");

    std::cout << "[OK] unit_detessdequant_output_info_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
