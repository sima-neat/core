#pragma once

#include "model/Model.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace graph_phase3_test {

std::filesystem::path repo_root();

std::filesystem::path resolve_yolov8n_variant_or_throw(const std::filesystem::path& root,
                                                       std::string_view filename);
std::vector<std::filesystem::path>
resolve_yolov8n_variants_or_throw(const std::filesystem::path& root);

std::filesystem::path resolve_resnet50_or_throw(const std::filesystem::path& root);
std::filesystem::path resolve_mnist_or_throw(const std::filesystem::path& root);

#if defined(SIMA_WITH_OPENCV)
cv::Mat load_people_bgr_or_throw(const std::filesystem::path& root);
cv::Mat load_goldfish_bgr_or_throw(const std::filesystem::path& root);
#endif

simaai::neat::Tensor make_rgb_tensor(int width, int height, std::uint8_t fill = 0x5a);
simaai::neat::Sample make_tensor_sample(int frame_id, std::string stream_id, int width = 32,
                                        int height = 24);

simaai::neat::Model::Options yolo_image_bgr_to_rgb_bf16_options();
simaai::neat::Model::Options resnet_imagenet_bgr_options();
simaai::neat::Model::Options mnist_gray8_options();

void require_nonempty_tensor_output(const simaai::neat::TensorList& out, std::string_view where);
void require_sample_tensor_output(const simaai::neat::Sample& sample, std::string_view where);

} // namespace graph_phase3_test
