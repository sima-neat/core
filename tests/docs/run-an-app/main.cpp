#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace neat = simaai::neat;

neat::Model::Options yolo_model_options() {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.25f;
  opt.nms_iou_threshold = 0.45f;
  opt.top_k = 100;
  return opt;
}

int main() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png");
  if (bgr.empty()) {
    throw std::runtime_error("failed to read assets/tutorial_sample_image.png");
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  neat::Model model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options());
  neat::Tensor input = neat::Tensor::from_cv_mat(rgb, neat::ImageSpec::PixelFormat::RGB);

  neat::Graph graph("hello_neat_app");
  graph.add(neat::nodes::Input("image"));
  graph.add(model);
  graph.add(neat::nodes::Output("detections"));

  neat::Run run = graph.build();
  run.push("image", neat::TensorList{input});
  run.close_input();
  neat::TensorList outputs = run.pull_tensors("detections", /*timeout_ms=*/2000);
  run.close();

  neat::TensorList decoded = neat::decode_bbox(outputs);
  const neat::Tensor& boxes = decoded.front();
  auto m = boxes.map_read();
  const float* d = static_cast<const float*>(m.data);
  for (int64_t i = 0; i < boxes.shape[0]; ++i) {
    const float* r = d + i * 6;
    const int cls = static_cast<int>(r[5]);
    const char* name = (cls == 0) ? "person" : (cls == 27) ? "tie" : "?";
    std::printf("%-8s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n", name, r[4], r[0], r[1], r[2], r[3]);
  }
  std::printf("[OK] Graph app completed\n");
  return 0;
}
