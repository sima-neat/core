/**
 * Canonical Model -> Preprocess -> Infer -> BoxDecode pose integration smoke.
 * The MPK is the only source of model contracts. Unlike the retired fixture,
 * this test never loads archive plugin JSON or assembles ProcessCVU pipelines.
 * Numerical decoder oracles live in Kernels; this test checks public payload
 * alignment, repeated async delivery and nonempty image-driven pose output.
 */
#include "gst/GstInit.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/Graph.h"
#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  try {
    namespace fs = std::filesystem;
    using namespace simaai::neat;
    const fs::path root = argc > 1 ? argv[1] : fs::current_path();
    const char* configured = std::getenv("SIMA_YOLO11_POSE_BF16_TAR");
    const fs::path archive =
        configured && *configured ? fs::path(configured) : root / "tmp/yolov11_mpk_bf16.tar.gz";
    if (!fs::exists(archive)) {
      if (configured && *configured)
        throw std::runtime_error("Configured SIMA_YOLO11_POSE_BF16_TAR does not exist");
      return skip_long_test("Set SIMA_YOLO11_POSE_BF16_TAR to a YOLO11 pose BF16 MPK archive");
    }
    const auto image = sima_yolov8_test::load_people_image_or_skip(root);
    Model::Options options;
    options.preprocess.kind = InputKind::Image;
    options.preprocess.enable = AutoFlag::On;
    options.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
    options.decode_type = BoxDecodeType::YoloV8Pose;
    options.score_threshold = 0.25f;
    options.nms_iou_threshold = 0.5f;
    options.top_k = 100;
    Model model(archive.string(), options);
    Graph graph;
    graph.add(nodes::Input());
    graph.add(nodes::groups::Preprocess(model));
    graph.add(nodes::groups::Infer(model));
    graph.add(nodes::SimaBoxDecode(model, BoxDecodeType::YoloV8Pose, 0.25, 0.5, 100));
    graph.add(nodes::Output());
    // Separate in-flight DMA allocations avoid same-buffer ownership contention.
    gst_init_once();
    std::vector<TensorList> inputs;
    for (int i = 0; i < 2; ++i)
      inputs.push_back(
          TensorList{Tensor::from_cv_mat(image, ImageSpec::PixelFormat::BGR, TensorMemory::EV74)});
    RunOptions run_options;
    run_options.queue_depth = 2;
    auto run = graph.build(inputs.front(), run_options);
    const auto validate = [&](const TensorList& output) {
      const auto decoded = decode_pose(output, image.cols, image.rows, 100, true);
      require(decoded.size() == 1, "pose route must publish one wire tensor");
      const auto& pose = decoded.front();
      require(pose.boxes.shape.size() == 2 && pose.boxes.shape[0] > 0 && pose.boxes.shape[1] == 6,
              "people image must produce nonempty pose boxes");
      require(pose.keypoints.shape == std::vector<int64_t>{pose.boxes.shape[0], 17, 3},
              "pose boxes and keypoints must remain aligned");
      const auto bytes = pose.keypoints.copy_payload_bytes();
      std::vector<float> values(bytes.size() / sizeof(float));
      std::memcpy(values.data(), bytes.data(), bytes.size());
      bool visible = false;
      for (std::size_t i = 0; i < values.size(); i += 3) {
        require(std::isfinite(values[i]) && std::isfinite(values[i + 1]) &&
                    std::isfinite(values[i + 2]),
                "pose output must be finite");
        require(values[i + 2] >= 0 && values[i + 2] <= 1, "pose visibility must be a probability");
        visible |= values[i + 2] > 0.5f;
      }
      require(visible, "people image must produce visible keypoints, not untouched zero output");
    };
    for (int i = 0; i < 8; ++i)
      validate(run.run(inputs[i % 2], 30000));
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) {
      require(run.push(inputs[0]) && run.push(inputs[1]), "async pose input was rejected");
      validate(run.pull_tensors(30000));
      validate(run.pull_tensors(30000));
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double fps = seconds > 0 ? 40.0 / seconds : 0;
    std::cout << "YOLO11_POSE outputs=40 fps=" << fps << '\n';
    require(fps >= 1.0, "pose integration smoke must sustain at least 1 FPS");
    return 0;
  } catch (const SkipTest& error) {
    return skip_long_test(error.what());
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
