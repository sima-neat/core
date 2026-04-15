#include "asset_utils.h"
#include "cli_utils.h"
#include "gst/GstInit.h"
#include "gst/GstHelpers.h"
#include "model/Model.h"
#include "pipeline/TensorOpenCV.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path find_repo_root() {
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

cv::Mat load_input_image_or_skip(const fs::path& root) {
  const std::vector<fs::path> candidates = {
      root / "tests" / "images" / "people.jpg",
      root / "test.jpg",
      root / "tmp" / "coco_sample.jpg",
  };

  for (const auto& path : candidates) {
    if (!fs::exists(path))
      continue;
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (!img.empty())
      return img;
  }

  const fs::path coco = root / "tmp" / "coco_sample.jpg";
  const std::string url =
      "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg";
  if (!sima_test::download_file(url, coco)) {
    skip_long_test_exception("failed to download sample image");
  }
  cv::Mat img = cv::imread(coco.string(), cv::IMREAD_COLOR);
  if (img.empty()) {
    skip_long_test_exception("failed to load downloaded sample image");
  }
  return img;
}

bool has_valid_tensor(const simaai::neat::Tensor& t, std::string& err) {
  if (!t.storage) {
    err = "tensor missing storage";
    return false;
  }
  if (t.storage->size_bytes == 0 && t.storage->kind != simaai::neat::StorageKind::DeviceHandle) {
    err = "tensor storage is empty";
    return false;
  }
  std::string validate_err;
  if (!t.validate(&validate_err)) {
    err = "tensor validation failed: " + validate_err;
    return false;
  }
  return true;
}

bool sample_has_tensor(const simaai::neat::Sample& s, std::string& err) {
  if (s.kind == simaai::neat::SampleKind::Tensor) {
    if (!s.tensor.has_value()) {
      err = "tensor output missing";
      return false;
    }
    return has_valid_tensor(*s.tensor, err);
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    if (s.fields.empty()) {
      err = "bundle output empty";
      return false;
    }
    for (const auto& field : s.fields) {
      std::string field_err;
      if (sample_has_tensor(field, field_err))
        return true;
    }
    err = "bundle output missing tensor";
    return false;
  }
  err = "unexpected sample kind";
  return false;
}

void run_model_once(const std::string& model_name, const std::string& tar_gz,
                    const cv::Mat& img_bgr) {
  if (tar_gz.empty()) {
    throw std::runtime_error("missing model tar.gz for " + model_name);
  }

  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "BGR";
  model_opt.input_max_width = img_bgr.cols;
  model_opt.input_max_height = img_bgr.rows;
  model_opt.input_max_depth = img_bgr.channels();
  simaai::neat::Model model(tar_gz, model_opt);

  simaai::neat::Sample sync_out = model.run(img_bgr);
  std::string sync_err;
  if (!sample_has_tensor(sync_out, sync_err)) {
    throw std::runtime_error("sync output invalid: " + sync_err);
  }

  auto runner = model.build(img_bgr);
  if (!runner) {
    throw std::runtime_error("async runner build failed");
  }
  if (!runner.push(img_bgr)) {
    runner.close();
    throw std::runtime_error("async push failed");
  }
  auto async_out = runner.pull(10000);
  if (!async_out.has_value()) {
    runner.close();
    throw std::runtime_error("async pull returned no output");
  }
  std::string async_err;
  if (!sample_has_tensor(*async_out, async_err)) {
    runner.close();
    throw std::runtime_error("async output invalid: " + async_err);
  }
  runner.close();
}

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test("OpenCV required for modelzoo bulk test");
#else
  try {
    const bool run_flag = sima_test::has_flag(argc, argv, "--run");
    if (!run_flag && !env_flag("SIMA_RUN_MODELZOO_BULK")) {
      return skip_long_test("set SIMA_RUN_MODELZOO_BULK=1 or pass --run to execute");
    }

    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocesscvu") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing SimaAI plugins (neatprocesscvu/mla)");
    }

    const fs::path root = find_repo_root();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    std::string only_model;
    sima_test::get_arg(argc, argv, "--only", only_model);
    int limit = -1;
    sima_test::parse_int_arg(argc, argv, "--limit", limit);

    const std::vector<std::string> models = {
        "anomaly_detection/fastflow_demo",
        "anomaly_detection/fastflow_mvtec",
        "anomaly_detection/stfpm_mvtec",
        "change_detection/fresunet",
        "depth_estimation/depth_anything_v2_vits",
        "depth_estimation/midas_v21_small_256",
        "face_detection/fld_68landmarks",
        "grasping/ggcnn_v2",
        "handwriting_classification/mnist_cnn",
        "handwriting_classification/mnist_fc",
        "image_classification/alexnet",
        "image_classification/convnext_base",
        "image_classification/convnext_small",
        "image_classification/convnext_tiny",
        "image_classification/densenet121_12",
        "image_classification/densenet121_9",
        "image_classification/densenet_121",
        "image_classification/densenet_161",
        "image_classification/densenet_169",
        "image_classification/densenet_201",
        "image_classification/dla_34",
        "image_classification/efficientnet-lite4-11",
        "image_classification/efficientnet-v2-L",
        "image_classification/efficientnet-v2-b0",
        "image_classification/efficientnet-v2-m",
        "image_classification/efficientnet_b0",
        "image_classification/efficientnet_b1",
        "image_classification/efficientnet_b2",
        "image_classification/efficientnet_b3",
        "image_classification/efficientnet_b4",
        "image_classification/efficientnet_b5",
        "image_classification/efficientnet_b6",
        "image_classification/efficientnet_b7",
        "image_classification/efficientnet_edgetpu_L",
        "image_classification/efficientnet_edgetpu_S",
        "image_classification/efficientnet_lite0",
        "image_classification/efficientnet_lite1",
        "image_classification/efficientnet_lite2",
        "image_classification/efficientnet_lite3",
        "image_classification/efficientnet_lite4",
        "image_classification/googlenet",
        "image_classification/googlenet_v3",
        "image_classification/hardnet_39",
        "image_classification/hardnet_68",
        "image_classification/hbonet-0.25",
        "image_classification/hbonet_1_0",
        "image_classification/hrnet_w32",
        "image_classification/inception_v3",
        "image_classification/mnasnet_a1_0.5",
        "image_classification/mnasnet_a1_0.75",
        "image_classification/mnasnet_a1_1.0",
        "image_classification/mnasnet_a1_1.3",
        "image_classification/mobilenet_v1",
        "image_classification/mobilenet_v2",
        "image_classification/mobilenet_v2_12",
        "image_classification/mobilenet_v2_7",
        "image_classification/mobilenet_v3_large",
        "image_classification/mobilenet_v3_small",
        "image_classification/nfnet_f0",
        "image_classification/regnet_x_1.6gf",
        "image_classification/regnet_x_16gf",
        "image_classification/regnet_x_3.2gf",
        "image_classification/regnet_x_32gf",
        "image_classification/regnet_x_400mf",
        "image_classification/regnet_x_800mf",
        "image_classification/regnet_x_8gf",
        "image_classification/regnet_y_16gf",
        "image_classification/regnet_y_1_6gf",
        "image_classification/regnet_y_32gf",
        "image_classification/regnet_y_3_2gf",
        "image_classification/regnet_y_400mf",
        "image_classification/regnet_y_800mf",
        "image_classification/regnet_y_8gf",
        "image_classification/repvgg-a0",
        "image_classification/repvgg-a1",
        "image_classification/repvgg-a2",
        "image_classification/repvgg-b1",
        "image_classification/repvgg-b3",
        "image_classification/resnet50_v1_7",
        "image_classification/resnet50_v2_7",
        "image_classification/resnet_101_v2",
        "image_classification/resnet_18",
        "image_classification/resnet_34",
        "image_classification/resnet_50",
        "image_classification/resnet_50_v15",
        "image_classification/resnext101_32x8d",
        "image_classification/resnext101_64x4d",
        "image_classification/resnext_101",
        "image_classification/resnext_50",
        "image_classification/shufflenet_10_v2",
        "image_classification/shufflenet_12_v2",
        "image_classification/shufflenet_v2_x05",
        "image_classification/shufflenet_v2_x10",
        "image_classification/shufflenet_v2_x15",
        "image_classification/shufflenet_v2_x20",
        "image_classification/squeezenet_v1",
        "image_classification/squeezenet_v1.1",
        "image_classification/vgg11",
        "image_classification/vgg11_bn",
        "image_classification/vgg13",
        "image_classification/vgg13_bn",
        "image_classification/vgg16",
        "image_classification/vgg16_bn",
        "image_classification/vgg19",
        "image_classification/vgg19_bn",
        "image_classification/vits14",
        "image_classification/wide-resnet-101",
        "image_classification/wide-resnet-50",
        "image_super_resolution/ABPN_AIMET",
        "image_super_resolution/carn_m",
        "image_super_resolution/msrganx4",
        "image_super_resolution/rdn",
        "image_super_resolution/sesr_m11_2x",
        "image_super_resolution/sesr_m7_2x",
        "instance_segmentation/yolo_v8l_seg",
        "instance_segmentation/yolo_v8m_seg",
        "instance_segmentation/yolo_v8n_seg",
        "instance_segmentation/yolo_v8s_seg",
        "instance_segmentation/yolo_v8x_seg",
        "instance_segmentation/yolo_v9_gelan_c_seg",
        "instance_segmentation/yolo_v9c_seg",
        "license_plate_recognition/lprnet",
        "object_detection/centernet",
        "object_detection/yolo_v3",
        "object_detection/yolo_v3_tiny",
        "object_detection/yolo_v7",
        "object_detection/yolo_v7w6",
        "object_detection/yolo_v7x",
        "object_detection/yolo_v8l",
        "object_detection/yolo_v8m",
        "object_detection/yolo_v8n",
        "object_detection/yolo_v8s",
        "object_detection/yolo_v8x",
        "object_detection/yolo_v9c",
        "object_detection/yolo_v9m",
        "object_detection/yolo_v9s",
        "object_detection/yolo_v9t",
        "object_detection/yolox_l",
        "object_detection/yolox_m",
        "object_detection/yolox_nano",
        "object_detection/yolox_s",
        "object_detection/yolox_tiny",
        "object_detection/yolox_x",
        "pose_estimation/open_pose",
        "re_identification/reid",
        "semantic_segmentation/fcn_hrnet18",
        "semantic_segmentation/fcn_hrnet48",
        "semantic_segmentation/lraspp_mnv3",
        "semantic_segmentation/yolov5l",
        "semantic_segmentation/yolov5m",
        "semantic_segmentation/yolov5n",
        "semantic_segmentation/yolov5s",
    };

    cv::Mat img_bgr = load_input_image_or_skip(root);

    int failures = 0;
    int ran = 0;

    for (const auto& name : models) {
      if (!only_model.empty() && name != only_model)
        continue;
      if (limit >= 0 && ran >= limit)
        break;

      std::cout << "[MODEL] " << name << "\n";
      std::string tar_gz = sima_test::resolve_modelzoo_tar(name, root);
      if (tar_gz.empty()) {
        std::cerr << "[FAIL] download failed for " << name << " (sima-cli modelzoo -v "
                  << sima_test::modelzoo_version() << " get " << name << ")\n";
        failures += 1;
        ran += 1;
        continue;
      }

      try {
        run_model_once(name, tar_gz, img_bgr);
        std::cout << "[OK] " << name << "\n";
      } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << name << ": " << e.what() << "\n";
        failures += 1;
      }

      std::error_code rm_ec;
      fs::remove(tar_gz, rm_ec);
      ran += 1;
    }

    if (ran == 0) {
      return skip_long_test("no models selected (use --only or --limit)");
    }

    if (failures != 0) {
      std::cerr << "[SUMMARY] failures=" << failures << " ran=" << ran << "\n";
      return 1;
    }

    std::cout << "[SUMMARY] all models passed (" << ran << ")\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}
