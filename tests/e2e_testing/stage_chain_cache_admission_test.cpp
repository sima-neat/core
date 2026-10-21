#include "neat.h"
#include "asset_utils.h"
#include "pipeline/StageRun.h"
#include "pipeline/internal/StageInputCacheIdentity.h"
#include "test_main.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unistd.h>

namespace {
using namespace simaai::neat;

// Capture the existing build diagnostic, not a replacement preparation provider.
// Restore/report all stderr even on failure so retirement diagnostics remain visible.
class BuildLog {
public:
  BuildLog() {
    stream_ = std::tmpfile();
    require(stream_ != nullptr, "could not open build diagnostic capture");
    std::fflush(stderr);
    saved_ = dup(STDERR_FILENO);
    require(saved_ >= 0, "could not preserve stderr");
    require(dup2(fileno(stream_), STDERR_FILENO) >= 0, "could not capture stderr");
  }
  ~BuildLog() {
    if (saved_ >= 0) {
      try {
        std::cerr << finish();
      } catch (...) {
        std::fputs("[FAIL] build diagnostic restoration failed\n", stderr);
      }
    }
    if (stream_) {
      std::fclose(stream_);
    }
  }
  std::string finish() {
    std::fflush(stderr);
    require(dup2(saved_, STDERR_FILENO) >= 0, "could not restore stderr");
    close(saved_);
    saved_ = -1;
    std::rewind(stream_);
    std::string log;
    char chunk[4096];
    while (const auto count = std::fread(chunk, 1U, sizeof(chunk), stream_)) {
      log.append(chunk, count);
    }
    return log;
  }

private:
  FILE* stream_ = nullptr;
  int saved_ = -1;
};

template <class Execute> std::pair<Sample, std::size_t> with_compile_count(Execute&& execute) {
  BuildLog capture;
  Sample output = execute();
  const auto log = capture.finish();
  std::cerr << log;
  std::size_t count = 0U;
  std::size_t offset = 0U;
  const std::string marker = "[GRAPH-BUILD-TIMING] session_build_compile_contracts";
  while ((offset = log.find(marker, offset)) != std::string::npos) {
    ++count;
    offset += marker.size();
  }
  return {std::move(output), count};
}

Sample heads_for_image(const cv::Mat& image, const Model& model) {
  Sample pre = sample_from_tensors(stages::Preproc(std::vector<cv::Mat>{image}, model));
  Sample infer = stages::Infer(pre, model);
  require(infer.kind == SampleKind::TensorSet && infer.tensors.size() == 6U,
          "the unchanged YOLOv8 model must publish all six typed heads");
  for (const auto& head : infer.tensors) {
    require(head.route.segment_name == infer.tensors.front().route.segment_name,
            "logical heads must preserve the shared producer carrier");
  }
  return infer;
}

void report_cache_identity_difference(
    const pipeline_internal::StageInputCacheIdentity& first,
    const std::optional<pipeline_internal::StageInputCacheIdentity>& second,
    const std::string& ineligible_reason) {
  if (!second) {
    std::cerr << "[stage-cache-identity-diff] second input ineligible: " << ineligible_reason
              << '\n';
    return;
  }
  // The identity contains metadata only: no addresses, mappings or tensor payload bytes.
  // Include both values at each changed JSON pointer rather than dumping all six heads.
  using Json = nlohmann::ordered_json;
  const auto before = Json::parse(first.value);
  const auto after = Json::parse(second->value);
  auto changes = Json::diff(before, after);
  for (auto& change : changes) {
    const Json::json_pointer path(change.at("path").get<std::string>());
    if (before.contains(path)) {
      change["before"] = before.at(path);
    }
    if (after.contains(path)) {
      change["after"] = after.at(path);
    }
  }
  std::cerr << "[stage-cache-identity-diff] " << changes.dump(2) << '\n';
}
} // namespace

RUN_TEST(
    "stage_chain_cache_admission_test", ([] {
      using namespace simaai::neat;
      namespace pi = pipeline_internal;
      const auto root = sima_test::test_source_root();
      const auto archive = sima_test::resolve_yolov8s_strict_mpk_tar(root);
      require(!archive.empty(), "strict YOLOv8 MPK fixture is required");
      const auto image =
          cv::imread(sima_test::test_image_fixture_path().string(), cv::IMREAD_COLOR);
      require(!image.empty(), "canonical test image is required");
      Model::Options options;
      options.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
      options.preprocess.input_max_width = image.cols;
      options.preprocess.input_max_height = image.rows;
      options.preprocess.input_max_depth = image.channels();
      options.decode_type = BoxDecodeType::YoloV8;
      options.score_threshold = 0.25F;
      options.nms_iou_threshold = 0.5F;
      options.top_k = 100;
      Model model(archive, options);

      require(setenv("SIMA_GRAPH_BUILD_TIMING", "1", 1) == 0, "could not enable build diagnostic");
      Sample first = heads_for_image(image, model);
      const auto first_key = pi::stage_input_cache_identity(
          first, nullptr, pi::StageInputCacheGeometryPolicy::DynamicOriginalSize);
      require(first_key.has_value(), "actual typed producer output must permit safe stage reuse");
      std::weak_ptr<TensorBuffer> first_view = first.tensors.front().storage;
      auto [cold, cold_compiles] =
          with_compile_count([&] { return stages::Postprocess(first, model); });
      require(cold_compiles == 1U, "cold Postprocess must execute one seeded stage compilation");
      require(cold.kind == SampleKind::TensorSet && cold.tensors.size() == 1U,
              "Postprocess must publish one detection payload");
      const auto retained = cold.tensors.front().copy_payload_bytes();
      const auto cold_boxes =
          decode_bbox_tensor(cold.tensors.front(), image.cols, image.rows, 100, true);
      first = Sample{};

      Sample second = heads_for_image(image, model);
      std::string second_ineligible_reason;
      const auto second_key =
          pi::stage_input_cache_identity(second, &second_ineligible_reason,
                                         pi::StageInputCacheGeometryPolicy::DynamicOriginalSize);
      if (second_key != first_key) {
        report_cache_identity_difference(*first_key, second_key, second_ineligible_reason);
      }
      require(second_key == first_key, "equivalent fresh producer views must have equal identity");
      auto [warm, warm_compiles] =
          with_compile_count([&] { return stages::Postprocess(second, model); });
      require(warm_compiles == 0U, "compatible warm Postprocess must not compile another stage");
      require(first_view.expired(), "cached plan must not retain its first input Tensor view");
      require(cold.tensors.front().copy_payload_bytes() == retained,
              "retained cold output must remain unchanged after warm execution");
      const auto warm_boxes =
          decode_bbox_tensor(warm.tensors.front(), image.cols, image.rows, 100, true);
      require(warm_boxes.boxes.size() == cold_boxes.boxes.size(), "warm detection count changed");
      for (std::size_t i = 0; i < cold_boxes.boxes.size(); ++i) {
        const auto& a = cold_boxes.boxes[i];
        const auto& b = warm_boxes.boxes[i];
        require(a.class_id == b.class_id && std::fabs(a.score - b.score) < 1e-5F &&
                    std::fabs(a.x1 - b.x1) < 1e-4F && std::fabs(a.y1 - b.y1) < 1e-4F &&
                    std::fabs(a.x2 - b.x2) < 1e-4F && std::fabs(a.y2 - b.y2) < 1e-4F,
                "warm detection values changed for identical image input");
      }
      // Actual ROI preprocessing changes frame geometry, not the selected MPK
      // inference contracts. Keep fresh affine/ROI facts on every output while
      // reusing that immutable stage, including a repeated earlier coordinate.
      require(image.cols >= 8 && image.rows >= 2, "ROI fixture image is too small");
      for (const bool mla_only : {false, true}) {
        int frame = 0;
        std::size_t inference_compiles = 0;
        double first_offset_x = 0;
        std::vector<std::vector<std::uint8_t>> first_roi_outputs;
        for (int x : {0, 1, 2, 3, 0}) {
          const std::vector<PreprocessRoi> rois{{0, x, 0, image.cols / 2, image.rows / 2}};
          Sample roi_input = sample_from_tensors(stages::Preproc({image}, model, rois));
          require(roi_input.tensors.size() == 1 &&
                      roi_input.tensors[0].semantic.preprocess.has_value(),
                  "ROI preprocessing must publish its current transform");
          const auto expected = *roi_input.tensors[0].semantic.preprocess;
          require(expected.rois.size() == 1 && expected.rois[0].x == x,
                  "ROI preprocessing did not publish the requested current coordinate");
          if (frame == 0)
            first_offset_x = expected.affine_m02;
          else if (x != 0)
            require(expected.affine_m02 != first_offset_x,
                    "moving ROI fixture did not change its model-to-original transform");
          auto [inferred, compiles] = with_compile_count([&] {
            return mla_only ? stages::MLA(roi_input, model) : stages::Infer(roi_input, model);
          });
          inference_compiles += compiles;
          require(compiles <= (frame == 0 ? 1U : 0U),
                  "moving frame ROI recompiled the immutable inference stage");
          require(inferred.tensors.size() == 6, "ROI inference lost model output heads");
          for (std::size_t i = 0; i < inferred.tensors.size(); ++i) {
            const auto& tensor = inferred.tensors[i];
            require(tensor.semantic.preprocess.has_value(), "inference lost frame ROI metadata");
            const auto& actual = *tensor.semantic.preprocess;
            // MLA publishes scalar image/affine facts, not the input ROI-list
            // extension. Check that supported publication against this frame.
            require(actual.original_width == expected.original_width &&
                        actual.original_height == expected.original_height &&
                        actual.resized_width == expected.resized_width &&
                        actual.resized_height == expected.resized_height &&
                        actual.affine_m00 == expected.affine_m00 &&
                        actual.affine_m01 == expected.affine_m01 &&
                        actual.affine_m02 == expected.affine_m02 &&
                        actual.affine_m10 == expected.affine_m10 &&
                        actual.affine_m11 == expected.affine_m11 &&
                        actual.affine_m12 == expected.affine_m12,
                    "cached inference reused stale image or affine metadata");
            if (frame == 0)
              first_roi_outputs.push_back(tensor.copy_payload_bytes());
            else if (frame == 4)
              require(tensor.copy_payload_bytes() == first_roi_outputs[i],
                      "repeated ROI inference changed numerical output");
          }
          ++frame;
        }
        require(inference_compiles <= 1, "moving ROI sequence retained per-coordinate stages");
        std::cout << "[stage-cache] route=" << (mla_only ? "MLA" : "Infer")
                  << " moving_roi_frames=" << frame
                  << " inference_stage_compiles=" << inference_compiles << '\n';
      }
      std::cout << "[stage-cache] cold_stage_compiles=" << cold_compiles
                << " warm_stage_compiles=" << warm_compiles
                << " retained_output_verified=1 first_input_view_released=1\n";
    }));
