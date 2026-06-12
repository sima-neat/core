#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <vector>

namespace {

std::string require_yolov9_tar() {
  const std::string tar =
      sima_test::resolve_modelzoo_tar("yolo_v9c_seg", sima_test::repo_root_for_modelzoo());
  require(!tar.empty(), "missing yolo_v9c_seg tar; run sima-cli modelzoo get yolo_v9c_seg");
  require(std::filesystem::exists(tar), "resolved yolo_v9c_seg tar does not exist: " + tar);
  return tar;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0)
      out += ",";
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

} // namespace

RUN_TEST(
    "unit_model_output_spec_contract_test", ([] {
      using namespace simaai::neat;

      {
        const std::string tar = require_yolov9_tar();
        Model model(tar);
        const Model::ModelInfo info = model.info();
        const std::vector<TensorSpec> specs = model.output_specs();

        require(info.output_topology.physical_outputs >= 1,
                "yolov9 should expose at least one physical output");
        require(info.output_topology.logical_outputs >= 1,
                "yolov9 should expose at least one logical output");
        const std::vector<std::vector<int64_t>> expected_shapes = {
            {1, 80, 80, 64}, {1, 40, 40, 64}, {1, 20, 20, 64}, {1, 80, 80, 80}, {1, 40, 40, 80},
            {1, 20, 20, 80}, {1, 80, 80, 32}, {1, 40, 40, 32}, {1, 20, 20, 32}, {1, 160, 160, 32},
        };
        require(specs.size() == expected_shapes.size(),
                "Model::output_specs should expose all yolov9 outputs, got " +
                    std::to_string(specs.size()));
        for (size_t i = 0; i < specs.size(); ++i) {
          const TensorSpec& spec = specs[i];
          require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                  "Model::output_specs dtype mismatch at index " + std::to_string(i));
          require(spec.rank == static_cast<int>(expected_shapes[i].size()),
                  "Model::output_specs rank mismatch at index " + std::to_string(i) + ", got " +
                      std::to_string(spec.rank));
          require(spec.shape == expected_shapes[i], "Model::output_specs shape mismatch at index " +
                                                        std::to_string(i) + ", got " +
                                                        shape_to_string(spec.shape));
        }

        const TensorSpec& first = specs.front();
        require(first.shape == expected_shapes.front(),
                "Model::output_specs first entry should expose first logical output shape");
        require(!first.dtypes.empty() && first.dtypes[0] == TensorDType::Float32,
                "Model::output_specs first entry dtype mismatch");
      }

      {
        Model::Options opt;
        opt.decode_type = simaai::neat::BoxDecodeType::Yolo;
        opt.score_threshold = 0.25f;
        opt.nms_iou_threshold = 0.45f;
        opt.top_k = 100;
        const std::string tar = require_yolov9_tar();
        Model model(tar, opt);
        const std::vector<TensorSpec> specs = model.output_specs();
        require(!specs.empty(), "Model::output_specs boxdecode should not be empty");
        const TensorSpec& spec = specs.front();

        require(spec.rank == -1,
                "Model::output_specs boxdecode Model boundary should return unknown rank");
        require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::UInt8,
                "Model::output_specs boxdecode Model boundary should expose UInt8 dtype");
        require(specs.size() == 1, "Model::output_specs boxdecode should expose one terminal spec");
        require(specs.front().rank == -1,
                "Model::output_specs boxdecode terminal rank should be unknown");
        require(!specs.front().dtypes.empty() && specs.front().dtypes[0] == TensorDType::UInt8,
                "Model::output_specs boxdecode terminal dtype should be UInt8");
      }

      {
        const auto root = sima_test::repo_root_for_modelzoo();
        const std::vector<std::filesystem::path> bf16_candidates = {
            root / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_mpk.tar.gz",
            root / "tmp" / "yolov8n_drive" / "yolov8n_A_W_BF16_MLATess.tar.gz",
        };
        for (const auto& bf16_model : bf16_candidates) {
          if (!std::filesystem::exists(bf16_model)) {
            continue;
          }
          Model model(bf16_model.string());
          const std::vector<TensorSpec> specs = model.output_specs();
          require(!specs.empty(), "BF16 model output_specs should not be empty");
          for (size_t i = 0; i < specs.size(); ++i) {
            require(specs[i].rank != 0,
                    "BF16 model output_specs rank should not be zero at index " +
                        std::to_string(i));
            require(!specs[i].dtypes.empty() && specs[i].dtypes[0] == TensorDType::Float32,
                    "BF16 cast egress must expose Float32 Model boundary output_specs at index " +
                        std::to_string(i));
          }
          break;
        }
      }

      {
        const auto legacy = sima_test::make_model_archive_fixture(
            "model_output_spec_legacy_missing_mpk", {
                                                        {"etc/pipeline_sequence.json",
                                                         R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                                                        {"etc/0_process_mla.json",
                                                         R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "decoder"}]
})json"},
                                                    });
        bool threw = false;
        try {
          Model legacy_model(legacy.tar_path);
          (void)legacy_model.output_specs();
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "strict MPK contract required",
                           "legacy missing-mpk fixture should fail with strict contract error");
        }
        require(threw, "legacy missing-mpk fixture must fail under strict contract");
      }
    }));
