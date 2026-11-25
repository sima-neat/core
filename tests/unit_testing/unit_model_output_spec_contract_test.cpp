#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "support/yolov8_variant_fixture.h"
#include "sam_decoder_fixtures.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string require_yolov9_tar() {
  // Metadata-only unit coverage: reuse the checked exact modelzoo manifest
  // and existing topology fixture builder; no inference binary is executed.
  static const auto fixture = sima_test::make_strict_model_archive_fixture(
      "unit_model_output_spec_contract_test", {}, true, "yolo_v9c_seg", true);
  return fixture.tar_path;
}

// A decoder package the board's dmabuf-plan admission accepts: the MPK plus
// synthetic topology evidence for each MLA executable it names.
sima_test::ModelArchiveFixture admissible_fixture(const std::string& tag, const std::string& file,
                                                  const std::string& mpk) {
  auto fixture = sima_test::make_model_archive_fixture(tag, {{file, mpk}}, false);
  sima_test::write_mpk_topology_elfs(fixture.root_dir, mpk);
  const std::string cmd = "tar -czf " + sima_test::model_archive_shell_quote(fixture.tar_path) +
                          " -C " + sima_test::model_archive_shell_quote(fixture.root_dir) + " .";
  if (std::system(cmd.c_str()) != 0)
    throw std::runtime_error("failed to archive " + tag);
  return fixture;
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
        // BoxDecodeType::EdgeTamTrack: an EdgeTAM decoder returns the BoxDecode tensor, then
        // exactly the tensors the same Model returns without BoxDecode; EdgeTamSeg the mask alone.
        const auto fixture =
            admissible_fixture("edgetam_track_raw_outputs", "etc/edgetam_mask_decoder_mpk.json",
                               sam_test::edgetam_decoder_mpk());
        Model raw_model(fixture.tar_path);
        const auto raw = raw_model.output_specs();
        const std::vector<std::vector<int64_t>> shapes = {
            {1, 256, 256, 4}, {1, 4}, {1, 1}, {1, 4, 256}};
        require(raw.size() == shapes.size(),
                "EdgeTAM without BoxDecode must return its four decoder outputs, got " +
                    std::to_string(raw.size()));
        for (std::size_t i = 0; i < raw.size(); ++i)
          require(raw[i].dtypes == std::vector<TensorDType>{TensorDType::Float32} &&
                      raw[i].shape == shapes[i],
                  "EdgeTAM raw output " + std::to_string(i) + " is " +
                      shape_to_string(raw[i].shape));
        Model::Options options;
        options.decode_type = BoxDecodeType::EdgeTamSeg;
        Model decoded(fixture.tar_path, options);
        require(decoded.output_specs().size() == 1U,
                "EdgeTamSeg must return the BoxDecode tensor alone");
        options.decode_type = BoxDecodeType::EdgeTamTrack;
        Model both(fixture.tar_path, options);
        const auto specs = both.output_specs();
        require(specs.size() == 1U + raw.size() &&
                    specs.front().dtypes == std::vector<TensorDType>{TensorDType::UInt8},
                "EdgeTamTrack must return the BoxDecode tensor first, then the raw outputs");
        for (std::size_t i = 0; i < raw.size(); ++i)
          require(specs[1 + i].dtypes == raw[i].dtypes && specs[1 + i].shape == raw[i].shape,
                  "raw output " + std::to_string(i) + " differs from the route without BoxDecode");
        const auto contract = internal::ModelAccess::build_boxdecode_stage_contract(both, false);
        require(contract.payload.raw_outputs.size() == raw.size(),
                "the BoxDecode stage must publish one raw output per input");
        // The names the route without BoxDecode publishes, '/' included, not
        // graph endpoint tokens ("cast_5_mask_logits").
        const std::vector<std::string> names = {"cast_5/mask_logits", "cast_6/iou_scores",
                                                "cast_7/object_score_logits",
                                                "cast_8/object_pointers"};
        for (std::size_t i = 0; i < raw.size(); ++i)
          require(contract.payload.raw_outputs[i].dtype == "FP32" &&
                      contract.payload.raw_outputs[i].shape == shapes[i] &&
                      contract.payload.raw_outputs[i].name == names[i],
                  "raw output " + std::to_string(i) + " is '" +
                      contract.payload.raw_outputs[i].name + "', not the published name '" +
                      names[i] + "' (or lost its dtype or shape)");
        require(internal::ModelAccess::build_boxdecode_stage_contract(decoded, false)
                    .payload.raw_outputs.empty(),
                "the EdgeTamSeg BoxDecode stage must publish no raw outputs");
      }

      {
        // Model::run keeps one runner per original image size for a Segment
        // Anything decoder, whose masks have that size: the size the inputs
        // publish keys the runner, so 720p then 1080p builds a second one.
        const auto fixture =
            admissible_fixture("edgetam_run_original_size", "etc/edgetam_mask_decoder_mpk.json",
                               sam_test::edgetam_decoder_mpk());
        Model::Options options;
        options.decode_type = BoxDecodeType::EdgeTamSeg;
        Model decoder(fixture.tar_path, options);
        Model raw(fixture.tar_path);
        const auto inputs = [](int width, int height) {
          Tensor tensor = Tensor::from_vector(std::vector<float>(4), {1, 4});
          if (width > 0) {
            PreprocessRuntimeMeta facts;
            facts.resize_mode = "stretch";
            facts.original_width = width;
            facts.original_height = height;
            facts.resized_width = facts.resized_height = 1024;
            tensor.semantic.preprocess = facts;
          }
          return TensorList{Tensor::from_vector(std::vector<float>(4), {1, 4}), tensor};
        };
        const auto key = [](const Model& model, const TensorList& list) {
          return internal::ModelAccess::fixed_boxdecode_original_size(model, list);
        };
        require(key(decoder, inputs(1280, 720)) == std::pair{1280, 720} &&
                    key(decoder, inputs(1920, 1080)) == std::pair{1920, 1080},
                "a Segment Anything decoder's runner must be keyed by the original image size");
        require(!key(decoder, inputs(0, 0)) && !key(raw, inputs(1280, 720)) &&
                    !key(Model(require_yolov9_tar()), inputs(1280, 720)),
                "only a Segment Anything BoxDecode route fixes its runner to one image size");
      }

      {
        // Options::multimask_output maps per model when the BoxDecode stage is
        // built; a choice the model cannot make fails there, by name.
        struct Choice {
          const char* tag;
          std::string mpk;
          BoxDecodeType type;
          bool multimask;
          const char* rejection;
        };
        const Choice choices[] = {
            {"mobilesam", sam_test::kMobileSamDecoderMpk, BoxDecodeType::MobileSam, false, nullptr},
            {"edgetam", sam_test::edgetam_decoder_mpk(), BoxDecodeType::EdgeTamSeg, false, nullptr},
            {"edgetam", sam_test::edgetam_decoder_mpk(), BoxDecodeType::EdgeTamTrack, false,
             nullptr},
            {"efficientsam", sam_test::efficientsam_decoder_mpk(), BoxDecodeType::EfficientSam,
             true, nullptr},
            {"efficientsam", sam_test::efficientsam_decoder_mpk(), BoxDecodeType::EfficientSam,
             false, "EfficientSAM exports no single-mask token"},
            {"efficientsam", sam_test::efficientsam_decoder_mpk(), BoxDecodeType::EfficientSam3,
             false, "EfficientSAM3 decoder with three mask channels exports no single-mask token"},
            {"efficientsam", sam_test::efficientsam_decoder_mpk(), BoxDecodeType::MobileSam, true,
             "MobileSAM decoders export 1 or 4 mask channels; this one has 3"}};
        for (const auto& c : choices) {
          const auto fixture =
              admissible_fixture(std::string("sam_multimask_") + c.tag,
                                 std::string("etc/") + c.tag + "_decoder_mpk.json", c.mpk);
          Model::Options options;
          options.decode_type = c.type;
          options.multimask_output = c.multimask;
          std::string error;
          try {
            Model model(fixture.tar_path, options);
            const auto contract =
                internal::ModelAccess::build_boxdecode_stage_contract(model, false);
            require(contract.payload.multimask_output == c.multimask,
                    std::string(c.tag) + ": the stage lost multimask_output");
          } catch (const std::exception& e) {
            error = e.what();
          }
          if (c.rejection)
            require(error.find(c.rejection) != std::string::npos,
                    std::string(c.tag) + " must reject multimask_output with \"" + c.rejection +
                        "\", got \"" + error + "\"");
          else
            require(error.empty(), std::string(c.tag) + ": " + error);
        }
        // Another decode type rejects a non-default value, as pose_classes does.
        Model::Options yolo;
        yolo.decode_type = BoxDecodeType::Yolo;
        yolo.multimask_output = false;
        std::string error;
        try {
          Model model(require_yolov9_tar(), yolo);
          (void)internal::ModelAccess::build_boxdecode_stage_contract(model, false);
        } catch (const std::exception& e) {
          error = e.what();
        }
        require(error.find("multimask_output applies only to the Segment Anything decode types") !=
                    std::string::npos,
                "a YOLO decoder must reject multimask_output=false, got \"" + error + "\"");
      }

      {
        const auto path = sima_test::test_model_archive_fixture_root_path() /
                          "strict-seeds/nemotron_joiner_argmax_mpk.json";
        std::ifstream input(path);
        require(input.is_open(), "joiner compiler manifest fixture must be installed");
        const std::string text((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
        const auto fixture = sima_test::make_model_archive_fixture(
            "joiner_cold_output_specs", {{"etc/joiner_argmax_mpk.json", text}}, true);
        Model model(fixture.tar_path);
        const auto specs = model.output_specs();
        require(specs.size() == 1U &&
                    specs.front().dtypes == std::vector<TensorDType>{TensorDType::Int32} &&
                    specs.front().shape == std::vector<int64_t>({1, 1}) && specs.front().rank == 2,
                "cold joiner output_specs must report the declared INT32 [1,1] egress");
        require(model.info().output_topology.logical_outputs == 1U,
                "cold joiner info must report one logical output");
      }

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
        Model model(sima_test::resolve_yolov8_variant_fixture("yolov8n_A_W_BF16_mpk").tar_path);
        const std::vector<TensorSpec> specs = model.output_specs();
        require(!specs.empty(), "BF16 model output_specs should not be empty");
        for (size_t i = 0; i < specs.size(); ++i) {
          require(specs[i].rank != 0,
                  "BF16 model output_specs rank should not be zero at index " + std::to_string(i));
          require(!specs[i].dtypes.empty() && specs[i].dtypes[0] == TensorDType::Float32,
                  "BF16 cast egress must expose Float32 Model boundary output_specs at index " +
                      std::to_string(i));
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
