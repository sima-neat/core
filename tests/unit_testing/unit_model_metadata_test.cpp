#include "model/Model.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/ErrorCodes.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <unordered_map>

namespace {

sima_test::ModelArchiveFixture make_metadata_fixture(const std::string& tag, bool include_metadata,
                                                     const std::string& metadata_json = "") {
  std::vector<std::pair<std::string, std::string>> files = {
      {"etc/pipeline_sequence.json",
       R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      }
    ]
  }]
})json"},
      {"etc/0_preproc.json",
       R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB"
})json"},
      {"etc/0_process_mla.json",
       R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
  };

  if (include_metadata) {
    files.push_back({"etc/metadata.json", metadata_json.empty() ? std::string(R"json({
  "model_name": "unit-meta",
  "version": 3,
  "calibration": true,
  "labels": ["cat", "dog"],
  "nested": {"a": 1}
})json")
                                                                : metadata_json});
  }

  return sima_test::make_strict_model_archive_fixture(tag, files, true);
}

} // namespace

RUN_TEST(
    "unit_model_metadata_test", ([] {
      using namespace simaai::neat;

      {
        const auto fixture = make_metadata_fixture("model_metadata_present", true);
        Model model(fixture.tar_path);
        const std::unordered_map<std::string, std::string> meta = model.metadata();

        require(!meta.empty(), "Model::metadata should return entries when metadata.json exists");
        require(meta.at("model_name") == "unit-meta",
                "Model::metadata should preserve string values verbatim");
        require(meta.at("version") == "3", "Model::metadata should stringify integer values");
        require(meta.at("calibration") == "true",
                "Model::metadata should stringify boolean values");
        require(meta.at("labels") == "[\"cat\",\"dog\"]",
                "Model::metadata should dump array values");
        require(meta.at("nested") == "{\"a\":1}", "Model::metadata should dump object values");

        require(!model.infer_output_name().empty(),
                "default inference naming must remain available without executable artifacts");
        require(!model.input_specs().empty(),
                "input metadata must remain available without executable artifacts");
        const auto output_specs = model.output_specs();
        require(output_specs.size() == 10U &&
                    output_specs.front().shape == std::vector<int64_t>({1, 80, 80, 64}) &&
                    output_specs.front().dtypes == std::vector<TensorDType>{TensorDType::Float32},
                "selected postprocess metadata must retain exact MPK outputs without artifacts");
        Model::Options options;
        options.decode_type = BoxDecodeType::YoloV8Seg;
        Model decoded(fixture.tar_path, options);
        require(decoded.metadata().at("model_name") == "unit-meta" &&
                    !decoded.input_specs().empty() && !decoded.output_specs().empty(),
                "selecting a compatible decoder must still permit metadata-only inspection");
        try {
          (void)nodes::SimaBoxDecode(decoded, options.decode_type);
          require(false, "model-bound BoxDecode must reject a missing executable");
        } catch (const std::runtime_error& error) {
          require_contains(error.what(), "missing-mla-executable",
                           "preserve physical admission error");
          const auto* structured = dynamic_cast<const NeatError*>(&error);
          require(!structured || structured->report().error_code != error_codes::kPipelineShape,
                  "a missing executable must not be reported as a decoder mismatch");
        }
        for (auto* inspected : {&model, &decoded}) {
          bool admission_rejected = false;
          try {
            (void)inspected->graph();
          } catch (const std::exception& error) {
            admission_rejected = true;
            require_contains(error.what(), "missing-mla-executable",
                             "execution admission must fail without the MLA executable");
          }
          require(admission_rejected, "metadata inspection must preserve graph admission");
        }
      }

      {
        const auto [filename, text] = sima_test::strict_contract_json_entry_from_modelzoo();
        auto mpk = nlohmann::json::parse(text);
        auto& terminal = mpk.at("plugins").back();
        terminal["name"] = "objectdecode_terminal";
        terminal["processor"] = "A65";
        terminal["config_params"]["kernel"] = "boxdecode";
        terminal["config_params"]["params"] = {{"decode_type", "yolov8-seg"}};
        const auto fixture = sima_test::make_model_archive_fixture(
            "model_metadata_authored_decoder", {{filename, mpk.dump()}}, true);
        Model::Options options;
        options.decode_type = BoxDecodeType::YoloV8Seg;
        const Model model(fixture.tar_path, options);
        require(!model.output_specs().empty(),
                "the authored BoxDecode terminal must be inspectable");
        options.decode_type = BoxDecodeType::Detr;
        require_neat_error([&] { Model mismatch(fixture.tar_path, options); },
                           error_codes::kPipelineShape, "does not match");
      }

      {
        const auto fixture = make_metadata_fixture("model_metadata_missing", false);
        Model model(fixture.tar_path);
        const auto meta = model.metadata();
        require(meta.empty(),
                "Model::metadata should return empty map when metadata.json is missing");
      }

      {
        const auto legacy = sima_test::make_model_archive_fixture(
            "model_metadata_legacy_missing_mpk", {
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
          (void)legacy_model.metadata();
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "strict MPK contract required",
                           "legacy missing-mpk fixture should fail with strict contract error");
        }
        require(threw, "legacy missing-mpk fixture must fail under strict contract");
      }

      {
        // Auxiliary JSON does not block model construction, but explicitly reading malformed
        // metadata surfaces a structured NeatError carrying an io.* error_code.
        const auto fixture =
            make_metadata_fixture("model_metadata_malformed", true, "{ not valid json ");
        Model model(fixture.tar_path);
        bool threw_neaterror = false;
        try {
          (void)model.metadata();
        } catch (const NeatError& e) {
          threw_neaterror = true;
          require_contains(
              std::string(e.what()), "io.",
              "malformed-metadata archive should surface a NeatError with an io.* error_code");
        }
        require(threw_neaterror, "malformed-metadata archive should throw simaai::neat::NeatError");

        // NeatError derives from std::runtime_error, so existing runtime_error catchers still
        // catch Model failures unchanged (backward compatibility guarantee).
        bool caught_as_runtime = false;
        try {
          Model model(
              make_metadata_fixture("model_metadata_malformed_rt", true, "{ also bad ").tar_path);
          (void)model.metadata();
        } catch (const std::runtime_error&) {
          caught_as_runtime = true;
        }
        require(caught_as_runtime, "Model NeatError must remain catchable as std::runtime_error");
      }
    }));
