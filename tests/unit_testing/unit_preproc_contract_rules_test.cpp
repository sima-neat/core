#include <simaai/scheduler/client/frame_layout.h>
#include "model/Model.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelInternal.h"
#endif
#include "model_archive_fixture_utils.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/Quant.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#ifdef SIMA_NEAT_INTERNAL
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include <gstsimaaitensorbuffer_meta.h>
#endif
#include "gst/GstInit.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gst/gst.h>

namespace {

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (found unexpected: " + needle + ")");
  }
}

void set_model_managed_preproc_max_input_shape(simaai::neat::PreprocOptions* opt,
                                               std::vector<int> shape) {
  if (!opt) {
    return;
  }
#ifdef SIMA_NEAT_INTERNAL
  auto lineage = std::make_shared<simaai::neat::internal::ModelLineageBinding>();
  lineage->preproc_max_input_shape = std::move(shape);
  opt->model_lineage = std::move(lineage);
#else
  (void)shape;
#endif
}

std::vector<int> model_managed_preproc_max_input_shape(const simaai::neat::PreprocOptions& opt) {
#ifdef SIMA_NEAT_INTERNAL
  if (opt.model_lineage) {
    return opt.model_lineage->preproc_max_input_shape;
  }
#else
  (void)opt;
#endif
  return {};
}

// The MLA input canvas is canvas x canvas x channels (640 x 640 x 3 unless a test needs
// another size).
sima_test::ModelArchiveFixture
make_preproc_fixture(const std::string& tag, int canvas = 640,
                     const nlohmann::json& input_range = nlohmann::json::array({0.0, 1.0}),
                     bool mpk_image_preproc = false, int channels = 3) {
  const auto fixture = sima_test::make_model_archive_fixture(tag,
                                                             {
                                                                 {"etc/test_model_mpk.json",
                                                                  R"json({
  "name": "preproc_contract_model",
  "model_path": "preproc_contract_model.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    {
      "name": "images",
      "type": "buffer",
      "size": 4915200,
      "input_range": [0.0, 1.0],
      "logical_shape": [1, 640, 640, 3],
      "logical_dtype": "FP32"
    }
  ],
  "plugins": [
    {
      "name": "cast_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "cast_transform",
        "params": {
          "out_dtype": "bfloat16",
          "input_shapes": [[1, 640, 640, 3]],
          "output_shapes": [[1, 640, 640, 3]]
        }
      },
      "input_nodes": [{"name": "images", "size": 4915200}],
      "output_nodes": [{
        "name": "cast_0",
        "type": "buffer",
        "size": 2457600,
        "logical_shape": [1, 640, 640, 3],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "kernel_name_tbd"}
    },
    {
      "name": "mla_0",
      "sequence": 2,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 4,
        "input_types": [{"scalar": "bfloat16", "shape": [1, 640, 640, 3]}],
        "output_types": [{"scalar": "bfloat16", "shape": [1, 80, 80, 6]}]
      },
      "input_nodes": [{
        "name": "cast_0",
        "size": 2457600,
        "logical_shape": [1, 640, 640, 3],
        "logical_dtype": "BF16"
      }],
      "output_nodes": [{
        "name": "mla_0",
        "type": "buffer",
        "size": 76800,
        "logical_shape": [1, 80, 80, 6],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "stage0.elf"}
    }
  ]
})json"},
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
  "graph_name": "preproc",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "normalize": true,
  "channel_mean": [0.0, 0.0, 0.0],
  "channel_stddev": [1.0, 1.0, 1.0],
  "output_dtype": "BF16",
  "tessellate": false
})json"},
                                                                 {"etc/0_process_mla.json",
                                                                  R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["EV81_BFLOAT16"],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                             },
                                                             false);
  // Another canvas or channel count rescales the 640x640x3 shapes and their FP32/BF16 byte
  // sizes.
  for (const char* name : {"etc/test_model_mpk.json", "etc/0_preproc.json"}) {
    if (canvas == 640 && channels == 3)
      continue;
    const auto path = std::filesystem::path(fixture.root_dir) / name;
    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const std::pair<std::string, std::string> sizes[] = {
        {"640, 3]", std::to_string(canvas) + ", " + std::to_string(channels) + "]"},
        {"4915200", std::to_string(canvas * canvas * 4 * channels)},
        {"2457600", std::to_string(canvas * canvas * 2 * channels)},
        {"640", std::to_string(canvas)}};
    for (const auto& [from, to] : sizes)
      for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
        text.replace(at, from.size(), to);
    std::ofstream(path, std::ios::trunc) << text;
  }
  const auto mpk_path = std::filesystem::path(fixture.root_dir) / "etc/test_model_mpk.json";
  std::ifstream mpk_input(mpk_path);
  auto manifest = nlohmann::json::parse(mpk_input);
  mpk_input.close();
  if (input_range.is_null()) {
    manifest["input_nodes"][0].erase("input_range");
  } else {
    manifest["input_nodes"][0]["input_range"] = input_range;
  }
  if (mpk_image_preproc) {
    // This historical token is only for route-plan inspection. Executable-contract
    // tests keep Cast -> MLA and let Model add its supported image preprocessing.
    manifest["plugins"][0]["config_params"]["kernel"] = "preproc";
  }
  std::ofstream(mpk_path, std::ios::trunc) << manifest.dump();
  // Image preprocessing comes from Model::Options; the MPK describes its tensor target.
  // Planning needs the declared MLA topology, not an executable MLA program.
  const auto share = std::filesystem::path(fixture.root_dir) / "share";
  std::filesystem::create_directories(share);
  sima_test::write_topology_elf(share / "stage0.elf", "data.ifm.b0",
                                static_cast<std::uint64_t>(canvas) * canvas * 2U * channels,
                                "data.ofm.b0", 76800U);
  const std::string archive = "tar -czf " + sima_test::model_archive_shell_quote(fixture.tar_path) +
                              " -C " + sima_test::model_archive_shell_quote(fixture.root_dir) +
                              " .";
  require(std::system(archive.c_str()) == 0, "failed to archive Preproc topology fixture");
  return fixture;
}

sima_test::ModelArchiveFixture make_infer_only_fixture(const std::string& tag) {
  return sima_test::make_model_archive_fixture(tag,
                                               {
                                                   {"etc/test_model_mpk.json",
                                                    R"json({
  "name": "infer_only_contract_model",
  "model_path": "infer_only_contract_model.onnx",
  "model_sdk_version": "2.0.0",
  "sequence": 1,
  "input_nodes": [
    {
      "name": "images",
      "type": "buffer",
      "size": 9830400,
      "input_range": [0.0, 1.0],
      "logical_shape": [1, 1280, 1280, 3],
      "logical_dtype": "BF16"
    }
  ],
  "plugins": [
    {
      "name": "mla_0",
      "sequence": 1,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "number_of_quads_to_user": 4,
        "input_shapes": [[1, 1280, 1280, 3]],
        "input_data_type": ["BF16"],
        "output_shapes": [[1, 80, 80, 6]],
        "data_type": ["BF16"]
      },
      "input_nodes": [{
        "name": "images",
        "size": 9830400,
        "logical_shape": [1, 1280, 1280, 3],
        "logical_dtype": "BF16"
      }],
      "output_nodes": [{
        "name": "mla_0",
        "type": "buffer",
        "size": 76800,
        "logical_shape": [1, 80, 80, 6],
        "logical_dtype": "BF16"
      }],
      "type": "sgpProcess",
      "resources": {"executable": "stage0.elf"}
    }
  ]
})json"},
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
  "input_buffers": [{"name": "decoder"}],
  "data_type": ["EV81_BFLOAT16"],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                               },
                                               true);
}

simaai::neat::OutputSpec make_rgb_input_spec(int width, int height) {
  simaai::neat::OutputSpec spec;
  spec.media_type = "video/x-raw";
  spec.format = "BGR";
  spec.width = width;
  spec.height = height;
  spec.depth = 3;
  spec.layout = "HWC";
  spec.dtype = "UInt8";
  spec.memory = "SystemMemory";
  return spec;
}

std::vector<int> shape3(int h, int w, int c) {
  std::vector<int> out;
  out.push_back(h);
  out.push_back(w);
  out.push_back(c);
  return out;
}

} // namespace

RUN_TEST(
    "unit_preproc_contract_rules_test", ([] {
      using namespace simaai::neat;

      // Calibration extrema never specify a normalization recipe. In particular,
      // [0.2, 0.8] may be observed after /255 without changing that preprocessing.
      {
        const auto check = [](const sima_test::ModelArchiveFixture& fixture,
                              const Model::Options& options, bool normalize, bool advisory,
                              const std::string& label) {
          Model model(fixture.tar_path, options);
          const auto plan = model.resolved_preprocess_plan();
          require(plan.effective.normalize.enable == (normalize ? AutoFlag::On : AutoFlag::Off),
                  label + ": normalization must follow explicit configuration only");
          std::size_t advisories = 0;
          for (const auto& warning : plan.warnings) {
            if (warning.find("preprocess.normalize=Auto resolved Off") == std::string::npos)
              continue;
            ++advisories;
            require_contains(warning, "preprocess.normalize.enable=On",
                             label + ": diagnostic must explain explicit normalization");
            require_contains(warning, "preprocess.preset",
                             label + ": diagnostic must mention presets");
          }
          require(advisories == (advisory ? 1U : 0U),
                  label + ": only unconfigured Auto on an image route needs one advisory");
          return plan;
        };
        const nlohmann::json ranges[] = {
            nlohmann::json::array({0.0, 1.0}), nlohmann::json::array({0.2, 0.8}),
            nlohmann::json::array({-1.0, 1.0}), nlohmann::json::array({0.0, 255.0}), nullptr};
        for (const auto& range : ranges) {
          const auto fixture = make_preproc_fixture("normalize_observed_range", 640, range, true);
          const auto plan = check(fixture, {}, false, true, "MPK image route " + range.dump());
          require(plan.requested.normalize.enable == AutoFlag::Auto,
                  "the public Auto option must be preserved");
          require(plan.effective.normalize.mean == std::array{0.0f, 0.0f, 0.0f} &&
                      plan.effective.normalize.stddev == std::array{1.0f, 1.0f, 1.0f},
                  "observed extrema must not replace the default statistics");
        }

        const auto fixture = make_preproc_fixture("normalize_explicit_policy", 640,
                                                  nlohmann::json::array({0.2, 0.8}));
        Model::Options image;
        image.preprocess.kind = InputKind::Image;
        image.preprocess.enable = AutoFlag::On;
        check(fixture, image, false, true, "explicit image route");
        require(!PreprocOptions(Model(fixture.tar_path, image)).normalize,
                "model-managed Preproc must keep unconfigured normalization Off");

        auto options = image;
        options.preprocess.normalize.enable = AutoFlag::Off;
        check(fixture, options, false, false, "explicit Off");
        options.preprocess.normalize.enable = AutoFlag::On;
        options.preprocess.normalize.mean = {0.1f, 0.2f, 0.3f};
        options.preprocess.normalize.stddev = {0.4f, 0.5f, 0.6f};
        const auto explicit_plan = check(fixture, options, true, false, "explicit On");
        require(explicit_plan.effective.normalize.mean == options.preprocess.normalize.mean &&
                    explicit_plan.effective.normalize.stddev == options.preprocess.normalize.stddev,
                "explicit On must preserve the authored statistics");
        options.preprocess.normalize.enable = AutoFlag::Auto;
        options.preprocess.normalize.has_explicit_stats = true;
        check(fixture, options, true, false, "Auto with explicit statistics");

        for (const auto preset : {NormalizePreset::ImageNet, NormalizePreset::COCO_YOLO}) {
          options = image;
          options.preprocess.preset = preset;
          check(fixture, options, true, false, "preset normalization");
        }

        // The effective transform list owns intent, including an Auto-only list
        // whose image route is selected from the MPK with InputKind::Auto.
        const auto transform_fixture = make_preproc_fixture(
            "normalize_transform_route", 640, nlohmann::json::array({0.2, 0.8}), true);
        options = {};
        options.preprocess.normalize.enable = AutoFlag::On;
        Transform transform;
        transform.type = TransformType::Normalize;
        options.preprocess.transforms = {transform};
        check(transform_fixture, options, false, true, "transform Auto overrides simple On");
        options.preprocess.transforms[0].normalize.enable = AutoFlag::Off;
        check(transform_fixture, options, false, false, "transform Off");
        options.preprocess.transforms[0].normalize.enable = AutoFlag::On;
        check(transform_fixture, options, true, false, "transform On");
        options.preprocess.transforms[0].normalize.enable = AutoFlag::Auto;
        options.preprocess.transforms[0].normalize.has_explicit_stats = true;
        check(transform_fixture, options, true, false, "transform statistics");

        const auto tensor_fixture = make_preproc_fixture("normalize_tensor_route");
        options = {};
        options.preprocess.kind = InputKind::Tensor;
        check(tensor_fixture, options, false, false, "tensor route");
        options.preprocess.enable = AutoFlag::Off;
        check(make_infer_only_fixture("normalize_preprocessing_disabled"), options, false, false,
              "preprocessing disabled");
      }

      {
        const auto fixture = make_infer_only_fixture("preproc_contract_missing");
        Model::Options model_opt;
        model_opt.preprocess.enable = AutoFlag::Off;
        model_opt.preprocess.kind = InputKind::Tensor;
        Model model(fixture.tar_path, model_opt);

        bool threw = false;
        try {
          (void)PreprocOptions(model);
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "does not contain Preproc",
                           "Preproc(Model) should fail when route omits Preproc");
          require_contains(
              std::string(e.what()), "Pass preprocess options to Model",
              "Preproc(Model) failure should explain how to enable model-managed Preproc");
        }
        require(threw, "Preproc(Model) must hard fail when route has no Preproc stage");

        threw = false;
        try {
          (void)QuantOptions(model);
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "does not contain Quant",
                           "Quant(Model) should fail when route omits Quant stage");
        }
        require(threw, "Quant(Model) must hard fail when route has no Quant stage");
      }

      {
        const auto fixture = make_preproc_fixture("preproc_contract_present");
        Model::Options model_opt;
        model_opt.preprocess.kind = InputKind::Image;
        model_opt.preprocess.enable = AutoFlag::On;
        model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
        Model model(fixture.tar_path, model_opt);

        const auto requirements = model.preprocess_requirements();
        require(requirements.has_preproc_stage,
                "Model::preprocess_requirements should report model-managed Preproc");
        require(!requirements.output_format.empty(),
                "Model::preprocess_requirements should expose preproc output format");

        PreprocOptions opt(model);
        require(opt.model_managed_contract,
                "PreprocOptions(Model) should mark the contract as model-managed");
        require(opt.input_width() > 0 && opt.input_height() > 0,
                "PreprocOptions(Model) should expose model-managed input shape");
        require(!opt.input_img_type.empty(),
                "PreprocOptions(Model) should expose model-managed input format");
        require(opt.output_width() > 0 && opt.output_height() > 0,
                "PreprocOptions(Model) should expose model-managed output shape");
        require(!opt.output_dtype.empty(),
                "PreprocOptions(Model) should expose model-managed output dtype");
        require(!opt.normalize,
                "PreprocOptions(Model) should derive normalize from the resolved preprocess plan, "
                "not from legacy graph defaults");
#ifdef SIMA_NEAT_INTERNAL
        require(opt.compiled_contract != nullptr,
                "model must retain its admitted preproc contract");
        const auto admitted = opt.compiled_contract;
        const auto original_source = admitted->runtime_contract.physical_inputs.front().size_bytes;
        const auto original_output = admitted->payload.output_tensors.front().descriptor();
        const auto original_logical = admitted->runtime_contract.logical_outputs.front();
        const auto slot_bytes = admitted->runtime_contract.physical_outputs.front().size_bytes;
        bool source_count_threw = false;
        try {
          (void)pipeline_internal::sima::specialize_preproc_roi_contract(*admitted, 51, 48, 64, 3,
                                                                         1);
        } catch (const std::invalid_argument&) {
          source_count_threw = true;
        }
        require(source_count_threw, "ROI source batch must respect the firmware's 50-image limit");
        gst_init_once();
        for (const int capacity : {1, 2, 3}) {
          auto inherited = *admitted;
          inherited.runtime_contract.allocation_policy =
              simaai::scheduler::client::AllocationPolicy::PreferDmsThenCma;
          auto invocation = pipeline_internal::sima::specialize_preproc_roi_contract(
              inherited, 2, 48, 64, 3, capacity);
          require(
              invocation.payload.batch_size == capacity &&
                  invocation.payload.input_shapes.front() == std::vector<int>({48, 64, 3}) &&
                  invocation.payload.input_tensors.front().descriptor().shape.sizes[0] == 2 &&
                  invocation.payload.input_tensors.front().descriptor().shape.axis_semantics[0] ==
                      SIMA_EV_AXIS_N,
              "ROI source N must be independent of output R and public HWC options");
          require(invocation.runtime_contract.physical_inputs.front().size_bytes == 18432U &&
                      invocation.runtime_contract.logical_inputs.front().size_bytes == 18432U &&
                      invocation.runtime_contract.input_bindings.front().src_physical_size_bytes ==
                          18432U,
                  "ROI source descriptor and binding must cover both source frames");
          require(std::memcmp(&invocation.payload.output_tensors.front().descriptor(),
                              &original_output, sizeof(original_output)) == 0,
                  "ROI specialization must preserve every one-member output descriptor byte");
          require(invocation.runtime_contract.physical_outputs.front().size_bytes ==
                          slot_bytes * capacity &&
                      invocation.runtime_contract.logical_outputs.front().size_bytes ==
                          original_logical.size_bytes * capacity &&
                      invocation.runtime_contract.logical_outputs.front().logical_name ==
                          original_logical.logical_name &&
                      invocation.runtime_contract.logical_outputs.front().backend_name ==
                          original_logical.backend_name &&
                      invocation.payload.q_scale == admitted->payload.q_scale &&
                      invocation.payload.q_zp == admitted->payload.q_zp,
                  "ROI carrier batching must preserve exact slot span, identities and qparams");
          require(!invocation.physical_command_role && !invocation.payload.dmabuf_plan_contract &&
                      !invocation.runtime_contract.allocation_policy,
                  "ROI invocation must obtain its own storage after executor resolution");
          auto roi_opt = opt;
          roi_opt.batch_size = capacity;
          roi_opt.compiled_contract =
              std::make_shared<const CompiledProcessCvuContract>(std::move(invocation));
          pipeline_internal::sima::ManifestBuildDiagnostics roi_diagnostics;
          const auto roi_compiled =
              compile_node_contracts(std::vector<std::shared_ptr<Node>>{nodes::Preproc(roi_opt)},
                                     ContractCompileInput{}, &roi_diagnostics);
          auto roi_manifest = render_manifest_from_compiled_contracts(
              roi_compiled, ContractCompileInput{}, &roi_diagnostics);
          require(
              roi_diagnostics.errors.empty() && roi_manifest && roi_manifest->stages.size() == 1U,
              "specialized immutable preproc must compile/render without options reconstruction");
          namespace ps = pipeline_internal::sima;
          const auto prepared = ps::prepare_kernel_stage_recipes(&*roi_manifest, {}, {}, {}, {});
          require(!prepared, prepared ? prepared->message : "");
          ps::finalize_pipeline_storage(*roi_manifest);
          require(!ps::finalize_stage_publications(&*roi_manifest), "ROI publication finalization");
          const auto& layout = roi_manifest->stages.front().frame_layout;
          namespace c = simaai::scheduler::client;
          require(layout && layout->ownership == c::FrameOwnership::Allocate &&
                      layout->allocations.size() == 1 &&
                      layout->allocations.front().policy == c::AllocationPolicy::RequireCma &&
                      layout->allocations.front().size_bytes >= slot_bytes * capacity &&
                      layout->outputs.size() == 1 &&
                      layout->outputs.front().span.byte_offset == 0 &&
                      layout->outputs.front().span.size_bytes == slot_bytes * capacity,
                  "ROI final layout must own CMA and preserve exact batched output capacity");
        }
        require(admitted == opt.compiled_contract &&
                    admitted->runtime_contract.physical_inputs.front().size_bytes ==
                        original_source &&
                    admitted->runtime_contract.logical_outputs.front().shape ==
                        original_logical.shape &&
                    std::memcmp(&admitted->payload.output_tensors.front().descriptor(),
                                &original_output, sizeof(original_output)) == 0,
                "ROI invocation must not mutate the shared model contract");

        // Explicit batch layouts do not reinterpret unqualified volume descriptors.
        std::uint8_t axes[SIMA_EV_MAX_RANK]{};
        pipeline_internal::sima::tensorsemantics::fill_axis_semantics_from_shape_layout(
            std::vector<int>{2, 48, 64, 3}, "NHWC", axes);
        require(axes[0] == SIMA_EV_AXIS_N, "explicit NHWC must preserve source N=2");
        pipeline_internal::sima::tensorsemantics::fill_axis_semantics_from_shape_layout(
            std::vector<int>{2, 48, 64, 3}, "HWC", axes);
        require(axes[0] == SIMA_EV_AXIS_D, "unqualified HWC volume inference must not change");

        const auto model_max_shape = model_managed_preproc_max_input_shape(opt);
        require(PreprocOptions::shape_dim(model_max_shape, 1) >= opt.input_width() &&
                    PreprocOptions::shape_dim(model_max_shape, 0) >= opt.input_height(),
                "PreprocOptions(Model) should preserve internal modelpack max input capacity");

        Preproc capacity_node(opt);
        InputContract oversized_contract;
        oversized_contract.media_type = "video/x-raw";
        oversized_contract.format = opt.input_img_type;
        oversized_contract.width = PreprocOptions::shape_dim(model_max_shape, 1) + 1;
        oversized_contract.height = opt.input_height();
        oversized_contract.depth = opt.input_channels();
        bool capacity_threw = false;
        try {
          capacity_node.apply_input_contract(oversized_contract, nullptr);
        } catch (const std::exception& e) {
          capacity_threw = true;
          require_contains(
              std::string(e.what()), "exceeds max_input_width",
              "PreprocOptions(Model) capacity violation should mention max_input_width");
        }
        require(capacity_threw,
                "PreprocOptions(Model) must reject inputs above modelpack capacity");
#endif

        Preproc node(opt);
        const std::string frag = node.backend_fragment(0);
        require_not_contains(
            frag, "config=", "model-managed Preproc fragment must not emit a legacy config path");
      }

      {
        const auto fixture = make_preproc_fixture("preproc_dynamic_capacity_rebind");
        Model::Options model_opt;
        model_opt.preprocess.kind = InputKind::Image;
        model_opt.preprocess.enable = AutoFlag::On;
        model_opt.preprocess.input_max_width = 1920;
        model_opt.preprocess.input_max_height = 1080;
        model_opt.preprocess.input_max_depth = 3;
        model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
        model_opt.preprocess.resize.enable = AutoFlag::On;
        model_opt.preprocess.resize.width = 640;
        model_opt.preprocess.resize.height = 640;
        model_opt.preprocess.resize.mode = ResizeMode::Letterbox;
        Model model(fixture.tar_path, model_opt);

        PreprocOptions opt(model);
        require(opt.input_width() == 1920 && opt.input_height() == 1080 &&
                    opt.input_channels() == 3,
                "model-managed Preproc must project the configured capacity into its static input "
                "shape before a smaller seed is bound");
#ifdef SIMA_NEAT_INTERNAL
        const auto initial_max_shape = model_managed_preproc_max_input_shape(opt);
        require(PreprocOptions::shape_dim(initial_max_shape, 1) == 1920 &&
                    PreprocOptions::shape_dim(initial_max_shape, 0) == 1080 &&
                    PreprocOptions::shape_channels(initial_max_shape) == 3,
                "model-managed Preproc must retain the configured capacity in model lineage");
#endif

        auto node = std::make_shared<Preproc>(opt);
        InputContract seed_contract;
        seed_contract.media_type = "video/x-raw";
        seed_contract.format = "RGB";
        seed_contract.width = 1280;
        seed_contract.height = 720;
        seed_contract.depth = 3;
        node->apply_input_contract(seed_contract, nullptr);
        require(node->options().input_width() == 1280 && node->options().input_height() == 720,
                "model-managed Preproc must bind the smaller seed as actual geometry");

        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const std::vector<std::shared_ptr<Node>> nodes{node};
        const auto compiled = compile_node_contracts(nodes, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty() && compiled.fully_renderable &&
                    compiled.stages.size() == 1U && compiled.stages.front().processcvu.has_value(),
                "model-managed Preproc capacity contract must compile without diagnostics");
        const auto& processcvu = *compiled.stages.front().processcvu;
        require(processcvu.payload.input_shapes.size() == 1U &&
                    processcvu.payload.input_shapes.front() == std::vector<int>({1080, 1920, 3}),
                "compiled Preproc payload must retain capacity after a smaller seed is bound");
        require(
            processcvu.runtime_contract.logical_inputs.size() == 1U &&
                processcvu.runtime_contract.logical_inputs.front().shape ==
                    std::vector<std::int64_t>({1080, 1920, 3}),
            "compiled Preproc logical input must retain capacity after a smaller seed is bound");

        auto manifest =
            render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty() && manifest.has_value() && manifest->stages.size() == 1U,
                "model-managed Preproc capacity contract must render without diagnostics");
        const auto& rendered_stage = manifest->stages.front();
        require(rendered_stage.logical_inputs.size() == 1U &&
                    rendered_stage.logical_inputs.front().shape ==
                        std::vector<std::int64_t>({1080, 1920, 3}) &&
                    rendered_stage.processcvu.input_shapes.size() == 1U &&
                    rendered_stage.processcvu.input_shapes.front() ==
                        std::vector<int>({1080, 1920, 3}),
                "rendered Preproc static contract must expose the full dynamic input capacity");
        require(rendered_stage.processcvu.dmabuf_plan_contract ==
                        processcvu.payload.dmabuf_plan_contract &&
                    rendered_stage.processcvu.graph_name == "preproc" &&
                    rendered_stage.input_bindings.size() == 1U &&
                    rendered_stage.physical_inputs.size() == 1U &&
                    rendered_stage.physical_inputs.front().size_bytes > 0U &&
                    rendered_stage.logical_outputs.size() == 1U &&
                    rendered_stage.physical_outputs.size() == 1U &&
                    rendered_stage.logical_outputs.front().backend_output_index >= 0 &&
                    rendered_stage.logical_outputs.front().physical_index ==
                        rendered_stage.physical_outputs.front().physical_index,
                "rendering must preserve inherited Preproc facts without creating final storage");
        const auto& input_binding = rendered_stage.input_bindings.front();
        const auto& physical_input = rendered_stage.physical_inputs.front();
        const auto& logical_input = rendered_stage.logical_inputs.front();
        // This node consumes an external image, not a model-stage output. Its sole
        // DMA-BUF still has explicit selectors; only the producer-stage identity is absent.
        require(
            input_binding.src_stage_id.empty() && input_binding.src_stage_index < 0 &&
                input_binding.sink_pad_index == 0 && input_binding.local_logical_input_index == 0 &&
                input_binding.src_logical_output_index == 0 && input_binding.src_output_slot == 0 &&
                input_binding.src_physical_output_index == 0 &&
                input_binding.src_physical_byte_offset == 0 &&
                input_binding.src_physical_size_bytes == physical_input.size_bytes &&
                input_binding.source_segment_name == physical_input.segment_name &&
                physical_input.physical_index == 0 && physical_input.allocator_index == 0 &&
                physical_input.source_physical_index == 0 &&
                physical_input.source_byte_offset == 0 && logical_input.logical_index == 0 &&
                logical_input.backend_input_index == 0 && logical_input.physical_index == 0 &&
                logical_input.byte_offset == 0,
            "model-managed Preproc must bind the sole external DMA-BUF without a producer stage");
        namespace ps = pipeline_internal::sima;
        const auto prepared = ps::prepare_kernel_stage_recipes(&*manifest, {}, {}, {}, {});
        require(!prepared, prepared ? prepared->field + ": " + prepared->message : "");
        ps::finalize_pipeline_storage(*manifest);
        require(!ps::finalize_stage_publications(&*manifest), "Preproc publication finalization");
        const auto& finalized_stage = manifest->stages.front();
        const auto& layout = finalized_stage.frame_layout;
        namespace c = simaai::scheduler::client;
        require(layout && layout->ownership == c::FrameOwnership::Allocate &&
                    layout->allocations.size() == 1U &&
                    layout->allocations.front().policy == c::AllocationPolicy::RequireCma &&
                    (layout->allocations.front().required_device_access & c::kAccessEv74) != 0U &&
                    layout->outputs.size() == 1U,
                "checked Preproc must receive one CMA allocation for its selected output");
        const auto& span = layout->outputs.front().span;
        require(span.allocation_index == 0U && span.size_bytes > 0U &&
                    span.byte_offset <= layout->allocations.front().size_bytes &&
                    span.size_bytes <= layout->allocations.front().size_bytes - span.byte_offset,
                "final Preproc carrier must fit its allocation");
        for (const auto& output : finalized_stage.physical_outputs)
          require(output.required_alignment_bytes > 0U && output.source_byte_offset >= 0 &&
                      static_cast<std::uint64_t>(output.source_byte_offset) %
                              output.required_alignment_bytes ==
                          0U,
                  "final Preproc carrier must satisfy its checked alignment");

        InputContract capacity_contract = seed_contract;
        capacity_contract.width = 1920;
        capacity_contract.height = 1080;
        node->apply_input_contract(capacity_contract, nullptr);
        require(node->options().input_width() == 1920 && node->options().input_height() == 1080,
                "model-managed Preproc must accept a later contract up to its configured capacity");
#ifdef SIMA_NEAT_INTERNAL
        const auto rebound_max_shape = model_managed_preproc_max_input_shape(node->options());
        require(PreprocOptions::shape_dim(rebound_max_shape, 1) == 1920 &&
                    PreprocOptions::shape_dim(rebound_max_shape, 0) == 1080,
                "runtime contract rebinding must not shrink the model-managed capacity");
#endif
      }

      // A single-channel model, as MNIST was: GRAY8 ingress with depth 1 must reach the MLA as
      // one channel instead of being converted to RGB. Planning only, no MLA run.
      {
        const auto fixture = make_preproc_fixture("preproc_gray8_single_channel", 28,
                                                  nlohmann::json::array({0.0, 1.0}), false, 1);
        Model::Options model_opt;
        model_opt.preprocess.kind = InputKind::Image;
        model_opt.preprocess.enable = AutoFlag::On;
        model_opt.preprocess.input_max_depth = 1;
        model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::GRAY8;
        model_opt.preprocess.normalize.enable = AutoFlag::On;
        model_opt.preprocess.normalize.mean = {0.1307f, 0.1307f, 0.1307f};
        model_opt.preprocess.normalize.stddev = {0.3081f, 0.3081f, 0.3081f};
        model_opt.preprocess.normalize.has_explicit_stats = true;
        Model model(fixture.tar_path, model_opt);

        const auto requirements = model.preprocess_requirements();
        require(requirements.has_preproc_stage && requirements.output_format == "GRAY8",
                "single-channel GRAY8 Preproc must publish GRAY8, got " +
                    requirements.output_format);

        PreprocOptions opt(model);
        require(opt.input_channels() == 1,
                "single-channel GRAY8 Preproc must take one input channel, got " +
                    std::to_string(opt.input_channels()));
        require(opt.output_img_type == "GRAY8",
                "single-channel GRAY8 Preproc must not convert to RGB, got " + opt.output_img_type);

        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const std::vector<std::shared_ptr<Node>> nodes{std::make_shared<Preproc>(opt)};
        const auto compiled = compile_node_contracts(nodes, ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty() && compiled.fully_renderable &&
                    compiled.stages.size() == 1U && compiled.stages.front().processcvu.has_value(),
                "single-channel GRAY8 Preproc contract must compile without diagnostics");
        const auto& input_shapes = compiled.stages.front().processcvu->payload.input_shapes;
        require(input_shapes.size() == 1U && input_shapes.front().size() == 3U &&
                    input_shapes.front().back() == 1,
                "compiled single-channel GRAY8 Preproc input must have one channel");
        const auto& output_shapes = compiled.stages.front().processcvu->payload.output_shapes;
        std::string output_shape_text;
        for (const auto& shape : output_shapes) {
          output_shape_text += "[";
          for (const int dim : shape)
            output_shape_text += std::to_string(dim) + ",";
          output_shape_text += "]";
        }
        require(output_shapes.size() == 1U && !output_shapes.front().empty() &&
                    output_shapes.front().back() == 1,
                "compiled single-channel GRAY8 Preproc output must have one channel, got " +
                    output_shape_text);
      }

#ifdef SIMA_NEAT_INTERNAL
      {
        const auto fixture = make_preproc_fixture("preproc_default_capacity_seed_rebind");
        Model::Options model_opt;
        model_opt.preprocess.kind = InputKind::Image;
        model_opt.preprocess.enable = AutoFlag::On;
        model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
        model_opt.preprocess.resize.enable = AutoFlag::On;
        model_opt.preprocess.resize.width = 640;
        model_opt.preprocess.resize.height = 640;
        model_opt.preprocess.resize.mode = ResizeMode::Letterbox;
        Model model(fixture.tar_path, model_opt);

        InputOptions seed;
        seed.payload_type = PayloadType::Image;
        seed.format = FormatTag::RGB;
        seed.width = 1280;
        seed.height = 720;
        seed.depth = 3;
        const auto seeded_nodes =
            internal::ModelAccess::build_preprocess_nodes_for_input(model, seed, false);
        std::shared_ptr<Preproc> seeded_preproc;
        for (const auto& candidate : seeded_nodes) {
          seeded_preproc = std::dynamic_pointer_cast<Preproc>(candidate);
          if (seeded_preproc) {
            break;
          }
        }
        require(seeded_preproc != nullptr, "seeded model route must contain Preproc");
        require(seeded_preproc->options().input_width() == 1920 &&
                    seeded_preproc->options().input_height() == 1080,
                "seeded model Preproc must size its static shape from the default input capacity");

        InputContract seed_contract;
        seed_contract.media_type = "video/x-raw";
        seed_contract.format = "RGB";
        seed_contract.width = 1280;
        seed_contract.height = 720;
        seed_contract.depth = 3;
        seeded_preproc->apply_input_contract(seed_contract, nullptr);
        require(seeded_preproc->options().input_width() == 1280 &&
                    seeded_preproc->options().input_height() == 720,
                "seeded model Preproc must bind the seed as actual frame geometry");

        const auto seeded_max_shape =
            model_managed_preproc_max_input_shape(seeded_preproc->options());
        require(PreprocOptions::shape_dim(seeded_max_shape, 1) == 1920 &&
                    PreprocOptions::shape_dim(seeded_max_shape, 0) == 1080,
                "a 720p seed must retain the documented default 1920x1080 input capacity");

        pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
        const auto compiled =
            compile_node_contracts(std::vector<std::shared_ptr<Node>>{seeded_preproc},
                                   ContractCompileInput{}, &diagnostics);
        require(diagnostics.errors.empty() && compiled.fully_renderable &&
                    compiled.stages.size() == 1U &&
                    compiled.stages.front().processcvu.has_value() &&
                    compiled.stages.front().processcvu->payload.input_shapes.size() == 1U &&
                    compiled.stages.front().processcvu->payload.input_shapes.front() ==
                        std::vector<int>({1080, 1920, 3}),
                "seeded Preproc must compile its static envelope at the default input capacity");

        InputContract later_contract;
        later_contract.media_type = "video/x-raw";
        later_contract.format = "RGB";
        later_contract.width = 1920;
        later_contract.height = 1080;
        later_contract.depth = 3;
        seeded_preproc->apply_input_contract(later_contract, nullptr);
        require(seeded_preproc->options().input_width() == 1920 &&
                    seeded_preproc->options().input_height() == 1080,
                "a default-capacity model seeded at 720p must admit a later 1080p frame");
      }
#endif

      {
        PreprocOptions opt;
        opt.model_managed_contract = true;
        opt.set_input_shape({720, 1280, 3});
        opt.input_img_type = "RGB";
        opt.set_output_shape({640, 640, 3});
        opt.scaled_width = 640;
        opt.scaled_height = 640;
        opt.output_img_type = "RGB";
        opt.output_dtype = "EVXX_BFLOAT16";
        opt.normalize = true;
        opt.tessellate = false;

        Preproc node(opt);
        InputContract contract;
        contract.media_type = "video/x-raw";
        contract.format = "BGR";
        contract.width = 1280;
        contract.height = 720;
        contract.depth = 3;
        node.apply_input_contract(contract, nullptr);

        require(
            node.options().input_img_type == "RGB",
            "model-managed Preproc must preserve resolved input format over upstream heuristics");
        require(node.options().input_width() == 1280 && node.options().input_height() == 720,
                "model-managed Preproc must bind actual input dimensions from upstream contract");
        const auto* cfg = node.config_json();
        require(cfg != nullptr && cfg->contains("input_img_type") &&
                    (*cfg)["input_img_type"].get<std::string>() == "RGB",
                "model-managed Preproc config must preserve resolved input_img_type");
      }

      {
        PreprocOptions opt;
        opt.model_managed_contract = true;
        opt.set_input_shape({1080, 1920, 3});
        set_model_managed_preproc_max_input_shape(&opt, shape3(1080, 1920, 3));
        opt.input_img_type = "RGB";
        opt.set_output_shape({640, 640, 3});
        opt.scaled_width = 640;
        opt.scaled_height = 640;
        opt.output_img_type = "RGB";
        opt.output_dtype = "EVXX_BFLOAT16";
        opt.normalize = true;
        opt.tessellate = false;

        Preproc node(opt);
        InputContract contract;
        contract.media_type = "video/x-raw";
        contract.format = "RGB";
        contract.width = 256;
        contract.height = 256;
        contract.depth = 3;
        node.apply_input_contract(contract, nullptr);

        require(node.options().input_width() == 256 && node.options().input_height() == 256,
                "model-managed Preproc must treat upstream contract as actual geometry");
        const auto max_shape = model_managed_preproc_max_input_shape(node.options());
        require(PreprocOptions::shape_dim(max_shape, 1) == 1920 &&
                    PreprocOptions::shape_dim(max_shape, 0) == 1080,
                "model-managed Preproc must preserve internal max input shape as capacity");
      }

      {
        PreprocOptions opt;
        opt.model_managed_contract = true;
        opt.set_input_shape({1080, 1920, 3});
        set_model_managed_preproc_max_input_shape(&opt, shape3(1080, 1920, 3));
        opt.input_img_type = "RGB";
        opt.set_output_shape({640, 640, 3});
        opt.scaled_width = 640;
        opt.scaled_height = 640;
        opt.output_img_type = "RGB";
        opt.output_dtype = "EVXX_BFLOAT16";
        opt.normalize = true;
        opt.tessellate = false;

        Preproc node(opt);
        InputContract contract;
        contract.media_type = "video/x-raw";
        contract.format = "RGB";
        contract.width = 2048;
        contract.height = 1080;
        contract.depth = 3;
        bool threw = false;
        try {
          node.apply_input_contract(contract, nullptr);
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "exceeds max_input_width",
                           "capacity violation should mention max_input_width");
        }
        require(threw, "model-managed Preproc must reject actual geometry beyond capacity");
      }

      {
        PreprocOptions opt;
        opt.set_input_shape({17, 13});
        opt.input_img_type = "RGB";
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT16";
        opt.tessellate = false;

        Preproc node(opt);
        const OutputSpec out = node.output_spec(make_rgb_input_spec(1920, 1080));
        const std::string frag = node.backend_fragment(0);
        require(out.width == 640 && out.height == 640,
                "standalone Preproc should preserve explicit output size");
        require(node.options().input_width() == 1920 && node.options().input_height() == 1080,
                "standalone Preproc must derive input width/height from actual upstream input");
        require(node.options().input_img_type == "BGR",
                "standalone Preproc must derive input format from actual upstream input");
        require_contains(frag, "neatkernelrunner name=", "Preproc must use Runtime by default");
        require_not_contains(frag,
                             "num-buffers=", "runner does not accept legacy buffer properties");
        require_not_contains(
            frag, "stage-id=", "Preproc identity travels through the typed Runtime context");
        require_not_contains(
            frag, "config=", "standalone Preproc fragment must not emit a legacy config path");
      }

      {
        PreprocOptions opt;
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT16";
        opt.tessellate = true;

        Preproc node(opt);
        bool threw = false;
        try {
          (void)node.output_spec(make_rgb_input_spec(1280, 720));
        } catch (const std::exception& e) {
          threw = true;
          require_contains(std::string(e.what()), "slice_shape",
                           "standalone tess Preproc must hard fail without explicit tile geometry");
        }
        require(threw,
                "standalone tessellated Preproc must hard fail when tile geometry is missing");
      }

      {
        PreprocOptions opt;
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT16";
        opt.tessellate = false;
        opt.set_slice_shape({32, 128, 3});

        Preproc node(opt);
        const OutputSpec out = node.output_spec(make_rgb_input_spec(1280, 720));
        require(
            out.width == 640 && out.height == 640,
            "standalone Preproc should still produce a valid output contract when tessellate=false "
            "and tile geometry is provided");
      }

      {
        PreprocOptions opt;
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT8";
        opt.tessellate = false;

        Preproc node(opt);
        const OutputSpec out = node.output_spec(make_rgb_input_spec(1280, 720));
        require(out.width == 640 && out.height == 640,
                "standalone quantized Preproc should still produce a valid output contract without "
                "explicit quant params");
      }

      {
        PreprocOptions opt;
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT16";
        opt.tessellate = false;
        opt.q_scale = 0.25;
        opt.q_zp = -7;

        Preproc node(opt);
        (void)node.output_spec(make_rgb_input_spec(1280, 720));
      }

      {
        PreprocOptions opt;
        opt.set_output_shape({640, 640});
        opt.output_img_type = "RGB";
        opt.output_dtype = "INT8";
        opt.tessellate = false;
        opt.q_scale = 0.5;
        opt.q_zp = -9;

        Preproc node(opt);
        InputContract contract;
        contract.media_type = "video/x-raw";
        contract.format = "BGR";
        contract.width = 1280;
        contract.height = 720;
        contract.depth = 3;
        node.apply_input_contract(contract, nullptr);

        const auto* cfg = node.config_json();
        require(cfg != nullptr,
                "standalone quantized Preproc should materialize config from input contract");
        require((*cfg).value("q_scale", 0.0) == 0.5,
                "standalone Preproc should serialize explicit q_scale");
        require((*cfg).value("q_zp", 0) == -9, "standalone Preproc should serialize explicit q_zp");
      }

      // An existing positional ResizeSpec initializer keeps its meaning; placement is appended.
      {
        const ResizeSpec legacy{AutoFlag::On, 640, 480, ResizeMode::Letterbox, 114, "BILINEAR"};
        require(legacy.scaling_type == "BILINEAR" && legacy.pad_value == 114 &&
                    legacy.placement == LetterboxPlacement::Center && !legacy.interpolation &&
                    !legacy.shorter_side,
                "positional ResizeSpec initializer changed meaning");
      }

      // ResizeSpec placement reaches the frame adapter through the model route:
      // a 1280x720 source on the 640x640 canvas lands at the origin only for TopLeft.
      {
        const auto fixture = make_preproc_fixture("preproc_letterbox_placement");
        for (const auto placement : {LetterboxPlacement::Center, LetterboxPlacement::TopLeft}) {
          Model::Options model_opt;
          model_opt.preprocess.kind = InputKind::Image;
          model_opt.preprocess.enable = AutoFlag::On;
          model_opt.preprocess.input_max_width = 1280;
          model_opt.preprocess.input_max_height = 720;
          model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
          model_opt.preprocess.resize.enable = AutoFlag::On;
          model_opt.preprocess.resize.mode = ResizeMode::Letterbox;
          model_opt.preprocess.resize.pad_value = 116;
          model_opt.preprocess.resize.placement = placement;
          Model model(fixture.tar_path, model_opt);
          PreprocOptions opt(model);
          const bool top_left = placement == LetterboxPlacement::TopLeft;
          require(opt.aspect_ratio && opt.padding_type == (top_left ? "TOPLEFT" : "CENTER") &&
                      opt.pad_value == 116,
                  "model letterbox placement did not reach PreprocOptions");
#ifdef SIMA_NEAT_INTERNAL
          require(opt.compiled_contract != nullptr &&
                      opt.compiled_contract->payload.padding_type == opt.padding_type,
                  "admitted Preproc contract lost the letterbox placement");
          using Adapter = pipeline_internal::sima::kernel_lowering::PreprocFrameAdapter;
          auto policy = opt.compiled_contract->payload;
          if (policy.resolved_exec_backend.empty())
            policy.resolved_exec_backend = "EVXX";
          auto checked = Adapter::check_policy(policy);
          require(std::holds_alternative<std::shared_ptr<const Adapter::CheckedPolicy>>(checked),
                  "frame adapter rejected the model letterbox placement");
          auto frozen = Adapter::checked_initial_preprocess_facts(
              std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked)));
          require(std::holds_alternative<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen),
                  "frame adapter could not publish initial letterbox facts");
          const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(
              std::get<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen));
          require(facts && facts->original_width == 1280 && facts->original_height == 720 &&
                      facts->placed[2] == 640 && facts->placed[3] == 360 && facts->placed[0] == 0 &&
                      facts->placed[1] == (top_left ? 0 : 140) && facts->affine[0] == 2.0 &&
                      facts->affine[4] == 2.0 && facts->affine[2] == 0.0 &&
                      facts->affine[5] == (top_left ? 0.0 : -280.0),
                  "letterbox placement published the wrong rectangle or affine");
#endif
        }
      }

      // A Model's top-left letterbox rounds the shorter side to the nearest pixel, as Segment
      // Anything and its BoxDecode crop do: int(side * 1024 / longest + 0.5) in double. A
      // centered letterbox keeps the legacy upward rounding (683.2 becomes 684).
      {
        const auto fixture = make_preproc_fixture("preproc_letterbox_rounding", 1024);
        struct Case {
          int width, height;
          LetterboxPlacement placement;
          std::int64_t scaled_width, scaled_height, pad_left, pad_top, pad_right, pad_bottom;
        };
        constexpr auto top_left = LetterboxPlacement::TopLeft;
        constexpr auto center = LetterboxPlacement::Center;
        for (const auto& c : std::vector<Case>{{640, 427, top_left, 1024, 683, 0, 0, 0, 341},
                                               {427, 640, top_left, 683, 1024, 0, 0, 341, 0},
                                               {1280, 720, top_left, 1024, 576, 0, 0, 0, 448},
                                               {1001, 667, top_left, 1024, 682, 0, 0, 0, 342},
                                               {667, 1001, top_left, 682, 1024, 0, 0, 342, 0},
                                               {1920, 1081, top_left, 1024, 577, 0, 0, 0, 447},
                                               {640, 427, center, 1024, 684, 0, 170, 0, 170},
                                               {427, 640, center, 684, 1024, 170, 0, 170, 0},
                                               {1001, 667, center, 1024, 683, 0, 170, 0, 171}}) {
          Model::Options model_opt;
          model_opt.preprocess.kind = InputKind::Image;
          model_opt.preprocess.enable = AutoFlag::On;
          model_opt.preprocess.input_max_width = c.width;
          model_opt.preprocess.input_max_height = c.height;
          model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::RGB;
          model_opt.preprocess.resize.enable = AutoFlag::On;
          model_opt.preprocess.resize.mode = ResizeMode::Letterbox;
          model_opt.preprocess.resize.pad_value = 116;
          model_opt.preprocess.resize.placement = c.placement;
          Model model(fixture.tar_path, model_opt);
          PreprocOptions opt(model);
          const std::string label = std::to_string(c.width) + "x" + std::to_string(c.height) +
                                    (c.placement == top_left ? " top-left" : " centered");
          require(opt.padding_type == (c.placement == top_left ? "TOPLEFT" : "CENTER"),
                  label + ": model letterbox placement did not reach PreprocOptions");
#ifdef SIMA_NEAT_INTERNAL
          require(opt.compiled_contract != nullptr &&
                      opt.compiled_contract->payload.letterbox_nearest_extent ==
                          (c.placement == top_left),
                  label + ": only the Model's top-left letterbox rounds to the nearest pixel");
          using Adapter = pipeline_internal::sima::kernel_lowering::PreprocFrameAdapter;
          auto policy = opt.compiled_contract->payload;
          if (policy.resolved_exec_backend.empty())
            policy.resolved_exec_backend = "EVXX";
          auto checked = Adapter::check_policy(policy);
          require(std::holds_alternative<std::shared_ptr<const Adapter::CheckedPolicy>>(checked),
                  label + ": frame adapter rejected the model letterbox");
          auto frozen = Adapter::checked_initial_preprocess_facts(
              std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked)));
          require(std::holds_alternative<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen),
                  label + ": frame adapter could not publish initial letterbox facts");
          const auto* facts = ::simaai::gst::tensor_buffer_preprocess_facts(
              std::get<::simaai::gst::TensorBufferPreprocessFactsOwner>(frozen));
          require(facts != nullptr && facts->original_width == c.width &&
                      facts->original_height == c.height && facts->resized_width == 1024 &&
                      facts->resized_height == 1024,
                  label + ": letterbox facts lost the source or canvas size");
          const auto pad_right = facts->resized_width - facts->placed[0] - facts->placed[2];
          const auto pad_bottom = facts->resized_height - facts->placed[1] - facts->placed[3];
          require(facts->placed[0] == c.pad_left && facts->placed[1] == c.pad_top &&
                      facts->placed[2] == c.scaled_width && facts->placed[3] == c.scaled_height &&
                      pad_right == c.pad_right && pad_bottom == c.pad_bottom,
                  label + ": letterbox published " + std::to_string(facts->placed[2]) + "x" +
                      std::to_string(facts->placed[3]) + " at (" +
                      std::to_string(facts->placed[0]) + ", " + std::to_string(facts->placed[1]) +
                      "), expected " + std::to_string(c.scaled_width) + "x" +
                      std::to_string(c.scaled_height) + " at (" + std::to_string(c.pad_left) +
                      ", " + std::to_string(c.pad_top) + ")");
          require(facts->affine[0] == double(c.width) / double(c.scaled_width) &&
                      facts->affine[4] == double(c.height) / double(c.scaled_height) &&
                      facts->affine[2] == -double(c.pad_left) * facts->affine[0] &&
                      facts->affine[5] == -double(c.pad_top) * facts->affine[4],
                  label + ": letterbox affine does not map the placed image to the source");
#endif
        }
      }
    }));
