#include "model/Model.h"
#include "model_archive_fixture_utils.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/Quant.h"
#include "test_main.h"
#include "test_utils.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

sima_test::ModelArchiveFixture make_preproc_fixture(const std::string& tag) {
  return sima_test::make_model_archive_fixture(tag,
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
      "size": 2764800,
      "input_range": [0.0, 1.0],
      "logical_shape": [1, 720, 1280, 3],
      "logical_dtype": "UINT8"
    }
  ],
  "plugins": [
    {
      "name": "preproc_0",
      "sequence": 1,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "preproc",
        "params": {
          "input_shapes": [[1, 720, 1280, 3]],
          "output_shapes": [[1, 640, 640, 3]],
          "input_dtype": ["UINT8"],
          "output_dtype": "BF16"
        }
      },
      "input_nodes": [{"name": "images", "size": 2764800}],
      "output_nodes": [{
        "name": "preproc_0",
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
        "input_shapes": [[1, 640, 640, 3]],
        "input_data_type": ["BF16"],
        "output_shapes": [[1, 80, 80, 6]],
        "data_type": ["BF16"]
      },
      "input_nodes": [{
        "name": "preproc_0",
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
                                               true);
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

RUN_TEST("unit_preproc_contract_rules_test", [] {
  using namespace simaai::neat;

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
      require_contains(std::string(e.what()), "Pass preprocess options to Model",
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
      require_contains(std::string(e.what()), "exceeds max_input_width",
                       "PreprocOptions(Model) capacity violation should mention max_input_width");
    }
    require(capacity_threw, "PreprocOptions(Model) must reject inputs above modelpack capacity");
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
    require(opt.input_width() == 1920 && opt.input_height() == 1080 && opt.input_channels() == 3,
            "model-managed Preproc must project the configured capacity into its static input "
            "shape before a smaller seed is bound");
#ifdef SIMA_NEAT_INTERNAL
    const auto initial_max_shape = model_managed_preproc_max_input_shape(opt);
    require(PreprocOptions::shape_dim(initial_max_shape, 1) == 1920 &&
                PreprocOptions::shape_dim(initial_max_shape, 0) == 1080 &&
                PreprocOptions::shape_channels(initial_max_shape) == 3,
            "model-managed Preproc must retain the configured capacity in model lineage");
#endif

    Preproc node(opt);
    InputContract seed_contract;
    seed_contract.media_type = "video/x-raw";
    seed_contract.format = "RGB";
    seed_contract.width = 1280;
    seed_contract.height = 720;
    seed_contract.depth = 3;
    node.apply_input_contract(seed_contract, nullptr);
    require(node.options().input_width() == 1280 && node.options().input_height() == 720,
            "model-managed Preproc must bind the smaller seed as actual geometry");

    InputContract capacity_contract = seed_contract;
    capacity_contract.width = 1920;
    capacity_contract.height = 1080;
    node.apply_input_contract(capacity_contract, nullptr);
    require(node.options().input_width() == 1920 && node.options().input_height() == 1080,
            "model-managed Preproc must accept a later contract up to its configured capacity");
#ifdef SIMA_NEAT_INTERNAL
    const auto rebound_max_shape = model_managed_preproc_max_input_shape(node.options());
    require(PreprocOptions::shape_dim(rebound_max_shape, 1) == 1920 &&
                PreprocOptions::shape_dim(rebound_max_shape, 0) == 1080,
            "runtime contract rebinding must not shrink the model-managed capacity");
#endif
  }

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

    require(node.options().input_img_type == "RGB",
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
    require_not_contains(
        frag, "stage-id=", "standalone Preproc fragment must not opt into manifest routing");
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
    require(threw, "standalone tessellated Preproc must hard fail when tile geometry is missing");
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
    require(out.width == 640 && out.height == 640,
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
});
