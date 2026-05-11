#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/nodes/StageModelExecutor.h"
#include "model/Model.h"
#include "asset_utils.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstring>

namespace {

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

simaai::neat::Tensor make_fp32_tensor(int w, int h, int d) {
  const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                            static_cast<std::size_t>(d) * sizeof(float);
  auto storage = simaai::neat::make_cpu_owned_storage(bytes);
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, d};
  t.strides_bytes = contiguous_strides_bytes(t.shape, static_cast<int64_t>(sizeof(float)));
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

sima_test::MpkFixture make_fixture() {
  return sima_test::make_strict_mpk_tar_fixture(
      "hybrid_graph_stage_model",
      {
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
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB"
})json"},
          {"etc/0_process_mla.json",
           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [64],
  "output_height": [48],
  "output_depth": [3]
})json"},
      },
      true);
}

} // namespace

RUN_TEST("hybrid_graph_stage_model_test", [] {
  const auto fixture = make_fixture();
  const std::string tar = fixture.tar_path;

  auto model = std::make_shared<simaai::neat::Model>(tar);

  simaai::neat::graph::nodes::StageModelExecutorOptions opt;
  opt.model = model;
  opt.do_preproc = false;
  opt.do_mla = false;
  opt.do_boxdecode = false;

  simaai::neat::graph::Graph g;
  auto node_id = g.add(simaai::neat::graph::nodes::StageModelExecutorNode(opt, "stage_model"));

  simaai::neat::graph::GraphSession session(std::move(g));
  simaai::neat::graph::GraphRun run = session.build();

  simaai::neat::InputOptions tensor_opt = model->input_appsrc_options(true);
  const int input_w = (tensor_opt.width > 0) ? tensor_opt.width : tensor_opt.max_width;
  const int input_h = (tensor_opt.height > 0) ? tensor_opt.height : tensor_opt.max_height;
  const int input_d = (tensor_opt.depth > 0) ? tensor_opt.depth : tensor_opt.max_depth;
  require(input_w > 0 && input_h > 0 && input_d > 0,
          "Model missing tensor input dims and max input limits");

  simaai::neat::Sample input;
  input.kind = simaai::neat::SampleKind::Tensor;
  input.tensor = make_fp32_tensor(input_w, input_h, input_d);
  input.frame_id = 1;
  input.stream_id = "model";

  require(run.push(node_id, input), "GraphRun::push failed");
  auto out = run.pull(node_id, 2000);
  require(out.has_value(), "GraphRun::pull timed out");
  require(out->tensor.has_value(), "StageModelExecutor output missing simaai::neat::Tensor");
  require(out->tensor->shape == input.tensor->shape, "Output shape mismatch");
  require(out->tensor->dtype == input.tensor->dtype, "Output dtype mismatch");

  run.stop();
});
