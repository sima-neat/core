#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

class FakeDownstreamKindNode final : public simaai::neat::Node {
public:
  explicit FakeDownstreamKindNode(
      std::string kind,
      simaai::neat::MemoryContract memory = simaai::neat::MemoryContract::AllowEitherButReport)
      : kind_(std::move(kind)), memory_(memory) {}

  std::string kind() const override {
    return kind_;
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }
  simaai::neat::MemoryContract memory_contract() const override {
    return memory_;
  }
  std::string backend_fragment(int node_index) const override {
    return "identity name=n" + std::to_string(node_index) + "_fake_downstream";
  }
  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_fake_downstream"};
  }

private:
  std::string kind_;
  simaai::neat::MemoryContract memory_;
};

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* key, const std::string& value) : key_(key), had_old_(false) {
    if (const char* old = std::getenv(key_.c_str())) {
      had_old_ = true;
      old_ = old;
    }
    setenv(key_.c_str(), value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (had_old_) {
      setenv(key_.c_str(), old_.c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

private:
  std::string key_;
  std::string old_;
  bool had_old_;
};

simaai::neat::Graph make_rgb_graph() {
  using namespace simaai::neat;

  Graph graph;
  InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  graph.add(nodes::Input(src_opt));
  graph.add(nodes::Output(OutputOptions::EveryFrame(64)));
  return graph;
}

simaai::neat::Sample tensor_to_sample(const simaai::neat::Tensor& tensor) {
  using namespace simaai::neat;
  Sample sample = sample_from_tensors(TensorList{tensor});
  sample.payload_tag = "RGB";
  sample.owned = true;
  return sample;
}

void require_tensor_outputs(const simaai::neat::TensorList& outputs, const std::string& where) {
  require(outputs.size() == 1U, where + ": expected one tensor output");
}

void require_tensor_sample_outputs(const simaai::neat::Sample& outputs, const std::string& where) {
  require(outputs.size() == 1U, where + ": expected one sample output");
  const auto& output = outputs.front();
  const bool has_single_tensor =
      (simaai::neat::sample_has_tensor_list(output) && output.tensors.size() == 1U) ||
      (output.kind == simaai::neat::SampleKind::Tensor && output.tensor.has_value());
  require(has_single_tensor, where + ": expected single-tensor sample output");
}

} // namespace

RUN_TEST(
    "unit_graph_build_input_strategy_test", ([] {
      using namespace simaai::neat;

      const ScopedEnvVar preflight("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "1");

      cv::Mat mat_seed(48, 64, CV_8UC3, cv::Scalar(20, 80, 140));
      Tensor tensor_seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x63);
      const Sample sample_seed = tensor_to_sample(tensor_seed);

      RunOptions run_opt;
      run_opt.queue_depth = 4;
      run_opt.overflow_policy = OverflowPolicy::Block;

      // Sync build path parity for Mat/Tensor/Sample.
      {
        Graph graph = make_rgb_graph();

        Run run_mat =
            graph.build_seeded_internal(std::vector<cv::Mat>{mat_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000), "sync build mat");
        run_mat.stop();

        Run run_tensor =
            graph.build_seeded_internal(TensorList{tensor_seed}, RunMode::Sync, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "sync build tensor");
        run_tensor.stop();

        Run run_sample = graph.build_seeded_internal(Sample{sample_seed}, RunMode::Sync, run_opt);
        require_tensor_sample_outputs(run_sample.run(Sample{sample_seed}, 1000),
                                      "sync build sample");
        run_sample.stop();
      }

      // Sync run cache parity for Mat/Tensor overloads.
      {
        Graph graph = make_rgb_graph();
        require_tensor_outputs(graph.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat first");
        require_tensor_outputs(graph.run(std::vector<cv::Mat>{mat_seed}, run_opt),
                               "sync run mat second");
        require_tensor_outputs(graph.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor first");
        require_tensor_outputs(graph.run(TensorList{tensor_seed}, run_opt),
                               "sync run tensor second");
      }

      // Async build parity for Mat/Tensor/Sample.
      {
        Graph graph = make_rgb_graph();

        Run run_mat = graph.build(std::vector<cv::Mat>{mat_seed}, run_opt);
        require_tensor_outputs(run_mat.run(std::vector<cv::Mat>{mat_seed}, 1000),
                               "async build mat");
        run_mat.stop();

        Run run_tensor = graph.build(TensorList{tensor_seed}, run_opt);
        require_tensor_outputs(run_tensor.run(TensorList{tensor_seed}, 1000), "async build tensor");
        run_tensor.stop();

        Run run_sample = graph.build(Sample{sample_seed}, run_opt);
        require_tensor_sample_outputs(run_sample.run(Sample{sample_seed}, 1000),
                                      "async build sample");
        run_sample.stop();
      }

      // Allocation preferences must not become admission requirements. Keep explicit
      // policies and the legacy pool opt-out authoritative without starting device plugins.
      {
        struct MemoryCase {
          const char* label;
          InputMemoryPolicy declared;
          bool use_pool;
          const char* downstream;
          MemoryContract contract;
          InputMemoryPolicy allocation;
          bool require_device;
        };
        const MemoryCase cases[] = {
            {"Auto CPU", InputMemoryPolicy::Auto, true, "Identity",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::SystemMemory, false},
            {"Auto preference", InputMemoryPolicy::Auto, true, "GenericDeviceConsumer",
             MemoryContract::PreferDeviceZeroCopy, InputMemoryPolicy::Ev74, false},
            {"Auto Preproc", InputMemoryPolicy::Auto, true, "Preproc",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Auto Quant", InputMemoryPolicy::Auto, true, "Quant",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Auto Tess", InputMemoryPolicy::Auto, true, "Tess",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Auto QuantTess", InputMemoryPolicy::Auto, true, "QuantTess",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Auto CastTess", InputMemoryPolicy::Auto, true, "CastTess",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Auto MLA", InputMemoryPolicy::Auto, true, "ModelFragment",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Dms0, true},
            {"Pool opt-out", InputMemoryPolicy::Auto, false, "Preproc",
             MemoryContract::PreferDeviceZeroCopy, InputMemoryPolicy::SystemMemory, false},
            {"Explicit CPU", InputMemoryPolicy::SystemMemory, true, "Preproc",
             MemoryContract::PreferDeviceZeroCopy, InputMemoryPolicy::SystemMemory, false},
            {"Explicit EV74", InputMemoryPolicy::Ev74, false, "Identity",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Ev74, true},
            {"Explicit MLA", InputMemoryPolicy::Dms0, false, "Identity",
             MemoryContract::AllowEitherButReport, InputMemoryPolicy::Dms0, true},
        };
        for (const auto& test : cases) {
          InputOptions options;
          options.memory_policy = test.declared;
          options.use_simaai_pool = test.use_pool;
          const InputOptions declared = options;
          const std::vector<std::shared_ptr<Node>> graph_nodes = {
              nodes::Input(declared), nullptr, std::make_shared<FakeDownstreamKindNode>("Cast"),
              std::make_shared<FakeDownstreamKindNode>(test.downstream, test.contract)};
          const auto memory = session_test::resolve_input_memory_for_test(declared, graph_nodes);
          require(memory.allocation == test.allocation,
                  std::string(test.label) + ": allocation policy mismatch");
          require(memory.require_device_visible_input == test.require_device,
                  std::string(test.label) + ": admission requirement mismatch");

          runtime::PipelineSegmentPlan segment;
          segment.nodes = graph_nodes;
          segment.resolved_input_memory_policy = memory.allocation;
          const auto unseeded = runtime::pipeline_segment_ingress_input(segment);
          require(unseeded && unseeded->memory_policy == test.allocation &&
                      unseeded->payload_type == PayloadType::Auto && unseeded->format.empty() &&
                      unseeded->width == declared.width,
                  std::string(test.label) + ": unseeded source lost policy or invented media");

          segment.input_spec.media_type = "video/x-raw";
          segment.input_spec.format = "BGR";
          segment.input_spec.width = 1280;
          segment.input_spec.height = 720;
          segment.input_complete = true;
          const auto seeded = runtime::pipeline_segment_ingress_input(segment);
          require(seeded && seeded->memory_policy == test.allocation &&
                      seeded->payload_type == PayloadType::Image && seeded->format.str() == "BGR" &&
                      seeded->width == 1280 && seeded->height == 720,
                  std::string(test.label) + ": seeded ingress lost allocation or source geometry");

          segment.boundary_hints.emplace();
          InputOptions hint;
          hint.payload_type = PayloadType::Tensor;
          hint.format = "FP32";
          hint.width = 300;
          hint.memory_policy = InputMemoryPolicy::Dms0;
          segment.boundary_hints->ingress_inputs.push_back(hint);
          const auto hinted = runtime::pipeline_segment_ingress_input(segment);
          const auto hint_allocation =
              declared.memory_policy != InputMemoryPolicy::Auto || !declared.use_simaai_pool
                  ? test.allocation
                  : hint.memory_policy;
          require(hinted && hinted->memory_policy == hint_allocation &&
                      hinted->payload_type == hint.payload_type && hinted->format == hint.format &&
                      hinted->width == hint.width,
                  std::string(test.label) + ": source policy and port metadata were conflated");

          const auto* input = dynamic_cast<const Input*>(graph_nodes.front().get());
          require(input && input->options().memory_policy == test.declared &&
                      input->options().use_simaai_pool == test.use_pool,
                  std::string(test.label) + ": source declaration changed");
        }
        const InputOptions standalone;
        const auto memory =
            session_test::resolve_input_memory_for_test(standalone, {nodes::Input(standalone)});
        require(memory.allocation == InputMemoryPolicy::SystemMemory &&
                    !memory.require_device_visible_input,
                "standalone Input should not infer a device requirement");
      }

      // Per-port descriptors do not inherit a scalar segment allocation. Unknown
      // indices retain their metadata fallback, not another port's memory policy.
      {
        runtime::PipelineSegmentPlan segment;
        segment.nodes = {nodes::Input()};
        segment.resolved_input_memory_policy = InputMemoryPolicy::Ev74;
        segment.boundary_hints.emplace();
        InputOptions first;
        first.payload_type = PayloadType::Image;
        first.format = "RGB";
        first.width = 640;
        first.memory_policy = InputMemoryPolicy::Dms0;
        InputOptions second = first;
        second.format = "BGR";
        second.width = 320;
        second.memory_policy = InputMemoryPolicy::Auto;
        second.use_simaai_pool = false;
        segment.boundary_hints->ingress_inputs = {first, second};
        for (std::size_t i = 0; i < 2; ++i) {
          const auto actual = runtime::pipeline_segment_ingress_input(segment, i);
          const auto& expected = segment.boundary_hints->ingress_inputs[i];
          require(actual && actual->memory_policy == expected.memory_policy &&
                      actual->use_simaai_pool == expected.use_simaai_pool &&
                      actual->format == expected.format && actual->width == expected.width,
                  "indexed ingress must preserve its own policy and geometry");
        }
        require(!runtime::pipeline_segment_ingress_input(segment, 2),
                "missing indexed ingress must not acquire the public source descriptor");
        segment.input_complete = true;
        segment.input_spec.media_type = "video/x-raw";
        const auto unknown = runtime::pipeline_segment_ingress_input(segment, 2);
        require(unknown && unknown->memory_policy == InputMemoryPolicy::Auto,
                "unknown indexed fallback must not inherit the segment allocation");

        InputOptions explicit_source;
        explicit_source.memory_policy = InputMemoryPolicy::SystemMemory;
        segment.nodes = {nodes::Input(explicit_source)};
        const auto overridden = runtime::pipeline_segment_ingress_input(segment, 1);
        require(overridden && overridden->memory_policy == explicit_source.memory_policy &&
                    overridden->format == second.format,
                "aggregate source allocation must apply without replacing port interpretation");
        const auto unknown_explicit = runtime::pipeline_segment_ingress_input(segment, 2);
        require(unknown_explicit && unknown_explicit->memory_policy == InputMemoryPolicy::Auto,
                "unknown indexed fallback must not acquire an explicit source policy");
        segment.input_edges = {0};
        const auto boundary = runtime::pipeline_segment_ingress_input(segment);
        require(boundary && boundary->memory_policy == first.memory_policy,
                "internal Input must not override port allocation as a public source");

        segment.boundary_hints.reset();
        segment.input_complete = false;
        require(!runtime::pipeline_segment_ingress_input(segment),
                "unknown internal boundary must remain unspecified");
      }
    }));
