#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Verifies that StageExecutor can emit graph outputs while on_input() is still
// running through the public Graph -> Run path, including terminal routing,
// downstream routing, and stop cancellation.
namespace {

struct StreamState {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> entered{false};
  std::atomic<bool> completed{false};
  std::atomic<bool> request_stop_called{false};
  bool release = false;
};

simaai::neat::Sample make_text_sample(const std::string& port, const std::string& text) {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample(port, simaai::neat::Tensor::from_text(text));
  sample.port_name = port;
  sample.stream_label = port;
  sample.stream_id = "stream";
  return sample;
}

std::string sample_text(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    require(sample.tensor.has_value(), "expected tensor payload");
    return sample.tensor->to_text();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    require(sample.tensors.size() == 1U, "expected one tensor payload");
    return sample.tensors.front().to_text();
  }
  throw std::runtime_error("expected tensor sample");
}

bool wait_until(const std::function<bool()>& pred, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return pred();
}

class BlockingEmitterStage final : public simaai::neat::graph::StageExecutor {
public:
  explicit BlockingEmitterStage(std::shared_ptr<StreamState> state) : state_(std::move(state)) {}

  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    out_ = ports.out_port("out");
  }

  void set_emitter(simaai::neat::graph::StageEmitter* emitter) override {
    emitter_ = emitter;
  }

  void request_stop() override {
    state_->request_stop_called.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(state_->mu);
      state_->release = true;
    }
    state_->cv.notify_all();
  }

  void on_input(simaai::neat::graph::StageMsg&&,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    require(emitter_ != nullptr, "stage emitter should be installed");
    require(emitter_->emit(simaai::neat::graph::StageOutMsg{
                .out_port = out_, .sample = make_text_sample("tokens", "token-1")}),
            "stage emitter failed to route token");

    state_->entered.store(true, std::memory_order_release);
    state_->cv.notify_all();

    std::unique_lock<std::mutex> lock(state_->mu);
    state_->cv.wait(lock, [&] { return state_->release; });
    state_->completed.store(true, std::memory_order_release);

    out.push_back(simaai::neat::graph::StageOutMsg{.out_port = out_,
                                                   .sample = make_text_sample("done", "done")});
  }

private:
  std::shared_ptr<StreamState> state_;
  simaai::neat::graph::StageEmitter* emitter_ = nullptr;
  simaai::neat::graph::PortId out_ = simaai::neat::graph::kInvalidPort;
};

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    out_ = ports.out_port("out");
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node>
make_blocking_emitter_node(const std::shared_ptr<StreamState>& state) {
  using simaai::neat::OutputSpec;
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;

  StageNode::StageExecutorFactory factory = [state] {
    return std::make_unique<BlockingEmitterStage>(state);
  };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
  return std::make_shared<StageNode>("BlockingEmitter", std::move(factory), std::move(inputs),
                                     std::move(outputs), "streamer");
}

std::shared_ptr<simaai::neat::graph::Node> make_pass_node() {
  using simaai::neat::OutputSpec;
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;

  StageNode::StageExecutorFactory factory = [] { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), "sink");
}

std::size_t add_input_endpoint(simaai::neat::Graph& graph, const char* name) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Tensor;
  return graph.append_pipeline_vertex_for_internal_graph_(
      std::make_shared<simaai::neat::Input>(std::string(name), std::move(opt)));
}

std::size_t add_output_endpoint(simaai::neat::Graph& graph, const char* name) {
  return graph.append_pipeline_vertex_for_internal_graph_(
      std::make_shared<simaai::neat::Output>(std::string(name)));
}

void connect_runtime(simaai::neat::Graph& graph, std::size_t from, const char* from_port,
                     std::size_t to, const char* to_port) {
  graph.connect_runtime_port_for_internal_graph_(from, from_port, to, to_port);
}

void release_stage(const std::shared_ptr<StreamState>& state) {
  {
    std::lock_guard<std::mutex> lock(state->mu);
    state->release = true;
  }
  state->cv.notify_all();
}

} // namespace

RUN_TEST("unit_graph_stage_emitter_test", [] {
  {
    simaai::neat::Graph graph;
    const auto input = add_input_endpoint(graph, "in");
    const auto stage = graph.append_runtime_vertex_for_internal_graph_(make_pass_node());
    const auto output = add_output_endpoint(graph, "out");
    connect_runtime(graph, input, "out", stage, "in");
    connect_runtime(graph, stage, "out", output, "in");

    simaai::neat::Run run = graph.build();

    simaai::neat::Sample input_sample = make_text_sample("prompt", "metadata");
    input_sample.frame_id = 42;
    input_sample.pts_ns = 123456789;
    input_sample.dts_ns = 123456000;
    input_sample.duration_ns = 33333;
    input_sample.input_seq = 7;
    input_sample.orig_input_seq = 6;
    input_sample.stream_id = "camera-0";
    input_sample.stream_label = "primary";

    require(run.push("in", input_sample), "push to metadata pass-through stage failed");
    auto out = run.pull("out", 1000);
    require(out.has_value(), "metadata pass-through stage output timed out");
    require(sample_text(*out) == "metadata", "metadata pass-through payload changed");
    require(out->frame_id == input_sample.frame_id, "runtime stage lost frame_id");
    require(out->pts_ns == input_sample.pts_ns, "runtime stage lost pts_ns");
    require(out->dts_ns == input_sample.dts_ns, "runtime stage lost dts_ns");
    require(out->duration_ns == input_sample.duration_ns, "runtime stage lost duration_ns");
    require(out->input_seq == input_sample.input_seq, "runtime stage lost input_seq");
    require(out->orig_input_seq == input_sample.orig_input_seq,
            "runtime stage lost orig_input_seq");
    require(out->stream_id == input_sample.stream_id, "runtime stage lost stream_id");
    require(out->stream_label == input_sample.stream_label, "runtime stage lost stream_label");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    const auto input = add_input_endpoint(graph, "in");
    const auto output_a = add_output_endpoint(graph, "left");
    const auto output_b = add_output_endpoint(graph, "right");
    connect_runtime(graph, input, "out", output_a, "in");
    connect_runtime(graph, input, "out", output_b, "in");

    simaai::neat::Run run = graph.build();

    require(run.push("in", make_text_sample("prompt", "fanout-token")),
            "push to direct input fanout failed");
    auto left = run.pull("left", 1000);
    auto right = run.pull("right", 1000);
    require(left.has_value(), "left direct fanout output timed out");
    require(right.has_value(), "right direct fanout output timed out");
    require(sample_text(*left) == "fanout-token", "left direct fanout text changed");
    require(sample_text(*right) == "fanout-token", "right direct fanout text changed");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    const auto input = add_input_endpoint(graph, "in");
    const auto stage_a = graph.append_runtime_vertex_for_internal_graph_(make_pass_node());
    const auto stage_b = graph.append_runtime_vertex_for_internal_graph_(make_pass_node());
    const auto output_a = add_output_endpoint(graph, "stage_left");
    const auto output_b = add_output_endpoint(graph, "stage_right");
    connect_runtime(graph, input, "out", stage_a, "in");
    connect_runtime(graph, input, "out", stage_b, "in");
    connect_runtime(graph, stage_a, "out", output_a, "in");
    connect_runtime(graph, stage_b, "out", output_b, "in");

    simaai::neat::Run run = graph.build();

    require(run.push("in", make_text_sample("prompt", "stage-fanout-token")),
            "push to direct input stage fanout failed");
    auto left = run.pull("stage_left", 1000);
    auto right = run.pull("stage_right", 1000);
    require(left.has_value(), "left stage fanout output timed out");
    require(right.has_value(), "right stage fanout output timed out");
    require(sample_text(*left) == "stage-fanout-token", "left stage fanout text changed");
    require(sample_text(*right) == "stage-fanout-token", "right stage fanout text changed");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    const auto input = add_input_endpoint(graph, "in");
    const auto output_a = add_output_endpoint(graph, "seeded_left");
    const auto output_b = add_output_endpoint(graph, "seeded_right");
    connect_runtime(graph, input, "out", output_a, "in");
    connect_runtime(graph, input, "out", output_b, "in");

    simaai::neat::Run run = graph.build(make_text_sample("prompt", "seed-token"));

    require(run.push("in", make_text_sample("prompt", "live-text")),
            "push after seeded direct input build failed");
    auto left = run.pull("seeded_left", 1000);
    auto right = run.pull("seeded_right", 1000);
    require(left.has_value(), "left seeded direct fanout output timed out");
    require(right.has_value(), "right seeded direct fanout output timed out");
    require(sample_text(*left) == "live-text", "left seeded direct fanout text changed");
    require(sample_text(*right) == "live-text", "right seeded direct fanout text changed");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    auto state = std::make_shared<StreamState>();
    const auto input = add_input_endpoint(graph, "in");
    const auto streamer =
        graph.append_runtime_vertex_for_internal_graph_(make_blocking_emitter_node(state));
    const auto output = add_output_endpoint(graph, "out");
    connect_runtime(graph, input, "out", streamer, "in");
    connect_runtime(graph, streamer, "out", output, "in");

    simaai::neat::Run run = graph.build();

    require(run.push("in", make_text_sample("prompt", "go")),
            "push to terminal streaming stage failed");
    auto first = run.pull("out", 1000);
    require(first.has_value(), "live emitter output did not reach terminal sink");
    require(sample_text(*first) == "token-1", "unexpected first streamed token");
    require(wait_until([&] { return state->entered.load(std::memory_order_acquire); }, 3000),
            "stage did not enter blocking region");
    require(!state->completed.load(std::memory_order_acquire),
            "stage completed before the live token was pulled");

    release_stage(state);
    auto final = run.pull("out", 1000);
    require(final.has_value(), "final returned output did not reach terminal sink");
    require(sample_text(*final) == "done", "unexpected final output");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    auto state = std::make_shared<StreamState>();
    const auto input = add_input_endpoint(graph, "in");
    const auto streamer =
        graph.append_runtime_vertex_for_internal_graph_(make_blocking_emitter_node(state));
    const auto sink = graph.append_runtime_vertex_for_internal_graph_(make_pass_node());
    const auto output = add_output_endpoint(graph, "out");
    connect_runtime(graph, input, "out", streamer, "in");
    connect_runtime(graph, streamer, "out", sink, "in");
    connect_runtime(graph, sink, "out", output, "in");

    simaai::neat::Run run = graph.build();

    require(run.push("in", make_text_sample("prompt", "go")),
            "push to downstream streaming stage failed");
    auto first = run.pull("out", 1000);
    require(first.has_value(), "live emitter output did not reach downstream stage");
    require(sample_text(*first) == "token-1", "unexpected downstream streamed token");
    require(!state->completed.load(std::memory_order_acquire),
            "source stage completed before downstream live token was pulled");

    release_stage(state);
    auto final = run.pull("out", 1000);
    require(final.has_value(), "final returned output did not reach downstream stage");
    require(sample_text(*final) == "done", "unexpected downstream final output");
    run.stop();
  }

  {
    simaai::neat::Graph graph;
    auto state = std::make_shared<StreamState>();
    const auto input = add_input_endpoint(graph, "in");
    const auto streamer =
        graph.append_runtime_vertex_for_internal_graph_(make_blocking_emitter_node(state));
    const auto output = add_output_endpoint(graph, "out");
    connect_runtime(graph, input, "out", streamer, "in");
    connect_runtime(graph, streamer, "out", output, "in");

    simaai::neat::Run run = graph.build();

    require(run.push("in", make_text_sample("prompt", "go")),
            "push to stop streaming stage failed");
    require(wait_until([&] { return state->entered.load(std::memory_order_acquire); }, 1000),
            "stage did not enter blocking region before stop");
    run.stop();
    require(state->request_stop_called.load(std::memory_order_acquire),
            "Run::stop did not call StageExecutor::request_stop");
  }
});
