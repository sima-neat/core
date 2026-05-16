#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
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
#include <utility>
#include <vector>

// Verifies that StageExecutor can emit graph outputs while on_input() is still
// running, including terminal routing, downstream routing, and stop cancellation.
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
    simaai::neat::graph::Graph graph;
    const auto in_port = graph.intern_port("in");
    auto state = std::make_shared<StreamState>();
    const auto streamer = graph.add(make_blocking_emitter_node(state));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(streamer, in_port, make_text_sample("prompt", "go")),
            "push to terminal streaming stage failed");
    auto first = run.pull(streamer, 1000);
    require(first.has_value(), "live emitter output did not reach terminal sink");
    require(sample_text(*first) == "token-1", "unexpected first streamed token");
    require(state->entered.load(std::memory_order_acquire), "stage did not enter blocking region");
    require(!state->completed.load(std::memory_order_acquire),
            "stage completed before the live token was pulled");

    release_stage(state);
    auto final = run.pull(streamer, 1000);
    require(final.has_value(), "final returned output did not reach terminal sink");
    require(sample_text(*final) == "done", "unexpected final output");
    run.stop();
  }

  {
    simaai::neat::graph::Graph graph;
    const auto in_port = graph.intern_port("in");
    auto state = std::make_shared<StreamState>();
    const auto streamer = graph.add(make_blocking_emitter_node(state));
    const auto sink = graph.add(make_pass_node());
    graph.connect(streamer, sink, "out", "in");

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(streamer, in_port, make_text_sample("prompt", "go")),
            "push to downstream streaming stage failed");
    auto first = run.pull(sink, 1000);
    require(first.has_value(), "live emitter output did not reach downstream stage");
    require(sample_text(*first) == "token-1", "unexpected downstream streamed token");
    require(!state->completed.load(std::memory_order_acquire),
            "source stage completed before downstream live token was pulled");

    release_stage(state);
    auto final = run.pull(sink, 1000);
    require(final.has_value(), "final returned output did not reach downstream stage");
    require(sample_text(*final) == "done", "unexpected downstream final output");
    run.stop();
  }

  {
    simaai::neat::graph::Graph graph;
    const auto in_port = graph.intern_port("in");
    auto state = std::make_shared<StreamState>();
    const auto streamer = graph.add(make_blocking_emitter_node(state));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(streamer, in_port, make_text_sample("prompt", "go")),
            "push to stop streaming stage failed");
    require(wait_until([&] { return state->entered.load(std::memory_order_acquire); }, 1000),
            "stage did not enter blocking region before stop");
    run.stop();
    require(state->request_stop_called.load(std::memory_order_acquire),
            "GraphRun::stop did not call StageExecutor::request_stop");
  }
});
