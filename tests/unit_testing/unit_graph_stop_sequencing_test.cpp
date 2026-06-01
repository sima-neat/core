#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key), had_(false) {
    const char* cur = std::getenv(key_);
    if (cur && *cur) {
      had_ = true;
      old_ = cur;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_) {
      ::setenv(key_, old_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_;
  std::string old_;
};

struct BlockState {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> entered{false};
  bool release = false;
};

class BlockingStage final : public simaai::neat::graph::StageExecutor {
public:
  explicit BlockingStage(std::shared_ptr<BlockState> state) : state_(std::move(state)) {}

  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    state_->entered.store(true, std::memory_order_release);
    state_->cv.notify_all();

    std::unique_lock<std::mutex> lock(state_->mu);
    state_->cv.wait(lock, [&] { return state_->release; });

    out.push_back(simaai::neat::graph::StageOutMsg{
        .out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                        ? simaai::neat::graph::kInvalidPort
                        : out_port_,
        .sample = std::move(msg.sample)});
  }

private:
  std::shared_ptr<BlockState> state_;
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node>
make_blocking_stage(const std::shared_ptr<BlockState>& s) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = [s] { return std::make_unique<BlockingStage>(s); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("BlockingStage", std::move(factory), std::move(inputs),
                                     std::move(outputs), "blocking");
}

bool wait_for_entered(const std::shared_ptr<BlockState>& s, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (s->entered.load(std::memory_order_acquire))
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return s->entered.load(std::memory_order_acquire);
}

} // namespace

RUN_TEST("unit_graph_stop_sequencing_test", ([] {
           using namespace simaai::neat;

           EnvVarGuard stop_timeout("SIMA_GRAPH_STOP_TIMEOUT_MS", "1");

           auto block_state = std::make_shared<BlockState>();

           graph::Graph g;
           const auto pipe_in = g.add(
               std::make_shared<graph::nodes::PipelineNode>(nodes::VideoConvert(), "convert"));
           const auto stage = g.add(make_blocking_stage(block_state));
           const auto pipe_out =
               g.add(std::make_shared<graph::nodes::PipelineNode>(nodes::Output(), "sink"));

           g.connect(pipe_in, stage);
           g.connect(stage, pipe_out);
           graph::GraphRunOptions run_opt;
           run_opt.edge_queue = 8;
           run_opt.push_timeout_ms = 250;
           run_opt.pull_timeout_ms = 20;

           graph::GraphRun run = simaai::neat::graph::build(std::move(g), run_opt);

           Sample sample;
           sample.kind = SampleKind::Tensor;
           sample.tensor = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x27);
           sample.frame_id = 17;
           sample.stream_id = "graph-stop-sequencing";

           require(run.push(pipe_in, simaai::neat::Sample{sample}),
                   "graph stop sequencing: initial push failed");
           require(wait_for_entered(block_state, 3000),
                   "graph stop sequencing: blocking stage did not receive input");

           const auto t0 = std::chrono::steady_clock::now();
           run.stop();
           const auto t1 = std::chrono::steady_clock::now();

           const int stop_ms = static_cast<int>(
               std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
           require(stop_ms < 3000, "graph stop sequencing: stop() exceeded expected bound");

           {
             std::lock_guard<std::mutex> lock(block_state->mu);
             block_state->release = true;
           }
           block_state->cv.notify_all();

           // idempotent stop should remain safe after stage release.
           run.stop();
         }));
