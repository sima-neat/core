#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "graph/nodes/StreamScheduler.h"
#include "test_main.h"
#include "test_utils.h"

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace {

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    const simaai::neat::graph::PortId out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                                                     ? simaai::neat::graph::kInvalidPort
                                                     : out_port_;
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node> make_pass_node(const std::string& label) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

simaai::neat::Sample make_sample(const std::string& stream_id, int frame_id) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(8, 6, simaai::neat::ImageSpec::PixelFormat::RGB);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  return sample;
}

struct ExpectedSample {
  std::string stream_id;
  int frame_id = -1;
};

class ExpectedScheduler {
public:
  ExpectedScheduler(int per_stream_queue, simaai::neat::graph::nodes::StreamDropPolicy drop_policy)
      : per_stream_queue_(per_stream_queue), drop_policy_(drop_policy) {}

  ExpectedSample push_and_emit(const std::string& stream_id, int frame_id) {
    const std::string key = stream_id.empty() ? "stream0" : stream_id;
    auto& q = queues_[key];
    const bool has_limit = per_stream_queue_ > 0;
    if (has_limit && q.size() >= static_cast<std::size_t>(per_stream_queue_)) {
      if (drop_policy_ == simaai::neat::graph::nodes::StreamDropPolicy::DropOldest) {
        q.pop_front();
      } else {
        return ExpectedSample{};
      }
    }
    q.push_back(ExpectedSample{key, frame_id});
    ensure_stream_(key);
    return emit_one_();
  }

private:
  void ensure_stream_(const std::string& stream_id) {
    if (stream_id.empty())
      return;
    if (active_.insert(stream_id).second) {
      rr_order_.push_back(stream_id);
    }
  }

  ExpectedSample emit_one_() {
    if (rr_order_.empty())
      return ExpectedSample{};
    std::size_t attempts = rr_order_.size();
    while (attempts-- > 0 && !rr_order_.empty()) {
      std::string stream_id = rr_order_.front();
      rr_order_.pop_front();

      auto it = queues_.find(stream_id);
      if (it == queues_.end() || it->second.empty()) {
        active_.erase(stream_id);
        continue;
      }

      ExpectedSample sample = it->second.front();
      it->second.pop_front();

      if (!it->second.empty()) {
        rr_order_.push_back(stream_id);
      } else {
        active_.erase(stream_id);
      }

      return sample;
    }
    return ExpectedSample{};
  }

  int per_stream_queue_ = 0;
  simaai::neat::graph::nodes::StreamDropPolicy drop_policy_;
  std::unordered_map<std::string, std::deque<ExpectedSample>> queues_;
  std::deque<std::string> rr_order_;
  std::unordered_set<std::string> active_;
};

} // namespace

RUN_TEST("hybrid_graph_multistream_scheduler_test", [] {
  simaai::neat::graph::Graph g;

  simaai::neat::graph::nodes::StreamSchedulerOptions opt;
  opt.per_stream_queue = 2;
  opt.drop_policy = simaai::neat::graph::nodes::StreamDropPolicy::DropOldest;
  opt.max_batch = 1;

  auto sched = g.add(simaai::neat::graph::nodes::StreamSchedulerNode(opt, "sched"));
  auto sink = g.add(make_pass_node("sink"));

  g.connect(sched, sink, "out", "in");
  simaai::neat::graph::GraphRun run = simaai::neat::graph::build(std::move(g));

  const int fast_inputs = 30;
  const int slow_inputs = 6;

  ExpectedScheduler expected(opt.per_stream_queue, opt.drop_policy);
  int fast_out = 0;
  int slow_out = 0;

  for (int i = 0; i < fast_inputs; ++i) {
    require(run.push(sched, simaai::neat::Sample{make_sample("fast", i)}), "push fast failed");
    ExpectedSample exp_fast = expected.push_and_emit("fast", i);
    auto out_fast = run.pull(sink, 5000);
    require(out_fast.has_value(), "GraphRun::pull timed out (fast)");
    require(out_fast->stream_id == exp_fast.stream_id, "scheduler output mismatch (fast)");
    if (out_fast->stream_id == "fast")
      fast_out++;
    if (out_fast->stream_id == "slow")
      slow_out++;

    if (i < slow_inputs) {
      require(run.push(sched, simaai::neat::Sample{make_sample("slow", i)}), "push slow failed");
      ExpectedSample exp_slow = expected.push_and_emit("slow", i);
      auto out_slow = run.pull(sink, 5000);
      require(out_slow.has_value(), "GraphRun::pull timed out (slow)");
      require(out_slow->stream_id == exp_slow.stream_id, "scheduler output mismatch (slow)");
      if (out_slow->stream_id == "fast")
        fast_out++;
      if (out_slow->stream_id == "slow")
        slow_out++;
    }
  }

  require(fast_out > 0, "fast stream missing outputs");
  require(slow_out > 0, "slow stream missing outputs");

  run.stop();
});
