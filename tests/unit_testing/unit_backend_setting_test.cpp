// The process's backend (neat/runtime.h: set_backend, backend) on the public API. A process runs
// on one backend, fixed by its first Model constructed or Graph built, so each case is a process
// of its own (--case=<name>; tests/CMakeLists.txt registers one test per case):
//   dataflow   before anything is built the setting reads back what was set, any number of
//              times and from several threads; a value that is no backend is refused; a build
//              takes the setting (a graph Dataflow does not carry fails with the not-carried
//              error, which names set_backend), and fixes it even though it failed: the other
//              value is then refused with the rule, the same value accepted; the calls that only
//              the GStreamer backend carries (Graph::validate, Graph::run_rtsp, the source
//              Graph::run()) fail on Dataflow at the call; a Run that Dataflow carries runs.
//   gstreamer  a build on the default (GStreamer) fixes it: Dataflow is then refused, and a graph
//              Dataflow does not carry (a Queue node) runs on GStreamer.
//   race       threads setting both values while another builds that graph: after the build
//              the other value is refused, and the build ran on the backend the process is fixed
//              to (on Dataflow it fails with the not-carried error, on GStreamer it builds).
//   without_dataflow  (a library built without the Dataflow runtime) Dataflow is refused, and the
//              process runs on GStreamer.

#include "neat/runtime.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr const char* kHint =
    "Hint: To run this program on the GStreamer backend, call "
    "simaai::neat::set_backend(simaai::neat::RunBackend::GStreamer) (Python: "
    "pyneat.set_backend(pyneat.RunBackend.GStreamer)) before the program constructs its first "
    "Model or builds its first Graph.";

template <class Middle> Graph input_output_with(Middle middle) {
  InputOptions in;
  in.shape = {4};
  in.dtype = TensorDType::UInt8;
  Graph g;
  g.add(nodes::Input(in));
  middle(g);
  g.add(nodes::Output());
  return g;
}

Graph input_output() {
  return input_output_with([](Graph&) {});
}

// A graph neither backend runs without a source behind it, which Dataflow refuses at the build
// (it has no public Input).
Graph no_input() {
  Graph g;
  g.add(nodes::Output());
  return g;
}

// Input -> Queue -> Output: GStreamer runs it (built with a seed, as GStreamer's linear Run of an
// Input with a declared shape needs); Dataflow refuses the Queue node at the build.
Graph with_queue() {
  Graph g = input_output_with([](Graph& graph) { graph.add(nodes::Queue()); });
  return g;
}

Tensor bytes4(std::uint8_t tag) {
  return Tensor::from_vector(std::vector<std::uint8_t>{tag, 1, 2, 3}, {4}, TensorMemory::CPU);
}

const char* name(RunBackend b) {
  return b == RunBackend::Dataflow ? "Dataflow" : "GStreamer";
}

void dataflow_case() {
  require(backend() == RunBackend::GStreamer, "the default is GStreamer");
  for (int i = 0; i < 3; ++i) {
    set_backend(RunBackend::Dataflow);
    require(backend() == RunBackend::Dataflow, "set Dataflow, read Dataflow");
    set_backend(RunBackend::GStreamer);
    require(backend() == RunBackend::GStreamer, "set GStreamer, read GStreamer");
  }
  {
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
      threads.emplace_back([t] {
        for (int i = 0; i < 1000; ++i) {
          set_backend((i + t) % 2 == 0 ? RunBackend::Dataflow : RunBackend::GStreamer);
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }
  require_neat_error([] { set_backend(static_cast<RunBackend>(42)); },
                     error_codes::kOptionOutOfRange, "simaai::neat::set_backend: not a backend");
  set_backend(RunBackend::Dataflow);
  require(backend() == RunBackend::Dataflow, "Dataflow before the first build");

  // The first build takes the setting and fixes it, even though it fails.
  require_neat_error([] { (void)no_input().build(); }, error_codes::kPipelineShape,
                     "Graph::build: this graph is not carried by the Dataflow backend: it has no "
                     "public Input (a source graph).\n" +
                         std::string(kHint));
  require_neat_error([] { set_backend(RunBackend::GStreamer); }, error_codes::kOptionOutOfRange,
                     "simaai::neat::set_backend(RunBackend::GStreamer): a process runs on one "
                     "backend, and this one runs on the Dataflow backend since its first Model was "
                     "constructed or its first Graph built");
  set_backend(RunBackend::Dataflow);
  require(backend() == RunBackend::Dataflow, "the same value is accepted, and nothing changes");

  // The GStreamer backend's own calls fail at the call.
  require_neat_error([] { (void)input_output().validate(ValidateOptions{}); },
                     error_codes::kPipelineShape,
                     "Graph::validate: the validation of this graph is not carried by the Dataflow "
                     "backend: it prerolls the graph as a GStreamer pipeline.\n" +
                         std::string(kHint));
  require_neat_error([] { (void)input_output().run_rtsp(RtspServerOptions{}); },
                     error_codes::kPipelineShape,
                     "Graph::run_rtsp: this graph's RTSP server is not carried by the Dataflow "
                     "backend: it streams the graph's H.264 output over RTSP.\n" +
                         std::string(kHint));
  require_neat_error([] { no_input().run(); }, error_codes::kPipelineShape,
                     "Graph::run: this source graph is not carried by the Dataflow backend: it "
                     "has no public Input: its frames come from its source node, and its tensor "
                     "callback takes each output.\n" +
                         std::string(kHint));

  // A Run Dataflow carries runs.
  Graph g = input_output();
  Run run = g.build();
  require(run.push(TensorList{bytes4(7)}), "push");
  Result<TensorList> out = run.pull(std::chrono::milliseconds(5000));
  require(static_cast<bool>(out) && out->size() == 1U &&
              out->front().copy_dense_bytes_tight() == bytes4(7).copy_dense_bytes_tight(),
          "the pushed bytes come back");
  run.close();
}

void gstreamer_case() {
  require(backend() == RunBackend::GStreamer, "the default is GStreamer");
  Graph g = with_queue();
  Run run = g.build(TensorList{bytes4(1)});
  require_neat_error([] { set_backend(RunBackend::Dataflow); }, error_codes::kOptionOutOfRange,
                     "this one runs on the GStreamer backend since its first Model was "
                     "constructed or its first Graph built");
  set_backend(RunBackend::GStreamer);
  require(run.push(TensorList{bytes4(9)}), "push");
  Result<TensorList> out = run.pull(std::chrono::milliseconds(5000));
  require(static_cast<bool>(out) && out->size() == 1U, "a GStreamer Run runs");
  run.close();
}

void without_dataflow_case() {
  require_neat_error([] { set_backend(RunBackend::Dataflow); }, error_codes::kOptionOutOfRange,
                     "simaai::neat::set_backend(RunBackend::Dataflow): this libsima_neat was built "
                     "without the Dataflow backend");
  require(backend() == RunBackend::GStreamer, "the process runs on GStreamer");
  gstreamer_case();
}

void race_case() {
  std::atomic<bool> go{false};
  std::atomic<bool> built{false};
  std::atomic<int> refused{0};
  std::atomic<int> refused_before_build{0};
  std::vector<std::thread> setters;
  for (int t = 0; t < 4; ++t) {
    setters.emplace_back([&, t] {
      while (!go.load()) {
        std::this_thread::yield();
      }
      for (int i = 0; i < 20000; ++i) {
        const bool was_built = built.load();
        try {
          set_backend((i + t) % 2 == 0 ? RunBackend::Dataflow : RunBackend::GStreamer);
        } catch (const NeatError& e) {
          require(e.report().error_code == error_codes::kOptionOutOfRange, "the rule's code");
          ++refused;
          if (!was_built) {
            // Refused while this thread had not yet seen the build: the build fixed it first.
            ++refused_before_build;
          }
        }
      }
    });
  }
  go.store(true);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  bool on_dataflow = false;
  {
    Graph g = with_queue();
    try {
      Run run = g.build(TensorList{bytes4(1)});
      run.close();
    } catch (const NeatError& e) {
      on_dataflow = std::string(e.what()).find("node Queue is not carried by the Dataflow "
                                               "backend") != std::string::npos;
      require(on_dataflow, std::string("the build fails only on Dataflow: ") + e.what());
    }
  }
  built.store(true);
  for (auto& t : setters) {
    t.join();
  }
  const RunBackend fixed = backend();
  require(on_dataflow == (fixed == RunBackend::Dataflow),
          std::string("the build ran on the backend the process is fixed to: ") + name(fixed));
  require(refused.load() > 0, "after the build the other value is refused");
  const RunBackend other =
      fixed == RunBackend::Dataflow ? RunBackend::GStreamer : RunBackend::Dataflow;
  require_neat_error([&] { set_backend(other); }, error_codes::kOptionOutOfRange,
                     "a process runs on one backend");
  set_backend(fixed);
  std::cout << "[INFO] race: fixed on " << name(fixed) << ", " << refused.load()
            << " sets refused (" << refused_before_build.load()
            << " by a thread that had not seen the build return)\n";
}

} // namespace

int main(int argc, char** argv) {
  std::string which = "dataflow";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--case=", 0) == 0) {
      which = arg.substr(7);
    }
  }
  return sima_test::run_test(("unit_backend_setting_test " + which).c_str(), [&] {
    if (which == "dataflow") {
      dataflow_case();
    } else if (which == "gstreamer") {
      gstreamer_case();
    } else if (which == "race") {
      race_case();
    } else if (which == "without_dataflow") {
      without_dataflow_case();
    } else {
      throw std::runtime_error("unknown --case=" + which);
    }
  });
}
