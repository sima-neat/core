#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/gst/InputStreamInternal.h"
#include "test_main.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <thread>
#include <vector>

namespace {
using namespace simaai::neat;
using namespace std::chrono_literals;
const cv::Mat image(24, 32, CV_8UC3, cv::Scalar(10, 20, 30));

Run build(int output_timeout, bool discard = false, bool preflight = false, bool latest = false) {
  InputOptions input;
  input.format = FormatTag::RGB;
  input.width = input.max_width = image.cols;
  input.height = input.max_height = image.rows;
  input.depth = input.max_depth = 3;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  Graph graph;
  graph.add(nodes::Input(input));
  if (discard)
    graph.custom("valve drop=true");
  graph.add(nodes::Output(latest ? OutputOptions::Latest() : OutputOptions::EveryFrame(8)));
  RunOptions options;
  options.input_timeout_ms = 200;
  options.output_idle_timeout_ms = output_timeout;
  options.output_memory = OutputMemory::Owned;
  options.startup_preflight = preflight;
  return graph.build(std::vector<cv::Mat>{image}, options);
}

void exchange(Run& run) {
  require(run.push(std::vector<cv::Mat>{image}), "Input rejected");
  Sample sample;
  PullError error;
  require(run.pull(1000, sample, &error) == PullStatus::Ok, "Expected output: " + run.last_error());
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 1, "Expected one output tensor");
  auto pixels = tensors.front().map(MapMode::Read);
  require(pixels.data && pixels.size_bytes >= 3 &&
              static_cast<const unsigned char*>(pixels.data)[0] == 10,
          "Output changed");
}

void close(Run& run) {
  const auto start = std::chrono::steady_clock::now();
  run.close();
  require(std::chrono::steady_clock::now() - start < 2s, "Close was not bounded");
}

void require_timeout(Run& run) {
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (run.last_error().empty() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(10ms);
  require(run.last_error().find("runtime.output_timeout") != std::string::npos,
          "Expected output-silence failure: " + run.last_error());
}
} // namespace

RUN_TEST("unit_inputstream_idle_timeout_test", ([] {
           require(RunOptions{}.output_idle_timeout_ms == -1, "Default must remain legacy");
           // Exercise the fast-output race deterministically without depending on scheduling.
           InputStream::State state;
           state.opt.output_idle_timeout_ms = 200;
           state.running = true;
           const auto submitted = state.output_submission_time();
           state.observe_terminal_output();
           state.accept_output_submission(submitted);
           const auto first = state.first_accepted_ns.load();
           require(first == submitted && state.terminal_output_ns >= first,
                   "Output before push return was lost");
           state.accept_output_submission(InputStream::State::monotonic_ns());
           require(state.first_accepted_ns == first, "Further input extended the deadline");

           for (int timeout : {0, 200, -1}) {
             std::cerr << "case: output_idle_timeout_ms=" << timeout << '\n';
             auto run = build(timeout);
             std::this_thread::sleep_for(450ms);
             require(run.last_error().empty(), "Unsubmitted run was armed");
             Sample sample;
             PullError error;
             const auto start = std::chrono::steady_clock::now();
             require(run.pull(40, sample, &error) == PullStatus::Timeout,
                     "Explicit empty pull must time out");
             const auto waited = std::chrono::steady_clock::now() - start;
             require(waited >= 30ms && waited < 500ms && run.last_error().empty(),
                     "Pull timeout changed or permanently failed the Run");
             exchange(run);
             std::this_thread::sleep_for(450ms);
             if (timeout == 0) {
               exchange(run); // Explicitly unlimited producer idle.
             } else {
               require_timeout(run); // Legacy still fails after producer inactivity.
             }
             close(run);
           }
           {
             std::cerr << "case: preflight, rejection and output progress\n";
             auto run = build(200, false, true);
             std::this_thread::sleep_for(450ms);
             require(run.last_error().empty(), "Preflight armed the output timer");
             bool rejected = false;
             try {
               rejected = !run.push(std::vector<cv::Mat>{cv::Mat{}});
             } catch (const std::exception&) {
               rejected = true;
             }
             require(rejected, "Fixture input was not rejected");
             std::this_thread::sleep_for(450ms);
             require(run.last_error().empty(), "Rejected input armed the timer");
             for (int i = 0; i < 5; ++i) {
               exchange(run);
               std::this_thread::sleep_for(100ms);
             }
             require(run.last_error().empty(), "Output progress failed to refresh the timer");
             require_timeout(run);
             close(run);
           }
           {
             std::cerr << "case: explicit duration independent of input timeout\n";
             auto run = build(600);
             const auto start = std::chrono::steady_clock::now();
             exchange(run);
             std::this_thread::sleep_for(300ms);
             require(run.last_error().empty(), "Input timeout overrode the output deadline");
             require_timeout(run);
             require(std::chrono::steady_clock::now() - start >= 600ms,
                     "Output deadline fired too early");
             close(run);
           }
           {
             std::cerr << "case: unconsumed latest outputs still count as progress\n";
             auto run = build(200, false, false, true);
             for (int i = 0; i < 6; ++i) {
               require(run.push(std::vector<cv::Mat>{image}), "Latest input rejected");
               std::this_thread::sleep_for(100ms);
             }
             require(run.last_error().empty(), "Output dropping hid terminal progress");
             require_timeout(run);
             close(run);
           }
           for (int timeout : {0, 200, -1}) {
             std::cerr << "case: intentional discard, timeout=" << timeout << '\n';
             auto run = build(timeout, true);
             const auto until = std::chrono::steady_clock::now() + 650ms;
             while (run.last_error().empty() && std::chrono::steady_clock::now() < until) {
               try {
                 require(run.push(std::vector<cv::Mat>{image}), "Discard input rejected");
               } catch (const std::exception&) {
                 if (timeout <= 0)
                   throw;
                 break; // The background failure can race this submission.
               }
               std::this_thread::sleep_for(30ms);
             }
             if (timeout > 0)
               require_timeout(run); // Silence includes intentional discard, despite new input.
             else
               require(run.last_error().empty(), "Disabled or legacy behavior changed");
             close(run);
           }
         }));
