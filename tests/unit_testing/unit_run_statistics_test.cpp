// A Run's statistics, on either backend (--backend=gstreamer|dataflow, a process
// each): the latency samples, RunStats and MeasureReport mean on Dataflow what they mean on
// GStreamer. Each lockstep push and pull is bracketed by the test's own clock reads, and every
// output's latency lies between "push returned -> pull entered" and "push entered -> pull
// returned", so the averages do too. Rows:
//   io           Input -> Output: lockstep inside a MeasureScope (counters, one end-to-end sample
//                per output, the gap between outputs, the push and pull averages, RunStats as the
//                exports read it), then free-running (one sample per output).
//   model        the same through Model::build(seed) + Model::Runner of a YOLOv8 detection model
//                (EV74 preproc, MLA, BoxDecode), and Model::benchmark's report.
//   keep_latest  that model's Run under OverflowPolicy::KeepLatest at depth 2, fed by try_push
//                faster than it runs (the link evicts): every pulled output still has its sample,
//                and no sample is longer than its own frame's push-to-pull interval.
// The model rows need a compiled YOLOv8 detection package (--model-package <X_mpk.tar.gz>,
// SIMA_TEST_BOXDECODE_MODEL_PACKAGE or SIMA_YOLO_TAR) and the EV74 on the Kernels firmware; without
// them they are skipped.

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "pipeline/runtime/RunInternal.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using Clock = std::chrono::steady_clock;

constexpr int kTimeoutMs = 30000;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double mean(const std::vector<double>& v) {
  double total = 0;
  for (const double x : v) {
    total += x;
  }
  return v.empty() ? 0.0 : total / static_cast<double>(v.size());
}

// A BGR image; a model's route takes it in EV74 memory (where its preproc stage reads it).
Tensor frame(int width, int height, unsigned index, TensorMemory memory = TensorMemory::CPU) {
  cv::Mat mat(height, width, CV_8UC3);
  for (int r = 0; r < height; ++r) {
    std::uint8_t* row = mat.ptr<std::uint8_t>(r);
    for (int c = 0; c < 3 * width; ++c) {
      row[c] = static_cast<std::uint8_t>((r * 7 + c * 3 + 31 * index) & 0xffU);
    }
  }
  return Tensor::from_cv_mat(mat, ImageSpec::PixelFormat::BGR, memory);
}

// A lockstep measurement of `n` push/pull pairs on `push` and `pull`, against the brackets.
template <class Push, class Pull>
void lockstep_measures(Run& run, int n, const Push& push, const Pull& pull, bool linear,
                       const std::string& what) {
  MeasureScope scope = run.start_measurement(MeasureOptions{});
  std::vector<double> lower;
  std::vector<double> upper;
  for (int i = 0; i < n; ++i) {
    const auto t0 = Clock::now();
    require(push(i), what + ": push");
    const auto t1 = Clock::now();
    const auto t2 = Clock::now();
    require(pull(), what + ": pull");
    const auto t3 = Clock::now();
    lower.push_back(ms_between(t1, t2));
    upper.push_back(ms_between(t0, t3));
  }
  const MeasureReport r = scope.stop();
  const auto count = static_cast<std::uint64_t>(n);
  require(r.counters.inputs_pushed == count && r.counters.outputs_pulled == count,
          what + ": counters " + std::to_string(r.counters.inputs_pushed) + " pushed, " +
              std::to_string(r.counters.outputs_pulled) + " pulled");
  require(r.end_to_end.count == static_cast<std::size_t>(n) && r.latency_samples_collected,
          what + ": one end-to-end sample per output (" + std::to_string(r.end_to_end.count) + ")");
  require(r.end_to_end.avg_ms >= mean(lower) && r.end_to_end.avg_ms <= mean(upper),
          what + ": the average latency " + std::to_string(r.end_to_end.avg_ms) +
              " ms lies between the brackets " + std::to_string(mean(lower)) + " and " +
              std::to_string(mean(upper)) + " ms");
  require(r.end_to_end.p50_ms > 0.0 && r.end_to_end.p99_ms >= r.end_to_end.p50_ms &&
              r.end_to_end.max_ms <= *std::max_element(upper.begin(), upper.end()),
          what + ": percentiles and maximum");
  if (linear) {
    require(r.frame_gap.count == static_cast<std::size_t>(n - 1),
            what + ": a gap between consecutive outputs (" + std::to_string(r.frame_gap.count) +
                ")");
  }
  // GStreamer's input stream counts and times the pushes it builds a buffer for (a tensor's, not
  // an image's, which it counts under decode), and its output thread's wait for each output;
  // Dataflow counts and times every accepted push and every pull.
  const bool pushes = sima_test::on_dataflow()
                          ? r.input.push_count == count && r.input.avg_push_us > 0.0
                          : r.input.push_count == count || r.input.push_count == 0U;
  require(pushes && r.input.pull_count == count && r.input.avg_pull_wait_us > 0.0,
          what + ": the push and pull averages (" + std::to_string(r.input.push_count) +
              " pushes, push " + std::to_string(r.input.avg_push_us) + " us, pull wait " +
              std::to_string(r.input.avg_pull_wait_us) + " us)");
  require(r.throughput_batches_per_s > 0.0 && r.elapsed_s > 0.0, what + ": throughput");
  const RunStats s = run_internal::stats(run);
  require(s.avg_latency_ms > 0.0 && s.min_latency_ms <= s.avg_latency_ms &&
              s.avg_latency_ms <= s.max_latency_ms,
          what + ": RunStats latency (" + std::to_string(s.min_latency_ms) + " <= " +
              std::to_string(s.avg_latency_ms) + " <= " + std::to_string(s.max_latency_ms) + ")");
}

// `n` blocking pushes from a thread of their own while this one pulls: one sample per output.
template <class Push, class Pull>
void free_running_measures(Run& run, int n, const Push& push, const Pull& pull,
                           const std::string& what) {
  MeasureScope scope = run.start_measurement(MeasureOptions{});
  std::thread pusher([&] {
    for (int i = 0; i < n; ++i) {
      if (!push(i)) {
        return;
      }
    }
  });
  int pulled = 0;
  while (pulled < n && pull()) {
    ++pulled;
  }
  pusher.join();
  const MeasureReport r = scope.stop();
  require(pulled == n && r.end_to_end.count == static_cast<std::size_t>(n),
          what + ": free-running, one sample per output (" + std::to_string(r.end_to_end.count) +
              " of " + std::to_string(pulled) + ")");
}

void io() {
  Graph g;
  g.add(nodes::Input());
  g.add(nodes::Output());
  Run run = g.build(TensorList{frame(64, 48, 0)});
  const auto push = [&](int i) { return run.push(TensorList{frame(64, 48, i)}); };
  const auto pull = [&] { return run.pull(std::chrono::milliseconds(kTimeoutMs)).ok(); };
  lockstep_measures(run, 200, push, pull, /*linear=*/true, "io");
  free_running_measures(run, 200, push, pull, "io");
  run.close();
}

bool ev74_runs_kernels_firmware() {
  for (int i = 0; i < 8; ++i) {
    const std::string dir = "/sys/class/remoteproc/remoteproc" + std::to_string(i) + "/";
    std::ifstream name(dir + "name");
    std::string value;
    if (name >> value && value == "ev74") {
      std::ifstream firmware(dir + "firmware");
      std::string fw;
      return firmware >> fw && fw == "sima-neat-native-cvu-fw";
    }
  }
  return false;
}

Model::Options detection_options() {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = 100;
  return o;
}

void model(const std::string& archive) {
  Model m(archive, detection_options());
  const TensorList seed{frame(1280, 720, 0, TensorMemory::EV74)};
  Model::Runner runner = m.build(seed);
  Run& run = internal::ModelAccess::run(runner);
  const auto push = [&](int i) {
    return runner.push(TensorList{frame(1280, 720, i % 4, TensorMemory::EV74)});
  };
  const auto pull = [&] { return !runner.pull(kTimeoutMs).empty(); };
  lockstep_measures(run, 100, push, pull, /*linear=*/true, "model");
  free_running_measures(run, 100, push, pull, "model");
  runner.close();

  BenchmarkOptions bo;
  bo.num_samples = 50;
  bo.min_duration_ms = 0;
  bo.inputs = seed;
  const BenchmarkReport report = m.benchmark(bo);
  require(report.latency_ms > 0.0 && report.latency_p50_ms > 0.0 && report.latency_p95_ms > 0.0 &&
              report.latency_p99_ms > 0.0 && report.fps > 0.0 && report.throughput_samples == 50 &&
              report.throughput_duration_ms > 0.0,
          "Model::benchmark: every latency and throughput field filled (latency " +
              std::to_string(report.latency_ms) + " ms, " + std::to_string(report.fps) + "/s)");
}

void keep_latest(const std::string& archive) {
  Model m(archive, detection_options());
  RunOptions ro;
  ro.queue_depth = 2;
  ro.overflow_policy = OverflowPolicy::KeepLatest;
  Model::Runner runner =
      m.build(TensorList{frame(1280, 720, 0, TensorMemory::EV74)}, Model::RouteOptions{}, ro);
  Run& run = internal::ModelAccess::run(runner);
  MeasureScope scope = run.start_measurement(MeasureOptions{});
  constexpr int kFrames = 120;
  std::vector<Clock::time_point> pushed(kFrames);
  std::thread pusher([&] {
    for (int i = 0; i < kFrames; ++i) {
      Stamp proposal;
      proposal.frame_id = i;
      pushed[static_cast<std::size_t>(i)] = Clock::now();
      (void)run.try_push(
          TensorList{frame(1280, 720, i % 4, TensorMemory::EV74).with_stamp(proposal)});
      std::this_thread::sleep_for(std::chrono::microseconds(700));
    }
  });
  std::vector<double> upper;
  int pulled = 0;
  for (;;) {
    Result<TensorList> out = run.pull(std::chrono::milliseconds(500));
    const auto at = Clock::now();
    if (!out.ok()) {
      break;
    }
    const std::int64_t id = out->front().stamp().frame_id;
    require(id >= 0 && id < kFrames, "keep_latest: the pulled frame's id");
    upper.push_back(ms_between(pushed[static_cast<std::size_t>(id)], at));
    ++pulled;
  }
  pusher.join();
  const MeasureReport r = scope.stop();
  require(pulled > 0 && r.end_to_end.count == static_cast<std::size_t>(pulled),
          "keep_latest: one sample per pulled output (" + std::to_string(r.end_to_end.count) +
              " of " + std::to_string(pulled) + ", " + std::to_string(r.counters.inputs_dropped) +
              " dropped)");
  require(r.end_to_end.max_ms <= *std::max_element(upper.begin(), upper.end()) &&
              r.end_to_end.avg_ms <= mean(upper),
          "keep_latest: no sample longer than its frame's push-to-pull interval (max " +
              std::to_string(r.end_to_end.max_ms) + " ms, average " +
              std::to_string(r.end_to_end.avg_ms) + " against " + std::to_string(mean(upper)) +
              " ms)");
  runner.close();
}

} // namespace

int main(int argc, char** argv) {
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_BOXDECODE_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_run_statistics_test", [&] {
    io();
    std::cout << "[row] io ok\n";
    if (archive.empty() || !std::ifstream(archive) || !ev74_runs_kernels_firmware()) {
      std::cout
          << "[skip] model rows: no YOLOv8 package (--model-package) or the EV74 does not run "
             "the Kernels firmware\n";
      return;
    }
    model(archive);
    std::cout << "[row] model ok\n";
    keep_latest(archive);
    std::cout << "[row] keep_latest ok\n";
  });
}
