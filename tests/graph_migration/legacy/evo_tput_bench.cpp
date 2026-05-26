#include <neat.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace neat = simaai::neat;

namespace {
std::string arg_value(int argc, char** argv, const std::string& key) {
  const std::string prefix = key + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
    if (arg == key && i + 1 < argc) return argv[++i] ? argv[i] : "";
  }
  return {};
}
int int_arg(int argc, char** argv, const std::string& key, int fallback) {
  const std::string raw = arg_value(argc, argv, key);
  if (raw.empty()) return fallback;
  try { return std::stoi(raw); } catch (...) { return fallback; }
}
bool bool_arg(int argc, char** argv, const std::string& key, bool fallback) {
  const std::string raw = arg_value(argc, argv, key);
  if (raw.empty()) return fallback;
  if (raw == "1" || raw == "true" || raw == "on" || raw == "yes") return true;
  if (raw == "0" || raw == "false" || raw == "off" || raw == "no") return false;
  return fallback;
}
std::size_t element_count(const std::vector<int64_t>& shape) {
  std::size_t count = 1;
  if (shape.empty()) return 1;
  for (auto dim : shape) count *= static_cast<std::size_t>(dim > 0 ? dim : 1);
  return count;
}
neat::Tensor make_tensor(const neat::TensorSpec& spec, float fill) {
  std::vector<int64_t> shape = spec.shape.empty() ? std::vector<int64_t>{1} : spec.shape;
  std::vector<float> data(element_count(shape), fill);
  return neat::Tensor::from_vector(data, shape, neat::TensorMemory::EV74);
}
double percentile_ms(std::vector<double> samples, double p) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const double idx = (p / 100.0) * static_cast<double>(samples.size() - 1U);
  const auto lo = static_cast<std::size_t>(idx);
  const auto hi = std::min<std::size_t>(lo + 1U, samples.size() - 1U);
  const double frac = idx - static_cast<double>(lo);
  return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}
std::string base_name(const std::string& path) {
  const auto pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}
}

int main(int argc, char** argv) {
  const std::string model_path = arg_value(argc, argv, "--model");
  const std::string pre = arg_value(argc, argv, "--pre");
  const std::string post = arg_value(argc, argv, "--post");
  const int warmup = int_arg(argc, argv, "--warmup", 5);
  const int measured = int_arg(argc, argv, "--measured", 20);
  const int timeout_ms = int_arg(argc, argv, "--timeout-ms", 60000);
  const bool cleanup = bool_arg(argc, argv, "--cleanup", true);
  const bool plugin_latency = bool_arg(argc, argv, "--plugin-latency", true);
  const bool startup_preflight = bool_arg(argc, argv, "--startup-preflight", true);
  const std::string mode = arg_value(argc, argv, "--mode").empty() ? "sync" : arg_value(argc, argv, "--mode");
  const int inflight = std::max(1, int_arg(argc, argv, "--inflight", 4));
  if (model_path.empty() || pre.empty() || post.empty() || measured <= 0 || warmup < 0) {
    std::cerr << "Usage: evo_tput_bench --model <path> --pre <A65|EV74> --post <A65|EV74> "
                 "[--warmup N] [--measured N] [--timeout-ms MS] [--cleanup 0|1] "
                 "[--mode sync|async] [--inflight N]\n";
    return 2;
  }
  if (mode != "sync" && mode != "async") {
    std::cerr << "EVO_TPUT_FAIL stage=args reason=bad_mode mode=" << mode << "\n";
    return 2;
  }
  try {
    neat::Model::Options opt;
    opt.preprocess.kind = neat::InputKind::Tensor;
    opt.preprocess.enable = neat::AutoFlag::On;
    opt.processcvu.pre_run_target = pre;
    opt.processcvu.post_run_target = post;
    opt.cleanup_extracted_model_data = cleanup;

    std::cout << "EVO_TPUT_CONFIG model=" << model_path << " model_name=" << base_name(model_path)
              << " pre=" << pre << " post=" << post << " warmup=" << warmup
              << " measured=" << measured << " timeout_ms=" << timeout_ms
              << " plugin_latency=" << (plugin_latency ? 1 : 0)
              << " startup_preflight=" << (startup_preflight ? 1 : 0)
              << " mode=" << mode << " inflight=" << inflight << "\n" << std::flush;

    const auto ctor0 = std::chrono::steady_clock::now();
    neat::Model model(model_path, opt);
    const auto ctor1 = std::chrono::steady_clock::now();
    std::cout << "EVO_TPUT_CTOR_DONE ctor_seconds="
              << std::chrono::duration<double>(ctor1 - ctor0).count() << "\n" << std::flush;

    neat::TensorList inputs;
    const auto specs = model.input_specs();
    inputs.reserve(specs.size());
    std::size_t elems = 0;
    for (std::size_t i = 0; i < specs.size(); ++i) {
      auto t = make_tensor(specs[i], 0.01f * static_cast<float>(i + 1));
      elems += element_count(specs[i].shape.empty() ? std::vector<int64_t>{1} : specs[i].shape);
      inputs.push_back(std::move(t));
    }
    std::cout << "EVO_TPUT_INPUTS count=" << inputs.size() << " float_elements=" << elems << "\n"
              << std::flush;

    const auto build0 = std::chrono::steady_clock::now();
    neat::RunOptions run_opt;
    run_opt.startup_preflight = startup_preflight;
    auto runner = model.build(inputs, neat::Model::RouteOptions{}, run_opt);
    const auto build1 = std::chrono::steady_clock::now();
    if (!runner) {
      std::cerr << "EVO_TPUT_FAIL stage=build reason=runner_not_ready\n";
      return 3;
    }
    std::cout << "EVO_TPUT_BUILD_READY build_seconds="
              << std::chrono::duration<double>(build1 - build0).count()
              << " ctor_seconds=" << std::chrono::duration<double>(ctor1 - ctor0).count() << "\n"
              << std::flush;

    std::size_t warm_outputs = 0;
    const auto warm0 = std::chrono::steady_clock::now();
    for (int i = 0; i < warmup; ++i) {
      auto outputs = runner.run(inputs, timeout_ms);
      warm_outputs += outputs.size();
    }
    const auto warm1 = std::chrono::steady_clock::now();
    std::cout << "EVO_TPUT_WARMUP_DONE frames=" << warmup << " outputs=" << warm_outputs
              << " seconds=" << std::chrono::duration<double>(warm1 - warm0).count() << "\n"
              << std::flush;

    neat::MeasureOptions measure_opt;
    measure_opt.duration_ms = std::max(1, measured * timeout_ms);
    measure_opt.warmup_ms = 0;
    measure_opt.timeout_ms = timeout_ms;
    measure_opt.include_plugin_latency = plugin_latency;
    measure_opt.include_power = false;
    measure_opt.title = "EVO throughput";
    measure_opt.model = base_name(model_path);
    measure_opt.placement = std::string("pre=") + pre + ",post=" + post;
    auto scope = runner.start_measurement(measure_opt);

    std::vector<double> lat_ms;
    lat_ms.reserve(static_cast<std::size_t>(measured));
    std::size_t output_count = 0;
    auto meas0 = std::chrono::steady_clock::now();
    if (mode == "sync") {
      for (int i = 0; i < measured; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto outputs = runner.run(inputs, timeout_ms);
        const auto t1 = std::chrono::steady_clock::now();
        if (outputs.empty()) {
          std::cerr << "EVO_TPUT_FAIL stage=measured iter=" << i << " reason=no_outputs\n";
          return 4;
        }
        output_count += outputs.size();
        lat_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
    } else {
      std::deque<std::chrono::steady_clock::time_point> push_times;
      int pushed = 0;
      int pulled = 0;
      const int seed = std::min(inflight, measured);
      for (; pushed < seed; ++pushed) {
        const auto t_push = std::chrono::steady_clock::now();
        if (!runner.push(inputs)) {
          std::cerr << "EVO_TPUT_FAIL stage=async_seed iter=" << pushed << " reason=push_failed\n";
          return 5;
        }
        push_times.push_back(t_push);
      }
      meas0 = std::chrono::steady_clock::now();
      while (pulled < measured) {
        auto out = runner.pull(timeout_ms);
        const auto t_pull = std::chrono::steady_clock::now();
        if (out.empty()) {
          std::cerr << "EVO_TPUT_FAIL stage=async_measured iter=" << pulled
                    << " reason=no_output\n";
          return 6;
        }
        if (push_times.empty()) {
          std::cerr << "EVO_TPUT_FAIL stage=async_measured iter=" << pulled
                    << " reason=missing_push_timestamp\n";
          return 7;
        }
        lat_ms.push_back(std::chrono::duration<double, std::milli>(t_pull - push_times.front()).count());
        push_times.pop_front();
        ++pulled;
        ++output_count;
        if (pushed < measured) {
          const auto t_push = std::chrono::steady_clock::now();
          if (!runner.push(inputs)) {
            std::cerr << "EVO_TPUT_FAIL stage=async_measured iter=" << pulled
                      << " reason=push_failed\n";
            return 8;
          }
          push_times.push_back(t_push);
          ++pushed;
        }
      }
    }
    const auto meas1 = std::chrono::steady_clock::now();
    const auto report = scope.stop();

    const double meas_s = std::chrono::duration<double>(meas1 - meas0).count();
    const double mean_ms = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) /
                           static_cast<double>(lat_ms.size());
    const auto [min_it, max_it] = std::minmax_element(lat_ms.begin(), lat_ms.end());
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "EVO_TPUT_RESULT status=PASS model_name=" << base_name(model_path)
              << " pre=" << pre << " post=" << post << " mode=" << mode
              << " inflight=" << inflight << " measured=" << measured
              << " outputs=" << output_count << " measured_seconds=" << meas_s
              << " fps=" << (static_cast<double>(measured) / meas_s)
              << " mean_ms=" << mean_ms << " min_ms=" << *min_it
              << " p50_ms=" << percentile_ms(lat_ms, 50.0)
              << " p90_ms=" << percentile_ms(lat_ms, 90.0)
              << " p95_ms=" << percentile_ms(lat_ms, 95.0)
              << " p99_ms=" << percentile_ms(lat_ms, 99.0)
              << " max_ms=" << *max_it << "\n";
    std::cout << report.to_text() << std::flush;

    const auto summary = runner.measurement_summary();
    std::cout << "EVO_RUN_STATS"
              << " avg_latency_ms=" << summary.stats.avg_latency_ms
              << " min_latency_ms=" << summary.stats.min_latency_ms
              << " max_latency_ms=" << summary.stats.max_latency_ms
              << " avg_push_us=" << summary.input_stats.avg_push_us
              << " avg_pull_wait_us=" << summary.input_stats.avg_pull_wait_us
              << " avg_alloc_us=" << summary.input_stats.avg_alloc_us
              << " avg_map_us=" << summary.input_stats.avg_map_us
              << " avg_copy_us=" << summary.input_stats.avg_copy_us
              << " push_count=" << summary.input_stats.push_count
              << " pull_count=" << summary.input_stats.pull_count
              << "\n";

    const auto diag = runner.diag_snapshot();
    std::vector<neat::RunStageStats> stages = diag.stages;
    std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(a.total_us, a.max_us, a.stage_name) >
             std::make_tuple(b.total_us, b.max_us, b.stage_name);
    });
    const std::size_t stage_n = std::min<std::size_t>(stages.size(), 8U);
    for (std::size_t i = 0; i < stage_n; ++i) {
      const auto& s = stages[i];
      const double avg_ms = s.samples ? (static_cast<double>(s.total_us) / s.samples / 1000.0) : 0.0;
      std::cout << "EVO_STAGE_TOP rank=" << (i + 1)
                << " name=" << s.stage_name
                << " samples=" << s.samples
                << " total_ms=" << (static_cast<double>(s.total_us) / 1000.0)
                << " avg_ms=" << avg_ms
                << " max_ms=" << (static_cast<double>(s.max_us) / 1000.0)
                << "\n";
    }
    std::vector<neat::RunElementTimingStats> elements = diag.element_timings;
    std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(a.total_us, a.max_us, a.element_name) >
             std::make_tuple(b.total_us, b.max_us, b.element_name);
    });
    const std::size_t elem_n = std::min<std::size_t>(elements.size(), 12U);
    for (std::size_t i = 0; i < elem_n; ++i) {
      const auto& e = elements[i];
      const double avg_ms = e.samples ? (static_cast<double>(e.total_us) / e.samples / 1000.0) : 0.0;
      std::cout << "EVO_ELEMENT_TOP rank=" << (i + 1)
                << " name=" << e.element_name
                << " samples=" << e.samples
                << " total_ms=" << (static_cast<double>(e.total_us) / 1000.0)
                << " avg_ms=" << avg_ms
                << " min_ms=" << (static_cast<double>(e.min_us) / 1000.0)
                << " max_ms=" << (static_cast<double>(e.max_us) / 1000.0)
                << " missed_in=" << e.missed_in
                << " missed_out=" << e.missed_out
                << "\n";
    }
    runner.close();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "EVO_TPUT_FAIL exception=" << ex.what() << "\n";
    return 1;
  }
}
