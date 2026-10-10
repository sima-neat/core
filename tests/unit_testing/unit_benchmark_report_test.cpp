#include "model/Model.h"
#include "test_main.h"

#include <glib.h>

#include <fstream>
#include <limits>
#include <locale>
#include <sstream>

namespace {

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    GError* error = nullptr;
    gchar* value = g_dir_make_tmp("neat-benchmark-report-XXXXXX", &error);
    if (!value) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      throw std::runtime_error(detail);
    }
    path = value;
    g_free(value);
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  require(file.good(), "saved report must be readable");
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

template <typename Callable> void require_throws(Callable&& operation) {
  bool threw = false;
  try {
    operation();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "invalid destination/report must throw");
}

struct CommaDecimal : std::numpunct<char> {
  char do_decimal_point() const override {
    return ',';
  }
};

struct LocaleGuard {
  std::locale original = std::locale();
  LocaleGuard() {
    std::locale::global(std::locale(original, new CommaDecimal));
  }
  ~LocaleGuard() {
    std::locale::global(original);
  }
};

void saves_exact_path_and_replaces_existing_report() {
  TemporaryDirectory dir;
  simaai::neat::BenchmarkReport report;
  report.latency_ms = 1.25;
  report.fps = 800;
  report.latency_p95_ms = 2.5;
  report.telemetry.cpu_cores_usage_pct = {{"cpu0", "Core 0", "%", 2, 42.25, 40, 44.5}};
  require(simaai::neat::LayerStats{}.stage_index == 0,
          "single-stage records default to stage zero");
  const auto target = dir.path / "model with spaces" / "yolo_benchmark.yaml";
  {
    LocaleGuard locale;
    require(report.save(target) == target, "save must return the caller's complete file path");
  }
  const auto contents = read_file(target);
  require(contents.find("latency_ms: 1.25\n") != std::string::npos,
          "decimal scalars must be locale independent");
  require(contents.find("layer_stats: []\n") != std::string::npos,
          "a report without profiling must save an empty sequence");
  require(!std::filesystem::exists(target.parent_path() / "benchmark.yaml"),
          "save must not append an invented filename");
  require(contents.find("latency_p95_ms: 2.5\n") != std::string::npos,
          "save must preserve latency percentiles");
  require(contents.find("\"mean\":42.25") != std::string::npos,
          "save must preserve per-core telemetry measurements");
  report.layer_stats = {{"MLA_0/noop", 0.0, 2}};
  report.save(target);
  require(read_file(target).find("stage_index: 2\n") != std::string::npos,
          "stage indices must preserve execution order gaps for A65 stages");
  require(read_file(target).find("total_execution_time_us: 0\n") != std::string::npos,
          "replacement must retain zero-time no-op records");
}

void failures_do_not_destroy_a_previous_report() {
  TemporaryDirectory dir;
  const auto target = dir.path / "report.yaml";
  simaai::neat::BenchmarkReport report;
  report.save(target);
  const auto original = read_file(target);
  for (double invalid :
       {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    report.layer_stats = {{"bad", invalid, 0}};
    require_throws([&] { report.save(target); });
    require(read_file(target) == original, "invalid layer timings must preserve the existing file");
  }
  report.layer_stats.clear();
  report.fps = std::numeric_limits<double>::infinity();
  require_throws([&] { report.save(target); });
  require(read_file(target) == original, "invalid headline values must preserve the existing file");
  report.fps = 0;
  require_throws([&] { report.save({}); });
  require_throws([&] { report.save(dir.path); });
  require_throws([&] { report.save(target / "child.yaml"); });
}

} // namespace

RUN_TEST("unit_benchmark_report_test", ([] {
           saves_exact_path_and_replaces_existing_report();
           failures_do_not_destroy_a_previous_report();
         }));
