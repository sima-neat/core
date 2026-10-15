#include "model/Model.h"
#include "test_main.h"

#include <glib.h>
#include <yaml-cpp/yaml.h>

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
  const auto parsed = YAML::LoadFile(target.string());
  require(parsed["latency_ms"].as<double>() == report.latency_ms,
          "decimal scalars must be locale independent");
  require(parsed["layer_stats"].IsSequence() && parsed["layer_stats"].size() == 0,
          "a report without profiling must save an empty sequence");
  require(!std::filesystem::exists(target.parent_path() / "benchmark.yaml"),
          "save must not append an invented filename");
  require(parsed["latency_p95_ms"].as<double>() == report.latency_p95_ms,
          "save must preserve latency percentiles");
  require(parsed["telemetry"]["cpu_cores_usage_pct"][0]["mean"].as<double>() == 42.25,
          "save must preserve per-core telemetry measurements");
  report.layer_stats = {{"MLA_0/noop", 0.0, 2}};
  report.save(target);
  const auto layer = YAML::LoadFile(target.string())["layer_stats"][0];
  require(layer["stage_index"].as<std::size_t>() == 2,
          "stage indices must preserve execution order gaps for A65 stages");
  require(layer["name"].as<std::string>() == "MLA_0/noop" &&
              layer["total_execution_time_us"].as<double>() == 0,
          "replacement must retain zero-time no-op records");
}

void telemetry_scientific_notation_remains_numeric() {
  TemporaryDirectory dir;
  simaai::neat::BenchmarkReport report;
  report.telemetry.cpu_cores_usage_pct = {
      {"cpu0", "Core 0: \"quoted\"\nline", "%", 2, 1e-9, -1e-20, 1e20}};
  const auto path = dir.path / "scientific.yaml";
  report.save(path);
  const auto contents = read_file(path);
  // Cover YAML 1.1 scientific notation, including negative telemetry values.
  require(contents.find("mean: 1.0e-09") != std::string::npos,
          "small telemetry means must remain YAML numbers");
  require(contents.find("minimum: -1.0e-20") != std::string::npos,
          "negative telemetry values must remain YAML numbers");
  require(contents.find("maximum: 1.0e+20") != std::string::npos,
          "large telemetry values must remain YAML numbers");
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
           telemetry_scientific_notation_remains_numeric();
           failures_do_not_destroy_a_previous_report();
         }));
