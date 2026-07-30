// Guards the cold model-archive load: decode plus extract, measured through the loader rather than
// through Model, because ModelPack caches by archive identity and would turn every sample after the
// first into a warm hit.
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/internal/ModelArchiveLoader.h"
#include "perf_metrics_common.h"

#include "asset_utils.h"
#include "model_archive_test_utils.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::size_t count_regular_files(const fs::path& root) {
  std::size_t files = 0;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file())
      ++files;
  }
  return files;
}

} // namespace

int main() {
  try {
    using simaai::neat::internal::ModelArchiveExtractResult;
    using simaai::neat::internal::ModelArchiveLoader;

    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 10);

    const std::string archive = sima_test::resolve_resnet50_tar();
    if (archive.empty()) {
      throw std::runtime_error("model download failed: ResNet50 model pack unavailable; set "
                               "SIMA_MODEL_TAR or SIMA_RESNET50_TAR");
    }

    const fs::path scratch = fs::path(sima_test::make_temp_dir("perf_runtime_model_archive_load"));

    // ModelPack's runtime configuration, so the scenario measures the load path production uses.
    // Strict defaults reject the auxiliary build artifacts that shipped model packs carry, and
    // leaving the snapshot on TMPDIR would measure a filesystem pair no real load uses.
    simaai::neat::internal::ModelArchiveLoaderOptions opt;
    opt.reject_unsupported_file_types = false;
    opt.require_pipeline_sequence = false;
    opt.staging_base = scratch.string();

    // Warm the page cache so the first sample measures the decoder rather than the first read of
    // the archive off storage.
    {
      const fs::path warmup_root = scratch / "warmup";
      (void)ModelArchiveLoader::extract(archive, warmup_root.string(), opt);
      fs::remove_all(warmup_root);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));
    std::size_t expected_files = 0;

    for (int i = 0; i < iterations; ++i) {
      // A fresh empty root per sample: extract() clears an existing package root, so reusing one
      // would measure a delete alongside the decode.
      const fs::path sample_root = scratch / ("sample_" + std::to_string(i));
      const auto t0 = sima_perf::Clock::now();
      const ModelArchiveExtractResult extracted =
          ModelArchiveLoader::extract(archive, sample_root.string(), opt);
      const auto t1 = sima_perf::Clock::now();
      latencies_ms.push_back(sima_perf::elapsed_ms(t0, t1));

      // Catches an empty extraction and run-to-run nondeterminism. It cannot catch a decoder that
      // truncates uniformly, since every sample decodes the same archive the same way; the
      // multi-member fixture in unit_model_archive_loader_test is what covers truncation.
      const std::size_t files = count_regular_files(extracted.package_root);
      if (files == 0) {
        throw std::runtime_error("model archive load produced an empty package at iteration " +
                                 std::to_string(i));
      }
      if (i == 0) {
        expected_files = files;
      } else if (files != expected_files) {
        throw std::runtime_error("model archive load extracted " + std::to_string(files) +
                                 " files at iteration " + std::to_string(i) + ", expected " +
                                 std::to_string(expected_files));
      }
      fs::remove_all(sample_root);
    }

    // From the measured loads only. Bracketing the loop instead would fold the file-count walk and
    // the per-sample delete into a metric named for the load, and gate them tighter than p50 does.
    double total_ms = 0.0;
    for (const double ms : latencies_ms) {
      total_ms += ms;
    }
    sima_perf::PerfMetrics metrics;
    metrics.throughput =
        (total_ms > 0.0) ? (1000.0 * static_cast<double>(iterations) / total_ms) : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = latencies_ms.empty() ? 0.0 : latencies_ms.front();
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();

    sima_perf::emit_metrics_json("runtime_model_archive_load", iterations, metrics,
                                 "model_archive_cold_load");
    std::error_code cleanup_ec;
    fs::remove_all(scratch, cleanup_ec);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "perf_runtime_model_archive_load_test exception: " << e.what() << "\n";
    return 1;
  }
}
