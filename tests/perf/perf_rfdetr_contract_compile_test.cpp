#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "perf_metrics_common.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

// Measures RF contract preparation, without model extraction or device startup.
int main() {
  try {
    using namespace simaai::neat;
    using namespace simaai::neat::pipeline_internal::sima;
    MpkContract mpk;
    MpkPluginIoContract mla;
    mla.name = "MLA_0";
    mla.sequence = 1;
    mla.processor = "MLA";
    mla.kernel = "mla";
    mla.canonical_output_dtype = "BF16";
    const std::array<std::vector<std::int64_t>, 3> shapes{
        {{1, 1, 200, 4}, {1, 1, 200, 91}, {1, 108, 108, 200}}};
    for (int i = 0; i < 3; ++i) {
      const auto& shape = shapes[i];
      mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = i,
          .physical_index = i,
          .name = "output_" + std::to_string(i),
          .dtype = "BF16",
          .mpk_shape = shape,
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = static_cast<std::size_t>(shape[1] * shape[2] * shape[3] * 2),
          .logical_shape = shape});
    }
    mpk.plugins.push_back(std::move(mla));
    ModelManagedRouteFlags flags;
    flags.boxdecode_selected = true;
    flags.requested_decode_type = BoxDecodeType::RfDetrSeg;
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 1000);
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = -100; i < iterations; ++i) {
      std::string error;
      const auto start = sima_perf::Clock::now();
      const auto contract = build_boxdecode_static_contract_from_mpk(mpk, flags, &error);
      if (!contract)
        throw std::runtime_error(error);
      const auto compiled = stagesemantics::build_boxdecode_compiled_contract(*contract);
      const auto end = sima_perf::Clock::now();
      if (compiled.runtime_contract.input_bindings.size() != 3 ||
          compiled.runtime_contract.input_bindings[2].src_logical_output_index != 2 ||
          compiled.runtime_contract.logical_inputs.size() != 3)
        throw std::runtime_error("RF contract lost native head bindings");
      if (i >= 0)
        samples.push_back(sima_perf::elapsed_ms(start, end));
    }
    double total_ms = 0;
    for (const auto sample : samples)
      total_ms += sample;
    sima_perf::PerfMetrics metrics;
    metrics.throughput = total_ms > 0 ? 1000.0 * iterations / total_ms : 0;
    metrics.p50 = sima_perf::percentile(samples, 50);
    metrics.p95 = sima_perf::percentile(samples, 95);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    sima_perf::emit_metrics_json("rfdetr_contract_compile", iterations, metrics,
                                 "contract_prepare_compile");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
