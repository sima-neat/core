// Manual timing with saved inputs. Both backends use the public single-flight
// push/pull path. ELF profiling is never invoked by this benchmark.
#include "../support/grid_sample_reference.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>

namespace gt = grid_sample_test;
namespace neat = simaai::neat;
namespace {
using Clock = std::chrono::steady_clock;
std::vector<float> read(const char* name) {
  std::ifstream file(name, std::ios::binary | std::ios::ate);
  gt::require(bool(file), std::string("cannot open ") + name);
  const auto bytes = file.tellg();
  gt::require(bytes > 0 && bytes % 4 == 0, "expected a nonempty raw FP32 file");
  std::vector<float> values(static_cast<std::size_t>(bytes) / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), bytes);
  gt::require(bool(file), "cannot read complete FP32 input");
  return values;
}
double percentile(const std::vector<double>& sorted, double p) {
  const auto position = p * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  return sorted[lower] + (sorted[std::min(lower + 1, sorted.size() - 1)] - sorted[lower]) *
                             (position - static_cast<double>(lower));
}
std::vector<std::int64_t> physical_shape(const char* text, std::size_t elements) {
  std::string value(text);
  std::replace(value.begin(), value.end(), ',', ' ');
  std::istringstream dimensions(value);
  std::vector<std::int64_t> shape;
  std::int64_t dimension;
  std::size_t product = 1;
  while (dimensions >> dimension) {
    gt::require(dimension > 0 && std::uint64_t(dimension) <= elements / product,
                "invalid physical input shape");
    product *= static_cast<std::size_t>(dimension);
    shape.push_back(dimension);
  }
  gt::require(dimensions.eof() && !shape.empty() && product == elements,
              "physical shape differs from saved input bytes");
  return shape;
}
} // namespace

int main(int argc, char** argv) {
  try {
    gt::require(argc == 6 || argc == 8 || argc == 10,
                "usage: grid_sample_pipeline_benchmark METHOD CASE FEATURES.bin GRID.bin RAW.csv "
                "[MPK EXPECTED_FP32.bin [FEATURES_SHAPE GRID_SHAPE]]");
    const std::string method = argv[1];
    const bool mla = method == "mla_bf16", bf16 = method == "a65_bf16";
    gt::require(mla || bf16 || method == "a65_fp32", "unknown method");
    gt::require(mla == (argc >= 8), "MLA requires MPK and independent expected output");
    const auto c = std::find_if(std::begin(gt::comparison_cases), std::end(gt::comparison_cases),
                                [&](auto value) { return value.name == std::string(argv[2]); });
    gt::require(c != std::end(gt::comparison_cases), "unknown comparison geometry");
    gt::Frame frame(*c, bf16);
    frame.features = read(argv[3]);
    frame.grid = read(argv[4]);
    gt::require(frame.features.size() == std::size_t(c->height) * c->width * c->channels &&
                    frame.grid.size() == std::size_t(c->output_height) * c->output_width * 2,
                "input sizes differ from comparison geometry");
    if (bf16)
      for (auto& value : frame.features)
        value = std::bit_cast<float>(std::uint32_t(gt::bf16(value)) << 16);
    std::optional<neat::Model> model;
    std::vector<float> expected;
    auto pipeline = gt::graph(frame.options, neat::InputMemoryPolicy::Ev74);
    neat::TensorList inputs;
    if (mla) {
      model.emplace(argv[6]);
      pipeline = model->graph({.include_input = true, .include_output = true});
      const auto specs = model->input_specs();
      gt::require(specs.size() == 2, "MLA model must have two inputs");
      auto feature_shape = specs[0].shape, grid_shape = specs[1].shape;
      if (argc == 10) {
        feature_shape = physical_shape(argv[8], frame.features.size());
        grid_shape = physical_shape(argv[9], frame.grid.size());
        gt::require((specs[0].shape.empty() || specs[0].shape == feature_shape) &&
                        (specs[1].shape.empty() || specs[1].shape == grid_shape),
                    "explicit physical shape differs from the model contract");
      }
      gt::require(!feature_shape.empty() && !grid_shape.empty(),
                  "model input shapes absent; supply shapes from the MPK manifest");
      inputs = {neat::Tensor::from_vector(frame.features, feature_shape, neat::TensorMemory::EV74),
                neat::Tensor::from_vector(frame.grid, grid_shape, neat::TensorMemory::EV74)};
      expected = read(argv[7]);
    } else {
      inputs = frame.tensors(neat::TensorMemory::EV74);
    }
    neat::RunOptions options;
    options.queue_depth = 1;
    options.output_memory = neat::OutputMemory::ZeroCopy;
    options.startup_preflight = false;
    const auto cold = Clock::now();
    auto run = pipeline.build(inputs, options);
    const double build_us = std::chrono::duration<double, std::micro>(Clock::now() - cold).count();
    const auto call = [&] {
      if (!run.push(inputs))
        throw std::runtime_error("benchmark push: " + run.last_error());
      neat::Sample output;
      if (run.pull(30000, output) != neat::PullStatus::Ok || output.empty())
        throw std::runtime_error("benchmark pull failed or produced an empty output");
      return output;
    };
    const auto verify = [&](const neat::Sample& sample) {
      const auto tensors = neat::tensors_from_sample(sample);
      if (!mla) {
        frame.check(tensors);
      } else {
        gt::require(tensors.size() == 1 && tensors[0].dtype == neat::TensorDType::Float32,
                    "MLA must publish one FP32 tensor");
        const auto bytes = tensors[0].copy_dense_bytes_tight();
        gt::require(bytes.size() == expected.size() * sizeof(float) &&
                        std::memcmp(bytes.data(), expected.data(), bytes.size()) == 0,
                    "MLA differs from independent saved expected output");
      }
    };
    verify(call());
    for (int i = 0; i < 100; ++i)
      call();
    std::vector<double> samples;
    samples.reserve(20000);
    const auto minimum_end = Clock::now() + std::chrono::seconds(1);
    while (samples.size() < 1000 || Clock::now() < minimum_end) {
      const auto begin = Clock::now();
      const auto output = call();
      const auto end = Clock::now();
      samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    verify(call());
    run.close();
    const std::filesystem::path report(argv[5]);
    if (!report.parent_path().empty())
      std::filesystem::create_directories(report.parent_path());
    std::ofstream raw(report);
    raw << "sample,latency_us\n" << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < samples.size(); ++i)
      raw << i << ',' << samples[i] << '\n';
    raw.close();
    gt::require(bool(raw), "cannot write chronological samples");
    const auto mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    // Runtime log filtering may drain stdout after main returns. Keep the
    // completed validation and timing result beside its chronological samples.
    std::ofstream summary(report.string() + ".json");
    summary << std::fixed << std::setprecision(6) << "{\"method\":\"" << method << "\",\"case\":\""
            << c->name << "\",\"samples\":" << samples.size()
            << ",\"warmup\":100,\"mean_us\":" << mean << ",\"p50_us\":" << percentile(samples, 0.50)
            << ",\"p95_us\":" << percentile(samples, 0.95) << ",\"build_us\":" << build_us
            << ",\"boundary\":\"public_push_pull\",\"validation\":\"PASS\"}\n";
    summary.close();
    gt::require(bool(summary), "cannot write completed benchmark result");
    std::cout << std::fixed << std::setprecision(6) << "GRID_SAMPLE_BENCHMARK method=" << method
              << " case=" << c->name << " samples=" << samples.size() << " warmup=100"
              << " mean_us=" << mean << " p50_us=" << percentile(samples, 0.50)
              << " p95_us=" << percentile(samples, 0.95) << " build_us=" << build_us
              << " boundary=public_push_pull validation=PASS" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GRID_SAMPLE_BENCHMARK result=FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
