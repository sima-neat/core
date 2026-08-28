#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include <neat.h>
#include "model/internal/ModelInternal.h"
#include "pipeline/runtime/RunInternal.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

namespace neat = simaai::neat;

namespace {
std::string arg_value(int argc, char** argv, const std::string& key) {
  const std::string prefix = key + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] ? argv[i] : "";
    if (arg.rfind(prefix, 0) == 0)
      return arg.substr(prefix.size());
    if (arg == key && i + 1 < argc)
      return argv[++i] ? argv[i] : "";
  }
  return {};
}
int int_arg(int argc, char** argv, const std::string& key, int fallback) {
  const std::string raw = arg_value(argc, argv, key);
  if (raw.empty())
    return fallback;
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}
bool bool_arg(int argc, char** argv, const std::string& key, bool fallback) {
  const std::string raw = arg_value(argc, argv, key);
  if (raw.empty())
    return fallback;
  if (raw == "1" || raw == "true" || raw == "on" || raw == "yes")
    return true;
  if (raw == "0" || raw == "false" || raw == "off" || raw == "no")
    return false;
  return fallback;
}
std::size_t element_count(const std::vector<int64_t>& shape) {
  std::size_t count = 1;
  if (shape.empty())
    return 1;
  for (auto dim : shape)
    count *= static_cast<std::size_t>(dim > 0 ? dim : 1);
  return count;
}
neat::Tensor make_tensor(const neat::TensorSpec& spec, float fill) {
  std::vector<int64_t> shape = spec.shape.empty() ? std::vector<int64_t>{1} : spec.shape;
  std::vector<float> data(element_count(shape), fill);
  return neat::Tensor::from_vector(data, shape, neat::TensorMemory::EV74);
}
std::size_t checked_static_element_count(const std::vector<int64_t>& shape) {
  std::size_t count = 1U;
  for (const int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("--input-fp32 requires a fully static positive input shape");
    }
    const auto extent = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::runtime_error("--input-fp32 shape element count overflows size_t");
    }
    count *= extent;
  }
  return count;
}
neat::Tensor load_fp32_tensor(const std::string& path, const neat::TensorSpec& spec) {
  if (!spec.dtypes.empty() &&
      std::find(spec.dtypes.begin(), spec.dtypes.end(), neat::TensorDType::Float32) ==
          spec.dtypes.end()) {
    throw std::runtime_error("--input-fp32 is incompatible with the model input dtype contract");
  }
  const std::vector<int64_t> shape =
      spec.shape.empty() ? std::vector<int64_t>{1} : spec.shape;
  const std::size_t elements = checked_static_element_count(shape);
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::runtime_error("--input-fp32 byte count overflows size_t");
  }
  const std::size_t expected_bytes = elements * sizeof(float);
  std::error_code ec;
  const std::uintmax_t actual_bytes = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error("failed to stat --input-fp32 file '" + path + "': " +
                             ec.message());
  }
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error("--input-fp32 file size mismatch: expected " +
                             std::to_string(expected_bytes) + " bytes, got " +
                             std::to_string(actual_bytes));
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw std::runtime_error("--input-fp32 supports exact little-endian FP32 files only");
  }
  if (expected_bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("--input-fp32 file is too large for std::ifstream");
  }
  std::vector<float> data(elements);
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open --input-fp32 file '" + path + "'");
  }
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(expected_bytes));
  if (!input || input.gcount() != static_cast<std::streamsize>(expected_bytes)) {
    throw std::runtime_error("failed to read exact --input-fp32 bytes from '" + path + "'");
  }
  return neat::Tensor::from_vector(data, shape, neat::TensorMemory::EV74);
}
const char* dtype_token(neat::TensorDType dtype) {
  switch (dtype) {
  case neat::TensorDType::UInt8:
    return "uint8";
  case neat::TensorDType::Int8:
    return "int8";
  case neat::TensorDType::UInt16:
    return "uint16";
  case neat::TensorDType::Int16:
    return "int16";
  case neat::TensorDType::Int32:
    return "int32";
  case neat::TensorDType::BFloat16:
    return "bfloat16";
  case neat::TensorDType::Float32:
    return "float32";
  case neat::TensorDType::Float64:
    return "float64";
  }
  throw std::runtime_error("cannot serialize unknown TensorDType");
}
double percentile_ms(std::vector<double> samples, double p) {
  if (samples.empty())
    return 0.0;
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
bool no_external_stage(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return token == "NONE" || token == "OFF";
}
std::uint64_t fnv1a64_append(std::uint64_t hash, const void* data,
                             std::size_t size) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= kPrime;
  }
  return hash;
}
std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}
void persist_public_outputs(const std::filesystem::path& output_dir,
                            const std::string& input_fp32,
                            const std::vector<int64_t>& input_shape,
                            const neat::TensorList& outputs,
                            const std::vector<std::vector<std::uint8_t>>& output_bytes,
                            const std::vector<std::uint64_t>& output_hashes,
                            std::uint64_t combined_hash) {
  std::error_code ec;
  if (std::filesystem::exists(output_dir, ec)) {
    throw std::runtime_error("--output-dir already exists: " + output_dir.string());
  }
  if (ec) {
    throw std::runtime_error("failed to inspect --output-dir '" + output_dir.string() + "': " +
                             ec.message());
  }
  if (!std::filesystem::create_directories(output_dir, ec) || ec) {
    throw std::runtime_error("failed to create --output-dir '" + output_dir.string() + "': " +
                             ec.message());
  }

  nlohmann::json manifest{
      {"schema", "sima.neat.raw-public-outputs"},
      {"version", 1},
      {"byte_order", "little"},
      {"storage", "dense-row-major"},
      {"input",
       {{"source", input_fp32.empty() ? "deterministic-fill" : "fp32-raw-file"},
        {"file", input_fp32},
        {"dtype", "float32"},
        {"shape", input_shape},
        {"bytes", checked_static_element_count(input_shape) * sizeof(float)}}},
      {"combined_fnv1a64", hex64(combined_hash)},
      {"outputs", nlohmann::json::array()},
  };

  for (std::size_t index = 0; index < outputs.size(); ++index) {
    std::ostringstream filename;
    filename << "output-" << std::setw(3) << std::setfill('0') << index << ".raw";
    const std::filesystem::path output_path = output_dir / filename.str();
    std::ofstream raw(output_path, std::ios::binary | std::ios::trunc);
    if (!raw.is_open()) {
      throw std::runtime_error("failed to create raw public output '" + output_path.string() +
                               "'");
    }
    const auto& bytes = output_bytes[index];
    if (!bytes.empty()) {
      if (bytes.size() >
          static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("raw public output is too large for std::ofstream");
      }
      raw.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    }
    if (!raw.good()) {
      throw std::runtime_error("failed to write raw public output '" + output_path.string() +
                               "'");
    }
    const auto& tensor = outputs[index];
    manifest["outputs"].push_back(
        {{"ordinal", index},
         {"file", filename.str()},
         {"dtype", dtype_token(tensor.dtype)},
         {"shape", tensor.shape},
         {"bytes", bytes.size()},
         {"fnv1a64", hex64(output_hashes[index])},
         {"name", tensor.route.name},
         {"logical_index", tensor.route.logical_index},
         {"backend_name", tensor.route.backend_name}});
  }

  const std::filesystem::path manifest_path = output_dir / "manifest.json";
  std::ofstream manifest_file(manifest_path, std::ios::binary | std::ios::trunc);
  if (!manifest_file.is_open()) {
    throw std::runtime_error("failed to create output manifest '" + manifest_path.string() +
                             "'");
  }
  manifest_file << manifest.dump(2) << '\n';
  if (!manifest_file.good()) {
    throw std::runtime_error("failed to write output manifest '" + manifest_path.string() +
                             "'");
  }
}
} // namespace

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
  const bool mla_only = bool_arg(argc, argv, "--mla-only", false);
  const bool verbose = bool_arg(argc, argv, "--verbose", false);
  const bool correctness_hash =
      bool_arg(argc, argv, "--correctness-hash", false);
  const std::string input_fp32 = arg_value(argc, argv, "--input-fp32");
  const std::string output_dir = arg_value(argc, argv, "--output-dir");
  const std::string arena_dump = arg_value(argc, argv, "--arena-dump");
  const int early_stop_after =
      std::max(0, int_arg(argc, argv, "--early-stop-after", 0));
  const int early_stop_delay_ms =
      std::max(0, int_arg(argc, argv, "--early-stop-delay-ms", 0));
  const std::string mode =
      arg_value(argc, argv, "--mode").empty() ? "sync" : arg_value(argc, argv, "--mode");
  const int inflight = std::max(1, int_arg(argc, argv, "--inflight", 4));
  if (model_path.empty() || pre.empty() || post.empty() || measured <= 0 || warmup < 0) {
    std::cerr << "Usage: evo_tput_bench --model <path> --pre <A65|EV74|NONE> "
                 "--post <A65|EV74|NONE> "
                 "[--warmup N] [--measured N] [--timeout-ms MS] [--cleanup 0|1] "
                 "[--mode sync|async] [--inflight N] [--mla-only 0|1] [--verbose 0|1] "
                 "[--correctness-hash 0|1] [--input-fp32 PATH] [--output-dir DIR] "
                 "[--arena-dump PATH]\n";
    return 2;
  }
  if (mode != "sync" && mode != "async") {
    std::cerr << "EVO_TPUT_FAIL stage=args reason=bad_mode mode=" << mode << "\n";
    return 2;
  }
  try {
    neat::Model::Options opt;
    opt.preprocess.kind = neat::InputKind::Tensor;
    // A compiler-authored full ModelExecutionPlan may consume the public FP32
    // tensor directly (RF-DETR is the qualification case). NONE means exactly
    // that application boundary: do not invent a second pre/post adapter
    // around the model-owned MLA/A65/CVU command schedule.
    opt.preprocess.enable = no_external_stage(pre) ? neat::AutoFlag::Off
                                                    : neat::AutoFlag::On;
    if (!no_external_stage(pre)) {
      opt.processcvu.pre_run_target = pre;
    }
    if (!no_external_stage(post)) {
      opt.processcvu.post_run_target = post;
    }
    opt.inference_terminal.mla_only = mla_only;
    if (verbose) {
      opt.verbose.level = neat::VerbosityLevel::Verbose;
      opt.verbose.gstreamer = true;
      opt.verbose.plugins = true;
    }
    opt.cleanup_extracted_model_data = cleanup;

    std::cout << "EVO_TPUT_CONFIG model=" << model_path << " model_name=" << base_name(model_path)
              << " pre=" << pre << " post=" << post << " warmup=" << warmup
              << " measured=" << measured << " timeout_ms=" << timeout_ms
              << " plugin_latency=" << (plugin_latency ? 1 : 0)
              << " startup_preflight=" << (startup_preflight ? 1 : 0) << " mode=" << mode
              << " inflight=" << inflight << " mla_only=" << (mla_only ? 1 : 0)
              << " verbose=" << (verbose ? 1 : 0)
              << " correctness_hash=" << (correctness_hash ? 1 : 0)
              << " input_fp32=" << (input_fp32.empty() ? "<generated>" : input_fp32)
              << " output_dir=" << (output_dir.empty() ? "<none>" : output_dir)
              << " arena_dump=" << (arena_dump.empty() ? "<none>" : arena_dump)
              << " early_stop_after=" << early_stop_after
              << " early_stop_delay_ms=" << early_stop_delay_ms << "\n"
              << std::flush;

    const auto ctor0 = std::chrono::steady_clock::now();
    neat::Model model(model_path, opt);
    const auto ctor1 = std::chrono::steady_clock::now();
    std::cout << "EVO_TPUT_CTOR_DONE ctor_seconds="
              << std::chrono::duration<double>(ctor1 - ctor0).count() << "\n"
              << std::flush;

    neat::TensorList inputs;
    const auto specs = model.input_specs();
    inputs.reserve(specs.size());
    std::size_t elems = 0;
    if (!input_fp32.empty()) {
      if (specs.size() != 1U) {
        throw std::runtime_error("--input-fp32 requires exactly one public model input");
      }
      auto t = load_fp32_tensor(input_fp32, specs.front());
      elems = checked_static_element_count(t.shape);
      inputs.push_back(std::move(t));
    } else {
      for (std::size_t i = 0; i < specs.size(); ++i) {
        auto t = make_tensor(specs[i], 0.01f * static_cast<float>(i + 1));
        elems += element_count(specs[i].shape.empty() ? std::vector<int64_t>{1} : specs[i].shape);
        inputs.push_back(std::move(t));
      }
    }
    std::cout << "EVO_TPUT_INPUTS count=" << inputs.size() << " float_elements=" << elems << "\n"
              << std::flush;

    const auto build0 = std::chrono::steady_clock::now();
    neat::RunOptions run_opt;
    run_opt.startup_preflight = startup_preflight;
    if (!arena_dump.empty()) {
      run_opt.output_memory = neat::OutputMemory::ZeroCopy;
    }
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

    if (correctness_hash || !output_dir.empty() || !arena_dump.empty()) {
      auto outputs = runner.run(inputs, timeout_ms);
      if (outputs.empty()) {
        std::cerr << "EVO_TPUT_FAIL stage=output_capture reason=no_outputs\n";
        return 10;
      }
      constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
      std::uint64_t combined = kOffsetBasis;
      std::size_t total_bytes = 0U;
      std::ostringstream tensor_hashes;
      tensor_hashes << std::hex << std::setfill('0');
      std::vector<std::vector<std::uint8_t>> captured_bytes;
      std::vector<std::uint64_t> captured_hashes;
      captured_bytes.reserve(outputs.size());
      captured_hashes.reserve(outputs.size());
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::vector<std::uint8_t> bytes = outputs[i].copy_dense_bytes_tight();
        if (bytes.empty() && outputs[i].dense_bytes_tight() != 0U) {
          std::cerr << "EVO_TPUT_FAIL stage=output_capture tensor=" << i
                    << " reason=copy_failed\n";
          return 11;
        }
        std::uint64_t tensor_hash = kOffsetBasis;
        tensor_hash = fnv1a64_append(tensor_hash, bytes.data(), bytes.size());
        const std::uint64_t size = static_cast<std::uint64_t>(bytes.size());
        combined = fnv1a64_append(combined, &size, sizeof(size));
        combined = fnv1a64_append(combined, bytes.data(), bytes.size());
        total_bytes += bytes.size();
        if (i != 0U) {
          tensor_hashes << ',';
        }
        tensor_hashes << std::setw(16) << tensor_hash;
        captured_hashes.push_back(tensor_hash);
        captured_bytes.push_back(std::move(bytes));
      }
      if (!arena_dump.empty()) {
        if (!outputs.front().storage) {
          throw std::runtime_error("--arena-dump requires zero-copy output storage");
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
          if (!outputs[i].storage) {
            throw std::runtime_error("--arena-dump output has no zero-copy storage");
          }
          std::cout << "EVO_ARENA_OUTPUT ordinal=" << i
                    << " storage_bytes=" << outputs[i].storage->size_bytes
                    << " byte_offset=" << outputs[i].byte_offset
                    << " physical_byte_offset=" << outputs[i].route.physical_byte_offset
                    << " memory_index=" << outputs[i].route.memory_index << "\n";
        }
        std::error_code ec;
        if (std::filesystem::exists(arena_dump, ec)) {
          throw std::runtime_error("--arena-dump path already exists: " + arena_dump);
        }
        if (ec) {
          throw std::runtime_error("failed to inspect --arena-dump path '" + arena_dump +
                                   "': " + ec.message());
        }
        const neat::Mapping arena = outputs.front().storage->map(neat::MapMode::Read);
        if (!arena.data || arena.size_bytes == 0U ||
            arena.size_bytes != outputs.front().storage->size_bytes) {
          throw std::runtime_error("--arena-dump could not map the exact output carrier");
        }
        if (arena.size_bytes >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
          throw std::runtime_error("--arena-dump carrier is too large for std::ofstream");
        }
        std::ofstream raw(arena_dump, std::ios::binary | std::ios::trunc);
        if (!raw.is_open()) {
          throw std::runtime_error("failed to create --arena-dump file '" + arena_dump + "'");
        }
        raw.write(static_cast<const char*>(arena.data),
                  static_cast<std::streamsize>(arena.size_bytes));
        if (!raw.good()) {
          throw std::runtime_error("failed to write --arena-dump file '" + arena_dump + "'");
        }
        std::cout << "EVO_ARENA_DUMP status=PASS file=" << arena_dump
                  << " bytes=" << arena.size_bytes << "\n";
      }
      if (!output_dir.empty()) {
        persist_public_outputs(output_dir, input_fp32, inputs.front().shape, outputs,
                               captured_bytes, captured_hashes, combined);
        std::cout << "EVO_OUTPUT_DUMP status=PASS dir=" << output_dir
                  << " outputs=" << outputs.size() << " bytes=" << total_bytes
                  << " combined=" << hex64(combined) << "\n";
      }
      runner.close();
      if (correctness_hash) {
        std::cout << "EVO_CORRECTNESS_HASH status=PASS pre=" << pre
                  << " post=" << post << " outputs=" << outputs.size()
                  << " bytes=" << total_bytes << " combined=" << std::hex
                  << std::setw(16) << std::setfill('0') << combined << std::dec
                  << " tensors=" << tensor_hashes.str() << "\n";
      }
      return 0;
    }

    if (early_stop_after > 0) {
      for (int i = 0; i < early_stop_after; ++i) {
        if (!runner.push(inputs)) {
          std::cerr << "EVO_TPUT_FAIL stage=early_stop_push iter=" << i
                    << " reason=push_failed\n";
          return 9;
        }
      }
      if (early_stop_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(early_stop_delay_ms));
      }
      const auto close_start = std::chrono::steady_clock::now();
      runner.close();
      const auto close_end = std::chrono::steady_clock::now();
      std::cout << "EVO_EARLY_STOP_RESULT status=PASS pushed="
                << early_stop_after << " delay_ms=" << early_stop_delay_ms
                << " close_ms="
                << std::chrono::duration<double, std::milli>(close_end - close_start).count()
                << "\n";
      return 0;
    }

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
        lat_ms.push_back(
            std::chrono::duration<double, std::milli>(t_pull - push_times.front()).count());
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
    const double mean_ms =
        std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / static_cast<double>(lat_ms.size());
    const auto [min_it, max_it] = std::minmax_element(lat_ms.begin(), lat_ms.end());
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "EVO_TPUT_RESULT status=PASS model_name=" << base_name(model_path)
              << " pre=" << pre << " post=" << post << " mode=" << mode << " inflight=" << inflight
              << " measured=" << measured << " outputs=" << output_count
              << " measured_seconds=" << meas_s
              << " fps=" << (static_cast<double>(measured) / meas_s) << " mean_ms=" << mean_ms
              << " min_ms=" << *min_it << " p50_ms=" << percentile_ms(lat_ms, 50.0)
              << " p90_ms=" << percentile_ms(lat_ms, 90.0)
              << " p95_ms=" << percentile_ms(lat_ms, 95.0)
              << " p99_ms=" << percentile_ms(lat_ms, 99.0) << " max_ms=" << *max_it << "\n";
    std::cout << report.to_text() << std::flush;

    std::cout << "EVO_RUN_STATS" << " avg_latency_ms=" << report.end_to_end.avg_ms
              << " p50_latency_ms=" << report.end_to_end.p50_ms
              << " p95_latency_ms=" << report.end_to_end.p95_ms
              << " avg_push_us=" << report.input.avg_push_us
              << " avg_pull_wait_us=" << report.input.avg_pull_wait_us
              << " avg_alloc_us=" << report.input.avg_alloc_us
              << " avg_map_us=" << report.input.avg_map_us
              << " avg_copy_us=" << report.input.avg_copy_us
              << " push_count=" << report.input.push_count
              << " pull_count=" << report.input.pull_count << "\n";

    std::vector<neat::GraphNodeMetrics> stages = report.node_metrics;
    std::sort(stages.begin(), stages.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(a.latency.total_ms, a.latency.max_ms, a.label, a.node_id) >
             std::make_tuple(b.latency.total_ms, b.latency.max_ms, b.label, b.node_id);
    });
    const std::size_t stage_n = std::min<std::size_t>(stages.size(), 8U);
    for (std::size_t i = 0; i < stage_n; ++i) {
      const auto& s = stages[i];
      const std::string name =
          !s.label.empty() ? s.label : (!s.node_id.empty() ? s.node_id : s.kind);
      std::cout << "EVO_STAGE_TOP rank=" << (i + 1) << " name=" << name
                << " samples=" << s.latency.samples << " total_ms=" << s.latency.total_ms
                << " avg_ms=" << s.latency.avg_ms << " max_ms=" << s.latency.max_ms << "\n";
    }
    std::vector<neat::GraphElementMetrics> elements;
    for (const auto& node : report.node_metrics) {
      elements.insert(elements.end(), node.elements.begin(), node.elements.end());
    }
    std::sort(elements.begin(), elements.end(), [](const auto& a, const auto& b) {
      return std::make_tuple(a.latency.total_ms, a.latency.max_ms, a.name) >
             std::make_tuple(b.latency.total_ms, b.latency.max_ms, b.name);
    });
    const std::size_t elem_n = std::min<std::size_t>(elements.size(), 12U);
    for (std::size_t i = 0; i < elem_n; ++i) {
      const auto& e = elements[i];
      std::cout << "EVO_ELEMENT_TOP rank=" << (i + 1) << " name=" << e.name
                << " samples=" << e.latency.samples << " total_ms=" << e.latency.total_ms
                << " avg_ms=" << e.latency.avg_ms << " min_ms=" << e.latency.min_ms
                << " max_ms=" << e.latency.max_ms << "\n";
    }
    runner.close();
    return 0;
  } catch (const neat::NeatError& ex) {
    for (const auto& message : ex.report().bus) {
      std::cerr << "EVO_TPUT_BUS type=" << message.type << " src=" << message.src
                << " detail=" << message.detail << "\n";
    }
    std::cerr << "EVO_TPUT_FAIL exception=" << ex.what() << "\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "EVO_TPUT_FAIL exception=" << ex.what() << "\n";
    return 1;
  }
}
