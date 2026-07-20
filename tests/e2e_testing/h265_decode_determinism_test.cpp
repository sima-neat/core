#include "asset_utils.h"
#include "gst/GstInit.h"
#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;
constexpr int kFrameCount = 30;
constexpr int kQueueDepth = 8;
constexpr int kPullTimeoutMs = 5000;

struct TuningCase {
  const char* name;
  const char* decoder_tuning;
  bool memory_opt;
};

struct FrameSignature {
  std::string contract;
  std::uint64_t payload_hash;
};

const std::array<TuningCase, 4> kTuningCases{{
    {"automatic", "auto", false},
    {"low-memory", "low-memory", false},
    {"throughput-low-latency", "throughput-low-latency", false},
    {"memory-opt", "auto", true},
}};

std::string h265_parser_fragment() {
  return "h265parse disable-passthrough=true config-interval=-1 ! capsfilter "
         "caps=\"video/x-h265,parsed=true,stream-format=(string)byte-stream,"
         "alignment=(string)au\"";
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U) {
      out << ",";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

simaai::neat::Sample pull_or_throw(simaai::neat::Run& run, const std::string& context) {
  simaai::neat::Sample sample;
  simaai::neat::PullError error;
  const simaai::neat::PullStatus status = run.pull(kPullTimeoutMs, sample, &error);
  if (status == simaai::neat::PullStatus::Ok) {
    return sample;
  }
  if (status == simaai::neat::PullStatus::Timeout) {
    throw std::runtime_error(context + ": timed out");
  }
  if (status == simaai::neat::PullStatus::Closed) {
    throw std::runtime_error(context + ": closed before all outputs arrived");
  }
  throw std::runtime_error(context + ": " +
                           (error.message.empty() ? "pull failed" : error.message));
}

std::vector<simaai::neat::Sample> extract_access_units() {
  namespace fs = std::filesystem;
  const fs::path fixture = sima_test::test_codec_perf_h265_fixture_path();
  require(fs::is_regular_file(fixture), "missing H.265 fixture: " + fixture.string());

  simaai::neat::Graph graph("h265-access-unit-extract");
  graph.add(simaai::neat::nodes::FileInput(fixture.string()));
  graph.add(simaai::neat::nodes::Custom(h265_parser_fragment()));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(kFrameCount + 1)));

  simaai::neat::RunOptions options;
  options.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run run = graph.build(options);

  std::vector<simaai::neat::Sample> access_units;
  while (true) {
    simaai::neat::Sample sample;
    simaai::neat::PullError error;
    const simaai::neat::PullStatus status = run.pull(kPullTimeoutMs, sample, &error);
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      throw std::runtime_error("H.265 fixture extraction timed out");
    }
    if (status == simaai::neat::PullStatus::Error) {
      throw std::runtime_error(error.message.empty() ? "H.265 fixture extraction failed"
                                                     : error.message);
    }

    require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Encoded,
            "H.265 fixture extraction produced a non-encoded sample");
    const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
    require(tensors.size() == 1U && tensors.front().storage != nullptr,
            "H.265 fixture extraction produced an invalid payload");
    require(!tensors.front().copy_payload_bytes().empty(),
            "H.265 fixture extraction produced an empty access unit");

    const int64_t frame_id = static_cast<int64_t>(access_units.size());
    const int64_t duration_ns = 1000000000LL / kFps;
    sample.frame_id = frame_id;
    sample.pts_ns = frame_id * duration_ns;
    sample.dts_ns = sample.pts_ns;
    sample.duration_ns = duration_ns;
    sample.stream_id = "h265-determinism";
    if (sample.caps_string.empty()) {
      sample.caps_string = "video/x-h265,parsed=true,stream-format=(string)byte-stream,"
                           "alignment=(string)au";
    }
    access_units.push_back(std::move(sample));
  }
  run.stop();

  require(access_units.size() == static_cast<std::size_t>(kFrameCount),
          "H.265 fixture must contain exactly " + std::to_string(kFrameCount) +
              " access units, got " + std::to_string(access_units.size()));
  return access_units;
}

simaai::neat::Graph make_decode_graph(const TuningCase& tuning, const simaai::neat::Sample& seed) {
  simaai::neat::Graph graph(std::string("h265-decode-") + tuning.name);

  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.caps_override = seed.caps_string;
  input.block = true;
  input.pool_max_buffers = kQueueDepth;
  input.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  graph.add(simaai::neat::nodes::Input(input));
  graph.add(simaai::neat::nodes::Custom(h265_parser_fragment()));

  simaai::neat::SimaDecodeOptions decode;
  decode.type = simaai::neat::SimaDecodeType::H265;
  decode.out_format = simaai::neat::FormatTag::NV12;
  decode.raw_output = false;
  decode.dec_width = kWidth;
  decode.dec_height = kHeight;
  decode.dec_fps = kFps;
  decode.decoder_tuning = tuning.decoder_tuning;
  decode.memory_opt = tuning.memory_opt;
  graph.add(simaai::neat::nodes::SimaDecode(decode));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(kQueueDepth)));
  return graph;
}

FrameSignature make_signature(const simaai::neat::Sample& sample, const std::string& context) {
  require(sample.owned, context + ": output sample is not owned");
  require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Image,
          context + ": output payload is not an Image");
  require(sample.media_type == "video/x-raw", context + ": output media type is not video/x-raw");

  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1U, context + ": expected one decoded tensor");
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.storage != nullptr, context + ": decoded tensor has no storage");
  require(tensor.storage->kind == simaai::neat::StorageKind::CpuOwned,
          context + ": decoded tensor is not stored in owned CPU memory");
  require(tensor.dtype == simaai::neat::TensorDType::UInt8,
          context + ": decoded tensor is not UInt8");
  require(tensor.width() == kWidth, context + ": decoded width mismatch");
  require(tensor.height() == kHeight, context + ": decoded height mismatch");
  require(tensor.is_nv12(), context + ": decoded format is not NV12");

  const std::vector<std::uint8_t> bytes = tensor.copy_nv12_contiguous();
  const std::size_t expected_bytes =
      static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) * 3U / 2U;
  require(bytes.size() == expected_bytes,
          context + ": decoded NV12 byte count mismatch: expected " +
              std::to_string(expected_bytes) + ", got " + std::to_string(bytes.size()));

  std::ostringstream contract;
  contract << "kind=" << static_cast<int>(sample.kind)
           << ";payload=" << static_cast<int>(simaai::neat::sample_payload_type(sample))
           << ";media=" << sample.media_type << ";tag=" << sample.payload_tag
           << ";format=" << sample.format << ";caps=" << sample.caps_string
           << ";owned=" << sample.owned << ";shape=" << shape_string(tensor.shape)
           << ";dtype=" << static_cast<int>(tensor.dtype)
           << ";layout=" << static_cast<int>(tensor.layout) << ";width=" << tensor.width()
           << ";height=" << tensor.height() << ";storage=" << static_cast<int>(tensor.storage->kind)
           << ";bytes=" << bytes.size() << ";pts=" << sample.pts_ns << ";dts=" << sample.dts_ns
           << ";duration=" << sample.duration_ns << ";frame_id=" << sample.frame_id;
  return {contract.str(), fnv1a64(bytes)};
}

std::vector<FrameSignature> decode_once(const TuningCase& tuning,
                                        const std::vector<simaai::neat::Sample>& access_units,
                                        int run_index) {
  simaai::neat::Graph graph = make_decode_graph(tuning, access_units.front());
  simaai::neat::RunOptions options;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.queue_depth = kQueueDepth;
  options.output_memory = simaai::neat::OutputMemory::Owned;
  options.startup_preflight = false;
  simaai::neat::Run run = graph.build(simaai::neat::Sample{access_units.front()}, options);

  std::exception_ptr producer_error;
  std::thread producer([&] {
    try {
      for (const simaai::neat::Sample& sample : access_units) {
        if (!run.push(simaai::neat::Sample{sample})) {
          throw std::runtime_error("push failed");
        }
      }
      run.close_input();
    } catch (...) {
      producer_error = std::current_exception();
      try {
        run.close_input();
      } catch (...) {
      }
    }
  });

  std::vector<FrameSignature> signatures;
  std::exception_ptr consumer_error;
  try {
    signatures.reserve(access_units.size());
    for (std::size_t frame = 0; frame < access_units.size(); ++frame) {
      const std::string context = std::string(tuning.name) + " run " + std::to_string(run_index) +
                                  " frame " + std::to_string(frame);
      signatures.push_back(make_signature(pull_or_throw(run, context), context));
    }
  } catch (...) {
    consumer_error = std::current_exception();
    try {
      run.close_input();
    } catch (...) {
    }
  }

  producer.join();
  if (!consumer_error && !producer_error) {
    simaai::neat::Sample extra;
    simaai::neat::PullError error;
    const simaai::neat::PullStatus status = run.pull(kPullTimeoutMs, extra, &error);
    if (status != simaai::neat::PullStatus::Closed) {
      consumer_error = std::make_exception_ptr(std::runtime_error(
          std::string(tuning.name) + " run " + std::to_string(run_index) +
          ": decoder did not close after exactly " + std::to_string(kFrameCount) + " outputs"));
    }
  }
  run.stop();

  if (producer_error) {
    std::rethrow_exception(producer_error);
  }
  if (consumer_error) {
    std::rethrow_exception(consumer_error);
  }
  return signatures;
}

void require_same_signatures(const std::vector<FrameSignature>& expected,
                             const std::vector<FrameSignature>& actual,
                             const std::string& context) {
  require(actual.size() == expected.size(), context + ": output count mismatch");
  for (std::size_t frame = 0; frame < expected.size(); ++frame) {
    require(actual[frame].contract == expected[frame].contract,
            context + " frame " + std::to_string(frame) + ": metadata mismatch\nexpected: " +
                expected[frame].contract + "\nactual:   " + actual[frame].contract);
    require(actual[frame].payload_hash == expected[frame].payload_hash,
            context + " frame " + std::to_string(frame) + ": NV12 payload hash mismatch");
  }
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    const std::vector<simaai::neat::Sample> access_units = extract_access_units();

    std::vector<FrameSignature> automatic_reference;
    for (const TuningCase& tuning : kTuningCases) {
      const std::vector<FrameSignature> first = decode_once(tuning, access_units, 1);
      const std::vector<FrameSignature> second = decode_once(tuning, access_units, 2);
      require_same_signatures(first, second, std::string(tuning.name) + " repeated run");

      if (automatic_reference.empty()) {
        automatic_reference = first;
      } else {
        require_same_signatures(automatic_reference, first,
                                std::string(tuning.name) + " versus automatic");
      }
      std::cout << "[OK] " << tuning.name << " decoded " << first.size()
                << " deterministic H.265 frames twice\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
