#include "asset_utils.h"
#include "gst/GstInit.h"
#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <glib.h>

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;
constexpr int kFrameCount = 30;
constexpr int kQueueDepth = 8;
constexpr int kPullTimeoutMs = 5000;

constexpr std::array<std::string_view, kFrameCount> kH265GoldenSha256{{
    "ec6af9d13bfe86c3fea63cc8dc30505db679ac71639a6cb6f59880a22d68fbc5",
    "008c76630f18a0f9f493dae905308f565ce9138a386ca640db19555f5e14e685",
    "eb13b5d4fded2262b79320a24f6a3559d01503b3743289f047570c54d583a966",
    "91a59ce3d09b4576d68d30168fd12b86644f44e5ce6df775d036c83d34bdcb23",
    "e408e8487a7a29bfd9d077f5d216b8cb87656e7e61c56353c031cfdf814b5e10",
    "e06a910884773c726341c840217c11d48a7b12bc7bf6979eaee6dde6169dca5b",
    "67e4db98b4319a59cf3ed95e26a8ca468f95f02c6b677466217039716cf7a208",
    "3d93f98542ac158480af359f38ed3101d211bee297feb136822a114559b9e53f",
    "f14c01f133fcd0c385797614eb4562e1a23e54b20ab91ea22d46794b25683cca",
    "f240de6d25579330a7bfb88b5e5c645e6cf0d45e85ff9df9c8484281c52d1820",
    "aabdf82ff3d1f26cd335e705f2925744fcb2db87976d8b74c408f80d5b8553c7",
    "2bf645877bace2bb5ee575dc3222c1b06951ecaa9b324817540b7724635ef0b7",
    "792dcba71f800f3711b120c2a0728ff3d95d18a0f6464d68a49d358205ba30e8",
    "60966e9d4231861a9f7069db6c00991e02a6f7e2aa3a04b5c3999f340e42003d",
    "1ee26515718832477b11640bd8dd54ae638ca6453e42f9400d9996888ed66e2f",
    "4afe394c055c27808e10fed515aacc9847652f69d15a2771076ae17c4e269285",
    "ac32e018613f56e05f042c48bef1c8bc93faba88c1c0fbc6c907e1ffc941acbe",
    "9f2f22f5ebdbac0c7d379e3a19361feb259a498d13c11b87bea4bca48a5f1414",
    "ca7c027e635607d5b036b7d8cf32695d6c0e87aafaa03b2d685213ed71765ebc",
    "8b344ea03a7d43768bfba88fc8c07100275c20f7de86707a2077d6b8bc3eb11a",
    "16eef8217dd291142b4e1d126361ba01ab4947f85f8b15440d0f80b6f7034651",
    "682279ee8b5d12d953cdc107df1b069fd8d5d8f9e71eab2d65f4fe37cc36593d",
    "8480ae67c8d9601ea920abc171362602e4429b4fc6c86ec2e475f040567c8ea2",
    "0c70fd929d99ca6f3e838026a178294fe4173f97b0a4c267126c7f4316be4283",
    "967c79655b1a7ac2ad913e5f2845b1e6f0a51fcdfe25a6d0fb3598d7893090e5",
    "fb254a33f94d763d5fbe98541c3e58006c237d4451a12d447be108391d657f8f",
    "7e3591272c67cf67f4d4e857e12e549b0c100d3fa990837c6a2826cc1cc2684e",
    "a1e43d4c9aaf4ba66f9f8c39db503d3eaef5385d3cef31f12628ac74b9af52e6",
    "d67b3720829692572053afa11a38c7c0e1f09df101d607d65fb78f194db5b51d",
    "cdb2ab4921a783006c1657dd069357e869e762cb3461ea2842a7c2b3aa3d1bc2",
}};

struct TuningCase {
  const char* name;
  const char* decoder_tuning;
  bool memory_opt;
};

struct FrameSignature {
  std::string contract;
  std::uint64_t payload_hash;
  std::string payload_sha256;
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

std::string sha256_bytes(const std::vector<std::uint8_t>& bytes) {
  gchar* digest = g_compute_checksum_for_data(G_CHECKSUM_SHA256, bytes.data(), bytes.size());
  require(digest != nullptr, "failed to compute decoded payload SHA-256");
  std::string result(digest);
  g_free(digest);
  return result;
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
  require(graph.describe_backend(false).find("h265parse") == std::string::npos,
          "direct parsed H.265 decode must not insert a parser");
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
  return {contract.str(), fnv1a64(bytes), sha256_bytes(bytes)};
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
      if (std::getenv("SIMA_H265_SIGNATURE_DEBUG") != nullptr) {
        std::cerr << "[H265-SIGNATURE] tuning=" << tuning.name
                  << " run=" << run_index << " frame=" << frame
                  << " fnv1a64=0x" << std::hex << signatures.back().payload_hash
                  << std::dec << " sha256=" << signatures.back().payload_sha256
                  << "\n";
      }
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
            context + " frame " + std::to_string(frame) +
                ": NV12 payload hash mismatch expected=0x" +
                [&] { std::ostringstream out; out << std::hex << expected[frame].payload_hash; return out.str(); }() +
                " actual=0x" +
                [&] { std::ostringstream out; out << std::hex << actual[frame].payload_hash; return out.str(); }());
    require(actual[frame].payload_sha256 == expected[frame].payload_sha256,
            context + " frame " + std::to_string(frame) +
                ": NV12 payload SHA-256 mismatch expected=" + expected[frame].payload_sha256 +
                " actual=" + actual[frame].payload_sha256);
  }
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    const std::vector<simaai::neat::Sample> access_units = extract_access_units();
    const char* const tuning_filter = std::getenv("SIMA_H265_TUNING_FILTER");

    std::vector<FrameSignature> automatic_reference;
    std::size_t tuning_count = 0;
    for (const TuningCase& tuning : kTuningCases) {
      if (tuning_filter != nullptr && std::string_view(tuning_filter) != tuning.name) {
        continue;
      }
      ++tuning_count;
      const std::vector<FrameSignature> first = decode_once(tuning, access_units, 1);
      const std::vector<FrameSignature> second = decode_once(tuning, access_units, 2);
      require_same_signatures(first, second, std::string(tuning.name) + " repeated run");

      if (automatic_reference.empty()) {
        automatic_reference = first;
        for (std::size_t frame = 0; frame < first.size(); ++frame) {
          require(first[frame].payload_sha256 == kH265GoldenSha256[frame],
                  "H.265 frame " + std::to_string(frame) +
                      " does not match the frozen .1.20 golden");
          std::cout << "[GOLDEN] codec=h265 frame=" << frame
                    << " sha256=" << first[frame].payload_sha256 << "\n";
        }
      } else {
        require_same_signatures(automatic_reference, first,
                                std::string(tuning.name) + " versus automatic");
      }
      std::cout << "[OK] " << tuning.name << " decoded " << first.size()
                << " deterministic H.265 frames twice\n";
    }
    require(tuning_count != 0,
            std::string("unknown SIMA_H265_TUNING_FILTER: ") +
                (tuning_filter != nullptr ? tuning_filter : ""));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
