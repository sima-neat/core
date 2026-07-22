#include "model/Model.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "perf/codec_perf_common.h"
#include "pipeline/Graph.h"

#include "asset_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;
constexpr int kFrames = 8;
constexpr int kQueueDepth = 16;
constexpr int kPullTimeoutMs = 20000;
constexpr std::size_t kResNet50Classes = 1000;

simaai::neat::RunOptions e2e_run_options() {
  simaai::neat::RunOptions options;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.queue_depth = kQueueDepth;
  // Decoded NV12 stays zero-copy into CVU; only the terminal 4 KB score tensor is app-owned.
  options.output_memory = simaai::neat::OutputMemory::Owned;
  options.advanced.copy_input = false;
  options.startup_preflight = false;
  return options;
}

simaai::neat::Graph make_graph(const simaai::neat::Sample& seed, const std::string& model_path,
                               const std::string& mode, int output_buffers) {
  const std::string decoder_name = "decoder_h265_" + mode;
  simaai::neat::Graph graph("h265-decode-resnet50-" + mode);

  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.caps_override = seed.caps_string;
  input.block = true;
  input.pool_max_buffers = kQueueDepth;
  input.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  graph.add(simaai::neat::nodes::Input(input));
  graph.add(simaai::neat::nodes::Custom(sima_codec_perf::h265_parser_fragment()));

  std::ostringstream decoder;
  decoder << "neatdecoder name=" << decoder_name
          << " sima-allocator-type=2 dec-type=h265 dec-fmt=NV12 next-element=CVU"
          << " dec-width=" << kWidth << " dec-height=" << kHeight << " dec-fps=" << kFps
          << " dec-ip-cnt=8 num-buffers=64 zero-copy-output=true";
  graph.add(simaai::neat::nodes::Custom(decoder.str()));
  graph.add(
      simaai::neat::nodes::CapsRaw("NV12", kWidth, kHeight, kFps, simaai::neat::CapsMemory::Any));

  simaai::neat::Model::Options model_options;
  model_options.preprocess.kind = simaai::neat::InputKind::Image;
  model_options.preprocess.enable = simaai::neat::AutoFlag::On;
  model_options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_options.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  model_options.upstream_name = decoder_name;
  simaai::neat::Model model(model_path, model_options);

  simaai::neat::Model::RouteOptions route;
  route.include_input = false;
  route.include_output = false;
  graph.add(model.graph(route));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(output_buffers)));
  return graph;
}

void require_zero_copy_pipeline(const simaai::neat::Graph& graph, const std::string& mode) {
  const std::string& pipeline = graph.last_pipeline();
  require(!pipeline.empty(), mode + ": built pipeline string is empty");
  require_contains(pipeline, "neatdecoder name=decoder_h265_" + mode,
                   mode + ": H.265 decoder is missing from the built pipeline");
  require_contains(pipeline, "next-element=CVU",
                   mode + ": decoder is not configured for the CVU path");
  require_contains(pipeline, "zero-copy-output=true",
                   mode + ": decoder is not configured for zero-copy output");
}

void require_valid_resnet50_output(const simaai::neat::Sample& sample, const std::string& context) {
  require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Tensor,
          context + ": model output payload is not a Tensor");
  require(sample.media_type == "application/vnd.simaai.tensor",
          context + ": model output media type mismatch");

  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1U, context + ": expected one ResNet50 output tensor");
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.storage != nullptr, context + ": output tensor has no storage");
  require(tensor.storage->kind == simaai::neat::StorageKind::CpuOwned,
          context + ": terminal model output is not stored in owned CPU memory");
  require(tensor.dtype == simaai::neat::TensorDType::Float32,
          context + ": output tensor is not Float32");
  require(tensor.is_dense(), context + ": output tensor is not dense");
  require(tensor.dense_bytes_tight() == kResNet50Classes * sizeof(float),
          context + ": output tensor is not exactly 1000 Float32 scores");

  const simaai::neat::Mapping mapping = tensor.map(simaai::neat::MapMode::Read);
  require(mapping.data != nullptr, context + ": output tensor is not CPU-readable");
  require(mapping.size_bytes >= kResNet50Classes * sizeof(float),
          context + ": output tensor contains fewer than 1000 scores");
  const auto* scores = static_cast<const float*>(mapping.data);
  require(std::all_of(scores, scores + kResNet50Classes,
                      [](float value) { return std::isfinite(value); }),
          context + ": output tensor contains a non-finite score");
}

void run_sync(const std::string& model_path,
              const std::vector<simaai::neat::Sample>& access_units) {
  simaai::neat::Graph graph = make_graph(access_units.front(), model_path, "sync", 1);
  const simaai::neat::RunOptions options = e2e_run_options();

  const simaai::neat::Sample output =
      graph.run(simaai::neat::Sample{access_units.front()}, options);
  require_valid_resnet50_output(output.front(), "sync frame 0");
  require_zero_copy_pipeline(graph, "sync");

  std::cout << "[OK] sync Graph::run H.265 -> ResNet50 zero-copy outputs=1\n";
}

void run_async(const std::string& model_path,
               const std::vector<simaai::neat::Sample>& access_units) {
  simaai::neat::Graph graph = make_graph(access_units.front(), model_path, "async", kQueueDepth);
  simaai::neat::Run run =
      graph.build(simaai::neat::Sample{access_units.front()}, e2e_run_options());
  require_zero_copy_pipeline(graph, "async");

  std::exception_ptr producer_error;
  std::thread producer([&] {
    try {
      for (const simaai::neat::Sample& sample : access_units) {
        if (!run.push(simaai::neat::Sample{sample})) {
          throw std::runtime_error("async push failed");
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

  std::exception_ptr consumer_error;
  try {
    for (std::size_t frame = 0; frame < access_units.size(); ++frame) {
      const std::optional<simaai::neat::Sample> output = run.pull(kPullTimeoutMs);
      require(output.has_value(), "async pull timed out at frame " + std::to_string(frame));
      require_valid_resnet50_output(*output, "async frame " + std::to_string(frame));
    }
  } catch (...) {
    consumer_error = std::current_exception();
    try {
      run.close_input();
    } catch (...) {
    }
  }

  producer.join();
  run.stop();
  if (producer_error) {
    std::rethrow_exception(producer_error);
  }
  if (consumer_error) {
    std::rethrow_exception(consumer_error);
  }

  std::cout << "[OK] async Graph::build push/pull H.265 -> ResNet50 zero-copy outputs="
            << access_units.size() << "\n";
}

} // namespace

int main() {
  try {
    const std::string model_path = sima_test::resolve_resnet50_tar();
    require(!model_path.empty(),
            "ResNet50 model pack not found; set SIMA_MODEL_TAR or SIMA_RESNET50_TAR");

    const sima_codec_perf::CodecPerfConfig config{
        .scenario_id = "h265-decode-resnet50-e2e",
        .run_mode = "e2e",
        .decode_type = simaai::neat::SimaDecodeType::H265,
        .width = kWidth,
        .height = kHeight,
        .fps = kFps,
    };
    const std::vector<simaai::neat::Sample> access_units = sima_codec_perf::make_sample_sequence(
        sima_codec_perf::extract_h265_access_units(kFrames), config, kFrames);

    run_sync(model_path, access_units);
    run_async(model_path, access_units);
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}
