#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "pipeline/Graph.h"

#include "asset_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kFrames = 10;
constexpr int kPullTimeoutMs = 20000;
constexpr std::size_t kResNet50Classes = 1000;
constexpr const char* kDecoderName = "decoder_h265_rtsp";

std::string trim_copy(const std::string& value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

std::string h265_url_from_env() {
  if (const char* url = std::getenv("SIMANEAT_TEST_RTSP_H265_URL"); url && *url) {
    return trim_copy(url);
  }

  const char* urls = std::getenv("SIMANEAT_TEST_RTSP_H265_URLS");
  if (!urls || !*urls) {
    return {};
  }
  const std::string values(urls);
  const std::size_t delimiter = values.find_first_of(",; \t\r\n");
  return trim_copy(values.substr(0, delimiter));
}

int h265_fps_from_env() {
  const char* value = std::getenv("SIMANEAT_TEST_RTSP_H265_FPS");
  if (!value || !*value) {
    throw std::runtime_error("SIMANEAT_TEST_RTSP_H265_FPS is required");
  }

  char* end = nullptr;
  const long fps = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || fps <= 0) {
    throw std::runtime_error("SIMANEAT_TEST_RTSP_H265_FPS must be a positive integer");
  }
  return static_cast<int>(fps);
}

simaai::neat::Graph make_graph(const std::string& url, int source_fps,
                               const std::string& model_path) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions source;
  source.url = url;
  source.codec = simaai::neat::nodes::groups::RtspCodec::H265;
  source.source_fps = source_fps;
  source.decoder_name = kDecoderName;
  source.decoder_raw_output = true;
  source.decoder_next_element = "CVU";

  simaai::neat::Model::Options model_options;
  model_options.preprocess.kind = simaai::neat::InputKind::Image;
  model_options.preprocess.enable = simaai::neat::AutoFlag::On;
  model_options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_options.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  model_options.upstream_name = kDecoderName;
  simaai::neat::Model model(model_path, model_options);

  simaai::neat::Model::RouteOptions route;
  route.include_input = false;
  route.include_output = false;

  simaai::neat::Graph graph("h265-rtsp-resnet50-e2e");
  graph.add(simaai::neat::nodes::groups::RtspDecodedInput(source));
  graph.add(model.graph(route));
  graph.add(simaai::neat::nodes::Output());
  return graph;
}

void require_valid_resnet50_output(const simaai::neat::Sample& sample, int frame) {
  const std::string context = "frame " + std::to_string(frame);
  require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Tensor,
          context + ": model output payload is not a Tensor");
  require(sample.media_type == "application/vnd.simaai.tensor",
          context + ": model output media type mismatch");
  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1U, context + ": expected one ResNet50 output tensor");
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.storage != nullptr, context + ": output tensor has no storage");
  require(tensor.storage->kind == simaai::neat::StorageKind::CpuOwned,
          context + ": output tensor is not stored in owned CPU memory");
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

} // namespace

int main() {
  const std::string url = h265_url_from_env();
  if (url.empty()) {
    return skip_long_test("set SIMANEAT_TEST_RTSP_H265_URL or SIMANEAT_TEST_RTSP_H265_URLS");
  }

  try {
    const int source_fps = h265_fps_from_env();
    const std::string model_path = sima_test::resolve_resnet50_tar();
    require(!model_path.empty(),
            "ResNet50 model pack not found; set SIMA_MODEL_TAR or SIMA_RESNET50_TAR");

    simaai::neat::Graph graph = make_graph(url, source_fps, model_path);
    simaai::neat::RunOptions run_options;
    run_options.output_memory = simaai::neat::OutputMemory::Owned;
    simaai::neat::Run run = graph.build(run_options);

    for (int frame = 0; frame < kFrames; ++frame) {
      const auto output = run.pull(kPullTimeoutMs);
      require(output.has_value(), "timed out waiting for frame " + std::to_string(frame));
      require_valid_resnet50_output(*output, frame);
    }
    run.close();

    std::cout << "[OK] H.265 RTSP -> ResNet50 outputs=" << kFrames << " source_fps=" << source_fps
              << "\n";
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}
