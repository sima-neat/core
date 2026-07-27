#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "pipeline/Graph.h"

#include "asset_utils.h"
#include "resnet50_test_utils.h"
#include "rtsp_probe_utils.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kFrames = 10;
constexpr int kPullTimeoutMs = 20000;
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

} // namespace

int main() {
  const std::string url = h265_url_from_env();
  if (url.empty()) {
    return skip_long_test("set SIMANEAT_TEST_RTSP_H265_URL or SIMANEAT_TEST_RTSP_H265_URLS");
  }

  try {
    const int source_fps = sima_test::probe_rtsp_source_fps(url);
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
      sima_test::require_valid_resnet50_output(*output, "frame " + std::to_string(frame));
    }
    run.close();

    std::cout << "[OK] H.265 RTSP -> ResNet50 outputs=" << kFrames << " source_fps=" << source_fps
              << "\n";
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}
