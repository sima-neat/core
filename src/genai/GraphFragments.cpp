#include "genai/GraphFragments.h"

#include "genai/ASRModel.h"
#include "genai/VisionLanguageModel.h"
#include "genai/internal/SpeechTranscriberNodeFactory.h"
#include "genai/internal/VisionLanguageNodeFactory.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/PayloadType.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::genai::graphs {
namespace {

std::size_t add_input_endpoint(Graph& graph, const char* name) {
  InputOptions opt;
  opt.payload_type = PayloadType::Tensor;
  return graph.append_pipeline_vertex_for_internal_graph_(
      std::make_shared<Input>(std::string(name), std::move(opt)));
}

std::size_t add_output_endpoint(Graph& graph, const char* name) {
  return graph.append_pipeline_vertex_for_internal_graph_(
      std::make_shared<Output>(std::string(name)));
}

void connect_input_to_stage(Graph& graph, std::size_t input, std::size_t stage, const char* port) {
  graph.connect_runtime_port_for_internal_graph_(input, "out", stage, port);
}

void connect_stage_to_output(Graph& graph, std::size_t stage, const char* port,
                             std::size_t output) {
  graph.connect_runtime_port_for_internal_graph_(stage, port, output, "in");
}

} // namespace

Graph VisionLanguage(std::shared_ptr<VisionLanguageModel> model, VisionLanguageOptions options,
                     std::string name) {
  if (!model) {
    throw std::invalid_argument("genai::graphs::VisionLanguage requires a non-null model");
  }

  Graph graph(std::move(name));
  const std::size_t prompt = add_input_endpoint(graph, "prompt");
  const std::size_t image = add_input_endpoint(graph, "image");
  const std::size_t cached = add_input_endpoint(graph, "use_cached_image");
  const std::size_t stage = graph.append_runtime_vertex_for_internal_graph_(
      nodes::VisionLanguage(std::move(model), std::move(options), "vision_language"));
  const std::size_t tokens = add_output_endpoint(graph, "tokens");
  const std::size_t done = add_output_endpoint(graph, "done");
  const std::size_t encoded = add_output_endpoint(graph, "encoded");
  const std::size_t error = add_output_endpoint(graph, "error");

  connect_input_to_stage(graph, prompt, stage, "prompt");
  connect_input_to_stage(graph, image, stage, "image");
  connect_input_to_stage(graph, cached, stage, "use_cached_image");
  connect_stage_to_output(graph, stage, "tokens", tokens);
  connect_stage_to_output(graph, stage, "done", done);
  connect_stage_to_output(graph, stage, "encoded", encoded);
  connect_stage_to_output(graph, stage, "error", error);
  return graph;
}

Graph SpeechTranscriber(std::shared_ptr<ASRModel> model, SpeechTranscriberOptions options,
                        std::string name) {
  if (!model) {
    throw std::invalid_argument("genai::graphs::SpeechTranscriber requires a non-null model");
  }

  Graph graph(std::move(name));
  const std::size_t audio = add_input_endpoint(graph, "audio");
  const std::size_t audio_path = add_input_endpoint(graph, "audio_path");
  const std::size_t stage = graph.append_runtime_vertex_for_internal_graph_(
      nodes::SpeechTranscriber(std::move(model), std::move(options), "speech_transcriber"));
  const std::size_t tokens = add_output_endpoint(graph, "tokens");
  const std::size_t done = add_output_endpoint(graph, "done");
  const std::size_t error = add_output_endpoint(graph, "error");

  connect_input_to_stage(graph, audio, stage, "audio");
  connect_input_to_stage(graph, audio_path, stage, "audio_path");
  connect_stage_to_output(graph, stage, "tokens", tokens);
  connect_stage_to_output(graph, stage, "done", done);
  connect_stage_to_output(graph, stage, "error", error);
  return graph;
}

} // namespace simaai::neat::genai::graphs
