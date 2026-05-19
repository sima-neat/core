#include "genai/VisionLanguageModel.h"
#include "genai/nodes/VisionLanguage.h"
#include "graph/Graph.h"
#include "graph/GraphRun.h"
#include "graph/GraphSession.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/TensorCore.h"
#include "test_utils.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

// Exercises the GenAI Language graph node end-to-end against a real LLiMa text
// model, including prompt, done metadata, and error output.
// Model fixture:
//   hf download simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4 --local-dir <model-dir>
//   export SIMA_TEST_LLIMA_TEXT_MODEL=<model-dir>
namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";

bool has_llima_vlm_config(const fs::path& model_dir) {
  std::error_code ec;
  return fs::is_regular_file(model_dir / "devkit" / "vlm_config.json", ec) && !ec;
}

std::string trim_env_value(const char* value) {
  if (value == nullptr) {
    return {};
  }

  std::string out(value);
  const auto first = out.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = out.find_last_not_of(" \t\r\n");
  return out.substr(first, last - first + 1);
}

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

fs::path resolve_model_dir() {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (env_model_dir.empty()) {
    skip_long_test_exception("set " + std::string(kModelEnv) +
                             " to an existing LLiMa text model directory");
  }

  fs::path model_dir(env_model_dir);
  if (has_llima_vlm_config(model_dir)) {
    return model_dir;
  }

  skip_long_test_exception(
      std::string(kModelEnv) +
      " does not point to a LLiMa text model directory: " + model_dir.string());
  return {};
}

simaai::neat::Sample make_text_input(std::string text, int64_t frame_id = 1) {
  simaai::neat::Sample sample =
      simaai::neat::make_tensor_sample("prompt", simaai::neat::Tensor::from_text(text));
  sample.frame_id = frame_id;
  sample.stream_id = "genai-graph";
  sample.pts_ns = frame_id * 1000;
  sample.duration_ns = 1000;
  return sample;
}

simaai::neat::Sample make_non_text_input() {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = simaai::neat::Tensor::from_vector(std::vector<uint8_t>{'x'}, {1},
                                                    simaai::neat::TensorMemory::CPU);
  sample.frame_id = 99;
  sample.stream_id = "genai-graph";
  return sample;
}

std::string sample_text(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    require(sample.tensor.has_value(), "Tensor sample missing tensor");
    return sample.tensor->to_text();
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    require(sample.tensors.size() == 1U, "TensorSet sample should carry one tensor");
    return sample.tensors.front().to_text();
  }
  throw std::runtime_error("sample is not text");
}

std::string bundle_field_text(const simaai::neat::Sample& bundle, const std::string& name) {
  require(bundle.kind == simaai::neat::SampleKind::Bundle, "done sample should be bundle");
  for (const auto& field : bundle.fields) {
    if (field.port_name == name || field.stream_label == name) {
      return sample_text(field);
    }
  }
  throw std::runtime_error("missing done bundle field: " + name);
}

struct GraphOutputs {
  std::string tokens;
  simaai::neat::Sample done;
  std::string error;
  int token_samples = 0;
  bool saw_done = false;
  bool saw_error = false;
};

GraphOutputs pull_graph_outputs(simaai::neat::graph::GraphRun& run,
                                simaai::neat::graph::NodeId node_id, bool stop_on_error = false) {
  GraphOutputs outputs;
  for (int i = 0; i < 64; ++i) {
    auto sample = run.pull(node_id, 60000);
    require(sample.has_value(), "GraphRun::pull timed out");
    if (sample->port_name == "tokens" || sample->stream_label == "tokens") {
      outputs.tokens += sample_text(*sample);
      outputs.token_samples += 1;
    } else if (sample->port_name == "done" || sample->stream_label == "done") {
      outputs.done = *sample;
      outputs.saw_done = true;
      break;
    } else if (sample->port_name == "error" || sample->stream_label == "error") {
      outputs.error = sample_text(*sample);
      outputs.saw_error = true;
      if (stop_on_error) {
        break;
      }
    } else {
      throw std::runtime_error("unexpected graph output port: " + sample->port_name);
    }
  }
  return outputs;
}

} // namespace

int main() {
  try {
    const fs::path model_dir = resolve_model_dir();

    std::cout << "GENAI_GRAPH_LLM model_dir=" << model_dir << "\n";

    auto model = std::make_shared<simaai::neat::genai::VisionLanguageModel>(model_dir);
    require(!model->accepts_image(), "Text-only LLiMa model should not accept image input");

    simaai::neat::graph::Graph graph;
    const auto prompt_port = graph.intern_port("prompt");
    const auto streaming_language = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .system_prompt = "You are concise.",
            .max_new_tokens = 24,
            .streaming = true,
        },
        "vision_language_streaming"));
    const auto sync_language = graph.add(simaai::neat::genai::nodes::VisionLanguage(
        model,
        simaai::neat::genai::nodes::VisionLanguageOptions{
            .system_prompt = "You are concise.",
            .max_new_tokens = 24,
            .streaming = false,
        },
        "vision_language_sync"));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(streaming_language, prompt_port,
                     make_text_input("What is the capital of Germany?", 1)),
            "GraphRun::push streaming prompt failed");
    const GraphOutputs streaming_prompt_outputs = pull_graph_outputs(run, streaming_language);
    require(streaming_prompt_outputs.saw_done, "GenAI graph streaming prompt did not emit done");
    require(!streaming_prompt_outputs.saw_error,
            "GenAI graph streaming prompt emitted error: " + streaming_prompt_outputs.error);
    require(streaming_prompt_outputs.token_samples > 0,
            "GenAI graph streaming prompt should emit token samples");
    std::cout << "GENAI_GRAPH_LLM_STREAM text=" << streaming_prompt_outputs.tokens << "\n";
    require(trim_text(streaming_prompt_outputs.tokens) == "The capital of Germany is Berlin.",
            "GenAI graph streaming generated unexpected text: " + streaming_prompt_outputs.tokens);
    const std::string streaming_finish_reason =
        bundle_field_text(streaming_prompt_outputs.done, "finish_reason");
    require(streaming_finish_reason == "stop" || streaming_finish_reason == "interrupted",
            "GenAI graph streaming returned unexpected finish reason: " + streaming_finish_reason);
    require(std::stoul(bundle_field_text(streaming_prompt_outputs.done, "generated_tokens")) > 0,
            "GenAI graph streaming done should report generated tokens");

    require(
        run.push(sync_language, prompt_port, make_text_input("What is the capital of Germany?", 2)),
        "GraphRun::push sync prompt failed");
    const GraphOutputs sync_prompt_outputs = pull_graph_outputs(run, sync_language);
    require(sync_prompt_outputs.saw_done, "GenAI graph sync prompt did not emit done");
    require(!sync_prompt_outputs.saw_error,
            "GenAI graph sync prompt emitted error: " + sync_prompt_outputs.error);
    require(sync_prompt_outputs.token_samples == 1,
            "GenAI graph sync prompt should emit exactly one token sample");
    std::cout << "GENAI_GRAPH_LLM_SYNC text=" << sync_prompt_outputs.tokens << "\n";
    require(trim_text(sync_prompt_outputs.tokens) == "The capital of Germany is Berlin.",
            "GenAI graph sync generated unexpected text: " + sync_prompt_outputs.tokens);
    const std::string finish_reason = bundle_field_text(sync_prompt_outputs.done, "finish_reason");
    require(finish_reason == "stop" || finish_reason == "interrupted",
            "GenAI graph returned unexpected finish reason: " + finish_reason);
    require(std::stoul(bundle_field_text(sync_prompt_outputs.done, "generated_tokens")) > 0,
            "GenAI graph done should report generated tokens");

    require(run.push(streaming_language, prompt_port, make_non_text_input()),
            "GraphRun::push invalid prompt failed");
    const GraphOutputs error_outputs = pull_graph_outputs(run, streaming_language, true);
    require(error_outputs.saw_error, "GenAI graph invalid prompt should emit error");
    require(!error_outputs.error.empty(), "GenAI graph error text should be non-empty");

    run.stop();
    std::cout << "[OK] genai_graph_llm_run_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
