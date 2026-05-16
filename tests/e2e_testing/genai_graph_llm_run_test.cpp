#include "genai/VisionLanguageModel.h"
#include "genai/nodes/Language.h"
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
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kRepoId = "simaai/LFM2-350M-a16w4";
constexpr const char* kModelName = "LFM2-350M-a16w4";

std::string shell_quote(const fs::path& path) {
  std::string in = path.string();
  std::string out = "'";
  for (char c : in) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool command_exists(const char* command) {
  std::string cmd = "command -v ";
  cmd += command;
  cmd += " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

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

class AutoModelDir {
public:
  AutoModelDir() = default;

  explicit AutoModelDir(fs::path path) : path_(std::move(path)), owned_(true) {}

  AutoModelDir(const AutoModelDir&) = delete;
  AutoModelDir& operator=(const AutoModelDir&) = delete;

  AutoModelDir(AutoModelDir&& other) noexcept
      : path_(std::move(other.path_)), owned_(other.owned_) {
    other.owned_ = false;
  }

  AutoModelDir& operator=(AutoModelDir&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    cleanup();
    path_ = std::move(other.path_);
    owned_ = other.owned_;
    other.owned_ = false;
    return *this;
  }

  ~AutoModelDir() {
    cleanup();
  }

  const fs::path& path() const {
    return path_;
  }

private:
  void cleanup() {
    if (!owned_ || path_.empty()) {
      return;
    }
    std::error_code ec;
    fs::remove_all(path_, ec);
    owned_ = false;
  }

  fs::path path_;
  bool owned_ = false;
};

AutoModelDir download_model_to_nvme() {
  if (!command_exists("hf")) {
    skip_long_test_exception("missing Hugging Face CLI command 'hf'; set " +
                             std::string(kModelEnv) + " to an existing model directory");
  }

  const fs::path root =
      fs::path("/media/nvme/tmp") /
      ("neat-genai-graph-e2e-" + std::to_string(static_cast<long long>(::getpid())));
  const fs::path model_dir = root / kModelName;

  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    skip_long_test_exception("failed to create temporary model directory " + root.string() + ": " +
                             ec.message());
  }

  std::ostringstream cmd;
  cmd << "hf download " << kRepoId << " --local-dir " << shell_quote(model_dir);
  const int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    skip_long_test_exception("failed to download " + std::string(kRepoId) + " with hf");
  }

  if (!has_llima_vlm_config(model_dir)) {
    throw std::runtime_error("downloaded model is missing devkit/vlm_config.json: " +
                             model_dir.string());
  }

  return AutoModelDir(root);
}

fs::path resolve_model_dir(AutoModelDir& auto_dir) {
  const std::string env_model_dir = trim_env_value(std::getenv(kModelEnv));
  if (!env_model_dir.empty()) {
    fs::path model_dir(env_model_dir);
    if (has_llima_vlm_config(model_dir)) {
      return model_dir;
    }

    std::cout << "[WARN] " << kModelEnv
              << " does not point to a LLiMa VLM model directory, falling back to temporary "
                 "download: "
              << model_dir << "\n";
  }

  auto_dir = download_model_to_nvme();
  return auto_dir.path() / kModelName;
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
  bool saw_done = false;
  bool saw_error = false;
};

GraphOutputs pull_graph_outputs(simaai::neat::graph::GraphRun& run,
                                simaai::neat::graph::NodeId node_id, bool stop_on_error = false) {
  GraphOutputs outputs;
  for (int i = 0; i < 8; ++i) {
    auto sample = run.pull(node_id, 60000);
    require(sample.has_value(), "GraphRun::pull timed out");
    if (sample->port_name == "tokens" || sample->stream_label == "tokens") {
      outputs.tokens += sample_text(*sample);
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
    AutoModelDir auto_dir;
    const fs::path model_dir = resolve_model_dir(auto_dir);

    std::cout << "GENAI_GRAPH_LLM model_dir=" << model_dir << "\n";

    auto model = std::make_shared<simaai::neat::genai::VisionLanguageModel>(model_dir);
    require(!model->accepts_image(), "Text-only LLiMa model should not accept image input");

    simaai::neat::graph::Graph graph;
    const auto prompt_port = graph.intern_port("prompt");
    const auto formatted_prompt_port = graph.intern_port("formatted_prompt");
    const auto language =
        graph.add(simaai::neat::genai::nodes::Language(model,
                                                       simaai::neat::genai::nodes::LanguageOptions{
                                                           .system_prompt = "You are concise.",
                                                           .max_new_tokens = 24,
                                                       },
                                                       "language"));

    simaai::neat::graph::GraphRun run = simaai::neat::graph::GraphSession(std::move(graph)).build();

    require(run.push(language, prompt_port, make_text_input("What is the capital of Germany?", 1)),
            "GraphRun::push prompt failed");
    const GraphOutputs prompt_outputs = pull_graph_outputs(run, language);
    require(prompt_outputs.saw_done, "GenAI graph prompt run did not emit done");
    require(!prompt_outputs.saw_error,
            "GenAI graph prompt run emitted error: " + prompt_outputs.error);
    require(trim_text(prompt_outputs.tokens) == "The capital of Germany is Berlin.",
            "GenAI graph generated unexpected text: " + prompt_outputs.tokens);
    const std::string finish_reason = bundle_field_text(prompt_outputs.done, "finish_reason");
    require(finish_reason == "stop" || finish_reason == "interrupted",
            "GenAI graph returned unexpected finish reason: " + finish_reason);
    require(std::stoul(bundle_field_text(prompt_outputs.done, "generated_tokens")) > 0,
            "GenAI graph done should report generated tokens");

    require(
        run.push(language, formatted_prompt_port, make_text_input("The capital of Germany is", 2)),
        "GraphRun::push formatted_prompt failed");
    const GraphOutputs formatted_outputs = pull_graph_outputs(run, language);
    require(formatted_outputs.saw_done, "GenAI graph formatted_prompt run did not emit done");
    require(!formatted_outputs.saw_error,
            "GenAI graph formatted_prompt run emitted error: " + formatted_outputs.error);
    require(!trim_text(formatted_outputs.tokens).empty(),
            "GenAI graph formatted_prompt should generate non-empty text");

    require(run.push(language, prompt_port, make_non_text_input()),
            "GraphRun::push invalid prompt failed");
    const GraphOutputs error_outputs = pull_graph_outputs(run, language, true);
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
