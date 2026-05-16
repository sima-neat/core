#include "genai/nodes/Language.h"

#include "genai/VisionLanguageModel.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/TensorCore.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simaai::neat::genai::nodes {
namespace {

constexpr const char* kPromptPort = "prompt";
constexpr const char* kFormattedPromptPort = "formatted_prompt";
constexpr const char* kTokensPort = "tokens";
constexpr const char* kDonePort = "done";
constexpr const char* kErrorPort = "error";

OutputSpec text_output_spec() {
  OutputSpec spec;
  spec.media_type = "application/vnd.simaai.tensor";
  spec.format = "TEXT";
  spec.dtype = "UInt8";
  spec.layout = "Unknown";
  spec.memory = "SystemMemory";
  spec.certainty = SpecCertainty::Derived;
  spec.note = "UTF-8 text tensor";
  return spec;
}

Sample make_text_sample(std::string text, const char* port_name, const Sample& source) {
  Sample out = make_tensor_sample(port_name, Tensor::from_text(text));
  out.frame_id = source.frame_id;
  out.stream_id = source.stream_id;
  out.pts_ns = source.pts_ns;
  out.duration_ns = source.duration_ns;
  out.port_name = port_name;
  out.stream_label = port_name;
  return out;
}

std::string format_double(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

Sample make_done_sample(const GenerationResult& result, const Sample& source) {
  Sample done;
  done.kind = SampleKind::Bundle;
  done.frame_id = source.frame_id;
  done.stream_id = source.stream_id;
  done.pts_ns = source.pts_ns;
  done.duration_ns = source.duration_ns;
  done.port_name = kDonePort;
  done.stream_label = kDonePort;
  done.fields = {
      make_text_sample(result.text, "text", source),
      make_text_sample(result.finish_reason, "finish_reason", source),
      make_text_sample(std::to_string(result.metrics.generated_tokens), "generated_tokens", source),
      make_text_sample(format_double(result.metrics.time_to_first_token_s), "time_to_first_token_s",
                       source),
      make_text_sample(format_double(result.metrics.tokens_per_second), "tokens_per_second",
                       source),
  };
  return done;
}

const Tensor& require_text_tensor(const Sample& sample) {
  if (sample.kind == SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("GenAI Language graph input is Tensor kind but has no tensor");
    }
    return *sample.tensor;
  }

  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.size() != 1U) {
      throw std::runtime_error("GenAI Language graph input TensorSet must contain exactly one "
                               "text tensor");
    }
    return sample.tensors.front();
  }

  throw std::runtime_error("GenAI Language graph input must be Tensor or TensorSet");
}

class LanguageExecutor final : public graph::StageExecutor {
public:
  LanguageExecutor(std::shared_ptr<VisionLanguageModel> model, LanguageOptions options)
      : model_(std::move(model)), options_(options) {}

  void set_ports(const graph::StagePorts& ports) override {
    prompt_port_ = ports.in_port(kPromptPort);
    formatted_prompt_port_ = ports.in_port(kFormattedPromptPort);
    tokens_port_ = ports.out_port(kTokensPort);
    done_port_ = ports.out_port(kDonePort);
    error_port_ = ports.out_port(kErrorPort);
  }

  void on_input(graph::StageMsg&& msg, std::vector<graph::StageOutMsg>& out) override {
    try {
      GenerationRequest request;
      if (!options_.system_prompt.empty()) {
        request.system_prompt = options_.system_prompt;
      }
      request.max_new_tokens = options_.max_new_tokens;
      request.temperature = options_.temperature;
      request.top_p = options_.top_p;

      const std::string text = require_text_tensor(msg.sample).to_text();
      if (msg.in_port == prompt_port_) {
        request.prompt = text;
      } else if (msg.in_port == formatted_prompt_port_) {
        request.formatted_prompt = text;
      } else {
        throw std::runtime_error("GenAI Language graph input arrived on an unknown port");
      }

      const GenerationResult result = model_->run(request);
      out.push_back(graph::StageOutMsg{
          .out_port = tokens_port_,
          .sample = make_text_sample(result.text, kTokensPort, msg.sample),
      });
      out.push_back(graph::StageOutMsg{
          .out_port = done_port_,
          .sample = make_done_sample(result, msg.sample),
      });
    } catch (const std::exception& e) {
      out.push_back(graph::StageOutMsg{
          .out_port = error_port_,
          .sample = make_text_sample(e.what(), kErrorPort, msg.sample),
      });
    }
  }

private:
  std::shared_ptr<VisionLanguageModel> model_;
  LanguageOptions options_;
  graph::PortId prompt_port_ = graph::kInvalidPort;
  graph::PortId formatted_prompt_port_ = graph::kInvalidPort;
  graph::PortId tokens_port_ = graph::kInvalidPort;
  graph::PortId done_port_ = graph::kInvalidPort;
  graph::PortId error_port_ = graph::kInvalidPort;
};

} // namespace

std::shared_ptr<graph::Node> Language(std::shared_ptr<VisionLanguageModel> model,
                                      LanguageOptions options, std::string label) {
  if (!model) {
    throw std::invalid_argument("genai::nodes::Language requires a non-null model");
  }

  std::vector<graph::PortDesc> inputs = {
      graph::PortDesc{.name = kPromptPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kFormattedPromptPort, .spec = text_output_spec()},
  };
  std::vector<graph::PortDesc> outputs = {
      graph::PortDesc{.name = kTokensPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kDonePort, .spec = text_output_spec()},
      graph::PortDesc{.name = kErrorPort, .spec = text_output_spec()},
  };
  graph::nodes::StageNode::StageExecutorFactory factory = [model = std::move(model), options] {
    return std::make_unique<LanguageExecutor>(model, options);
  };
  return std::make_shared<graph::nodes::StageNode>(
      "GenAILanguage", std::move(factory), std::move(inputs), std::move(outputs), std::move(label));
}

} // namespace simaai::neat::genai::nodes
