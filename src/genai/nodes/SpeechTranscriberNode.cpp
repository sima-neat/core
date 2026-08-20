#include "genai/internal/SpeechTranscriberNodeFactory.h"

#include "genai/ASRModel.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"

#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::genai::nodes {
namespace {

constexpr const char* kAudioPort = "audio";
constexpr const char* kAudioPathPort = "audio_path";
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

OutputSpec audio_output_spec() {
  OutputSpec spec;
  spec.media_type = "application/vnd.simaai.tensor";
  spec.format = "PCM_F32";
  spec.dtype = "Float32";
  spec.layout = "Unknown";
  spec.memory = "SystemMemory";
  spec.certainty = SpecCertainty::Derived;
  spec.note = "float32 mono 16 kHz PCM audio tensor";
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

Sample make_done_sample(const GenerationResult& result, const Sample& source,
                        const std::string& language) {
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
      make_text_sample(result.language.empty() ? language : result.language, "language", source),
  };
  if (result.no_speech_prob.has_value()) {
    done.fields.push_back(
        make_text_sample(format_double(*result.no_speech_prob), "no_speech_prob", source));
  }
  if (result.avg_logprob.has_value()) {
    done.fields.push_back(
        make_text_sample(format_double(*result.avg_logprob), "avg_logprob", source));
  }
  return done;
}

const Tensor& require_single_tensor(const Sample& sample, const char* context) {
  if (sample.kind == SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error(std::string(context) + " is Tensor kind but has no tensor");
    }
    return *sample.tensor;
  }

  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.size() != 1U) {
      throw std::runtime_error(std::string(context) + " TensorSet must contain exactly one tensor");
    }
    return sample.tensors.front();
  }

  throw std::runtime_error(std::string(context) + " must be Tensor or TensorSet");
}

class SpeechTranscriberExecutor final : public graph::StageExecutor {
public:
  SpeechTranscriberExecutor(std::shared_ptr<ASRModel> model, SpeechTranscriberOptions options)
      : model_(std::move(model)), options_(std::move(options)) {}

  void set_ports(const graph::StagePorts& ports) override {
    audio_port_ = ports.in_port(kAudioPort);
    audio_path_port_ = ports.in_port(kAudioPathPort);
    tokens_port_ = ports.out_port(kTokensPort);
    done_port_ = ports.out_port(kDonePort);
    error_port_ = ports.out_port(kErrorPort);
  }

  void set_emitter(graph::StageEmitter* emitter) override {
    emitter_ = emitter;
  }

  void request_stop() override {
    GenerationStream* active = nullptr;
    {
      std::lock_guard<std::mutex> lock(active_stream_mutex_);
      active = active_stream_;
    }
    if (active != nullptr) {
      active->cancel();
    }
  }

  void on_input(graph::StageMsg&& msg, std::vector<graph::StageOutMsg>& out) override {
    try {
      GenerationRequest request;
      request.language = options_.language.empty() ? "auto" : options_.language;
      request.asr_task = options_.task;

      if (msg.in_port == audio_port_) {
        request.audio = require_single_tensor(msg.sample, "GenAI SpeechTranscriber audio input");
      } else if (msg.in_port == audio_path_port_) {
        request.audio_file = std::filesystem::path(
            require_single_tensor(msg.sample, "GenAI SpeechTranscriber audio_path input")
                .to_text());
      } else {
        throw std::runtime_error("GenAI SpeechTranscriber input arrived on an unknown port");
      }

      if (!options_.streaming) {
        GenerationResult result = model_->run(request);
        if (!emit_or_append(tokens_port_, make_text_sample(result.text, kTokensPort, msg.sample),
                            out)) {
          return;
        }
        (void)emit_or_append(done_port_, make_done_sample(result, msg.sample, request.language),
                             out);
        return;
      }

      GenerationStream stream = model_->stream(request);
      struct ActiveStreamGuard {
        SpeechTranscriberExecutor& owner;
        GenerationStream* stream;

        ActiveStreamGuard(SpeechTranscriberExecutor& owner_in, GenerationStream* stream_in)
            : owner(owner_in), stream(stream_in) {
          std::lock_guard<std::mutex> lock(owner.active_stream_mutex_);
          owner.active_stream_ = stream;
        }

        ~ActiveStreamGuard() {
          std::lock_guard<std::mutex> lock(owner.active_stream_mutex_);
          if (owner.active_stream_ == stream) {
            owner.active_stream_ = nullptr;
          }
        }
      } active_stream{*this, &stream};

      std::string transcript;
      GenerationMetrics last_metrics;
      bool saw_final = false;

      while (auto token = stream.next()) {
        last_metrics = token->metrics;
        if (token->is_final) {
          GenerationResult result;
          result.text = transcript;
          result.metrics = token->metrics;
          result.finish_reason = token->finish_reason.empty() ? "stop" : token->finish_reason;
          result.language = token->language;
          result.no_speech_prob = token->no_speech_prob;
          result.avg_logprob = token->avg_logprob;
          (void)emit_or_append(done_port_, make_done_sample(result, msg.sample, request.language),
                               out);
          saw_final = true;
          break;
        }

        if (!token->text.empty()) {
          transcript += token->text;
          if (!emit_or_append(tokens_port_, make_text_sample(token->text, kTokensPort, msg.sample),
                              out)) {
            stream.cancel();
            return;
          }
        }

        if (emitter_ != nullptr && emitter_->stop_requested()) {
          stream.cancel();
          return;
        }
      }

      if (!saw_final && (emitter_ == nullptr || !emitter_->stop_requested())) {
        GenerationResult result;
        result.text = transcript;
        result.metrics = last_metrics;
        result.finish_reason = "interrupted";
        (void)emit_or_append(done_port_, make_done_sample(result, msg.sample, request.language),
                             out);
      }
    } catch (const std::exception& e) {
      (void)emit_or_append(error_port_, make_text_sample(e.what(), kErrorPort, msg.sample), out);
    }
  }

private:
  bool emit_or_append(graph::PortId port, Sample sample, std::vector<graph::StageOutMsg>& out) {
    graph::StageOutMsg msg{.out_port = port, .sample = std::move(sample)};
    if (emitter_ != nullptr) {
      return emitter_->emit(std::move(msg));
    }
    out.push_back(std::move(msg));
    return true;
  }

  std::shared_ptr<ASRModel> model_;
  SpeechTranscriberOptions options_;
  graph::StageEmitter* emitter_ = nullptr;
  std::mutex active_stream_mutex_;
  GenerationStream* active_stream_ = nullptr;
  graph::PortId audio_port_ = graph::kInvalidPort;
  graph::PortId audio_path_port_ = graph::kInvalidPort;
  graph::PortId tokens_port_ = graph::kInvalidPort;
  graph::PortId done_port_ = graph::kInvalidPort;
  graph::PortId error_port_ = graph::kInvalidPort;
};

} // namespace

std::shared_ptr<graph::Node> SpeechTranscriber(std::shared_ptr<ASRModel> model,
                                               SpeechTranscriberOptions options,
                                               std::string label) {
  if (!model) {
    throw std::invalid_argument("genai::nodes::SpeechTranscriber requires a non-null model");
  }

  std::vector<graph::PortDesc> inputs = {
      graph::PortDesc{.name = kAudioPort, .spec = audio_output_spec()},
      graph::PortDesc{.name = kAudioPathPort, .spec = text_output_spec()},
  };
  std::vector<graph::PortDesc> outputs = {
      graph::PortDesc{.name = kTokensPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kDonePort, .spec = text_output_spec()},
      graph::PortDesc{.name = kErrorPort, .spec = text_output_spec()},
  };
  graph::nodes::StageNode::StageExecutorFactory factory = [model = std::move(model), options] {
    return std::make_unique<SpeechTranscriberExecutor>(model, options);
  };
  return std::make_shared<graph::nodes::StageNode>("GenAISpeechTranscriber", std::move(factory),
                                                   std::move(inputs), std::move(outputs),
                                                   std::move(label));
}

} // namespace simaai::neat::genai::nodes
