#include "genai/nodes/VisionLanguage.h"

#include "genai/TensorToVision.h"
#include "genai/VisionLanguageModel.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "pipeline/SessionOptions.h"
#include "pipeline/TensorCore.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat::genai::nodes {
namespace {

constexpr const char* kPromptPort = "prompt";
constexpr const char* kImagePort = "image";
constexpr const char* kUseCachedImagePort = "use_cached_image";
constexpr const char* kTokensPort = "tokens";
constexpr const char* kDonePort = "done";
constexpr const char* kEncodedPort = "encoded";
constexpr const char* kErrorPort = "error";

enum class ImageSource { None, Direct, Cached };

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

OutputSpec image_output_spec() {
  OutputSpec spec;
  spec.media_type = "video/x-raw";
  spec.format = "RGB";
  spec.dtype = "UInt8";
  spec.layout = "HWC";
  spec.memory = "SystemMemory";
  spec.certainty = SpecCertainty::Derived;
  spec.note = "uint8 HWC RGB image tensor";
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
                        std::size_t cached_image_count) {
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
      make_text_sample(std::to_string(cached_image_count), "cached_image_count", source),
  };
  return done;
}

Sample make_encoded_sample(std::size_t image_count, std::size_t cached_image_count,
                           const std::string& mode, const Sample& source) {
  Sample encoded;
  encoded.kind = SampleKind::Bundle;
  encoded.frame_id = source.frame_id;
  encoded.stream_id = source.stream_id;
  encoded.pts_ns = source.pts_ns;
  encoded.duration_ns = source.duration_ns;
  encoded.port_name = kEncodedPort;
  encoded.stream_label = kEncodedPort;
  encoded.fields = {
      make_text_sample(mode, "mode", source),
      make_text_sample(std::to_string(image_count), "image_count", source),
      make_text_sample(std::to_string(cached_image_count), "cached_image_count", source),
  };
  return encoded;
}

const Tensor& require_single_text_tensor(const Sample& sample, const char* context) {
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

std::vector<Tensor> require_image_tensors(const Sample& sample) {
  std::vector<Tensor> inputs;
  if (sample.kind == SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("GenAI VisionLanguage image input is Tensor kind but has no tensor");
    }
    inputs.push_back(*sample.tensor);
  } else if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.empty()) {
      throw std::runtime_error("GenAI VisionLanguage image TensorSet must not be empty");
    }
    inputs = sample.tensors;
  } else {
    throw std::runtime_error("GenAI VisionLanguage image input must be Tensor or TensorSet");
  }

  std::vector<Tensor> out;
  out.reserve(inputs.size());
#if defined(SIMA_WITH_OPENCV)
  for (const Tensor& image : inputs) {
    cv::Mat rgb = internal::tensor_to_rgb_mat(image);
    out.push_back(Tensor::from_cv_mat(rgb, ImageSpec::PixelFormat::RGB, TensorMemory::CPU));
  }
#else
  throw std::runtime_error("GenAI VisionLanguage image graph node requires OpenCV support");
#endif
  return out;
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

bool parse_bool_text(std::string text) {
  text = lowercase_ascii(trim_text(std::move(text)));
  if (text == "true" || text == "1" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "false" || text == "0" || text == "no" || text == "off") {
    return false;
  }
  throw std::runtime_error("GenAI VisionLanguage use_cached_image input must be true/false");
}

class VisionLanguageExecutor final : public graph::StageExecutor {
public:
  VisionLanguageExecutor(std::shared_ptr<VisionLanguageModel> model, VisionLanguageOptions options)
      : model_(std::move(model)), options_(options) {}

  void set_ports(const graph::StagePorts& ports) override {
    prompt_port_ = ports.in_port(kPromptPort);
    image_port_ = ports.in_port(kImagePort);
    use_cached_image_port_ = ports.in_port(kUseCachedImagePort);
    tokens_port_ = ports.out_port(kTokensPort);
    done_port_ = ports.out_port(kDonePort);
    encoded_port_ = ports.out_port(kEncodedPort);
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
      if (msg.in_port == image_port_) {
        handle_image_input(msg, out);
        return;
      }
      if (msg.in_port == use_cached_image_port_) {
        handle_use_cached_input(msg, out);
        return;
      }
      if (msg.in_port == prompt_port_) {
        handle_prompt_input(msg, out);
        return;
      }
      throw std::runtime_error("GenAI VisionLanguage input arrived on an unknown port");
    } catch (const std::exception& e) {
      (void)emit_or_append(error_port_, make_text_sample(e.what(), kErrorPort, msg.sample), out);
    }
  }

private:
  void handle_image_input(const graph::StageMsg& msg, std::vector<graph::StageOutMsg>& out) {
    if (!model_->accepts_image()) {
      throw std::runtime_error("GenAI VisionLanguage image input requires an image-capable model");
    }

    std::vector<Tensor> images = require_image_tensors(msg.sample);
    if (options_.encode_images_on_input) {
      if (!model_->encode(images)) {
        throw std::runtime_error("GenAI VisionLanguage image encode failed");
      }
      latest_images_.clear();
      latest_image_source_ = ImageSource::Cached;
      cached_image_count_ = model_->cached_image_count();
      use_cached_image_ = true;
      (void)emit_or_append(
          encoded_port_,
          make_encoded_sample(images.size(), cached_image_count_, "cached", msg.sample), out);
      return;
    }

    cached_image_count_ = 0;
    latest_images_ = std::move(images);
    latest_image_source_ = ImageSource::Direct;
    use_cached_image_ = false;
    (void)emit_or_append(encoded_port_,
                         make_encoded_sample(latest_images_.size(), 0, "direct", msg.sample), out);
  }

  void handle_use_cached_input(const graph::StageMsg& msg, std::vector<graph::StageOutMsg>&) {
    const std::string text =
        require_single_text_tensor(msg.sample, "GenAI VisionLanguage use_cached_image input")
            .to_text();
    use_cached_image_ = parse_bool_text(text);
    latest_image_source_ = use_cached_image_ ? ImageSource::Cached : ImageSource::Direct;
    cached_image_count_ = model_->cached_image_count();
  }

  void handle_prompt_input(const graph::StageMsg& msg, std::vector<graph::StageOutMsg>& out) {
    GenerationRequest request;
    if (!options_.system_prompt.empty()) {
      request.system_prompt = options_.system_prompt;
    }
    request.max_new_tokens = options_.max_new_tokens;
    request.temperature = options_.temperature;
    request.top_p = options_.top_p;
    request.prompt =
        require_single_text_tensor(msg.sample, "GenAI VisionLanguage prompt input").to_text();

    std::size_t cached_count_for_done = 0;
    if (model_->accepts_image()) {
      if (use_cached_image_ || latest_image_source_ == ImageSource::Cached) {
        const std::size_t count = model_->cached_image_count();
        if (count == 0U) {
          throw std::runtime_error("GenAI VisionLanguage prompt requested cached image but no "
                                   "cached image set is available");
        }
        request.use_cached_images = true;
        cached_count_for_done = count;
      } else if (!latest_images_.empty()) {
        request.images = latest_images_;
      } else {
        throw std::runtime_error("GenAI VisionLanguage prompt requires an image input or cached "
                                 "image set");
      }
    }

    if (!options_.streaming) {
      const GenerationResult result = model_->run(request);
      if (!emit_or_append(tokens_port_, make_text_sample(result.text, kTokensPort, msg.sample),
                          out)) {
        return;
      }
      (void)emit_or_append(done_port_, make_done_sample(result, msg.sample, cached_count_for_done),
                           out);
      return;
    }

    GenerationStream stream = model_->stream(request);
    struct ActiveStreamGuard {
      VisionLanguageExecutor& owner;
      GenerationStream* stream;

      ActiveStreamGuard(VisionLanguageExecutor& owner_in, GenerationStream* stream_in)
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

    std::string generated_text;
    GenerationMetrics last_metrics;
    bool saw_final = false;

    while (auto token = stream.next()) {
      last_metrics = token->metrics;
      if (token->is_final) {
        GenerationResult result;
        result.text = generated_text;
        result.metrics = token->metrics;
        result.finish_reason = token->finish_reason.empty() ? "stop" : token->finish_reason;
        (void)emit_or_append(done_port_,
                             make_done_sample(result, msg.sample, cached_count_for_done), out);
        saw_final = true;
        break;
      }

      if (!token->text.empty()) {
        generated_text += token->text;
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
      result.text = generated_text;
      result.metrics = last_metrics;
      result.finish_reason = "interrupted";
      (void)emit_or_append(done_port_, make_done_sample(result, msg.sample, cached_count_for_done),
                           out);
    }
  }

  bool emit_or_append(graph::PortId port, Sample sample, std::vector<graph::StageOutMsg>& out) {
    graph::StageOutMsg msg{.out_port = port, .sample = std::move(sample)};
    if (emitter_ != nullptr) {
      return emitter_->emit(std::move(msg));
    }
    out.push_back(std::move(msg));
    return true;
  }

  std::shared_ptr<VisionLanguageModel> model_;
  VisionLanguageOptions options_;
  graph::StageEmitter* emitter_ = nullptr;
  std::mutex active_stream_mutex_;
  GenerationStream* active_stream_ = nullptr;
  std::vector<Tensor> latest_images_;
  ImageSource latest_image_source_ = ImageSource::None;
  bool use_cached_image_ = false;
  std::size_t cached_image_count_ = 0;
  graph::PortId prompt_port_ = graph::kInvalidPort;
  graph::PortId image_port_ = graph::kInvalidPort;
  graph::PortId use_cached_image_port_ = graph::kInvalidPort;
  graph::PortId tokens_port_ = graph::kInvalidPort;
  graph::PortId done_port_ = graph::kInvalidPort;
  graph::PortId encoded_port_ = graph::kInvalidPort;
  graph::PortId error_port_ = graph::kInvalidPort;
};

} // namespace

std::shared_ptr<graph::Node> VisionLanguage(std::shared_ptr<VisionLanguageModel> model,
                                            VisionLanguageOptions options, std::string label) {
  if (!model) {
    throw std::invalid_argument("genai::nodes::VisionLanguage requires a non-null model");
  }

  std::vector<graph::PortDesc> inputs = {
      graph::PortDesc{.name = kPromptPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kImagePort, .spec = image_output_spec()},
      graph::PortDesc{.name = kUseCachedImagePort, .spec = text_output_spec()},
  };
  std::vector<graph::PortDesc> outputs = {
      graph::PortDesc{.name = kTokensPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kDonePort, .spec = text_output_spec()},
      graph::PortDesc{.name = kEncodedPort, .spec = text_output_spec()},
      graph::PortDesc{.name = kErrorPort, .spec = text_output_spec()},
  };
  graph::nodes::StageNode::StageExecutorFactory factory = [model = std::move(model), options] {
    return std::make_unique<VisionLanguageExecutor>(model, options);
  };
  return std::make_shared<graph::nodes::StageNode>("GenAIVisionLanguage", std::move(factory),
                                                   std::move(inputs), std::move(outputs),
                                                   std::move(label));
}

} // namespace simaai::neat::genai::nodes
