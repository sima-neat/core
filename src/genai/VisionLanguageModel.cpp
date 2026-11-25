#include "genai/VisionLanguageModel.h"
#include "genai/GenAIInternal.h"
#include "genai/internal/ModelOperations.h"

#include <sima_lmm/chat.hpp>
#include <sima_lmm/reasoning_parser.hpp>
#include <sima_lmm/tool_call_parser.hpp>
#include <sima_lmm/vision_language_model.hpp>

#include <atomic>
#include <limits>
#include <mutex>
#include <stop_token>
#include <utility>

namespace simaai::neat::genai {
namespace {

cv::Mat require_genai_rgb_image_tensor(const Tensor& image) {
  if (image.dtype != TensorDType::UInt8) {
    throw std::runtime_error("GenAI image tensor must have UInt8 dtype");
  }
  if (image.layout != TensorLayout::HWC) {
    throw std::runtime_error("GenAI image tensor must use HWC layout");
  }
  if (image.shape.size() != 3U || image.shape[2] != 3) {
    throw std::runtime_error("GenAI image tensor must have shape [H, W, 3]");
  }
  if (!image.semantic.image.has_value() ||
      image.semantic.image->format != ImageSpec::PixelFormat::RGB) {
    throw std::runtime_error("GenAI image tensor must declare RGB image semantics");
  }
  return image.to_cv_mat_copy(ImageSpec::PixelFormat::RGB);
}

void validate_lora_adapter_name(const std::string& adapter_name) {
  if (adapter_name.empty() || adapter_name == "." || adapter_name == ".." ||
      std::filesystem::path(adapter_name).is_absolute() ||
      adapter_name.find('/') != std::string::npos || adapter_name.find('\\') != std::string::npos ||
      adapter_name.find('\0') != std::string::npos) {
    throw std::invalid_argument("LoRA adapter name must be one non-empty directory name");
  }
}

std::vector<std::string> tool_names_from_definitions(const Json& tools) {
  std::vector<std::string> names;
  if (!tools.is_array()) {
    return names;
  }

  for (const auto& tool : tools) {
    if (tool.is_object() && tool.contains("function") && tool.at("function").is_object() &&
        tool.at("function").contains("name") && tool.at("function").at("name").is_string()) {
      names.push_back(tool.at("function").at("name").get<std::string>());
    }
  }
  return names;
}

Json make_llima_message(const ChatMessage& message, const Json& content) {
  Json out;
  out["role"] = message.role;
  if (message.role == "assistant" && !message.tool_calls.empty()) {
    out["content"] = message.content.empty() ? Json(nullptr) : Json(message.content);
    out["tool_calls"] = message.tool_calls;
  } else {
    out["content"] = content;
  }
  if (message.tool_call_id.has_value()) {
    out["tool_call_id"] = *message.tool_call_id;
  }
  if (message.name.has_value()) {
    out["name"] = *message.name;
  }
  return out;
}

// The chat template needs image placeholders. Pixels and encoded images travel separately.
Json message_content(const ChatMessage& message, std::size_t image_count) {
  if (!image_count)
    return message.content;
  auto content = Json::array();
  for (std::size_t i = 0; i < image_count; ++i)
    content.push_back({{"type", "image"}, {"image", ""}});
  content.push_back({{"type", "text"}, {"text", message.content}});
  return content;
}

std::string finish_reason(llima::ChatResult::Finish finish, bool tool_calls) {
  if (finish == llima::ChatResult::Finish::Cancelled)
    return "interrupted";
  if (finish == llima::ChatResult::Finish::Length)
    return "length";
  return tool_calls ? "tool_calls" : "stop";
}

} // namespace

struct VisionLanguageModel::Impl {
  explicit Impl(internal::ModelLoadContext context) : info(std::move(context.info)) {
    if (info.task != GenAITask::VisionLanguage)
      throw std::runtime_error("GenAI model directory is not a vision-language model: " +
                               info.root.string());

    internal::ensure_llima_runtime_connected();
    runtime = std::make_unique<llima::VisionLanguageModel>(
        info.root, std::nullopt, std::nullopt, std::move(context.files),
        llima::ModelOptions{context.options.max_kv_cache_slots});
    if (info.draft_root) {
      llima::VisionLanguageModel draft(*info.draft_root, std::nullopt, std::nullopt,
                                       std::move(context.draft_files),
                                       llima::ModelOptions{context.options.max_kv_cache_slots});
      runtime->set_draft_vlm(&draft);
    }
    tool_call_format = runtime->tool_call_format();
    reasoning_format = llima::reasoning_format_for_model(runtime->model_type());
  }

  void validate_thinking_request(const GenerationRequest& request) const {
    if (request.enable_thinking && reasoning_format == llima::ReasoningFormat::None)
      throw std::invalid_argument("Thinking is not supported for this model");
  }

  bool prompt_opens_reasoning(const GenerationRequest& request) const {
    return reasoning_format == llima::ReasoningFormat::Gemma4 && !request.messages.empty() &&
           request.messages.back().role == "tool";
  }

  llima::ChatResult generate(const GenerationRequest& request, llima::GenerationOptions options) {
    if (!operations.lock(options.stop)) {
      llima::ChatResult cancelled;
      cancelled.finish = llima::ChatResult::Finish::Cancelled;
      return cancelled;
    }
    std::lock_guard lock(operations, std::adopt_lock);
    if (request.max_new_tokens > std::numeric_limits<uint16_t>::max())
      throw std::runtime_error("GenerationRequest::max_new_tokens exceeds LLiMa token limit");
    if (request.max_new_tokens)
      options.max_new_tokens = static_cast<uint16_t>(request.max_new_tokens);
    options.performance_summary = false;
    options.reject_oversized_prompt = true;
    options.cache_id = request.cache_id;

    auto chat = runtime->create_chat();
    chat.set_enable_thinking(request.enable_thinking);
    chat.set_reasoning_effort(request.reasoning_effort);
    if (internal::tool_calls_enabled(request))
      chat.set_tools(request.tools);

    auto messages = Json::array();
    std::vector<cv::Mat> images;
    std::optional<std::size_t> cached_at;
    for (const auto& message : internal::build_text_messages(request)) {
      if ((!message.images.empty() || message.use_cached_images) && !info.accepts_image)
        throw std::runtime_error(
            "GenerationRequest uses images, but this model does not accept images");
      if (message.use_cached_images) {
        if (cached_images.empty())
          throw std::runtime_error("GenerationRequest requested cached images but none are cached");
        cached_at = images.size();
      }
      for (const auto& tensor : message.images.tensors())
        images.push_back(require_genai_rgb_image_tensor(tensor));
      const auto count = message.use_cached_images ? cached_images.size() : message.images.size();
      messages.push_back(make_llima_message(message, message_content(message, count)));
    }
    chat.set_messages(std::move(messages));
    options.images = images;

    // LLiMa accepts raw placeholders first, followed by encoded placeholders. Encode any
    // raw images after the cached turn so the original conversation order is preserved.
    std::vector<llima::EncodedImage> ordered;
    if (cached_at) {
      options.encoded_images = cached_images;
      if (*cached_at < images.size()) {
        auto suffix = runtime->encode_images(options.images.subspan(*cached_at));
        ordered = cached_images;
        ordered.insert(ordered.end(), suffix.begin(), suffix.end());
        options.images = options.images.first(*cached_at);
        options.encoded_images = ordered;
      }
    }
    try {
      return runtime->generate(chat, options);
    } catch (const llima::KVCacheCapacityError& e) {
      throw KVCacheCapacityError(e.what());
    }
  }

  GenerationResult run(const GenerationRequest& request) {
    validate_thinking_request(request);
    GenerationResult result;
    llima::GenerationOptions options;
    options.callbacks.info = [&](const std::string& metric, double value) {
      if (metric == "ttft")
        result.metrics.time_to_first_token_s = value;
      if (metric == "tps")
        result.metrics.tokens_per_second = value;
    };
    options.callbacks.text = [](const std::string&, bool, bool) {};
    const auto output = generate(request, std::move(options));
    result.metrics.generated_tokens = output.performance.generated_tokens;
    result.metrics.cached_prompt_tokens = output.performance.cached_prompt_tokens;
    result.metrics.cache_created = output.performance.cache_created;
    simaai::llima::ReasoningStreamParser reasoning_parser(reasoning_format, request.enable_thinking,
                                                          prompt_opens_reasoning(request));
    for (auto& event : reasoning_parser.add(output.text, true)) {
      if (event.reasoning) {
        result.reasoning += event.text;
      } else {
        result.text += event.text;
      }
    }
    if (internal::tool_calls_enabled(request)) {
      auto tool_calls = simaai::llima::try_parse_tool_calls(
          tool_call_format, result.text, tool_names_from_definitions(request.tools));
      if (!tool_calls.is_null()) {
        result.tool_calls = std::move(tool_calls);
        result.text.clear();
      }
    }
    result.finish_reason = finish_reason(output.finish, !result.tool_calls.empty());
    return result;
  }

  bool encode(const std::vector<Tensor>& images) {
    if (!info.accepts_image)
      throw std::runtime_error("VisionLanguageModel::encode requires a vision-capable model");
    if (images.empty())
      throw std::runtime_error("VisionLanguageModel::encode requires at least one image");
    std::vector<cv::Mat> pixels;
    pixels.reserve(images.size());
    for (const auto& image : images)
      pixels.push_back(require_genai_rgb_image_tensor(image));
    std::lock_guard lock(operations);
    cached_images = runtime->encode_images(pixels);
    cached_count.store(cached_images.size());
    return true;
  }

  internal::ModelDirectoryInfo info;
  std::unique_ptr<llima::VisionLanguageModel> runtime;
  llima::ToolCallFormat tool_call_format;
  llima::ReasoningFormat reasoning_format;
  // Adapter changes must wait for an in-flight request. Callbacks only touch request-local state.
  internal::ModelOperations operations;
  std::vector<llima::EncodedImage> cached_images;
  std::atomic<std::size_t> cached_count{0};
};

VisionLanguageModel::VisionLanguageModel(std::filesystem::path model_dir)
    : VisionLanguageModel(internal::local_model_context(model_dir)) {}

VisionLanguageModel::VisionLanguageModel(std::filesystem::path model_dir, GenAIModelOptions options)
    : VisionLanguageModel([&] {
        auto context = internal::local_model_context(model_dir);
        context.options = options;
        return context;
      }()) {}

VisionLanguageModel internal::ModelAccess::vision(ModelLoadContext context) {
  return VisionLanguageModel(std::move(context));
}

VisionLanguageModel::VisionLanguageModel(internal::ModelLoadContext context)
    : impl_(std::make_shared<Impl>(std::move(context))) {}

VisionLanguageModel::~VisionLanguageModel() = default;

VisionLanguageModel::VisionLanguageModel(VisionLanguageModel&&) noexcept = default;

VisionLanguageModel& VisionLanguageModel::operator=(VisionLanguageModel&&) noexcept = default;

bool VisionLanguageModel::accepts_image() const {
  return impl_->info.accepts_image;
}

bool VisionLanguageModel::supports_thinking() const {
  return impl_->reasoning_format != simaai::llima::ReasoningFormat::None;
}

std::string VisionLanguageModel::model_id() const {
  return internal::model_id_from_path(impl_->info.package_root);
}

void VisionLanguageModel::set_lora(const std::string& adapter_name) {
  validate_lora_adapter_name(adapter_name);
  if (impl_->info.draft_root) {
    throw std::invalid_argument("Dynamic LoRA is not supported with speculative decoding");
  }
  std::lock_guard lock(impl_->operations);
  impl_->runtime->set_reloc(adapter_name);
}

void VisionLanguageModel::unset_lora() {
  if (impl_->info.draft_root) {
    throw std::invalid_argument("Dynamic LoRA is not supported with speculative decoding");
  }
  std::lock_guard lock(impl_->operations);
  impl_->runtime->unset_reloc();
}

std::size_t VisionLanguageModel::kv_cache_count() const {
  std::unique_lock lock(impl_->operations, std::try_to_lock);
  if (!lock.owns_lock())
    throw std::logic_error("KV cache control requires an idle model");
  return impl_->runtime->kv_cache_count();
}

bool VisionLanguageModel::remove_kv_cache(const std::string& cache_id) {
  std::unique_lock lock(impl_->operations, std::try_to_lock);
  if (!lock.owns_lock())
    throw std::logic_error("KV cache control requires an idle model");
  return impl_->runtime->remove_kv_cache(cache_id);
}

void VisionLanguageModel::clear_kv_caches() {
  std::unique_lock lock(impl_->operations, std::try_to_lock);
  if (!lock.owns_lock())
    throw std::logic_error("KV cache control requires an idle model");
  impl_->runtime->clear_kv_caches();
}

std::size_t VisionLanguageModel::kv_cache_bytes_per_slot() const {
  return impl_->runtime->kv_cache_bytes_per_slot();
}

std::size_t VisionLanguageModel::cached_image_count() const {
  return impl_->cached_count.load();
}

bool VisionLanguageModel::encode(const Tensor& image) {
  return impl_->encode(std::vector<Tensor>{image});
}

bool VisionLanguageModel::encode(const std::vector<Tensor>& images) {
  return impl_->encode(images);
}

#if defined(SIMA_WITH_OPENCV)
bool VisionLanguageModel::encode(const cv::Mat& image) {
  return encode(std::vector<cv::Mat>{image});
}

bool VisionLanguageModel::encode(const std::vector<cv::Mat>& images) {
  const ImageList image_list(images);
  return impl_->encode(image_list.tensors());
}
#endif

GenerationResult VisionLanguageModel::run(const GenerationRequest& request) {
  return impl_->run(request);
}

GenerationStream VisionLanguageModel::stream(const GenerationRequest& request) {
  internal::validate_text_generation_request(request);
  impl_->validate_thinking_request(request);
  std::stop_source stop;
  return GenerationStream(
      [model = impl_, request, token = stop.get_token()](GenerationStream::Producer& producer) {
        llima::GenerationOptions options;
        options.stop = token;
        options.callbacks.info = [&producer](const std::string& metric, double value) {
          if (metric == "ttft" || metric == "tps" || metric == "cache_created" ||
              metric == "cached_prompt_tokens")
            producer.record_metric(metric, value);
        };
        const bool parse_tools = internal::tool_calls_enabled(request);
        simaai::llima::ToolCallStreamParser tool_parser(model->tool_call_format,
                                                        tool_names_from_definitions(request.tools));
        simaai::llima::ReasoningStreamParser reasoning_parser(
            model->reasoning_format, request.enable_thinking,
            model->prompt_opens_reasoning(request));
        bool emitted_tool_calls = false;
        auto handle_tool_parser_events =
            [&producer,
             &emitted_tool_calls](std::vector<simaai::llima::ToolCallStreamParser::Event> events) {
              for (auto& event : events) {
                TokenSample sample;
                sample.metrics = producer.current_metrics();
                if (std::holds_alternative<simaai::llima::ToolCallStreamParser::Content>(event)) {
                  sample.text =
                      std::move(std::get<simaai::llima::ToolCallStreamParser::Content>(event).text);
                } else {
                  sample.tool_calls = std::move(
                      std::get<simaai::llima::ToolCallStreamParser::ToolCalls>(event).calls);
                  emitted_tool_calls = true;
                }
                producer.push(std::move(sample));
              }
            };
        options.callbacks.text = [&producer, parse_tools, &reasoning_parser, &tool_parser,
                                  &handle_tool_parser_events](const std::string& text,
                                                              bool stream_end, bool) {
          for (auto& event : reasoning_parser.add(text, stream_end)) {
            if (event.reasoning) {
              TokenSample sample;
              sample.reasoning = std::move(event.text);
              sample.metrics = producer.current_metrics();
              producer.push(std::move(sample));
            } else if (parse_tools) {
              handle_tool_parser_events(tool_parser.add(event.text, false));
            } else {
              producer.record_text(event.text, false);
            }
          }
          if (stream_end) {
            if (parse_tools) {
              handle_tool_parser_events(tool_parser.add("", true));
            }
            producer.record_text("", true);
          }
        };
        const auto output = model->generate(request, std::move(options));
        producer.finish(finish_reason(output.finish, emitted_tool_calls),
                        output.performance.generated_tokens);
      },
      [stop] { stop.request_stop(); });
}

} // namespace simaai::neat::genai
