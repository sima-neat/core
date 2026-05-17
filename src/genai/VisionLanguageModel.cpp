#include "genai/VisionLanguageModel.h"
#include "genai/GenAIInternal.h"
#include "genai/TensorToVision.h"

#include <sima_lmm/chat.hpp>
#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/language_model.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/setup.hpp>
#include <sima_lmm/text_streamer.hpp>
#include <sima_lmm/utils.hpp>
#include <sima_lmm/vlm_config.hpp>
#include <sima_lmm/vlm_helper.hpp>
#include <sima_lmm/vision_model.hpp>

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

namespace simaai::neat::genai {
namespace {

void ensure_llima_runtime_connected() {
  static std::once_flag once;
  std::call_once(once, [] {
    simaai::llima::set_log_level(spdlog::level::warn);
    simaai::llima::connect_mla_rt({});
    simaai::llima::MLAModelWithBuffer::read_env_vars();
    simaai::llima::ImageProcessor::read_env_vars();
  });
}

simaai::llima::VlmConfig load_vlm_config(const std::filesystem::path& model_root) {
  const auto config_path = model_root / "devkit" / "vlm_config.json";
  std::ifstream in(config_path);
  if (!in) {
    throw std::runtime_error("Unable to open LLiMa VLM config: " + config_path.string());
  }
  try {
    return nlohmann::json::parse(in).get<simaai::llima::VlmConfig>();
  } catch (const std::exception& e) {
    throw std::runtime_error("Unable to parse LLiMa VLM config " + config_path.string() + ": " +
                             e.what());
  }
}

std::optional<uint16_t> make_max_total_tokens(std::size_t input_token_count,
                                              std::uint32_t max_new_tokens) {
  if (max_new_tokens == 0) {
    return std::nullopt;
  }
  const auto max_uint16 = static_cast<std::size_t>(std::numeric_limits<uint16_t>::max());
  if (input_token_count > max_uint16 || max_new_tokens > max_uint16 ||
      input_token_count + static_cast<std::size_t>(max_new_tokens) > max_uint16) {
    throw std::runtime_error("GenerationRequest::max_new_tokens exceeds LLiMa token limit");
  }
  return static_cast<uint16_t>(input_token_count + max_new_tokens);
}

std::string token_content_from_json(const nlohmann::json& token_json, const char* token_name) {
  if (token_json.is_null()) {
    return {};
  }
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  }
  if (token_json.contains("content") && token_json.at("content").is_string()) {
    return token_json.at("content").get<std::string>();
  }
  throw std::runtime_error(std::string("Failed to read ") + token_name + " from tokenizer config");
}

std::string load_bos_token(const std::filesystem::path& model_root) {
  const auto config_path = model_root / "devkit" / "tokenizer_config.json";
  std::ifstream in(config_path);
  if (!in) {
    return {};
  }
  try {
    const auto config = nlohmann::json::parse(in);
    if (!config.contains("bos_token")) {
      return {};
    }
    return token_content_from_json(config.at("bos_token"), "bos_token");
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("Unable to parse LLiMa tokenizer config " + config_path.string() +
                             ": " + e.what());
  }
}

std::unique_ptr<simaai::llima::ImageProcessor>
make_image_processor(const simaai::llima::VlmConfig& cfg, const std::filesystem::path& devkit_dir) {
  const auto p1 = devkit_dir / "preprocessor_config.json";
  const auto p2 = devkit_dir / "processor_config.json";
  const auto json_root =
      nlohmann::json::parse(std::ifstream(std::filesystem::exists(p1) ? p1 : p2));
  const auto& json =
      json_root.contains("image_processor") ? json_root["image_processor"] : json_root;

  bool do_pad_to_square = false;
  if (json.contains("do_pad") && json["do_pad"].is_boolean()) {
    do_pad_to_square = json["do_pad"];
  } else if (cfg.model_type.starts_with("vlm-lfm2") || cfg.model_type.starts_with("vlm-qwen")) {
    do_pad_to_square = true;
  }

  const bool do_center_crop = json.contains("do_center_crop") && json["do_center_crop"].is_boolean()
                                  ? json["do_center_crop"].get<bool>()
                                  : false;

  const int resample = json.value("resample", 3);
  cv::InterpolationFlags interpolation = cv::INTER_CUBIC;
  switch (resample) {
  case 2:
    interpolation = cv::INTER_LINEAR;
    break;
  case 3:
    interpolation = cv::INTER_CUBIC;
    break;
  default:
    throw std::runtime_error("Unsupported GenAI image resample type: " + std::to_string(resample));
  }

  return std::make_unique<simaai::llima::ImageProcessor>(
      cfg, do_pad_to_square, do_center_crop, interpolation,
      json.value("rescale_factor", 1.0 / 255.0), json.at("image_mean").get<std::vector<double>>(),
      json.at("image_std").get<std::vector<double>>());
}

const std::string& dummy_image_data_uri() {
  static const std::string uri =
      "data:image/jpeg;base64,"
      "/9j/4AAQSkZJRgABAQAAAQABAAD/"
      "2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
      "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/"
      "wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/"
      "8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoK"
      "So0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKzt"
      "LW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/"
      "8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/"
      "8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJ"
      "ygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqs"
      "rO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q==";
  return uri;
}

void append_image_items(nlohmann::ordered_json& content, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    content.push_back({{"type", "image"}, {"image", dummy_image_data_uri()}});
  }
}

void append_text_item(nlohmann::ordered_json& content, const std::string& text) {
  content.push_back({{"type", "text"}, {"text", text}});
}

struct BuiltMessages {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  std::vector<Tensor> images;
  bool use_cached_images = false;
  std::size_t expected_image_count = 0;
};

BuiltMessages build_llima_messages(const GenerationRequest& request,
                                   std::size_t cached_image_count) {
  BuiltMessages built;

  auto make_content = [](const std::string& text, std::size_t image_count) {
    if (image_count == 0U) {
      return nlohmann::ordered_json(text);
    }
    nlohmann::ordered_json content = nlohmann::ordered_json::array();
    append_image_items(content, image_count);
    append_text_item(content, text);
    return content;
  };

  if (!request.messages.empty()) {
    for (const auto& message : internal::build_text_messages(request)) {
      const std::size_t direct_count = message.images.size();
      const std::size_t cached_count = message.use_cached_images ? cached_image_count : 0U;
      if (message.use_cached_images && cached_image_count == 0U) {
        throw std::runtime_error("GenerationRequest requested cached images but none are cached");
      }
      if (message.use_cached_images) {
        built.use_cached_images = true;
      }
      built.expected_image_count += direct_count + cached_count;
      built.images.insert(built.images.end(), message.images.tensors().begin(),
                          message.images.tensors().end());
      built.messages.push_back(
          {{"role", message.role},
           {"content", make_content(message.content, direct_count + cached_count)}});
    }
    return built;
  }

  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  if (request.system_prompt.has_value()) {
    messages.push_back({{"role", "system"}, {"content", *request.system_prompt}});
  }

  if (request.use_cached_images && cached_image_count == 0U) {
    throw std::runtime_error("GenerationRequest requested cached images but none are cached");
  }
  const std::size_t cached_count = request.use_cached_images ? cached_image_count : 0U;
  const std::size_t direct_count = request.images.size();
  built.expected_image_count = direct_count + cached_count;
  built.use_cached_images = request.use_cached_images;
  built.images = request.images.tensors();
  messages.push_back({{"role", "user"},
                      {"content", make_content(request.prompt.value_or(std::string{}),
                                               built.expected_image_count)}});
  built.messages = std::move(messages);
  return built;
}

std::vector<std::vector<Eigen::bfloat16>>
preprocess_images(simaai::llima::ImageProcessor& image_processor,
                  const std::vector<Tensor>& images) {
  std::vector<std::vector<Eigen::bfloat16>> out;
  out.reserve(images.size());
  for (const auto& image : images) {
    out.push_back(image_processor.preprocess(internal::tensor_to_rgb_mat(image)));
  }
  return out;
}

} // namespace

struct VisionLanguageModel::Impl {
  class ActiveRunGuard {
  public:
    explicit ActiveRunGuard(Impl& owner_in) : owner(&owner_in) {}

    ActiveRunGuard(const ActiveRunGuard&) = delete;
    ActiveRunGuard& operator=(const ActiveRunGuard&) = delete;

    ActiveRunGuard(ActiveRunGuard&& other) noexcept : owner(other.owner) {
      other.owner = nullptr;
    }

    ActiveRunGuard& operator=(ActiveRunGuard&& other) noexcept {
      if (this == &other) {
        return *this;
      }
      release();
      owner = other.owner;
      other.owner = nullptr;
      return *this;
    }

    ~ActiveRunGuard() {
      release();
    }

    static ActiveRunGuard acquire(Impl& owner) {
      std::unique_lock<std::mutex> lock(owner.run_state_mutex);
      owner.run_state_cv.wait(lock, [&] { return !owner.run_active; });
      owner.run_active = true;
      return ActiveRunGuard(owner);
    }

  private:
    void release() {
      if (!owner) {
        return;
      }
      {
        std::lock_guard<std::mutex> lock(owner->run_state_mutex);
        owner->run_active = false;
      }
      owner->run_state_cv.notify_one();
      owner = nullptr;
    }

    Impl* owner = nullptr;
  };

  explicit Impl(std::filesystem::path model_dir_in)
      : info(internal::inspect_model_directory(std::move(model_dir_in))) {
    if (info.task != GenAITask::VisionLanguage) {
      throw std::runtime_error("GenAI model directory is not a vision-language model: " +
                               info.root.string());
    }
  }

  void load() {
    std::lock_guard<std::mutex> lock(load_mutex);
    if (language_model) {
      return;
    }

    ensure_llima_runtime_connected();
    cfg = load_vlm_config(info.root);
    bos_token = load_bos_token(info.root);
    vlm_helper = std::make_unique<simaai::llima::VlmHelper>(cfg, info.root / "devkit", std::nullopt,
                                                            std::nullopt);
    text_streamer = std::make_unique<simaai::llima::TextStreamer>(
        vlm_helper->get_tokenizer(),
        [this](const std::string& metric, double value) { record_metric(metric, value); },
        [](const std::string&, bool) {});
    language_model = std::make_unique<simaai::llima::LanguageModel>(
        info.root, vlm_helper->get_stop_token_ids(), vlm_helper->get_image_token_id(),
        *text_streamer, true);
    if (info.accepts_image) {
      image_processor = make_image_processor(cfg, info.root / "devkit");
      vision_model = std::make_unique<simaai::llima::VisionModel>(info.root);
    }
    configure_run_callbacks();
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_text_generation_request(request);

    auto active_run = ActiveRunGuard::acquire(*this);
    reset_metrics();
    configure_run_callbacks();

    auto output_token_ids = generate_tokens(request);

    GenerationResult result;
    result.metrics = current_metrics();
    if (!output_token_ids.has_value()) {
      result.finish_reason = "interrupted";
      return result;
    }

    result.metrics.generated_tokens = static_cast<std::uint32_t>(output_token_ids->size());
    result.text = vlm_helper->get_tokenizer()->decode(output_token_ids.value(), true);
    result.finish_reason = "stop";
    return result;
  }

  std::optional<std::vector<uint32_t>> generate_tokens(const GenerationRequest& request) {
    simaai::llima::ChronoTimer timer_ttft{true};
    auto prepared = prepare_input(request);
    auto vision_ofm_maps = language_model->create_input_buffers(prepared.input_token_ids);
    fill_vision_buffers(vision_ofm_maps, prepared);

    const auto max_total_tokens =
        make_max_total_tokens(prepared.input_token_ids.size(), request.max_new_tokens);
    return language_model->run_model(prepared.input_token_ids, timer_ttft, max_total_tokens);
  }

  struct PreparedInput {
    std::vector<uint32_t> input_token_ids;
    std::vector<std::vector<Eigen::bfloat16>> image_tensors;
    std::vector<std::vector<Eigen::bfloat16>> cached_image_tensors;
    std::size_t expected_image_count = 0;
  };

  PreparedInput prepare_input(const GenerationRequest& request) {
    const auto cached = cached_images_copy();
    const auto built = build_llima_messages(request, cached.size());
    if (built.expected_image_count > 0U && !info.accepts_image) {
      throw std::runtime_error(
          "GenerationRequest uses images, but this model does not accept images");
    }

    simaai::llima::Chat chat(*vlm_helper);
    chat.set_messages(built.messages);
    auto preprocessed = vlm_helper->preprocess(chat);
    if (preprocessed.image_tensors.size() != built.expected_image_count) {
      throw std::runtime_error("LLiMa chat template image count did not match GenAI request");
    }

    PreparedInput prepared;
    prepared.input_token_ids = std::move(preprocessed.input_token_ids);
    prepared.expected_image_count = built.expected_image_count;
    if (!built.images.empty()) {
      if (!image_processor) {
        throw std::runtime_error("GenAI image processor is unavailable for this model");
      }
      prepared.image_tensors = preprocess_images(*image_processor, built.images);
    }
    if (built.use_cached_images) {
      prepared.cached_image_tensors = cached;
    }
    return prepared;
  }

  void fill_vision_buffers(std::vector<std::map<uint8_t, simaai::llima::MLABufferSlice>>& ofm_maps,
                           const PreparedInput& prepared) {
    if (ofm_maps.size() != prepared.expected_image_count) {
      throw std::runtime_error("LLiMa language model image-token count did not match request");
    }
    if (ofm_maps.empty()) {
      return;
    }
    if (!vision_model) {
      throw std::runtime_error("GenAI vision model is unavailable for this model");
    }

    std::size_t map_idx = 0;
    for (const auto& image_tensor : prepared.image_tensors) {
      vision_model->run_model(image_tensor, &ofm_maps.at(map_idx));
      ++map_idx;
    }
    for (const auto& cached_tensor : prepared.cached_image_tensors) {
      upload_cached_image(cached_tensor, ofm_maps.at(map_idx));
      ++map_idx;
    }
  }

  static void upload_cached_image(const std::vector<Eigen::bfloat16>& cached_tensor,
                                  const std::map<uint8_t, simaai::llima::MLABufferSlice>& ofm_map) {
    if (ofm_map.size() != 1U || !ofm_map.contains(0)) {
      throw std::runtime_error(
          "Cached image reuse is not supported for this model's multi-output vision encoder");
    }

    const auto& slice = ofm_map.at(0);
    auto* buffer = slice.get_buf_ptr();
    if (buffer == nullptr) {
      throw std::runtime_error("Cached image destination buffer is null");
    }
    const auto data_size = buffer->get_buf_len(slice.get_buf_shapes());
    const auto expected_size = cached_tensor.size() * sizeof(Eigen::bfloat16);
    if (data_size != expected_size) {
      throw std::runtime_error("Cached image tensor shape does not match language input buffer");
    }
    buffer->upload(cached_tensor.data(), buffer->get_buf_addr_offset(slice.get_buf_begins()),
                   data_size);
  }

  std::vector<std::vector<Eigen::bfloat16>> cached_images_copy() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return cached_image_tensors;
  }

  bool encode(const std::vector<Tensor>& images) {
    if (!info.accepts_image) {
      throw std::runtime_error("VisionLanguageModel::encode requires a vision-capable model");
    }
    if (images.empty()) {
      throw std::runtime_error("VisionLanguageModel::encode requires at least one image");
    }

    auto active_run = ActiveRunGuard::acquire(*this);
    if (!image_processor || !vision_model) {
      throw std::runtime_error("GenAI vision runtime is unavailable for this model");
    }
    if (!cached_image_reuse_supported()) {
      throw std::runtime_error(
          "VisionLanguageModel::encode cached reuse is not supported for this model's "
          "multi-output vision encoder");
    }

    auto image_tensors = preprocess_images(*image_processor, images);
    std::vector<std::vector<Eigen::bfloat16>> encoded;
    encoded.reserve(image_tensors.size());
    for (const auto& image_tensor : image_tensors) {
      encoded.push_back(vision_model->run_model(image_tensor));
    }
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      cached_image_tensors = std::move(encoded);
    }
    return true;
  }

  bool cached_image_reuse_supported() const {
    return !cfg.vm_cfg.has_value() || cfg.vm_cfg->deepstack_visual_indexes.empty();
  }

  std::size_t cached_image_count() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return cached_image_tensors.size();
  }

  void configure_run_callbacks() {
    text_streamer->set_info_callback(
        [this](const std::string& metric, double value) { record_metric(metric, value); });
    text_streamer->set_text_callback([](const std::string&, bool) {});
  }

  void reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    metrics = {};
  }

  void record_metric(const std::string& metric, double value) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    if (metric == "ttft") {
      metrics.time_to_first_token_s = value;
    } else if (metric == "tps") {
      metrics.tokens_per_second = value;
    }
  }

  GenerationMetrics current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    return metrics;
  }

  internal::ModelDirectoryInfo info;
  simaai::llima::VlmConfig cfg;
  std::string bos_token;
  std::unique_ptr<simaai::llima::VlmHelper> vlm_helper;
  std::unique_ptr<simaai::llima::TextStreamer> text_streamer;
  std::unique_ptr<simaai::llima::LanguageModel> language_model;
  std::unique_ptr<simaai::llima::ImageProcessor> image_processor;
  std::unique_ptr<simaai::llima::VisionModel> vision_model;
  std::mutex load_mutex;
  std::mutex run_state_mutex;
  std::condition_variable run_state_cv;
  bool run_active = false;
  mutable std::mutex cache_mutex;
  std::vector<std::vector<Eigen::bfloat16>> cached_image_tensors;
  mutable std::mutex metrics_mutex;
  GenerationMetrics metrics;
};

struct GenerationStream::Impl {
  Impl(std::shared_ptr<VisionLanguageModel::Impl> model_in, GenerationRequest request_in)
      : model(std::move(model_in)), request(std::move(request_in)) {
    worker = std::thread([this] { run_worker(); });
  }

  ~Impl() {
    cancel();
    if (worker.joinable()) {
      worker.join();
    }
  }

  std::optional<TokenSample> next() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [&] { return !samples.empty() || closed || error != nullptr; });
    if (!samples.empty()) {
      TokenSample sample = std::move(samples.front());
      samples.pop();
      return sample;
    }
    if (error) {
      std::rethrow_exception(error);
    }
    return std::nullopt;
  }

  void cancel() {
    cancelled = true;
    if (model && model->language_model) {
      model->language_model->stop_model();
    }
  }

  void run_worker() {
    try {
      auto active_run = VisionLanguageModel::Impl::ActiveRunGuard::acquire(*model);
      model->reset_metrics();
      model->text_streamer->set_info_callback(
          [this](const std::string& metric, double value) { record_metric(metric, value); });
      model->text_streamer->set_text_callback(
          [this](const std::string& text, bool stream_end) { record_text(text, stream_end); });

      auto output_token_ids = model->generate_tokens(request);
      {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        metrics.generated_tokens =
            output_token_ids.has_value() ? static_cast<std::uint32_t>(output_token_ids->size()) : 0;
        if (finish_reason.empty()) {
          finish_reason = output_token_ids.has_value() ? "stop" : "interrupted";
        }
        if (cancelled && !output_token_ids.has_value()) {
          finish_reason = "interrupted";
        }
      }
      push_final();
      model->configure_run_callbacks();
    } catch (...) {
      if (model) {
        model->configure_run_callbacks();
      }
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        error = std::current_exception();
        closed = true;
      }
      queue_cv.notify_all();
    }
  }

  void record_metric(const std::string& metric, double value) {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    if (metric == "ttft") {
      metrics.time_to_first_token_s = value;
    } else if (metric == "tps") {
      metrics.tokens_per_second = value;
    } else if (metric == "FULL") {
      finish_reason = "cache_full";
    }
  }

  GenerationMetrics current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    return metrics;
  }

  std::string current_finish_reason() const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    return finish_reason;
  }

  void record_text(const std::string& text, bool stream_end) {
    if (!text.empty()) {
      push_sample(TokenSample{.text = text, .metrics = current_metrics()});
    }
    if (stream_end) {
      saw_stream_end = true;
    }
  }

  void push_sample(TokenSample sample) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      samples.push(std::move(sample));
    }
    queue_cv.notify_one();
  }

  void push_final() {
    TokenSample sample;
    sample.metrics = current_metrics();
    sample.is_final = true;
    sample.finish_reason = current_finish_reason();
    if (sample.finish_reason.empty()) {
      sample.finish_reason = saw_stream_end ? "stop" : "interrupted";
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      samples.push(std::move(sample));
      closed = true;
    }
    queue_cv.notify_all();
  }

  std::shared_ptr<VisionLanguageModel::Impl> model;
  GenerationRequest request;
  std::thread worker;
  std::atomic<bool> cancelled = false;
  std::atomic<bool> saw_stream_end = false;

  mutable std::mutex metrics_mutex;
  GenerationMetrics metrics;
  std::string finish_reason;

  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::queue<TokenSample> samples;
  bool closed = false;
  std::exception_ptr error;
};

GenerationStream::GenerationStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GenerationStream::~GenerationStream() = default;

GenerationStream::GenerationStream(GenerationStream&&) noexcept = default;

GenerationStream& GenerationStream::operator=(GenerationStream&&) noexcept = default;

std::optional<TokenSample> GenerationStream::next() {
  if (!impl_) {
    return std::nullopt;
  }
  return impl_->next();
}

void GenerationStream::cancel() {
  if (impl_) {
    impl_->cancel();
  }
}

GenerationStream::iterator GenerationStream::begin() {
  return iterator(this);
}

GenerationStream::iterator GenerationStream::end() {
  return iterator();
}

GenerationStream::iterator::iterator(GenerationStream* stream) : stream_(stream) {
  advance();
}

GenerationStream::iterator::reference GenerationStream::iterator::operator*() const {
  return *current_;
}

GenerationStream::iterator::pointer GenerationStream::iterator::operator->() const {
  return &*current_;
}

GenerationStream::iterator& GenerationStream::iterator::operator++() {
  advance();
  return *this;
}

void GenerationStream::iterator::operator++(int) {
  advance();
}

void GenerationStream::iterator::advance() {
  if (!stream_) {
    current_.reset();
    return;
  }
  current_ = stream_->next();
  if (!current_.has_value()) {
    stream_ = nullptr;
  }
}

bool operator==(const GenerationStream::iterator& lhs, const GenerationStream::iterator& rhs) {
  return lhs.stream_ == rhs.stream_ && lhs.current_.has_value() == rhs.current_.has_value();
}

VisionLanguageModel::VisionLanguageModel(std::filesystem::path model_dir)
    : impl_(std::make_shared<Impl>(std::move(model_dir))) {
  impl_->load();
}

VisionLanguageModel::~VisionLanguageModel() = default;

VisionLanguageModel::VisionLanguageModel(VisionLanguageModel&&) noexcept = default;

VisionLanguageModel& VisionLanguageModel::operator=(VisionLanguageModel&&) noexcept = default;

bool VisionLanguageModel::accepts_image() const {
  return impl_->info.accepts_image;
}

std::string VisionLanguageModel::model_id() const {
  return internal::model_id_from_path(impl_->info.root);
}

std::string VisionLanguageModel::describe() const {
  return "VisionLanguageModel(" + impl_->info.root.string() + ")";
}

std::size_t VisionLanguageModel::cached_image_count() const {
  return impl_->cached_image_count();
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
  return GenerationStream(std::make_unique<GenerationStream::Impl>(impl_, request));
}

} // namespace simaai::neat::genai
