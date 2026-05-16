#include "genai/VisionLanguageModel.h"
#include "genai/GenAIInternal.h"

#include <sima_lmm/nlohmann_optional.hpp>

#include <sima_lmm/chat.hpp>
#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/language_model.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/setup.hpp>
#include <sima_lmm/text_streamer.hpp>
#include <sima_lmm/utils.hpp>
#include <sima_lmm/vlm_config.hpp>
#include <sima_lmm/vlm_helper.hpp>

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
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

nlohmann::ordered_json build_llima_messages(const GenerationRequest& request) {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  for (const auto& message : internal::build_text_messages(request)) {
    messages.push_back({{"role", message.role}, {"content", message.content}});
  }
  return messages;
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

    ~ActiveRunGuard() { release(); }

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
    vlm_helper = std::make_unique<simaai::llima::VlmHelper>(
        cfg, info.root / "devkit", std::nullopt, std::nullopt);
    text_streamer = std::make_unique<simaai::llima::TextStreamer>(
        vlm_helper->get_tokenizer(),
        [this](const std::string& metric, double value) { record_metric(metric, value); },
        [](const std::string&, bool) {});
    language_model = std::make_unique<simaai::llima::LanguageModel>(
        info.root, vlm_helper->get_stop_token_ids(), vlm_helper->get_image_token_id(),
        *text_streamer, true);
    configure_run_callbacks();
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_text_generation_request(request);
    if (!request.formatted_prompt.empty()) {
      throw std::logic_error("GenerationRequest::formatted_prompt is not implemented yet");
    }

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
    simaai::llima::Chat chat(*vlm_helper);
    chat.set_messages(build_llima_messages(request));
    auto preprocessed = vlm_helper->preprocess(chat);
    language_model->create_input_buffers(preprocessed.input_token_ids);

    const auto max_total_tokens =
        make_max_total_tokens(preprocessed.input_token_ids.size(), request.max_new_tokens);
    return language_model->run_model(
        preprocessed.input_token_ids, simaai::llima::ChronoTimer{true}, max_total_tokens);
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
  std::unique_ptr<simaai::llima::VlmHelper> vlm_helper;
  std::unique_ptr<simaai::llima::TextStreamer> text_streamer;
  std::unique_ptr<simaai::llima::LanguageModel> language_model;
  std::mutex load_mutex;
  std::mutex run_state_mutex;
  std::condition_variable run_state_cv;
  bool run_active = false;
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

GenerationResult VisionLanguageModel::run(const GenerationRequest& request) {
  return impl_->run(request);
}

GenerationStream VisionLanguageModel::stream(const GenerationRequest& request) {
  internal::validate_text_generation_request(request);
  if (!request.formatted_prompt.empty()) {
    throw std::logic_error("GenerationRequest::formatted_prompt is not implemented yet");
  }
  return GenerationStream(std::make_unique<GenerationStream::Impl>(impl_, request));
}

} // namespace simaai::neat::genai
