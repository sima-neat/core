#include "genai/VisionLanguageModel.h"
#include "genai/GenAIInternal.h"

#include <sima_lmm/chat.hpp>
#include <sima_lmm/image_processor.hpp>
#include <sima_lmm/language_model.hpp>
#include <sima_lmm/mla_model.hpp>
#include <sima_lmm/text_streamer.hpp>
#include <sima_lmm/utils.hpp>
#include <sima_lmm/vlm_config.hpp>
#include <sima_lmm/vlm_helper.hpp>

#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace simaai::neat::genai {
namespace {

void ensure_llima_runtime_connected() {
  static std::once_flag once;
  std::call_once(once, [] {
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
  }

  GenerationResult run(const GenerationRequest& request) {
    internal::validate_text_generation_request(request);
    if (!request.formatted_prompt.empty()) {
      throw std::logic_error("GenerationRequest::formatted_prompt is not implemented yet");
    }

    std::lock_guard<std::mutex> lock(run_mutex);
    reset_metrics();

    simaai::llima::Chat chat(*vlm_helper);
    chat.set_messages(build_llima_messages(request));
    auto preprocessed = vlm_helper->preprocess(chat);
    language_model->create_input_buffers(preprocessed.input_token_ids);

    const auto max_total_tokens =
        make_max_total_tokens(preprocessed.input_token_ids.size(), request.max_new_tokens);
    auto output_token_ids = language_model->run_model(
        preprocessed.input_token_ids, simaai::llima::ChronoTimer{true}, max_total_tokens);

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
  std::mutex run_mutex;
  mutable std::mutex metrics_mutex;
  GenerationMetrics metrics;
};

VisionLanguageModel::VisionLanguageModel(std::filesystem::path model_dir)
    : impl_(std::make_unique<Impl>(std::move(model_dir))) {
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

} // namespace simaai::neat::genai
