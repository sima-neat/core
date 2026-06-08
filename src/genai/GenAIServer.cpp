#include "genai/GenAIServer.h"

#include "genai/GenAIInternal.h"
#include "genai/GenAIModel.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgcodecs.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace simaai::neat::genai {
namespace {

std::uint64_t unix_time_s() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
}

nlohmann::json parse_json_body(const httplib::Request& req) {
  try {
    return nlohmann::json::parse(req.body);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("Malformed JSON request body: ") + e.what());
  }
}

void set_json(httplib::Response& res, const nlohmann::json& body, int status = 200) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

void set_error(httplib::Response& res, const std::string& message, int status) {
  set_json(res, {{"error", {{"message", message}, {"type", "invalid_request_error"}}}}, status);
}

void write_sink(httplib::DataSink& sink, const std::string& text) {
  sink.write(text.data(), text.size());
}

bool parse_bool_string(const std::string& value) {
  return value == "true" || value == "1" || value == "True" || value == "TRUE";
}

std::string normalize_model_name(std::string name) {
  auto starts_with = [](const std::string& text, const std::string& prefix) {
    return text.rfind(prefix, 0) == 0;
  };

  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  for (const std::string& prefix : {"vlm-", "llm-", "asr-"}) {
    if (starts_with(name, prefix)) {
      name.erase(0, prefix.size());
      break;
    }
  }

  std::string out;
  bool last_dash = false;
  for (unsigned char ch : name) {
    const bool keep = std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-';
    const char normalized = keep ? static_cast<char>(ch) : '-';
    if (normalized == '-') {
      if (!last_dash) {
        out.push_back(normalized);
      }
      last_dash = true;
    } else {
      out.push_back(normalized);
      last_dash = false;
    }
  }
  while (!out.empty() && out.front() == '-') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == '-') {
    out.pop_back();
  }
  return out;
}

nlohmann::json parse_config_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to open GenAI model config: " + path.string());
  }
  try {
    return nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("Malformed GenAI model config " + path.string() + ": " + e.what());
  }
}

std::string config_model_type(const std::filesystem::path& model_root) {
  const auto vlm_config = model_root / "devkit" / "vlm_config.json";
  const auto whisper_config = model_root / "devkit" / "whisper_config.json";
  std::error_code ec;
  if (std::filesystem::is_regular_file(vlm_config, ec)) {
    const auto config = parse_config_file(vlm_config);
    return config.value("model_type", std::string{});
  }
  if (std::filesystem::is_regular_file(whisper_config, ec)) {
    const auto config = parse_config_file(whisper_config);
    return config.value("model_type", std::string{});
  }
  return {};
}

std::string default_served_name(const std::filesystem::path& model_root) {
  std::string name = normalize_model_name(config_model_type(model_root));
  if (name.empty()) {
    name = normalize_model_name(internal::model_id_from_path(model_root));
  }
  if (name.empty()) {
    throw std::runtime_error("Unable to derive served model name from " + model_root.string());
  }
  return name;
}

std::optional<std::uint32_t> json_u32(const nlohmann::json& body,
                                      std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    if (body.contains(key) && body.at(key).is_number_unsigned()) {
      return body.at(key).get<std::uint32_t>();
    }
    if (body.contains(key) && body.at(key).is_number_integer()) {
      const auto value = body.at(key).get<std::int64_t>();
      if (value >= 0 && value <= std::numeric_limits<std::uint32_t>::max()) {
        return static_cast<std::uint32_t>(value);
      }
    }
  }
  return std::nullopt;
}

bool json_bool(const nlohmann::json& body, const char* key, bool default_value = false) {
  if (!body.contains(key)) {
    return default_value;
  }
  const auto& value = body.at(key);
  if (value.is_boolean()) {
    return value.get<bool>();
  }
  if (value.is_string()) {
    return parse_bool_string(value.get<std::string>());
  }
  return default_value;
}

std::string choice_finish_reason(const std::string& finish_reason) {
  return finish_reason.empty() ? "stop" : finish_reason;
}

nlohmann::json chat_message_response(const GenerationResult& result) {
  nlohmann::json message = {{"role", "assistant"}};
  if (!result.tool_calls.empty()) {
    message["content"] = nullptr;
    message["tool_calls"] = result.tool_calls;
  } else {
    message["content"] = result.text;
  }
  return message;
}

std::string chat_chunk(const std::string& model_name, const std::string& text,
                       const std::optional<std::string>& finish_reason = std::nullopt,
                       const std::optional<GenerationMetrics>& metrics = std::nullopt) {
  nlohmann::json chunk;
  chunk["id"] = "chatcmpl-" + std::to_string(unix_time_s());
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = unix_time_s();
  chunk["model"] = model_name;
  if (metrics.has_value()) {
    if (metrics->time_to_first_token_s > 0.0) {
      chunk["ttft"] = metrics->time_to_first_token_s;
    }
    if (metrics->tokens_per_second > 0.0) {
      chunk["tps"] = metrics->tokens_per_second;
    }
    if (metrics->generated_tokens > 0U) {
      chunk["generated_tokens"] = metrics->generated_tokens;
    }
  }

  nlohmann::json choice;
  choice["index"] = 0;
  if (finish_reason.has_value()) {
    choice["delta"] = nlohmann::json::object();
    choice["finish_reason"] = *finish_reason;
  } else {
    choice["delta"] = {{"content", text}};
    choice["finish_reason"] = nullptr;
  }
  chunk["choices"] = nlohmann::json::array({choice});
  return "data: " + chunk.dump() + "\n\n";
}

std::string chat_tool_call_chunk(const std::string& model_name, const Json& tool_calls,
                                 const GenerationMetrics& metrics) {
  nlohmann::json delta_tool_calls = nlohmann::json::array();
  for (std::size_t i = 0; i < tool_calls.size(); ++i) {
    nlohmann::json tool_call = tool_calls.at(i);
    tool_call["index"] = static_cast<int>(i);
    delta_tool_calls.push_back(std::move(tool_call));
  }

  nlohmann::json chunk;
  chunk["id"] = "chatcmpl-" + std::to_string(unix_time_s());
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = unix_time_s();
  chunk["model"] = model_name;
  if (metrics.time_to_first_token_s > 0.0) {
    chunk["ttft"] = metrics.time_to_first_token_s;
  }
  if (metrics.tokens_per_second > 0.0) {
    chunk["tps"] = metrics.tokens_per_second;
  }
  if (metrics.generated_tokens > 0U) {
    chunk["generated_tokens"] = metrics.generated_tokens;
  }
  chunk["choices"] = nlohmann::json::array(
      {{{"index", 0}, {"delta", {{"tool_calls", delta_tool_calls}}}, {"finish_reason", nullptr}}});
  return "data: " + chunk.dump() + "\n\n";
}

Json openai_tool_calls_to_ollama(const Json& tool_calls) {
  Json out = Json::array();
  for (const auto& tool_call : tool_calls) {
    if (!tool_call.contains("function") || !tool_call.at("function").is_object()) {
      continue;
    }
    const auto& fn = tool_call.at("function");
    Json args = Json::object();
    if (fn.contains("arguments")) {
      const auto& value = fn.at("arguments");
      if (value.is_object()) {
        args = value;
      } else if (value.is_string()) {
        try {
          args = nlohmann::json::parse(value.get<std::string>());
        } catch (const nlohmann::json::exception&) {
          args = Json::object();
        }
      }
    }
    out.push_back({{"function", {{"name", fn.value("name", std::string{})}, {"arguments", args}}}});
  }
  return out;
}

std::string completion_chunk(const std::string& model_name, const std::string& text,
                             const std::optional<std::string>& finish_reason = std::nullopt,
                             const std::optional<GenerationMetrics>& metrics = std::nullopt) {
  nlohmann::json chunk;
  chunk["id"] = "cmpl-" + std::to_string(unix_time_s());
  chunk["object"] = "text_completion";
  chunk["created"] = unix_time_s();
  chunk["model"] = model_name;
  if (metrics.has_value()) {
    if (metrics->time_to_first_token_s > 0.0) {
      chunk["ttft"] = metrics->time_to_first_token_s;
    }
    if (metrics->tokens_per_second > 0.0) {
      chunk["tps"] = metrics->tokens_per_second;
    }
    if (metrics->generated_tokens > 0U) {
      chunk["generated_tokens"] = metrics->generated_tokens;
    }
  }

  nlohmann::json choice;
  choice["index"] = 0;
  choice["text"] = finish_reason.has_value() ? "" : text;
  choice["finish_reason"] =
      finish_reason.has_value() ? nlohmann::json(*finish_reason) : nlohmann::json(nullptr);
  chunk["choices"] = nlohmann::json::array({choice});
  return "data: " + chunk.dump() + "\n\n";
}

std::string audio_chunk(const std::string& text, bool finished,
                        const std::optional<std::string>& finish_reason = std::nullopt) {
  nlohmann::json chunk;
  chunk["object"] = finished ? "audio.transcription.done" : "audio.transcription.chunk";
  chunk["text"] = text;
  if (finished) {
    chunk["finish_reason"] = finish_reason.value_or("stop");
  }
  return "data: " + chunk.dump() + "\n\n";
}

std::string ollama_chat_line(const std::string& model_name, const std::string& text, bool done,
                             const std::optional<std::string>& finish_reason = std::nullopt,
                             const std::optional<GenerationMetrics>& metrics = std::nullopt) {
  nlohmann::json body;
  body["model"] = model_name;
  body["message"] = {{"role", "assistant"}, {"content", done ? "" : text}};
  body["done"] = done;
  if (done) {
    body["done_reason"] = finish_reason.value_or("stop");
  }
  if (metrics.has_value()) {
    if (metrics->time_to_first_token_s > 0.0) {
      body["ttft"] = metrics->time_to_first_token_s;
    }
    if (metrics->tokens_per_second > 0.0) {
      body["tps"] = metrics->tokens_per_second;
    }
    if (metrics->generated_tokens > 0U) {
      body["eval_count"] = metrics->generated_tokens;
    }
  }
  return body.dump() + "\n";
}

std::string ollama_generate_line(const std::string& model_name, const std::string& text, bool done,
                                 const std::optional<std::string>& finish_reason = std::nullopt,
                                 const std::optional<GenerationMetrics>& metrics = std::nullopt) {
  nlohmann::json body;
  body["model"] = model_name;
  body["response"] = done ? "" : text;
  body["done"] = done;
  if (done) {
    body["done_reason"] = finish_reason.value_or("stop");
  }
  if (metrics.has_value()) {
    if (metrics->time_to_first_token_s > 0.0) {
      body["ttft"] = metrics->time_to_first_token_s;
    }
    if (metrics->tokens_per_second > 0.0) {
      body["tps"] = metrics->tokens_per_second;
    }
    if (metrics->generated_tokens > 0U) {
      body["eval_count"] = metrics->generated_tokens;
    }
  }
  return body.dump() + "\n";
}

std::vector<std::uint8_t> decode_base64(std::string_view input) {
  static constexpr unsigned char kInvalid = 255;
  std::array<unsigned char, 256> table{};
  table.fill(kInvalid);
  const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (std::size_t i = 0; i < alphabet.size(); ++i) {
    table[static_cast<unsigned char>(alphabet[i])] = static_cast<unsigned char>(i);
  }

  std::vector<std::uint8_t> out;
  int value = 0;
  int bits = -8;
  for (unsigned char ch : input) {
    if (std::isspace(ch)) {
      continue;
    }
    if (ch == '=') {
      break;
    }
    const unsigned char decoded = table[ch];
    if (decoded == kInvalid) {
      throw std::runtime_error("Invalid base64 image data");
    }
    value = (value << 6) + decoded;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<std::uint8_t>((value >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

#if defined(SIMA_WITH_OPENCV)
Tensor tensor_from_data_uri(const std::string& uri) {
  const auto comma = uri.find(',');
  if (comma == std::string::npos || uri.rfind("data:image/", 0) != 0 ||
      uri.find(";base64", 0) == std::string::npos) {
    throw std::runtime_error("OpenAI image_url must be a data:image/...;base64 URI");
  }
  const auto bytes = decode_base64(std::string_view(uri).substr(comma + 1));
  const cv::Mat raw(1, static_cast<int>(bytes.size()), CV_8UC1,
                    const_cast<std::uint8_t*>(bytes.data()));
  const cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to decode OpenAI image_url image data");
  }
  ImageList image_list{bgr};
  return image_list.tensors().front();
}

void append_data_uri_image(ChatMessage& message, const std::string& uri) {
  message.images.tensors().push_back(tensor_from_data_uri(uri));
}
#else
void append_data_uri_image(ChatMessage&, const std::string&) {
  throw std::runtime_error("OpenAI image input requires SIMA_WITH_OPENCV");
}
#endif

void append_content_text(std::string& content, const std::string& text) {
  if (!content.empty() && !text.empty()) {
    content.push_back(' ');
  }
  content += text;
}

void append_openai_content_part(ChatMessage& message, const nlohmann::json& part) {
  const std::string type = part.value("type", std::string{});
  if (type == "text") {
    append_content_text(message.content, part.value("text", std::string{}));
    return;
  }

  if (type == "image_url") {
    const auto& image_url = part.at("image_url");
    const std::string url = image_url.is_string() ? image_url.get<std::string>()
                                                  : image_url.value("url", std::string{});
    append_data_uri_image(message, url);
    return;
  }

  if (type == "image" && part.contains("image")) {
    append_data_uri_image(message, part.at("image").get<std::string>());
  }
}

void append_ollama_images(ChatMessage& message, const nlohmann::json& images) {
  if (!images.is_array()) {
    return;
  }
  for (const auto& image : images) {
    if (image.is_string()) {
      append_data_uri_image(message, "data:image/jpeg;base64," + image.get<std::string>());
    }
  }
}

Tensor make_warmup_image() {
  std::vector<std::uint8_t> pixels(224U * 224U * 3U, 0U);
  Tensor tensor = Tensor::from_vector(std::move(pixels), {224, 224, 3}, TensorMemory::CPU);
  tensor.layout = TensorLayout::HWC;
  tensor.semantic.image = ImageSpec{ImageSpec::PixelFormat::RGB, ""};
  return tensor;
}

Tensor make_warmup_audio() {
  std::vector<float> samples(1600U, 0.0F);
  Tensor tensor = Tensor::from_vector(std::move(samples), {1600}, TensorMemory::CPU);
  tensor.semantic.audio = AudioSpec{
      .sample_rate = 16000,
      .channels = 1,
      .interleaved = true,
  };
  return tensor;
}

GenerationRequest make_warmup_request(const GenAIModel& model) {
  GenerationRequest request;
  request.max_new_tokens = 1;
  if (model.accepts_audio()) {
    request.audio = make_warmup_audio();
    request.language = "en";
    return request;
  }

  request.prompt = "Hello";
  if (model.accepts_image()) {
    request.prompt = "Describe the image.";
    request.images = ImageList{make_warmup_image()};
  }
  return request;
}

std::vector<ChatMessage> parse_chat_messages(const nlohmann::json& body) {
  if (!body.contains("messages") || !body.at("messages").is_array()) {
    throw std::runtime_error("Chat request requires messages array");
  }

  std::vector<ChatMessage> messages;
  for (const auto& item : body.at("messages")) {
    ChatMessage message;
    message.role = item.value("role", "user");

    if (item.contains("content")) {
      const auto& content = item.at("content");
      if (content.is_string()) {
        message.content = content.get<std::string>();
      } else if (content.is_array()) {
        for (const auto& part : content) {
          append_openai_content_part(message, part);
        }
      } else if (!content.is_null()) {
        throw std::runtime_error("OpenAI chat message content must be string or array");
      }
    }

    if (item.contains("images")) {
      append_ollama_images(message, item.at("images"));
    }

    if (item.contains("tool_calls") && item.at("tool_calls").is_array()) {
      message.tool_calls = item.at("tool_calls");
    } else if (item.contains("function_call") && item.at("function_call").is_object()) {
      const auto& function_call = item.at("function_call");
      message.tool_calls =
          Json::array({{{"id", "call_0"}, {"type", "function"}, {"function", function_call}}});
    }

    if (item.contains("tool_call_id") && item.at("tool_call_id").is_string()) {
      message.tool_call_id = item.at("tool_call_id").get<std::string>();
    }
    if (item.contains("name") && item.at("name").is_string()) {
      message.name = item.at("name").get<std::string>();
    }

    if (message.content.empty() && message.images.empty() && message.tool_calls.empty() &&
        !message.tool_call_id.has_value()) {
      continue;
    }
    messages.push_back(std::move(message));
  }
  return messages;
}

std::filesystem::path write_uploaded_file(const httplib::MultipartFormData& file) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path extension = std::filesystem::path(file.filename).extension();
  if (extension.empty()) {
    extension = ".wav";
  }
  const auto path = std::filesystem::temp_directory_path() /
                    ("sima-neat-openai-audio-" + std::to_string(now) + extension.string());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Unable to create temporary audio upload: " + path.string());
  }
  out.write(file.content.data(), static_cast<std::streamsize>(file.content.size()));
  if (!out) {
    throw std::runtime_error("Failed to write temporary audio upload: " + path.string());
  }
  return path;
}

struct TempFileGuard {
  std::filesystem::path path;
  ~TempFileGuard() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

} // namespace

struct GenAIServer::Impl {
  explicit Impl(GenAIServerOptions options_in) : options(std::move(options_in)) {
    configure_routes();
  }

  struct Entry {
    std::shared_ptr<GenAIModel> model;
  };

  struct ActiveStreamHandle {
    void attach(GenerationStream& stream_in) {
      std::lock_guard<std::mutex> lock(mutex);
      stream = &stream_in;
      if (cancelled) {
        stream->cancel();
      }
    }

    void cancel() {
      std::lock_guard<std::mutex> lock(mutex);
      cancelled = true;
      if (stream) {
        stream->cancel();
      }
    }

    std::mutex mutex;
    GenerationStream* stream = nullptr;
    bool cancelled = false;
  };

  class ActiveStreamRegistration {
  public:
    ActiveStreamRegistration(Impl& owner_in, std::string model_name_in)
        : owner(&owner_in), model_name(std::move(model_name_in)),
          handle(std::make_shared<ActiveStreamHandle>()) {
      owner->register_active_stream(model_name, handle);
    }

    ActiveStreamRegistration(const ActiveStreamRegistration&) = delete;
    ActiveStreamRegistration& operator=(const ActiveStreamRegistration&) = delete;

    ~ActiveStreamRegistration() {
      if (owner) {
        owner->unregister_active_stream(model_name, handle);
      }
    }

    void attach(GenerationStream& stream) {
      handle->attach(stream);
    }

  private:
    Impl* owner;
    std::string model_name;
    std::shared_ptr<ActiveStreamHandle> handle;
  };

  void configure_routes() {
    http.Get("/v1/models",
             [this](const httplib::Request&, httplib::Response& res) { handle_models(res); });
    http.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
      handle_chat(req, res);
    });
    http.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
      handle_completion(req, res);
    });
    http.Post(
        "/v1/audio/transcriptions",
        [this](const httplib::Request& req, httplib::Response& res) { handle_audio(req, res); });
    http.Post("/audio/transcriptions", [this](const httplib::Request& req, httplib::Response& res) {
      handle_audio(req, res);
    });
    http.Post("/api/chat", [this](const httplib::Request& req, httplib::Response& res) {
      handle_ollama_chat(req, res);
    });
    http.Post("/api/generate", [this](const httplib::Request& req, httplib::Response& res) {
      handle_ollama_generate(req, res);
    });
    http.Post("/stop", [this](const httplib::Request& req, httplib::Response& res) {
      handle_stop(req, res);
    });
    auto options_handler = [](const httplib::Request&, httplib::Response& res) {
      set_cors(res);
      res.status = 200;
    };
    http.Options("/v1/models", options_handler);
    http.Options("/v1/chat/completions", options_handler);
    http.Options("/v1/completions", options_handler);
    http.Options("/v1/audio/transcriptions", options_handler);
    http.Options("/audio/transcriptions", options_handler);
    http.Options("/api/chat", options_handler);
    http.Options("/api/generate", options_handler);
    http.Options("/stop", options_handler);
  }

  static void set_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
  }

  std::string add_model(std::filesystem::path model_dir) {
    const auto info = internal::inspect_model_directory(model_dir);
    return add_model(info.root, default_served_name(info.root));
  }

  std::string add_model(std::filesystem::path model_dir, std::string served_name) {
    if (served_name.empty()) {
      throw std::invalid_argument("GenAIServer::add_model requires a non-empty served name");
    }
    const std::string registered_name = served_name;
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (registry.contains(served_name)) {
        throw std::runtime_error("OpenAI served model already exists: " + served_name);
      }
    }

    auto model = std::make_shared<GenAIModel>(std::move(model_dir));
    add_model(std::move(served_name), std::move(model));
    return registered_name;
  }

  void add_model(std::string served_name, std::shared_ptr<GenAIModel> model) {
    if (served_name.empty()) {
      throw std::invalid_argument("GenAIServer::add_model requires a non-empty served name");
    }
    if (!model) {
      throw std::invalid_argument("GenAIServer::add_model requires a non-null model");
    }
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (registry.contains(served_name)) {
      throw std::runtime_error("OpenAI served model already exists: " + served_name);
    }
    registry.emplace(std::move(served_name), Entry{std::move(model)});
  }

  bool remove_model(const std::string& served_name) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return registry.erase(served_name) > 0U;
  }

  std::vector<std::string> model_names() const {
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::vector<std::string> names;
    names.reserve(registry.size());
    for (const auto& [name, _] : registry) {
      names.push_back(name);
    }
    return names;
  }

  std::shared_ptr<GenAIModel> find_model(const std::string& served_name) const {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find(served_name);
    return it == registry.end() ? nullptr : it->second.model;
  }

  void register_active_stream(const std::string& model_name,
                              const std::shared_ptr<ActiveStreamHandle>& handle) {
    std::lock_guard<std::mutex> lock(active_streams_mutex);
    active_streams[model_name].push_back(handle);
  }

  void unregister_active_stream(const std::string& model_name,
                                const std::shared_ptr<ActiveStreamHandle>& handle) {
    std::lock_guard<std::mutex> lock(active_streams_mutex);
    auto it = active_streams.find(model_name);
    if (it == active_streams.end()) {
      return;
    }
    auto& handles = it->second;
    handles.erase(std::remove_if(handles.begin(), handles.end(),
                                 [&handle](const std::weak_ptr<ActiveStreamHandle>& stored) {
                                   const auto locked = stored.lock();
                                   return !locked || locked == handle;
                                 }),
                  handles.end());
    if (handles.empty()) {
      active_streams.erase(it);
    }
  }

  std::size_t cancel_active_streams(const std::optional<std::string>& model_name) {
    std::vector<std::shared_ptr<ActiveStreamHandle>> handles;
    {
      std::lock_guard<std::mutex> lock(active_streams_mutex);
      if (model_name.has_value()) {
        const auto it = active_streams.find(*model_name);
        if (it != active_streams.end()) {
          for (const auto& weak_handle : it->second) {
            if (auto handle = weak_handle.lock()) {
              handles.push_back(std::move(handle));
            }
          }
        }
      } else {
        for (const auto& [_, weak_handles] : active_streams) {
          for (const auto& weak_handle : weak_handles) {
            if (auto handle = weak_handle.lock()) {
              handles.push_back(std::move(handle));
            }
          }
        }
      }
    }

    for (const auto& handle : handles) {
      handle->cancel();
    }
    return handles.size();
  }

  void handle_models(httplib::Response& res) const {
    set_cors(res);
    nlohmann::json data = nlohmann::json::array();
    for (const auto& name : model_names()) {
      data.push_back({{"id", name}, {"object", "model"}, {"owned_by", "simaai"}});
    }
    set_json(res, {{"object", "list"}, {"data", data}});
  }

  std::optional<std::string> stop_model_name(const httplib::Request& req) const {
    if (req.has_param("model")) {
      return req.get_param_value("model");
    }
    if (req.has_file("model")) {
      return req.get_file_value("model").content;
    }
    if (req.body.empty()) {
      return std::nullopt;
    }
    const auto body = parse_json_body(req);
    const std::string model = body.value("model", std::string{});
    return model.empty() ? std::nullopt : std::optional<std::string>{model};
  }

  static bool messages_use_images(const std::vector<ChatMessage>& messages) {
    return std::any_of(messages.begin(), messages.end(), [](const ChatMessage& message) {
      return !message.images.empty() || message.use_cached_images;
    });
  }

  static std::string completion_prompt(const nlohmann::json& body) {
    if (!body.contains("prompt")) {
      throw std::runtime_error("OpenAI completion request requires prompt");
    }
    const auto& prompt = body.at("prompt");
    if (prompt.is_string()) {
      return prompt.get<std::string>();
    }
    if (prompt.is_array()) {
      std::string out;
      for (const auto& item : prompt) {
        if (!item.is_string()) {
          throw std::runtime_error("OpenAI completion prompt array must contain strings");
        }
        if (!out.empty()) {
          out.push_back('\n');
        }
        out += item.get<std::string>();
      }
      return out;
    }
    throw std::runtime_error("OpenAI completion prompt must be string or string array");
  }

  static std::optional<std::uint32_t> ollama_max_tokens(const nlohmann::json& body) {
    if (const auto max_tokens = json_u32(body, {"max_tokens", "max_completion_tokens"})) {
      return max_tokens;
    }
    if (body.contains("options") && body.at("options").is_object()) {
      return json_u32(body.at("options"), {"num_predict"});
    }
    return std::nullopt;
  }

  std::shared_ptr<GenAIModel> require_model(const std::string& model_name, httplib::Response& res,
                                            const std::string& request_kind) const {
    if (model_name.empty()) {
      set_error(res, request_kind + " request requires model", 400);
      return nullptr;
    }
    auto model = find_model(model_name);
    if (!model) {
      set_error(res, "Unknown model: " + model_name, 404);
      return nullptr;
    }
    return model;
  }

  bool require_text_or_vision_model(const GenAIModel& model, const std::string& model_name,
                                    httplib::Response& res) const {
    if (model.accepts_audio()) {
      set_error(res, "Model is not a text or vision-language model: " + model_name, 400);
      return false;
    }
    return true;
  }

  bool require_image_capability(const GenAIModel& model, const std::string& model_name,
                                const GenerationRequest& request, httplib::Response& res) const {
    if ((!request.images.empty() || messages_use_images(request.messages)) &&
        !model.accepts_image()) {
      set_error(res, "Model does not accept image input: " + model_name, 400);
      return false;
    }
    return true;
  }

  void handle_stop(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto model_name = stop_model_name(req);
      if (model_name.has_value() && !find_model(*model_name)) {
        set_error(res, "Unknown model: " + *model_name, 404);
        return;
      }
      const auto cancelled = cancel_active_streams(model_name);
      set_json(res, {{"status", "stopping"},
                     {"model", model_name.value_or(std::string{"*"})},
                     {"cancelled_streams", cancelled}});
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_chat(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto body = parse_json_body(req);
      const std::string model_name = body.value("model", std::string{});
      auto model = require_model(model_name, res, "OpenAI chat");
      if (!model) {
        return;
      }
      if (!require_text_or_vision_model(*model, model_name, res)) {
        return;
      }

      GenerationRequest request;
      request.messages = parse_chat_messages(body);
      if (body.contains("tools") && body.at("tools").is_array()) {
        request.tools = body.at("tools");
      }
      if (body.contains("tool_choice")) {
        request.tool_choice = body.at("tool_choice");
      }
      if (!require_image_capability(*model, model_name, request, res)) {
        return;
      }
      if (const auto max_tokens = json_u32(body, {"max_tokens", "max_completion_tokens"})) {
        request.max_new_tokens = *max_tokens;
      }
      const bool stream = json_bool(body, "stream");
      if (stream) {
        handle_chat_stream(res, model_name, std::move(model), std::move(request));
      } else {
        const auto result = model->run(request);
        const nlohmann::json message = chat_message_response(result);
        set_json(res, {{"id", "chatcmpl-" + std::to_string(unix_time_s())},
                       {"object", "chat.completion"},
                       {"created", unix_time_s()},
                       {"model", model_name},
                       {"choices",
                        nlohmann::json::array(
                            {{{"index", 0},
                              {"message", message},
                              {"finish_reason", choice_finish_reason(result.finish_reason)}}})},
                       {"usage", {{"completion_tokens", result.metrics.generated_tokens}}}});
      }
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_completion(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto body = parse_json_body(req);
      const std::string model_name = body.value("model", std::string{});
      auto model = require_model(model_name, res, "OpenAI completion");
      if (!model) {
        return;
      }
      if (!require_text_or_vision_model(*model, model_name, res)) {
        return;
      }

      GenerationRequest request;
      request.prompt = completion_prompt(body);
      if (const auto max_tokens = json_u32(body, {"max_tokens", "max_completion_tokens"})) {
        request.max_new_tokens = *max_tokens;
      }
      const bool stream = json_bool(body, "stream");
      if (stream) {
        handle_completion_stream(res, model_name, std::move(model), std::move(request));
      } else {
        const auto result = model->run(request);
        set_json(res, {{"id", "cmpl-" + std::to_string(unix_time_s())},
                       {"object", "text_completion"},
                       {"created", unix_time_s()},
                       {"model", model_name},
                       {"choices",
                        nlohmann::json::array(
                            {{{"index", 0},
                              {"text", result.text},
                              {"finish_reason", choice_finish_reason(result.finish_reason)}}})},
                       {"usage", {{"completion_tokens", result.metrics.generated_tokens}}}});
      }
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_ollama_chat(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto body = parse_json_body(req);
      const std::string model_name = body.value("model", std::string{});
      auto model = require_model(model_name, res, "Ollama chat");
      if (!model) {
        return;
      }
      if (!require_text_or_vision_model(*model, model_name, res)) {
        return;
      }

      GenerationRequest request;
      request.messages = parse_chat_messages(body);
      if (body.contains("tools") && body.at("tools").is_array()) {
        request.tools = body.at("tools");
      }
      if (body.contains("tool_choice")) {
        request.tool_choice = body.at("tool_choice");
      }
      if (!require_image_capability(*model, model_name, request, res)) {
        return;
      }
      if (const auto max_tokens = ollama_max_tokens(body)) {
        request.max_new_tokens = *max_tokens;
      }
      const bool stream = json_bool(body, "stream", true);
      if (stream) {
        handle_ollama_chat_stream(res, model_name, std::move(model), std::move(request));
      } else {
        const auto result = model->run(request);
        set_json(res, {{"model", model_name},
                       {"message", chat_message_response(result)},
                       {"done", true},
                       {"done_reason", choice_finish_reason(result.finish_reason)},
                       {"eval_count", result.metrics.generated_tokens}});
      }
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_ollama_generate(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto body = parse_json_body(req);
      const std::string model_name = body.value("model", std::string{});
      auto model = require_model(model_name, res, "Ollama generate");
      if (!model) {
        return;
      }
      if (!require_text_or_vision_model(*model, model_name, res)) {
        return;
      }
      if (!body.contains("prompt") || !body.at("prompt").is_string()) {
        set_error(res, "Ollama generate request requires string prompt", 400);
        return;
      }

      ChatMessage message;
      message.role = "user";
      message.content = body.at("prompt").get<std::string>();
      if (body.contains("images")) {
        append_ollama_images(message, body.at("images"));
      }

      GenerationRequest request;
      request.messages.push_back(std::move(message));
      if (!require_image_capability(*model, model_name, request, res)) {
        return;
      }
      if (const auto max_tokens = ollama_max_tokens(body)) {
        request.max_new_tokens = *max_tokens;
      }
      const bool stream = json_bool(body, "stream", true);
      if (stream) {
        handle_ollama_generate_stream(res, model_name, std::move(model), std::move(request));
      } else {
        const auto result = model->run(request);
        set_json(res, {{"model", model_name},
                       {"response", result.text},
                       {"done", true},
                       {"done_reason", choice_finish_reason(result.finish_reason)},
                       {"eval_count", result.metrics.generated_tokens}});
      }
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_chat_stream(httplib::Response& res, std::string model_name,
                          std::shared_ptr<GenAIModel> model, GenerationRequest request) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, model_name = std::move(model_name), model = std::move(model),
         request = std::move(request)](std::size_t, httplib::DataSink& sink) mutable {
          try {
            ActiveStreamRegistration active_stream{*this, model_name};
            auto stream = model->stream(request);
            active_stream.attach(stream);
            for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
              if (sample->is_final) {
                const auto final_chunk =
                    chat_chunk(model_name, "", choice_finish_reason(sample->finish_reason),
                               sample->metrics) +
                    "data: [DONE]\n\n";
                write_sink(sink, final_chunk);
                sink.done();
                return true;
              }
              if (!sample->tool_calls.empty()) {
                write_sink(sink,
                           chat_tool_call_chunk(model_name, sample->tool_calls, sample->metrics));
                continue;
              }
              const auto chunk =
                  chat_chunk(model_name, sample->text, std::nullopt, sample->metrics);
              write_sink(sink, chunk);
            }
            const auto done = chat_chunk(model_name, "", "stop") + "data: [DONE]\n\n";
            write_sink(sink, done);
          } catch (const std::exception& e) {
            const nlohmann::json error = {{"error", {{"message", e.what()}}}};
            const std::string chunk = "data: " + error.dump() + "\n\ndata: [DONE]\n\n";
            write_sink(sink, chunk);
          }
          sink.done();
          return true;
        });
  }

  void handle_completion_stream(httplib::Response& res, std::string model_name,
                                std::shared_ptr<GenAIModel> model, GenerationRequest request) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, model_name = std::move(model_name), model = std::move(model),
         request = std::move(request)](std::size_t, httplib::DataSink& sink) mutable {
          try {
            ActiveStreamRegistration active_stream{*this, model_name};
            auto stream = model->stream(request);
            active_stream.attach(stream);
            for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
              if (sample->is_final) {
                const auto final_chunk =
                    completion_chunk(model_name, "", choice_finish_reason(sample->finish_reason),
                                     sample->metrics) +
                    "data: [DONE]\n\n";
                write_sink(sink, final_chunk);
                sink.done();
                return true;
              }
              write_sink(sink,
                         completion_chunk(model_name, sample->text, std::nullopt, sample->metrics));
            }
            write_sink(sink, completion_chunk(model_name, "", "stop") + "data: [DONE]\n\n");
          } catch (const std::exception& e) {
            const nlohmann::json error = {{"error", {{"message", e.what()}}}};
            write_sink(sink, "data: " + error.dump() + "\n\ndata: [DONE]\n\n");
          }
          sink.done();
          return true;
        });
  }

  void handle_ollama_chat_stream(httplib::Response& res, std::string model_name,
                                 std::shared_ptr<GenAIModel> model, GenerationRequest request) {
    res.set_header("Content-Type", "application/x-ndjson");
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "application/x-ndjson",
        [this, model_name = std::move(model_name), model = std::move(model),
         request = std::move(request)](std::size_t, httplib::DataSink& sink) mutable {
          try {
            ActiveStreamRegistration active_stream{*this, model_name};
            auto stream = model->stream(request);
            active_stream.attach(stream);
            Json pending_tool_calls = nullptr;
            GenerationMetrics pending_tool_metrics;
            for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
              if (sample->is_final) {
                if (!pending_tool_calls.is_null()) {
                  nlohmann::json body;
                  body["model"] = model_name;
                  body["message"] = {
                      {"role", "assistant"},
                      {"content", ""},
                      {"tool_calls", openai_tool_calls_to_ollama(pending_tool_calls)}};
                  body["done"] = true;
                  body["done_reason"] = choice_finish_reason(sample->finish_reason);
                  if (pending_tool_metrics.time_to_first_token_s > 0.0) {
                    body["ttft"] = pending_tool_metrics.time_to_first_token_s;
                  }
                  if (pending_tool_metrics.tokens_per_second > 0.0) {
                    body["tps"] = pending_tool_metrics.tokens_per_second;
                  }
                  if (sample->metrics.generated_tokens > 0U) {
                    body["eval_count"] = sample->metrics.generated_tokens;
                  }
                  write_sink(sink, body.dump() + "\n");
                  sink.done();
                  return true;
                }
                write_sink(sink, ollama_chat_line(model_name, "", true,
                                                  choice_finish_reason(sample->finish_reason),
                                                  sample->metrics));
                sink.done();
                return true;
              }
              if (!sample->tool_calls.empty()) {
                pending_tool_calls = sample->tool_calls;
                pending_tool_metrics = sample->metrics;
                continue;
              }
              write_sink(sink, ollama_chat_line(model_name, sample->text, false, std::nullopt,
                                                sample->metrics));
            }
            write_sink(sink, ollama_chat_line(model_name, "", true, "stop"));
          } catch (const std::exception& e) {
            write_sink(sink, nlohmann::json({{"error", e.what()}, {"done", true}}).dump() + "\n");
          }
          sink.done();
          return true;
        });
  }

  void handle_ollama_generate_stream(httplib::Response& res, std::string model_name,
                                     std::shared_ptr<GenAIModel> model, GenerationRequest request) {
    res.set_header("Content-Type", "application/x-ndjson");
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "application/x-ndjson",
        [this, model_name = std::move(model_name), model = std::move(model),
         request = std::move(request)](std::size_t, httplib::DataSink& sink) mutable {
          try {
            ActiveStreamRegistration active_stream{*this, model_name};
            auto stream = model->stream(request);
            active_stream.attach(stream);
            for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
              if (sample->is_final) {
                write_sink(sink, ollama_generate_line(model_name, "", true,
                                                      choice_finish_reason(sample->finish_reason),
                                                      sample->metrics));
                sink.done();
                return true;
              }
              write_sink(sink, ollama_generate_line(model_name, sample->text, false, std::nullopt,
                                                    sample->metrics));
            }
            write_sink(sink, ollama_generate_line(model_name, "", true, "stop"));
          } catch (const std::exception& e) {
            write_sink(sink, nlohmann::json({{"error", e.what()}, {"done", true}}).dump() + "\n");
          }
          sink.done();
          return true;
        });
  }

  void handle_audio(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      std::string model_name;
      if (req.has_param("model")) {
        model_name = req.get_param_value("model");
      } else if (req.has_file("model")) {
        model_name = req.get_file_value("model").content;
      }
      if (model_name.empty()) {
        set_error(res, "OpenAI audio transcription request requires model", 400);
        return;
      }
      auto model = find_model(model_name);
      if (!model) {
        set_error(res, "Unknown model: " + model_name, 404);
        return;
      }
      if (!model->accepts_audio()) {
        set_error(res, "Model is not an ASR model: " + model_name, 400);
        return;
      }
      if (!req.has_file("file")) {
        set_error(res, "OpenAI audio transcription request requires file", 400);
        return;
      }

      std::string language = "en";
      if (req.has_param("language")) {
        language = req.get_param_value("language");
      } else if (req.has_file("language")) {
        language = req.get_file_value("language").content;
      }

      bool stream = false;
      if (req.has_param("stream")) {
        stream = parse_bool_string(req.get_param_value("stream"));
      } else if (req.has_file("stream")) {
        stream = parse_bool_string(req.get_file_value("stream").content);
      }

      const auto audio_path = write_uploaded_file(req.get_file_value("file"));
      if (stream) {
        handle_audio_stream(res, model_name, std::move(model), audio_path, language);
      } else {
        TempFileGuard guard{audio_path};
        GenerationRequest request;
        request.audio_file = audio_path;
        request.language = language;
        const auto result = model->run(request);
        set_json(res, {{"text", result.text},
                       {"model", model_name},
                       {"finish_reason", choice_finish_reason(result.finish_reason)}});
      }
    } catch (const std::exception& e) {
      set_error(res, e.what(), 500);
    }
  }

  void handle_audio_stream(httplib::Response& res, std::string model_name,
                           std::shared_ptr<GenAIModel> model, std::filesystem::path audio_path,
                           std::string language) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider("text/event-stream", [this, model_name = std::move(model_name),
                                                           model = std::move(model),
                                                           audio_path = std::move(audio_path),
                                                           language = std::move(language)](
                                                              std::size_t,
                                                              httplib::DataSink& sink) mutable {
      TempFileGuard guard{audio_path};
      try {
        GenerationRequest request;
        request.audio_file = audio_path;
        request.language = language;
        ActiveStreamRegistration active_stream{*this, model_name};
        auto stream = model->stream(request);
        active_stream.attach(stream);
        for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
          if (sample->is_final) {
            const auto final_chunk =
                audio_chunk("", true, choice_finish_reason(sample->finish_reason)) +
                "data: [DONE]\n\n";
            write_sink(sink, final_chunk);
            sink.done();
            return true;
          }
          const auto chunk = audio_chunk(sample->text, false);
          write_sink(sink, chunk);
        }
        const auto done = audio_chunk("", true, "stop") + "data: [DONE]\n\n";
        write_sink(sink, done);
      } catch (const std::exception& e) {
        const nlohmann::json error = {{"object", "audio.transcription.error"}, {"error", e.what()}};
        const std::string chunk = "data: " + error.dump() + "\n\ndata: [DONE]\n\n";
        write_sink(sink, chunk);
      }
      sink.done();
      return true;
    });
  }

  void warmup_models_once() {
    {
      std::lock_guard<std::mutex> lock(warmup_mutex);
      if (warmup_complete) {
        return;
      }
    }

    std::vector<std::pair<std::string, std::shared_ptr<GenAIModel>>> models;
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      models.reserve(registry.size());
      for (const auto& [name, entry] : registry) {
        models.emplace_back(name, entry.model);
      }
    }

    for (const auto& [name, model] : models) {
      try {
        (void)model->run(make_warmup_request(*model));
      } catch (const std::exception& e) {
        throw std::runtime_error("GenAIServer warmup failed for model '" + name +
                                 "': " + e.what());
      }
    }

    {
      std::lock_guard<std::mutex> lock(warmup_mutex);
      warmup_complete = true;
    }
  }

  void serve() {
    if (running.exchange(true)) {
      throw std::runtime_error("GenAIServer is already running");
    }
    try {
      warmup_models_once();
      const bool ok = http.listen(options.host, options.port);
      running.store(false);
      if (!ok && !stopping.load()) {
        throw std::runtime_error("GenAIServer failed to listen on " + options.host + ":" +
                                 std::to_string(options.port));
      }
    } catch (...) {
      running.store(false);
      throw;
    }
  }

  void start() {
    if (worker.joinable()) {
      throw std::runtime_error("GenAIServer is already started");
    }
    stopping.store(false);
    warmup_models_once();
    worker = std::thread([this] { serve(); });
  }

  void stop() {
    stopping.store(true);
    http.stop();
    if (worker.joinable()) {
      worker.join();
    }
    running.store(false);
  }

  GenAIServerOptions options;
  httplib::Server http;
  mutable std::mutex registry_mutex;
  std::map<std::string, Entry> registry;
  mutable std::mutex active_streams_mutex;
  std::map<std::string, std::vector<std::weak_ptr<ActiveStreamHandle>>> active_streams;
  std::mutex warmup_mutex;
  bool warmup_complete = false;
  std::thread worker;
  std::atomic<bool> running = false;
  std::atomic<bool> stopping = false;
};

GenAIServer::GenAIServer(GenAIServerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

GenAIServer::~GenAIServer() {
  if (impl_) {
    impl_->stop();
  }
}

GenAIServer::GenAIServer(GenAIServer&&) noexcept = default;

GenAIServer& GenAIServer::operator=(GenAIServer&&) noexcept = default;

std::string GenAIServer::add_model(std::filesystem::path model_dir) {
  return impl_->add_model(std::move(model_dir));
}

std::string GenAIServer::add_model(std::filesystem::path model_dir, std::string served_name) {
  return impl_->add_model(std::move(model_dir), std::move(served_name));
}

void GenAIServer::add_model(std::string served_name, std::shared_ptr<GenAIModel> model) {
  impl_->add_model(std::move(served_name), std::move(model));
}

bool GenAIServer::remove_model(const std::string& served_name) {
  return impl_->remove_model(served_name);
}

std::vector<std::string> GenAIServer::model_names() const {
  return impl_->model_names();
}

void GenAIServer::serve() {
  impl_->serve();
}

void GenAIServer::start() {
  impl_->start();
}

void GenAIServer::stop() {
  impl_->stop();
}

} // namespace simaai::neat::genai
