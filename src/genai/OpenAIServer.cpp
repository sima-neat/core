#include "genai/OpenAIServer.h"

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

std::string chat_chunk(const std::string& model_name, const std::string& text,
                       const std::optional<std::string>& finish_reason = std::nullopt) {
  nlohmann::json chunk;
  chunk["id"] = "chatcmpl-" + std::to_string(unix_time_s());
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = unix_time_s();
  chunk["model"] = model_name;

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
#endif

void append_content_text(std::string& content, const std::string& text) {
  if (!content.empty() && !text.empty()) {
    content.push_back(' ');
  }
  content += text;
}

std::vector<ChatMessage> parse_chat_messages(const nlohmann::json& body) {
  if (!body.contains("messages") || !body.at("messages").is_array()) {
    throw std::runtime_error("OpenAI chat request requires messages array");
  }

  std::vector<ChatMessage> messages;
  for (const auto& item : body.at("messages")) {
    ChatMessage message;
    message.role = item.value("role", "user");

    if (!item.contains("content")) {
      messages.push_back(std::move(message));
      continue;
    }
    const auto& content = item.at("content");
    if (content.is_string()) {
      message.content = content.get<std::string>();
    } else if (content.is_array()) {
      for (const auto& part : content) {
        const std::string type = part.value("type", std::string{});
        if (type == "text") {
          append_content_text(message.content, part.value("text", std::string{}));
        } else if (type == "image_url") {
#if defined(SIMA_WITH_OPENCV)
          const auto& image_url = part.at("image_url");
          const std::string url = image_url.is_string() ? image_url.get<std::string>()
                                                        : image_url.value("url", std::string{});
          message.images.tensors().push_back(tensor_from_data_uri(url));
#else
          throw std::runtime_error("OpenAI image_url input requires SIMA_WITH_OPENCV");
#endif
        }
      }
    } else if (!content.is_null()) {
      throw std::runtime_error("OpenAI chat message content must be string or array");
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

struct OpenAIServer::Impl {
  explicit Impl(OpenAIServerOptions options_in) : options(std::move(options_in)) {
    configure_routes();
  }

  struct Entry {
    std::shared_ptr<GenAIModel> model;
  };

  void configure_routes() {
    http.Get("/v1/models",
             [this](const httplib::Request&, httplib::Response& res) { handle_models(res); });
    http.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
      handle_chat(req, res);
    });
    http.Post(
        "/v1/audio/transcriptions",
        [this](const httplib::Request& req, httplib::Response& res) { handle_audio(req, res); });
    auto options_handler = [](const httplib::Request&, httplib::Response& res) {
      set_cors(res);
      res.status = 200;
    };
    http.Options("/v1/models", options_handler);
    http.Options("/v1/chat/completions", options_handler);
    http.Options("/v1/audio/transcriptions", options_handler);
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
      throw std::invalid_argument("OpenAIServer::add_model requires a non-empty served name");
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
      throw std::invalid_argument("OpenAIServer::add_model requires a non-empty served name");
    }
    if (!model) {
      throw std::invalid_argument("OpenAIServer::add_model requires a non-null model");
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

  void handle_models(httplib::Response& res) const {
    set_cors(res);
    nlohmann::json data = nlohmann::json::array();
    for (const auto& name : model_names()) {
      data.push_back({{"id", name}, {"object", "model"}, {"owned_by", "simaai"}});
    }
    set_json(res, {{"object", "list"}, {"data", data}});
  }

  void handle_chat(const httplib::Request& req, httplib::Response& res) {
    set_cors(res);
    try {
      const auto body = parse_json_body(req);
      const std::string model_name = body.value("model", std::string{});
      if (model_name.empty()) {
        set_error(res, "OpenAI chat request requires model", 400);
        return;
      }
      auto model = find_model(model_name);
      if (!model) {
        set_error(res, "Unknown OpenAI model: " + model_name, 404);
        return;
      }
      if (model->accepts_audio()) {
        set_error(res, "Model is not a text or vision-language model: " + model_name, 400);
        return;
      }

      GenerationRequest request;
      request.messages = parse_chat_messages(body);
      if (const auto max_tokens = json_u32(body, {"max_tokens", "max_completion_tokens"})) {
        request.max_new_tokens = *max_tokens;
      }
      const bool stream = json_bool(body, "stream");
      if (stream) {
        handle_chat_stream(res, model_name, std::move(model), std::move(request));
      } else {
        const auto result = model->run(request);
        nlohmann::json message = {{"role", "assistant"}, {"content", result.text}};
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

  void handle_chat_stream(httplib::Response& res, std::string model_name,
                          std::shared_ptr<GenAIModel> model, GenerationRequest request) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider(
        "text/event-stream",
        [model_name = std::move(model_name), model = std::move(model),
         request = std::move(request)](std::size_t, httplib::DataSink& sink) mutable {
          try {
            auto stream = model->stream(request);
            for (auto sample = stream.next(); sample.has_value(); sample = stream.next()) {
              if (sample->is_final) {
                const auto final_chunk =
                    chat_chunk(model_name, "", choice_finish_reason(sample->finish_reason)) +
                    "data: [DONE]\n\n";
                write_sink(sink, final_chunk);
                sink.done();
                return true;
              }
              const auto chunk = chat_chunk(model_name, sample->text);
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
        set_error(res, "Unknown OpenAI model: " + model_name, 404);
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
        handle_audio_stream(res, std::move(model), audio_path, language);
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

  void handle_audio_stream(httplib::Response& res, std::shared_ptr<GenAIModel> model,
                           std::filesystem::path audio_path, std::string language) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_chunked_content_provider("text/event-stream", [model = std::move(model),
                                                           audio_path = std::move(audio_path),
                                                           language = std::move(language)](
                                                              std::size_t,
                                                              httplib::DataSink& sink) mutable {
      TempFileGuard guard{audio_path};
      try {
        GenerationRequest request;
        request.audio_file = audio_path;
        request.language = language;
        auto stream = model->stream(request);
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

  void serve() {
    if (running.exchange(true)) {
      throw std::runtime_error("OpenAIServer is already running");
    }
    const bool ok = http.listen(options.host, options.port);
    running.store(false);
    if (!ok && !stopping.load()) {
      throw std::runtime_error("OpenAIServer failed to listen on " + options.host + ":" +
                               std::to_string(options.port));
    }
  }

  void start() {
    if (worker.joinable()) {
      throw std::runtime_error("OpenAIServer is already started");
    }
    stopping.store(false);
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

  OpenAIServerOptions options;
  httplib::Server http;
  mutable std::mutex registry_mutex;
  std::map<std::string, Entry> registry;
  std::thread worker;
  std::atomic<bool> running = false;
  std::atomic<bool> stopping = false;
};

OpenAIServer::OpenAIServer(OpenAIServerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

OpenAIServer::~OpenAIServer() {
  if (impl_) {
    impl_->stop();
  }
}

OpenAIServer::OpenAIServer(OpenAIServer&&) noexcept = default;

OpenAIServer& OpenAIServer::operator=(OpenAIServer&&) noexcept = default;

std::string OpenAIServer::add_model(std::filesystem::path model_dir) {
  return impl_->add_model(std::move(model_dir));
}

std::string OpenAIServer::add_model(std::filesystem::path model_dir, std::string served_name) {
  return impl_->add_model(std::move(model_dir), std::move(served_name));
}

void OpenAIServer::add_model(std::string served_name, std::shared_ptr<GenAIModel> model) {
  impl_->add_model(std::move(served_name), std::move(model));
}

bool OpenAIServer::remove_model(const std::string& served_name) {
  return impl_->remove_model(served_name);
}

std::vector<std::string> OpenAIServer::model_names() const {
  return impl_->model_names();
}

void OpenAIServer::serve() {
  impl_->serve();
}

void OpenAIServer::start() {
  impl_->start();
}

void OpenAIServer::stop() {
  impl_->stop();
}

} // namespace simaai::neat::genai
