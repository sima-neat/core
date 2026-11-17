#include "genai/GenAIServer.h"
#include "genai_test_utils.h"
#include "test_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr const char* kTextModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kVlmModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kAsrModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";
constexpr const char* kExpectedText = "The capital of Germany is Berlin.";
constexpr const char* kExpectedAsrText = "tell me a joke please";
constexpr const char* kExpectedTranslation = "please tell me a joke";

std::string trim_text(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string normalize_transcript(const std::string& text) {
  std::string out;
  bool pending_space = false;
  for (const unsigned char ch : text) {
    if (std::isalnum(ch)) {
      if (pending_space && !out.empty()) {
        out.push_back(' ');
      }
      out.push_back(static_cast<char>(std::tolower(ch)));
      pending_space = false;
    } else {
      pending_space = true;
    }
  }
  return out;
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to read file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string base64_encode(const std::string& bytes) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((bytes.size() + 2U) / 3U) * 4U);
  std::uint32_t value = 0;
  int bits = -6;
  for (const unsigned char ch : bytes) {
    value = (value << 8U) | ch;
    bits += 8;
    while (bits >= 0) {
      out.push_back(kAlphabet[(value >> bits) & 0x3FU]);
      bits -= 6;
    }
  }
  if (bits > -6) {
    out.push_back(kAlphabet[((value << 8U) >> (bits + 8)) & 0x3FU]);
  }
  while (out.size() % 4U != 0U) {
    out.push_back('=');
  }
  return out;
}

fs::path fixture_path(const char* repo_root_arg, const fs::path& rel) {
  fs::path path = fs::path(repo_root_arg) / rel;
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    throw std::runtime_error("missing fixture: " + path.string());
  }
  return path;
}

int choose_free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    const std::string error = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("bind failed while choosing server port: " + error);
  }

  socklen_t len = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    const std::string error = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("getsockname failed while choosing server port: " + error);
  }
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

class ServerGuard {
public:
  explicit ServerGuard(simaai::neat::genai::GenAIServer& server_in) : server(server_in) {}
  ~ServerGuard() {
    server.stop();
  }

  ServerGuard(const ServerGuard&) = delete;
  ServerGuard& operator=(const ServerGuard&) = delete;

private:
  simaai::neat::genai::GenAIServer& server;
};

httplib::Client make_client(int port) {
  httplib::Client client("127.0.0.1", port);
  client.set_connection_timeout(5, 0);
  client.set_read_timeout(180, 0);
  client.set_write_timeout(30, 0);
  return client;
}

Json parse_response(const httplib::Result& result, const std::string& label) {
  if (!result) {
    throw std::runtime_error(label + " request failed: " + httplib::to_string(result.error()));
  }
  if (result->status != 200) {
    throw std::runtime_error(label + " returned HTTP " + std::to_string(result->status) + ": " +
                             result->body);
  }
  return Json::parse(result->body);
}

void wait_for_server(int port) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    auto client = make_client(port);
    if (auto result = client.Get("/v1/models"); result && result->status == 200) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("GenAIServer did not become ready on port " + std::to_string(port));
}

void require_model_list_contains(int port) {
  auto client = make_client(port);
  const Json body = parse_response(client.Get("/v1/models"), "GET /v1/models");
  std::vector<std::string> names;
  for (const auto& item : body.at("data")) {
    names.push_back(item.at("id").get<std::string>());
  }
  for (const char* expected : {"llm", "vlm", "asr"}) {
    require(std::find(names.begin(), names.end(), expected) != names.end(),
            std::string("GET /v1/models missing registered model: ") + expected);
  }
}

void request_text_completion(int port) {
  auto client = make_client(port);
  const Json request = {
      {"model", "llm"},
      {"messages",
       Json::array({{{"role", "user"}, {"content", "What is the capital of Germany?"}}})},
      {"max_tokens", 24},
      {"stream", false},
  };
  const Json body =
      parse_response(client.Post("/v1/chat/completions", request.dump(), "application/json"),
                     "POST /v1/chat/completions text");
  const std::string text = body.at("choices").at(0).at("message").at("content").get<std::string>();
  std::cout << "GENAI_SERVER_TEXT text=" << text << "\n";
  require(trim_text(text) == kExpectedText, "server text completion returned unexpected text");
}

// For requests the server must reject: checks the status and returns the error object,
// putting the status and body in the message when either is not what was expected.
Json parse_error_response(const httplib::Result& result, int expected_status,
                          const std::string& label) {
  require(static_cast<bool>(result),
          label + ": request failed" + (result ? "" : ": " + httplib::to_string(result.error())));
  const std::string seen = "HTTP " + std::to_string(result->status) + ": " + result->body;
  require(result->status == expected_status,
          label + ": expected HTTP " + std::to_string(expected_status) + ", got " + seen);
  const Json body = Json::parse(result->body, nullptr, false);
  require(!body.is_discarded() && body.contains("error") && body.at("error").is_object(),
          label + ": expected an error object, got " + seen);
  return body.at("error");
}

Json post_chat(int port, const Json& request, const std::string& label) {
  auto client = make_client(port);
  return parse_response(client.Post("/v1/chat/completions", request.dump(), "application/json"),
                        "POST /v1/chat/completions " + label);
}

// The fields of an OpenAI chat completion that clients read before the generated text.
void require_chat_completion_shape(const Json& body, const std::string& label) {
  require(body.at("object") == "chat.completion", label + ": object should be chat.completion");
  require(body.at("model") == "llm", label + ": model should echo the requested model");
  require(body.at("id").is_string() && !body.at("id").get<std::string>().empty(),
          label + ": id should be a non-empty string");
  require(body.at("created").is_number_integer(), label + ": created should be an integer");
  const Json& choice = body.at("choices").at(0);
  require(choice.at("index") == 0, label + ": first choice should have index 0");
  require(choice.at("message").at("role") == "assistant",
          label + ": message role should be assistant");
  require(choice.at("finish_reason").is_string(), label + ": finish_reason should be a string");
}

const Json& weather_and_time_tools() {
  static const Json tools = Json::array({
      {{"type", "function"},
       {"function",
        {{"name", "get_time"},
         {"description", "Get the current local time for a city"},
         {"parameters",
          {{"type", "object"},
           {"properties", {{"city", {{"type", "string"}}}}},
           {"required", Json::array({"city"})}}}}}},
      {{"type", "function"},
       {"function",
        {{"name", "get_weather"},
         {"description", "Get the current weather for a city"},
         {"parameters",
          {{"type", "object"},
           {"properties",
            {{"city", {{"type", "string"}}},
             {"unit", {{"type", "string"}, {"enum", Json::array({"celsius", "fahrenheit"})}}}}},
           {"required", Json::array({"city"})}}}}}},
  });
  return tools;
}

const Json* find_tool(const Json& tools, const std::string& name) {
  for (const Json& tool : tools) {
    if (tool.at("function").at("name") == name) {
      return &tool;
    }
  }
  return nullptr;
}

bool is_one_of(const Json& value, std::initializer_list<const char*> allowed) {
  for (const char* candidate : allowed) {
    if (value == candidate) {
      return true;
    }
  }
  return false;
}

// Which tool the model calls, if any, is up to the model. What the server guarantees is a
// consistent answer, checked here against the paths that build it:
// - chat_message_response sends content null with tool calls, and the text otherwise;
// - VisionLanguageModel sets finish_reason to tool_calls or stop, and length takes
//   precedence when the token limit is reached;
// - llima's try_parse_tool_calls keeps only declared tool names and object arguments,
//   and the server passes its id, type and arguments string through unchanged.
// A malformed call the parser can't read comes back as text and passes here; that path
// needs a fixed-output parser test in llima.
void require_tool_calls_or_text(const Json& choice, const Json& tools, const std::string& label) {
  const Json& message = choice.at("message");
  const bool has_calls = message.contains("tool_calls") && !message.at("tool_calls").empty();
  if (!has_calls) {
    require(is_one_of(choice.at("finish_reason"), {"stop", "length"}),
            label + ": without tool calls finish_reason should be stop or length");
    require(message.at("content").is_string(),
            label + ": without tool calls content should be a string");
    return;
  }
  require(is_one_of(choice.at("finish_reason"), {"tool_calls", "length"}),
          label + ": with tool calls finish_reason should be tool_calls or length");
  require(message.at("content").is_null(), label + ": with tool calls content should be null");
  for (const Json& call : message.at("tool_calls")) {
    require(call.at("type") == "function", label + ": tool call type should be function");
    require(call.at("id").is_string() && !call.at("id").get<std::string>().empty(),
            label + ": tool call id should be a non-empty string");
    const std::string name = call.at("function").at("name").get<std::string>();
    require(find_tool(tools, name) != nullptr,
            label + ": tool call names an undeclared tool: " + name);
    // OpenAI sends the arguments as a JSON string, not as an object.
    require(call.at("function").at("arguments").is_string(),
            label + ": tool call arguments should be a JSON string");
    const Json arguments =
        Json::parse(call.at("function").at("arguments").get<std::string>(), nullptr, false);
    require(arguments.is_object(), label + ": tool call arguments should parse to an object");
  }
}

// Integration scenario pinned to the CI text model (kDefaultTextModelName). The server
// ignores sampling parameters and llima decodes greedily, so the answer is stable for a
// given model and llima build: the weather question must come back as get_weather for Paris.
// get_weather is declared second, so a server that kept only the first tool would fail. The
// consistency checks below accept a text answer, so without this one a server that stopped
// passing tools to the model, or a parser that stopped reading calls, would still pass.
// Revisit the expectation when the CI text model or llima's decoding changes. main() runs
// this only when the default text model is selected.
void request_pinned_weather_tool_call(int port) {
  const std::string label = "pinned weather tool call";
  const Json body = post_chat(
      port,
      {{"model", "llm"},
       {"stream", false},
       {"max_tokens", 96},
       {"tools", weather_and_time_tools()},
       {"messages",
        Json::array({{{"role", "user"}, {"content", "What's the weather in Paris in celsius?"}}})}},
      label);
  require_chat_completion_shape(body, label);
  const Json& choice = body.at("choices").at(0);
  std::cout << "GENAI_SERVER_PINNED_TOOL_CALL message=" << choice.at("message").dump() << "\n";
  require_tool_calls_or_text(choice, weather_and_time_tools(), label);
  const Json& message = choice.at("message");
  require(message.contains("tool_calls") && !message.at("tool_calls").empty(),
          label + ": the CI model should answer with a tool call");
  bool called_weather_for_paris = false;
  for (const Json& call : message.at("tool_calls")) {
    if (call.at("function").at("name") != "get_weather") {
      continue;
    }
    const Json arguments =
        Json::parse(call.at("function").at("arguments").get<std::string>(), nullptr, false);
    require(arguments.is_object() && arguments.contains("city") && arguments.at("city").is_string(),
            label + ": get_weather should have a string city argument, got " + call.dump());
    std::string city = arguments.at("city").get<std::string>();
    for (char& ch : city) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    called_weather_for_paris = called_weather_for_paris || city == "paris";
  }
  require(called_weather_for_paris, label +
                                        ": the CI model should call get_weather for Paris, got " +
                                        message.at("tool_calls").dump());
}

void request_with_tools(int port) {
  // A time question and a question that needs no tool; the weather question is the pinned
  // scenario above. The model may or may not call a tool for each; every answer must be
  // consistent with the declared tools.
  for (const char* question :
       {"What time is it in Paris right now?", "Say hello in French, one word."}) {
    const std::string label = std::string("tools: ") + question;
    const Json body =
        post_chat(port,
                  {{"model", "llm"},
                   {"stream", false},
                   {"max_tokens", 96},
                   {"tools", weather_and_time_tools()},
                   {"messages", Json::array({{{"role", "user"}, {"content", question}}})}},
                  label);
    require_chat_completion_shape(body, label);
    const Json& choice = body.at("choices").at(0);
    std::cout << "GENAI_SERVER_TOOLS question=" << question
              << " finish_reason=" << choice.at("finish_reason").get<std::string>()
              << " message=" << choice.at("message").dump() << "\n";
    require_tool_calls_or_text(choice, weather_and_time_tools(), label);
  }
}

void request_tool_choice_none(int port) {
  // With tool_choice "none" the server does not parse tool calls (tool_calls_enabled is
  // false), so the answer can't contain any. This does not show whether the tool
  // descriptions reached the model.
  const std::string label = "tool_choice none";
  const Json body = post_chat(
      port,
      {{"model", "llm"},
       {"stream", false},
       {"max_tokens", 96},
       {"tools", weather_and_time_tools()},
       {"tool_choice", "none"},
       {"messages",
        Json::array({{{"role", "user"}, {"content", "What's the weather in Paris in celsius?"}}})}},
      label);
  require_chat_completion_shape(body, label);
  const Json& choice = body.at("choices").at(0);
  require(!choice.at("message").contains("tool_calls"),
          label + ": the answer should not contain tool calls");
  require(choice.at("message").at("content").is_string(), label + ": content should be a string");
  require(is_one_of(choice.at("finish_reason"), {"stop", "length"}),
          label + ": finish_reason should be stop or length");
}

void reject_invalid_tool_options(int port) {
  // apply_tool_options validates these before generation and answers 400.
  const std::vector<std::pair<Json, std::string>> requests = {
      {{{"model", "llm"},
        {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
        {"tools", "get_weather"}},
       "tools must be an array"},
      {{{"model", "llm"},
        {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
        {"tools", weather_and_time_tools()},
        {"tool_choice", "required"}},
       "Only tool_choice 'auto' or 'none' is supported"},
  };
  for (const auto& [request, expected_message] : requests) {
    const std::string label = "invalid tool options (" + expected_message + ")";
    auto client = make_client(port);
    const Json error = parse_error_response(
        client.Post("/v1/chat/completions", request.dump(), "application/json"), 400, label);
    require(error.value("type", Json()) == "invalid_request_error",
            label + ": error type should be invalid_request_error, got " + error.dump());
    require(error.value("message", Json()) == expected_message,
            label + ": unexpected error message, got " + error.dump());
  }
}

void request_multi_turn(int port) {
  // The server must accept a history with an assistant turn and answer it like any request.
  // This checks that the request is accepted and answered, not what the model remembers.
  // The non-empty answer depends on the model; the server does not guarantee it.
  const Json body = post_chat(
      port,
      {{"model", "llm"},
       {"stream", false},
       {"max_tokens", 96},
       {"messages", Json::array({{{"role", "user"}, {"content", "My name is Alice. Remember it."}},
                                 {{"role", "assistant"}, {"content", "Got it, Alice."}},
                                 {{"role", "user"},
                                  {"content", "What is my name? Reply with the name only."}}})}},
      "multi-turn");
  require_chat_completion_shape(body, "multi-turn");
  const std::string text = body.at("choices").at(0).at("message").at("content").get<std::string>();
  std::cout << "GENAI_SERVER_MULTI_TURN text=" << text << "\n";
  require(!trim_text(text).empty(), "multi-turn: expected a text answer");
}

void reject_unknown_model(int port) {
  auto client = make_client(port);
  const Json request = {{"model", "no-such-model"},
                        {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})}};
  const Json error =
      parse_error_response(client.Post("/v1/chat/completions", request.dump(), "application/json"),
                           404, "unknown model");
  require(error.value("type", Json()) == "invalid_request_error",
          "unknown model: error type should be invalid_request_error, got " + error.dump());
  require(error.value("message", Json()).is_string(),
          "unknown model: error message should be a string, got " + error.dump());
  require_contains(error.at("message").get<std::string>(), "no-such-model",
                   "unknown model: error message should name the model");
}

void request_streamed_chunks(int port) {
  auto client = make_client(port);
  const Json request = {
      {"model", "llm"},
      {"stream", true},
      {"max_tokens", 8},
      {"messages",
       Json::array({{{"role", "user"}, {"content", "What is the capital of Germany?"}}})}};
  const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
  require(static_cast<bool>(response),
          "streamed chat completion request failed: " + httplib::to_string(response.error()));
  require(response->status == 200, "streamed chat completion returned HTTP " +
                                       std::to_string(response->status) + ": " + response->body);
  std::vector<Json> chunks;
  bool done = false;
  std::size_t start = 0;
  const std::string& body = response->body;
  while (start < body.size()) {
    std::size_t end = body.find('\n', start);
    if (end == std::string::npos) {
      end = body.size();
    }
    const std::string line = trim_text(body.substr(start, end - start));
    start = end + 1;
    if (line.rfind("data:", 0) != 0) {
      continue;
    }
    const std::string payload = trim_text(line.substr(5));
    if (payload == "[DONE]") {
      done = true;
      break;
    }
    Json chunk = Json::parse(payload, nullptr, false);
    require(!chunk.is_discarded(), "stream chunk is not valid JSON: " + payload);
    // A generation error arrives as its own chunk under HTTP 200.
    require(!chunk.contains("error"), "stream returned an error chunk: " + payload);
    chunks.push_back(std::move(chunk));
  }
  require(done, "stream should end with data: [DONE]");
  require(!chunks.empty(), "stream should send at least one chunk before [DONE]");
  const std::string id = chunks.front().at("id").get<std::string>();
  std::string streamed_text;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    const Json& chunk = chunks.at(i);
    require(chunk.at("object") == "chat.completion.chunk",
            "stream chunks should have object chat.completion.chunk");
    require(chunk.at("id") == id, "every chunk of one stream should carry the same id");
    // chat_chunk sets finish_reason null on content chunks and only the final chunk ends it.
    const Json& finish_reason = chunk.at("choices").at(0).at("finish_reason");
    const Json& delta = chunk.at("choices").at(0).value("delta", Json::object());
    if (delta.contains("content") && delta.at("content").is_string()) {
      streamed_text += delta.at("content").get<std::string>();
    }
    if (i + 1 < chunks.size()) {
      require(finish_reason.is_null(), "only the last stream chunk should carry finish_reason");
    } else {
      require(is_one_of(finish_reason, {"stop", "length"}),
              "the last stream chunk should finish with stop or length");
    }
  }
  std::cout << "GENAI_SERVER_STREAM text=" << streamed_text << "\n";
  // Non-empty text depends on the model; the server only forwards what the model emits.
  require(!trim_text(streamed_text).empty(),
          "the stream should carry generated text, got body: " + body);
}

void check_context_limits(int port, const fs::path& model_dir) {
  const auto config = Json::parse(read_file(model_dir / "devkit/vlm_config.json"));
  const auto window = config.at("pipeline_cfg").at("max_num_tokens").get<std::size_t>();
  std::string oversized;
  for (std::size_t i = 0; i <= window; ++i) {
    oversized += " lorem ipsum";
  }
  for (const bool stream : {false, true}) {
    auto client = make_client(port);
    Json request = {{"model", "llm"},
                    {"max_tokens", 1},
                    {"stream", stream},
                    {"messages", Json::array({{{"role", "user"}, {"content", oversized}}})}};
    const auto rejected = client.Post("/v1/chat/completions", request.dump(), "application/json");
    require(rejected != nullptr, "context-limit request failed");
    require(rejected->status == (stream ? 200 : 400), "wrong context-limit HTTP status");
    if (stream) {
      require_contains(rejected->body, "\"code\":\"context_length_exceeded\"",
                       "stream omitted context-limit error");
      require_contains(rejected->body, "data: [DONE]", "context error did not finish stream");
    } else {
      const auto error = Json::parse(rejected->body).at("error");
      require(error.at("code") == "context_length_exceeded", "wrong context-limit code");
      require(error.at("type") == "invalid_request_error", "wrong context-limit error type");
    }

    request["messages"][0]["content"] = "What is the capital of Germany?";
    const auto limited = client.Post("/v1/chat/completions", request.dump(), "application/json");
    require(limited && limited->status == 200, "generation after context error failed");
    require_contains(limited->body, "\"finish_reason\":\"length\"",
                     "token limit was reported as a normal stop");
  }
}

void reject_unsupported_thinking(int port) {
  const std::vector<std::pair<std::string, Json>> requests = {
      {"/v1/chat/completions",
       {{"model", "llm"},
        {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
        {"enable_thinking", true},
        {"stream", true}}},
      {"/v1/completions",
       {{"model", "llm"}, {"prompt", "Hello"}, {"enable_thinking", true}, {"stream", true}}},
      {"/api/chat",
       {{"model", "llm"},
        {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
        {"think", true},
        {"stream", true}}},
      {"/api/generate", {{"model", "llm"}, {"prompt", "Hello"}, {"think", true}, {"stream", true}}},
  };

  for (const auto& [path, request] : requests) {
    auto client = make_client(port);
    const auto response = client.Post(path, request.dump(), "application/json");
    require(response != nullptr, path + " unsupported-thinking request failed");
    require(response->status == 400, path + " should reject unsupported thinking with HTTP 400");
    const Json body = Json::parse(response->body);
    require(body.at("error").at("message") == "Thinking is not supported for this model",
            path + " returned an unexpected unsupported-thinking error");
  }
}

void validate_lora_requests(int port) {
  const std::vector<std::pair<std::string, Json>> bad_requests = {
      {"/set_lora", {{"model", "llm"}, {"name", "../adapter"}}},
      {"/set_lora", {{"model", "llm"}}},
      {"/unset_lora", Json::object()},
      {"/unset_lora", {{"model", "asr"}}},
  };
  for (const auto& [path, request] : bad_requests) {
    auto client = make_client(port);
    const auto response = client.Post(path, request.dump(), "application/json");
    require(response != nullptr, path + " validation request failed");
    require(response->status == 400, path + " should reject invalid LoRA input with HTTP 400");
  }

  auto client = make_client(port);
  const auto unknown = client.Post("/unset_lora", R"({"model":"unknown"})", "application/json");
  require(unknown != nullptr, "/unset_lora unknown-model request failed");
  require(unknown->status == 404, "/unset_lora should reject an unknown model with HTTP 404");
}

void request_image_completion(int port, const fs::path& image_path) {
  auto client = make_client(port);
  const std::string data_uri = "data:image/jpeg;base64," + base64_encode(read_file(image_path));
  const Json request = {
      {"model", "vlm"},
      {"messages",
       Json::array(
           {{{"role", "user"},
             {"content",
              Json::array({{{"type", "text"}, {"text", "Describe this image in a short phrase."}},
                           {{"type", "image_url"}, {"image_url", {{"url", data_uri}}}}})}}})},
      {"max_tokens", 48},
      {"stream", false},
  };
  const Json body =
      parse_response(client.Post("/v1/chat/completions", request.dump(), "application/json"),
                     "POST /v1/chat/completions image");
  const std::string text = body.at("choices").at(0).at("message").at("content").get<std::string>();
  std::cout << "GENAI_SERVER_VLM text=" << text << "\n";
  require(!trim_text(text).empty(), "server image chat expected non-empty generated text");
}

void request_audio(int port, const fs::path& audio_path, const std::string& endpoint,
                   const std::string& task, const std::string& expected_text,
                   const std::string& expected_language) {
  auto client = make_client(port);
  httplib::MultipartFormDataItems items = {
      {"model", "asr", "", ""},
      {"file", read_file(audio_path), audio_path.filename().string(), "audio/wav"},
  };
  const Json body = parse_response(client.Post(endpoint, items), "POST " + endpoint);
  const std::string text = body.at("text").get<std::string>();
  std::cout << "GENAI_SERVER_ASR text=" << text << "\n";
  require(normalize_transcript(text) == expected_text,
          "server audio " + task + " returned unexpected text");
  require(body.at("task") == task, "server audio response should report task=" + task);
  require(body.at("language") == expected_language,
          "server audio request should report detected language " + expected_language);
  const double no_speech_prob = body.at("no_speech_prob").get<double>();
  require(std::isfinite(no_speech_prob), "server audio no_speech_prob should be finite");
  require(no_speech_prob >= 0.0 && no_speech_prob <= 1.0,
          "server audio no_speech_prob should be within [0, 1]");
  require(std::isfinite(body.at("avg_logprob").get<double>()),
          "server audio avg_logprob should be finite");
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("genai_server_http_test requires repository root argument");
    }

    const fs::path text_model_dir = simaai::neat::test::resolve_genai_model_dir(
        kTextModelEnv, simaai::neat::test::kDefaultTextModelName, "LLiMa text",
        "devkit/vlm_config.json");
    const fs::path vlm_model_dir = simaai::neat::test::resolve_genai_model_dir(
        kVlmModelEnv, simaai::neat::test::kDefaultVlmModelName, "LLiMa VLM",
        "devkit/vlm_config.json");
    const fs::path asr_model_dir = simaai::neat::test::resolve_genai_model_dir(
        kAsrModelEnv, simaai::neat::test::kDefaultAsrModelName, "LLiMa ASR",
        "devkit/whisper_config.json");
    const fs::path image_path = fixture_path(argv[1], "tests/images/people.jpg");
    const fs::path audio_path = fixture_path(argv[1], "tests/assets/genai/audio.wav");
    const fs::path german_audio_path = fixture_path(argv[1], "tests/assets/genai/audio_de.wav");
    const int port = choose_free_port();

    std::cout << "GENAI_SERVER text_model_dir=" << text_model_dir << "\n";
    std::cout << "GENAI_SERVER vlm_model_dir=" << vlm_model_dir << "\n";
    std::cout << "GENAI_SERVER asr_model_dir=" << asr_model_dir << "\n";
    std::cout << "GENAI_SERVER port=" << port << "\n";

    simaai::neat::genai::GenAIServerOptions options;
    options.host = "127.0.0.1";
    options.port = port;
    simaai::neat::genai::GenAIServer server(options);
    ServerGuard guard(server);
    server.add_model(text_model_dir, "llm");
    server.add_model(vlm_model_dir, "vlm");
    server.add_model(asr_model_dir, "asr");
    server.start();
    wait_for_server(port);

    require_model_list_contains(port);
    reject_unsupported_thinking(port);
    reject_unknown_model(port);
    reject_invalid_tool_options(port);
    validate_lora_requests(port);
    request_text_completion(port);
    request_streamed_chunks(port);
    // The pinned scenario expects the answer of the default text model. With another model
    // selected through SIMA_TEST_LLIMA_TEXT_MODEL the scenario is not run; the remaining new
    // checks need only a model that answers with text.
    if (text_model_dir.filename() == simaai::neat::test::kDefaultTextModelName) {
      request_pinned_weather_tool_call(port);
    } else {
      std::cout << "GENAI_SERVER_PINNED_TOOL_CALL not run: text model "
                << text_model_dir.filename().string() << " is not the default\n";
    }
    request_with_tools(port);
    request_tool_choice_none(port);
    request_multi_turn(port);
    check_context_limits(port, text_model_dir);
    request_image_completion(port, image_path);
    request_audio(port, audio_path, "/v1/audio/transcriptions", "transcribe", kExpectedAsrText,
                  "en");
    request_audio(port, german_audio_path, "/v1/audio/translations", "translate",
                  kExpectedTranslation, "de");

    server.stop();
    std::cout << "[OK] genai_server_http_test passed\n";
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
