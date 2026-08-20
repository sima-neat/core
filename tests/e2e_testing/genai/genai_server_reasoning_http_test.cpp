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
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr const char* kQwenModelEnv = "SIMA_TEST_LLIMA_REASONING_QWEN_MODEL";
constexpr const char* kQwenModel = "Qwen3-0.6B-Autoround-a16w4";
constexpr const char* kGemmaModelEnv = "SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL";
constexpr const char* kGemmaModel = "Gemma-4-E2B-it-TextOnly-GPTQ-a16w4";
constexpr const char* kQuery = "Solve x + 7 = 12. Think briefly, then give x.";

int choose_free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("failed to create socket");
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(fd);
    throw std::runtime_error("failed to bind free-port socket");
  }
  socklen_t length = sizeof(address);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    ::close(fd);
    throw std::runtime_error("failed to inspect free-port socket");
  }
  const int port = ntohs(address.sin_port);
  ::close(fd);
  return port;
}

void wait_for_server(int port) {
  for (int attempt = 0; attempt < 600; ++attempt) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(0, 100000);
    if (const auto response = client.Get("/v1/models"); response && response->status == 200) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("GenAIServer did not become ready");
}

Json post_json(int port, const std::string& path, const Json& request) {
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(180, 0);
  const auto response = client.Post(path, request.dump(), "application/json");
  require(response != nullptr, path + " returned no response");
  require(response->status == 200, path + " returned HTTP " + std::to_string(response->status));
  return Json::parse(response->body);
}

std::string post_text(int port, const std::string& path, const Json& request) {
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(180, 0);
  const auto response = client.Post(path, request.dump(), "application/json");
  require(response != nullptr, path + " returned no response");
  require(response->status == 200, path + " returned HTTP " + std::to_string(response->status));
  return response->body;
}

void require_no_markers(const std::string& text) {
  for (const char* marker : {"<think>", "</think>", "<|channel>", "<channel|>", "<tool_call>",
                             "</tool_call>", "<|tool_call>", "<tool_call|>"}) {
    require(text.find(marker) == std::string::npos,
            std::string("response leaked structural marker: ") + marker);
  }
}

void require_reasoning_answer(const std::string& reasoning, const std::string& content) {
  require(!reasoning.empty(), "reasoning output is empty");
  require(content.find('5') != std::string::npos, "final answer does not contain 5");
  require_no_markers(reasoning + content);
}

void require_openai_stream(int port) {
  const std::string body =
      post_text(port, "/v1/chat/completions",
                {{"model", "reasoning"},
                 {"messages", Json::array({{{"role", "user"}, {"content", kQuery}}})},
                 {"chat_template_kwargs", {{"enable_thinking", true}}},
                 {"stream", true}});
  std::string reasoning;
  std::string content;
  bool saw_content = false;
  bool saw_done = false;
  for (std::size_t start = 0; start < body.size();) {
    const auto end = body.find('\n', start);
    const std::string line = body.substr(start, end - start);
    start = end == std::string::npos ? body.size() : end + 1U;
    if (!line.starts_with("data: ")) {
      continue;
    }
    const std::string payload = line.substr(6);
    if (payload == "[DONE]") {
      saw_done = true;
      continue;
    }
    const Json chunk = Json::parse(payload);
    const auto& delta = chunk.at("choices").at(0).at("delta");
    if (delta.contains("reasoning_content")) {
      require(!saw_content, "OpenAI reasoning arrived after final content");
      reasoning += delta.at("reasoning_content").get<std::string>();
    }
    if (delta.contains("content") && delta.at("content").is_string()) {
      saw_content = true;
      content += delta.at("content").get<std::string>();
    }
  }
  require(saw_done, "OpenAI reasoning stream omitted [DONE]");
  require_reasoning_answer(reasoning, content);
}

void require_ollama_stream(int port, const std::string& path, Json request) {
  request["stream"] = true;
  const std::string body = post_text(port, path, request);
  std::string reasoning;
  std::string content;
  bool saw_content = false;
  bool saw_done = false;
  for (std::size_t start = 0; start < body.size();) {
    const auto end = body.find('\n', start);
    const std::string line = body.substr(start, end - start);
    start = end == std::string::npos ? body.size() : end + 1U;
    if (line.empty()) {
      continue;
    }
    const Json chunk = Json::parse(line);
    saw_done = saw_done || chunk.value("done", false);
    const Json& value = path == "/api/chat" ? chunk.at("message") : chunk;
    if (value.contains("thinking")) {
      require(!saw_content, "Ollama reasoning arrived after final content");
      reasoning += value.at("thinking").get<std::string>();
    }
    const char* content_key = path == "/api/chat" ? "content" : "response";
    if (value.contains(content_key) && value.at(content_key).is_string() &&
        !value.at(content_key).get<std::string>().empty()) {
      saw_content = true;
      content += value.at(content_key).get<std::string>();
    }
  }
  require(saw_done, "Ollama reasoning stream omitted final chunk");
  require_reasoning_answer(reasoning, content);
}

void exercise_model(const std::filesystem::path& model_dir) {
  const int port = choose_free_port();
  simaai::neat::genai::GenAIServerOptions options;
  options.host = "127.0.0.1";
  options.port = port;
  simaai::neat::genai::GenAIServer server(options);
  server.add_model(model_dir, "reasoning");
  server.start();
  wait_for_server(port);

  const Json openai_request = {
      {"model", "reasoning"},
      {"messages", Json::array({{{"role", "user"}, {"content", kQuery}}})},
      {"enable_thinking", true},
      {"stream", false},
  };
  const Json openai = post_json(port, "/v1/chat/completions", openai_request);
  const auto& openai_message = openai.at("choices").at(0).at("message");
  require_reasoning_answer(openai_message.at("reasoning_content").get<std::string>(),
                           openai_message.at("content").get<std::string>());
  require_openai_stream(port);

  Json disabled_request = openai_request;
  disabled_request["enable_thinking"] = false;
  const Json disabled = post_json(port, "/v1/chat/completions", disabled_request);
  const auto& disabled_message = disabled.at("choices").at(0).at("message");
  require(!disabled_message.contains("reasoning_content"),
          "thinking-disabled response included reasoning_content");
  require(disabled_message.at("content").get<std::string>().find('5') != std::string::npos,
          "thinking-disabled final answer does not contain 5");
  require_no_markers(disabled.dump());

  const Json ollama_chat =
      post_json(port, "/api/chat",
                {{"model", "reasoning"},
                 {"messages", Json::array({{{"role", "user"}, {"content", kQuery}}})},
                 {"think", true},
                 {"stream", false}});
  const auto& ollama_message = ollama_chat.at("message");
  require_reasoning_answer(ollama_message.at("thinking").get<std::string>(),
                           ollama_message.at("content").get<std::string>());
  require_ollama_stream(port, "/api/chat",
                        {{"model", "reasoning"},
                         {"messages", Json::array({{{"role", "user"}, {"content", kQuery}}})},
                         {"think", true}});

  const Json ollama_generate =
      post_json(port, "/api/generate",
                {{"model", "reasoning"}, {"prompt", kQuery}, {"think", true}, {"stream", false}});
  require_reasoning_answer(ollama_generate.at("thinking").get<std::string>(),
                           ollama_generate.at("response").get<std::string>());
  require_ollama_stream(port, "/api/generate",
                        {{"model", "reasoning"}, {"prompt", kQuery}, {"think", true}});

  const Json tool_request = {
      {"model", "reasoning"},
      {"messages",
       Json::array(
           {{{"role", "user"},
             {"content", "Call get_temperature once for Berlin. Do not answer directly."}}})},
      {"enable_thinking", true},
      {"tools", Json::array({{{"type", "function"},
                              {"function",
                               {{"name", "get_temperature"},
                                {"description", "Get the current outside temperature for a city."},
                                {"parameters",
                                 {{"type", "object"},
                                  {"properties", {{"city", {{"type", "string"}}}}},
                                  {"required", Json::array({"city"})}}}}}}})},
      {"tool_choice", "auto"},
      {"stream", false},
  };
  const Json tool_response = post_json(port, "/v1/chat/completions", tool_request);
  const auto& choice = tool_response.at("choices").at(0);
  const auto& tool_message = choice.at("message");
  require(choice.at("finish_reason") == "tool_calls", "tool call finish reason mismatch");
  require(!tool_message.at("reasoning_content").get<std::string>().empty(),
          "tool-call reasoning output is empty");
  require(tool_message.at("content").is_null(), "tool-call content must be null");
  require(tool_message.at("tool_calls").size() == 1U, "expected one tool call");
  const auto& function = tool_message.at("tool_calls").at(0).at("function");
  require(function.at("name") == "get_temperature", "unexpected tool name");
  require(Json::parse(function.at("arguments").get<std::string>()).at("city") == "Berlin",
          "unexpected tool arguments");
  require_no_markers(tool_response.dump());

  server.stop();
}

} // namespace

int main() {
  try {
    const auto qwen = simaai::neat::test::resolve_genai_model_dir(
        kQwenModelEnv, kQwenModel, "Qwen reasoning", "devkit/vlm_config.json");
    const auto gemma = simaai::neat::test::resolve_genai_model_dir(
        kGemmaModelEnv, kGemmaModel, "Gemma reasoning", "devkit/vlm_config.json");
    exercise_model(qwen);
    exercise_model(gemma);
    std::cout << "[OK] genai_server_reasoning_http_test passed\n";
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
