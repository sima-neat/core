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
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr const char* kTextModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kVlmModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";
constexpr const char* kAsrModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";
constexpr const char* kExpectedText = "The capital of Germany is Berlin.";
constexpr const char* kExpectedVlmText = "Skier in the air.";
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
  require(trim_text(text) == kExpectedVlmText, "server image chat returned unexpected text");
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
    request_text_completion(port);
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
