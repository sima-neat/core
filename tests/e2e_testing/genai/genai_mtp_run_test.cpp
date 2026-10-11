#include "genai/GenAIServer.h"
#include "genai/VisionLanguageModel.h"
#include "test_utils.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

// Manual regression until a compiled MTP pair is included in the CI fixtures.
// Usage: genai_mtp_run_test MODEL_PACKAGE PORT [IMAGE_FILE]
int main(int argc, char** argv) {
  using namespace simaai::neat::genai;
  using Json = nlohmann::json;
  try {
    require(argc == 3 || argc == 4, "Usage: genai_mtp_run_test MODEL_PACKAGE PORT [IMAGE_FILE]");
    const int port = std::stoi(argv[2]);
    require(port > 0 && port <= 65535, "PORT must be between 1 and 65535");
    GenerationRequest request;
    request.prompt = "What is the capital of Germany?";
    request.max_new_tokens = 32;
    if (argc == 4) {
      auto image = cv::imread(argv[3]);
      require(!image.empty(), "Could not read IMAGE_FILE");
      request.images = {image};
      request.prompt = "Describe this image in a short phrase.";
    }

    std::string reference;
    {
      VisionLanguageModel model(argv[1]);
      const auto result = model.run(request);
      require(!result.text.empty() && result.metrics.generated_tokens > 0 &&
                  result.metrics.generated_tokens <= request.max_new_tokens,
              "Direct MTP generation produced no output or exceeded its token limit");
      reference = result.text;
      auto stream = model.stream(request);
      std::string streamed;
      bool finished = false;
      while (auto sample = stream.next()) {
        streamed += sample->text;
        if (sample->is_final) {
          finished = true;
          require(sample->metrics.generated_tokens == result.metrics.generated_tokens,
                  "Direct MTP stream changed the token count");
        }
      }
      require(finished && streamed == reference, "Direct MTP stream changed its output");
      auto short_request = request;
      short_request.max_new_tokens = 1;
      require(model.run(short_request).metrics.generated_tokens == 1,
              "MTP dispatch did not support a one-token request");
    }

    GenAIServer server({"127.0.0.1", static_cast<std::uint16_t>(port)});
    server.add_model(argv[1], "mtp");
    server.start();
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(1, 0);
    client.set_read_timeout(180, 0);
    bool ready = false;
    for (int attempt = 0; attempt < 100 && !ready; ++attempt) {
      const auto response = client.Get("/v1/models");
      ready = response && response->status == 200;
      if (!ready)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    require(ready, "GenAIServer did not become ready");
    // Text HTTP requests also exercise VLM targets without an attached image.
    const Json body = {
        {"model", "mtp"},
        {"messages",
         Json::array({{{"role", "user"}, {"content", "What is the capital of Germany?"}}})},
        {"max_tokens", 32},
        {"stream", false},
    };
    auto response = client.Post("/v1/chat/completions", body.dump(), "application/json");
    require(response && response->status == 200, "Non-streaming HTTP MTP request failed: " +
                                                     (response ? response->body : "no response"));
    const auto result = Json::parse(response->body);
    const auto text = result.at("choices").at(0).at("message").at("content").get<std::string>();
    require(!text.empty(), "HTTP MTP generation produced no output");
    if (argc == 3)
      require(text == reference, "HTTP MTP generation differs from the direct API");

    auto streaming_body = body;
    streaming_body["stream"] = true;
    response = client.Post("/v1/chat/completions", streaming_body.dump(), "application/json");
    require(response && response->status == 200, "Streaming HTTP MTP request failed");
    std::istringstream events(response->body);
    std::string line, streamed;
    bool done = false;
    while (std::getline(events, line)) {
      if (!line.starts_with("data: "))
        continue;
      const auto payload = line.substr(6);
      if (payload == "[DONE]") {
        done = true;
        continue;
      }
      const auto delta = Json::parse(payload).at("choices").at(0).at("delta");
      if (delta.contains("content") && delta.at("content").is_string())
        streamed += delta.at("content").get<std::string>();
    }
    require(done && streamed == text, "HTTP MTP stream changed its output or omitted [DONE]");
    server.stop();
    std::cout << "[OK] MTP direct generation, streaming, token limit and HTTP tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << '\n';
    return 1;
  }
}
