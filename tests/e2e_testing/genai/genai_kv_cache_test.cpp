#include "genai/GenAIModel.h"
#include "genai/GenAIServer.h"
#include "genai_test_utils.h"
#include "test_utils.h"

#include <httplib.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using namespace simaai::neat::genai;

namespace {
int free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  require(fd >= 0, "Could not create port-probe socket");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    throw std::runtime_error("Could not bind port-probe socket");
  }
  socklen_t size = sizeof(addr);
  const auto status = ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &size);
  ::close(fd);
  require(status == 0, "Could not read port-probe socket");
  return ntohs(addr.sin_port);
}

void check_http(const std::shared_ptr<GenAIModel>& model) {
  const auto port = static_cast<uint16_t>(free_port());
  GenAIServer server({"127.0.0.1", port});
  server.add_model("llm", model);
  server.add_model("alias", model);
  server.start();
  httplib::Client client("127.0.0.1", port);
  client.set_read_timeout(120, 0);
  for (int i = 0; i < 100; ++i) {
    if (const auto ready = client.Get("/v1/models"); ready && ready->status == 200)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  require(server.kv_cache_count("llm") == 0, "Server warmup retained an assigned slot");
  auto post = [&](const std::string& route, const Json& body, int status) {
    const auto response = client.Post(route, body.dump(), "application/json");
    require(response && response->status == status, "Unexpected HTTP status for " + route);
    return Json::parse(response->body);
  };
  Json request = {{"model", "llm"}, {"cache_id", "a"}, {"prompt", "Say hello."}, {"max_tokens", 8}};
  auto cold = post("/v1/completions", request, 200);
  require(cold.at("cache_created") == true, "HTTP cold context was not created");
  require(cold.at("usage").at("cached_prompt_tokens") == 0, "HTTP cold context reused tokens");
  request["model"] = "alias";
  auto warm = post("/v1/completions", request, 200);
  require(warm.at("cache_created") == false, "Aliases did not share the cache pool");
  require(warm.at("usage").at("cached_prompt_tokens").get<unsigned>() > 0,
          "HTTP warm context did not reuse tokens");
  request["cache_id"] = "b";
  post("/v1/completions", request, 200);
  request["cache_id"] = "c";
  require(post("/v1/completions", request, 429).at("error").at("type") == "cache_capacity_error",
          "HTTP capacity error lost its type");
  request["stream"] = true;
  auto response = client.Post("/v1/completions", request.dump(), "application/json");
  require(response && response->status == 200 &&
              response->body.find("cache_capacity_error") != std::string::npos &&
              response->body.find("[DONE]") != std::string::npos,
          "SSE capacity failure did not finish with a typed error");
  post("/remove_cache", {{"model", "llm"}, {"cache_id", "a"}}, 200);
  require(server.kv_cache_count("alias") == 1, "HTTP removal did not release a slot");
  require(post("/remove_cache", {{"model", "llm"}, {"cache_id", "a"}}, 200).at("removed") == false,
          "Removal should be idempotent");
  request["cache_id"] = "";
  post("/v1/completions", request, 400);
  post("/clear_caches", {{"model", "alias"}}, 200);
  require(server.kv_cache_count("llm") == 0, "HTTP clear retained cache associations");
  server.stop();
  GenerationRequest existing;
  existing.prompt = "Say hello.";
  existing.max_new_tokens = 8;
  existing.cache_id = "a";
  model->run(existing);
  existing.cache_id = "b";
  model->run(existing);
  server.add_model("llm", model);
  server.start();
  require(server.kv_cache_count("llm") == 2, "Startup changed pre-existing associations");
  require(model->run(existing).metrics.cached_prompt_tokens > 0,
          "Startup invalidated a pre-existing prompt context");
  server.stop();
}
} // namespace

int main() {
  try {
    const auto path = simaai::neat::test::resolve_genai_model_dir(
        "SIMA_TEST_LLIMA_TEXT_MODEL", simaai::neat::test::kDefaultTextModelName, "LLiMa text",
        "devkit/vlm_config.json");
    auto model = std::make_shared<GenAIModel>(path, GenAIModelOptions{2});
    GenerationRequest a;
    a.prompt = "What is the capital of Germany?";
    a.max_new_tokens = 24;
    a.cache_id = "a";
    auto b = a;
    b.prompt = "What is the capital of France?";
    b.cache_id = "b";
    const auto gold_a = model->run(a);
    const auto gold_b = model->run(b);
    require(gold_a.metrics.cache_created && gold_b.metrics.cache_created,
            "New public API contexts were not created");
    require(!gold_a.text.empty() && !gold_b.text.empty(), "Expected generated text");
    const auto warm_a = model->run(a);
    require(warm_a.text == gold_a.text && warm_a.metrics.cached_prompt_tokens > 0 &&
                !warm_a.metrics.cache_created,
            "A/B/A did not retain A");
    auto stream = model->stream(b);
    std::string text;
    bool final = false;
    while (const auto sample = stream.next()) {
      text += sample->text;
      if (sample->is_final) {
        final = true;
        require(sample->metrics.cached_prompt_tokens > 0 && !sample->metrics.cache_created,
                "Streaming lost warm-context metrics");
      }
    }
    require(final && text == gold_b.text, "Warm B stream changed output or omitted completion");
    auto excess = a;
    excess.cache_id = "excess";
    bool typed = false;
    try {
      model->run(excess);
    } catch (const KVCacheCapacityError&) {
      typed = true;
    }
    require(typed, "Direct API lost the capacity exception type");
    require(model->kv_cache_count() == 2 && model->kv_cache_bytes_per_slot() > 0,
            "Cache count or memory accounting incorrect");
    require(model->remove_kv_cache("a"), "Direct removal failed");
    a.cache_id.reset();
    require(model->run(a).metrics.cache_created, "Legacy context did not acquire released slot");
    require(model->kv_cache_count() == 2, "Legacy context bypassed slot accounting");
    model->clear_kv_caches();
#if defined(SIMA_WITH_OPENCV)
    if (model->accepts_image()) {
      GenerationRequest image;
      image.prompt = "Name the dominant color.";
      image.max_new_tokens = 24;
      image.cache_id = "red";
      const cv::Mat red(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));
      const cv::Mat blue(64, 64, CV_8UC3, cv::Scalar(255, 0, 0));
      image.images = {red};
      const auto red_result = model->run(image);
      image.cache_id = "blue";
      image.images = {blue};
      const auto blue_result = model->run(image);
      require(red_result.metrics.cached_prompt_tokens == 0 &&
                  blue_result.metrics.cached_prompt_tokens == 0,
              "Independent image contexts should start cold");
      image.cache_id = "red";
      image.images = {red};
      const auto red_again = model->run(image);
      require(red_again.text == red_result.text && red_again.metrics.cached_prompt_tokens > 0,
              "Image identity was not retained across context switches");
      image.images = {blue};
      const auto changed = model->run(image);
      require(changed.text == blue_result.text && changed.metrics.cached_prompt_tokens == 0 &&
                  !changed.metrics.cache_created,
              "Changing pixels did not invalidate the selected context");
      image.cache_id = "blue";
      const auto unchanged = model->run(image);
      require(unchanged.text == blue_result.text && unchanged.metrics.cached_prompt_tokens > 0,
              "Changing one image context invalidated another");
      model->clear_kv_caches();
      std::cout << "GENAI_KV_CACHE_IMAGE result=PASS\n";
    }
#endif
    check_http(model);
    std::cout << "GENAI_KV_CACHE result=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GENAI_KV_CACHE result=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
