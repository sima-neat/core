#include "Protocol.h"
#include "genai/MediaStage.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace g = simaai::neat::pcie::genai;
namespace wire = simaai::neat::pcie::genai::wire;
namespace pcie = simaai::neat::pcie;

namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class Fn> void rejects(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Expected rejection");
}
struct Directory {
  std::filesystem::path path;
  Directory() {
    char name[] = "/tmp/neat-genai-values-XXXXXX";
    const auto p = mkdtemp(name);
    if (!p)
      throw std::runtime_error("mkdtemp failed");
    path = p;
  }
  ~Directory() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};
} // namespace

int main() {
  try {
    const std::string session(24, 'a');
    require(wire::tag(session, true).size() < 32, "Platform tag limit");
    const auto e = wire::envelope(session, UINT64_MAX, "sample");
    require(wire::parse(e.dump(), session).at("request") == UINT64_MAX, "64-bit identity");
    rejects([&] { wire::parse(e.dump(), std::string(24, 'b')); });
    auto invalid = e;
    invalid["v"] = wire::version + 1;
    rejects([&] { wire::parse(invalid.dump(), session); });
    invalid = e;
    invalid["request"] = -1;
    rejects([&] { wire::parse(invalid.dump(), session); });
    rejects([&] { wire::parse(std::string(wire::max_message_bytes + 1, ' '), session); });
    for (auto path : {"/tmp/a", "../a", "a/../b", "a/./b", "a\\b", ""})
      rejects([&] { wire::relative_name(path); });
    require(wire::relative_name("model/devkit/config.json") == "model/devkit/config.json",
            "Asset path");

    g::TokenSample sample;
    sample.text = "answer";
    sample.reasoning = "reason";
    sample.is_final = true;
    sample.finish_reason = "stop";
    sample.language = "en";
    sample.no_speech_prob = .2f;
    sample.avg_logprob = -.1f;
    sample.metrics.generated_tokens = 7;
    sample.metrics.time_to_first_token_s = .5;
    sample.metrics.tokens_per_second = 9;
    sample.tool_calls = g::Json::array({{{"id", "tool-1"}}});
    require(wire::encode(wire::decode(wire::encode(sample))) == wire::encode(sample),
            "Result parity");

    Directory directory;
    g::ConnectionOptions options;
    options.media_directory = directory.path / "does-not-exist";
    g::GenerationRequest request;
    request.prompt = "hello";
    {
      g::internal::MediaStage stage(options, session, 1);
      require(stage.encode(request).at("prompt") == "hello", "Text-only staging");
      require(!std::filesystem::exists(options.media_directory),
              "Text must not need a serve directory");
    }
    options.media_directory = directory.path;
    unsigned char pixels[] = {1, 2, 3, 99, 4, 5, 6};
    pcie::Tensor image;
    image.data = pixels;
    image.size_bytes = sizeof(pixels);
    image.dtype = pcie::TensorDType::UInt8;
    image.shape = {1, 2, 3};
    image.strides_bytes = {7, 4, 1};
    request.images = {image};
    std::filesystem::path staged;
    {
      g::internal::MediaStage stage(options, session, 2);
      const auto encoded = stage.encode(request);
      staged = directory.path / encoded.at("images").at(0).at("name").get<std::string>();
      std::ifstream input(staged, std::ios::binary);
      const std::string bytes((std::istreambuf_iterator<char>(input)), {});
      require(bytes == std::string("P6\n2 1\n255\n") + std::string("\1\2\3\4\5\6", 6),
              "Strided RGB staging");
    }
    require(!std::filesystem::exists(staged), "Owned media cleanup");
    request.images[0].image_format = pcie::PixelFormat::BGR;
    rejects([&] {
      g::internal::MediaStage stage(options, session, 3);
      stage.encode(request);
    });
    request.images.clear();
    request.messages.push_back({});
    rejects([&] {
      g::internal::MediaStage stage(options, session, 4);
      stage.encode(request);
    });
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
