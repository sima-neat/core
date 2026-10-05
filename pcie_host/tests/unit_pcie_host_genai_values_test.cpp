#include "Protocol.h"
#include "genai/MediaStage.h"
#include "genai/RemoteSession.h"
#include "genai/ModelAssets.h"
#include "AssetTransfer.h"
#include "SshRunner.h"

#include <filesystem>
#include <cstdlib>
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
struct SearchPath {
  std::optional<std::string> previous;
  explicit SearchPath(const std::filesystem::path& directory) {
    if (const auto* value = std::getenv("PATH"))
      previous = value;
    const auto path = directory.string() + ":" + previous.value_or("/usr/bin:/bin");
    if (setenv("PATH", path.c_str(), 1) != 0)
      throw std::runtime_error("Cannot set test executable search path");
  }
  ~SearchPath() {
    if (previous)
      setenv("PATH", previous->c_str(), 1);
    else
      unsetenv("PATH");
  }
};
} // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--shutdown-worker") {
    alarm(5); // Bound the helper lifetime even if its parent fails.
    for (;;)
      pause();
  }
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
    using Runner = simaai::neat::pcie::internal::SshRunner;
    const auto ssh_arguments = directory.path / "ssh-arguments";
    const auto fake_ssh = directory.path / "ssh";
    std::ofstream(fake_ssh) << "#!/bin/sh\nprintf '%s\\n' \"$@\" > "
                            << Runner::shell_escape(ssh_arguments.string()) << '\n';
    std::filesystem::permissions(fake_ssh, std::filesystem::perms::owner_all);
    {
      // Verify the actual launch destination without SSH or board access.
      SearchPath search(directory.path);
      for (const auto card : {0, 1, 2}) {
        for (const bool override_address : {false, true}) {
          g::ConnectionOptions connection;
          connection.card_id = card;
          connection.user = "sima";
          connection.ssh_key = "test-key";
          if (override_address)
            connection.card_host = "192.168.1.42";
          g::internal::RemoteSession remote(connection);
          remote.start();
          std::ifstream arguments(ssh_arguments);
          const std::string contents((std::istreambuf_iterator<char>(arguments)), {});
          const auto expected =
              override_address ? "192.168.1.42" : "10.0." + std::to_string(card) + ".2";
          require(contents.find("\nsima@" + expected + "\n") != std::string::npos,
                  "Card-specific address or explicit override reaches SSH");
        }
      }
    }
    g::ConnectionOptions invalid_connection;
    invalid_connection.card_id = -1;
    rejects([&] { g::internal::RemoteSession remote(invalid_connection); });
    const auto model_root = directory.path / "models";
    const auto model_path = model_root / "nested/model with spaces";
    std::filesystem::create_directories(model_path);
    require(g::internal::host_model_path(model_path) == model_path, "Absolute host model path");
    const auto relative_model =
        std::filesystem::relative(model_path, std::filesystem::current_path());
    require(g::internal::host_model_path(relative_model) == model_path,
            "Relative model path uses application cwd");
    require(g::internal::host_model_path(model_path.string() + "/") == model_path,
            "Trailing separators");
    rejects([&] { g::internal::host_model_path(model_root / "missing"); });
    rejects([&] { g::internal::host_model_path(""); });
    const auto asset = model_path / "layer 1.bin";
    std::ofstream(asset) << "weights";
    require(g::internal::model_asset_path(model_path, "layer 1.bin") == asset,
            "Host asset paths preserve spaces");
    require(!g::internal::model_asset_path(model_path, "missing.bin"), "Optional missing asset");
    const auto model_link = model_root / "link";
    std::filesystem::create_directory_symlink(model_path, model_link);
    require(g::internal::host_model_path(model_link) == model_path, "Canonical model directory");
    const auto inside_link = model_path / "inside.bin";
    std::filesystem::create_symlink(asset, inside_link);
    require(g::internal::model_asset_path(model_path, "inside.bin") == asset,
            "In-directory asset symlinks resolve safely");
    const auto outside = directory.path / "outside.bin";
    std::ofstream(outside) << "outside";
    std::filesystem::create_symlink(outside, model_path / "escape.bin");
    rejects([&] { g::internal::model_asset_path(model_path, "escape.bin"); });
    for (const auto* name : {"../outside.bin", "/etc/passwd", ".", "a/../layer 1.bin", "a\\b"})
      rejects([&] { g::internal::model_asset_path(model_path, name); });
    std::filesystem::create_directory(model_path / "directory");
    rejects([&] { g::internal::model_asset_path(model_path, "directory"); });
    rejects([&] { g::internal::host_model_path(asset); });
    require(wire::asset_tag(session, false) != wire::tag(session, false) &&
                wire::asset_tag(session, true).size() < 32,
            "Isolated asset channel tags");
    auto reply = wire::envelope(session, 1, "asset");
    reply["found"] = true;
    reply["bytes"] = 7;
    int sends = 0;
    auto send_asset = [&](const std::string& text) {
      const auto request = wire::parse(text, session);
      require(request.at("name") == "layer 1.bin", "Only model-relative names cross PCIe");
      ++sends;
    };
    auto exchange = [&](auto receive, auto cancelled, int timeout = 1500) {
      return wire::request_asset(session, 1, "layer 1.bin", timeout, send_asset, receive,
                                 cancelled);
    };
    require(exchange([&](int) { return std::optional(reply.dump()); }, [] { return false; }) == 7,
            "Successful asset response");
    sends = 0;
    require(exchange(
                [&](int ms) -> std::optional<std::string> {
                  if (sends < 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                    return std::nullopt;
                  }
                  return reply.dump();
                },
                [] { return false; }) == 7 &&
                sends == 2,
            "Retry after a lost reply");
    reply["found"] = false;
    require(!exchange([&](int) { return std::optional(reply.dump()); }, [] { return false; }),
            "Optional missing-file response");
    reply["error"] = "permission denied";
    rejects(
        [&] { exchange([&](int) { return std::optional(reply.dump()); }, [] { return false; }); });
    reply.erase("error");
    reply["request"] = 2;
    rejects(
        [&] { exchange([&](int) { return std::optional(reply.dump()); }, [] { return false; }); });
    rejects([&] {
      exchange([](int) -> std::optional<std::string> { return std::nullopt; }, [] { return true; });
    });
    rejects([&] {
      exchange(
          [](int ms) -> std::optional<std::string> {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            return std::nullopt;
          },
          [] { return false; }, 10);
    });
    // Exercise the actual shutdown script locally, including exit between
    // its ownership check and kill. No SSH server or model fixture is needed.
    namespace remote = g::internal;
    const auto executable = Runner::shell_escape(std::filesystem::read_symlink("/proc/self/exe"));
    const auto session_directory =
        Runner::shell_escape((directory.path / ".cache/neat-genai" / session).string());
    auto stop_script = remote::RemoteSession::stop_script(session);
    // Redirect only the test's session directory; leave the real HOME unchanged.
    stop_script.replace(stop_script.find("$HOME"), 5, directory.path.string());
    const std::string launch =
        "set -eu; d=" + session_directory + "; mkdir -p \"$d\"; " + executable +
        " --shutdown-worker " + session +
        " & p=$!; trap 'command kill -KILL \"$p\" 2>/dev/null || :; wait \"$p\" 2>/dev/null || :' "
        "EXIT; "
        "echo \"$p\" >\"$d/pid\"; awk '{print $22}' /proc/$p/stat >\"$d/start\"; "
        "while ! tr '\\0' '\\n' </proc/$p/cmdline | grep -Fxq " +
        session + "; do sleep 0.01; done; ";
    const auto normal_stop = Runner::run({"/bin/sh", "-c", launch + stop_script}, 3);
    require(normal_stop.exit_code == 0 && !normal_stop.timed_out, "Normal worker shutdown");
    const auto raced_stop =
        Runner::run({"/bin/sh", "-c",
                     launch +
                         "kill() { command kill \"$@\"; wait \"$p\" 2>/dev/null || :; "
                         "command kill \"$@\"; }; " +
                         stop_script},
                    3);
    require(raced_stop.exit_code == 0 && !raced_stop.timed_out,
            "Worker exiting between ownership check and kill must be successful");
    const auto denied_stop =
        Runner::run({"/bin/sh", "-c", launch + "kill() { return 1; }; " + stop_script}, 3);
    require(denied_stop.exit_code != 0 && !denied_stop.timed_out,
            "A failed kill of a still-owned worker must not be suppressed");
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
