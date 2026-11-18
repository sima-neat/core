#include "genai/ModelAssets.h"
#include "HostPlatform.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

extern "C" {
void test_genai_receive_root(const char*);
int test_genai_copies();
void test_genai_put_error(int);
void test_genai_block_put(bool);
void test_genai_pause_put(bool);
bool test_genai_put_paused();
bool test_genai_put_waiting();
int test_genai_open_clients();
}
namespace wire = simaai::neat::pcie::genai::wire;
namespace internal = simaai::neat::pcie::genai::internal;
namespace host = simaai::neat::pcie::internal;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class Ready> bool wait_until(Ready ready, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!ready() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return ready();
}
// A hung close() cannot be joined, so report it and exit instead of waiting for ctest.
template <class During>
void close_within(internal::ModelAssets& assets, std::chrono::milliseconds bound, During during,
                  const char* message) {
  std::atomic<bool> closed{false};
  std::thread closer([&] {
    assets.close();
    closed = true;
  });
  during();
  if (!wait_until([&] { return closed.load(); }, bound)) {
    std::cerr << message << '\n';
    std::_Exit(1);
  }
  closer.join();
}
struct Directory {
  std::filesystem::path path;
  Directory() {
    static std::atomic<unsigned long> sequence{0};
    path = std::filesystem::temp_directory_path() /
           ("neat-genai-assets-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
            std::to_string(++sequence));
    if (!std::filesystem::create_directory(path))
      throw std::runtime_error("temporary directory creation failed");
  }
  ~Directory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};
} // namespace
int main() {
  try {
    Directory directory;
    const auto model = directory.path / "ordinary host directory";
    std::filesystem::create_directory(model);
    std::ofstream(model / "layer.bin") << "weights";
    std::filesystem::create_directories(model / "target/elf_files");
    std::filesystem::create_directories(model / "draft/elf_files");
    std::ofstream(model / "target/elf_files/model.elf") << "target";
    std::ofstream(model / "draft/elf_files/model.elf") << "draft";
    const auto recv = directory.path / "received";
    auto rejects_asset = [&](const std::filesystem::path& path, uint64_t size,
                             const std::string& message) {
      try {
        wire::validate_received_asset(path, size, "layer.bin");
      } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find(message) != std::string::npos &&
                    std::string(e.what()).find(path.string()) != std::string::npos,
                "Asset failure distinguishes its cause and includes the expected path");
        return;
      }
      throw std::runtime_error("Expected asset validation failure");
    };
    wire::validate_received_asset(model / "layer.bin", 7, "layer.bin");
    rejects_asset(model / "missing.bin", 7, "missing at expected path");
    rejects_asset(model, 7, "not a regular file");
    rejects_asset(model / "layer.bin", 8, "expected 8, received 7 bytes");
    const auto recv_string = recv.string();
    test_genai_receive_root(recv_string.c_str());
    const std::string session(24, 'b');
    internal::ModelAssets assets(model, session, 0);
    wire::Service worker(0);
    worker.subscribe(wire::asset_tag(session, true));
    auto request = [&](uint64_t id, const std::string& name) {
      return wire::request_asset(
          session, id, name, 2000,
          [&](const std::string& text) { worker.send(wire::asset_tag(session, false), text); },
          [&](int ms) { return worker.receive(ms); }, [] { return false; });
    };
    require(request(1, "layer.bin") == 7, "File transfer size");
    const auto target = recv / "neat-genai" / session / "model/layer.bin";
    std::ifstream input(target);
    std::string contents;
    input >> contents;
    require(contents == "weights", "File transfer contents");
    input.close();
    std::filesystem::remove(target);
    require(request(1, "layer.bin") == 7 && test_genai_copies() == 1 &&
                !std::filesystem::exists(target),
            "Retry replays response without restoring evicted file");
    require(!request(2, "missing.bin") && test_genai_copies() == 1, "Optional asset not found");
    require(request(3, "layer.bin") == 7 && test_genai_copies() == 2, "Fetch again after eviction");
    require(request(4, "target/elf_files/model.elf") == 6 &&
                request(5, "draft/elf_files/model.elf") == 5 && test_genai_copies() == 4,
            "Target and draft share one sequential transfer channel");
    auto read_asset = [&](const std::string& name) {
      std::ifstream stream(recv / "neat-genai" / session / "model" / name);
      return std::string((std::istreambuf_iterator<char>(stream)), {});
    };
    require(read_asset("target/elf_files/model.elf") == "target" &&
                read_asset("draft/elf_files/model.elf") == "draft",
            "Identically named target/draft files retain distinct contents");
    bool rejected = false;
    try {
      request(6, "../unavailable");
    } catch (const std::invalid_argument&) {
      rejected = true;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && test_genai_copies() == 4, "Reject path escape before upload");
#ifndef _WIN32
    std::filesystem::create_symlink(directory.path, model / "escape");
    rejected = false;
    try {
      request(6, "escape/unavailable");
    } catch (const std::invalid_argument&) {
      rejected = true;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && test_genai_copies() == 4, "Reject symlink escape before upload");
#else
    require(!request(6, "another-missing.bin") && test_genai_copies() == 4,
            "Missing asset preserves the sequential transfer channel");
#endif
    test_genai_put_error(-EACCES);
    rejected = false;
    try {
      request(7, "layer.bin");
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "Propagate daemon transfer failure");
    test_genai_put_error(0);
    test_genai_block_put(true);
    auto blocked = wire::envelope(session, 8, "asset");
    blocked["name"] = "layer.bin";
    worker.send(wire::asset_tag(session, false), blocked.dump());
    // copies counts entry into put(); wait until its read is pending instead.
    require(wait_until(test_genai_put_waiting, std::chrono::seconds(2)) && test_genai_copies() == 6,
            "Reach blocked transfer");
    const auto started = std::chrono::steady_clock::now();
    assets.close();
    require(std::chrono::steady_clock::now() - started < std::chrono::seconds(1),
            "Close interrupts a blocked transfer");
    assets.close();
    assets.check();
    test_genai_block_put(false);
    auto send_asset = [&](const std::string& id) {
      auto asset = wire::envelope(id, 1, "asset");
      asset["name"] = "layer.bin";
      worker.send(wire::asset_tag(id, false), asset.dump());
    };
    {
      // Windows cancels only pending I/O: close() before put() starts waiting
      // must interrupt again once it does.
      const std::string late(24, 'c');
      internal::ModelAssets paused(model, late, 0);
      test_genai_block_put(true);
      test_genai_pause_put(true);
      send_asset(late);
      require(wait_until(test_genai_put_paused, std::chrono::seconds(2)),
              "Pause before the transfer waits");
      close_within(
          paused, std::chrono::seconds(1),
          [] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            test_genai_pause_put(false);
          },
          "Close interrupts a transfer that starts waiting after close begins");
      paused.check();
      test_genai_block_put(false);
    }
    const int clients = test_genai_open_clients();
    {
      const std::string stuck(24, 'd');
      internal::ModelAssets ignored(model, stuck, 0);
      test_genai_pause_put(true);
      send_asset(stuck);
      require(wait_until(test_genai_put_paused, std::chrono::seconds(2)),
              "Pause a transfer that ignores interruption");
      close_within(
          ignored, std::chrono::seconds(4), [] {},
          "Close leaks a transfer that ignores interruption instead of hanging");
    }
    test_genai_pause_put(false);
    require(
        wait_until([&] { return test_genai_open_clients() == clients; }, std::chrono::seconds(2)),
        "Leaked worker releases its service once the transfer returns");
    {
      // GenAIModel receives UTF-8 names; a non-ASCII root must reach the upload intact.
      const auto tokyo = directory.path / std::filesystem::path(u8"\u6771\u4eac"); // 東京
      std::filesystem::create_directory(tokyo);
      std::ofstream(tokyo / "layer.bin") << "weights";
      const auto root =
          internal::host_model_path(host::host_path_from_string(host::host_path_string(tokyo)));
      const std::string unicode(24, 'e');
      internal::ModelAssets unicode_assets(root, unicode, 0);
      wire::Service replies(0);
      replies.subscribe(wire::asset_tag(unicode, true));
      require(
          wire::request_asset(
              unicode, 1, "layer.bin", 2000,
              [&](const std::string& text) { replies.send(wire::asset_tag(unicode, false), text); },
              [&](int ms) { return replies.receive(ms); }, [] { return false; }) == 7,
          "Upload an asset under a non-ASCII model directory");
      std::ifstream uploaded(recv / "neat-genai" / unicode / "model/layer.bin");
      require(std::string((std::istreambuf_iterator<char>(uploaded)), {}) == "weights",
              "Non-ASCII model asset contents");
    }
    std::cout << "GenAI on-demand assets: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
