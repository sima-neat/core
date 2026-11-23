#include "genai/ModelAssets.h"
#include "HostPlatform.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

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
// Returns the error close() reported, or an empty string.
template <class Close, class During>
std::string close_within(Close close, std::chrono::milliseconds bound, During during,
                         const char* message) {
  std::atomic<bool> closed{false};
  std::string error;
  std::thread closer([&] {
    try {
      close();
    } catch (const std::exception& e) {
      error = e.what();
    }
    closed = true;
  });
  during();
  if (!wait_until([&] { return closed.load(); }, bound)) {
    std::cerr << message << '\n';
    std::_Exit(1);
  }
  closer.join();
  return error;
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
      require(close_within([&] { paused.close(); }, std::chrono::seconds(1),
                           [] {
                             std::this_thread::sleep_for(std::chrono::milliseconds(100));
                             test_genai_pause_put(false);
                           },
                           "Close interrupts a transfer that starts waiting after close begins")
                  .empty(),
              "Close confirms an interrupted transfer");
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
      auto close = [&] { ignored.close(); };
      require(close_within(
                  close, std::chrono::seconds(4), [] {},
                  "Close returns while a transfer ignores interruption")
                      .find("upload is still running") != std::string::npos,
              "Close reports an unfinished transfer");
      test_genai_pause_put(false);
      require(close_within(
                  close, std::chrono::seconds(4), [] {}, "Close confirms a transfer that finished")
                  .empty(),
              "A later close confirms the transfer finished");
    }
    {
      const std::string abandoned(24, 'f');
      std::optional<internal::ModelAssets> leaked(std::in_place, model, abandoned, 0);
      test_genai_pause_put(true);
      send_asset(abandoned);
      require(wait_until(test_genai_put_paused, std::chrono::seconds(2)),
              "Pause a transfer the destructor abandons");
      close_within([&] { leaked.reset(); }, std::chrono::seconds(4), [] {},
                   "Destruction leaks a transfer that ignores interruption instead of hanging");
    }
    test_genai_pause_put(false);
    require(
        wait_until([&] { return test_genai_open_clients() == clients; }, std::chrono::seconds(2)),
        "Leaked worker releases its service once the transfer returns");
    // GenAIModel receives UTF-8 names; a non-ASCII root must reach the upload intact.
    auto upload_under = [&](const std::u8string& folder, char id) -> std::string {
      const auto parent = directory.path / std::filesystem::path(folder);
      std::filesystem::create_directory(parent);
      std::ofstream(parent / "layer.bin") << "weights";
      const auto root =
          internal::host_model_path(host::host_path_from_string(host::host_path_string(parent)));
      const std::string unicode(24, id);
      internal::ModelAssets unicode_assets(root, unicode, 0);
      wire::Service replies(0);
      replies.subscribe(wire::asset_tag(unicode, true));
      try {
        require(wire::request_asset(
                    unicode, 1, "layer.bin", 2000,
                    [&](const std::string& text) {
                      replies.send(wire::asset_tag(unicode, false), text);
                    },
                    [&](int ms) { return replies.receive(ms); }, [] { return false; }) == 7,
                "Non-ASCII model asset size");
      } catch (const std::exception& e) {
        return e.what();
      }
      std::ifstream uploaded(recv / "neat-genai" / unicode / "model/layer.bin");
      require(std::string((std::istreambuf_iterator<char>(uploaded)), {}) == "weights",
              "Non-ASCII model asset contents");
      return {};
    };
#ifdef _WIN32
    // The released SiMaPCIe.dll, like the fake, opens host files in the active code page.
    const auto code_page_holds = [](const wchar_t* text) {
      BOOL defaulted = FALSE;
      return GetACP() == CP_UTF8 || (WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, text, -1,
                                                         nullptr, 0, nullptr, &defaulted) > 0 &&
                                     !defaulted);
    };
    if (code_page_holds(L"\u00e9"))
      require(upload_under(u8"caf\u00e9", '0').empty(),
              "Upload an asset under a folder the code page represents");
    // No ANSI code page holds both Thai and Hebrew.
    const auto error = upload_under(u8"\u0e01\u05d0", '1');
    require(GetACP() == CP_UTF8 ? error.empty()
                                : error.find("system code page") != std::string::npos,
            "Report a model folder the code page cannot represent");
#else
    require(upload_under(u8"\u6771\u4eac", '0').empty(), // 東京
            "Upload an asset under a non-ASCII model directory");
#endif
    std::cout << "GenAI on-demand assets: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
