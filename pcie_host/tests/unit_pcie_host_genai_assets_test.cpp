#include "genai/ModelAssets.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

extern "C" {
void test_genai_receive_root(const char*);
int test_genai_copies();
void test_genai_put_error(int);
void test_genai_block_put(bool);
}
namespace wire = simaai::neat::pcie::genai::wire;
namespace internal = simaai::neat::pcie::genai::internal;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
struct Directory {
  std::filesystem::path path;
  Directory() {
    char name[] = "/tmp/neat-genai-assets-XXXXXX";
    const auto* made = mkdtemp(name);
    if (!made)
      throw std::runtime_error("mkdtemp failed");
    path = made;
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
    test_genai_receive_root(recv.c_str());
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
    std::filesystem::create_symlink(directory.path, model / "escape");
    bool rejected = false;
    try {
      request(6, "escape/unavailable");
    } catch (const std::invalid_argument&) {
      rejected = true;
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && test_genai_copies() == 4, "Reject symlink escape before upload");
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
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (test_genai_copies() != 6 && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(test_genai_copies() == 6, "Reach blocked transfer");
    const auto started = std::chrono::steady_clock::now();
    assets.close();
    require(std::chrono::steady_clock::now() - started < std::chrono::seconds(1),
            "Close interrupts a blocked transfer");
    assets.close();
    assets.check();
    test_genai_block_put(false);
    std::cout << "GenAI on-demand assets: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
