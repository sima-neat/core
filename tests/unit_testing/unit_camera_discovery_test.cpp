#include "nodes/io/CameraDiscovery.h"

#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace simaai::neat::camera_discovery;
namespace {
void check(bool value) {
  if (!value)
    throw std::runtime_error("camera discovery assertion failed");
}
void rejects(const std::vector<Topology>& graphs, const std::string& device = "") {
  bool rejected = false;
  try {
    select_simor_device(graphs, device);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected);
}
struct SysfsFixture {
  std::filesystem::path root;
  SysfsFixture() {
    const auto base = std::filesystem::temp_directory_path();
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
      root = base / ("neat-camera-sysfs-" + std::to_string(std::random_device{}()));
      if (std::filesystem::create_directory(root))
        return;
    }
    throw std::runtime_error("cannot create isolated sysfs fixture");
  }
  ~SysfsFixture() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
};

void check_controller_isolation() {
  SysfsFixture fixture;
  const auto owner = fixture.root / "devices/camera";
  const auto selected = fixture.root / "dev/char/81:7";
  std::filesystem::create_directories(owner / "media9");
  std::filesystem::create_directories(owner / "media2");
  std::filesystem::create_directories(owner / "media-invalid");
  std::filesystem::create_directories(owner / "media");
  std::ofstream(owner / "media3"); // A regular file is not a controller.
  std::filesystem::create_directories(fixture.root / "devices/unrelated/media0");
  std::filesystem::create_directories(selected);
  std::filesystem::create_directory_symlink(owner, selected / "device");
  check(media_nodes_for_video_device(selected) ==
        std::vector<std::string>({"/dev/media2", "/dev/media9"}));

  // Missing ownership must fail closed, not scan unrelated controllers.
  std::filesystem::remove(selected / "device");
  bool rejected = false;
  try {
    media_nodes_for_video_device(selected);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected);
  std::filesystem::create_directory_symlink(owner, selected / "device");
  std::filesystem::remove(owner / "media2");
  std::filesystem::remove(owner / "media9");
  rejected = false;
  try {
    media_nodes_for_video_device(selected);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected);
}
} // namespace
int main() {
  try {
    check_controller_isolation();
    // Saved board topology: sensor -> CSI -> VDMA -> video. IDs and device
    // numbering intentionally differ from the board to catch hard-coded paths.
    Topology graph{{{103, "simor_metoak 9-0066", true},
                    {42, "csi", false},
                    {73, "vdma", false},
                    {6, "capture", false}},
                   {{103, 42, true}, {42, 73, true}, {73, 6, true}},
                   {{6, "/dev/video7"}}};
    check(select_simor_device({graph}, "") == "/dev/video7");
    check(select_simor_device({graph}, "/dev/video7") == "/dev/video7");
    rejects({graph}, "/dev/video1");
    rejects({});
    auto changed = graph;
    changed.entities[0].name = "imx477 9-001a";
    rejects({changed});
    changed.entities[0].name = "not_simor_metoak 9-0066";
    rejects({changed});
    changed = graph;
    changed.entities[0].sensor = false;
    rejects({changed});
    changed = graph;
    changed.links[0].enabled = false;
    rejects({changed});
    changed = graph;
    changed.entities.push_back({104, "imx477", true});
    changed.links.push_back({104, 42, true});
    rejects({changed}); // Ambiguous mux: never infer stream routing.
    changed.links.back().enabled = false;
    check(select_simor_device({changed}, "") == "/dev/video7");
    changed = graph;
    changed.endpoints[0].device = "/dev/video8";
    rejects({graph, changed});
    check(select_simor_device({graph, changed}, "/dev/video8") == "/dev/video8");
    changed = graph;
    changed.links.push_back({6, 42, true}); // Malformed cycle terminates.
    check(select_simor_device({changed}, "") == "/dev/video7");
    changed.links.push_back({999, 42, true});
    rejects({changed});
    std::cout
        << "PASS camera topology selection, controller isolation, ambiguity and legacy isolation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
