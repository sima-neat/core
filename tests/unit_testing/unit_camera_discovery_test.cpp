#include "nodes/io/CameraDiscovery.h"

#include <iostream>
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
} // namespace
int main() {
  try {
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
    std::cout << "PASS camera topology selection, ambiguity and legacy isolation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
