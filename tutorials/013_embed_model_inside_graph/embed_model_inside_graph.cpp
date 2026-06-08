// Hybrid graph composition: a Model is added directly to a public Graph.
//
// Usage:
//   tutorial_013_embed_model_inside_graph --model /path/to/model.tar.gz

#include "neat.h"

#include <iostream>
#include <string>

namespace {

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path;
    if (!get_arg(argc, argv, "--model", model_path)) {
      std::cerr << "Usage: tutorial_013_embed_model_inside_graph --model <path>\n";
      return 1;
    }

    // STEP load-model
    simaai::neat::Model model(model_path);
    // END STEP

    // CORE LOGIC
    // Model is now a Graph-compatible object. `graph.add(model)` appends the
    // model route fragment (preprocess/inference/postprocess as needed) without
    // exposing the internal low-level runtime graph.
    // STEP compose-graph
    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input("image"));
    graph.add(model);
    graph.add(simaai::neat::nodes::Output("result"));

    std::cout << graph.describe() << "\n";
    // END STEP
    // END CORE LOGIC

    // STEP inspect-model
    const auto info = model.info();
    std::cout << "model=" << (info.model_name.empty() ? "<unnamed>" : info.model_name)
              << " physical_outputs=" << info.output_topology.physical_outputs
              << " logical_outputs=" << info.output_topology.logical_outputs << "\n";
    // END STEP
    std::cout << "[OK] 013_embed_model_inside_graph\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
