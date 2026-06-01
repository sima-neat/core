#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_ordered(const std::string& text, const std::string& first, const std::string& second,
                     const std::string& third) {
  const auto p1 = text.find(first);
  const auto p2 = text.find(second);
  const auto p3 = text.find(third);
  require(p1 != std::string::npos, "missing fragment: " + first + " in " + text);
  require(p2 != std::string::npos, "missing fragment: " + second + " in " + text);
  require(p3 != std::string::npos, "missing fragment: " + third + " in " + text);
  require(p1 < p2 && p2 < p3, "fragments are not in linear add order: " + text);
}

std::filesystem::path temp_graph_path() {
  const auto base = std::filesystem::temp_directory_path();
  const auto name = "graph_phase2_composition_storage_" + std::to_string(::getpid()) + ".json";
  return base / name;
}

} // namespace

int main() {
  try {
    simaai::neat::Graph graph;
    graph.custom("identity name=phase2_first");
    graph.custom("queue name=phase2_second");
    graph.custom("fakesink name=phase2_third");

    const std::string pipeline = graph.describe_backend(false);
    require_ordered(pipeline, "phase2_first", "phase2_second", "phase2_third");

    const std::filesystem::path path = temp_graph_path();
    graph.save(path.string());

    simaai::neat::Graph loaded = simaai::neat::Graph::load(path.string());
    const std::string loaded_pipeline = loaded.describe_backend(false);
    require_ordered(loaded_pipeline, "phase2_first", "phase2_second", "phase2_third");
    std::filesystem::remove(path);

    simaai::neat::Graph moved = std::move(loaded);
    require_ordered(moved.describe_backend(false), "phase2_first", "phase2_second", "phase2_third");

    loaded.custom("identity name=phase2_moved_from_reused");
    require(loaded.describe_backend(false).find("phase2_moved_from_reused") != std::string::npos,
            "moved-from Graph should remain reusable by add/custom");

    simaai::neat::Graph null_graph;
    bool add_threw = false;
    try {
      null_graph.add(std::shared_ptr<simaai::neat::Node>{});
    } catch (...) {
      add_threw = true;
    }
    require(!add_threw, "Graph::add(nullptr) should preserve deferred validation behavior");

    bool validate_failed_on_null = false;
    try {
      (void)null_graph.validate();
    } catch (const simaai::neat::NeatError& e) {
      validate_failed_on_null = std::string(e.what()).find("node is null") != std::string::npos;
    }
    require(validate_failed_on_null, "null node should still fail through validation/build path");

    std::cout << "[OK] graph_migration_phase2_composition_storage_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] graph_migration_phase2_composition_storage_test: " << e.what() << "\n";
    return 1;
  }
}
