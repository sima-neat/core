#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/graph/internal/GraphTestHooks.h"

#include <chrono>
#include <filesystem>
#include <functional>
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

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = text.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

void require_throws_contains(const std::function<void()>& fn, const std::string& needle,
                             const std::string& message) {
  try {
    fn();
  } catch (const std::exception& e) {
    require(std::string(e.what()).find(needle) != std::string::npos,
            message + ": unexpected exception: " + e.what());
    return;
  }
  throw std::runtime_error(message + ": expected exception containing '" + needle + "'");
}

std::filesystem::path temp_graph_path() {
  const auto base = std::filesystem::temp_directory_path();
  const auto name = "graph_phase2_composition_storage_" + std::to_string(::getpid()) + ".json";
  return base / name;
}

} // namespace

int main() {
  try {
    {
      auto shared = simaai::neat::nodes::VideoConvert();
      simaai::neat::Graph identity;
      identity.add(shared);
      const std::string before = identity.describe_backend(false);
      require_throws_contains([&] { identity.add(shared); }, "already exists",
                              "duplicate Node identity should be rejected");
      require(identity.describe_backend(false) == before,
              "failed duplicate add must leave the composition unchanged");

      simaai::neat::Graph moved_identity = std::move(identity);
      require_throws_contains([&] { moved_identity.add(shared); }, "already exists",
                              "identity index should move with the composition");

      simaai::neat::Graph distinct;
      distinct.add(simaai::neat::nodes::VideoConvert());
      distinct.add(simaai::neat::nodes::VideoConvert());
      require(!distinct.describe_backend(false).empty(),
              "separate instances of one Node type should remain valid");
    }

    {
      auto shared = simaai::neat::nodes::VideoConvert();
      simaai::neat::Graph left("left");
      left.add(shared);
      simaai::neat::Graph right("right");
      right.add(shared);

      simaai::neat::Graph destination;
      destination.add(simaai::neat::nodes::VideoScale());
      const std::string before = destination.describe_backend(false);
      require_throws_contains([&] { destination.connect(left, right); }, "already exists",
                              "overlapping fragment import should be rejected");
      require(destination.describe_backend(false) == before,
              "failed multi-fragment connect must roll back the first import");

      destination.add(simaai::neat::nodes::VideoConvert());
      require(destination.describe_backend(false).find("videoconvert") != std::string::npos,
              "a valid mutation should succeed after rollback");
    }

    {
      auto shared_source = simaai::neat::nodes::VideoConvert();
      auto first_sink = simaai::neat::nodes::VideoScale();
      auto second_sink = simaai::neat::nodes::VideoScale();
      simaai::neat::Graph fanout;
      fanout.add(shared_source);
      fanout.connect(shared_source, first_sink);
      fanout.connect(shared_source, second_sink);
      const std::string description = fanout.describe();
      require(count_occurrences(description, "VideoConvert") == 1U,
              "connect() should reuse an existing Node vertex");
      require(count_occurrences(description, "VideoScale") == 2U,
              "fan-out should retain two distinct sink vertices");
    }

    {
      // Guard the indexed insertion path against accidentally returning to an incremental
      // full-vector scan. Keep the limit deliberately generous for loaded CI/DevKit runs; this is
      // a setup-path complexity tripwire rather than a microbenchmark.
      constexpr std::size_t kNodeCount = 10'000U;
      simaai::neat::Graph incremental;
      std::shared_ptr<simaai::neat::Node> retained;
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t i = 0; i < kNodeCount; ++i) {
        auto node = simaai::neat::nodes::VideoConvert();
        if (i == kNodeCount / 2U) {
          retained = node;
        }
        incremental.add(std::move(node));
      }
      const auto elapsed = std::chrono::steady_clock::now() - start;
      require(elapsed < std::chrono::seconds(10),
              "10,000 indexed incremental insertions exceeded the complexity budget");
      require_throws_contains([&] { incremental.add(retained); }, "already exists",
                              "the large-graph identity index should remain complete");
    }

    {
      simaai::neat::Graph linear;
      linear.add(simaai::neat::nodes::Input("in"));
      linear.add(simaai::neat::nodes::Output("out"));
      const std::string before = linear.describe();
      simaai::neat::session_test::arm_composition_failure_for_test(
          simaai::neat::session_test::CompositionFailurePoint::EndpointEdgesReplaced);
      require_throws_contains([&] { linear.connect("in", "out"); }, "injected",
                              "endpoint-promotion failure should be injected after replacement");
      require(linear.describe() == before,
              "endpoint-promotion failure must restore the original implicit edge vector");
      linear.connect("in", "out");
    }

    {
      simaai::neat::Graph source_a("image");
      source_a.add(simaai::neat::nodes::Input("raw_a"));
      simaai::neat::Graph source_b("image");
      source_b.add(simaai::neat::nodes::Input("raw_b"));
      simaai::neat::Graph sink("image");
      sink.add(simaai::neat::nodes::Input("image"));
      sink.add(simaai::neat::nodes::Output("classes"));
      sink.connect("image", "classes");

      simaai::neat::GraphLinkOptions realtime;
      realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      realtime.queue_depth = 3;
      simaai::neat::Graph app;
      app.connect(source_a, sink, realtime);
      const std::string before = app.describe();

      simaai::neat::session_test::arm_composition_failure_for_test(
          simaai::neat::session_test::CompositionFailurePoint::BeforeConnectionEdgeAppend);
      require_throws_contains([&] { app.connect(source_b, sink, realtime); }, "injected",
                              "realtime fan-in failure should occur after existing-edge writes");
      require(app.describe() == before,
              "realtime fan-in failure must restore link options, stream IDs, and imports");
      app.connect(source_b, sink, realtime);
    }

    {
      simaai::neat::Graph batched;
      batched.add(simaai::neat::nodes::VideoConvert());
      const std::string before = batched.describe_backend(false);
      simaai::neat::session_test::arm_composition_failure_for_test(
          simaai::neat::session_test::CompositionFailurePoint::PipelineVertexAppended, 2U);
      require_throws_contains([&] { batched.add_output_tensor({}); }, "injected",
                              "add_output_tensor failure should occur after a partial append");
      require(batched.describe_backend(false) == before,
              "add_output_tensor must roll back all four Nodes when any insertion fails");
      batched.add_output_tensor({});
    }

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
