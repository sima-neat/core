#include "graph_migration/common/phase3_graph_test_utils.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path tmp_path(const std::string& name) {
  const char* tmp = std::getenv("TMPDIR");
  return std::filesystem::path(tmp ? tmp : "/tmp") / name;
}

void require_file_contains(const std::filesystem::path& path, const std::string& needle,
                           const std::string& label) {
  std::ifstream in(path);
  const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (body.find(needle) == std::string::npos) {
    throw std::runtime_error(label + ": missing '" + needle + "' in\n" + body);
  }
}

} // namespace

RUN_TEST("graph_migration_phaseA4_connected_save_load_test", [] {
  {
    simaai::neat::Graph input("image");
    input.add(simaai::neat::nodes::Input("image"));
    simaai::neat::Graph output("classes");
    output.add(simaai::neat::nodes::Output("classes"));

    simaai::neat::Graph app("connected_save_load");
    app.connect(input, output);

    const std::filesystem::path path = tmp_path("connected_save_load.neat.json");
    std::filesystem::remove(path);
    app.save(path.string());
    require_file_contains(path, "\"version\": 2", "connected save schema version");
    require_file_contains(path, "\"edges\"", "connected save topology edges");
    require_file_contains(path, "\"from_endpoint\":\"image\"", "connected save from endpoint");
    require_file_contains(path, "\"to_endpoint\":\"classes\"", "connected save to endpoint");

    simaai::neat::Graph loaded = simaai::neat::Graph::load(path.string());
    require(loaded.describe().find("endpoint image -> classes") != std::string::npos,
            "loaded connected describe should preserve endpoint edge");
    require(loaded.validate().error_code.empty(),
            "loaded connected graph should validate structurally");

    simaai::neat::Run run = loaded.build();
    require(run.push("image",
                     simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x71)}),
            "loaded connected push failed: " + run.last_error());
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors, "loaded connected pull_tensors");
    run.close();
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByFrame);
    const std::filesystem::path path = tmp_path("connected_combine_save_load.neat.json");
    std::filesystem::remove(path);
    join.save(path.string());
    require_file_contains(path, "\"combine_policy\":\"ByFrame\"", "connected save CombinePolicy");

    simaai::neat::Graph loaded = simaai::neat::Graph::load(path.string());
    require(loaded.describe().find("combine=ByFrame") != std::string::npos,
            "loaded connected graph should preserve CombinePolicy");

    simaai::neat::Run run = loaded.build();
    simaai::neat::Sample left = graph_phase3_test::make_tensor_sample(900, "left-stream");
    simaai::neat::Sample right = graph_phase3_test::make_tensor_sample(900, "right-stream");
    require(run.push("left", left), "loaded combine left push failed: " + run.last_error());
    require(run.push("right", right), "loaded combine right push failed: " + run.last_error());
    auto out = run.pull("combined", 5000);
    require(out.has_value(), "loaded combine pull timed out: " + run.last_error());
    require(out->kind == simaai::neat::SampleKind::Bundle,
            "loaded combine output should be a bundle");
    require(out->fields.size() == 2U, "loaded combine should preserve two fields");
    require(out->fields[0].stream_label == "left",
            "loaded combine field order/name mismatch for left");
    require(out->fields[1].stream_label == "right",
            "loaded combine field order/name mismatch for right");
    run.close();
  }

  {
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    const std::filesystem::path path = tmp_path("connected_round_robin_save_load.neat.json");
    std::filesystem::remove(path);
    rr.save(path.string());
    require_file_contains(path, "\"combine_policy\":\"RoundRobin\"",
                          "connected save RoundRobin CombinePolicy");

    simaai::neat::Graph loaded = simaai::neat::Graph::load(path.string());
    require(loaded.describe().find("combine=RoundRobin") != std::string::npos,
            "loaded connected graph should preserve RoundRobin CombinePolicy");

    simaai::neat::Run run = loaded.build();
    simaai::neat::Sample left = graph_phase3_test::make_tensor_sample(910, "");
    simaai::neat::Sample right = graph_phase3_test::make_tensor_sample(911, "");
    require(run.push("left", left), "loaded round-robin left push failed: " + run.last_error());
    require(run.push("right", right), "loaded round-robin right push failed: " + run.last_error());
    auto first = run.pull("combined", 5000);
    require(first.has_value(), "loaded round-robin first pull timed out: " + run.last_error());
    auto second = run.pull("combined", 5000);
    require(second.has_value(), "loaded round-robin second pull timed out: " + run.last_error());
    require(first->kind != simaai::neat::SampleKind::Bundle,
            "loaded round-robin should forward original samples");
    require(second->kind != simaai::neat::SampleKind::Bundle,
            "loaded round-robin should forward original samples");
    require(first->stream_id == "left", "loaded round-robin left stream_id mismatch");
    require(second->stream_id == "right", "loaded round-robin right stream_id mismatch");
    run.close();
  }
});
