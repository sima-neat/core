#include <neat.h>

#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require_port(const simaai::neat::ContractPortSpec& port, const std::string& name,
                  const char* context) {
  require(port.port_id == name,
          std::string(context) + ": expected port '" + name + "' got '" + port.port_id + "'");
  require(port.media_type == "application/vnd.simaai.tensor",
          std::string(context) + ": visual frontend ports should be tensor media");
  require(!port.required_segment_names.empty() && port.required_segment_names.front() == name,
          std::string(context) + ": required segment name should match public port");
}

void test_feature_histogram_public_api() {
  simaai::neat::FeatureHistogramOptions opt;
  opt.width = 320;
  opt.height = 240;
  opt.batch_size = 2;

  const std::string summary = opt.summary();
  require_contains(summary, "FeatureHistogramOptions", "summary should name the options type");
  require_contains(summary, "graph=feature_histogram", "summary should name graph");
  require_contains(summary, "graph_id=235", "summary should expose graph id for diagnostics");
  require_contains(summary, "input_shape=[2,240,320]", "summary should expose logical input shape");
  require_contains(summary, "output_shape=[2,256]", "summary should expose logical output shape");

  simaai::neat::FeatureHistogram node(opt);
  require(node.kind() == "FeatureHistogram", "FeatureHistogram kind mismatch");
  const auto def = node.contract_definition();
  require(def.plugin_kind == "processcvu", "FeatureHistogram should compile to processcvu");
  require(def.inputs.size() == 1U, "FeatureHistogram should have one public input");
  require(def.outputs.size() == 1U, "FeatureHistogram should have one public output");
  require_port(def.inputs.front(), "input_image", "FeatureHistogram input");
  require_port(def.outputs.front(), "output_hist", "FeatureHistogram output");

  auto factory = simaai::neat::nodes::FeatureHistogram(opt);
  require(factory && factory->kind() == "FeatureHistogram", "FeatureHistogram factory mismatch");
}

void test_grider_fast_public_api() {
  simaai::neat::GriderFastOptions opt;
  opt.width = 320;
  opt.height = 240;
  opt.batch_size = 2;
  opt.max_features = 64;

  const std::string summary = opt.summary();
  require_contains(summary, "GriderFastOptions", "summary should name the options type");
  require_contains(summary, "graph=grider_fast", "summary should name graph");
  require_contains(summary, "graph_id=236", "summary should expose graph id for diagnostics");
  require_contains(summary, "feature_shape=[2,193]", "summary should expose feature shape");

  simaai::neat::GriderFast node(opt);
  require(node.kind() == "GriderFast", "GriderFast kind mismatch");
  const auto def = node.contract_definition();
  require(def.inputs.size() == 1U, "GriderFast should have one public input");
  require(def.outputs.size() == 1U, "GriderFast should have one public output");
  require_port(def.inputs.front(), "input_image", "GriderFast input");
  require_port(def.outputs.front(), "output_features", "GriderFast output");

  auto factory = simaai::neat::nodes::GriderFast(opt);
  require(factory && factory->kind() == "GriderFast", "GriderFast factory mismatch");
}

void test_track_descriptor_public_api() {
  simaai::neat::TrackDescriptorOptions opt;
  opt.width = 320;
  opt.height = 240;
  opt.batch_size = 2;
  opt.max_features = 64;

  const std::string summary = opt.summary();
  require_contains(summary, "TrackDescriptorOptions", "summary should name the options type");
  require_contains(summary, "graph=track_descriptor", "summary should name graph");
  require_contains(summary, "graph_id=237", "summary should expose graph id for diagnostics");
  require_contains(summary, "feature_shape=[2,193]", "summary should expose feature shape");
  require_contains(summary, "descriptor_shape=[2,64,8]", "summary should expose descriptor shape");

  simaai::neat::TrackDescriptor node(opt);
  require(node.kind() == "TrackDescriptor", "TrackDescriptor kind mismatch");
  const auto def = node.contract_definition();
  require(def.inputs.size() == 1U, "TrackDescriptor should have one public input");
  require(def.outputs.size() == 2U, "TrackDescriptor should have two public outputs");
  require_port(def.inputs.front(), "input_image", "TrackDescriptor input");
  require_port(def.outputs[0], "output_features", "TrackDescriptor features output");
  require_port(def.outputs[1], "output_descriptors", "TrackDescriptor descriptors output");

  auto factory = simaai::neat::nodes::TrackDescriptor(opt);
  require(factory && factory->kind() == "TrackDescriptor", "TrackDescriptor factory mismatch");
}

void test_track_klt_public_api() {
  simaai::neat::TrackKLTOptions opt;
  opt.width = 320;
  opt.height = 240;
  opt.batch_size = 2;
  opt.num_points = 32;
  opt.max_features = 64;

  const std::string no_detect_summary = opt.summary();
  require_contains(no_detect_summary, "TrackKLTOptions", "summary should name the options type");
  require_contains(no_detect_summary, "graph=track_klt", "summary should name graph");
  require_contains(no_detect_summary, "graph_id=238", "summary should expose graph id");
  require_contains(no_detect_summary, "input_points_shape=[2,32,2]",
                   "summary should expose input point shape");
  require_contains(no_detect_summary, "published_feature_shape=<disabled>",
                   "summary should reflect disabled detect path");

  simaai::neat::TrackKLT node(opt);
  require(node.kind() == "TrackKLT", "TrackKLT kind mismatch");
  auto def = node.contract_definition();
  require(def.inputs.size() == 3U, "TrackKLT should have three public inputs");
  require(def.outputs.size() == 2U, "TrackKLT no-detect should publish points/status only");
  require_port(def.inputs[0], "prev_image", "TrackKLT prev image input");
  require_port(def.inputs[1], "cur_image", "TrackKLT cur image input");
  require_port(def.inputs[2], "input_points", "TrackKLT point input");
  require_port(def.outputs[0], "output_points", "TrackKLT point output");
  require_port(def.outputs[1], "output_status", "TrackKLT status output");

  opt.detect_new_features = 1;
  const std::string detect_summary = opt.summary();
  require_contains(detect_summary, "published_feature_shape=[2,193]",
                   "summary should expose detect feature shape");
  simaai::neat::TrackKLT detect_node(opt);
  def = detect_node.contract_definition();
  require(def.outputs.size() == 3U, "TrackKLT detect should publish detected features");
  require_port(def.outputs[2], "output_features", "TrackKLT detected features output");

  auto factory = simaai::neat::nodes::TrackKLT(opt);
  require(factory && factory->kind() == "TrackKLT", "TrackKLT factory mismatch");
}

void test_metoak_depth_public_api() {
  using namespace simaai::neat;
  MetoakDepthOptions opt;
  opt.width = 640;
  opt.height = 360;
  MetoakDepth node(opt);
  const auto def = node.contract_definition();
  require(def.inputs.size() == 6U && def.outputs.size() == 3U,
          "MetoakDepth must expose all six inputs and three outputs");
  const char* inputs[] = {"y_src", "u_src", "v_src", "disp_src", "bf_mm_src", "proj_src"};
  const char* outputs[] = {"rgb_dst", "depth_dst", "points_dst"};
  for (std::size_t i = 0; i < 6U; ++i)
    require_port(def.inputs[i], inputs[i], "Metoak input");
  for (std::size_t i = 0; i < 3U; ++i)
    require_port(def.outputs[i], outputs[i], "Metoak output");
  require_contains(opt.summary(), "graph_id=20", "Metoak graph identity");
  require(nodes::MetoakDepth(opt)->kind() == "MetoakDepth", "Metoak factory");
  require(node.backend_fragment(7) == node.backend_fragment(7), "Metoak fragment deterministic");
  for (const auto& shape : {std::pair{8, 8}, std::pair{2048, 1536}}) {
    opt.width = shape.first;
    opt.height = shape.second;
    require(MetoakDepth(opt).contract_definition().outputs.size() == 3U,
            "Metoak supported envelope boundary");
  }
  for (const auto& shape : {std::pair{0, 360}, std::pair{6, 360}, std::pair{9, 360},
                            std::pair{640, 9}, std::pair{2050, 360}, std::pair{640, 1538}}) {
    opt.width = shape.first;
    opt.height = shape.second;
    bool rejected = false;
    try {
      (void)MetoakDepth(opt).contract_definition();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "Metoak invalid geometry must fail before dispatch");
  }
  opt.width = 640;
  opt.height = 360;
  opt.disp_name = "custom_disparity";
  bool rejected = false;
  try {
    (void)MetoakDepth(opt).contract_definition();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "Metoak unsupported aliases must not silently misbind");
}

void test_metoak_depth_launch_buffer_contract() {
  using namespace simaai::neat;
  MetoakDepthOptions opt;
  opt.width = 640;
  opt.height = 360;
  const auto launch = [&](const MetoakDepthOptions& options) {
    const std::vector<std::shared_ptr<Node>> graph_nodes = {
        nodes::Input(), nodes::MetoakDepth(options), nodes::Output()};
    // This is the real launch builder and strict pre-parse gate, without
    // parsing, setting pipeline state, allocating EV buffers or dispatching.
    return build_pipeline_full(graph_nodes, false, "mysink", false, {}).pipeline_string;
  };
  const std::string automatic = launch(opt);
  require_contains(automatic, "num-buffers=4", "Metoak auto pool must be explicit");
  session_build_enforce_mla_num_buffers(automatic, "Metoak default async regression", false);
  opt.num_buffers = 4;
  session_build_enforce_mla_num_buffers(launch(opt), "Metoak explicit async regression", false);
  opt.num_buffers = 2;
  const std::string explicit_two = launch(opt);
  require_contains(explicit_two, "num-buffers=2", "Metoak explicit override must remain visible");
  bool rejected = false;
  try {
    session_build_enforce_mla_num_buffers(explicit_two, "Metoak invalid async regression", false);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "Metoak must not weaken the async four-buffer gate");
  const std::string sync = session_build_clamp_sync_pipeline(automatic, 1);
  require_contains(sync, "num-buffers=2", "Sync terminal output must retain its spare buffer");
  session_build_enforce_mla_num_buffers(sync, "Metoak sync regression", true);
  rejected = false;
  try {
    session_build_enforce_mla_num_buffers("neatprocesscvu name=missing_pool async=true",
                                          "Metoak absent pool regression", false);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "Missing pool properties must still fail before pipeline parsing");
}

} // namespace

int main() {
  try {
    test_feature_histogram_public_api();
    test_grider_fast_public_api();
    test_track_descriptor_public_api();
    test_track_klt_public_api();
    test_metoak_depth_public_api();
    test_metoak_depth_launch_buffer_contract();
    std::cout << "[OK] unit_visual_frontend_node_api_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
