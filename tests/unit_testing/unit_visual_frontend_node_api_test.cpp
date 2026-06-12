#include <neat.h>

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

} // namespace

int main() {
  try {
    test_feature_histogram_public_api();
    test_grider_fast_public_api();
    test_track_descriptor_public_api();
    test_track_klt_public_api();
    std::cout << "[OK] unit_visual_frontend_node_api_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
