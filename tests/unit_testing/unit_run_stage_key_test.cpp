// The stage key of a model route's output (TensorRouteMeta::stage_key) and the Sample label drawn
// from the stage's name, on both backends, on the public API only:
//   Input -> Model::preprocess() -> Model::inference() [-> Model::postprocess()] -> Output
// with the kernel stages on the A65, built from user Graphs whose GraphOptions set an element-name
// prefix, a suffix, both, or neither.
//
// - A suffix (and a prefix with it): the stage key, the Sample's stream_label and its segment_name
//   are the same on both backends: the stage's name in the model package with the prefix and the
//   suffix (recorded by the GStreamer run, compared by the Dataflow run).
// - Neither: Dataflow gives the stage's name in the model package, the same for two Graphs built
//   one after the other; GStreamer gives that name with Core's per-process Graph number appended,
//   a different number for each Graph.
// - A prefix alone: the prefix and the package name on Dataflow; GStreamer appends its number.
//
// A process runs on one backend (set_backend): the test runs twice, --backend=gstreamer
// --reference=<file> recording, then --backend=dataflow --reference=<file> comparing. It needs the
// MLA and a compiled model package whose route is quantize + tessellate, the MLA, detessellate +
// dequantize (ResNet-50 INT8): --model-package <X_mpk.tar.gz> or SIMA_TEST_MODEL_PACKAGE. Without
// one the test is skipped (it downloads nothing).

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Result.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

// An FP32 "image" for the model's input.
std::vector<float> image(std::size_t elements) {
  std::vector<float> out(elements);
  std::uint32_t x = 12345U;
  for (auto& value : out) {
    x = x * 1664525U + 1013904223U;
    value = -2.6F + 5.7F * (static_cast<float>(x >> 8) / 16777216.0F);
  }
  return out;
}

struct Route {
  Model model;
  std::vector<std::int64_t> shape;
  std::size_t elements = 1;

  explicit Route(const std::string& archive) : model(archive, options()) {
    const auto specs = model.input_specs();
    require(specs.size() == 1U, "the test model has one input");
    shape = specs.front().shape;
    for (const auto dim : shape) {
      require(dim > 0, "the test model's input shape is fixed");
      elements *= static_cast<std::size_t>(dim);
    }
  }

  static Model::Options options() {
    Model::Options o;
    o.preprocess.kind = InputKind::Tensor;
    return o;
  }

  // `post`: the whole route (a kernel stage last); otherwise the MLA stage last.
  Graph graph(const std::string& prefix, const std::string& suffix, bool post) const {
    GraphOptions o;
    o.processcvu.pre_run_target = "A65";
    o.processcvu.post_run_target = "A65";
    o.element_name_prefix = prefix;
    o.element_name_suffix = suffix;
    OutputOptions out;
    out.max_buffers = 4;
    out.sync = false;
    Graph g(o);
    g.add(nodes::Input(model.input_appsrc_options(true)));
    g.add(model.preprocess());
    g.add(model.inference());
    if (post) {
      g.add(model.postprocess());
    }
    g.add(nodes::Output(out));
    return g;
  }
};

struct Names {
  std::string key;     // the tensor's route.stage_key
  std::string label;   // Sample::stream_label
  std::string segment; // Sample::segment_name
};

Names pulled_names(const Route& route, const std::string& prefix, const std::string& suffix,
                   bool post) {
  Graph g = route.graph(prefix, suffix, post);
  const TensorList input{
      Tensor::from_vector(image(route.elements), route.shape, TensorMemory::EV74)};
  RunOptions opt;
  opt.queue_depth = 4;
  opt.overflow_policy = OverflowPolicy::Block;
  Run run = g.build(input, opt);
  require(run.push(input), "push");
  Sample sample;
  PullError error;
  require(run.pull(30000, sample, &error) == PullStatus::Ok, "Sample pull: " + error.message);
  require(!sample.tensors.empty(), "the Sample holds a tensor");
  Names names{sample.tensors.front().route.stage_key, sample.stream_label, sample.segment_name};
  run.close();
  std::cout << "[names] prefix '" << prefix << "' suffix '" << suffix << "' "
            << (post ? "whole route" : "MLA last") << ": stage_key '" << names.key
            << "', stream_label '" << names.label << "', segment_name '" << names.segment << "'\n";
  return names;
}

bool ends_with(const std::string& value, const std::string& end) {
  return value.size() >= end.size() &&
         value.compare(value.size() - end.size(), end.size(), end) == 0;
}

// `value` is `base` followed by "_<digits>": GStreamer's per-process Graph number.
bool numbered(const std::string& value, const std::string& base) {
  if (value.size() < base.size() + 2U || value.compare(0, base.size(), base) != 0 ||
      value[base.size()] != '_') {
    return false;
  }
  for (std::size_t i = base.size() + 1U; i < value.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(value[i])) == 0) {
      return false;
    }
  }
  return true;
}

void stage_key(const Route& route, bool post, sima_test::Reference& ref) {
  const std::string row = post ? "whole route" : "MLA last";
  const bool dataflow = sima_test::on_dataflow();

  // A suffix, and a prefix with it: the same names on both backends.
  const Names suffixed = pulled_names(route, "", "_cam0", post);
  require(ends_with(suffixed.key, "_cam0"), row + ": the suffix ends the stage key");
  const std::string base = suffixed.key.substr(0, suffixed.key.size() - 5U);
  ref.equal(row + " suffix: stage_key", suffixed.key);
  ref.equal(row + " suffix: Sample stream_label", suffixed.label);
  ref.equal(row + " suffix: Sample segment_name", suffixed.segment);
  const Names both = pulled_names(route, "cam0_", "_cam0", post);
  require(both.key == "cam0_" + base + "_cam0", row + ": the prefix and the suffix frame the key");
  ref.equal(row + " prefix and suffix: stage_key", both.key);
  ref.equal(row + " prefix and suffix: Sample stream_label", both.label);
  ref.equal(row + " prefix and suffix: Sample segment_name", both.segment);
  // A label drawn from the stage's name takes the names as the key does; an output's own name
  // does not.
  const std::string label = ends_with(suffixed.label, "_cam0")
                                ? suffixed.label.substr(0, suffixed.label.size() - 5U)
                                : suffixed.label;

  // Neither: the package's name on Dataflow, for every Graph; GStreamer numbers each Graph.
  const Names first = pulled_names(route, "", "", post);
  const Names second = pulled_names(route, "", "", post);
  if (dataflow) {
    require(first.key == base && second.key == base,
            row + ": two Graphs give the stage's name in the package: '" + first.key + "', '" +
                second.key + "', '" + base + "'");
    require(first.label == label && second.label == label,
            row + ": the Sample label without a suffix: '" + first.label + "'");
  } else {
    require(numbered(first.key, base) && numbered(second.key, base) && first.key != second.key,
            row + ": GStreamer numbers each Graph: '" + first.key + "', '" + second.key + "'");
    require((first.label == label || numbered(first.label, label)),
            row + ": the Sample label without a suffix: '" + first.label + "'");
  }
  ref.prefix(row + " none: stage_key", first.key);
  ref.prefix(row + " none: Sample stream_label", first.label);

  // A prefix alone: Dataflow appends nothing; GStreamer its number.
  const Names prefixed = pulled_names(route, "cam0_", "", post);
  if (dataflow) {
    require(prefixed.key == "cam0_" + base, row + ": the prefix alone: '" + prefixed.key + "'");
  } else {
    require(numbered(prefixed.key, "cam0_" + base),
            row + ": the prefix alone, numbered: '" + prefixed.key + "'");
  }
  ref.prefix(row + " prefix: stage_key", prefixed.key);
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  if (const char* staged = std::getenv("SIMA_TEST_MODEL_PACKAGE"); archive.empty() && staged) {
    archive = staged;
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_run_stage_key_test needs a model package with quantize + tessellate "
                     "and detessellate + dequantize stages: --model-package <X_mpk.tar.gz> or "
                     "SIMA_TEST_MODEL_PACKAGE");
  }
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_run_stage_key_test", [&] {
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const Route route(archive);
    stage_key(route, true, ref);
    stage_key(route, false, ref);
    ref.finish();
  });
}
