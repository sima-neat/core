// A model package's whole route on the Dataflow backend (H1), on the public API only:
//   Input -> Model::preprocess() -> Model::inference() -> Model::postprocess() -> Output
// with the kernel stages on the A65 (GraphOptions::processcvu.pre_run_target / post_run_target =
// "A65"), on the Dataflow backend (Feed -> KernelOp -> MlaOp -> KernelOp -> Fetch) and, for the
// comparison rows, on GStreamer (appsrc ! neatkernelrunner ! neatprocessmla ! neatkernelrunner !
// appsink); and, on the Dataflow backend only, with the kernel stages on the EV74 (row ev74). A
// process runs on one backend (set_backend): the test runs twice, --backend=gstreamer
// --reference=<file> running the comparison rows and recording what they pull, then
// --backend=dataflow --reference=<file> running every row and comparing. No runtime type is
// named.
//
// It needs the MLA and a compiled model package: --model-package <X_mpk.tar.gz>, or
// SIMA_TEST_MODEL_PACKAGE (SIMA_TEST_MLA_ONLY_MODEL is read too: the same archive serves both
// tests). Without one the test is skipped (it downloads nothing). The model has one FP32 input
// that the package quantizes and tessellates, one MLA stage and one output that the package
// detessellates and dequantizes; ResNet-50 INT8 is the one it was written with.
//
// A kernel stage on the EV74 is only ever asked of the Dataflow backend (no fallback), and only
// run where the EV74 runs the Kernels firmware (the
// remoteproc "ev74" running "sima-neat-native-cvu-fw", read here from sysfs). On a board with the
// stock firmware the same builds must be refused at the build, before anything runs, and the
// rows check that instead. GStreamer is never built with a kernel stage on the EV74 here.
//
// Rows:
//   differential     the same inputs (4 images in rotation) on each backend: every pulled
//                    tensor's dtype, layout, shape, strides, byte offset, route, quantization and
//                    format tag, the Sample's shape, and every byte (its digest). Excluded, each
//                    printed with both values: the storage's size (GStreamer: the stage's
//                    allocation; Dataflow: the kernel's output buffer), frame_id / input_seq /
//                    timestamps / caps_string (documented for every Dataflow Run:
//                    unit_run_backend_differential_test).
//   reference        the logits against a reference computed here: the image quantized and
//                    tessellated in the test, inferred through the MLA-only route, dequantized in
//                    the test. Both backends against it, for every image.
//   partial          the two shorter routes: preprocess + inference of an MLA-only terminal
//                    (kernel -> MLA) and inference + postprocess of a model without preprocess
//                    (MLA -> kernel): both backends, the same bytes. The second is not run for
//                    a model whose MLA input buffer is longer than its INT8 tensor (a buffer
//                    rounded up to 16 bytes): neither backend takes that tensor there.
//   input_contract   which tensors the route takes: TensorMemory::EV74 and MLA tensors, with and
//                    without the batch axis, seeded and not; a CPU tensor is refused with
//                    GStreamer's guard message; a tensor of another dtype or size is refused.
//   reuse            what applications do with their input tensors: one tensor rewritten in place
//                    between pushes; a new tensor for every frame (300 of them).
//   held             what the Run keeps alive of pushed tensors the application let go of: of 96
//                    new tensors pushed once each, never more than the first operator's bound
//                    (32 of the input's size, the runtime's held_buffers) and one in flight;
//                    all of them gone once the Run is closed. Under the default placement and
//                    with both kernel stages on the A65.
//   lifecycle        a pull before any push times out; try_push; close_input drains what was
//                    pushed and then closes; a push after it is refused; stop twice.
//   in_flight        a Run destroyed with frames queued, in the kernels and on the MLA; then a
//                    new Run works; stop with outputs unpulled.
//   recreate         Runs one after the other in one process give the same outputs.
//   owned            OutputMemory::Owned: a CPU copy with the same bytes, which outlives its Run.
//   alloc            Run::alloc lends a slot of the input's device pool; what the CPU writes
//                    there is what the first kernel reads.
//   policies         try_push under KeepLatest and DropIncoming: what the input's queue lets
//                    through runs through every stage.
//   declined         what stays off this backend: OutputOptions::drop behind a model, and, on a
//                    board whose EV74 runs the stock firmware, a kernel stage placed on the EV74
//                    (the default placement of the pre stage, and an explicit one of the post
//                    stage): explicit Dataflow fails at the build.
//   ev74             the kernel stages on the EV74, four placements (the default one: pre on the
//                    EV74, post on the A65; EV74 + A65 asked; A65 + EV74; both on the EV74): every
//                    field and every byte of 64 frames against GStreamer's with both stages on
//                    the A65; then, under each placement, the rows input_contract (a
//                    TensorMemory::MLA tensor in front of a kernel on the EV74 is copied, with
//                    the output of the same bytes in EV74 memory), reuse, held, lifecycle,
//                    in_flight, recreate, owned, alloc and policies.

#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/Result.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

constexpr int kPatterns = 4;

std::string dims(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(shape[i]);
  }
  return out + "]";
}

// An "image": FP32 values in [-2.6, 3.1), past both ends of what ResNet-50's input quantization
// represents, so that the saturation of the quantizer is exercised too.
std::vector<float> image(std::size_t elements, unsigned index) {
  std::vector<float> out(elements);
  std::uint32_t x = 0x9e3779b9U * (index + 1U) + 12345U;
  for (auto& value : out) {
    x = x * 1664525U + 1013904223U;
    value = -2.6F + 5.7F * (static_cast<float>(x >> 8) / 16777216.0F);
  }
  return out;
}

std::vector<std::uint8_t> bytes_of(const Tensor& tensor) {
  return tensor.copy_dense_bytes_tight();
}

GraphOptions on_a65() {
  GraphOptions o;
  o.processcvu.pre_run_target = "A65";
  o.processcvu.post_run_target = "A65";
  return o;
}

// Where the Dataflow-only rows place the two kernel stages (row ev74 moves it).
GraphOptions& placement() {
  static GraphOptions options = on_a65();
  return options;
}

GraphOptions placed(const char* pre, const char* post) {
  GraphOptions o;
  if (pre) {
    o.processcvu.pre_run_target = pre;
  }
  if (post) {
    o.processcvu.post_run_target = post;
  }
  return o;
}

// The EV74 runs the firmware of the Kernels package: the remoteproc named "ev74" is running
// "sima-neat-native-cvu-fw".
bool ev74_runs_kernels_firmware() {
  const auto read = [](const std::string& path) {
    std::ifstream file(path);
    std::string text;
    std::getline(file, text);
    return text;
  };
  for (int i = 0; i < 16; ++i) {
    const std::string dir = "/sys/class/remoteproc/remoteproc" + std::to_string(i) + "/";
    if (read(dir + "name") == "ev74") {
      return read(dir + "state") == "running" &&
             read(dir + "firmware") == "sima-neat-native-cvu-fw";
    }
  }
  return false;
}

struct Route {
  Model model;
  std::vector<std::int64_t> shape; // the model's input, with its batch axis
  std::size_t elements = 0;

  explicit Route(const std::string& archive) : model(archive, options()) {
    const auto specs = model.input_specs();
    require(specs.size() == 1U, "the test model has one input");
    shape = specs.front().shape;
    elements = 1;
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

  // The route's Input: the model's own options (its public FP32 tensor).
  InputOptions input() const {
    return model.input_appsrc_options(true);
  }

  // `named`: named endpoints, a graph Run (built without a seed too; a plain linear Run is not).
  Graph graph(const InputOptions& in, int max_buffers = 4, bool drop = false,
              const GraphOptions& options = placement(), bool named = false) const {
    OutputOptions out;
    out.max_buffers = max_buffers;
    out.drop = drop;
    out.sync = false;
    Graph g(options);
    g.add(named ? nodes::Input("in", in) : nodes::Input(in));
    g.add(model.preprocess());
    g.add(model.inference());
    g.add(model.postprocess());
    g.add(named ? nodes::Output("out", out) : nodes::Output(out));
    return g;
  }

  Tensor tensor(unsigned index, TensorMemory memory = TensorMemory::EV74) const {
    return Tensor::from_vector(image(elements, index), shape, memory);
  }
};

RunOptions on(int depth = 4) {
  RunOptions o;
  o.queue_depth = depth;
  o.overflow_policy = OverflowPolicy::Block;
  return o;
}

TensorList pull_ok(Run& run, const std::string& what) {
  Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
  require(static_cast<bool>(out), what + ": pull failed: " + (out ? "" : out.error().message));
  require(out->size() == 1U, what + ": one tensor per inference");
  return std::move(*out);
}

// ---- differential ------------------------------------------------------------------------------

// One pulled frame, kept without its storage: a pulled tensor holds an output buffer of the Run
// (GStreamer lends a bounded number of them), so the test keeps the facts and the bytes.
struct Frame {
  Tensor tensor;                   // the pulled tensor's description; storage dropped
  std::vector<std::uint8_t> bytes; // its dense bytes
  StorageKind storage_kind = StorageKind::Unknown;
  DeviceType storage_device = DeviceType::UNKNOWN;
  std::uint64_t storage_target_flags = 0;
  std::size_t storage_segments = 0;
  std::size_t storage_bytes = 0;
  std::string storage_segment;
  std::int64_t frame_id = -1;
  std::uint64_t seq = 0;
  bool from_sample = false;
  Sample sample; // the Sample's own fields (its tensors dropped)
};

Frame keep(Tensor tensor) {
  Frame frame;
  frame.bytes = bytes_of(tensor);
  require(static_cast<bool>(tensor.storage), "a pulled tensor has storage");
  frame.storage_kind = tensor.storage->kind;
  frame.storage_device = tensor.storage->device.type;
  frame.storage_target_flags = tensor.storage->sima_mem_target_flags;
  frame.storage_segments = tensor.storage->sima_segments.size();
  frame.storage_segment =
      tensor.storage->sima_segments.empty() ? "" : tensor.storage->sima_segments.front().name;
  frame.storage_bytes = tensor.storage->size_bytes;
  frame.frame_id = tensor.stamp().frame_id;
  frame.seq = tensor.stamp().seq;
  tensor.storage.reset();
  tensor.envelope.reset();
  frame.tensor = std::move(tensor);
  return frame;
}

std::vector<Frame> run_frames(Graph g, const std::vector<TensorList>& inputs, int frames) {
  Run run = g.build(inputs[0], on());
  std::vector<Frame> seen;
  for (int i = 0; i < frames; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), "push");
    if ((i % 2) == 0) {
      seen.push_back(keep(pull_ok(run, "differential").front()));
    } else {
      Sample sample;
      PullError error;
      require(run.pull(30000, sample, &error) == PullStatus::Ok,
              "differential: Sample pull: " + error.message);
      require(sample.tensors.size() == 1U, "differential: the Sample holds one tensor");
      Frame frame = keep(sample.tensors.front());
      sample.tensors.clear();
      frame.from_sample = true;
      frame.sample = std::move(sample);
      seen.push_back(std::move(frame));
    }
  }
  run.close_input();
  Sample none;
  PullError closed;
  require(run.pull(5000, none, &closed) == PullStatus::Closed,
          "differential: closed after the drain: " + closed.message);
  run.stop();
  return seen;
}

// A device buffer's storage is each backend's own kind (GStreamer's GstSample, Dataflow's
// DeviceHandle: qualification/expected_differences.json, device_storage_kind); the differential
// compares it as the one "device buffer", every other kind as it is.
std::string storage_kind_text(StorageKind kind) {
  const StorageKind device =
      sima_test::on_dataflow() ? StorageKind::DeviceHandle : StorageKind::GstSample;
  return kind == device ? std::string("device buffer") : sima_test::text(kind);
}

// Every field a user reads of a backend's frames, and every byte: recorded by the GStreamer run
// and compared by the Dataflow run (`row` names the record's keys). GStreamer numbers some
// elements per pipeline built in the process (MLA_0_<n>), so the stage's name is compared by
// prefix.
void compare(sima_test::Reference& ref, const std::vector<Frame>& frames, const std::string& row) {
  ref.equal(row + ": frames", frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto at = [&](const char* what) {
      return row + " frame " + std::to_string(i) + ": " + what;
    };
    const Frame& f = frames[i];
    const Tensor& t = f.tensor;
    ref.equal(at("dtype"), t.dtype);
    ref.equal(at("layout"), t.layout);
    ref.equal(at("shape"), t.shape);
    ref.equal(at("strides"), t.strides_bytes);
    ref.equal(at("byte_offset"), t.byte_offset);
    ref.equal(at("axis_semantics"), t.axis_semantics);
    ref.equal(at("device"), sima_test::text(t.device.type) + "/" + sima_test::text(t.device.id));
    ref.equal(at("read_only"), t.read_only);
    ref.equal(at("planes"), t.planes.size());
    ref.equal(at("route"), sima_test::text(t.route.logical_index) + "," +
                               sima_test::text(t.route.backend_output_index) + "," +
                               sima_test::text(t.route.route_slot) + "," +
                               sima_test::text(t.route.physical_index) + "," +
                               sima_test::text(t.route.memory_index) + "," +
                               sima_test::text(t.route.physical_byte_offset) + ",name '" +
                               t.route.name + "',backend '" + t.route.backend_name + "',segment '" +
                               t.route.segment_name + "'");
    ref.prefix(at("route.stage_key"), t.route.stage_key);
    ref.equal(at("semantic.tess"),
              t.semantic.tess ? sima_test::text(t.semantic.tess->format) : std::string("none"));
    ref.equal(at("semantic.quant"), t.semantic.quant.has_value());
    require(!t.semantic.image, at("no image semantic"));
    ref.equal(at("storage kind, placement, segments"),
              storage_kind_text(f.storage_kind) + "/" + sima_test::text(f.storage_device) + "/" +
                  sima_test::text(f.storage_target_flags) + "/" +
                  sima_test::text(f.storage_segments) + "/'" + f.storage_segment + "'");
    require(!f.bytes.empty(), at("bytes"));
    ref.equal(at("bytes"), sima_test::bytes_digest(f.bytes));
    ref.equal(at("from a Sample"), f.from_sample);
    if (f.from_sample) {
      const Sample& sa = f.sample;
      ref.equal(at("Sample kind / owned"),
                sima_test::text(sa.kind) + "/" + sima_test::text(sa.owned));
      ref.equal(at("Sample media"), sima_test::text(sa.media_type) + "/" +
                                        sima_test::text(sa.payload_type) + "/'" + sa.payload_tag +
                                        "'/" + sima_test::text(sa.format));
      ref.equal(at("Sample routing"),
                sima_test::text(sa.output_index) + "/" + sima_test::text(sa.logical_output_index) +
                    "/" + sima_test::text(sa.route_slot) + "/" + sima_test::text(sa.memory_index));
      ref.equal(at("Sample stream_id"), sa.stream_id);
      ref.prefix(at("Sample stream_label"), sa.stream_label);
      ref.prefix(at("Sample segment_name"), sa.segment_name);
    }
  }
  if (frames.empty()) {
    return;
  }
  ref.note(row + " excluded: stage_key", frames[0].tensor.route.stage_key);
  ref.note(row + " excluded: storage bytes", frames[0].storage_bytes);
  ref.note(row + " excluded: first frame_id", frames[0].frame_id);
  ref.note(row + " excluded: first seq", frames[0].seq);
  ref.note(row + " excluded: caps_string", frames.size() > 1U ? frames[1].sample.caps_string : "");
}

std::vector<TensorList> images(const Route& route) {
  std::vector<TensorList> inputs;
  for (int p = 0; p < kPatterns; ++p) {
    inputs.push_back({route.tensor(static_cast<unsigned>(p))});
  }
  return inputs;
}

void differential(const Route& route, sima_test::Reference& ref) {
  constexpr int kFrames = 64;
  const std::vector<TensorList> inputs = images(route);
  const std::vector<Frame> df = run_frames(route.graph(route.input()), inputs, kFrames);
  compare(ref, df, "differential");
  // The four inputs give four different outputs, and each input the same one every time.
  for (int p = 1; p < kPatterns; ++p) {
    require(df[0].bytes != df[static_cast<std::size_t>(p)].bytes,
            "the outputs of different inputs differ");
    require(df[static_cast<std::size_t>(p)].bytes ==
                df[static_cast<std::size_t>(p + kPatterns)].bytes,
            "an input gives the same output again");
  }
}

// ---- the MLA's own input ----------------------------------------------------------------------

// Whether the MLA's input buffer is longer than the model's INT8 tensor of `elements` bytes: the
// model compiler makes the buffer a multiple of 16 bytes. Then the model's tensor is not what a
// route whose first stage is the MLA takes, on either backend (GStreamer's inference-only ingress
// refuses it by its count of bytes and builds no such route for the buffer either), and the rows
// that push it there are not run.
bool padded_mla_input(std::size_t elements) {
  return elements % 16U != 0U;
}

// ---- reference ---------------------------------------------------------------------------------

struct Dequant {
  double scale = 7.11897298351226; // ResNet-50 INT8, dequantize_1 of its package
  int zero_point = -63;
};

// The MLA's input for `values`, as the package's pre stage makes it: each value multiplied by the
// quantization scale, rounded to nearest even, the zero point added, saturated to INT8; then the
// tensor [1,H,W,C] cut into tiles of slice [h,w,c], tile after tile in raster order, each tile's
// values in H, W, C order (every tile whole and the channels compact: H, W divisible, c == C).
std::vector<std::int8_t> quantize_and_tile(const std::vector<float>& values,
                                           const std::vector<std::int64_t>& shape,
                                           const std::vector<int>& slice, float scale, int zero) {
  const auto h = static_cast<std::size_t>(shape[1]);
  const auto w = static_cast<std::size_t>(shape[2]);
  const auto c = static_cast<std::size_t>(shape[3]);
  const auto th = static_cast<std::size_t>(slice[0]);
  const auto tw = static_cast<std::size_t>(slice[1]);
  std::vector<std::int8_t> out(values.size());
  std::size_t at = 0;
  for (std::size_t ty = 0; ty < h; ty += th) {
    for (std::size_t tx = 0; tx < w; tx += tw) {
      for (std::size_t y = ty; y < ty + th; ++y) {
        for (std::size_t x = tx; x < tx + tw; ++x) {
          for (std::size_t k = 0; k < c; ++k) {
            const float scaled = values[(y * w + x) * c + k] * scale;
            const long q = std::lrintf(scaled) + zero; // the default rounding mode: nearest even
            out[at++] = static_cast<std::int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
          }
        }
      }
    }
  }
  return out;
}

void reference(const Route& route, const std::string& archive, const Dequant& dequant) {
  const Model::PreprocessRequirements pre = route.model.preprocess_requirements();
  if (!pre.quant_needed || !pre.tess_needed || !pre.q_scale || !pre.q_zp ||
      pre.slice_shape.size() != 3U || route.shape.size() != 4U || route.shape[0] != 1 ||
      route.shape[1] % pre.slice_shape[0] != 0 || route.shape[2] % pre.slice_shape[1] != 0 ||
      route.shape[3] != pre.slice_shape[2]) {
    std::cout << "[reference] not computed for this model (its pre stage is not a whole-tile "
                 "quantize + tessellate)\n";
    return;
  }
  if (padded_mla_input(route.elements)) {
    std::cout << "[reference] not computed for this model (its MLA input buffer is longer than "
                 "its tensor: the MLA alone is no route for the tensor)\n";
    return;
  }
  // The reference: the test's own quantization and tessellation, the MLA alone, the test's own
  // dequantization.
  Model::Options mla_only;
  mla_only.preprocess.kind = InputKind::Tensor;
  mla_only.preprocess.enable = AutoFlag::Off;
  mla_only.inference_terminal.mla_only = true;
  const Model mla(archive, mla_only);
  InputOptions in = mla.input_appsrc_options(true);
  in.payload_type = PayloadType::Tensor;
  in.format = FormatTag::INT8;
  in.dtype = TensorDType::Int8;
  std::vector<std::vector<float>> expected;
  {
    Graph g;
    g.add(nodes::Input(in));
    g.add(mla.inference());
    g.add(nodes::Output());
    std::vector<TensorList> coded;
    for (int p = 0; p < kPatterns; ++p) {
      coded.push_back({Tensor::from_vector(
          quantize_and_tile(image(route.elements, static_cast<unsigned>(p)), route.shape,
                            pre.slice_shape, static_cast<float>(*pre.q_scale),
                            static_cast<int>(*pre.q_zp)),
          route.shape, TensorMemory::MLA)});
    }
    Run run = g.build(coded[0], on());
    const float scale = 1.0F / static_cast<float>(dequant.scale);
    for (int p = 0; p < kPatterns; ++p) {
      require(run.push(coded[static_cast<std::size_t>(p)]), "reference: push");
      const std::vector<std::uint8_t> codes = bytes_of(pull_ok(run, "reference").front());
      std::vector<float> logits(codes.size());
      for (std::size_t i = 0; i < codes.size(); ++i) {
        logits[i] = static_cast<float>(static_cast<int>(static_cast<std::int8_t>(codes[i])) -
                                       dequant.zero_point) *
                    scale;
      }
      expected.push_back(std::move(logits));
    }
    run.close();
  }
  {
    Graph g = route.graph(route.input());
    const std::vector<TensorList> inputs = images(route);
    Run run = g.build(inputs[0], on());
    std::size_t different = 0;
    std::size_t compared = 0;
    double worst = 0;
    for (int p = 0; p < kPatterns; ++p) {
      require(run.push(inputs[static_cast<std::size_t>(p)]), "reference: push the image");
      const std::vector<std::uint8_t> out = bytes_of(pull_ok(run, "reference route").front());
      const std::vector<float>& want = expected[static_cast<std::size_t>(p)];
      require(out.size() == want.size() * sizeof(float),
              "reference: the route's output is one FP32 per MLA output value");
      for (std::size_t i = 0; i < want.size(); ++i) {
        float got = 0;
        std::memcpy(&got, out.data() + i * sizeof(float), sizeof(float));
        const double d = std::fabs(static_cast<double>(got) - static_cast<double>(want[i]));
        different += d != 0 ? 1U : 0U;
        worst = d > worst ? d : worst;
        ++compared;
      }
    }
    run.close();
    std::cout << "[reference] " << sima_test::backend_name() << ": of " << compared << " logits "
              << different << " differ from the reference computed in the test (worst " << worst
              << ")\n";
    require(different == 0, "the route's logits are the reference's, bit for bit");
  }
}

// ---- partial -----------------------------------------------------------------------------------

void partial(const Route& route, const std::string& archive, sima_test::Reference& ref) {
  // preprocess + inference, the inference ending at the MLA: kernel -> MLA.
  {
    Model::Options o = Route::options();
    o.inference_terminal.mla_only = true;
    const Model model(archive, o);
    const auto graph = [&] {
      Graph g(on_a65());
      g.add(nodes::Input(model.input_appsrc_options(true)));
      g.add(model.preprocess());
      g.add(model.inference());
      g.add(nodes::Output());
      return g;
    };
    const std::vector<TensorList> inputs = images(route);
    compare(ref, run_frames(graph(), inputs, 16), "partial kernel -> MLA");
  }
  // inference + postprocess of a model without preprocess: MLA -> kernel. Its input is the MLA's.
  if (padded_mla_input(route.elements)) {
    std::cout << "[partial] MLA -> kernel: not run for this model (its MLA input buffer is longer "
                 "than its tensor of "
              << route.elements << " bytes, which neither backend takes in front of the MLA)\n";
  } else {
    Model::Options o = Route::options();
    o.preprocess.enable = AutoFlag::Off;
    const Model model(archive, o);
    InputOptions in = model.input_appsrc_options(true);
    in.payload_type = PayloadType::Tensor;
    in.format = FormatTag::INT8;
    in.dtype = TensorDType::Int8;
    const auto graph = [&] {
      Graph g(on_a65());
      g.add(nodes::Input(in));
      g.add(model.inference());
      g.add(model.postprocess());
      g.add(nodes::Output());
      return g;
    };
    std::vector<TensorList> inputs;
    for (int p = 0; p < kPatterns; ++p) {
      std::vector<std::int8_t> codes(route.elements);
      std::uint32_t x = 0x9e3779b9U * (static_cast<unsigned>(p) + 1U) + 12345U;
      for (auto& value : codes) {
        x = x * 1664525U + 1013904223U;
        value = static_cast<std::int8_t>(static_cast<unsigned char>(x >> 24));
      }
      inputs.push_back({Tensor::from_vector(codes, route.shape, TensorMemory::MLA)});
    }
    compare(ref, run_frames(graph(), inputs, 16), "partial MLA -> kernel");
  }
}

// ---- input contract ----------------------------------------------------------------------------

// What a push of `tensor` does: "accepted" with an output pulled, or the error's text.
std::string push_outcome(const Route& route, const InputOptions& in, const Tensor& tensor,
                         const TensorList* seed) {
  Graph g = route.graph(in, 4, false, placement(), /*named=*/seed == nullptr);
  std::string outcome;
  try {
    Run run = seed ? g.build(*seed, on()) : g.build(on());
    outcome = run.push({tensor}) ? "accepted" : "push returned false";
    if (outcome == "accepted") {
      (void)pull_ok(run, "input_contract");
    }
    run.close();
  } catch (const std::exception& error) {
    outcome = error.what();
  }
  return outcome;
}

void input_contract(const Route& route) {
  const InputOptions in = route.input();
  const TensorList seed{route.tensor(0)};
  require(push_outcome(route, in, route.tensor(1, TensorMemory::EV74), &seed) == "accepted",
          "a TensorMemory::EV74 tensor is taken");
  require(push_outcome(route, in, route.tensor(1, TensorMemory::MLA), &seed) == "accepted",
          "a TensorMemory::MLA tensor is taken");
  require(push_outcome(route, in, route.tensor(1, TensorMemory::EV74), nullptr) == "accepted",
          "an unseeded Run binds by its first push");
  if (route.shape.size() == 4U && route.shape.front() == 1) {
    // Without the batch axis: seeded with it and without it, and both in one Run.
    const std::vector<std::int64_t> unbatched(route.shape.begin() + 1, route.shape.end());
    const Tensor t = Tensor::from_vector(image(route.elements, 1), unbatched, TensorMemory::EV74);
    const TensorList unbatched_seed{t};
    require(push_outcome(route, in, t, &unbatched_seed) == "accepted",
            "the input without its batch axis is taken");
    require(push_outcome(route, in, t, &seed) == "accepted",
            "the input without its batch axis is taken by a Run seeded with it");
    require(push_outcome(route, in, t, nullptr) == "accepted",
            "the input without its batch axis binds an unseeded Run");
    // The same bytes with and without the axis give the same output.
    Graph g = route.graph(in);
    Run run = g.build(seed, on());
    require(run.push({route.tensor(1)}), "push with the batch axis");
    const std::vector<std::uint8_t> batched = bytes_of(pull_ok(run, "batched").front());
    require(run.push({t}), "push without the batch axis");
    require(bytes_of(pull_ok(run, "unbatched").front()) == batched,
            "the output does not depend on the batch axis");
    run.close();
  }
  // GStreamer's guard of a device-visible route, with its message.
  const std::string cpu = push_outcome(route, in, route.tensor(1, TensorMemory::CPU), &seed);
  require_contains(cpu, "CPU-backed Tensor pushed into a device-visible EV74/DMS route",
                   "a CPU tensor is refused as on GStreamer: " + cpu);
  // Another dtype.
  const std::vector<std::int8_t> codes(route.elements, 7);
  const std::string int8 =
      push_outcome(route, in, Tensor::from_vector(codes, route.shape, TensorMemory::EV74), &seed);
  require(int8 != "accepted", "an INT8 tensor is refused: " + int8);
  std::cout << "[input_contract] INT8 under the FP32 declaration: " << int8 << "\n";
  // Another size.
  if (route.shape.size() == 4U) {
    std::vector<std::int64_t> half = route.shape;
    half[1] = std::max<std::int64_t>(1, half[1] / 2);
    std::size_t n = 1;
    for (const auto dim : half) {
      n *= static_cast<std::size_t>(dim);
    }
    const Tensor small = Tensor::from_vector(image(n, 1), half, TensorMemory::EV74);
    const std::string outcome = push_outcome(route, in, small, &seed);
    require(outcome != "accepted", "a tensor of another size is refused: " + outcome);
    std::cout << "[input_contract] another size: " << outcome << "\n";
  }
  // A declared shape that is not the model's input is the build's error.
  {
    InputOptions other = in;
    other.shape = route.shape;
    other.shape.back() += 1;
    require_neat_error(
        [&] {
          Graph g = route.graph(other);
          (void)g.build(on());
        },
        error_codes::kInputShape, "the model's input is");
  }
}

// ---- reuse -------------------------------------------------------------------------------------

void reuse(const Route& route) {
  const std::vector<TensorList> inputs = images(route);
  std::vector<std::vector<std::uint8_t>> expected;
  Graph g = route.graph(route.input());
  Run run = g.build(inputs[0], on());
  for (int p = 0; p < kPatterns; ++p) {
    require(run.push(inputs[static_cast<std::size_t>(p)]), "push");
    expected.push_back(bytes_of(pull_ok(run, "reuse reference").front()));
  }
  // One tensor, rewritten in place between pushes: each output is the bytes it held at its push.
  Tensor one = route.tensor(0);
  for (int i = 0; i < 12; ++i) {
    const int p = (i * 3 + 1) % kPatterns;
    const std::vector<float> values = image(route.elements, static_cast<unsigned>(p));
    {
      Mapping map = one.map_write();
      require(map.data != nullptr && map.size_bytes >= values.size() * sizeof(float),
              "a pushed tensor maps for write again once its frame was pulled");
      std::memcpy(map.data, values.data(), values.size() * sizeof(float));
    }
    require(run.push({one}), "push the rewritten tensor");
    require(bytes_of(pull_ok(run, "rewritten").front()) == expected[static_cast<std::size_t>(p)],
            "the output of a tensor rewritten in place is its new bytes' (round " +
                std::to_string(i) + ")");
  }
  // A new tensor for every frame.
  for (int i = 0; i < 300; ++i) {
    const int p = i % kPatterns;
    require(run.push({route.tensor(static_cast<unsigned>(p))}), "push a new tensor");
    require(bytes_of(pull_ok(run, "new tensor").front()) == expected[static_cast<std::size_t>(p)],
            "the output of new tensor " + std::to_string(i));
  }
  run.close();
}

// ---- held --------------------------------------------------------------------------------------

// The Run wraps a pushed device tensor and its first operator keeps the wrap's holder, which is
// the tensor's storage, with the device's admission of it. That memory is the EV74's or the MLA's
// heap, so what a Run keeps of tensors the application dropped is bounded (the runtime's
// held_buffers: a count under a byte cap): here, 32 tensors of the model's input.
void held(const Route& route) {
  constexpr int kFrames = 96;
  constexpr std::size_t kKeep = 32;
  Graph g = route.graph(route.input());
  Run run = g.build({route.tensor(0)}, on());
  std::vector<std::weak_ptr<TensorBuffer>> pushed;
  std::size_t most = 0;
  std::size_t last = 0;
  for (int i = 0; i < kFrames; ++i) {
    {
      Tensor tensor = route.tensor(static_cast<unsigned>(i % kPatterns));
      pushed.emplace_back(tensor.storage);
      require(run.push({tensor}), "held: push a new tensor");
    }
    (void)pull_ok(run, "held");
    last = static_cast<std::size_t>(std::count_if(
        pushed.begin(), pushed.end(), [](const auto& storage) { return !storage.expired(); }));
    most = std::max(most, last);
  }
  std::cout << "[held] " << kFrames << " new tensors of " << route.elements * sizeof(float)
            << " bytes pushed once each: at most " << most << " kept alive at once ("
            << most * route.elements * sizeof(float) << " bytes), " << last << " after the last\n";
  require(most <= kKeep + 1U, "held: the Run keeps at most 32 pushed tensors and the one in "
                              "flight alive");
  run.close();
  const auto alive = std::count_if(pushed.begin(), pushed.end(),
                                   [](const auto& storage) { return !storage.expired(); });
  require(alive == 0, "held: a closed Run keeps no pushed tensor alive");
}

// ---- lifecycle ---------------------------------------------------------------------------------

void lifecycle(const Route& route) {
  Graph g = route.graph(route.input(), /*max_buffers=*/8);
  const TensorList a{route.tensor(0)};
  const TensorList b{route.tensor(1)};
  Run run = g.build(a, on(8));
  require(run.running() && run.can_push() && run.can_pull(), "a built Run runs");

  // Nothing pushed: a pull times out and says so; a poll says nothing.
  Sample sample;
  PullError error;
  require(run.pull(50, sample, &error) == PullStatus::Timeout, "a pull before any push times out");
  require(run.pull(0, sample, &error) == PullStatus::Timeout && error.message.empty(),
          "a poll before any push is silent");

  const std::vector<std::uint8_t> out_a = [&] {
    require(run.push(a), "push a");
    return bytes_of(pull_ok(run, "lifecycle a").front());
  }();
  const std::vector<std::uint8_t> out_b = [&] {
    require(run.try_push(b), "try_push b into an empty Run");
    return bytes_of(pull_ok(run, "lifecycle b").front());
  }();
  require(out_a != out_b, "two inputs, two outputs");

  // close_input: what was pushed still runs through every stage and is pulled, in order, then the
  // Run is closed.
  for (int i = 0; i < 6; ++i) {
    require(run.push((i % 2) == 0 ? a : b), "push before close_input");
  }
  run.close_input();
  for (int i = 0; i < 6; ++i) {
    require(bytes_of(pull_ok(run, "drain").front()) == ((i % 2) == 0 ? out_a : out_b),
            "the drained output " + std::to_string(i) + " is its input's");
  }
  require(run.pull(5000, sample, &error) == PullStatus::Closed, "closed after the drain");
  require(!run.push(a), "a push after close_input is refused");
  require(!run.try_push(a), "a try_push after close_input is refused");
  run.stop();
  run.stop();
  require(!run.running(), "stopped");
  // What last_error says is the refused push's reason, as on GStreamer (develop #1431): no error
  // of the Run was latched.
  require(run.last_error() == "[runtime.pull] Run::push: input is closed",
          "last_error is the reason of the refused push: " + run.last_error());
}

// A Run that goes away with frames queued in front of the first kernel, in the kernels, on the
// MLA and waiting to be pulled.
void in_flight(const Route& route) {
  const TensorList a{route.tensor(0)};
  std::vector<std::uint8_t> expected;
  {
    Graph g = route.graph(route.input(), /*max_buffers=*/4);
    Run run = g.build(a, on(4));
    require(run.push(a), "push");
    expected = bytes_of(pull_ok(run, "in_flight first").front());
    for (int i = 0; i < 10; ++i) {
      require(run.try_push(a) || i > 0, "the first try_push finds room");
    }
    // No pull, no close_input, no stop: the destructor ends the Run.
  }
  {
    Graph g = route.graph(route.input(), 4);
    Run run = g.build(a, on(4));
    require(run.push(a), "push into the next Run");
    require(bytes_of(pull_ok(run, "in_flight next").front()) == expected,
            "the next Run infers as the first");
    for (int i = 0; i < 3; ++i) {
      require(run.push(a), "push before stop");
    }
    run.stop(); // with outputs unpulled
    Sample sample;
    PullError error;
    const PullStatus after = run.pull(100, sample, &error);
    require(after == PullStatus::Closed || after == PullStatus::Ok,
            "a pull after stop is closed, or hands out what was ready");
    require(!run.push(a), "a push after stop is refused");
  }
}

void recreate(const Route& route) {
  const TensorList a{route.tensor(2)};
  std::vector<std::uint8_t> first;
  for (int round = 0; round < 4; ++round) {
    Graph g = route.graph(route.input());
    Run run = g.build(a, on());
    for (int i = 0; i < 8; ++i) {
      require(run.push(a), "push");
      const std::vector<std::uint8_t> out = bytes_of(pull_ok(run, "recreate").front());
      if (first.empty()) {
        first = out;
      }
      require(out == first, "round " + std::to_string(round) + " frame " + std::to_string(i));
    }
    run.close_input();
    run.stop();
  }
}

void owned(const Route& route) {
  const TensorList a{route.tensor(3)};
  std::vector<std::uint8_t> zero_copy;
  {
    Graph g = route.graph(route.input());
    Run run = g.build(a, on());
    require(run.push(a), "push");
    const Tensor t = pull_ok(run, "owned reference").front();
    require(t.storage && t.storage->kind == (sima_test::on_dataflow() ? StorageKind::DeviceHandle
                                                                      : StorageKind::GstSample),
            "the default output is the last stage's buffer");
    zero_copy = bytes_of(t);
    run.close();
  }
  Graph g = route.graph(route.input());
  RunOptions o = on();
  o.output_memory = OutputMemory::Owned;
  Run run = g.build(a, o);
  Tensor kept;
  for (int i = 0; i < 3; ++i) {
    require(run.push(a), "push");
    kept = pull_ok(run, "owned").front();
    require(kept.storage && kept.storage->kind == StorageKind::CpuOwned, "an Owned pull is a copy");
    require(bytes_of(kept) == zero_copy, "the copy holds the output");
  }
  run.close();
  require(bytes_of(kept) == zero_copy, "the copy outlives its Run");
}

// Run::alloc: the application writes the image into a slot of the route's device pool.
void alloc(const Route& route) {
  const TensorList reference_input{route.tensor(1)};
  std::vector<std::uint8_t> expected;
  {
    Graph g = route.graph(route.input());
    Run run = g.build(reference_input, on());
    require(run.push(reference_input), "push");
    expected = bytes_of(pull_ok(run, "alloc reference").front());
    run.close();
  }
  InputOptions in = route.input();
  in.shape = route.shape;
  in.dtype = TensorDType::Float32;
  OutputOptions out;
  out.max_buffers = 4;
  Graph g(placement());
  g.add(nodes::Input("x", in));
  g.add(route.model.preprocess());
  g.add(route.model.inference());
  g.add(route.model.postprocess());
  g.add(nodes::Output("y", out));
  Run run = g.build(on());
  const std::vector<float> values = image(route.elements, 1);
  for (int i = 0; i < 3; ++i) {
    Tensor t = run.alloc("x");
    {
      Mapping map = t.map_write();
      require(map.data != nullptr && map.size_bytes >= values.size() * sizeof(float),
              "the slot maps for write");
      std::memcpy(map.data, values.data(), values.size() * sizeof(float));
    }
    require(run.push("x", {t}), "push the allocated tensor");
    Result<TensorList> pulled = run.pull("y", std::chrono::milliseconds(30000));
    require(pulled && pulled->size() == 1U, "pull");
    require(bytes_of(pulled->front()) == expected, "the kernel read what the CPU wrote");
  }
  run.close();
}

// try_push under the two lossy policies: the Feed's link, in front of the first stage, evicts
// (KeepLatest) or refuses (DropIncoming), as GStreamer's input queue does; what it lets through
// runs through every stage.
void policies(const Route& route) {
  const std::vector<TensorList> inputs = images(route);
  std::vector<std::vector<std::uint8_t>> expected;
  {
    Graph g = route.graph(route.input());
    Run run = g.build(inputs[0], on());
    for (int p = 0; p < kPatterns; ++p) {
      require(run.push(inputs[static_cast<std::size_t>(p)]), "push");
      expected.push_back(bytes_of(pull_ok(run, "policies reference").front()));
    }
    run.close();
  }
  constexpr int kPushes = 24;
  for (const OverflowPolicy policy : {OverflowPolicy::KeepLatest, OverflowPolicy::DropIncoming}) {
    const bool latest = policy == OverflowPolicy::KeepLatest;
    Graph g = route.graph(route.input(), /*max_buffers=*/2);
    RunOptions o = on(2);
    o.overflow_policy = policy;
    Run run = g.build(inputs[0], o);
    int accepted = 0;
    for (int i = 0; i < kPushes; ++i) {
      accepted += run.try_push(inputs[static_cast<std::size_t>(i % kPatterns)]) ? 1 : 0;
    }
    run.close_input();
    std::vector<std::vector<std::uint8_t>> pulled;
    for (;;) {
      Result<TensorList> out = run.pull(std::chrono::milliseconds(30000));
      if (!out) {
        break;
      }
      require(out->size() == 1U, "policies: one tensor per frame");
      pulled.push_back(bytes_of(out->front()));
    }
    require(!pulled.empty(), "policies: at least one frame came through");
    for (const auto& bytes : pulled) {
      require(std::find(expected.begin(), expected.end(), bytes) != expected.end(),
              "policies: every output is the output of an image");
    }
    if (latest) {
      require(accepted == kPushes, "KeepLatest accepts every try_push");
      require(pulled.back() == expected[static_cast<std::size_t>((kPushes - 1) % kPatterns)],
              "KeepLatest: the last frame pushed is the last one pulled");
    } else {
      require(static_cast<int>(pulled.size()) == accepted,
              "DropIncoming: every accepted frame is pulled");
      require(pulled.front() == expected[0], "DropIncoming: the first frame pushed came through");
    }
    require(run.last_error().empty(), "policies: no error latched: " + run.last_error());
    std::cout << "[policies] " << (latest ? "KeepLatest" : "DropIncoming") << ": " << kPushes
              << " try_push, " << accepted << " accepted, " << pulled.size() << " pulled\n";
    run.stop();
  }
}

void declined(const Route& route, bool ev74) {
  const TensorList seed{route.tensor(0)};
  // The code: an option's for OutputOptions::drop, the board's for a stage on an EV74 that does
  // not run the Kernels firmware.
  const auto refused = [&](const GraphOptions& options, bool drop, const char* text,
                           const char* what) {
    const std::string code =
        drop ? error_codes::kOptionOutOfRange : error_codes::kDispatcherUnavailable;
    try {
      Graph g = route.graph(route.input(), 2, drop, options);
      (void)g.build(seed, on());
    } catch (const NeatError& error) {
      require(error.report().error_code == code,
              std::string(what) + ": the build error " + code + ": " + error.what());
      require_contains(error.what(), text, std::string(what) + ": " + error.what());
      std::cout << "[declined] " << what << ": " << error.what() << "\n";
      return;
    }
    throw std::runtime_error(std::string("declined: ") + what + " built on the Dataflow backend");
  };
  if (!ev74) {
    // The EV74 runs the stock firmware: a kernel stage placed there keeps the route off this
    // backend. The default placement puts the pre stage on the EV74.
    refused(GraphOptions{}, false, "sima-neat-native-cvu-fw", "the default placement");
    // The pre stage on the A65 and the post stage asked onto the EV74.
    refused(placed("A65", "EV74"), false, "is not carried by the Dataflow backend on this board",
            "post_run_target EV74");
    refused(placed("EV74", "EV74"), false, "is not carried by the Dataflow backend on this board",
            "both on the EV74");
  }
  // OutputOptions::drop behind a model.
  refused(on_a65(), true, "OutputOptions::drop behind a model", "OutputOptions::drop");
  if (ev74) {
    refused(GraphOptions{}, true, "OutputOptions::drop behind a model",
            "OutputOptions::drop, the default placement");
  }
}

// ---- the kernel stages on the EV74 ---------------------------------------------------------

void on_ev74(const Route& route, bool ev74, sima_test::Reference& ref) {
  if (!ev74) {
    std::cout << "[ev74] not run: the EV74 does not run the Kernels firmware (row declined "
                 "checks that the route stays off the Dataflow backend)\n";
    return;
  }
  struct Placement {
    const char* name;
    GraphOptions options;
  };
  const Placement placements[] = {{"default (pre EV74, post A65)", GraphOptions{}},
                                  {"pre EV74, post A65", placed("EV74", "A65")},
                                  {"pre A65, post EV74", placed("A65", "EV74")},
                                  {"pre EV74, post EV74", placed("EV74", "EV74")}};
  constexpr int kFrames = 64;
  const std::vector<TensorList> inputs = images(route);
  if (!sima_test::on_dataflow()) {
    // The reference of every placement: GStreamer with both kernel stages on the A65.
    const std::vector<Frame> gst =
        run_frames(route.graph(route.input(), 4, false, on_a65()), inputs, kFrames);
    for (const Placement& p : placements) {
      compare(ref, gst, std::string("ev74 ") + p.name);
    }
    return;
  }
  for (const Placement& p : placements) {
    const std::string row = std::string("ev74 ") + p.name;
    const std::vector<Frame> df =
        run_frames(route.graph(route.input(), 4, false, p.options), inputs, kFrames);
    compare(ref, df, row);
    // A TensorMemory::MLA tensor (the DMS heap) gives the output of the same bytes in EV74 memory,
    // whatever the first stage's target: wrapped where the first stage reads either heap, copied
    // into a slot of the input's pool in front of a kernel on the EV74.
    {
      Graph g = route.graph(route.input(), 4, false, p.options);
      Run run = g.build(inputs[0], on());
      for (int i = 0; i < 2 * kPatterns; ++i) {
        const unsigned image_index = static_cast<unsigned>(i % kPatterns);
        require(run.push({route.tensor(image_index, TensorMemory::MLA)}), row + ": push (MLA)");
        require(bytes_of(pull_ok(run, row).front()) == df[image_index].bytes,
                row + ": a TensorMemory::MLA tensor gives the EV74 tensor's output");
      }
      run.close();
    }
    placement() = p.options;
    input_contract(route);
    reuse(route);
    held(route);
    lifecycle(route);
    in_flight(route);
    recreate(route);
    owned(route);
    alloc(route);
    policies(route);
    placement() = on_a65();
    std::cout << "[ev74] " << p.name
              << ": differential, input_contract, reuse, lifecycle, in_flight, recreate, owned, "
                 "alloc, policies ok\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_MODEL_PACKAGE", "SIMA_TEST_MLA_ONLY_MODEL"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_dataflow_model_route_test needs a model package with quantize + "
                     "tessellate and detessellate + dequantize stages: --model-package "
                     "<X_mpk.tar.gz> or SIMA_TEST_MODEL_PACKAGE");
  }
  std::string only;
  sima_test::get_arg(argc, argv, "--row", only);
  Dequant dequant;
  std::string value;
  if (sima_test::get_arg(argc, argv, "--dequant-scale", value)) {
    dequant.scale = std::atof(value.c_str());
  }
  if (sima_test::get_arg(argc, argv, "--dequant-zero-point", value)) {
    dequant.zero_point = std::atoi(value.c_str());
  }
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_dataflow_model_route_test", [&] {
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const Route route(archive);
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    const bool ev74 = ev74_runs_kernels_firmware();
    std::cout << "[ev74] the EV74 " << (ev74 ? "runs" : "does not run")
              << " the Kernels firmware\n";
    row("differential", [&] { differential(route, ref); });
    row("reference", [&] { reference(route, archive, dequant); });
    row("partial", [&] { partial(route, archive, ref); });
    if (!sima_test::on_dataflow()) {
      // GStreamer: the comparison rows' records (and the reference row, checked on its own).
      row("ev74", [&] { on_ev74(route, ev74, ref); });
      ref.finish();
      return;
    }
    row("input_contract", [&] { input_contract(route); });
    row("reuse", [&] { reuse(route); });
    row("held", [&] { held(route); });
    row("lifecycle", [&] { lifecycle(route); });
    row("in_flight", [&] { in_flight(route); });
    row("recreate", [&] { recreate(route); });
    row("owned", [&] { owned(route); });
    row("alloc", [&] { alloc(route); });
    row("policies", [&] { policies(route); });
    row("declined", [&] { declined(route, ev74); });
    row("ev74", [&] { on_ev74(route, ev74, ref); });
    if (only.empty()) {
      ref.finish();
    }
  });
}
