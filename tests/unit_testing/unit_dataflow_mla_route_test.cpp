// The MLA-only model route on the Dataflow backend (H1), on the public API only:
//   Input -> Model::inference() -> Output      (Model: preprocess off, inference_terminal.mla_only)
// on the Dataflow backend (Feed -> MlaOp -> Fetch) and, for the comparison row, on GStreamer
// (appsrc ! neatprocessmla ! appsink). A process runs on one backend (set_backend): the test runs
// twice, --backend=gstreamer --reference=<file> recording what the comparison row pulls, then
// --backend=dataflow --reference=<file> running every row and comparing. No runtime type is
// named.
//
// It needs the MLA and a compiled model: --mla-model <X_mpk.tar.gz>, or SIMA_TEST_MLA_ONLY_MODEL.
// Without one the test is skipped (it downloads nothing). The model has one INT8 input and one
// output; ResNet-50 INT8 is the one it was written with.
//
// Rows:
//   differential     the same inputs (4 patterns in rotation) on each backend: every pulled
//                    tensor's dtype, layout, shape, strides, byte offset, route, quantization and
//                    format tag, the Sample's shape, and every byte (its digest). Excluded, each
//                    printed with both values: the stage's element name (GStreamer numbers its
//                    elements per process: MLA_0_<n>), the storage's size (GStreamer: the
//                    allocation; Dataflow: the MLA's output buffer), frame_id / input_seq /
//                    timestamps / caps_string (documented for every Dataflow Run:
//                    unit_run_backend_differential_test).
//   input_contract   which tensors the route takes: TensorMemory::MLA and EV74 tensors, with and
//                    without the batch axis; a CPU tensor is refused with GStreamer's guard
//                    message; a tensor of another dtype with GStreamer's push error.
//   lifecycle        a pull before any push times out; try_push; close_input drains what was
//                    pushed and then closes; a push after it is refused; stop twice.
//   in_flight        a Run destroyed with frames queued and on the MLA; then a new Run works.
//   recreate         Runs one after the other in one process give the same outputs.
//   owned            OutputMemory::Owned: a CPU copy with the same bytes.
//   alloc            Run::alloc lends a slot of the input's device pool; what the CPU writes
//                    there is what the MLA reads.
//   padding          for a model whose MLA input buffer is longer than its INT8 tensor (the model
//                    compiler rounds the buffer up to 16 bytes; Inception-v3's 299x299x3): the
//                    bytes behind the tensor do not reach the outputs. The buffer is pushed
//                    whole under a plain Input, the same values in front of paddings of zeros, of
//                    ones and of noise: every output byte is the same. The model's tensor alone
//                    is refused. (A model whose buffer is its tensor has no padding: the row
//                    says so.)
//   declined         what stays off this backend: OutputOptions::drop behind a model, and the
//                    model's preprocessing in front of its inference (explicit Dataflow fails at
//                    the build; nothing runs).

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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
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

std::vector<std::int8_t> pattern(std::size_t bytes, unsigned index) {
  std::vector<std::int8_t> out(bytes);
  std::uint32_t x = 0x9e3779b9U * (index + 1U) + 12345U;
  for (auto& value : out) {
    x = x * 1664525U + 1013904223U;
    value = static_cast<std::int8_t>(static_cast<unsigned char>(x >> 24));
  }
  return out;
}

std::vector<std::uint8_t> bytes_of(const Tensor& tensor) {
  return tensor.copy_dense_bytes_tight();
}

struct Route {
  Model model;
  std::vector<std::int64_t> shape; // the model's input, with its batch axis
  std::size_t bytes = 0;

  explicit Route(const std::string& archive) : model(archive, options()) {
    const auto specs = model.input_specs();
    require(specs.size() == 1U, "the test model has one input");
    shape = specs.front().shape;
    bytes = 1;
    for (const auto dim : shape) {
      require(dim > 0, "the test model's input shape is fixed");
      bytes *= static_cast<std::size_t>(dim);
    }
  }

  static Model::Options options() {
    Model::Options o;
    o.preprocess.kind = InputKind::Tensor;
    o.preprocess.enable = AutoFlag::Off;
    o.inference_terminal.mla_only = true;
    return o;
  }

  // The route's Input: the model's own options, with the MLA's INT8 input declared.
  InputOptions input() const {
    InputOptions in = model.input_appsrc_options(true);
    in.payload_type = PayloadType::Tensor;
    in.format = FormatTag::INT8;
    in.dtype = TensorDType::Int8;
    return in;
  }

  // `named`: named endpoints, a graph Run (built without a seed too; a plain linear Run is not).
  Graph graph(const InputOptions& in, int max_buffers = 4, bool drop = false,
              bool named = false) const {
    OutputOptions out;
    out.max_buffers = max_buffers;
    out.drop = drop;
    out.sync = false;
    Graph g;
    g.add(named ? nodes::Input("in", in) : nodes::Input(in));
    g.add(model.inference());
    g.add(named ? nodes::Output("out", out) : nodes::Output(out));
    return g;
  }

  Tensor tensor(unsigned index, TensorMemory memory = TensorMemory::MLA) const {
    return Tensor::from_vector(pattern(bytes, index), shape, memory);
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
  frame.storage_bytes = tensor.storage->size_bytes;
  frame.frame_id = tensor.stamp().frame_id;
  frame.seq = tensor.stamp().seq;
  tensor.storage.reset();
  tensor.envelope.reset();
  frame.tensor = std::move(tensor);
  return frame;
}

std::vector<Frame> run_frames(const Route& route, int frames) {
  Graph g = route.graph(route.input());
  std::vector<TensorList> inputs;
  for (int p = 0; p < kPatterns; ++p) {
    inputs.push_back({route.tensor(static_cast<unsigned>(p))});
  }
  Run run = g.build(inputs[0], on());
  std::vector<Frame> seen;
  for (int i = 0; i < frames; ++i) {
    require(run.push(inputs[static_cast<std::size_t>(i % kPatterns)]), "push");
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

// What both backends give the same, recorded by the GStreamer run and compared by the Dataflow run;
// the values each backend gives its own way are printed side by side.
void differential(const Route& route, sima_test::Reference& ref) {
  constexpr int kFrames = 64;
  const std::vector<Frame> frames = run_frames(route, kFrames);
  ref.equal("differential: frames", frames.size());
  for (int i = 0; i < kFrames; ++i) {
    const auto at = [&](const char* what) {
      return std::string("differential frame ") + std::to_string(i) + ": " + what;
    };
    const Frame& f = frames[static_cast<std::size_t>(i)];
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
    // GStreamer names the element MLA_0_<n>, n counting the elements this process built.
    ref.prefix(at("route.stage_key"), t.route.stage_key);
    ref.equal(at("semantic.tess"),
              t.semantic.tess ? sima_test::text(t.semantic.tess->format) : std::string("none"));
    ref.equal(at("semantic.quant"), t.semantic.quant.has_value());
    require(!t.semantic.image, at("no image semantic"));
    ref.equal(at("storage kind, placement, segments"),
              storage_kind_text(f.storage_kind) + "/" + sima_test::text(f.storage_device) + "/" +
                  sima_test::text(f.storage_target_flags) + "/" +
                  sima_test::text(f.storage_segments));
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
  // The four inputs give four different outputs, and each input the same one every time.
  for (int p = 1; p < kPatterns; ++p) {
    require(frames[0].bytes != frames[static_cast<std::size_t>(p)].bytes,
            "the outputs of different inputs differ");
    require(frames[static_cast<std::size_t>(p)].bytes ==
                frames[static_cast<std::size_t>(p + kPatterns)].bytes,
            "an input gives the same output again");
  }
  // Excluded: each backend's own.
  ref.note("differential excluded: stage_key", frames[0].tensor.route.stage_key);
  ref.note("differential excluded: storage bytes", frames[0].storage_bytes);
  ref.note("differential excluded: first frame_id", frames[0].frame_id);
  ref.note("differential excluded: first seq", frames[0].seq);
  ref.note("differential excluded: caps_string", frames[1].sample.caps_string);
}

// ---- input contract ----------------------------------------------------------------------------

// What a push of `tensor` does: "accepted" with an output pulled, or the error's text.
std::string push_outcome(const Route& route, const InputOptions& in, const Tensor& tensor,
                         const TensorList* seed) {
  Graph g = route.graph(in, 4, false, /*named=*/seed == nullptr);
  std::string outcome;
  try {
    // A seed the route refuses fails the build (its preflight pushes it), as on GStreamer.
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
  require(push_outcome(route, in, route.tensor(1, TensorMemory::MLA), &seed) == "accepted",
          "a TensorMemory::MLA tensor is taken");
  require(push_outcome(route, in, route.tensor(1, TensorMemory::EV74), &seed) == "accepted",
          "a TensorMemory::EV74 tensor is taken");
  require(push_outcome(route, in, route.tensor(1, TensorMemory::MLA), nullptr) == "accepted",
          "an unseeded Run binds by its first push");
  if (route.shape.size() == 4U && route.shape.front() == 1) {
    const std::vector<std::int64_t> unbatched(route.shape.begin() + 1, route.shape.end());
    const Tensor t = Tensor::from_vector(pattern(route.bytes, 1), unbatched, TensorMemory::MLA);
    const TensorList unbatched_seed{t};
    require(push_outcome(route, in, t, &unbatched_seed) == "accepted",
            "the input without its batch axis is taken");
  }
  // GStreamer's guard of a device-visible route, with its message.
  const std::string cpu = push_outcome(route, in, route.tensor(1, TensorMemory::CPU), &seed);
  require_contains(cpu, "CPU-backed Tensor pushed into a device-visible EV74/DMS route",
                   "a CPU tensor is refused as on GStreamer: " + cpu);
  // GStreamer's push check of the declared format.
  const std::vector<std::uint8_t> zeros(route.bytes, 7);
  const Tensor u8 = Tensor::from_vector(zeros, route.shape, TensorMemory::MLA);
  const TensorList u8_seed{u8};
  const std::string wrong = push_outcome(route, in, u8, &u8_seed);
  require_contains(
      wrong, "tensor format does not match dtype",
      "a UInt8 tensor is refused (as the seed, by the build's preflight), as on GStreamer: " +
          wrong);
  // The model's own options declare its public FP32 input; the MLA's INT8 tensor does not fit them.
  const std::string fp32 =
      push_outcome(route, route.model.input_appsrc_options(true), route.tensor(1), &seed);
  require_contains(fp32, "tensor format does not match dtype",
                   "an INT8 tensor under the FP32 declaration: " + fp32);
  // A declared shape that is not the MLA's input is the build's error.
  {
    InputOptions other = in;
    other.shape = route.shape;
    other.shape.back() += 1;
    require_neat_error(
        [&] {
          Graph g = route.graph(other);
          (void)g.build(on());
        },
        error_codes::kInputShape, "the model's MLA input is");
  }
  // A tensor of the right dtype and another size is not the MLA's input.
  if (route.shape.size() == 4U) {
    std::vector<std::int64_t> half = route.shape;
    half[1] = std::max<std::int64_t>(1, half[1] / 2);
    std::size_t n = 1;
    for (const auto dim : half) {
      n *= static_cast<std::size_t>(dim);
    }
    const Tensor small = Tensor::from_vector(pattern(n, 1), half, TensorMemory::MLA);
    const std::string outcome = push_outcome(route, in, small, &seed);
    require(outcome != "accepted", "a tensor of another size is refused: " + outcome);
    std::cout << "[input_contract] another size: " << outcome << "\n";
  }
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

  // close_input: what was pushed is still inferred and pulled, in order, then the Run is closed.
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

// A Run that goes away with frames queued in front of the MLA, on it, and waiting to be pulled.
void in_flight(const Route& route) {
  const TensorList a{route.tensor(0)};
  std::vector<std::uint8_t> expected;
  {
    Graph g = route.graph(route.input(), /*max_buffers=*/4);
    Run run = g.build(a, on(4));
    require(run.push(a), "push");
    expected = bytes_of(pull_ok(run, "in_flight first").front());
    for (int i = 0; i < 6; ++i) {
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
            "the default output is the MLA's buffer");
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

// Run::alloc: the application writes the input into a slot of the route's device pool.
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
  OutputOptions out;
  out.max_buffers = 4;
  Graph g;
  g.add(nodes::Input("x", in));
  g.add(route.model.inference());
  g.add(nodes::Output("y", out));
  Run run = g.build(on());
  const std::vector<std::int8_t> values = pattern(route.bytes, 1);
  for (int i = 0; i < 3; ++i) {
    Tensor t = run.alloc("x");
    {
      Mapping map = t.map_write();
      require(map.data != nullptr && map.size_bytes >= values.size(), "the slot maps for write");
      std::memcpy(map.data, values.data(), values.size());
    }
    require(run.push("x", {t}), "push the allocated tensor");
    Result<TensorList> pulled = run.pull("y", std::chrono::milliseconds(30000));
    require(pulled && pulled->size() == 1U, "pull");
    require(bytes_of(pulled->front()) == expected, "the MLA read what the CPU wrote");
  }
  run.close();
}

// ---- padding -----------------------------------------------------------------------------------

// The model compiler makes the MLA's input buffer a multiple of 16 bytes, so it may be up to 15
// bytes longer than the model's tensor. What the bytes behind the tensor hold must not reach the
// outputs (in front of a kernel stage they are the zeros its port's pool wrote once). Here the
// MLA's buffer is the route's input: a plain Input of INT8 tensors takes a tensor of the
// buffer's bytes, and the same values are pushed in front of paddings of zeros, of ones and of
// noise. Every output byte is the same. The model's tensor alone, under the model's own Input
// options, is not the route's input (GStreamer's inference-only ingress refuses it too, and
// builds no inference-only route for such a model's buffer either: this row has no comparison
// row on GStreamer).
void padding(const Route& route) {
  const std::size_t buffer = (route.bytes + 15U) / 16U * 16U;
  if (buffer == route.bytes) {
    std::cout << "[padding] the MLA's input buffer is the model's tensor (" << route.bytes
              << " bytes, a multiple of 16): no padding to vary\n";
    return;
  }
  const std::size_t tail = buffer - route.bytes;
  // The model's tensor alone: refused at the build (its preflight pushes the seed), with the
  // buffer's bytes in the text.
  {
    const TensorList seed{route.tensor(0)};
    const std::string outcome = push_outcome(route, route.input(), route.tensor(1), &seed);
    require(outcome != "accepted", "padding: the model's tensor alone is refused");
    require_contains(outcome, "in a buffer of " + std::to_string(buffer) + " bytes",
                     "padding: the refusal says what the buffer is: " + outcome);
    std::cout << "[padding] the model's tensor alone: " << outcome << "\n";
  }
  InputOptions in;
  in.payload_type = PayloadType::Tensor;
  in.format = FormatTag::INT8;
  in.dtype = TensorDType::Int8;
  const auto whole = [&](unsigned p, int fill) {
    std::vector<std::int8_t> bytes(buffer);
    const std::vector<std::int8_t> values = pattern(route.bytes, p);
    std::memcpy(bytes.data(), values.data(), route.bytes);
    const std::vector<std::int8_t> noise = pattern(tail, 100U + p);
    for (std::size_t i = 0; i < tail; ++i) {
      bytes[route.bytes + i] = fill < 0 ? noise[i] : static_cast<std::int8_t>(fill);
    }
    return Tensor::from_vector(bytes, {static_cast<std::int64_t>(buffer)}, TensorMemory::MLA);
  };
  Graph g = route.graph(in);
  Run run = g.build(TensorList{whole(0, 0x00)}, on());
  std::size_t compared = 0;
  std::vector<std::vector<std::uint8_t>> outputs;
  for (unsigned p = 0; p < kPatterns; ++p) {
    std::vector<std::uint8_t> zeros;
    for (const int fill : {0x00, 0xFF, -1}) {
      require(run.push({whole(p, fill)}), "padding: push");
      std::vector<std::uint8_t> out = bytes_of(pull_ok(run, "padding").front());
      if (fill == 0x00) {
        zeros = std::move(out);
      } else {
        require(out == zeros, "padding: pattern " + std::to_string(p) + ", a padding of " +
                                  (fill < 0 ? "noise" : "ones") +
                                  ": the outputs differ from those in front of zeros");
        ++compared;
      }
    }
    outputs.push_back(std::move(zeros));
  }
  run.close();
  for (unsigned p = 1; p < kPatterns; ++p) {
    require(outputs[p] != outputs[0], "padding: the outputs of different inputs differ");
  }
  std::cout << "[padding] " << tail << " bytes of padding behind the tensor's " << route.bytes
            << ": " << compared << " outputs in front of ones and of noise, each equal to the "
            << "output of the same values in front of zeros (" << kPatterns << " patterns, "
            << outputs[0].size() << " bytes each)\n";
}

void declined(const Route& route, const std::string& archive) {
  const TensorList seed{route.tensor(0)};
  // OutputOptions::drop behind a model.
  require_neat_error(
      [&] {
        Graph g = route.graph(route.input(), 2, /*drop=*/true);
        (void)g.build(seed, on());
      },
      error_codes::kOptionOutOfRange, "OutputOptions::drop behind a model");
  // The model's preprocessing in front of its inference: stages this backend does not run yet.
  // Explicit Dataflow fails at the build, naming what it does not take; nothing runs.
  Model full(archive);
  Graph g;
  g.add(nodes::Input("in", full.input_appsrc_options(false)));
  g.add(full.preprocess());
  g.add(full.inference());
  g.add(nodes::Output("out"));
  // Since the model route takes kernel stages, this one (a quantize + tessellate stage placed on
  // the EV74 by default) is declined only where the EV74 runs the stock firmware; on the Kernels
  // firmware it builds (nothing is pushed, so nothing runs).
  bool kernels_firmware = false;
  for (int i = 0; i < 16; ++i) {
    const std::string dir = "/sys/class/remoteproc/remoteproc" + std::to_string(i) + "/";
    const auto read = [&dir](const char* name) {
      std::ifstream file(dir + name);
      std::string text;
      std::getline(file, text);
      return text;
    };
    if (read("name") == "ev74") {
      kernels_firmware =
          read("state") == "running" && read("firmware") == "sima-neat-native-cvu-fw";
      break;
    }
  }
  try {
    (void)g.build(on());
    if (kernels_firmware) {
      std::cout << "[declined] the preprocess stage built: the EV74 runs the Kernels firmware\n";
      return;
    }
    throw std::runtime_error("declined: a preprocess stage built on the Dataflow backend");
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kDispatcherUnavailable,
            std::string("a stage on an EV74 without the Kernels firmware is the board's build "
                        "error: ") +
                error.what());
    std::cout << "[declined] " << error.what() << "\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string archive;
  sima_test::get_arg(argc, argv, "--mla-model", archive);
  if (const char* staged = std::getenv("SIMA_TEST_MLA_ONLY_MODEL"); archive.empty() && staged) {
    archive = staged;
  }
  if (archive.empty() || !file_exists(archive)) {
    return skip_test("unit_dataflow_mla_route_test needs an MLA-only INT8 model: "
                     "--mla-model <X_mpk.tar.gz> or SIMA_TEST_MLA_ONLY_MODEL");
  }
  std::string only;
  sima_test::get_arg(argc, argv, "--row", only);
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_dataflow_mla_route_test", [&] {
    sima_test::Reference ref = sima_test::Reference::from_args(argc, argv);
    const Route route(archive);
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    row("differential", [&] { differential(route, ref); });
    if (!sima_test::on_dataflow()) {
      // GStreamer: the comparison row's record only.
      ref.finish();
      return;
    }
    row("input_contract", [&] { input_contract(route); });
    row("lifecycle", [&] { lifecycle(route); });
    row("in_flight", [&] { in_flight(route); });
    row("recreate", [&] { recreate(route); });
    row("owned", [&] { owned(route); });
    row("alloc", [&] { alloc(route); });
    row("padding", [&] { padding(route); });
    row("declined", [&] { declined(route, archive); });
    if (only.empty() || only == "differential") {
      ref.finish();
    }
  });
}
