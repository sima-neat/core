// The MLA-only model route on the Dataflow backend (H1), on the public API only:
//   Input -> Model::inference() -> Output      (Model: preprocess off, inference_terminal.mla_only)
// built with RunOptions::backend = RunBackend::Dataflow (Feed -> MlaOp -> Fetch) and, for the
// comparison rows, with RunBackend::GStreamer (appsrc ! neatprocessmla ! appsink) in the same
// process. No runtime type is named.
//
// It needs the MLA and a compiled model: --mla-model <X_mpk.tar.gz>, or SIMA_TEST_MLA_ONLY_MODEL.
// Without one the test is skipped (it downloads nothing). The model has one INT8 input and one
// output; ResNet-50 INT8 is the one it was written with.
//
// Rows:
//   differential     the same inputs (4 patterns in rotation) through both backends: every pulled
//                    tensor's dtype, layout, shape, strides, byte offset, route, quantization and
//                    format tag, the Sample's shape, and every byte. Excluded, each printed with
//                    both values: the stage's element name (GStreamer numbers its elements per
//                    process: MLA_0_<n>), the storage's size (GStreamer: the allocation; Dataflow:
//                    the MLA's output buffer), frame_id / input_seq / timestamps / caps_string
//                    (documented for every Dataflow Run: unit_run_backend_differential_test).
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

  Graph graph(const InputOptions& in, int max_buffers = 4, bool drop = false) const {
    OutputOptions out;
    out.max_buffers = max_buffers;
    out.drop = drop;
    out.sync = false;
    Graph g;
    g.add(nodes::Input(in));
    g.add(model.inference());
    g.add(nodes::Output(out));
    return g;
  }

  Tensor tensor(unsigned index, TensorMemory memory = TensorMemory::MLA) const {
    return Tensor::from_vector(pattern(bytes, index), shape, memory);
  }
};

RunOptions on(RunBackend backend, int depth = 4) {
  RunOptions o;
  o.backend = backend;
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

std::vector<Frame> run_frames(const Route& route, RunBackend backend, int frames) {
  Graph g = route.graph(route.input());
  std::vector<TensorList> inputs;
  for (int p = 0; p < kPatterns; ++p) {
    inputs.push_back({route.tensor(static_cast<unsigned>(p))});
  }
  Run run = g.build(inputs[0], on(backend));
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

void differential(const Route& route) {
  constexpr int kFrames = 64;
  const std::vector<Frame> gst = run_frames(route, RunBackend::GStreamer, kFrames);
  const std::vector<Frame> df = run_frames(route, RunBackend::Dataflow, kFrames);
  require(gst.size() == df.size(), "both backends produced every frame");
  for (int i = 0; i < kFrames; ++i) {
    const auto at = [&](const char* what) {
      return std::string("differential frame ") + std::to_string(i) + ": " + what;
    };
    const Frame& fa = gst[static_cast<std::size_t>(i)];
    const Frame& fb = df[static_cast<std::size_t>(i)];
    const Tensor& a = fa.tensor;
    const Tensor& b = fb.tensor;
    require(a.dtype == b.dtype, at("dtype"));
    require(a.layout == b.layout, at("layout"));
    require(a.shape == b.shape, at("shape ") + dims(a.shape) + " / " + dims(b.shape));
    require(a.strides_bytes == b.strides_bytes, at("strides"));
    require(a.byte_offset == b.byte_offset, at("byte_offset"));
    require(a.axis_semantics == b.axis_semantics, at("axis_semantics"));
    require(a.device.type == b.device.type && a.device.id == b.device.id, at("device"));
    require(a.read_only == b.read_only, at("read_only"));
    require(a.planes.size() == b.planes.size(), at("planes"));
    require(a.route.logical_index == b.route.logical_index &&
                a.route.backend_output_index == b.route.backend_output_index &&
                a.route.route_slot == b.route.route_slot &&
                a.route.physical_index == b.route.physical_index &&
                a.route.memory_index == b.route.memory_index &&
                a.route.physical_byte_offset == b.route.physical_byte_offset &&
                a.route.name == b.route.name && a.route.backend_name == b.route.backend_name &&
                a.route.segment_name == b.route.segment_name,
            at("route: name '") + a.route.name + "' / '" + b.route.name + "', segment '" +
                a.route.segment_name + "' / '" + b.route.segment_name + "'");
    // GStreamer names the element MLA_0_<n>, n counting the elements this process built.
    require(a.route.stage_key.rfind(b.route.stage_key, 0) == 0,
            at("route.stage_key '") + a.route.stage_key + "' / '" + b.route.stage_key + "'");
    require(a.semantic.tess.has_value() == b.semantic.tess.has_value() &&
                (!a.semantic.tess || a.semantic.tess->format == b.semantic.tess->format),
            at("semantic.tess"));
    require(a.semantic.quant.has_value() == b.semantic.quant.has_value(), at("semantic.quant"));
    require(!a.semantic.image && !b.semantic.image, at("no image semantic"));
    require(fa.storage_kind == fb.storage_kind && fa.storage_device == fb.storage_device &&
                fa.storage_target_flags == fb.storage_target_flags &&
                fa.storage_segments == fb.storage_segments,
            at("storage kind, placement, segments"));
    require(!fa.bytes.empty() && fa.bytes == fb.bytes, at("bytes"));
    if (fa.from_sample) {
      const Sample& sa = fa.sample;
      const Sample& sb = fb.sample;
      require(sa.kind == sb.kind && sa.owned == sb.owned, at("Sample kind / owned"));
      require(sa.media_type == sb.media_type && sa.payload_type == sb.payload_type &&
                  sa.payload_tag == sb.payload_tag && sa.format == sb.format,
              at("Sample media: '") + sa.payload_tag + "' / '" + sb.payload_tag + "'");
      require(sa.output_index == sb.output_index &&
                  sa.logical_output_index == sb.logical_output_index &&
                  sa.route_slot == sb.route_slot && sa.memory_index == sb.memory_index,
              at("Sample routing"));
      require(sa.stream_id == sb.stream_id, at("Sample stream_id"));
      require(sa.stream_label.rfind(sb.stream_label, 0) == 0 &&
                  sa.segment_name.rfind(sb.segment_name, 0) == 0,
              at("Sample stream_label '") + sa.stream_label + "' / '" + sb.stream_label + "'");
    }
  }
  // The four inputs give four different outputs, and each input the same one every time.
  for (int p = 1; p < kPatterns; ++p) {
    require(df[0].bytes != df[static_cast<std::size_t>(p)].bytes,
            "the outputs of different inputs differ");
    require(df[static_cast<std::size_t>(p)].bytes ==
                df[static_cast<std::size_t>(p + kPatterns)].bytes,
            "an input gives the same output again");
  }
  std::cout << "[differential] excluded: stage_key '" << gst[0].tensor.route.stage_key << "' / '"
            << df[0].tensor.route.stage_key << "'; storage bytes " << gst[0].storage_bytes << " / "
            << df[0].storage_bytes << "; first frame_id " << gst[0].frame_id << " / "
            << df[0].frame_id << "; first seq " << gst[0].seq << " / " << df[0].seq
            << "; caps_string '" << gst[1].sample.caps_string << "' / '" << df[1].sample.caps_string
            << "'\n";
}

// ---- input contract ----------------------------------------------------------------------------

// What a push of `tensor` does: "accepted" with an output pulled, or the error's text.
std::string push_outcome(const Route& route, const InputOptions& in, const Tensor& tensor,
                         const TensorList* seed) {
  Graph g = route.graph(in);
  Run run = seed ? g.build(*seed, on(RunBackend::Dataflow)) : g.build(on(RunBackend::Dataflow));
  std::string outcome;
  try {
    outcome = run.push({tensor}) ? "accepted" : "push returned false";
    if (outcome == "accepted") {
      (void)pull_ok(run, "input_contract");
    }
  } catch (const std::exception& error) {
    outcome = error.what();
  }
  run.close();
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
  require_contains(wrong, "tensor format does not match dtype",
                   "a UInt8 tensor is refused at the push, as on GStreamer: " + wrong);
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
          (void)g.build(on(RunBackend::Dataflow));
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
  Run run = g.build(a, on(RunBackend::Dataflow, 8));
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
  require(run.last_error().empty(), "no error latched: " + run.last_error());
}

// A Run that goes away with frames queued in front of the MLA, on it, and waiting to be pulled.
void in_flight(const Route& route) {
  const TensorList a{route.tensor(0)};
  std::vector<std::uint8_t> expected;
  {
    Graph g = route.graph(route.input(), /*max_buffers=*/4);
    Run run = g.build(a, on(RunBackend::Dataflow, 4));
    require(run.push(a), "push");
    expected = bytes_of(pull_ok(run, "in_flight first").front());
    for (int i = 0; i < 6; ++i) {
      require(run.try_push(a) || i > 0, "the first try_push finds room");
    }
    // No pull, no close_input, no stop: the destructor ends the Run.
  }
  {
    Graph g = route.graph(route.input(), 4);
    Run run = g.build(a, on(RunBackend::Dataflow, 4));
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
    Run run = g.build(a, on(RunBackend::Dataflow));
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
    Run run = g.build(a, on(RunBackend::Dataflow));
    require(run.push(a), "push");
    const Tensor t = pull_ok(run, "owned reference").front();
    require(t.storage && t.storage->kind == StorageKind::GstSample,
            "the default output is the MLA's buffer");
    zero_copy = bytes_of(t);
    run.close();
  }
  Graph g = route.graph(route.input());
  RunOptions o = on(RunBackend::Dataflow);
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
    Run run = g.build(reference_input, on(RunBackend::Dataflow));
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
  Run run = g.build(on(RunBackend::Dataflow));
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

void declined(const Route& route, const std::string& archive) {
  const TensorList seed{route.tensor(0)};
  // OutputOptions::drop behind a model.
  require_neat_error(
      [&] {
        Graph g = route.graph(route.input(), 2, /*drop=*/true);
        (void)g.build(seed, on(RunBackend::Dataflow));
      },
      error_codes::kPipelineShape, "OutputOptions::drop behind a model");
  // The model's preprocessing in front of its inference: stages this backend does not run yet.
  // Explicit Dataflow fails at the build, naming what it does not take; nothing runs.
  Model full(archive);
  Graph g;
  g.add(nodes::Input(full.input_appsrc_options(false)));
  g.add(full.preprocess());
  g.add(full.inference());
  g.add(nodes::Output());
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
    (void)g.build(on(RunBackend::Dataflow));
    if (kernels_firmware) {
      std::cout << "[declined] the preprocess stage built: the EV74 runs the Kernels firmware\n";
      return;
    }
    throw std::runtime_error("declined: a preprocess stage built on the Dataflow backend");
  } catch (const NeatError& error) {
    require(error.report().error_code == error_codes::kPipelineShape,
            std::string("explicit Dataflow on stages beside the MLA is a build error: ") +
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
  return sima_test::run_test("unit_dataflow_mla_route_test", [&] {
    const Route route(archive);
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only.empty() && only != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    row("differential", [&] { differential(route); });
    row("input_contract", [&] { input_contract(route); });
    row("lifecycle", [&] { lifecycle(route); });
    row("in_flight", [&] { in_flight(route); });
    row("recreate", [&] { recreate(route); });
    row("owned", [&] { owned(route); });
    row("alloc", [&] { alloc(route); });
    row("declined", [&] { declined(route, archive); });
  });
}
