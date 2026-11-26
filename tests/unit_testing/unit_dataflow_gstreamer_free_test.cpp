// A process on the Dataflow backend never starts GStreamer and makes no GStreamer object
// (DECISIONS.md 31). After each scenario below the process's GStreamer footprint
// (support/gstreamer_footprint.h: gst_is_initialized(), and every GLib type whose name starts with
// "Gst") is empty; the first scenario that leaves one fails, named with what it left.
//
// Scenarios, each a customer's calls on the public API:
//   prewarm         prewarm_runtime(), prewarm_runtime_async()
//   cpu tensors     Tensor::from_vector in CPU memory, clone, cpu(), read and write maps
//   host graph      Input -> Output with CPU tensors: a seeded build, a push of a TensorList and of
//                   a Sample, try_push, the pull of a Sample and of a TensorList, Graph::run
//   refusals        an InputOptions::caps_override (refused unparsed) and an RTSP source, each in
//                   the one not-carried form
//   device tensors  where the build has device memory: Tensor::from_vector in EV74 and in MLA
//                   memory, cvu(), mla(true), to(), their storage the runtime's
//                   (StorageKind::DeviceHandle) with the bytes; cpu(), clone(), write and read maps
//   device graph    Input in EV74 and in DMS0 memory -> Output: a device tensor pushed (its
//                   DMA-BUF backs the slot), a CPU tensor pushed (copied, with the compatibility
//                   copy allowed); the pulled tensors are DeviceHandle storage with the bytes
//   models          each --model=<package> [--opts=<file|->] (the qualification harness's option
//                   files): the Model, Model::build(seed) + Runner push and pull, Model::run,
//                   Model::benchmark, Model::execution_plan, and a Graph of the Model's
//                   preprocess, inference and postprocess; inputs in EV74 memory, as the harness
//                   pushes them (a 1280x720 frame of the view's format, or tensors of the model's
//                   input specs). A form of a view whose route shape Dataflow does not carry is
//                   refused in the one not-carried form; the refusal is a scenario like the others
// Registered with --backend=dataflow only. The model scenarios run where packages are given (a
// board); without device memory (a host build) the device scenarios are skipped and say so.

#include "model/Model.h"
#include "neat/runtime.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/TensorCore.h"
#include "qualification/qualification_common.h"
#include "support/backend_selection.h"
#include "support/gstreamer_footprint.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
namespace q = simaai::neat::qualification;

// The scenario `name` ran; the process must still hold nothing of GStreamer.
void untouched(const std::string& name) {
  const sima_test::GStreamerFootprint gst = sima_test::gstreamer_footprint();
  require(gst.empty(), "after " + name + ": GStreamer is " + gst.text());
  std::cout << "[untouched] " << name << "\n";
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

std::vector<std::uint8_t> pattern(std::size_t n, std::uint8_t salt) {
  std::vector<std::uint8_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<std::uint8_t>(salt + i * 13U);
  }
  return v;
}

// An RGB frame (h, w, 3) of `memory`, its bytes from `salt`.
Tensor rgb(int w, int h, std::uint8_t salt, TensorMemory memory) {
  Tensor t =
      Tensor::from_vector(pattern(static_cast<std::size_t>(w) * h * 3U, salt), {h, w, 3}, memory);
  t.layout = TensorLayout::HWC;
  t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::RGB, ""};
  return t;
}

Tensor pull_one(Run& run, const std::string& what) {
  Result<TensorList> out = run.pull(std::chrono::milliseconds(5000));
  require(out.ok() && !out.value().empty(), what + ": a pulled tensor");
  return out.value().front();
}

InputOptions image_input(InputMemoryPolicy memory) {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::RGB;
  in.memory_policy = memory;
  return in;
}

Graph input_output(const InputOptions& in = {}) {
  Graph g;
  g.add(nodes::Input(in));
  g.add(nodes::Output());
  return g;
}

void prewarm() {
  prewarm_runtime();
  prewarm_runtime_async().get();
  untouched("prewarm");
}

void cpu_tensors() {
  Tensor t = rgb(16, 8, 0x11, TensorMemory::CPU);
  const Tensor copy = t.clone();
  require(bytes_of(copy) == bytes_of(t) && bytes_of(t.cpu()) == bytes_of(t), "CPU tensor bytes");
  {
    Mapping map = t.map(MapMode::Write);
    require(map.data != nullptr, "a CPU write map");
    static_cast<std::uint8_t*>(map.data)[0] = 0x5A;
  }
  require(bytes_of(t).front() == 0x5A, "the write is read back");
  untouched("cpu tensors");
}

void host_graph() {
  const Tensor seed = rgb(32, 24, 0x21, TensorMemory::CPU);
  {
    Run run = input_output(image_input(InputMemoryPolicy::Auto)).build(TensorList{seed});
    for (int i = 0; i < 4; ++i) {
      const Tensor t = rgb(32, 24, static_cast<std::uint8_t>(0x30 + i), TensorMemory::CPU);
      Sample sample;
      sample.kind = SampleKind::Tensor;
      sample.tensor = t;
      require(i % 2 == 0 ? run.push(TensorList{t}) : run.push(sample), "a host push");
      require(bytes_of(pull_one(run, "host graph")) == bytes_of(t), "host graph bytes");
    }
    require(run.try_push(TensorList{seed}), "try_push");
    Sample s;
    PullError e;
    require(run.pull(5000, s, &e) == PullStatus::Ok && !s.tensors.empty(), "a Sample pull");
    run.close();
  }
  Graph g = input_output(image_input(InputMemoryPolicy::Auto));
  const TensorList out = g.run(TensorList{seed});
  require(!out.empty() && bytes_of(out.front()) == bytes_of(seed), "Graph::run");
  untouched("host graph");
}

// A build Dataflow refuses in the one not-carried form; GStreamer is not asked first.
void refused(Graph&& g, const std::string& subject) {
  bool threw = false;
  try {
    (void)g.build(TensorList{rgb(32, 24, 0x41, TensorMemory::CPU)});
  } catch (const NeatError& e) {
    threw = true;
    require_contains(e.what(), "is not carried by the Dataflow backend", subject);
    require_contains(e.what(), subject, "the refusal names it");
  }
  require(threw, subject + ": refused");
}

void refusals() {
  InputOptions caps = image_input(InputMemoryPolicy::Auto);
  caps.caps_override = "video/x-raw,format=(string";
  refused(input_output(caps), "option InputOptions::caps_override");
  Graph rtsp;
  rtsp.add(nodes::RTSPInput("rtsp://127.0.0.1:18554/cam"));
  rtsp.add(nodes::Output());
  bool threw = false;
  try {
    (void)rtsp.build(RunOptions{});
  } catch (const NeatError& e) {
    threw = true;
    require_contains(e.what(), "is not carried by the Dataflow backend", "an RTSP source");
  }
  require(threw, "an RTSP source: refused");
  untouched("refusals");
}

// Whether this build of the Dataflow backend has device memory: a device tensor is made, or the
// build says it has none.
bool device_memory() {
  try {
    (void)Tensor::from_vector(pattern(64, 1), {64}, TensorMemory::EV74);
    return true;
  } catch (const NeatError& e) {
    if (std::string(e.what()).find("configured without the Runner's memory package") !=
        std::string::npos) {
      return false;
    }
    throw;
  }
}

// A device tensor of the runtime's memory placed on `device` with `bytes`. A pulled one's storage
// is placed (the tensor itself is labelled CPU, as a GStreamer Run labels a pulled DMA-BUF tensor).
void device_tensor(const Tensor& t, DeviceType device, const std::vector<std::uint8_t>& bytes,
                   const std::string& what, bool pulled = false) {
  require(t.storage && t.storage->kind == StorageKind::DeviceHandle,
          what + ": the runtime's device memory");
  require(t.storage->device.type == device && (pulled || t.device.type == device),
          what + ": placed");
  require(bytes_of(t) == bytes, what + ": its bytes");
}

void device_tensors() {
  const std::vector<std::uint8_t> v = pattern(4096, 0x51);
  const Tensor cpu = Tensor::from_vector(v, {4096}, TensorMemory::CPU);
  device_tensor(Tensor::from_vector(v, {4096}, TensorMemory::EV74), DeviceType::SIMA_CVU, v,
                "from_vector EV74");
  device_tensor(Tensor::from_vector(v, {4096}, TensorMemory::MLA), DeviceType::SIMA_MLA, v,
                "from_vector MLA");
  const Tensor ev74 = cpu.cvu();
  device_tensor(ev74, DeviceType::SIMA_CVU, v, "cvu()");
  device_tensor(cpu.mla(true), DeviceType::SIMA_MLA, v, "mla(true)");
  device_tensor(cpu.to(Device{DeviceType::SIMA_MLA, 0}), DeviceType::SIMA_MLA, v, "to(MLA)");
  require(bytes_of(ev74.clone()) == v, "clone()");
  require(ev74.cpu().device.type == DeviceType::CPU && bytes_of(ev74.cpu()) == v, "cpu()");
  {
    Tensor w = cpu.cvu();
    {
      Mapping map = w.map(MapMode::Write);
      require(map.data != nullptr && map.size_bytes >= v.size(), "a device write map");
      std::memset(map.data, 0x7E, v.size());
    }
    require(bytes_of(w) == std::vector<std::uint8_t>(v.size(), 0x7E), "the write is read back");
  }
  // A transfer per frame takes memory a released tensor gave back: many in a row hold.
  for (int i = 0; i < 64; ++i) {
    const std::vector<std::uint8_t> f = pattern(256 * 1024, static_cast<std::uint8_t>(i));
    require(bytes_of(Tensor::from_vector(f, {256 * 1024}, TensorMemory::EV74)) == f,
            "a transfer per frame");
  }
  untouched("device tensors");
}

void device_graph(InputMemoryPolicy memory, DeviceType device) {
  ::setenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY", "1", 1);
  const TensorMemory placement =
      device == DeviceType::SIMA_MLA ? TensorMemory::MLA : TensorMemory::EV74;
  const Tensor seed = rgb(64, 48, 0x60, placement);
  Run run = input_output(image_input(memory)).build(TensorList{seed});
  for (int i = 0; i < 4; ++i) {
    const Tensor t = rgb(64, 48, static_cast<std::uint8_t>(0x61 + i), placement);
    require(run.push(TensorList{t}), "a device tensor into a device input");
    device_tensor(pull_one(run, "device graph"), device, bytes_of(t), "pulled", true);
  }
  const Tensor cpu = rgb(64, 48, 0x70, TensorMemory::CPU);
  require(run.push(TensorList{cpu}), "a CPU tensor into a device input");
  device_tensor(pull_one(run, "device graph, CPU"), device, bytes_of(cpu), "pulled copy", true);
  run.close();
  ::unsetenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY");
  untouched(std::string("device graph ") +
            (memory == InputMemoryPolicy::Dms0 ? "(DMS0 memory)" : "(EV74 memory)"));
}

// ---- models ---------------------------------------------------------------------------------

struct ModelCase {
  std::string package;
  std::string opts = "-";
};

std::vector<ModelCase> model_cases(int argc, char** argv) {
  std::vector<ModelCase> out;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--model=", 0) == 0) {
      out.push_back({arg.substr(8)});
    } else if (arg.rfind("--opts=", 0) == 0 && !out.empty()) {
      out.back().opts = arg.substr(7);
    }
  }
  return out;
}

// What the qualification harness pushes (compare.cpp): a 1280x720 frame of the view's format, or
// tensors of the model's input specs, in EV74 memory.
TensorList model_inputs(const Model& model, const q::OptionSet& set, int salt) {
  TensorList in;
  if (set.image) {
    Tensor t = rgb(1280, 720, static_cast<std::uint8_t>(salt), TensorMemory::CPU);
    if (set.frame_format == "BGR") {
      t.semantic.image->format = ImageSpec::PixelFormat::BGR;
    }
    in.push_back(t.cvu());
    return in;
  }
  for (const TensorSpec& spec : model.input_specs()) {
    const TensorDType dtype = spec.dtypes.empty() ? TensorDType::UInt8 : spec.dtypes.front();
    const std::size_t elem = dtype == TensorDType::UInt8 || dtype == TensorDType::Int8      ? 1
                             : dtype == TensorDType::Float64                                ? 8
                             : dtype == TensorDType::Int32 || dtype == TensorDType::Float32 ? 4
                                                                                            : 2;
    std::vector<std::int64_t> shape;
    std::size_t count = 1;
    for (const std::int64_t d : spec.shape) {
      shape.push_back(d > 0 ? d : 1);
      count *= static_cast<std::size_t>(d > 0 ? d : 1);
    }
    // Small values in every dtype: bytes of 0..63 make no NaN or infinity of a float.
    std::vector<std::uint8_t> bytes(count * elem);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<std::uint8_t>((i * 7U + static_cast<std::size_t>(salt)) & 0x3FU);
    }
    Tensor t =
        Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())}, TensorMemory::CPU);
    t.dtype = dtype;
    t.shape = shape;
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t step = static_cast<std::int64_t>(elem);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides[static_cast<std::size_t>(i)] = step;
      step *= shape[static_cast<std::size_t>(i)];
    }
    t.strides_bytes = strides;
    if (spec.image_format.has_value()) {
      t.semantic.image = ImageSpec{*spec.image_format, ""};
      if (shape.size() == 3U) {
        t.layout = TensorLayout::HWC;
      }
    }
    in.push_back(t.cvu());
  }
  return in;
}

// One form of a Model: it runs, or the Dataflow backend refuses it in the one not-carried form (a
// view whose route shape the backend does not carry, which the qualification harness expects as
// not_carried). Either way the process must still hold nothing of GStreamer.
void model_form(const std::string& what, const std::function<void()>& form) {
  try {
    form();
  } catch (const NeatError& e) {
    require_contains(e.what(), "is not carried by the Dataflow backend", what);
    std::cout << "[refused] " << what << " (not carried)\n";
  }
  untouched(what);
}

void model_case(const ModelCase& mc) {
  const q::OptionSet set = q::read_option_set(mc.opts);
  const std::string name = mc.package.substr(mc.package.find_last_of('/') + 1);
  Model model(mc.package, set.options);
  std::vector<TensorList> inputs;
  for (int k = 0; k < 4; ++k) {
    inputs.push_back(model_inputs(model, set, 0x10 * (k + 1)));
  }
  untouched(name + ": Model");
  model_form(name + ": Model::build + Runner", [&] {
    Model::Runner runner = model.build(inputs.front(), Model::RouteOptions{}, RunOptions{});
    for (int i = 0; i < 8; ++i) {
      require(runner.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), "Runner push");
      require(!runner.pull(20000).empty(), "Runner pull");
    }
    runner.close();
  });
  model_form(name + ": Model::run", [&] {
    for (int i = 0; i < 4; ++i) {
      require(!model.run(inputs[static_cast<std::size_t>(i)], 20000).empty(), "Model::run");
    }
  });
  model_form(name + ": a Graph of the Model's fragments", [&] {
    GraphOptions go;
    go.processcvu = set.options.processcvu;
    go.processmla = set.options.processmla;
    go.advanced_execution = set.options.advanced_execution;
    go.resolve_advanced_execution();
    Graph g(go);
    g.add(nodes::Input());
    g.add(model.preprocess());
    g.add(model.inference());
    g.add(model.postprocess());
    g.add(nodes::Output());
    Run run = g.build(inputs.front(), RunOptions{});
    for (int i = 0; i < 8; ++i) {
      require(run.push(inputs[static_cast<std::size_t>(i) % inputs.size()]), "Graph push");
      Result<TensorList> out = run.pull(std::chrono::milliseconds(20000));
      require(out.ok() && !out.value().empty(), "Graph pull");
    }
    run.close();
  });
  model_form(name + ": Model::benchmark", [&] {
    BenchmarkOptions bo;
    bo.num_samples = 20;
    bo.min_duration_ms = 0;
    if (set.image) {
      bo.inputs = inputs.front();
    }
    (void)model.benchmark(bo);
  });
  model_form(name + ": Model::execution_plan", [&] {
    require(!model.execution_plan().to_json().empty(), "Model::execution_plan");
  });
}

} // namespace

int main(int argc, char** argv) {
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  const std::vector<ModelCase> models = model_cases(argc, argv);
  return sima_test::run_test("unit_dataflow_gstreamer_free_test", [&] {
    require(sima_test::on_dataflow(), "registered with --backend=dataflow only");
    untouched("start");
    prewarm();
    cpu_tensors();
    host_graph();
    refusals();
    if (device_memory()) {
      untouched("the device memory probe");
      device_tensors();
      device_graph(InputMemoryPolicy::Ev74, DeviceType::SIMA_CVU);
      device_graph(InputMemoryPolicy::Dms0, DeviceType::SIMA_MLA);
    } else {
      std::cout << "[skipped] device tensors and device graphs: this build has no device memory\n";
    }
    for (const ModelCase& mc : models) {
      model_case(mc);
    }
    if (models.empty()) {
      std::cout << "[skipped] models: no --model=<package> given\n";
    }
    untouched("the end");
  });
}
