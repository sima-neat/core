// neat_qualification_census: what the Dataflow backend's planner answers for each catalogue
// component under each option view (README.md, tier T0). CPU only: a package becomes nodes, a
// manifest and a plan; no device is opened, nothing is loaded onto the MLA, no Run is built.
//
// The question is the one a customer's call asks: Input -> preprocess() -> inference() ->
// postprocess() -> Output under the Model's own Graph options, the Input as the script feeds it
// (an image view: one frame of the script's format and size as the build's seed; a tensor view:
// the model's own tensor declaration). The node list equals the one Model::build makes for both
// views on every catalogue component (the census of the merged tree compared the two, X0.md).
// Each question is asked twice: on a board whose EV74 runs the Kernels firmware, and on one with
// the stock firmware, whose answer must be the firmware refusal for every EV74 kernel stage.
//
//   neat_qualification_census --list <file> --out <file.jsonl> [--frame-width 1280 --frame-height
//   720] list lines:  <component id> <package .tar.gz> <option file or ->
//
// One JSON line per (component, view), written as it is known (a crash loses one component).

#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/runtime/DataflowRoutePlan.h"

#include "qualification_common.h"

#include <simaai/scheduler/client/kernel.h>

#include <dirent.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
namespace q = simaai::neat::qualification;
namespace si = pipeline_internal::sima;
namespace client = ::simaai::scheduler::client;

// What the planner is told when the board's EV74 runs the stock firmware (the runtime's own
// sentence comes from the remoteproc; the census states the condition).
constexpr const char* kStockFirmware = "the EV74 runs the stock firmware";

// A frame as the planner reads it: shape, dtype, layout, planes; CPU memory, zero bytes.
Tensor packed_frame(int width, int height, int channels) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * channels, 0);
  Tensor t = Tensor::from_vector(bytes, {height, width, channels}, TensorMemory::CPU);
  t.layout = TensorLayout::HWC;
  return t;
}

Tensor nv12_frame(int width, int height) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * 3U / 2U, 0);
  Tensor t =
      Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())}, TensorMemory::CPU);
  t.layout = TensorLayout::HW;
  t.shape = {height, width};
  t.strides_bytes = {width, 1};
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::NV12, ""};
  Plane luma;
  luma.role = PlaneRole::Y;
  luma.shape = {height, width};
  luma.strides_bytes = {width, 1};
  Plane chroma;
  chroma.role = PlaneRole::UV;
  chroma.shape = {height / 2, width};
  chroma.strides_bytes = {width, 1};
  chroma.byte_offset = static_cast<std::int64_t>(width) * height;
  t.planes = {luma, chroma};
  return t;
}

std::string stage_kind(const si::StageStaticSpec& s) {
  if (s.payload_kind == si::StagePayloadKind::ProcessMla) {
    return "mla";
  }
  if (s.native_kernel) {
    const auto& r = *s.native_kernel;
    const std::string op = r.boxdecode_program                ? std::string("boxdecode")
                           : !r.requests.empty()              ? r.requests.front().operation
                           : !s.processcvu.graph_name.empty() ? s.processcvu.graph_name
                                                              : s.kernel_kind;
    return op + "@" + (r.target == client::Hardware::A65 ? "A65" : "EV74");
  }
  return (s.kernel_kind.empty() ? s.plugin_kind : s.kernel_kind) + "(no kernel)";
}

// The planner's answer: accepted, or the refusal's code, subject and fact.
struct Answer {
  bool accepted = false;
  std::string code, subject, fact, sentence, error;
};

Answer refusal(const pipeline_internal::DataflowNotCarried& declined) {
  Answer a;
  a.code = declined.code ? declined.code : "misconfig.pipeline_shape";
  a.subject = declined.subject;
  a.fact = declined.fact;
  a.sentence = pipeline_internal::dataflow_not_carried_sentence(declined);
  return a;
}

void put(q::Obj& o, const std::string& prefix, const Answer& a) {
  o.boolean(prefix + "accepted", a.accepted);
  if (!a.accepted) {
    o.str(prefix + "code", a.code).str(prefix + "subject", a.subject).str(prefix + "fact", a.fact);
    o.str(prefix + "sentence", a.sentence);
  }
  if (!a.error.empty()) {
    o.str(prefix + "error", a.error);
  }
}

std::ofstream g_out;

void emit(const std::string& line) {
  g_out << line << "\n";
  g_out.flush();
}

// One component under one view.
void ask(const std::string& id, const std::string& package, const std::string& view,
         const q::OptionSet& set, int width, int height) {
  const auto t0 = std::chrono::steady_clock::now();
  q::Obj o;
  o.str("record", "census").str("id", id).str("view", view);
  o.boolean("options_given", set.given).boolean("image", set.image);
  if (set.image) {
    o.str("frame_format", set.frame_format);
  }
  o.raw("options_not_applied", q::json_strings(set.not_applied));
  const auto finish = [&](q::Obj& rec) {
    rec.num("seconds",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    emit(rec.text());
  };
  std::unique_ptr<Model> model;
  try {
    model = std::make_unique<Model>(package, set.options);
    internal::ModelAccess::prepare_for_execution(*model, false);
  } catch (const std::exception& e) {
    o.str("error", std::string("model: ") + e.what());
    finish(o);
    return;
  }
  // The Model's own Graph options, as Model::build gives them to its Graph.
  GraphOptions graph;
  graph.processcvu = set.options.processcvu;
  graph.processmla = set.options.processmla;
  graph.advanced_execution = set.options.advanced_execution;
  graph.resolve_advanced_execution();
  InputOptions declared;
  std::vector<Tensor> seed;
  try {
    if (set.image) {
      seed.push_back(set.frame_format == "NV12"
                         ? nv12_frame(width, height)
                         : packed_frame(width, height, set.frame_format == "GRAY8" ? 1 : 3));
      o.str("seed", set.frame_format + " " + std::to_string(width) + "x" + std::to_string(height));
    } else {
      const auto list = model->input_appsrc_options_list(true);
      if (!list.empty()) {
        declared = list.front();
      }
      o.integer("declared_inputs", static_cast<long long>(list.size()));
    }
  } catch (const std::exception& e) {
    o.str("error", std::string("input: ") + e.what());
    finish(o);
    return;
  }
  std::vector<std::shared_ptr<Node>> nodes{nodes::Input(declared)};
  try {
    for (auto part : {internal::ModelAccess::build_public_preprocess_nodes(*model),
                      internal::ModelAccess::build_public_inference_nodes(*model),
                      internal::ModelAccess::build_public_postprocess_nodes(*model)}) {
      nodes.insert(nodes.end(), part.begin(), part.end());
    }
  } catch (const std::exception& e) {
    o.str("error", std::string("nodes: ") + e.what());
    finish(o);
    return;
  }
  OutputOptions out;
  out.max_buffers = 4;
  nodes.push_back(nodes::Output(out));
  std::optional<si::SimaPluginStaticManifest> manifest;
  Answer kernels, stock;
  try {
    pipeline_internal::DataflowNotCarried declined;
    manifest = runtime::dataflow_model_manifest(nodes, graph, &declined);
    if (!manifest) {
      kernels = refusal(declined);
      stock = kernels;
    }
  } catch (const std::exception& e) {
    kernels.error = stock.error = std::string("manifest: ") + e.what();
  }
  if (manifest) {
    std::vector<std::string> kinds;
    for (const auto& s : manifest->stages) {
      kinds.push_back(stage_kind(s));
    }
    o.raw("kinds", q::json_strings(kinds));
    for (Answer* a : {&kernels, &stock}) {
      try {
        pipeline_internal::DataflowNotCarried declined;
        const bool ok = static_cast<bool>(runtime::plan_dataflow_route_stages(
            *manifest, declared, seed, a == &kernels ? std::string() : std::string(kStockFirmware),
            &declined));
        if (ok) {
          a->accepted = true;
        } else {
          *a = refusal(declined);
        }
      } catch (const std::exception& e) {
        a->error = std::string("plan: ") + e.what();
      }
    }
  }
  put(o, "", kernels);
  put(o, "stock_", stock);
  finish(o);
}

} // namespace

int main(int argc, char** argv) {
  const q::Args a = q::Args::parse(argc, argv);
  if (!a.has("list") || !a.has("out")) {
    std::cerr << "usage: neat_qualification_census --list <file> --out <file.jsonl>\n";
    return 2;
  }
  g_out.open(a.get("out"), std::ios::binary | std::ios::app);
  if (!g_out) {
    std::cerr << "cannot write " << a.get("out") << "\n";
    return 2;
  }
  const int width = static_cast<int>(a.num("frame-width", 1280));
  const int height = static_cast<int>(a.num("frame-height", 720));
  std::ifstream list(a.get("list"));
  std::string line;
  while (std::getline(list, line)) {
    std::istringstream words(line);
    std::string id, package, opts;
    if (!(words >> id >> package)) {
      continue;
    }
    if (!(words >> opts)) {
      opts = "-";
    }
    q::OptionSet app;
    try {
      app = q::read_option_set(opts);
    } catch (const std::exception& e) {
      q::Obj o;
      o.str("record", "census").str("id", id).str("view", "app").str("error", e.what());
      emit(o.text());
    }
    ask(id, package, "default", q::OptionSet{}, width, height);
    // A component whose script sets no ModelOptions is built with the default options: its app
    // view is the default question again (recorded once more, under its own name).
    ask(id, package, "app", app, width, height);
    q::Obj done;
    done.str("record", "component_done").str("id", id);
    emit(done.text());
  }
  // The census opens no device: the device nodes this process holds at its end, as evidence.
  std::vector<std::string> devices;
  if (DIR* dir = opendir("/proc/self/fd")) {
    while (const dirent* e = readdir(dir)) {
      char target[512];
      const std::string link = std::string("/proc/self/fd/") + e->d_name;
      const ssize_t n = readlink(link.c_str(), target, sizeof target - 1);
      if (n <= 0) {
        continue;
      }
      target[n] = 0;
      const std::string t(target);
      if (t.rfind("/dev/", 0) == 0 && t != "/dev/null" && t.rfind("/dev/pts/", 0) != 0 &&
          t.rfind("/dev/shm/", 0) != 0 && t != "/dev/urandom" && t != "/dev/random") {
        devices.push_back(t);
      }
    }
    closedir(dir);
  }
  q::Obj end;
  end.str("record", "end").raw("open_device_nodes", q::json_strings(devices));
  emit(end.text());
  return 0;
}
