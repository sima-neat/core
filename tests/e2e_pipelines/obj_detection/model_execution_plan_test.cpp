#include "model/Model.h"
#include "pipeline/ExecutionPlan.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/runtime/ExecutionPlanAssembly.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/RunCore.h"
#include "gst/GstInit.h"

#include <gst/gst.h>

#include "asset_utils.h"
#include "cli_utils.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

json parse_plan(const simaai::neat::ExecutionPlan& plan, const std::string& where) {
  require(plan.schema_version() == 1, where + ": schema_version should be 1");
  json doc = json::parse(plan.to_json());
  require(doc.at("schema").get<std::string>() == "sima.neat.execution_plan",
          where + ": wrong schema name");
  require(doc.at("schema_version").get<int>() == 1, where + ": wrong schema_version");
  return doc;
}

// Test-only observation of the actual temporary graph. No element/pad references
// are retained: weak notifications prove observed Gst finalization, while source
// pad probes count buffers that cross appsrc. This does not prove universal zero
// enqueue or lower-level DMA retirement.
class GraphObservation {
  struct State {
    std::mutex mutex;
    std::unordered_set<GObject*> live;
    std::atomic<unsigned> pipelines{0}, sources{0}, buffers{0};
    std::atomic<bool> failed{false};
  };
  std::shared_ptr<State> state_ = std::make_shared<State>();
  guint signal_ = 0;
  gulong hook_ = 0;

  static bool track(const std::shared_ptr<State>& state, GObject* object) {
    std::lock_guard lock(state->mutex);
    if (!state->live.insert(object).second)
      return false;
    auto context = std::make_unique<std::shared_ptr<State>>(state);
    g_object_weak_ref(object, finalized, context.get());
    context.release();
    return true;
  }
  static void finalized(gpointer data, GObject* object) noexcept {
    std::unique_ptr<std::shared_ptr<State>> context(static_cast<std::shared_ptr<State>*>(data));
    const auto state = *context;
    try {
      std::lock_guard lock(state->mutex);
      state->live.erase(object);
    } catch (...) {
      state->failed = true;
    }
  }
  static GstPadProbeReturn buffer(GstPad*, GstPadProbeInfo* info, gpointer data) noexcept {
    auto& state = **static_cast<std::shared_ptr<State>*>(data);
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER)
      ++state.buffers;
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER_LIST)
      state.buffers += gst_buffer_list_length(GST_PAD_PROBE_INFO_BUFFER_LIST(info));
    return GST_PAD_PROBE_OK;
  }
  static gboolean added(GSignalInvocationHint*, guint count, const GValue* values,
                        gpointer data) noexcept {
    const auto state = *static_cast<std::shared_ptr<State>*>(data);
    try {
      require(count == 3, "unexpected graph observation signature");
      auto* root = G_OBJECT(g_value_get_object(&values[0]));
      if (GST_IS_PIPELINE(root) && track(state, root))
        ++state->pipelines;
      auto* element = GST_ELEMENT(g_value_get_object(&values[2]));
      if (!track(state, G_OBJECT(element)))
        return TRUE;
      auto* factory = gst_element_get_factory(element);
      if (!factory ||
          !g_str_equal(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "appsrc"))
        return TRUE;
      std::unique_ptr<GstPad, decltype(&gst_object_unref)> pad(
          gst_element_get_static_pad(element, "src"), gst_object_unref);
      require(bool(pad), "observed appsrc has no source pad");
      auto context = std::make_unique<std::shared_ptr<State>>(state);
      const auto types =
          static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST);
      const auto id = gst_pad_add_probe(pad.get(), types, buffer, context.get(), [](gpointer p) {
        delete static_cast<std::shared_ptr<State>*>(p);
      });
      require(id != 0, "cannot observe appsrc buffers");
      context.release();
      ++state->sources;
    } catch (...) {
      state->failed = true;
    }
    return TRUE;
  }

public:
  GraphObservation() {
    simaai::neat::gst_init_once();
    auto* klass = g_type_class_ref(GST_TYPE_BIN);
    signal_ = g_signal_lookup("deep-element-added", GST_TYPE_BIN);
    g_type_class_unref(klass);
    require(signal_ != 0, "graph observation signal unavailable");
    auto context = std::make_unique<std::shared_ptr<State>>(state_);
    hook_ = g_signal_add_emission_hook(signal_, 0, added, context.get(), [](gpointer p) {
      delete static_cast<std::shared_ptr<State>*>(p);
    });
    require(hook_ != 0, "cannot observe graph construction");
    context.release();
  }
  GraphObservation(const GraphObservation&) = delete;
  GraphObservation& operator=(const GraphObservation&) = delete;
  ~GraphObservation() {
    if (hook_)
      g_signal_remove_emission_hook(signal_, hook_);
  }
  void verify_closed(bool executed) {
    require(!state_->failed, "graph observation failed");
    require(state_->pipelines > 0 && state_->sources > 0,
            "graph observation must see a real pipeline and appsrc");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
      {
        std::lock_guard lock(state_->mutex);
        if (state_->live.empty())
          break;
      }
      require(std::chrono::steady_clock::now() < deadline,
              "observed Gst objects must finalize after the Runner closes");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(!state_->failed, "graph finalization observation failed");
    require(executed ? state_->buffers > 0 : state_->buffers == 0,
            executed ? "subsequent execution must exercise the appsrc observer"
                     : "inspection must emit zero observed appsrc buffers");
  }
};

// Every node says where it runs, accelerator nodes name their kernel, and every edge between
// two compiled nodes carries batch-first tensors.
void require_plan_shape(const json& doc, const std::string& where) {
  const json& model = doc.at("model");
  require(!model.at("name").get<std::string>().empty(), where + ": model name is empty");
  const int batch = model.at("batch").get<int>();
  require(batch >= 1, where + ": batch should be at least 1");

  const std::set<std::string> engines = {"A65", "EV74", "MLA"};
  bool has_mla = false;
  for (const auto& node : doc.at("nodes")) {
    const std::string kind = node.at("kind").get<std::string>();
    const std::string runs_on = node.at("runs_on").get<std::string>();
    require(engines.count(runs_on) == 1U, where + ": node " + kind + " runs on " + runs_on);
    has_mla = has_mla || runs_on == "MLA";
    const bool endpoint = kind == "Input" || kind == "Output";
    if (!endpoint && runs_on != "MLA") {
      require(node.contains("kernel"), where + ": node " + kind + " names no kernel");
    }
  }
  require(has_mla, where + ": no node runs on the MLA");

  const auto& nodes = doc.at("nodes");
  const auto& edges = doc.at("edges");
  require(edges.size() + 1U == nodes.size(), where + ": a linear plan has one edge per node pair");
  for (const auto& edge : edges) {
    if (!edge.contains("tensors")) {
      continue;
    }
    for (const auto& tensor : edge.at("tensors")) {
      const auto& shape = tensor.at("shape");
      require(!shape.empty() && shape.front().get<int>() == batch,
              where + ": tensor " + tensor.at("name").get<std::string>() +
                  " shape is not batch-first");
    }
  }
}

// A zero tensor in EV74 memory shaped like the Model input, as applications seed build.
simaai::neat::TensorList ev74_seed(const simaai::neat::Model& model) {
  const simaai::neat::InputOptions opt = model.input_appsrc_options(true);
  const int height = opt.height > 0 ? opt.height : opt.max_height;
  const int width = opt.width > 0 ? opt.width : opt.max_width;
  const int depth = opt.depth > 0 ? opt.depth : opt.max_depth;
  const std::vector<float> zeros(static_cast<std::size_t>(height) * width * depth, 0.0F);
  return {simaai::neat::Tensor::from_vector(zeros, {height, width, depth},
                                            simaai::neat::TensorMemory::EV74)};
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  std::string tar_gz;
  sima_test::get_arg(argc, argv, "--model", tar_gz);
  if (tar_gz.empty()) {
    tar_gz = sima_test::resolve_yolov8s_tar();
  }
  if (tar_gz.empty() || !fs::exists(tar_gz)) {
    std::cerr << "Failed to resolve the yolov8s model tar.gz. Set SIMA_YOLO_TAR or pass "
                 "--model <path>.\n";
    return 3;
  }

  try {
    simaai::neat::Model model(tar_gz);

    simaai::neat::RunOptions inspection_options;
    inspection_options.startup_preflight = true; // execution_plan must override this.
    auto inspected = [&](const auto& query, const std::string& label) {
      GraphObservation observation;
      const auto owned_plan = query();
      observation.verify_closed(false);
      // The immutable report must remain readable after temporary graph finalization.
      return parse_plan(owned_plan, label);
    };
    const json planned = inspected([&] { return model.execution_plan(inspection_options); },
                                   "Model::execution_plan");
    require_plan_shape(planned, "Model::execution_plan");
    require(inspected([&] { return model.execution_plan(inspection_options); },
                      "second Model::execution_plan") == planned,
            "planning twice should give the same plan");

    const simaai::neat::TensorList seed = ev74_seed(model);
    require(inspected([&] { return model.execution_plan(seed, {}, inspection_options); },
                      "seeded Model::execution_plan") == planned,
            "a seed shaped like the Model input should not change the plan");

    {
      GraphObservation observation;
      simaai::neat::RunOptions execution_options;
      execution_options.startup_preflight = false;
      auto runner = model.build(seed, {}, execution_options);
      auto prepared_plan = [&] {
        // Read the actual runner's prepared graph; do not manufacture a separate plan.
        const auto core =
            simaai::neat::run_internal::core(simaai::neat::internal::ModelAccess::run(runner));
        const auto diag = core ? core->pipeline.stream.diag_ctx() : nullptr;
        require(diag && diag->build_plan, "subsequent runner must retain its prepared plan");
        return parse_plan(simaai::neat::internal::ExecutionPlanAccess::make(
                              *diag->build_plan, planned.at("model").at("name")),
                          "subsequent runner");
      };
      require(prepared_plan() == planned, "inspection must match the next prepared graph");
      require(!runner.run(seed, 30000).empty(), "subsequent real model execution must succeed");
      require(prepared_plan() == planned, "execution must retain the inspected prepared graph");
      runner.close();
      observation.verify_closed(true);
    }

    require(simaai::neat::ExecutionPlan{}.to_json(-1) == "{}",
            "a default ExecutionPlan should be empty");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] model_execution_plan_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "[OK] model_execution_plan_test passed\n";
  return 0;
}
