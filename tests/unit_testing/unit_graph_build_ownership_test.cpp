#include <gstsimaaitensorbuffer_meta.h>
#include "pipeline/ErrorCodes.h"
#include "gst/GstInit.h"
#include "pipeline/NeatError.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <simaai/scheduler/client/kernel.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace simaai::neat;
namespace si = pipeline_internal::sima;

std::size_t created = 0;
std::size_t finalized = 0;
std::size_t contexts_seen = 0;
std::size_t contexts_finalized = 0;
GstContext* observed_context = nullptr; // Weak identity only; never dereferenced after destruction.

struct OwnershipProbe {
  GstElement parent;
};
struct OwnershipProbeClass {
  GstElementClass parent;
};
G_DEFINE_TYPE(OwnershipProbe, ownership_probe, GST_TYPE_ELEMENT)

void ownership_probe_init(OwnershipProbe*) {
  ++created;
}

void ownership_probe_class_init(OwnershipProbeClass* klass) {
  auto* object = G_OBJECT_CLASS(klass);
  auto* element = GST_ELEMENT_CLASS(klass);
  object->finalize = +[](GObject* value) {
    ++finalized;
    G_OBJECT_CLASS(ownership_probe_parent_class)->finalize(value);
  };
  element->set_context = +[](GstElement* value, GstContext* context) {
    if (sima_plugin_manifest_context_matches(context) && context != observed_context) {
      observed_context = context;
      ++contexts_seen;
      gst_mini_object_weak_ref(
          GST_MINI_OBJECT(context), +[](gpointer, GstMiniObject*) { ++contexts_finalized; },
          nullptr);
    }
    GST_ELEMENT_CLASS(ownership_probe_parent_class)->set_context(value, context);
  };
  gst_element_class_set_static_metadata(element, "Build ownership probe", "Testing",
                                        "Counts actual graph and context finalization", "SiMa.ai");
}

BuildResult build_input() {
  BuildResult build;
  build.pipeline_string = "coreownershipprobe name=ownership identity name=companion";
  build.diag = std::make_shared<DiagCtx>();
  build.rendered_manifest.emplace();
  auto& stage = build.rendered_manifest->stages.emplace_back();
  stage.element_name = "ownership";
  stage.logical_stage_id = "ownership_stage";
  stage.plugin_kind = "metadata";
  return build;
}

void reset_observation() {
  require(created == finalized && contexts_seen == contexts_finalized,
          "previous graph/context must close before starting a new case");
  created = finalized = contexts_seen = contexts_finalized = 0;
  observed_context = nullptr;
  session_test::clear_composition_failure_for_test();
}

void assert_closed(bool context_expected) {
  require(created == 1 && finalized == 1, "every parsed probe must finalize on the cold exit");
  require(contexts_seen == (context_expected ? 1U : 0U) && contexts_finalized == contexts_seen,
          "temporary and pipeline metadata-context references must both close");
}

struct OriginalFactoryException final : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void cold_rejection_releases_graph_and_recipe(bool reject_output_span,
                                             bool derived_exception = false) {
  namespace rt = simaai::scheduler::client;
  namespace gs = rt;
  namespace kernels = simaai::scheduler::client;
  constexpr const char* failure = "owned cold resolver factory failed";
  std::weak_ptr<const si::KernelStageRecipe> observer;
  bool threw = false;
  {
    auto build = build_input();
    build.pipeline_string += " fakesrc ! neatkernelrunner name=native ! fakesink";
    auto recipe = std::make_shared<si::KernelStageRecipe>();
    observer = recipe;
    recipe->target = rt::Target::EV74;
    const std::array inputs{rt::TensorSpec{{2}, rt::DType::Float32, kernels::Contiguous{}}};
    const std::array outputs{rt::TensorSpec{{2}, rt::DType::BFloat16, kernels::Contiguous{}}};
    auto checked = simaai::scheduler::client::check_kernel(
        {"cast", kernels::Target::EV74, inputs, outputs, {}});
    require(std::holds_alternative<rt::KernelExecutable>(checked),
            "real cold Cast checking must succeed before the factory failure test");
    auto executable = std::get<rt::KernelExecutable>(std::move(checked));
    auto inspected = rt::inspect_kernel(executable);
    require(std::holds_alternative<rt::KernelInspection>(inspected),
            "real cold Cast inspection must succeed before constructing its placement");
    const auto& inspection = std::get<rt::KernelInspection>(inspected);
    require(inspection.input_requirements().size() == 1 &&
                inspection.output_requirements().size() == 1,
            "ownership fixture requires exactly one actual Cast input and output");
    const auto input_requirement = inspection.input_requirements().front();
    const auto output_requirement = inspection.output_requirements().front();
    require(input_requirement.storage_bytes == 8 && output_requirement.storage_bytes == 4,
            "ownership fixture must retain its FP32-to-BF16 checked extents");
    recipe->members.push_back({std::move(executable),
                               {{0, 0, gs::TensorStagePortSource::Input}},
                               {{0, 0, gs::TensorStagePortSource::Output}},
                               {},
                               {}});
    if (!reject_output_span)
      recipe->resolver_factories.push_back({0, [failure, derived_exception]() -> si::KernelStageRecipe::Resolver {
                                              if (derived_exception)
                                                throw OriginalFactoryException(failure);
                                              throw std::runtime_error(failure);
                                            }});
    gs::TensorStagePublication publication;
    publication.input_caps = {gst_caps_new_empty_simple("application/vnd.simaai.tensor"),
                              gst_caps_unref};
    publication.output_caps = publication.input_caps;
    auto contract = std::make_shared<simaai::gst::TensorBufferPublishContract>();
    contract->stage_key = "native";
    contract->physical_outputs.push_back({0, 4, "output"});
    simaai::gst::TensorBufferPublishLogicalOutput output;
    output.logical_index = output.physical_index = output.memory_index = 0;
    output.backend_output_index = output.route_slot = 0;
    output.logical_name = output.backend_name = output.segment_name = "output";
    // The physical carrier remains large enough. A short logical publication
    // must fail cold admission; enlarging the parent is not authorization.
    output.size_bytes = reject_output_span ? 3 : 4;
    output.dtype = SIMA_TENSOR_SET_DTYPE_BF16_V1;
    output.shape = {2};
    output.stride_bytes = {2};
    contract->logical_outputs.push_back(std::move(output));
    publication.contract = std::move(contract);
    recipe->publication = std::move(publication);
    auto& stage = build.rendered_manifest->stages.emplace_back();
    stage.element_name = stage.logical_stage_id = "native";
    stage.plugin_kind = "kernel";
    stage.native_kernel = recipe;
    // Author the same real arena and checked routes as a compiled Core stage.
    // Missing placement must not preempt the intended factory/span failures.
    stage.frame_arena_size_bytes = 4096;
    stage.frame_arena_role = si::FrameArenaRole::Allocate;
    stage.frame_arena_storage_domain = si::static_contract::ArenaStorageDomain::Cma;
    stage.frame_arena_provenance = si::static_contract::ArenaAllocationProvenance::CoreAllocated;
    stage.frame_arena_required_device_access =
        static_cast<std::uint32_t>(si::static_contract::ArenaDeviceAccess::Ev74);
    stage.logical_inputs.push_back(si::specbuilders::build_logical_input_static_spec(
        0, 0, 0, {2}, "FP32", "C", "input", "input", "input"));
    stage.input_bindings.push_back(si::specbuilders::build_input_binding_static_spec(
        0, 0, "input", "input"));
    stage.physical_inputs.push_back(si::specbuilders::build_physical_buffer_static_spec(
        0, 0, input_requirement.storage_bytes, si::DeviceKind::Cpu, "input", -1, 0,
        input_requirement.alignment_bytes));
    stage.physical_outputs.push_back(si::specbuilders::build_physical_buffer_static_spec(
        0, 0, output_requirement.storage_bytes, si::DeviceKind::Cpu, "output", -1, 0,
        output_requirement.alignment_bytes));
    try {
      std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline(
          session_build_parse_pipeline_or_throw(build, "ownership-factory-test"),
          &gst_object_unref);
    } catch (const std::runtime_error& error) {
      threw = true;
      if (reject_output_span) {
        const auto* typed = dynamic_cast<const NeatError*>(&error);
        require(typed && typed->report().error_code == error_codes::kKernelBinding,
                "invalid publication must retain a typed kernel-binding diagnostic");
        require_contains(error.what(), "output.span",
                         "cold span rejection must not be replaced by cleanup failure");
      } else {
        require(std::string(error.what()) == failure,
                "the actual cold provider-factory exception must remain unchanged; actual=" +
                    std::string(error.what()));
        require(!derived_exception || dynamic_cast<const OriginalFactoryException*>(&error),
                "the original cold factory exception dynamic type must survive cleanup");
      }
    }
    require(!observer.expired(), "the caller still deliberately owns its reusable recipe");
  }
  require(threw, "the real cold construction rejection was not exercised");
  require(observer.expired(), "a failed graph must not record or retain the caller's recipe");
  assert_closed(true);
}

} // namespace

RUN_TEST("unit_graph_build_ownership_test", ([] {
           // This tests cold ownership, not full platform/provider startup. Use the
           // supported manual-init profile, then retain Core metadata/element setup.
           g_setenv("SIMA_ALLOW_GST_INIT", "1", TRUE);
           gst_init(nullptr, nullptr);
           gst_init_once();
           // CTest intentionally disables automatic system scanning. Core has now
           // selected its filtered system-plugin directory; load that real cohort.
           if (const char* paths = g_getenv("GST_PLUGIN_SYSTEM_PATH_1_0")) {
             gchar** directories = g_strsplit(paths, G_SEARCHPATH_SEPARATOR_S, -1);
             for (gchar** directory = directories; *directory; ++directory) {
               if (**directory)
                 gst_registry_scan_path(gst_registry_get(), *directory);
             }
             g_strfreev(directories);
           }
           for (const char* name : {"identity", "fakesrc", "fakesink", "neatkernelrunner"}) {
             GstElementFactory* factory = gst_element_factory_find(name);
             require(factory != nullptr, std::string("required real factory absent: ") + name);
             gst_object_unref(factory);
           }
           require(gst_element_register(nullptr, "coreownershipprobe", GST_RANK_NONE,
                                        ownership_probe_get_type()),
                   "ownership probe factory registration must succeed");
           const std::array points{
               session_test::CompositionFailurePoint::PipelineBeforeManifest,
               session_test::CompositionFailurePoint::PipelineStaticManifestAcquired,
               session_test::CompositionFailurePoint::PipelinePrepared};
           for (const auto point : points) {
             reset_observation();
             auto build = build_input();
             session_test::arm_composition_failure_for_test(point);
             bool threw = false;
             try {
               std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline(
                   session_build_parse_pipeline_or_throw(build, "ownership-exception-test"),
                   &gst_object_unref);
             } catch (const std::runtime_error& error) {
               threw = true;
               require(std::string(error.what()) == "injected Graph composition failure",
                       "the original injected exception must survive cold cleanup unchanged");
             }
             require(threw, "each actual graph construction failure point must be exercised");
             assert_closed(point != session_test::CompositionFailurePoint::PipelineBeforeManifest);
           }
           reset_observation();
           auto invalid = build_input();
           invalid.kernel_runner_a65_workers = 0;
           bool rejected = false;
           try {
             std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline(
                 session_build_parse_pipeline_or_throw(invalid, "ownership-typed-error-test"),
                 &gst_object_unref);
           } catch (const NeatError& error) {
             rejected = true;
             require(error.report().error_code == error_codes::kKernelBinding,
                     "typed invalid-worker diagnostics must remain kernel-binding errors; actual=" +
                         error.report().error_code + "\n" + error.what());
             require_contains(error.what(), "A65 worker count must be between 1 and 64",
                              "typed preparation diagnostics must remain actionable; actual=" +
                                  std::string(error.what()) +
                                  "\nrepro=" + error.report().repro_note);
           }
           require(rejected, "invalid cold cohort options must fail after metadata attachment");
           assert_closed(true);
           reset_observation();
           cold_rejection_releases_graph_and_recipe(false);
           reset_observation();
           cold_rejection_releases_graph_and_recipe(false, true);
           reset_observation();
           cold_rejection_releases_graph_and_recipe(true);
           reset_observation();
           auto build = build_input();
           std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline(
               session_build_parse_pipeline_or_throw(build, "ownership-success-test"),
               &gst_object_unref);
           require(pipeline && created == 1 && finalized == 0 && contexts_seen == 1 &&
                       contexts_finalized == 0,
                   "successful construction must transfer a live pipeline to its caller");
           pipeline.reset();
           assert_closed(true);
         }));
