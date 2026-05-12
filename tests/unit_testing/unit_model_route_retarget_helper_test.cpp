#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelRouteRetarget.h"
#include "mpk_fixture_utils.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/BoxDecodeType.h"
#include "pipeline/session/internal/SessionBuildInternal.h"
#include "test_main.h"

#include <filesystem>

namespace {

sima_test::MpkFixture make_quanttess_detessdequant_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
                                                    {"etc/pipeline_sequence.json",
                                                     R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "quanttess_0",
        "pluginId": "processcvu",
        "configPath": "0_quanttess.json",
        "processor": "CVU",
        "kernel": "quanttess",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "quanttess_0"
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_postproc.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
      }
    ]
  }]
})json"},
                                                    {"etc/0_quanttess.json",
                                                     R"json({
  "node_name": "quanttess_0",
  "input_width": 640,
  "input_height": 640,
  "input_depth": 3
})json"},
                                                    {"etc/0_process_mla.json",
                                                     R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "quanttess_0"}],
  "input_format": ["EV81_INT8"],
  "data_type": ["EV81_INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                                    {"etc/0_postproc.json",
                                                     R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                },
                                                true);
}

class ManualPostProbeNode final : public simaai::neat::Node,
                                  public simaai::neat::internal::ModelLineageProvider {
public:
  explicit ManualPostProbeNode(
      std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> binding)
      : binding_(std::move(binding)) {}

  std::string kind() const override {
    return "ManualPostProbe";
  }

  std::string backend_fragment(int /*node_index*/) const override {
    return "";
  }

  std::vector<std::string> element_names(int /*node_index*/) const override {
    return {"manual_post_probe"};
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  const simaai::neat::internal::ModelLineageBinding* model_lineage_binding() const override {
    return binding_.get();
  }

private:
  std::shared_ptr<const simaai::neat::internal::ModelLineageBinding> binding_;
};

} // namespace

RUN_TEST("unit_model_route_retarget_helper_test", ([] {
           using namespace simaai::neat;

           const auto fixture =
               make_quanttess_detessdequant_fixture("model_route_retarget_boxdecode");

           Model model(fixture.tar_path);

           require(internal::ModelAccess::resolved_post_kind(model) ==
                       internal::PostRouteStageKind::DetessDequant,
                   "fixture should start on detessdequant route");

           const auto base_binding = internal::make_model_lineage_binding(
               model, internal::ModelLineageStageRole::ManualPost,
               internal::RequestedPostRouteKind::BoxDecode, "SimaBoxDecode");
           auto probe_binding = std::make_shared<internal::ModelLineageBinding>(*base_binding);
           probe_binding->base_options.decode_type = BoxDecodeType::YoloV8;

           bool changed = false;
           std::string err;
           auto effective = internal::build_effective_model_for_requested_post(
               *base_binding, BoxDecodeType::YoloV8, &changed, &err);
           require(effective != nullptr, "retarget helper should produce an effective model");
           require(err.empty(), "retarget helper should not emit an error");
           require(changed, "retarget helper should report a changed model");
           require(internal::ModelAccess::resolved_post_kind(*effective) ==
                       internal::PostRouteStageKind::BoxDecode,
                   "retargeted model should resolve to boxdecode");

           // The bundled strict_seed_mpk does not declare an explicit
           // decode_type, so the retargeted model has no canonical
           // model-managed boxdecode contract and the model-bound
           // SimaBoxDecode constructor must fail. Validate the negative
           // path; coverage of the positive retarget+boxdecode flow is
           // gated on a fixture whose MPK declares decode_type.
           bool boxdecode_unavailable = false;
           try {
             (void)internal::ModelAccess::build_boxdecode_stage_contract(*effective, false);
           } catch (const std::exception&) {
             boxdecode_unavailable = true;
           }
           require(boxdecode_unavailable,
                   "MPK without decode_type should not expose a model-managed boxdecode contract");

           std::vector<std::shared_ptr<Node>> nodes;
           nodes.push_back(std::make_shared<ManualPostProbeNode>(probe_binding));

           bool threw_conflict = false;
           try {
             std::vector<std::shared_ptr<Node>> conflicting = nodes;
             conflicting.push_back(std::make_shared<ManualPostProbeNode>(probe_binding));
             (void)session_build_materialize_model_bound_nodes(conflicting, true);
           } catch (const std::exception&) {
             threw_conflict = true;
           }
           require(threw_conflict,
                   "multiple manual post nodes on one model lineage should be rejected");
         }));
