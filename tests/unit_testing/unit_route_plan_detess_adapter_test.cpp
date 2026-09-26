#include "model/Model.h"
#include "model/internal/RoutePlanner.h"
#include "test_main.h"
#include "nodes/sima/Detess.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"

RUN_TEST("unit_route_plan_detess_adapter_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::internal;

           {
             DetessOptions options;
             auto compiled = std::make_shared<CompiledProcessCvuContract>();
             compiled->runtime_contract.plugin_kind = "processcvu";
             options.compiled_contract = compiled;
             const Detess node(options);
             require(node.contract_definition().plugin_kind == "processcvu" &&
                         node.backend_fragment(0).find("neatprocesscvu name=") == 0U,
                     "model-managed Detess preserves its physical CVU renderer");
             const Detess standalone;
             require(standalone.contract_definition().plugin_kind == "neatdetess",
                     "standalone Detess retains its existing renderer");
           }

           {
             Model::Options options;
             ModelSemantics semantics;
             semantics.tess_needed = true;
             semantics.quant_needed = false;
             semantics.has_post_detess_adapter = false;
             semantics.has_post_dequant_adapter = true;

             const SessionRoutePlan plan = build_route_plan(options, semantics, nullptr);
             require(plan.post_chain.empty(),
                     "detess-only route must not be satisfied by dequant adapter alone");
             require(plan.selected_post_kind == PostRouteStageKind::None,
                     "detess-only route must not select Detess without a detess adapter");
           }

           {
             Model::Options options;
             ModelSemantics semantics;
             semantics.tess_needed = true;
             semantics.quant_needed = false;
             semantics.has_post_detess_adapter = true;
             semantics.has_post_dequant_adapter = false;

             const SessionRoutePlan plan = build_route_plan(options, semantics, nullptr);
             require(plan.post_chain.size() == 1U &&
                         plan.post_chain.front() == SessionPostStageOp::Detess,
                     "detess-only route should select Detess when a detess adapter exists");
             require(plan.selected_post_kind == PostRouteStageKind::Detess,
                     "detess-only route selected_post_kind mismatch");
           }
         }));
