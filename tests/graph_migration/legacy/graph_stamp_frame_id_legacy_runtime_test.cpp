#include "graph/nodes/StampFrameId.h"
#include "graph_test_utils.h"
#include "test_main.h"

#include <string>
#include <vector>

RUN_TEST("graph_migration_legacy_graph_stamp_frame_id_test", ([] {
           using simaai::neat::graph::PortId;
           using simaai::neat::graph::StageMsg;
           using simaai::neat::graph::StageOutMsg;
           using simaai::neat::graph::StagePorts;
           using simaai::neat::graph::nodes::StampFrameId;

           StampFrameId stage;
           StagePorts ports;
           ports.out["out"] = static_cast<PortId>(7);
           stage.set_ports(ports);

           std::vector<StageOutMsg> out;

           StageMsg a0;
           a0.in_port = 1;
           a0.sample = sima_test::make_tensor_sample(-1, "");
           stage.on_input(std::move(a0), out);
           require(out.size() == 1, "StampFrameId should emit exactly one output");
           require(out.back().out_port == static_cast<PortId>(7), "StampFrameId out_port mismatch");
           require(out.back().sample.stream_id == "stream0",
                   "StampFrameId default stream_id mismatch");
           require(out.back().sample.frame_id == 0,
                   "StampFrameId should start stream0 counter at zero");

           StageMsg a1;
           a1.sample = sima_test::make_tensor_sample(-1, "stream0");
           stage.on_input(std::move(a1), out);
           require(out.back().sample.frame_id == 1,
                   "StampFrameId should stamp monotonic frame ids per stream");

           StageMsg b0;
           b0.sample = sima_test::make_tensor_sample(-1, "cam-b");
           stage.on_input(std::move(b0), out);
           require(out.back().sample.frame_id == 0,
                   "StampFrameId should keep an independent counter for each stream");

           StageMsg b_preserve;
           b_preserve.sample = sima_test::make_tensor_sample(42, "cam-b");
           stage.on_input(std::move(b_preserve), out);
           require(out.back().sample.frame_id == 42,
                   "StampFrameId should preserve pre-existing frame_id values");

           StageMsg b1;
           b1.sample = sima_test::make_tensor_sample(-1, "cam-b");
           stage.on_input(std::move(b1), out);
           require(out.back().sample.frame_id == 1,
                   "StampFrameId should continue monotonic stamping after preserved IDs");
         }));
