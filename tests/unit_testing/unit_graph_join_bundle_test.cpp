#include "graph/nodes/JoinBundle.h"
#include "graph_test_utils.h"
#include "test_main.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

RUN_TEST("unit_graph_join_bundle_test", ([] {
           using simaai::neat::graph::PortId;
           using simaai::neat::graph::StageMsg;
           using simaai::neat::graph::StageOutMsg;
           using simaai::neat::graph::StagePorts;
           using simaai::neat::graph::nodes::JoinBundle;
           using simaai::neat::graph::nodes::JoinBundleOptions;
           using simaai::neat::graph::nodes::JoinKeyPolicy;

           const PortId encoded_port = 11;
           const PortId meta_port = 12;
           const PortId out_port = 19;

           // StreamFrame determinism: full emission, stable order and cardinality.
           {
             JoinBundleOptions opt;
             opt.inputs = {"encoded", "meta"};
             opt.key_policy = JoinKeyPolicy::StreamFrame;

             JoinBundle join(opt);
             StagePorts ports;
             ports.in["encoded"] = encoded_port;
             ports.in["meta"] = meta_port;
             ports.out["bundle"] = out_port;
             join.set_ports(ports);

             std::vector<StageOutMsg> out;
             for (int fid = 0; fid < 6; ++fid) {
               StageMsg encoded;
               encoded.in_port = encoded_port;
               encoded.sample =
                   sima_test::make_tensor_sample(fid, "cam-frame", -1, static_cast<uint8_t>(fid));
               join.on_input(std::move(encoded), out);

               StageMsg meta;
               meta.in_port = meta_port;
               meta.sample = sima_test::make_tensor_sample(fid, "cam-frame", -1,
                                                           static_cast<uint8_t>(fid + 1));
               join.on_input(std::move(meta), out);
             }

             require(out.size() == 6,
                     "JoinBundle StreamFrame should emit exactly one bundle per frame pair");
             for (size_t i = 0; i < out.size(); ++i) {
               require(out[i].out_port == out_port, "JoinBundle StreamFrame out_port mismatch");
               require(out[i].sample.kind == simaai::neat::SampleKind::Bundle,
                       "JoinBundle StreamFrame output kind mismatch");
               require(out[i].sample.frame_id == static_cast<int64_t>(i),
                       "JoinBundle StreamFrame output order must be stable");
               require(out[i].sample.fields.size() == 2,
                       "JoinBundle StreamFrame bundle field cardinality mismatch");
               require(out[i].sample.fields[0].stream_label == "encoded",
                       "JoinBundle StreamFrame field[0] name mismatch");
               require(out[i].sample.fields[1].stream_label == "meta",
                       "JoinBundle StreamFrame field[1] name mismatch");
             }
           }

           // StreamPts determinism.
           {
             JoinBundleOptions opt;
             opt.inputs = {"encoded", "meta"};
             opt.key_policy = JoinKeyPolicy::StreamPts;

             JoinBundle join(opt);
             StagePorts ports;
             ports.in["encoded"] = encoded_port;
             ports.in["meta"] = meta_port;
             ports.out["bundle"] = out_port;
             join.set_ports(ports);

             std::vector<StageOutMsg> out;
             for (int i = 0; i < 4; ++i) {
               const int64_t pts = static_cast<int64_t>(1000 + i * 33);

               StageMsg encoded;
               encoded.in_port = encoded_port;
               encoded.sample =
                   sima_test::make_tensor_sample(-1, "cam-pts", pts, static_cast<uint8_t>(i + 10));
               join.on_input(std::move(encoded), out);

               StageMsg meta;
               meta.in_port = meta_port;
               meta.sample =
                   sima_test::make_tensor_sample(-1, "cam-pts", pts, static_cast<uint8_t>(i + 20));
               join.on_input(std::move(meta), out);
             }

             require(out.size() == 4,
                     "JoinBundle StreamPts should emit exactly one bundle per pts pair");
             for (size_t i = 0; i < out.size(); ++i) {
               const int64_t expected_pts = static_cast<int64_t>(1000 + i * 33);
               require(out[i].sample.pts_ns == expected_pts,
                       "JoinBundle StreamPts should preserve pts key deterministically");
             }
           }

           // emit_partial=true behavior.
           {
             JoinBundleOptions opt;
             opt.inputs = {"encoded", "meta"};
             opt.emit_partial = true;

             JoinBundle join(opt);
             StagePorts ports;
             ports.in["encoded"] = encoded_port;
             ports.in["meta"] = meta_port;
             ports.out["bundle"] = out_port;
             join.set_ports(ports);

             std::vector<StageOutMsg> out;

             StageMsg encoded;
             encoded.in_port = encoded_port;
             encoded.sample = sima_test::make_tensor_sample(77, "partial");
             join.on_input(std::move(encoded), out);

             require(out.size() == 1,
                     "JoinBundle emit_partial should emit on first available field");
             require(
                 out[0].sample.fields.size() == 1,
                 "JoinBundle emit_partial should produce single-field bundle when peer missing");
             require(out[0].sample.fields[0].stream_label == "encoded",
                     "JoinBundle emit_partial field name mismatch");
           }

           // timeout_ms and max_pending_keys eviction behavior.
           {
             JoinBundleOptions opt;
             opt.inputs = {"encoded", "meta"};
             opt.timeout_ms = 1;
             opt.max_pending_keys = 1;

             JoinBundle join(opt);
             StagePorts ports;
             ports.in["encoded"] = encoded_port;
             ports.in["meta"] = meta_port;
             ports.out["bundle"] = out_port;
             join.set_ports(ports);

             std::vector<StageOutMsg> out;

             StageMsg e0;
             e0.in_port = encoded_port;
             e0.sample = sima_test::make_tensor_sample(1, "evict");
             join.on_input(std::move(e0), out);

             StageMsg e1;
             e1.in_port = encoded_port;
             e1.sample = sima_test::make_tensor_sample(2, "evict");
             join.on_input(std::move(e1), out);

             StageMsg stale_meta;
             stale_meta.in_port = meta_port;
             stale_meta.sample = sima_test::make_tensor_sample(1, "evict");
             join.on_input(std::move(stale_meta), out);
             require(out.empty(), "JoinBundle should evict oldest pending key at max_pending_keys");

             join.on_tick(std::numeric_limits<int64_t>::max() / 2, out);

             StageMsg fresh_meta;
             fresh_meta.in_port = meta_port;
             fresh_meta.sample = sima_test::make_tensor_sample(2, "evict");
             join.on_input(std::move(fresh_meta), out);
             require(out.empty(),
                     "JoinBundle should evict timed-out pending key before complete bundle");
           }
         }));
