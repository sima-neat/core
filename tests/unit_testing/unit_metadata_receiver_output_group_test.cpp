#include "nodes/groups/MetadataReceiverOutputGroup.h"
#include "rtsp_port_utils.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>

RUN_TEST("unit_metadata_receiver_output_group_test", ([] {
           using nlohmann::json;
           using simaai::neat::nodes::groups::MetadataReceiverOutputGroup;
           using simaai::neat::nodes::groups::MetadataReceiverOutputGroupOptions;

           {
             MetadataReceiverOutputGroup group;
             MetadataReceiverOutputGroupOptions opt;
             std::string err;
             require(!group.init(opt, 0, &err),
                     "MetadataReceiverOutputGroup should reject zero streams");
             require_contains(err, "streams must be > 0",
                              "MetadataReceiverOutputGroup zero streams error mismatch");
           }

           const int metadata_base = rtsp_find_free_port_range(/*base_port=*/19000,
                                                               /*ports_needed=*/2,
                                                               /*stride=*/1,
                                                               /*max_tries=*/3000);
           require(metadata_base > 0, "failed to reserve contiguous UDP ports");

           sima_test::UdpReceiver rx0(metadata_base, "127.0.0.1");
           sima_test::UdpReceiver rx1(metadata_base + 1, "127.0.0.1");

           MetadataReceiverOutputGroup group;
           MetadataReceiverOutputGroupOptions opt;
           opt.host = "127.0.0.1";
           opt.metadata_port_base = metadata_base;

           std::string init_err;
           require(group.init(opt, 2, &init_err),
                   "MetadataReceiverOutputGroup init failed: " + init_err);
           require(group.size() == 2, "MetadataReceiverOutputGroup size mismatch");
           require(group.metadata_port(0) == metadata_base,
                   "MetadataReceiverOutputGroup stream 0 port mismatch");
           require(group.metadata_port(1) == metadata_base + 1,
                   "MetadataReceiverOutputGroup stream 1 port mismatch");
           require(group.metadata_port(2) == -1,
                   "MetadataReceiverOutputGroup invalid stream port mismatch");

           std::string err;
           require(!group.send_raw_json(2, "{}", &err),
                   "MetadataReceiverOutputGroup should reject invalid raw stream index");
           require_contains(err, "invalid stream index",
                            "MetadataReceiverOutputGroup invalid stream error mismatch");

           require(group.send_raw_json(0, R"({"type":"raw","data":{"stream":0}})", &err),
                   "MetadataReceiverOutputGroup send_raw_json failed: " + err);
           require(group.send_metadata(1, "tracking", R"({"tracks":[{"id":"trk-1"}]})", 42,
                                       "frame-42", &err),
                   "MetadataReceiverOutputGroup send_metadata failed: " + err);

           std::string payload0;
           std::string payload1;
           require(rx0.recv_one(&payload0, 2000),
                   "MetadataReceiverOutputGroup stream 0 payload not received");
           require(rx1.recv_one(&payload1, 2000),
                   "MetadataReceiverOutputGroup stream 1 payload not received");

           const json parsed0 = json::parse(payload0);
           require(parsed0["data"]["stream"].get<int>() == 0,
                   "MetadataReceiverOutputGroup raw payload mismatch");

           const json parsed1 = json::parse(payload1);
           require(parsed1["type"].get<std::string>() == "tracking",
                   "MetadataReceiverOutputGroup metadata type mismatch");
           require(parsed1["timestamp"].get<int64_t>() == 42,
                   "MetadataReceiverOutputGroup metadata timestamp mismatch");
           require(parsed1["frame_id"].get<std::string>() == "frame-42",
                   "MetadataReceiverOutputGroup metadata frame id mismatch");

           group.stop();
           require(group.size() == 0, "MetadataReceiverOutputGroup stop should clear senders");
         }));
