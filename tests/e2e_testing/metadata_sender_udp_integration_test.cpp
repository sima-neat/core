#include "nodes/io/MetadataSenderGroup.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>

RUN_TEST("metadata_sender_udp_integration_test", ([] {
           using nlohmann::json;
           using simaai::neat::MetadataSenderGroup;
           using simaai::neat::MetadataSenderGroupOptions;

           sima_test::UdpReceiver rx;

           MetadataSenderGroup group;
           MetadataSenderGroupOptions opt;
           opt.host = "127.0.0.1";
           opt.metadata_port_base = rx.port();

           std::string init_err;
           require(group.init(opt, 1, &init_err), "MetadataSenderGroup init failed: " + init_err);

           const std::string data_json = R"({
             "objects": [
               {"id":"obj_1","label":"car","confidence":0.92,"bbox":[10,20,30,40]},
               {"id":"obj_2","label":"dog","confidence":0.75,"bbox":[100,120,50,40]}
             ]
           })";

           std::string send_err;
           require(group.send_metadata(0, "object-detection", data_json, 4444, "777", &send_err),
                   "MetadataSender integration send_metadata failed: " + send_err);

           std::string payload;
           require(rx.recv_one(&payload, 2000),
                   "MetadataSender integration expected UDP payload not received");

           const json parsed = json::parse(payload);
           require(parsed["type"].get<std::string>() == "object-detection",
                   "MetadataSender integration type mismatch");
           require(parsed["frame_id"].get<std::string>() == "777",
                   "MetadataSender integration frame id mismatch");
           require(parsed["timestamp"].get<int64_t>() == 4444,
                   "MetadataSender integration timestamp mismatch");
           require(parsed["data"]["objects"].size() == 2,
                   "MetadataSender integration object count mismatch");
           require(parsed["data"]["objects"][0]["label"].get<std::string>() == "car",
                   "MetadataSender integration data payload mismatch");

           group.stop();
         }));
