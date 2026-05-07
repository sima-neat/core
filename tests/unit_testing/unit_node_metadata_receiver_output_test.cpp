#include "nodes/io/MetadataReceiverOutput.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

RUN_TEST("unit_node_metadata_receiver_output_test", ([] {
           using nlohmann::json;

           sima_test::UdpReceiver rx;

           simaai::neat::MetadataReceiverChannelOptions opt;
           opt.host = "127.0.0.1";
           opt.channel = 0;
           opt.metadata_port_base = rx.port();

           std::string init_err;
           simaai::neat::MetadataReceiverOutput sender(opt, &init_err);
           require(sender.ok(), "MetadataReceiverOutput initialization failed: " + init_err);
           require(sender.metadata_port() == rx.port(),
                   "MetadataReceiverOutput metadata_port mismatch");

           simaai::neat::MetadataReceiverPayload generic;
           generic.type = "tracking";
           generic.data_json = R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})";
           generic.timestamp_ms = 12345;
           generic.frame_id = "frame-7";

           std::string generic_json;
           std::string make_err;
           require(simaai::neat::MetadataReceiverMakeJson(generic, &generic_json, &make_err),
                   "MetadataReceiverMakeJson failed: " + make_err);
           const json parsed_generic = json::parse(generic_json);
           require(parsed_generic["type"].get<std::string>() == "tracking",
                   "MetadataReceiverMakeJson type mismatch");
           require(parsed_generic["timestamp"].get<int64_t>() == 12345,
                   "MetadataReceiverMakeJson timestamp mismatch");
           require(parsed_generic["frame_id"].get<std::string>() == "frame-7",
                   "MetadataReceiverMakeJson frame_id mismatch");
           require(parsed_generic["data"]["tracks"][0]["id"].get<std::string>() == "trk-1",
                   "MetadataReceiverMakeJson data mismatch");

           std::string send_err;
           require(sender.send_metadata(generic, &send_err),
                   "MetadataReceiverOutput send_metadata failed: " + send_err);

           std::string received;
           require(rx.recv_one(&received, 2000),
                   "MetadataReceiverOutput send_metadata payload not received");
           require(received == generic_json,
                   "MetadataReceiverOutput send_metadata payload mismatch");

           const std::vector<simaai::neat::MetadataReceiverObject> objects = {
               simaai::neat::MetadataReceiverObject{
                   .x = 10, .y = 20, .w = 30, .h = 40, .score = 0.95f, .class_id = 0},
               simaai::neat::MetadataReceiverObject{
                   .x = 1, .y = 2, .w = 3, .h = 4, .score = 0.15f, .class_id = 5},
           };
           const std::vector<std::string> labels = {"person"};
           const std::string detection_json =
               simaai::neat::MetadataReceiverMakeObjectDetectionJson(111, "f-1", objects, labels);
           const json parsed_detection = json::parse(detection_json);
           require(parsed_detection["type"].get<std::string>() == "object-detection",
                   "MetadataReceiver object detection type mismatch");
           require(parsed_detection["data"]["objects"].size() == 2,
                   "MetadataReceiver object count mismatch");
           require(parsed_detection["data"]["objects"][0]["label"].get<std::string>() == "person",
                   "MetadataReceiver known label mismatch");
           require(parsed_detection["data"]["objects"][1]["label"].get<std::string>() == "Unknown",
                   "MetadataReceiver unknown label fallback mismatch");

           require(sender.send_object_detection(222, "f-2", objects, labels, &send_err),
                   "MetadataReceiverOutput send_object_detection failed: " + send_err);
           require(rx.recv_one(&received, 2000),
                   "MetadataReceiverOutput send_object_detection payload not received");
           const json sent_detection = json::parse(received);
           require(sent_detection["timestamp"].get<int64_t>() == 222,
                   "MetadataReceiverOutput send_object_detection timestamp mismatch");

           simaai::neat::MetadataReceiverPayload invalid;
           invalid.type = "tracking";
           invalid.data_json = "{bad-json";
           require(!simaai::neat::MetadataReceiverMakeJson(invalid, &generic_json, &make_err),
                   "MetadataReceiverMakeJson should reject invalid data_json");
           require_contains(make_err, "parse failed",
                            "MetadataReceiverMakeJson invalid JSON error mismatch");

           simaai::neat::MetadataReceiverChannelOptions bad_opt;
           bad_opt.host = "256.256.256.256";
           bad_opt.channel = 0;
           bad_opt.metadata_port_base = 9300;

           std::string bad_init_err;
           simaai::neat::MetadataReceiverOutput bad_sender(bad_opt, &bad_init_err);
           require(!bad_sender.ok(), "MetadataReceiverOutput should fail with invalid host");
           require_contains(bad_init_err, "getaddrinfo",
                            "MetadataReceiverOutput bad-host error mismatch");

           std::string bad_send_err;
           require(!bad_sender.send_json("{}", &bad_send_err),
                   "MetadataReceiverOutput should reject send_json when uninitialized");
           require_contains(bad_send_err, "not initialized",
                            "MetadataReceiverOutput uninitialized send error mismatch");
         }));
