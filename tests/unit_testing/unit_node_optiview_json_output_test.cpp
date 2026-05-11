#include "nodes/io/OptiViewJsonOutput.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

RUN_TEST("unit_node_optiview_json_output_test", ([] {
           using nlohmann::json;

           sima_test::UdpReceiver rx;

           simaai::neat::OptiViewChannelOptions opt;
           opt.host = "127.0.0.1";
           opt.channel = 0;
           opt.json_port_base = rx.port();
           opt.video_port_base = 9200;

           std::string init_err;
           simaai::neat::OptiViewJsonOutput sender(opt, &init_err);
           require(sender.ok(), "OptiViewJsonOutput initialization failed: " + init_err);
           require(sender.json_port() == rx.port(), "OptiViewJsonOutput json_port mismatch");
           require(sender.video_port() == 9200, "OptiViewJsonOutput video_port mismatch");

           // Schema snapshot for helper JSON creator.
           const std::vector<simaai::neat::OptiViewObject> objects = {
               simaai::neat::OptiViewObject{
                   .x = 10, .y = 20, .w = 30, .h = 40, .score = 0.95f, .class_id = 0},
               simaai::neat::OptiViewObject{
                   .x = 1, .y = 2, .w = 3, .h = 4, .score = 0.15f, .class_id = 5},
           };
           const std::vector<std::string> labels = {"person"};
           const std::string json_text =
               simaai::neat::OptiViewMakeJson(12345, "frame-7", objects, labels);
           const json parsed_helper = json::parse(json_text);
           require(parsed_helper["type"].get<std::string>() == "object-detection",
                   "OptiViewMakeJson type mismatch");
           require(parsed_helper["timestamp"].get<int64_t>() == 12345,
                   "OptiViewMakeJson timestamp mismatch");
           require(parsed_helper["frame_id"].get<std::string>() == "frame-7",
                   "OptiViewMakeJson frame_id mismatch");
           require(parsed_helper["data"]["objects"].size() == 2,
                   "OptiViewMakeJson object count mismatch");
           require(parsed_helper["data"]["objects"][0]["label"].get<std::string>() == "person",
                   "OptiViewMakeJson known label mismatch");
           require(parsed_helper["data"]["objects"][1]["label"].get<std::string>() == "Unknown",
                   "OptiViewMakeJson unknown label fallback mismatch");
           const auto default_labels = simaai::neat::OptiViewDefaultLabels();
           require(default_labels.size() == 80, "OptiViewDefaultLabels should emit 80 labels");
           require(default_labels.front() == "label_0",
                   "OptiViewDefaultLabels first label mismatch");

           // send_json loopback behavior.
           std::string send_err;
           require(sender.send_json(json_text, &send_err),
                   "OptiViewJsonOutput send_json failed: " + send_err);

           std::string received;
           require(rx.recv_one(&received, 2000),
                   "OptiViewJsonOutput send_json payload not received");
           require(received == json_text, "OptiViewJsonOutput send_json payload mismatch");

           // send_detection convenience path.
           require(sender.send_detection(111, "f-1", objects, labels, &send_err),
                   "OptiViewJsonOutput send_detection failed: " + send_err);
           require(rx.recv_one(&received, 2000),
                   "OptiViewJsonOutput send_detection payload not received");
           const json parsed_detection = json::parse(received);
           require(parsed_detection["timestamp"].get<int64_t>() == 111,
                   "OptiViewJsonOutput send_detection timestamp mismatch");

           // Host-resolution failure path.
           simaai::neat::OptiViewChannelOptions bad_opt;
           bad_opt.host = "256.256.256.256";
           bad_opt.channel = 0;
           bad_opt.json_port_base = 9300;

           std::string bad_init_err;
           simaai::neat::OptiViewJsonOutput bad_sender(bad_opt, &bad_init_err);
           require(!bad_sender.ok(), "OptiViewJsonOutput should fail with invalid host");
           require_contains(bad_init_err, "getaddrinfo",
                            "OptiViewJsonOutput bad-host error mismatch");

           std::string bad_send_err;
           require(!bad_sender.send_json("{}", &bad_send_err),
                   "OptiViewJsonOutput should reject send_json when uninitialized");
           require_contains(bad_send_err, "not initialized",
                            "OptiViewJsonOutput uninitialized send error mismatch");
         }));
