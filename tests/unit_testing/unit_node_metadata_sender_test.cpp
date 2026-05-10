#include "nodes/io/MetadataSender.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <string>

RUN_TEST(
    "unit_node_metadata_sender_test", ([] {
      using nlohmann::json;

      sima_test::UdpReceiver rx;

      simaai::neat::MetadataSenderOptions opt;
      opt.host = "127.0.0.1";
      opt.channel = 0;
      opt.metadata_port_base = rx.port();

      std::string init_err;
      simaai::neat::MetadataSender sender(opt, &init_err);
      require(sender.ok(), "MetadataSender initialization failed: " + init_err);
      require(sender.metadata_port() == rx.port(), "MetadataSender metadata_port mismatch");

      std::string send_err;
      require(sender.send_raw_json(R"({"type":"raw","data":{"ok":true}})", &send_err),
              "MetadataSender send_raw_json failed: " + send_err);

      std::string received;
      require(rx.recv_one(&received, 2000), "MetadataSender send_raw_json payload not received");
      require(json::parse(received)["type"].get<std::string>() == "raw",
              "MetadataSender send_raw_json payload mismatch");

      const std::string data_json = R"({"tracks":[{"id":"trk-1","bbox":[10,20,30,40]}]})";
      require(sender.send_metadata("tracking", data_json, 12345, "frame-7", &send_err),
              "MetadataSender send_metadata failed: " + send_err);

      require(rx.recv_one(&received, 2000), "MetadataSender send_metadata payload not received");
      const json parsed = json::parse(received);
      require(parsed["type"].get<std::string>() == "tracking",
              "MetadataSender send_metadata type mismatch");
      require(parsed["timestamp"].get<int64_t>() == 12345,
              "MetadataSender send_metadata timestamp mismatch");
      require(parsed["frame_id"].get<std::string>() == "frame-7",
              "MetadataSender send_metadata frame_id mismatch");
      require(parsed["data"]["tracks"][0]["id"].get<std::string>() == "trk-1",
              "MetadataSender send_metadata data mismatch");

      require(!sender.send_metadata("", data_json, 1, "frame", &send_err),
              "MetadataSender should reject empty metadata type");
      require_contains(send_err, "type must not be empty",
                       "MetadataSender empty type error mismatch");

      require(!sender.send_metadata("tracking", "{bad-json", 1, "frame", &send_err),
              "MetadataSender should reject invalid data_json");
      require_contains(send_err, "parse failed", "MetadataSender invalid data_json error mismatch");

      require(!sender.send_metadata("tracking", "[]", 1, "frame", &send_err),
              "MetadataSender should reject non-object data_json");
      require_contains(send_err, "must be a JSON object",
                       "MetadataSender non-object data_json error mismatch");

      simaai::neat::MetadataSenderOptions bad_opt;
      bad_opt.host = "256.256.256.256";
      bad_opt.channel = 0;
      bad_opt.metadata_port_base = 9300;

      std::string bad_init_err;
      simaai::neat::MetadataSender bad_sender(bad_opt, &bad_init_err);
      require(!bad_sender.ok(), "MetadataSender should fail with invalid host");
      require_contains(bad_init_err, "getaddrinfo", "MetadataSender bad-host error mismatch");

      std::string bad_send_err;
      require(!bad_sender.send_raw_json("{}", &bad_send_err),
              "MetadataSender should reject send_raw_json when uninitialized");
      require_contains(bad_send_err, "not initialized",
                       "MetadataSender uninitialized send error mismatch");
    }));
