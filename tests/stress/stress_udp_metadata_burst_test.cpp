#include "nodes/io/MetadataSenderGroup.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(40, std::min(value, 4000));
}

bool is_parseable_metadata_receiver_json(const std::string& payload) {
  try {
    const auto parsed = nlohmann::json::parse(payload);
    if (!parsed.contains("type") || !parsed["type"].is_string())
      return false;
    if (!parsed.contains("data") || !parsed["data"].is_object())
      return false;
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace

RUN_TEST("stress_udp_metadata_burst_test", ([] {
           using simaai::neat::MetadataSenderGroup;
           using simaai::neat::MetadataSenderGroupOptions;

           const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 180));
           const int streams = 2;
           const int metadata_port_base = 9900;

           sima_test::UdpReceiver rx0(metadata_port_base);
           sima_test::UdpReceiver rx1(metadata_port_base + 1);

           MetadataSenderGroup group;
           MetadataSenderGroupOptions opt;
           opt.host = "127.0.0.1";
           opt.metadata_port_base = metadata_port_base;

           std::string init_err;
           require(group.init(opt, streams, &init_err),
                   "MetadataSenderGroup init failed: " + init_err);

           int emitted = 0;
           int emit_fail = 0;

           for (int i = 0; i < iters; ++i) {
             for (int s = 0; s < streams; ++s) {
               nlohmann::json data;
               data["objects"] = nlohmann::json::array(
                   {nlohmann::json{{"id", "obj_1"},
                                   {"label", s == 0 ? "person" : "car"},
                                   {"confidence", 0.90},
                                   {"bbox", {10 + (i % 30), 20 + (i % 25), 40, 35}}},
                    nlohmann::json{{"id", "obj_2"},
                                   {"label", "dog"},
                                   {"confidence", 0.75},
                                   {"bbox", {120 + (i % 15), 100 + (i % 20), 30, 28}}}});

               std::string send_err;
               if (group.send_metadata(static_cast<size_t>(s), "object-detection", data.dump(),
                                       1000 + i, std::to_string(i), &send_err)) {
                 ++emitted;
               } else {
                 ++emit_fail;
               }
             }
           }

           std::vector<std::string> packets0;
           std::vector<std::string> packets1;
           const int expected_per_stream = iters;
           const int got0 = rx0.drain(&packets0, expected_per_stream, 120);
           const int got1 = rx1.drain(&packets1, expected_per_stream, 120);

           int parseable = 0;
           for (const auto& payload : packets0) {
             if (is_parseable_metadata_receiver_json(payload))
               ++parseable;
           }
           for (const auto& payload : packets1) {
             if (is_parseable_metadata_receiver_json(payload))
               ++parseable;
           }

           const int total_expected = streams * iters;
           const int total_received = got0 + got1;
           const int dropped = total_expected - total_received;

           require(emit_fail == 0, "UDP JSON burst stress should not fail send_metadata calls");
           require(emitted == total_expected, "UDP JSON burst stress emitted count mismatch");
           require(total_received > 0,
                   "UDP JSON burst stress should receive at least one datagram");
           require(parseable == total_received,
                   "UDP JSON burst stress should produce parseable metadata datagrams");
           require(dropped <= (total_expected / 2),
                   "UDP JSON burst stress drop count exceeded bounded threshold");

           group.stop();
         }));
