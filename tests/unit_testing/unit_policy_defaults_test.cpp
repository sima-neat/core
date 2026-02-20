#include "policy/DefaultPolicy.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_policy_defaults_test", ([] {
           using namespace simaai::neat::policy;

           const DefaultPolicy policy = make_default_policy();

           {
             const auto ok = policy.decoder.validate_resolution(1280, 720);
             require(ok.ok(), "default decoder policy should allow 1280x720");
           }
           {
             const auto bad = policy.decoder.validate_resolution(-1, 720);
             require(!bad.ok(), "decoder policy should reject negative width");
             require_contains(bad.reason, "decoder_policy",
                              "decoder rejection should be actionable");
           }

           {
             const auto ok = policy.encoder.validate_bitrate(4000);
             require(ok.ok(), "default encoder policy should allow moderate bitrate");
           }
           {
             const auto bad = policy.encoder.validate_bitrate(1'000'000);
             require(!bad.ok(), "encoder policy should reject out-of-range bitrate");
             require_contains(bad.reason, "encoder_policy",
                              "encoder rejection should be actionable");
           }

           {
             const auto ok = policy.memory.validate_pool_bytes(1 << 20);
             require(ok.ok(), "default memory policy should allow 1MiB");
           }
           {
             const auto bad = policy.memory.validate_pool_bytes(0);
             require(!bad.ok(), "memory policy should reject zero bytes");
             require_contains(bad.reason, "memory_policy", "memory rejection should be actionable");
           }

           {
             const auto ok = policy.rtsp.validate_port(8554);
             require(ok.ok(), "default RTSP policy should allow 8554");
           }
           {
             const auto bad = policy.rtsp.validate_port(80);
             require(!bad.ok(), "RTSP policy should reject privileged/out-of-range ports");
             require_contains(bad.reason, "rtsp_policy", "rtsp rejection should be actionable");
           }
         }));
