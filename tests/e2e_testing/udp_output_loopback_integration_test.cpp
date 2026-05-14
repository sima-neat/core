#include "nodes/io/Input.h"
#include "nodes/io/UdpOutput.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "udp_test_utils.h"

#include <chrono>
#include <string>

RUN_TEST("udp_output_loopback_integration_test", ([] {
           using namespace simaai::neat;

           sima_test::UdpReceiver rx;

           Session session;
           InputOptions src_opt;
           src_opt.media_type = "video/x-raw";
           src_opt.format = simaai::neat::FormatTag::RGB;
           src_opt.use_simaai_pool = false;
           src_opt.max_width = 32;
           src_opt.max_height = 24;
           src_opt.max_depth = 3;
           session.add(nodes::Input(src_opt));

           UdpOutputOptions udp_opt;
           udp_opt.host = "127.0.0.1";
           udp_opt.port = rx.port();
           udp_opt.sync = false;
           udp_opt.async = false;
           session.add(nodes::UdpOutput(udp_opt));

           Run run;
           try {
             const Tensor seed = make_color_tensor(12, 8, ImageSpec::PixelFormat::RGB, 0x44);
             RunOptions run_opt;
             run_opt.queue_depth = 32;
             run = session.build(TensorList{seed}, RunMode::Async, run_opt);
           } catch (const std::exception& e) {
             if (sima_test::likely_runtime_missing(e.what())) {
               throw std::runtime_error(
                   "Skipping UDP loopback integration due runtime limitations: " +
                   std::string(e.what()));
             }
             throw;
           }

           struct RunStopGuard {
             Run* run_ptr = nullptr;
             ~RunStopGuard() {
               if (!run_ptr)
                 return;
               try {
                 run_ptr->stop();
               } catch (...) {
               }
             }
           } guard{&run};

           for (int i = 0; i < 6; ++i) {
             Sample s;
             s.kind = SampleKind::Tensor;
             s.tensor = make_color_tensor(12, 8, ImageSpec::PixelFormat::RGB,
                                          static_cast<uint8_t>(0x50 + i));
             s.frame_id = i;
             s.stream_id = "udp-loopback";
             require(run.push(SampleList{s}), "udp loopback: push failed");
           }

           std::string payload;
           require(rx.recv_one(&payload, 2500), "udp loopback: expected datagram was not received");
           require(!payload.empty(), "udp loopback: received payload should be non-empty");

           run.stop();
           guard.run_ptr = nullptr;
         }));
