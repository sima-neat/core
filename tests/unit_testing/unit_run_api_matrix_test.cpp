#include "pipeline/ErrorCodes.h"
#include "pipeline/Session.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <string>

namespace {

std::string run_api_case(const char* id, const std::string& msg) {
  return std::string("RUN_API_CASE=") + id + " " + msg;
}

simaai::neat::Run make_async_rgb_run_with_copy_input(const simaai::neat::Tensor& seed,
                                                      int queue_depth = 1) {
  using namespace simaai::neat;

  Session session;
  InputOptions src_opt;
  src_opt.media_type = "video/x-raw";
  src_opt.format = simaai::neat::FormatTag::RGB;
  src_opt.use_simaai_pool = false;
  src_opt.max_width = 96;
  src_opt.max_height = 96;
  src_opt.max_depth = 3;
  session.add(nodes::Input(src_opt));
  session.add(nodes::Output(OutputOptions::EveryFrame(128)));

  RunOptions run_opt;
  run_opt.queue_depth = queue_depth;
  run_opt.overflow_policy = OverflowPolicy::Block;
  run_opt.advanced.copy_input = true;
  return session.build(TensorList{seed}, RunMode::Async, run_opt);
}

} // namespace

RUN_TEST("unit_run_api_matrix_test", ([] {
           using namespace simaai::neat;

           const Tensor seed = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x58);

           // Closed/default behavior.
           {
             Run closed;
             require(sima_test::throws_with([&]() { (void)closed.try_push(TensorList{seed}); }, "stream is closed"),
                     run_api_case("closed_try_push_tensor",
                                  "Run::try_push should reject default/closed run"));
             require(sima_test::throws_with([&]() { (void)closed.try_push_holder({}); },
                                            "stream is closed"),
                     run_api_case("closed_try_push_holder",
                                  "Run::try_push_holder should reject default/closed run"));

             Sample tmp;
             PullError err;
             const PullStatus status = closed.pull(10, tmp, &err);
             require(status == PullStatus::Closed,
                     run_api_case("closed_pull_status", "Run::pull closed status mismatch"));
             require(err.code == error_codes::kRuntimePull,
                     run_api_case("closed_pull_error_code",
                                  "Run::pull closed status should set runtime.pull code"));
             require_contains(err.message, "closed",
                              run_api_case("closed_pull_error_msg",
                                           "Run::pull closed message should include context"));
             TensorList closed_tensors = closed.pull_tensors(10);
             require(closed_tensors.empty(),
                     run_api_case("closed_pull_tensors", "Run::pull_tensors should return empty on closed run"));
           }

           // Active async matrix.
           {
             Run run = sima_test::make_async_rgb_run(seed, 32, 32);

             auto no_output = run.pull_tensors(120);
             require(no_output.empty(),
                     run_api_case("active_pull_tensors_timeout",
                                  "pull_tensors should timeout on empty output queue"));

             Sample tmp;
             PullError err;
             const PullStatus timeout_status = run.pull(120, tmp, &err);
             require(timeout_status == PullStatus::Timeout,
                     run_api_case("active_pull_timeout_status",
                                  "Run::pull timeout status mismatch"));
             require(err.code == error_codes::kRuntimePull,
                     run_api_case("active_pull_timeout_error_code",
                                  "Run::pull timeout should set runtime.pull code"));
             require_contains(err.message, "runtime.pull",
                              run_api_case("active_pull_timeout_error_msg",
                                           "Run::pull timeout should include runtime.pull code"));
             require(run.pull_tensors(120).empty(),
                     run_api_case("active_pull_tensors_timeout_repeat",
                                  "Run::pull_tensors should stay empty on timeout"));

             require(run.push(TensorList{seed}),
                     run_api_case("active_push_tensor", "Run::push(Tensor) should succeed"));
             auto matched = run.pull_tensors(1000);
             require(matched.size() == 1,
                     run_api_case("active_pull_tensors_hit",
                                  "pull_tensors should return one RGB payload"));

             require(run.push(TensorList{seed}),
                     run_api_case("active_push_tensor_for_miss",
                                  "Run::push(Tensor) should succeed before tag miss"));
             auto miss = run.pull_tensors(1000);
             require(miss.size() == 1,
                     run_api_case("active_pull_tensors_second_hit",
                                  "pull_tensors should return one tensor again"));

             {
               const TensorList single_input = {seed};
               require(run.push(single_input),
                       run_api_case("active_push_tensor_list",
                                    "Run::push(TensorList) should succeed"));
               TensorList list_out = run.pull_tensors(1000);
               require(list_out.size() == 1,
                       run_api_case("active_pull_tensors",
                                    "Run::pull_tensors should return one tensor for single-output run"));
               require(list_out.front().shape == seed.shape,
                       run_api_case("active_pull_tensors_shape",
                                    "Run::pull_tensors should preserve tensor shape"));

               TensorList roundtrip = run.run(single_input, 1000);
               require(roundtrip.size() == 1,
                       run_api_case("active_run_tensor_list",
                                    "Run::run(TensorList) should return one tensor"));
               require(roundtrip.front().dtype == seed.dtype,
                       run_api_case("active_run_tensor_list_dtype",
                                    "Run::run(TensorList) should preserve tensor dtype"));
             }

             Sample copy_msg;
             copy_msg.kind = SampleKind::Tensor;
             copy_msg.tensor = seed;
             copy_msg.media_type = "video/x-raw";
             copy_msg.format = "RGB";
             try {
               require(run.try_push(SampleList{copy_msg}),
                       run_api_case("active_try_push_sample_copy",
                                    "Run::try_push(Sample copy) should succeed"));
             } catch (const std::exception& e) {
               throw std::runtime_error(run_api_case(
                   "active_try_push_sample_copy_throw", std::string("unexpected throw: ") + e.what()));
             }
             auto copy_out = run.pull(1000);
             require(copy_out.has_value() && !tensors_from_sample(*copy_out, true).empty(),
                     run_api_case("active_try_push_sample_copy_output",
                                  "Run::try_push(Sample copy) should produce tensor output"));

             const auto fill = sima_test::fill_try_push_queue_non_blocking(run, seed, 4096);
             require(fill.saw_backpressure,
                     run_api_case("active_backpressure_signal",
                                  "Run::try_push should eventually report backpressure"));
             require(fill.max_call_ms < 50,
                     run_api_case("active_backpressure_non_blocking",
                                  "Run::try_push should remain non-blocking under pressure"));

             run.close_input();
             require(!run.try_push(TensorList{seed}),
                     run_api_case("after_close_input_try_push",
                                  "Run::try_push should fail after close_input"));

             run.stop();
             require(!run.try_push(TensorList{seed}),
                     run_api_case("after_stop_try_push", "Run::try_push should fail after stop"));
           }

           // Holder-specific matrix on a fresh run.
           {
             Run holder_run = make_async_rgb_run_with_copy_input(seed, 1);

             require(holder_run.push(TensorList{seed}),
                     run_api_case("active_holder_seed_push",
                                  "Run::push(TensorList) should seed holder path"));
             auto first_samples = holder_run.pull_samples(1000);
             require(first_samples.size() == 1,
                     run_api_case("active_holder_seed_pull",
                                  "Run::pull_samples should return one sample"));
             require(holder_run.last_error().empty(),
                     run_api_case("active_holder_seed_pull_last_error",
                                  "Run::pull_samples drain poll should not leave a timeout error"));
             Sample drain_probe;
             PullError drain_err;
             const PullStatus drain_status = holder_run.pull(0, drain_probe, &drain_err);
             require(drain_status == PullStatus::Timeout,
                     run_api_case("active_holder_zero_timeout_poll_status",
                                  "Run::pull(0) should report timeout when no extra sample is queued"));
             require(drain_err.code.empty() && drain_err.message.empty(),
                     run_api_case("active_holder_zero_timeout_poll_error",
                                  "Run::pull(0) should behave like a benign non-blocking poll"));
             const Sample first = first_samples.front();
             const TensorList first_tensors = tensors_from_sample(first, true);
             require(!first_tensors.empty() && first_tensors.front().storage != nullptr &&
                         first_tensors.front().storage->holder != nullptr,
                     run_api_case("active_holder_seed",
                                  "TensorList seed path should produce a holder-backed tensor"));

             const std::shared_ptr<void> holder = first_tensors.front().storage->holder;
             try {
               require(holder_run.try_push_holder(holder),
                       run_api_case("active_try_push_holder",
                                    "Run::try_push_holder should accept valid holder"));
             } catch (const std::exception& e) {
               throw std::runtime_error(run_api_case(
                   "active_try_push_holder_throw", std::string("unexpected throw: ") + e.what()));
             }
             auto pushed_holder_out = holder_run.pull(1000);
             require(pushed_holder_out.has_value() &&
                         !tensors_from_sample(*pushed_holder_out, true).empty(),
                     run_api_case("active_try_push_holder_output",
                                  "Run::try_push_holder should produce output"));

             require(holder_run.push_holder(holder),
                     run_api_case("active_push_holder",
                                  "Run::push_holder should accept valid holder"));
             auto holder_roundtrip = holder_run.pull_samples(1000);
             require(holder_roundtrip.size() == 1 &&
                         !tensors_from_sample(holder_roundtrip.front(), true).empty(),
                     run_api_case("active_push_holder_output",
                                  "Run::push_holder should produce output"));

             holder_run.stop();
             require(!holder_run.try_push_holder(holder),
                     run_api_case("holder_stale_after_stop",
                                  "Run::try_push_holder should fail after stop"));
           }

           // Sample(holder) path on its own run to avoid holder-state coupling.
           {
             Run holder_sample_run = make_async_rgb_run_with_copy_input(seed, 1);
             require(holder_sample_run.push(TensorList{seed}),
                     run_api_case("active_holder_sample_seed_push",
                                  "Run::push(TensorList) should seed holder sample path"));
             auto first_samples = holder_sample_run.pull_samples(1000);
             require(first_samples.size() == 1,
                     run_api_case("active_holder_sample_seed_pull",
                                  "Run::pull_samples should return one sample"));
             const Sample first = first_samples.front();
             const TensorList first_tensors = tensors_from_sample(first, true);
             require(!first_tensors.empty() && first_tensors.front().storage != nullptr &&
                         first_tensors.front().storage->holder != nullptr,
                     run_api_case("active_holder_sample_seed",
                                  "holder sample path should produce holder-backed tensor"));

             try {
               require(holder_sample_run.try_push(SampleList{first}),
                       run_api_case("active_try_push_sample_holder",
                                    "Run::try_push(Sample holder) should succeed"));
             } catch (const std::exception& e) {
               throw std::runtime_error(run_api_case(
                   "active_try_push_sample_holder_throw",
                   std::string("unexpected throw: ") + e.what()));
             }
             auto holder_sample_out = holder_sample_run.pull(1000);
             require(holder_sample_out.has_value() &&
                         !tensors_from_sample(*holder_sample_out, true).empty(),
                     run_api_case("active_try_push_sample_holder_output",
                                  "Run::try_push(Sample holder) should produce output"));

             require(
                 sima_test::throws_with([&]() { (void)holder_sample_run.try_push_holder({}); },
                                        "missing holder"),
                 run_api_case("active_null_holder",
                              "Run::try_push_holder should reject null holder"));
             holder_sample_run.stop();
           }

           // Sync contract: run() only, no push/pull.
           {
             Session session;
             InputOptions src_opt;
             src_opt.media_type = "application/vnd.simaai.tensor";
             src_opt.format = simaai::neat::FormatTag::RGB;
             session.add(nodes::Input(src_opt));
             session.add(nodes::Output(OutputOptions::EveryFrame(16)));

             Run sync_run = session.build(TensorList{seed}, RunMode::Sync);
             require(sima_test::throws_with(
                         [&]() { (void)sync_run.push(TensorList{seed}); },
                         "not allowed in sync mode"),
                     run_api_case("sync_push_tensor_throws",
                                  "Run::push should throw in sync mode"));
             require(sima_test::throws_with(
                         [&]() { (void)sync_run.try_push(TensorList{seed}); },
                         "not allowed in sync mode"),
                     run_api_case("sync_try_push_tensor_throws",
                                  "Run::try_push should throw in sync mode"));
             require(sima_test::throws_with(
                         [&]() { (void)sync_run.pull_tensors(10); },
                         "not allowed in sync mode"),
                     run_api_case("sync_pull_tensors_throws",
                                  "Run::pull_tensors should throw in sync mode"));
             require(sima_test::throws_with(
                         [&]() { (void)sync_run.pull_samples(10); },
                         "not allowed in sync mode"),
                     run_api_case("sync_pull_samples_throws",
                                  "Run::pull_samples should throw in sync mode"));

             TensorList sync_out = sync_run.run(TensorList{seed}, 1000);
             require(sync_out.size() == 1,
                     run_api_case("sync_run_tensor_list",
                                  "Run::run(TensorList) should remain the sync path"));
           }

           // Format/depth mismatch path.
           {
             Session session;
             InputOptions src_opt;
             src_opt.media_type = "video/x-raw";
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.use_simaai_pool = false;
             src_opt.max_width = 96;
             src_opt.max_height = 96;
             src_opt.max_depth = 3;
             session.add(nodes::Input(src_opt));
             session.add(nodes::Output(OutputOptions::EveryFrame(32)));

             RunOptions run_opt;
             run_opt.queue_depth = 8;

             cv::Mat rgb_seed(48, 64, CV_8UC3, cv::Scalar(20, 40, 60));
             Run run = session.build(std::vector<cv::Mat>{rgb_seed}, RunMode::Async, run_opt);

             bool threw = false;
             std::string msg;
             try {
               cv::Mat gray(48, 64, CV_8UC1, cv::Scalar(70));
               (void)run.run(std::vector<cv::Mat>{gray}, 300);
             } catch (const std::exception& e) {
               threw = true;
               msg = e.what();
             }
             run.stop();

             require(threw, run_api_case("format_depth_mismatch_throw",
                                         "Run should throw for incompatible frame depth"));
             require(!msg.empty(),
                     run_api_case("format_depth_mismatch_error_msg",
                                  "Run should report non-empty mismatch diagnostics"));
           }

           // Empty input build path.
           {
             Session session;
             InputOptions src_opt;
             src_opt.media_type = "video/x-raw";
             src_opt.format = simaai::neat::FormatTag::RGB;
             src_opt.use_simaai_pool = false;
             src_opt.max_width = 96;
             src_opt.max_height = 96;
             src_opt.max_depth = 3;
             session.add(nodes::Input(src_opt));
             session.add(nodes::Output(OutputOptions::EveryFrame(32)));

             RunOptions run_opt;
             run_opt.queue_depth = 8;

             require(sima_test::throws_with(
                         [&]() {
                           cv::Mat empty;
                           (void)session.build(std::vector<cv::Mat>{empty}, RunMode::Async, run_opt);
                         },
                         "empty image input at index 0"),
                     run_api_case("empty_input_build",
                                  "Session::build should reject empty cv::Mat input"));
           }
         }));
