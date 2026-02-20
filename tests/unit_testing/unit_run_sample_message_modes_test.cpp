#include "runtime_test_utils.h"
#include "test_main.h"

RUN_TEST("unit_run_sample_message_modes_test", [] {
  using namespace simaai::neat;

  const Tensor seed = make_color_tensor(48, 32, ImageSpec::PixelFormat::RGB, 0x55);
  Run run = sima_test::make_async_rgb_run(seed, 16, 16);

  Sample copy_msg;
  copy_msg.kind = SampleKind::Tensor;
  copy_msg.tensor = seed;
  copy_msg.media_type = "video/x-raw";
  copy_msg.format = "RGB";
  require(run.try_push(copy_msg), "Run::try_push(Sample copy) should succeed");
  auto copy_out = run.pull(1000);
  require(copy_out.has_value(), "Run::try_push(Sample copy) should produce output");
  require(copy_out->tensor.has_value(), "Run::try_push(Sample copy) output missing tensor");

  Sample first = run.push_and_pull(seed, 1000);
  require(first.tensor.has_value(), "holder mode seed output missing tensor");
  require(first.tensor->storage != nullptr, "holder mode seed tensor missing storage");
  require(first.tensor->storage->holder != nullptr, "holder mode seed tensor missing holder");

  require(run.try_push(first), "Run::try_push(Sample holder) should succeed");
  auto holder_out = run.pull(1000);
  require(holder_out.has_value(), "Run::try_push(Sample holder) should produce output");
  require(holder_out->tensor.has_value(), "Run::try_push(Sample holder) output missing tensor");

  run.stop();
});
