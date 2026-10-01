#include "v4l2_copy_mock.h"
#include <algorithm>
#include <iostream>

namespace cc = simaai::neat::camera_copy;
template <class F> void rejects(F action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  copy_check(rejected, "expected rejection");
}
int main() {
  try {
    std::atomic<bool> cancel{false};
    const cc::Format request{8, 4, cc::parse_fourcc("BA81"), 0, 0};
    for (const auto* fourcc : {"GREY", "BA81", "GBRG", "GRBG", "RGGB"})
      copy_check(cc::parse_fourcc(fourcc) != 0, "valid fourcc");
    rejects([] { cc::parse_fourcc("NV12"); });
    rejects([] { cc::parse_fourcc("RGB"); });
    auto mock = std::make_shared<CopyCameraMock>();
    cc::Capture capture(mock);
    capture.start("/dev/mock", request, 4);
    copy_check(capture.format().stride == 10 && capture.format().size == 48,
               "negotiated padding/trailer");
    std::vector<std::uint8_t> out(48);
    mock->retry_dequeue = true;
    const auto frame = capture.next(out, cancel, 1000);
    copy_check(frame.bytes == 48 && frame.sequence == 0, "full payload");
    copy_check(std::all_of(out.begin(), out.end(), [](auto b) { return b == 1; }),
               "copy includes padding/trailer before requeue");
    copy_check(mock->buffers[0][0] == 5 && out[0] == 1, "camera reuse corrupted owned output");
    const auto retained = out;
    capture.next(out, cancel, 1000);
    copy_check(retained[0] == 1 && out[0] == 5, "retained output changed");
    mock->bytes = 38; // Final row padding may be absent.
    copy_check(capture.next(out, cancel, 1000).bytes == 38, "valid last row extent");
    for (auto bad : {0U, 37U, 49U}) {
      mock->bytes = bad;
      const auto before = mock->queues;
      rejects([&] { capture.next(out, cancel, 1000); });
      copy_check(mock->queues == before + 1, "invalid frame not requeued");
    }
    mock->bytes = 48;
    mock->flags = V4L2_BUF_FLAG_ERROR;
    rejects([&] { capture.next(out, cancel, 1000); });
    mock->flags = 0;
    mock->index = 200;
    rejects([&] { capture.next(out, cancel, 1000); });
    mock->index = 0;
    rejects([&] { capture.next(std::span(out).first(47), cancel, 1000); });
    mock->poll_fail = true;
    rejects([&] { capture.next(out, cancel, 1000); });
    mock->poll_fail = false;
    mock->cancel = &cancel;
    copy_check(capture.next(out, cancel, 1000).bytes == 0, "cancellation failed");
    copy_check(capture.stop() && capture.stop(), "idempotent stop");
    copy_check(mock->unmaps == 4 && mock->close_count == 1, "cleanup mismatch");
    for (auto failure : {VIDIOC_QUERYCAP, VIDIOC_S_FMT, VIDIOC_REQBUFS, VIDIOC_QUERYBUF,
                         VIDIOC_QBUF, VIDIOC_STREAMON}) {
      auto m = std::make_shared<CopyCameraMock>();
      m->fail_request = failure;
      {
        cc::Capture c(m);
        rejects([&] { c.start("mock", request, 4); });
      }
      copy_check(m->close_count == 1, "failed start leaked fd");
      if (failure == VIDIOC_QBUF || failure == VIDIOC_STREAMON)
        copy_check(std::find(m->calls.begin(), m->calls.end(), VIDIOC_STREAMOFF) != m->calls.end(),
                   "failed start did not retire queue");
    }
    for (int scenario = 0; scenario < 6; ++scenario) {
      auto m = std::make_shared<CopyCameraMock>();
      if (scenario == 0)
        m->platform = true;
      if (scenario == 1)
        m->changed_format = true;
      if (scenario == 2)
        m->stride = 7;
      if (scenario == 3)
        m->size = 39;
      if (scenario == 4)
        m->map_fail = true;
      if (scenario == 5)
        m->count = 129;
      {
        cc::Capture c(m);
        rejects([&] { c.start("mock", request, 4); });
      }
      copy_check(m->close_count == 1 && m->queues == 0, "invalid config reached DMA queue");
      if (scenario == 0)
        copy_check(m->requests == 0, "platform gate ran too late");
    }
    auto parked = std::make_shared<CopyCameraMock>();
    {
      cc::Capture c(parked);
      c.start("mock", request, 4);
      parked->stop_fail = true;
      copy_check(!c.stop(), "failed stop accepted");
      rejects([&] { c.start("mock", request, 4); });
    }
    copy_check(parked->unmaps == 0 && parked->close_count == 0, "unknown DMA retired unsafely");
    std::cout
        << "PASS V4L2 copy capture: payload, ownership, cancellation, validation and lifecycle\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
