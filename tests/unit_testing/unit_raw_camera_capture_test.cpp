#include "gst/RawCameraCapture.h"
#include "gst/RawCameraMemoryContract.h"

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <functional>
#include <iostream>
#include <linux/videodev2.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace raw = simaai::neat::raw_camera;
namespace {
unsigned checks = 0;
void check(bool condition) {
  ++checks;
  if (!condition)
    throw std::runtime_error("raw camera test assertion failed");
}
void rejected(const std::function<void()>& action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  check(failed);
}
struct Fake : raw::Backend {
  raw::Format format{1920, 360, V4L2_PIX_FMT_SBGGR8, 1920, 691232};
  bool platform = false, fail_stop = false, fail_start = false, fail_queue = false;
  bool fail_req = false;
  std::uint64_t physical = 0x100000ULL, bus = 0x80100000ULL;
  bool enforce_shipping_driver_gate = false;
  unsigned format_calls = 0;
  unsigned queues = 0, closes = 0, allocated = 0, released = 0;
  unsigned queue_attempts = 0, fail_queue_attempt = 0, streamon_calls = 0;
  unsigned long fail_request = 0;
  v4l2_buffer next{};
  std::vector<std::weak_ptr<void>> owners;
  Fake() {
    next.index = 0;
    next.bytesused = 691200;
    next.length = 691232;
    next.sequence = 19;
    next.timestamp.tv_sec = 17;
    next.timestamp.tv_usec = 45;
    next.flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
  }
  int open_device(const std::string&) override {
    return 17;
  }
  int io(int, unsigned long request, void* argument) override {
    if (request == fail_request) {
      errno = EIO;
      return -1;
    }
    if (request == VIDIOC_QUERYCAP) {
      auto* value = static_cast<v4l2_capability*>(argument);
      value->capabilities = V4L2_CAP_DEVICE_CAPS;
      value->device_caps = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
      std::strcpy(reinterpret_cast<char*>(value->bus_info),
                  platform ? "platform:sima-csi" : "usb:test");
      if (enforce_shipping_driver_gate)
        raw::require_qualified_capture_driver(std::string_view(
            reinterpret_cast<const char*>(value->bus_info), sizeof(value->bus_info)));
    } else if (request == VIDIOC_S_FMT) {
      ++format_calls;
      auto& value = static_cast<v4l2_format*>(argument)->fmt.pix;
      value.width = format.width;
      value.height = format.height;
      value.pixelformat = format.fourcc;
      value.bytesperline = format.stride;
      value.sizeimage = format.size;
      value.field = V4L2_FIELD_NONE;
    } else if (request == VIDIOC_REQBUFS) {
      if (fail_req) {
        errno = EINVAL;
        return -1;
      }
    } else if (request == VIDIOC_QBUF) {
      ++queue_attempts;
      if (fail_queue || queue_attempts == fail_queue_attempt) {
        errno = EIO;
        return -1;
      }
      const auto* value = static_cast<v4l2_buffer*>(argument);
      check(value->length == format.size);
      check(value->memory == V4L2_MEMORY_DMABUF);
      ++queues;
    } else if (request == VIDIOC_DQBUF) {
      *static_cast<v4l2_buffer*>(argument) = next;
    } else if (request == VIDIOC_STREAMOFF) {
      if (fail_stop) {
        errno = EIO;
        return -1;
      }
    } else if (request == VIDIOC_STREAMON) {
      ++streamon_calls;
      if (fail_start) {
        errno = EIO;
        return -1;
      }
    }
    return 0;
  }
  int wait(int, int) override {
    return 1;
  }
  void close_device(int) override {
    ++closes;
  }
  raw::Allocation allocate(std::uint32_t size, const std::string&) override {
    ++allocated;
    auto owner = std::shared_ptr<void>(new int(0), [this](void* p) {
      delete static_cast<int*>(p);
      ++released;
    });
    owners.push_back(owner);
    return {100 + static_cast<int>(allocated), physical, bus, size, owner};
  }
};
std::shared_ptr<raw::Capture> begin(std::shared_ptr<Fake> fake) {
  auto capture = std::make_shared<raw::Capture>(fake);
  capture->start("/dev/video-test", "raw_src", {1920, 360, V4L2_PIX_FMT_SBGGR8, 0, 0}, 4);
  return capture;
}
} // namespace
int main() {
  try {
    {
      GstNeatCameraMemoryApiV1 api{
          GST_NEAT_CAMERA_MEMORY_API_VERSION_1,
          sizeof(GstNeatCameraMemoryApiV1),
          GST_NEAT_CAMERA_MEMORY_CAP_DMABUF_EXPORT |
              GST_NEAT_CAMERA_MEMORY_CAP_PACKED_LAYOUT |
              GST_NEAT_CAMERA_MEMORY_CAP_DEVICE_WRITTEN,
          +[] {},
          +[]() -> GstAllocator* { return nullptr; },
          +[](GstSimaaiAllocationParams*) {},
          +[](GstSimaaiAllocationParams*, gsize, const gchar*) -> gboolean { return TRUE; },
          +[](const GstMemory*) -> gboolean { return TRUE; },
          +[](const GstMemory*, const gchar*, guint32) -> gint { return -1; },
          +[](const GstMemory*, const gchar*) -> GstMemory* { return nullptr; },
          +[](GstMemory*) {},
      };
      check(!raw::supports_raw_camera_memory(nullptr));
      // Same v1 table and complete entry points, but old allocator loses owners.
      check(!raw::supports_raw_camera_memory(&api));
      api.capabilities |= GST_NEAT_CAMERA_MEMORY_CAP_SHARED_OWNER_RETENTION;
      check(raw::supports_raw_camera_memory(&api));
      api.capabilities |= G_GUINT64_CONSTANT(1) << 63;
      check(raw::supports_raw_camera_memory(&api)); // Future additive flags are fine.
      api.struct_size = sizeof(api) - 1;
      check(!raw::supports_raw_camera_memory(&api));
      api.struct_size = sizeof(api);
      api.abi_version = 2;
      check(!raw::supports_raw_camera_memory(&api));
      api.abi_version = GST_NEAT_CAMERA_MEMORY_API_VERSION_1;
      api.share_packed = nullptr;
      check(!raw::supports_raw_camera_memory(&api));
    }
    raw::require_qualified_capture_driver("usb:test");
    rejected([] { raw::require_qualified_capture_driver("platform:sima-csi"); });
    {
      auto fake = std::make_shared<Fake>();
      fake->platform = true;
      fake->enforce_shipping_driver_gate = true;
      auto capture = std::make_shared<raw::Capture>(fake);
      rejected([&] {
        capture->start("/dev/video-test", "raw_src", {1920, 360, V4L2_PIX_FMT_SBGGR8, 0, 0}, 4);
      });
      check(fake->format_calls == 0 && fake->allocated == 0);
      check(fake->queue_attempts == 0 && fake->streamon_calls == 0);
      check(capture->stop());
      capture.reset();
      check(fake->closes == 1 && fake->released == 0);
    }
    std::atomic<std::uint64_t> generations{0};
    check(raw::next_capture_generation(generations) == 1);
    check(raw::next_capture_generation(generations) == 2);
    generations = UINT64_MAX;
    rejected([&] { raw::next_capture_generation(generations); });
    check(generations == UINT64_MAX);
    check(raw::parse_fourcc("BA81") == V4L2_PIX_FMT_SBGGR8);
    rejected([] { raw::parse_fourcc("BA8"); });
    rejected([] { raw::parse_fourcc("B\n81"); });
    const raw::Format requested{1920, 360, V4L2_PIX_FMT_SBGGR8, 0, 0};
    raw::Format padded{1920, 360, V4L2_PIX_FMT_SBGGR8, 2048, 737312};
    raw::validate_format(requested, padded);
    raw::validate_frame(padded, {0, 737152, 0, 0}, 4); // final-row padding omitted
    raw::validate_frame(padded, {0, 737280, 0, 0}, 4);
    rejected([&] { raw::validate_frame(padded, {0, 737151, 0, 0}, 4); });
    rejected([&] { raw::validate_frame(padded, {0, 737312, 0, 0}, 4); });
    rejected([&] { raw::validate_frame(padded, {4, 737280, 0, 0}, 4); });
    auto bad = padded;
    bad.size = 100;
    rejected([&] { raw::validate_format(requested, bad); });
    bad = padded;
    bad.width = 1919;
    rejected([&] { raw::validate_format(requested, bad); });
    bad = padded;
    bad.fourcc = V4L2_PIX_FMT_SRGGB8;
    rejected([&] { raw::validate_format(requested, bad); });
    bad = padded;
    bad.stride = 1919;
    rejected([&] { raw::validate_format(requested, bad); });
    bad = padded;
    bad.fourcc = V4L2_PIX_FMT_SBGGR16;
    rejected([&] { raw::validate_format(bad, bad); });

    std::atomic<bool> interrupted{false};
    {
      auto fake = std::make_shared<Fake>();
      auto capture = begin(fake);
      check(fake->queues == 4 && fake->allocated == 4);
      auto lease = capture->next(interrupted);
      check(lease->frame.timestamp_ns == 17000045000ULL);
      check(lease->frame.bytes == 691200);
      check(lease->frame.discontinuity);
      const auto generation = lease->frame.capture_generation;
      check(generation != 0);
      check(capture->allocation(0).size == 691232); // trailer allocated, never published
      auto copy = lease;
      lease.reset();
      check(fake->queues == 4);
      copy.reset();
      check(fake->queues == 5);
      fake->next.sequence += 20; // dropped frames are not a capture-session restart
      auto held = capture->next(interrupted);
      check(!held->frame.discontinuity);
      check(held->frame.capture_generation == generation);
      check(capture->stop());
      capture.reset();
      check(fake->released == 0 && fake->closes == 0); // consumer still owns capture
      held.reset();
      check(fake->released == 4 && fake->closes == 1 && fake->queues == 5);
    }
    {
      auto fake = std::make_shared<Fake>();
      auto capture = begin(fake);
      interrupted = true;
      check(!capture->next(interrupted));
      interrupted = false;
      auto first = capture->next(interrupted);
      check(first->frame.discontinuity); // cancellation did not consume the boundary
      const auto previous_generation = first->frame.capture_generation;
      first.reset();
      check(capture->stop());
      capture.reset();
      auto restarted = begin(fake);
      auto fresh = restarted->next(interrupted);
      check(fresh->frame.discontinuity);
      check(fresh->frame.capture_generation > previous_generation);
      check(restarted->stop());
    }
    for (int error = 0; error < 6; ++error) {
      auto fake = std::make_shared<Fake>();
      auto capture = begin(fake);
      if (error == 0)
        fake->next.bytesused = 691232;
      if (error == 1)
        fake->next.index = 4;
      if (error == 2)
        fake->next.flags |= V4L2_BUF_FLAG_ERROR;
      if (error == 3)
        fake->next.timestamp.tv_usec = 1000000;
      if (error == 4)
        fake->next.flags = 0;
      if (error == 5)
        fake->next.length = 691233;
      rejected([&] { capture->next(interrupted); });
      check(capture->stop());
    }
    {
      auto fake = std::make_shared<Fake>();
      auto capture = begin(fake);
      auto lease = capture->next(interrupted);
      rejected([&] { capture->next(interrupted); });
      check(capture->stop());
    }
    {
      auto fake = std::make_shared<Fake>();
      fake->physical = 0x1000700000ULL;
      fake->bus = 0x80700000ULL;
      auto capture = begin(fake);
      check(capture->allocation(0).physical == 0x1000700000ULL);
      check(capture->allocation(0).bus == 0x80700000ULL);
      check(capture->stop());
      capture.reset();
      check(fake->released == fake->allocated && fake->closes == 1);
    }
    for (const std::uint64_t bus : {0ULL, 0x100000000ULL, 0xfffffff0ULL}) {
      auto fake = std::make_shared<Fake>();
      fake->physical = 0x1000700000ULL;
      fake->bus = bus;
      rejected([&] { begin(fake); });
      check(fake->queue_attempts == 0 && fake->streamon_calls == 0);
      check(fake->released == fake->allocated && fake->closes == 1);
    }
    for (int error = 0; error < 4; ++error) {
      auto fake = std::make_shared<Fake>();
      fake->fail_req = error == 0;
      fake->bus = error == 1 ? (1ULL << 32) : 0x80100000ULL;
      fake->format.width = error == 2 ? 1919 : 1920;
      fake->fail_queue = error == 3;
      rejected([&] { begin(fake); });
      check(fake->closes == 1);
      check(fake->allocated == fake->released);
    }
    {
      auto fake = std::make_shared<Fake>();
      fake->fail_start = true;
      auto capture = std::make_shared<raw::Capture>(fake);
      rejected([&] { capture->start("/dev/video-test", "raw_src", requested, 4); });
      check(!capture->stop());
      capture.reset();
      check(fake->released == 0 && fake->closes == 0);
    }
    // A platform QBUF can submit DMA before STREAMON. Fail the first and the
    // second attempt: neither failed ioctl nor !streaming proves safe cleanup.
    for (unsigned failed_attempt : {1U, 2U}) {
      auto fake = std::make_shared<Fake>();
      fake->platform = true;
      fake->fail_queue_attempt = failed_attempt;
      auto capture = std::make_shared<raw::Capture>(fake);
      rejected([&] { capture->start("/dev/video-test", "raw_src", requested, 4); });
      check(fake->queue_attempts == failed_attempt);
      check(fake->queues == failed_attempt - 1 && fake->streamon_calls == 0);
      check(!capture->stop());
      std::weak_ptr<raw::Capture> parked = capture;
      capture.reset();
      check(!parked.expired());
      check(fake->released == 0 && fake->closes == 0);
      for (const auto& owner : fake->owners)
        check(!owner.expired());
    }
    // Generic non-platform vb2 cleanup keeps its established failure behavior.
    {
      auto fake = std::make_shared<Fake>();
      fake->fail_queue_attempt = 2;
      auto capture = std::make_shared<raw::Capture>(fake);
      rejected([&] { capture->start("/dev/video-test", "raw_src", requested, 4); });
      check(capture->stop());
      capture.reset();
      check(fake->released == fake->allocated && fake->closes == 1);
    }
    // Unknown retirement intentionally retains all DMA resources, even after
    // every outside owner drops its reference. Tests never touch a device.
    for (int reason = 0; reason < 3; ++reason) {
      auto fake = std::make_shared<Fake>();
      fake->platform = reason == 0;
      fake->fail_stop = reason == 1;
      auto capture = begin(fake);
      if (reason == 2) {
        auto lease = capture->next(interrupted);
        fake->fail_queue = true;
        lease.reset();
      }
      check(!capture->stop());
      std::weak_ptr<raw::Capture> parked = capture;
      capture.reset();
      check(!parked.expired() && fake->released == 0 && fake->closes == 0);
    }
    std::cout << "PASS raw camera capture " << checks << " checks\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
