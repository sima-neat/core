#pragma once
#include "gst/V4L2CopyCapture.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <linux/videodev2.h>
#include <stdexcept>
#include <sys/mman.h>
#include <vector>

class CopyCameraMock final : public simaai::neat::camera_copy::Backend {
public:
  unsigned open_count = 0, close_count = 0, unmaps = 0, queues = 0, requests = 0;
  std::atomic<unsigned> dequeues{0};
  unsigned stride = 10, height = 4, width = 8, size = 48, bytes = 48, count = 4;
  unsigned fail_request = 0, flags = 0, index = 0;
  bool platform = false, map_fail = false, stop_fail = false, changed_format = false;
  bool poll_fail = false, retry_dequeue = false, short_second_frame = false;
  std::atomic<bool>* cancel = nullptr;
  std::vector<std::vector<unsigned char>> buffers;
  std::vector<unsigned long> calls;
  int open(const std::string&) override {
    ++open_count;
    return 123;
  }
  void close(int) noexcept override {
    ++close_count;
  }
  void* map(int, std::uint32_t offset, std::uint32_t) override {
    return map_fail ? MAP_FAILED : buffers.at(offset).data();
  }
  void unmap(void*, std::uint32_t) noexcept override {
    ++unmaps;
  }
  int wait(int, int) override {
    if (cancel) {
      cancel->store(true);
      return 0;
    }
    if (poll_fail) {
      errno = EIO;
      return -1;
    }
    return 1;
  }
  int io(int, unsigned long request, void* arg) override {
    calls.push_back(request);
    if (request == fail_request) {
      errno = EIO;
      return -1;
    }
    switch (request) {
    case VIDIOC_QUERYCAP: {
      auto& c = *static_cast<v4l2_capability*>(arg);
      c.capabilities = V4L2_CAP_DEVICE_CAPS;
      c.device_caps = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
      std::strcpy(reinterpret_cast<char*>(c.bus_info), platform ? "platform:test" : "usb:test");
      break;
    }
    case VIDIOC_S_FMT: {
      auto& f = static_cast<v4l2_format*>(arg)->fmt.pix;
      f.width = changed_format ? 9 : width;
      f.height = height;
      f.bytesperline = stride;
      f.sizeimage = size;
      break;
    }
    case VIDIOC_REQBUFS: {
      auto& b = *static_cast<v4l2_requestbuffers*>(arg);
      if (b.memory != V4L2_MEMORY_MMAP)
        throw std::runtime_error("not MMAP");
      b.count = count;
      ++requests;
      buffers.assign(count, std::vector<unsigned char>(size, 0xA5));
      break;
    }
    case VIDIOC_QUERYBUF: {
      auto& b = *static_cast<v4l2_buffer*>(arg);
      b.length = size;
      b.m.offset = b.index;
      break;
    }
    case VIDIOC_QBUF: {
      auto& b = *static_cast<v4l2_buffer*>(arg);
      if (b.memory != V4L2_MEMORY_MMAP)
        throw std::runtime_error("not MMAP queue");
      ++queues;
      // Simulate immediate device reuse: the delivered copy must not change.
      if (b.index < buffers.size())
        std::fill(buffers[b.index].begin(), buffers[b.index].end(),
                  static_cast<unsigned char>(queues));
      break;
    }
    case VIDIOC_DQBUF: {
      if (retry_dequeue) {
        retry_dequeue = false;
        errno = EAGAIN;
        return -1;
      }
      auto& b = *static_cast<v4l2_buffer*>(arg);
      b.index = index;
      b.length = size;
      b.bytesused = bytes;
      b.flags = flags;
      b.sequence = dequeues.fetch_add(1);
      if (short_second_frame && b.sequence == 1)
        b.bytesused = 38;
      break;
    }
    case VIDIOC_STREAMOFF:
      if (stop_fail) {
        errno = EIO;
        return -1;
      }
      break;
    }
    return 0;
  }
};
inline void copy_check(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
