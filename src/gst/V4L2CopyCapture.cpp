#include "gst/V4L2CopyCapture.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace simaai::neat::camera_copy {
namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
class LinuxBackend final : public Backend {
public:
  int open(const std::string& device) override {
    const int fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) < 0) {
      ::close(fd);
      return -1;
    }
    return fd;
  }
  int io(int fd, unsigned long request, void* arg) override {
    int result;
    do {
      result = ::ioctl(fd, request, arg);
    } while (result < 0 && errno == EINTR);
    return result;
  }
  void* map(int fd, std::uint32_t offset, std::uint32_t length) override {
    return ::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
  }
  void unmap(void* address, std::uint32_t length) noexcept override {
    ::munmap(address, length);
  }
  void close(int fd) noexcept override {
    ::close(fd);
  }
  int wait(int fd, int timeout_ms) override {
    pollfd event{fd, POLLIN, 0};
    const int result = ::poll(&event, 1, timeout_ms);
    if (result > 0 && (event.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      errno = EIO;
      return -1;
    }
    return result > 0 && !(event.revents & POLLIN) ? 0 : result;
  }
};
} // namespace
std::shared_ptr<Backend> linux_backend() {
  return std::make_shared<LinuxBackend>();
}
std::uint32_t parse_fourcc(const std::string& text) {
  require(text.size() == 4, "CameraInput V4L2 fourcc must have four bytes");
  std::uint32_t code = 0;
  for (std::size_t i = 0; i < 4; ++i)
    code |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[i])) << (8 * i);
  require(code == V4L2_PIX_FMT_GREY || code == V4L2_PIX_FMT_SBGGR8 || code == V4L2_PIX_FMT_SGBRG8 ||
              code == V4L2_PIX_FMT_SGRBG8 || code == V4L2_PIX_FMT_SRGGB8,
          "CameraInput V4L2 requires GREY, BA81, GBRG, GRBG or RGGB eight-bit wire format");
  return code;
}
Capture::Capture(std::shared_ptr<Backend> backend) : backend_(std::move(backend)) {
  require(backend_ != nullptr, "CameraInput V4L2 backend is missing");
}
Capture::~Capture() {
  stop();
}
void Capture::queue(std::uint32_t index) {
  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  buffer.index = index;
  // A driver's queue callback may submit DMA even before STREAMON.
  queue_attempted_ = true;
  require(backend_->io(fd_, VIDIOC_QBUF, &buffer) == 0, "CameraInput V4L2 QBUF failed");
}
void Capture::prepare(const std::string& device, Format requested, std::uint32_t count) {
  require(state_ == State::Closed, "CameraInput V4L2 session already used or parked");
  require(count >= 4 && count <= 128, "CameraInput V4L2 capture count must be in [4,128]");
  require(requested.width && requested.height, "CameraInput V4L2 dimensions must be positive");
  const char bytes[] = {
      static_cast<char>(requested.fourcc), static_cast<char>(requested.fourcc >> 8),
      static_cast<char>(requested.fourcc >> 16), static_cast<char>(requested.fourcc >> 24)};
  parse_fourcc(std::string(bytes, 4));
  fd_ = backend_->open(device);
  require(fd_ >= 0, "CameraInput cannot open/lock V4L2 device");
  try {
    v4l2_capability caps{};
    require(backend_->io(fd_, VIDIOC_QUERYCAP, &caps) == 0, "CameraInput V4L2 QUERYCAP failed");
    // Bus topology is not a buffer-ownership capability. Platform and USB
    // capture use the same MMAP/STREAMOFF contract; downstream owns only copies.
    const auto flags =
        (caps.capabilities & V4L2_CAP_DEVICE_CAPS) ? caps.device_caps : caps.capabilities;
    require((flags & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING)) ==
                (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING),
            "CameraInput requires single-plane V4L2 streaming capture");
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = requested.width;
    format.fmt.pix.height = requested.height;
    format.fmt.pix.pixelformat = requested.fourcc;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    require(backend_->io(fd_, VIDIOC_S_FMT, &format) == 0, "CameraInput V4L2 S_FMT failed");
    const auto& pix = format.fmt.pix;
    require(pix.width == requested.width && pix.height == requested.height &&
                pix.pixelformat == requested.fourcc && pix.field == V4L2_FIELD_NONE,
            "CameraInput V4L2 changed the requested geometry, format or progressive field");
    require(pix.bytesperline >= pix.width &&
                static_cast<std::uint64_t>(pix.bytesperline) * pix.height <= pix.sizeimage,
            "CameraInput V4L2 invalid row stride or sizeimage");
    format_ = {pix.width, pix.height, pix.pixelformat, pix.bytesperline, pix.sizeimage};
    v4l2_requestbuffers request{};
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    request.count = count;
    require(backend_->io(fd_, VIDIOC_REQBUFS, &request) == 0,
            "CameraInput requires V4L2 MMAP buffers");
    buffers_allocated_ = true;
    require(request.count >= count && request.count <= 128,
            "CameraInput V4L2 invalid buffer count");
    mappings_.reserve(request.count);
    for (std::uint32_t i = 0; i < request.count; ++i) {
      v4l2_buffer buffer{};
      buffer.type = request.type;
      buffer.memory = request.memory;
      buffer.index = i;
      require(backend_->io(fd_, VIDIOC_QUERYBUF, &buffer) == 0, "CameraInput V4L2 QUERYBUF failed");
      require(buffer.length >= format_.size, "CameraInput V4L2 mapping smaller than sizeimage");
      void* address = backend_->map(fd_, buffer.m.offset, buffer.length);
      require(address != MAP_FAILED, "CameraInput V4L2 mmap failed");
      mappings_.push_back({address, buffer.length});
    }
    state_ = State::Prepared;
  } catch (...) {
    stop();
    throw;
  }
}
void Capture::start() {
  require(state_ == State::Prepared, "CameraInput V4L2 session is not prepared");
  try {
    for (std::uint32_t i = 0; i < mappings_.size(); ++i)
      queue(i);
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    require(backend_->io(fd_, VIDIOC_STREAMON, &type) == 0, "CameraInput V4L2 STREAMON failed");
    state_ = State::Streaming;
  } catch (...) {
    stop();
    throw;
  }
}
Frame Capture::next(std::span<std::uint8_t> destination, const std::atomic<bool>& interrupted,
                    std::uint32_t timeout_ms) {
  require(state_ == State::Streaming, "CameraInput V4L2 capture is not streaming");
  require(destination.size() >= format_.size,
          "CameraInput copy destination smaller than sizeimage");
  require(timeout_ms > 0, "CameraInput V4L2 frame timeout must be positive");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!interrupted.load()) {
    require(std::chrono::steady_clock::now() < deadline, "CameraInput V4L2 frame timeout");
    const int ready = backend_->wait(fd_, 50);
    if (ready == 0 || (ready < 0 && errno == EINTR))
      continue;
    if (interrupted.load())
      return {};
    require(ready > 0, "CameraInput V4L2 poll failed");
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (backend_->io(fd_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN || errno == EINTR)
        continue;
      throw std::runtime_error("CameraInput V4L2 DQBUF failed");
    }
    require(buffer.index < mappings_.size(), "CameraInput V4L2 invalid dequeued index");
    const auto& mapping = mappings_[buffer.index];
    const auto minimum =
        static_cast<std::uint64_t>(format_.height - 1) * format_.stride + format_.width;
    const bool valid = buffer.bytesused >= minimum && buffer.bytesused <= format_.size &&
                       buffer.length >= buffer.bytesused && buffer.length <= mapping.size &&
                       !(buffer.flags & V4L2_BUF_FLAG_ERROR);
    if (valid)
      std::memcpy(destination.data(), mapping.address, buffer.bytesused);
    // Requeue before handing the copied tensor to any consumer. Invalid frames
    // are returned to the driver too, but never emitted as usable data.
    queue(buffer.index);
    require(valid, "CameraInput V4L2 damaged or truncated frame / invalid bytesused");
    return {buffer.bytesused, buffer.sequence};
  }
  return {};
}
bool Capture::stop() noexcept {
  if (state_ == State::Failed)
    return false;
  if (fd_ < 0)
    return true;
  state_ = State::Stopping;
  if (queue_attempted_) {
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (backend_->io(fd_, VIDIOC_STREAMOFF, &type) != 0) {
      // Intentionally retain the fd and mappings. Never claim safe recovery or
      // automatically retry a device with an ambiguous driver-owned DMA queue.
      state_ = State::Failed;
      return false;
    }
  }
  for (const auto& mapping : mappings_)
    backend_->unmap(mapping.address, mapping.size);
  mappings_.clear();
  if (buffers_allocated_) {
    v4l2_requestbuffers release{};
    release.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    release.memory = V4L2_MEMORY_MMAP;
    if (backend_->io(fd_, VIDIOC_REQBUFS, &release) != 0) {
      // DMA has retired, but the queue was not released. Keep its owner and
      // report failure rather than hiding it in close() or permitting restart.
      state_ = State::Failed;
      return false;
    }
    buffers_allocated_ = false;
  }
  backend_->close(fd_);
  fd_ = -1;
  queue_attempted_ = false;
  state_ = State::Closed;
  format_ = {};
  return true;
}
} // namespace simaai::neat::camera_copy
