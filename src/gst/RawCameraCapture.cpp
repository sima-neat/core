#include "gst/RawCameraCapture.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <linux/videodev2.h>
#include <stdexcept>
#include <utility>

namespace simaai::neat::raw_camera {
namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

void require_qualified_capture_driver(std::string_view bus_info) {
  require(!bus_info.starts_with("platform:"),
          "raw camera platform capture is blocked before buffer allocation: requires a "
          "qualified kernel DMA-quiescence contract for STREAMOFF, failed start and fd release");
}

std::uint64_t next_capture_generation(std::atomic<std::uint64_t>& counter) {
  auto value = counter.load(std::memory_order_relaxed);
  for (;;) {
    require(value != UINT64_MAX, "raw camera capture generation exhausted");
    if (counter.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
      return value + 1;
  }
}

std::uint32_t parse_fourcc(const std::string& text) {
  require(text.size() == 4, "raw camera fourcc must contain exactly four printable bytes");
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto ch = static_cast<unsigned char>(text[i]);
    require(ch >= 32 && ch <= 126, "raw camera fourcc is not printable");
    value |= static_cast<std::uint32_t>(ch) << (i * 8);
  }
  return value;
}

void validate_format(const Format& requested, const Format& actual) {
  const bool byte_format =
      requested.fourcc == V4L2_PIX_FMT_GREY || requested.fourcc == V4L2_PIX_FMT_SBGGR8 ||
      requested.fourcc == V4L2_PIX_FMT_SGBRG8 || requested.fourcc == V4L2_PIX_FMT_SGRBG8 ||
      requested.fourcc == V4L2_PIX_FMT_SRGGB8;
  require(byte_format, "raw camera supports only explicit unpacked eight-bit wire formats");
  require(requested.width && requested.height, "raw camera dimensions must be positive");
  require(actual.width == requested.width && actual.height == requested.height &&
              actual.fourcc == requested.fourcc,
          "V4L2 changed the requested raw camera format or geometry");
  require(actual.stride >= actual.width, "raw camera row stride is smaller than UInt8 width");
  const auto span = static_cast<std::uint64_t>(actual.stride) * actual.height;
  require(span <= actual.size, "raw camera sizeimage is smaller than the negotiated rows");
  require(actual.size > 0, "raw camera sizeimage must be positive");
}

void validate_frame(const Format& format, const Frame& frame, std::size_t slots) {
  require(format.height > 0 && format.stride >= format.width && format.width > 0,
          "raw camera frame has an invalid format");
  require(frame.index < slots, "raw camera dequeued an invalid buffer index");
  // Last-row padding may be absent; never include driver-private trailers in
  // the logical tensor. sizeimage remains the allocation/import requirement.
  const auto minimum = static_cast<std::uint64_t>(format.height - 1) * format.stride + format.width;
  const auto maximum = static_cast<std::uint64_t>(format.height) * format.stride;
  require(frame.bytes >= minimum && frame.bytes <= maximum && frame.bytes <= format.size,
          "raw camera bytesused is truncated or includes unexpected trailing data");
}

Lease::Lease(std::shared_ptr<Capture> owner, Frame value)
    : frame(value), owner_(std::move(owner)) {}
Lease::~Lease() {
  owner_->release(frame.index);
}
Capture::Capture(std::shared_ptr<Backend> backend) : backend_(std::move(backend)) {}
Capture::~Capture() {
  // stop() must be called by the owner before destruction; unknown sessions
  // are parked by the source. Only a session with no ambiguous driver-owned
  // descriptors can be closed normally; QBUF may precede STREAMON.
  if (fd_ >= 0 && !streaming_ && !unknown_)
    backend_->close_device(fd_);
}

void Capture::queue(std::uint32_t index) {
  v4l2_buffer value{};
  value.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  value.memory = V4L2_MEMORY_DMABUF;
  value.index = index;
  value.m.fd = buffers_.at(index).fd;
  value.length = buffers_.at(index).size;
  if (requires_quiescence_proof_) {
    // Some platform vb2 buf_queue callbacks submit DMA descriptors before
    // STREAMON. Even a failed/interrupted QBUF need not mean nothing was
    // submitted. Retain all owners before crossing that side-effect boundary.
    queue_attempted_ = true;
    quarantine_ = shared_from_this();
  }
  if (backend_->io(fd_, VIDIOC_QBUF, &value) != 0) {
    if (requires_quiescence_proof_)
      unknown_ = true;
    throw std::runtime_error("raw camera VIDIOC_QBUF failed");
  }
}

void Capture::start(const std::string& device, const std::string& name, Format requested,
                    std::uint32_t count) {
  std::lock_guard lock(mutex_);
  require(fd_ < 0, "raw camera session already opened");
  require(count >= 4 && count <= 128, "raw camera buffer count must be in [4,128]");
  static std::atomic<std::uint64_t> generations{0};
  generation_ = next_capture_generation(generations);
  fd_ = backend_->open_device(device);
  require(fd_ >= 0, "cannot open raw camera device");
  v4l2_capability caps{};
  require(backend_->io(fd_, VIDIOC_QUERYCAP, &caps) == 0, "raw camera VIDIOC_QUERYCAP failed");
  // Platform capture drivers require a positive retirement proof. In the
  // deployed SiMa driver STREAMOFF can return success before DMA termination.
  // No unsafe public bypass is provided; a future qualified driver protocol
  // can replace this conservative classification.
  requires_quiescence_proof_ = std::memcmp(caps.bus_info, "platform:", 9) == 0;
  const auto capabilities =
      (caps.capabilities & V4L2_CAP_DEVICE_CAPS) ? caps.device_caps : caps.capabilities;
  require((capabilities & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING)) ==
              (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING),
          "raw camera requires single-plane streaming capture");
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = requested.width;
  format.fmt.pix.height = requested.height;
  format.fmt.pix.pixelformat = requested.fourcc;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  require(backend_->io(fd_, VIDIOC_S_FMT, &format) == 0, "raw camera VIDIOC_S_FMT failed");
  format_ = {format.fmt.pix.width, format.fmt.pix.height, format.fmt.pix.pixelformat,
             format.fmt.pix.bytesperline, format.fmt.pix.sizeimage};
  validate_format(requested, format_);
  require(format.fmt.pix.field == V4L2_FIELD_NONE, "raw camera requires progressive capture");
  v4l2_requestbuffers request{};
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_DMABUF;
  request.count = count;
  require(backend_->io(fd_, VIDIOC_REQBUFS, &request) == 0,
          "raw camera requires V4L2 DMA-BUF import (no implicit copy fallback)");
  require(request.count >= count && request.count <= 128,
          "raw camera driver returned invalid queue depth");
  for (std::uint32_t i = 0; i < request.count; ++i) {
    auto allocation = backend_->allocate(format_.size, name);
    require(allocation.owner && allocation.fd >= 0 && allocation.size >= format_.size,
            "raw camera allocator returned an invalid DMA-BUF");
    require(allocation.physical, "raw camera allocation has no physical identity");
    require(allocation.bus && allocation.bus <= UINT32_MAX &&
                allocation.size - 1 <= UINT32_MAX - allocation.bus,
            "raw camera allocation is outside the EV74 32-bit bus address space");
    buffers_.push_back(std::move(allocation));
  }
  held_.resize(buffers_.size(), false);
  for (std::uint32_t i = 0; i < buffers_.size(); ++i)
    queue(i);
  auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // Even a failed STREAMON can have submitted DMA. The caller must stop/park.
  streaming_ = true;
  quarantine_ = shared_from_this();
  if (backend_->io(fd_, VIDIOC_STREAMON, &type) != 0) {
    unknown_ = true;
    throw std::runtime_error("raw camera VIDIOC_STREAMON failed; DMA retirement is unknown");
  }
  quarantine_.reset();
}

std::shared_ptr<Lease> Capture::next(const std::atomic<bool>& interrupted) {
  while (!interrupted.load()) {
    const int ready = backend_->wait(fd_, 100);
    if (ready == 0 || (ready < 0 && errno == EINTR))
      continue;
    require(ready > 0, "raw camera poll failed");
    std::lock_guard lock(mutex_);
    require(streaming_ && !unknown_, "raw camera session is not streaming");
    v4l2_buffer value{};
    value.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    value.memory = V4L2_MEMORY_DMABUF;
    if (backend_->io(fd_, VIDIOC_DQBUF, &value) < 0) {
      if (errno == EAGAIN || errno == EINTR)
        continue;
      throw std::runtime_error("raw camera VIDIOC_DQBUF failed");
    }
    require(value.index < buffers_.size() && !held_[value.index],
            "raw camera duplicate or invalid dequeue");
    require(value.length >= value.bytesused && value.length <= buffers_[value.index].size,
            "raw camera dequeue length exceeds its allocation or valid payload");
    held_[value.index] = true;
    Frame frame{value.index, value.bytesused, value.sequence, 0};
    // Capture timestamp is metadata, not part of the kernel's tensor bytes.
    require(value.timestamp.tv_sec >= 0 && value.timestamp.tv_usec >= 0 &&
                value.timestamp.tv_usec < 1000000,
            "raw camera returned an invalid timestamp");
    require((value.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC,
            "raw camera requires a monotonic capture timestamp");
    require(static_cast<std::uint64_t>(value.timestamp.tv_sec) <=
                (UINT64_MAX - 999999000ULL) / 1000000000ULL,
            "raw camera timestamp overflows nanoseconds");
    frame.timestamp_ns = static_cast<std::uint64_t>(value.timestamp.tv_sec) * 1000000000ULL +
                         static_cast<std::uint64_t>(value.timestamp.tv_usec) * 1000ULL;
    validate_frame(format_, frame, buffers_.size());
    require(!(value.flags & V4L2_BUF_FLAG_ERROR), "raw camera returned a damaged frame");
    frame.discontinuity = first_frame_;
    frame.capture_generation = generation_;
    first_frame_ = false;
    return std::make_shared<Lease>(shared_from_this(), frame);
  }
  return {};
}

void Capture::release(std::uint32_t index) noexcept {
  std::lock_guard lock(mutex_);
  if (index >= held_.size() || !held_[index])
    return;
  held_[index] = false;
  if (streaming_ && !unknown_) {
    try {
      queue(index);
    } catch (...) {
      unknown_ = true;
    }
  }
}

bool Capture::stop() {
  std::lock_guard lock(mutex_);
  if (!streaming_) {
    unknown_ = unknown_ || (requires_quiescence_proof_ && queue_attempted_);
    if (unknown_)
      quarantine_ = shared_from_this();
    return !unknown_;
  }
  auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  const bool stopped = backend_->io(fd_, VIDIOC_STREAMOFF, &type) == 0;
  streaming_ = false;
  unknown_ = unknown_ || !stopped || requires_quiescence_proof_;
  if (unknown_)
    quarantine_ = shared_from_this(); // Intentional parked ownership, never DMA-unsafe destruction.
  return !unknown_;
}
bool Capture::retired_unknown() const {
  std::lock_guard lock(mutex_);
  return unknown_;
}
} // namespace simaai::neat::raw_camera
