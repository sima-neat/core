#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::camera_copy {
struct Format {
  std::uint32_t width = 0, height = 0, fourcc = 0, stride = 0, size = 0;
};
struct Frame {
  std::uint32_t bytes = 0, sequence = 0;
};
// Injectable syscalls: tests never need a camera or device allocator.
class Backend {
public:
  virtual ~Backend() = default;
  virtual int open(const std::string& device) = 0;
  virtual int io(int fd, unsigned long request, void* arg) = 0;
  virtual void* map(int fd, std::uint32_t offset, std::uint32_t length) = 0;
  virtual void unmap(void* address, std::uint32_t length) noexcept = 0;
  virtual int wait(int fd, int timeout_ms) = 0;
  virtual void close(int fd) noexcept = 0;
};
std::shared_ptr<Backend> linux_backend();
std::uint32_t parse_fourcc(const std::string& text);

// Single capture thread. stop() runs only after next() has been interrupted and
// joined. Kernel MMAP storage never escapes this class; next() copies to the
// caller's independent storage before requeueing, including valid trailer bytes.
class Capture {
public:
  explicit Capture(std::shared_ptr<Backend> backend);
  ~Capture();
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  // Prepare without queueing: downstream caps and storage must be ready first.
  void prepare(const std::string& device, Format requested, std::uint32_t count);
  void start();
  Frame next(std::span<std::uint8_t> destination, const std::atomic<bool>& interrupted,
             std::uint32_t timeout_ms);
  bool stop() noexcept;
  const Format& format() const {
    return format_;
  }

private:
  struct Mapping {
    void* address;
    std::uint32_t size;
  };
  void queue(std::uint32_t index);
  std::shared_ptr<Backend> backend_;
  std::vector<Mapping> mappings_;
  Format format_;
  int fd_ = -1;
  enum class State { Closed, Prepared, Streaming, Stopping, Failed };
  State state_ = State::Closed;
  bool buffers_allocated_ = false;
  bool queue_attempted_ = false;
};
} // namespace simaai::neat::camera_copy
