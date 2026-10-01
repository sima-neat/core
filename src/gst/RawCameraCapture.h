#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::raw_camera {

struct Format {
  std::uint32_t width = 0, height = 0, fourcc = 0, stride = 0, size = 0;
};
struct Allocation {
  int fd = -1;
  std::uint64_t physical = 0; // CPU physical identity, preserved in buffer metadata.
  std::uint64_t bus = 0; // EV/STU device address; distinct from the physical identity.
  std::uint32_t size = 0;
  std::shared_ptr<void> owner;
};
struct Frame {
  std::uint32_t index = 0, bytes = 0, sequence = 0;
  std::uint64_t timestamp_ns = 0;
  bool discontinuity = false;
  std::uint64_t capture_generation = 0;
};

// Syscalls and allocation are injectable so lifecycle and driver-error paths are
// tested without opening a device. io() follows ioctl's return/errno convention.
class Backend {
public:
  virtual ~Backend() = default;
  virtual int open_device(const std::string& device) = 0;
  virtual int io(int fd, unsigned long request, void* arg) = 0;
  virtual int wait(int fd, int timeout_ms) = 0;
  virtual void close_device(int fd) = 0;
  virtual Allocation allocate(std::uint32_t size, const std::string& name) = 0;
};

// Shipping backend gate: process-local retention cannot make fd release safe
// when an unqualified platform driver still owns queued DMA descriptors.
void require_qualified_capture_driver(std::string_view bus_info);

std::uint64_t next_capture_generation(std::atomic<std::uint64_t>& counter);
std::uint32_t parse_fourcc(const std::string& text);
void validate_format(const Format& requested, const Format& actual);
void validate_frame(const Format& format, const Frame& frame, std::size_t slots);

class Capture;
class Lease {
public:
  Lease(std::shared_ptr<Capture> owner, Frame frame);
  ~Lease();
  Lease(const Lease&) = delete;
  Lease& operator=(const Lease&) = delete;
  const Frame frame;

private:
  std::shared_ptr<Capture> owner_;
};

class Capture : public std::enable_shared_from_this<Capture> {
public:
  explicit Capture(std::shared_ptr<Backend> backend);
  ~Capture();
  void start(const std::string& device, const std::string& name, Format requested,
             std::uint32_t count);
  std::shared_ptr<Lease> next(const std::atomic<bool>& interrupted);
  // Returns false when DMA retirement is unknown. Such sessions are retained
  // rather than freeing memory that a device may still write.
  bool stop();
  void release(std::uint32_t index) noexcept;
  const Format& format() const {
    return format_;
  }
  const Allocation& allocation(std::uint32_t index) const {
    return buffers_.at(index);
  }
  bool retired_unknown() const;

private:
  void queue(std::uint32_t index);
  std::shared_ptr<Backend> backend_;
  mutable std::mutex mutex_;
  int fd_ = -1;
  Format format_;
  std::vector<Allocation> buffers_;
  std::vector<bool> held_;
  bool streaming_ = false;
  bool first_frame_ = true;
  std::uint64_t generation_ = 0;
  bool unknown_ = false;
  bool requires_quiescence_proof_ = false;
  bool queue_attempted_ = false;
  std::shared_ptr<Capture> quarantine_;
};

} // namespace simaai::neat::raw_camera
