#pragma once

#include "simaai/neat/internal/dmabuf/DmaBufPool.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sima_test {

struct DmaBufBackingIdentity {
  std::uint64_t device = 0;
  std::uint64_t inode = 0;

  bool operator==(const DmaBufBackingIdentity&) const = default;
};

struct DmaBufSpan {
  DmaBufBackingIdentity backing;
  std::size_t offset = 0;
  std::size_t length = 0;

  bool operator==(const DmaBufSpan&) const = default;
};

// Identity inspection never maps payloads. Duplicated FDs identify the same
// backing object; shared GstMemory views can have different offsets/lengths.
inline DmaBufSpan dmabuf_span(GstMemory* memory) {
  if (!memory || !gst_is_dmabuf_memory(memory)) {
    throw std::runtime_error("expected standard DMA-BUF GstMemory");
  }
  struct stat status {};
  if (::fstat(gst_dmabuf_memory_get_fd(memory), &status) != 0) {
    throw std::runtime_error(std::string("identify DMA-BUF: ") + std::strerror(errno));
  }
  gsize offset = 0;
  const gsize length = gst_memory_get_sizes(memory, &offset, nullptr);
  return {{static_cast<std::uint64_t>(status.st_dev), static_cast<std::uint64_t>(status.st_ino)},
          offset,
          length};
}

inline DmaBufBackingIdentity dmabuf_identity(GstMemory* memory) {
  return dmabuf_span(memory).backing;
}

inline DmaBufSpan dmabuf_span(GstBuffer* buffer) {
  if (!buffer || gst_buffer_n_memory(buffer) != 1U) {
    throw std::runtime_error("expected one DMA-BUF payload memory");
  }
  return dmabuf_span(gst_buffer_peek_memory(buffer, 0U));
}

// Real exporter required for CPU synchronization and accelerator tests. Never
// silently replace a missing heap with a memfd or a CPU allocation.
inline GstBuffer* allocate_cma_dmabuf(std::size_t size) {
  namespace dma = simaai::neat::internal::dmabuf;
  dma::Error error;
  GstBuffer* buffer = dma::allocateDmaBufBuffer(dma::HeapKind::Cma, size, {}, &error);
  if (!buffer) {
    throw std::runtime_error("allocate test CMA DMA-BUF: " + error.message());
  }
  return buffer;
}

// Host-compatible bookkeeping fixture ONLY. Although wrapped by the standard
// allocator, memfd is not a DMA exporter: do not map it through DmaBufView or
// submit it to hardware. It tests classification, identity and reference life.
inline GstBuffer* make_bookkeeping_dmabuf(std::size_t size) {
  if (size == 0U) {
    throw std::runtime_error("bookkeeping DMA-BUF fixture requires nonempty storage");
  }
  const int fd = ::memfd_create("neat-dmabuf-bookkeeping", MFD_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::string("create bookkeeping memfd: ") + std::strerror(errno));
  }
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
    const int error = errno;
    ::close(fd);
    throw std::runtime_error(std::string("size bookkeeping memfd: ") + std::strerror(error));
  }
  GstAllocator* allocator = gst_dmabuf_allocator_new();
  if (!allocator) {
    ::close(fd);
    throw std::runtime_error("create bookkeeping DMA-BUF allocator");
  }
  GstMemory* memory = gst_dmabuf_allocator_alloc(allocator, fd, size);
  gst_object_unref(allocator);
  if (!memory) {
    ::close(fd);
    throw std::runtime_error("wrap bookkeeping DMA-BUF memory");
  }
  GstBuffer* buffer = gst_buffer_new();
  if (!buffer) {
    gst_memory_unref(memory);
    throw std::runtime_error("create bookkeeping DMA-BUF buffer");
  }
  gst_buffer_append_memory(buffer, memory);
  return buffer;
}

} // namespace sima_test
