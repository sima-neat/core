#pragma once

// Which DMA-BUF a Tensor's storage is, for the device-memory tests (H0b): two tensors share one
// DMA-BUF exactly when their descriptors name the same inode, whatever GstMemory or descriptor
// each holds it through.

#include "pipeline/TensorCore.h"
#include "pipeline/internal/SampleUtil.h"

#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>

#include <cstdint>

#include <sys/stat.h>

namespace sima_test {

// The inode of the tensor's DMA-BUF; 0 when its storage is not one standard DMA-BUF memory.
inline std::uint64_t dmabuf_inode(const simaai::neat::Tensor& t) {
  if (!simaai::neat::pipeline_internal::tensor_has_dmabuf_memory(t)) {
    return 0;
  }
  GstBuffer* buffer = gst_sample_get_buffer(static_cast<GstSample*>(t.storage->holder.get()));
  const int fd = gst_dmabuf_memory_get_fd(gst_buffer_peek_memory(buffer, 0));
  struct stat st {};
  return fd >= 0 && fstat(fd, &st) == 0 ? static_cast<std::uint64_t>(st.st_ino) : 0U;
}

} // namespace sima_test
