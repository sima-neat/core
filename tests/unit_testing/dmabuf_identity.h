#pragma once

// Which DMA-BUF a Tensor's storage is, for the device-memory tests (H0b): two tensors share one
// DMA-BUF exactly when their bytes are mapped from the same inode, whatever storage, descriptor or
// mapping each holds it through. Found from the storage's own CPU read map: the line of
// /proc/self/maps that holds the mapped address names the DMA-BUF it maps (the kernel's dmabuf
// file) and its inode. Public API only, so GStreamer's storage (StorageKind::GstSample) and the
// Dataflow backend's own (StorageKind::DeviceHandle) are read alike.

#include "pipeline/TensorCore.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace sima_test {

// The inode of the DMA-BUF the tensor's storage maps for a CPU read; 0 when it maps none (CPU
// memory), or when the tensor has no storage or no CPU map.
inline std::uint64_t dmabuf_inode(const simaai::neat::Tensor& t) {
  if (!t.storage) {
    return 0;
  }
  const simaai::neat::Mapping map = t.storage->map(simaai::neat::MapMode::Read);
  if (!map.data) {
    return 0;
  }
  const auto at = reinterpret_cast<std::uintptr_t>(map.data);
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream fields(line);
    std::string range, perms, offset, dev, path;
    std::uint64_t inode = 0;
    fields >> range >> perms >> offset >> dev >> inode;
    std::getline(fields, path);
    const std::size_t dash = range.find('-');
    if (dash == std::string::npos) {
      continue;
    }
    const auto begin = static_cast<std::uintptr_t>(std::stoull(range.substr(0, dash), nullptr, 16));
    const auto end = static_cast<std::uintptr_t>(std::stoull(range.substr(dash + 1), nullptr, 16));
    if (at >= begin && at < end) {
      return path.find("dmabuf") != std::string::npos ? inode : 0U;
    }
  }
  return 0;
}

} // namespace sima_test
