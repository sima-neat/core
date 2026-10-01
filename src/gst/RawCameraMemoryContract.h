#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/SimaaiGstCompat.h"

namespace simaai::neat::raw_camera {

// ABI v1 is additive: older allocators remain valid for existing consumers, but
// raw capture needs every descendant alias to retain its dequeue lease owner.
inline bool supports_raw_camera_memory(const GstNeatCameraMemoryApiV1* api) {
  constexpr guint64 required = GST_NEAT_CAMERA_MEMORY_CAP_DMABUF_EXPORT |
                               GST_NEAT_CAMERA_MEMORY_CAP_PACKED_LAYOUT |
                               GST_NEAT_CAMERA_MEMORY_CAP_DEVICE_WRITTEN |
                               GST_NEAT_CAMERA_MEMORY_CAP_SHARED_OWNER_RETENTION;
  return api && api->abi_version == GST_NEAT_CAMERA_MEMORY_API_VERSION_1 &&
         api->struct_size >= sizeof(*api) && (api->capabilities & required) == required &&
         api->init_once && api->get_allocator && api->allocation_params_init &&
         api->allocation_params_add_segment && api->has_packed_segments &&
         api->export_dmabuf_fd && api->share_packed && api->mark_device_written;
}

} // namespace simaai::neat::raw_camera
