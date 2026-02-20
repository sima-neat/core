#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <gst/gstbufferpool.h>
#include <gst/gstmemory.h>

#ifndef SIMA_HAS_SIMAAI_POOL
#define SIMA_HAS_SIMAAI_POOL 0
#endif

#ifndef GST_SIMAAI_MEMORY_TARGET_GENERIC
#define GST_SIMAAI_MEMORY_TARGET_GENERIC (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 0))
#define GST_SIMAAI_MEMORY_TARGET_OCM (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 1))
#define GST_SIMAAI_MEMORY_TARGET_DMS0 (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 2))
#define GST_SIMAAI_MEMORY_TARGET_DMS1 (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 3))
#define GST_SIMAAI_MEMORY_TARGET_DMS2 (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 4))
#define GST_SIMAAI_MEMORY_TARGET_DMS3 (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 5))
#define GST_SIMAAI_MEMORY_TARGET_EV74 (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 6))
#define GST_SIMAAI_MEMORY_FLAG_CACHED (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 7))
#define GST_SIMAAI_MEMORY_FLAG_RDONLY (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 8))
#define GST_SIMAAI_MEMORY_FLAG_DEFAULT (static_cast<GstMemoryFlags>(GST_MEMORY_FLAG_LAST << 9))
#endif

#ifndef GST_SIMAAI_MEMORY_FLAGS_DEFINED
typedef GstMemoryFlags GstSimaaiMemoryFlags;
#define GST_SIMAAI_MEMORY_FLAGS_DEFINED 1
#endif

#ifndef MAX_ALLOCATION_SEGMENTS
#define MAX_ALLOCATION_SEGMENTS 16
#endif

typedef struct {
  gsize size;
  const gchar* name;
} GstSimaaiSegment;

typedef struct GstSimaaiAllocationParamsTag {
  GstAllocationParams parent;
  GstSimaaiSegment segments[MAX_ALLOCATION_SEGMENTS];
  gsize num_of_segments;
} GstSimaaiAllocationParams;

#ifdef __cplusplus
extern "C" {
#endif

void gst_simaai_segment_memory_init_once(void);
GstAllocator* gst_simaai_memory_get_segment_allocator(void);
guintptr gst_simaai_segment_memory_get_phys_addr(const GstMemory* memory);
void gst_simaai_memory_allocation_params_init(GstSimaaiAllocationParams* params);
gboolean gst_simaai_memory_allocation_params_add_segment(GstSimaaiAllocationParams* params,
                                                         gsize size, const gchar* name);
void* gst_simaai_memory_get_segment(const GstMemory* memory, const gchar* name);

GstBufferPool* gst_simaai_allocate_buffer_pool(GstObject* object, GstAllocator* allocator,
                                               guint buffer_size, guint min_buffers,
                                               guint max_buffers, GstMemoryFlags flags);

GstBufferPool* gst_simaai_allocate_buffer_pool2(GstObject* object, GstAllocator* allocator,
                                                guint min_buffers, guint max_buffers,
                                                GstMemoryFlags flags, gsize number_of_segments,
                                                const gsize seg_sizes[], const gchar* seg_names[]);

gboolean gst_simaai_free_buffer_pool(GstBufferPool* pool);

#ifdef __cplusplus
}
#endif
