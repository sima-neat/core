#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gstneatallocatorabi.h>
#ifndef SIMA_HAS_SIMAAI_POOL
#define SIMA_HAS_SIMAAI_POOL 0
#endif

// Canonical Scheduler transport ABI; no local flag or allocator declarations.
using GstSimaaiSegment = segment_t;
