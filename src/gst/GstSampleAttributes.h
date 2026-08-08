/**
 * @file
 * @brief Shared conversion between `SampleAttributes` and the nested `attributes`
 *        `GstStructure` carried inside `GstSimaMeta`.
 *
 * Every boundary that reconstructs fixed Sample identity uses these helpers so that
 * attribute ownership, deep-copy, and clearing behave the same way everywhere. Copies are
 * always deep and independently owned: mutating one branch or Sample cannot mutate another.
 *
 * Not installed; internal to the runtime.
 */
#pragma once

#include "pipeline/GraphOptions.h"

#include <gst/gst.h>

namespace simaai::neat::gst_internal {

/// Field name of the nested attribute structure inside `GstSimaMeta`.
inline constexpr const char* kSimaMetaAttributesField = "attributes";

/// Read attributes from `buffer`'s `GstSimaMeta`. Clears `out` first; a buffer with no
/// meta or no nested structure yields an empty map.
void read_attributes(GstBuffer* buffer, SampleAttributes* out);

/// Read attributes from an existing `GstSimaMeta` structure.
void read_attributes_from_structure(const GstStructure* meta_structure, SampleAttributes* out);

/// Replace `buffer`'s attributes with `attributes`, adding `GstSimaMeta` if needed.
///
/// Passing an empty map removes the nested structure so a reused destination buffer cannot
/// retain stale attributes. Returns false when the buffer is not writable or the meta could
/// not be attached.
bool write_attributes(GstBuffer* buffer, const SampleAttributes& attributes);

/// Replace the attributes of an existing `GstSimaMeta` structure.
void write_attributes_to_structure(GstStructure* meta_structure,
                                   const SampleAttributes& attributes);

/// Remove any attributes from `buffer`. Safe when no meta or no attributes are present.
bool clear_attributes(GstBuffer* buffer);

/// True when `buffer` currently carries a non-empty nested attribute structure.
///
/// Read-only: callers use it to decide whether a write is needed at all, so a hot path
/// carrying no attributes never has to force a buffer writable.
bool has_attributes(GstBuffer* buffer);

/// Deep-copy attributes from `src` onto `dst`, clearing `dst`'s previous attributes.
///
/// When `src` carries none, `dst`'s are removed rather than retained. This is the operation
/// every pooled or reused destination needs.
bool copy_attributes(GstBuffer* src, GstBuffer* dst);

} // namespace simaai::neat::gst_internal
