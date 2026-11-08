#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Compatibility projection between Core's Sample identity fields and the per-message context the
// Neat Dataflow runtime keeps in its Stamp (H0 plan 3.5; APPLICATION_MODEL.md 2.1). Internal in
// this step: the Dataflow backend uses it on push and pull. It does not depend on the runtime's
// types, so the public Stamp of the new surface can reuse it.
//
// Push (Sample -> stamp): frame_id -> frame_id (-1 is unset), stream_id -> stream (interned per
// Run), pts_ns -> source_time (-1 is unset). input_seq, orig_input_seq, dts_ns, duration_ns,
// stream_label, port_name and attributes have no place in the stamp and are not carried.
// Pull (stamp -> Sample): frame_id, stream_id and pts_ns back; input_seq = seq - 1 (the runtime's
// publish sequence is 1-based and contiguous per input; Core's own counter starts at 0);
// orig_input_seq = frame_id when set, else input_seq.

#include "pipeline/GraphOptions.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal {

// The runtime-independent fields of a Dataflow stamp that the projection reads and writes.
struct ProjectedStamp {
  std::uint32_t stream = 0;                   ///< per-Run stream id; 0 is unassigned
  std::uint64_t seq = 0;                      ///< publish sequence, 1-based; 0 is unpublished
  std::int64_t frame_id = -1;                 ///< author's correlation key; -1 is unset
  std::optional<std::int64_t> source_time_ns; ///< Sample::pts_ns
};

// Core's string stream_id against the runtime's numeric StreamId: interned per Run. Id 0 stays
// "unassigned"; the empty name maps to the Run's default stream. Pushes intern, pulls resolve;
// both may run on different threads.
class StreamNames {
public:
  explicit StreamNames(std::string default_name);
  StreamNames(const StreamNames&) = delete;
  StreamNames& operator=(const StreamNames&) = delete;

  /// The id of `name`; the default stream's id for an empty name.
  std::uint32_t intern(std::string_view name);
  /// The name of `id`; empty for an unknown id or the unassigned id 0.
  std::string name(std::uint32_t id) const;
  /// The default stream's id (0 when the default name is empty).
  std::uint32_t default_id() const noexcept {
    return default_id_;
  }

private:
  std::string default_name_;
  std::uint32_t default_id_ = 0;
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::uint32_t> ids_;
  std::vector<std::string> names_; // index = id; names_[0] is the unassigned id
};

/// The stamp fields a push proposes for `sample` (seq is the runtime's, left 0).
ProjectedStamp project_push_identity(const Sample& sample, StreamNames& streams);

/// Writes the identity fields of a pulled `stamp` into `out`.
void apply_pulled_identity(const ProjectedStamp& stamp, const StreamNames& streams, Sample& out);

/// The payload tag of an image format ("RGB", "NV12", ...), as a pulled Sample's payload_tag
/// carries it; empty for UNKNOWN.
const char* pixel_format_token(ImageSpec::PixelFormat format) noexcept;

} // namespace simaai::neat::pipeline_internal
