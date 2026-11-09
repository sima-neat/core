#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// Compatibility projection between Core's Sample and a Tensor with its Stamp (H0 plan 3.5;
// APPLICATION_MODEL.md 2.1). Runtime-independent: both backends use it, the public
// to_tensor / to_sample (pipeline/GraphOptions.h) are built on it, and the
// Dataflow backend converts between this Stamp and the runtime's at its own boundary.
//
// Field meanings are develop's, on both backends (APPLICATION_MODEL.md decision 2):
//   Sample -> Stamp (what a push proposes): stream_id -> stream_name; frame_id -> frame_id (-1 is
//     unset); pts_ns -> source_time on ClockId::SensorLocal (-1 is unset).
//   Stamp -> Sample (a pulled message): stream_name -> stream_id, frame_id, source_time -> pts_ns
//     (any clock; -1 when unset); input_seq = seq - 1 (seq is 1-based and contiguous per input,
//     Core's counter 0-based); orig_input_seq = input_seq, as develop gives when the push carried
//     no orig_input_seq. A pushed input_seq or orig_input_seq has no field in the stamp: the
//     Dataflow backend cannot carry it (reported, c2.md 4).
//   Tensor -> Sample: a leaf is a TensorSet of one tensor (what push(TensorList{t}) makes); a
//     positional record (field names "0", "1", ...) a TensorSet of its fields; a record with other
//     names a Bundle of one field Sample each, in schema order.
//   Sample -> Tensor: Tensor or TensorSet of one tensor -> a leaf; TensorSet of several -> a
//     positional record; Bundle -> a record of its fields, positional.

#include "pipeline/GraphOptions.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal {

// Core's string stream_id against a numeric stream id: interned per Run. Id 0 stays
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

/// The stamp fields a push of `sample` proposes (stream_name, frame_id, source_time).
Stamp stamp_from_sample(const Sample& sample);

/// Writes the identity of a pulled `stamp` into `out`: stream_id, frame_id, pts_ns, input_seq,
/// orig_input_seq.
void apply_pulled_stamp(const Stamp& stamp, Sample& out);

/// Writes what a pushed Tensor's stamp proposes into `out` (and into a Bundle's fields): frame_id
/// when set, stream_id when the name is set, pts_ns when source_time is set. Fields it does not
/// set keep their values.
void apply_stamp_proposals(const Stamp& stamp, Sample& out);

/// The stamp of a pulled Sample: its identity as Stamp fields, `stream` the id of its name in
/// `streams` (0 without), `seq` = input_seq + 1 (0 when unset). No publish_time: the Sample has
/// none.
Stamp stamp_of_pulled_sample(const Sample& sample, StreamNames* streams);

/// True iff `tensor` is a record whose field names are "0", "1", ... in order.
bool is_positional_record(const Tensor& tensor) noexcept;

/// The tensors one push of `input` carries: a leaf is itself; a positional record its fields (a
/// record of one field: that field). Throws NeatError (misconfig.input_shape) for a record with
/// other names or a field that is itself a record (not supported before H1).
TensorList tensor_push_items(const Tensor& input, const char* where);

/// A positional record of `fields` (names "0", "1", ...) carrying `stamp`; every field carries it
/// too, through one envelope shared by all of them.
Tensor make_positional_record(std::vector<Tensor> fields, const Stamp& stamp);

/// The Tensor of a pulled Sample (see the header comment), carrying `stamp`. nullopt when the
/// Sample carries no tensor.
std::optional<Tensor> tensor_from_pulled_sample(const Sample& sample, const Stamp& stamp);

/// The payload tag of an image format ("RGB", "NV12", ...), as a pulled Sample's payload_tag
/// carries it; empty for UNKNOWN.
const char* pixel_format_token(ImageSpec::PixelFormat format) noexcept;

} // namespace simaai::neat::pipeline_internal
