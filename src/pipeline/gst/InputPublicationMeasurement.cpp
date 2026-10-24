#include "pipeline/internal/InputPublicationMeasurement.h"

#include <gstsimaaitensorbuffer_meta.h>

#include <stdexcept>
#include <utility>

namespace simaai::neat::pipeline_internal {
namespace {

InputPublicationIdentity read_identity(GstBuffer* buffer) {
  InputPublicationIdentity identity;
  std::string error;
  if (const auto* frame = simaai::gst::tensor_buffer_frame_metadata(buffer, &error)) {
    identity.stream_id = frame->stream_id;
    identity.original_sequence = frame->original_sequence.value_or(-1);
    identity.input_sequence = frame->input_sequence.value_or(-1);
    identity.frame_id = frame->frame_id;
    return identity;
  }
  if (!error.empty()) {
    throw std::runtime_error("Input publication measurement: " + error);
  }
  GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
  const GstStructure* fields = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (fields) {
    if (const char* stream = gst_structure_get_string(fields, "stream-id"))
      identity.stream_id = stream;
    gst_structure_get_int64(fields, "orig-input-seq", &identity.original_sequence);
    gst_structure_get_int64(fields, "input-seq", &identity.input_sequence);
    gst_structure_get_int64(fields, "frame-id", &identity.frame_id);
  }
  return identity;
}

} // namespace

InputPublicationMeasurement::Reservation::Reservation(Reservation&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), epoch_(other.epoch_), index_(other.index_) {}

void InputPublicationMeasurement::start(std::uint64_t epoch) {
  if (epoch == 0)
    throw std::invalid_argument("Input publication measurement requires a nonzero epoch");
  std::lock_guard lock(mutex_);
  entries_.clear();
  failed_ = false;
  epoch_.store(epoch, std::memory_order_release);
}

InputPublicationSnapshot InputPublicationMeasurement::stop(std::uint64_t epoch) {
  InputPublicationSnapshot selected;
  std::lock_guard lock(mutex_);
  if (epoch_.load(std::memory_order_relaxed) != epoch)
    return selected;
  epoch_.store(0, std::memory_order_release);
  selected.failed = failed_;
  selected.entries.reserve(entries_.size());
  for (auto& entry : entries_) {
    if (!entry.cancelled)
      selected.entries.push_back(std::move(entry.identity));
  }
  entries_.clear();
  return selected;
}

void InputPublicationMeasurement::discard(std::uint64_t epoch) noexcept {
  std::lock_guard lock(mutex_);
  if (epoch_.load(std::memory_order_relaxed) == epoch) {
    epoch_.store(0, std::memory_order_release);
    entries_.clear();
  }
}

InputPublicationMeasurement::Reservation
InputPublicationMeasurement::observe_active(GstBuffer* buffer, std::uint64_t epoch) noexcept {
  std::lock_guard lock(mutex_);
  if (epoch_.load(std::memory_order_relaxed) != epoch)
    return {};
  try {
    const auto index = entries_.size();
    auto identity = read_identity(buffer);
    if (identity.original_sequence < 0 && identity.input_sequence < 0 && identity.frame_id < 0)
      throw std::runtime_error("Input publication has no correlation identity");
    entries_.push_back({std::move(identity), false});
    return Reservation(this, epoch, index);
  } catch (...) {
    // Observation must not add a throwing edge while a transport owns a raw
    // buffer. Report measurement failure without replacing runtime validation.
    failed_ = true;
    return {};
  }
}

void InputPublicationMeasurement::cancel(std::uint64_t epoch, std::size_t index) noexcept {
  std::lock_guard lock(mutex_);
  if (epoch_.load(std::memory_order_relaxed) == epoch && index < entries_.size())
    entries_[index].cancelled = true;
}

} // namespace simaai::neat::pipeline_internal
