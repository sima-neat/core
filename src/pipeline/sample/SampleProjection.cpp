#include "pipeline/internal/SampleProjection.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::pipeline_internal {

StreamNames::StreamNames(std::string default_name) : default_name_(std::move(default_name)) {
  names_.emplace_back(); // id 0: unassigned
  if (!default_name_.empty()) {
    default_id_ = 1;
    names_.push_back(default_name_);
    ids_.emplace(default_name_, default_id_);
  }
}

std::uint32_t StreamNames::intern(std::string_view name) {
  if (name.empty() || name == default_name_) {
    return default_id_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  const std::string key(name);
  if (const auto it = ids_.find(key); it != ids_.end()) {
    return it->second;
  }
  const auto id = static_cast<std::uint32_t>(names_.size());
  names_.push_back(key);
  ids_.emplace(key, id);
  return id;
}

std::string StreamNames::name(std::uint32_t id) const {
  if (id == 0) {
    return {};
  }
  if (id == default_id_) {
    return default_name_;
  }
  std::lock_guard<std::mutex> lock(mu_);
  return id < names_.size() ? names_[id] : std::string{};
}

Stamp stamp_from_sample(const Sample& sample) {
  Stamp out;
  out.stream_name = sample.stream_id;
  out.frame_id = sample.frame_id >= 0 ? sample.frame_id : -1;
  if (sample.pts_ns >= 0) {
    out.source_time = TimePoint{sample.pts_ns, ClockId::SensorLocal};
  }
  return out;
}

void apply_pulled_stamp(const Stamp& stamp, Sample& out) {
  out.frame_id = stamp.frame_id >= 0 ? stamp.frame_id : -1;
  out.stream_id = stamp.stream_name;
  out.pts_ns = stamp.source_time.valid() ? stamp.source_time.ns : -1;
  out.input_seq = stamp.seq >= 1 ? static_cast<std::int64_t>(stamp.seq - 1) : -1;
  // develop: the pushed orig_input_seq, else input_seq (InputStreamPush.cpp:248-249,
  // InputStreamPull.cpp:803-807). The stamp has no field for a pushed one.
  out.orig_input_seq = out.input_seq;
}

void apply_stamp_proposals(const Stamp& stamp, Sample& out) {
  if (stamp.frame_id >= 0) {
    out.frame_id = stamp.frame_id;
  }
  if (!stamp.stream_name.empty()) {
    out.stream_id = stamp.stream_name;
  }
  if (stamp.source_time.valid()) {
    out.pts_ns = stamp.source_time.ns;
  }
  if (out.kind == SampleKind::Bundle) {
    for (Sample& field : out.fields) {
      apply_stamp_proposals(stamp, field);
    }
  }
}

Stamp stamp_of_pulled_sample(const Sample& sample, StreamNames* streams) {
  Stamp out = stamp_from_sample(sample);
  out.stream = streams && !sample.stream_id.empty() ? streams->intern(sample.stream_id) : 0U;
  out.seq = sample.input_seq >= 0 ? static_cast<std::uint64_t>(sample.input_seq) + 1U : 0U;
  return out;
}

bool is_positional_record(const Tensor& tensor) noexcept {
  const auto names = tensor.field_names();
  if (names.empty() || names.size() != tensor.fields().size()) {
    return false;
  }
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] != std::to_string(i)) {
      return false;
    }
  }
  return true;
}

TensorList tensor_push_items(const Tensor& input, const char* where) {
  if (!input.is_record()) {
    return TensorList{input};
  }
  if (!is_positional_record(input)) {
    error_util::throw_session_error(
        error_codes::kInputShape,
        std::string(where) + ": a record with named fields is not supported before H1; push a "
                             "positional record (Tensor::record)");
  }
  const auto fields = input.fields();
  for (const Tensor& field : fields) {
    if (field.is_record()) {
      error_util::throw_session_error(error_codes::kInputShape,
                                      std::string(where) +
                                          ": a record whose field is a record is not supported "
                                          "before H1");
    }
  }
  return TensorList(fields.begin(), fields.end());
}

std::optional<TensorList> list_record_fields(const TensorList& inputs, const char* where) {
  if (inputs.size() == 1U) {
    if (!inputs.front().is_record()) {
      return std::nullopt;
    }
    return tensor_push_items(inputs.front(), where);
  }
  for (const Tensor& tensor : inputs) {
    if (tensor.is_record()) {
      error_util::throw_session_error(error_codes::kInputShape,
                                      std::string(where) +
                                          ": a record inside a list of several tensors is not "
                                          "supported before H1; push the record alone, or its "
                                          "fields as the list");
    }
  }
  return std::nullopt;
}

namespace {

Tensor make_record(std::vector<Tensor> fields, std::vector<std::string> names, const Stamp& stamp) {
  auto leaf_envelope = std::make_shared<TensorEnvelope>();
  leaf_envelope->stamp = stamp;
  for (Tensor& field : fields) {
    if (!field.is_record()) {
      field.envelope = leaf_envelope;
    }
  }
  auto envelope = std::make_shared<TensorEnvelope>();
  envelope->stamp = stamp;
  envelope->fields = std::move(fields);
  envelope->names = std::move(names);
  Tensor out;
  out.envelope = std::move(envelope);
  return out;
}

std::vector<std::string> positional_names(std::size_t n) {
  std::vector<std::string> names;
  names.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    names.push_back(std::to_string(i));
  }
  return names;
}

Tensor leaf_with_stamp(Tensor tensor, const Stamp& stamp) {
  auto envelope = std::make_shared<TensorEnvelope>();
  envelope->stamp = stamp;
  tensor.envelope = std::move(envelope);
  return tensor;
}

} // namespace

Tensor make_positional_record(std::vector<Tensor> fields, const Stamp& stamp) {
  std::vector<std::string> names = positional_names(fields.size());
  return make_record(std::move(fields), std::move(names), stamp);
}

std::optional<Tensor> tensor_from_pulled_sample(const Sample& sample, const Stamp& stamp) {
  if (sample.kind == SampleKind::Bundle) {
    if (sample.fields.empty()) {
      return std::nullopt;
    }
    if (sample.fields.size() == 1U) {
      return tensor_from_pulled_sample(sample.fields.front(), stamp);
    }
    std::vector<Tensor> fields;
    std::vector<std::string> names;
    std::set<std::string> unique;
    fields.reserve(sample.fields.size());
    for (const Sample& field : sample.fields) {
      std::optional<Tensor> tensor = tensor_from_pulled_sample(field, stamp);
      if (!tensor) {
        return std::nullopt;
      }
      fields.push_back(std::move(*tensor));
      names.push_back(field.port_name);
      unique.insert(field.port_name);
    }
    // The fields' port names when each has a distinct one (to_sample writes them), else
    // positional names.
    const bool named = unique.size() == names.size() &&
                       std::none_of(names.begin(), names.end(),
                                    [](const std::string& name) { return name.empty(); });
    if (!named) {
      names = positional_names(fields.size());
    }
    return make_record(std::move(fields), std::move(names), stamp);
  }
  if (sample_has_tensor_list(sample)) {
    if (sample.tensors.size() == 1U) {
      return leaf_with_stamp(sample.tensors.front(), stamp);
    }
    return make_positional_record(TensorList(sample.tensors), stamp);
  }
  if (sample.tensor.has_value()) {
    return leaf_with_stamp(*sample.tensor, stamp);
  }
  return std::nullopt;
}

std::optional<TensorList> tensors_from_pulled_sample(const Sample& sample, const Stamp& stamp) {
  if (sample.kind != SampleKind::Bundle && sample_has_tensor_list(sample)) {
    // The common case, without a record envelope: every tensor shares one leaf envelope.
    auto envelope = std::make_shared<TensorEnvelope>();
    envelope->stamp = stamp;
    TensorList out(sample.tensors);
    for (Tensor& tensor : out) {
      tensor.envelope = envelope;
    }
    return out;
  }
  std::optional<Tensor> item = tensor_from_pulled_sample(sample, stamp);
  if (!item) {
    return std::nullopt;
  }
  if (!item->is_record()) {
    return TensorList{std::move(*item)};
  }
  const auto fields = item->fields();
  return TensorList(fields.begin(), fields.end());
}

const char* pixel_format_token(ImageSpec::PixelFormat format) noexcept {
  switch (format) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::NV16:
    return "NV16";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

} // namespace simaai::neat::pipeline_internal

namespace simaai::neat {

namespace {

// A Sample's media fields from a leaf's semantic, as an Input -> Output Run returns them.
void set_media_fields(const Tensor& leaf, Sample& out) {
  if (leaf.semantic.image.has_value()) {
    out.media_type = "video/x-raw";
    out.payload_type = PayloadType::Image;
    out.payload_tag = pipeline_internal::pixel_format_token(leaf.semantic.image->format);
    out.format = out.payload_tag;
  } else {
    out.media_type = "application/vnd.simaai.tensor";
    out.payload_type = PayloadType::Tensor;
  }
}

} // namespace

Tensor to_tensor(const Sample& sample) {
  const Stamp stamp = pipeline_internal::stamp_of_pulled_sample(sample, nullptr);
  std::optional<Tensor> out = pipeline_internal::tensor_from_pulled_sample(sample, stamp);
  if (!out) {
    pipeline_internal::error_util::throw_session_error(error_codes::kInputShape,
                                                       "to_tensor: the Sample carries no tensor");
  }
  return std::move(*out);
}

Sample to_sample(const Tensor& tensor) {
  Sample out;
  if (tensor.is_record() && !pipeline_internal::is_positional_record(tensor)) {
    out.kind = SampleKind::Bundle;
    const auto fields = tensor.fields();
    const auto names = tensor.field_names();
    out.fields.reserve(fields.size());
    for (std::size_t i = 0; i < fields.size(); ++i) {
      Sample field = to_sample(fields[i]);
      field.port_name = i < names.size() ? names[i] : std::string{};
      out.fields.push_back(std::move(field));
    }
  } else {
    out.kind = SampleKind::TensorSet;
    if (tensor.is_record()) {
      out.tensors.assign(tensor.fields().begin(), tensor.fields().end());
    } else {
      out.tensors.push_back(tensor);
    }
    if (!out.tensors.empty()) {
      set_media_fields(out.tensors.front(), out);
    }
  }
  pipeline_internal::apply_pulled_stamp(tensor.stamp(), out);
  if (out.kind == SampleKind::Bundle) {
    for (Sample& field : out.fields) {
      pipeline_internal::apply_pulled_stamp(tensor.stamp(), field);
    }
  }
  return out;
}

} // namespace simaai::neat
