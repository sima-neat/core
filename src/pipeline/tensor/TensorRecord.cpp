// Tensor's envelope (H0 plan 3.1): the Stamp a Run gives a Tensor and the fields of a record. A
// Tensor without an envelope is a leaf with the default Stamp; the envelope is shared by every
// copy of the Tensor and never changed once shared (with_stamp makes a new one).

#include "pipeline/TensorCore.h"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simaai::neat {

namespace {

const Stamp& default_stamp() noexcept {
  static const Stamp stamp{};
  return stamp;
}

std::string field_list(const TensorEnvelope& envelope) {
  std::string out = "[";
  for (std::size_t i = 0; i < envelope.names.size(); ++i) {
    out += (i == 0 ? "" : ", ") + envelope.names[i];
  }
  return out + "]";
}

void append_leaves(const Tensor& tensor, std::vector<Tensor>& out) {
  if (!tensor.is_record()) {
    out.push_back(tensor);
    return;
  }
  for (const Tensor& field : tensor.fields()) {
    append_leaves(field, out);
  }
}

} // namespace

const Stamp& Tensor::stamp() const noexcept {
  return envelope ? envelope->stamp : default_stamp();
}

Tensor Tensor::with_stamp(Stamp value) const {
  auto next = std::make_shared<TensorEnvelope>();
  next->stamp = std::move(value);
  if (envelope) {
    next->fields = envelope->fields;
    next->names = envelope->names;
  }
  Tensor out = *this;
  out.envelope = std::move(next);
  return out;
}

bool Tensor::is_record() const noexcept {
  return envelope && !envelope->fields.empty();
}

const Tensor& Tensor::operator[](std::string_view name) const {
  if (is_record()) {
    for (std::size_t i = 0; i < envelope->names.size() && i < envelope->fields.size(); ++i) {
      if (envelope->names[i] == name) {
        return envelope->fields[i];
      }
    }
    throw std::out_of_range("Tensor::operator[]: the record has no field '" + std::string(name) +
                            "'; its fields are " + field_list(*envelope));
  }
  throw std::out_of_range("Tensor::operator[]: field '" + std::string(name) +
                          "' of a Tensor that is not a record");
}

const Tensor& Tensor::operator[](FieldId id) const {
  if (is_record() && id.value < envelope->fields.size()) {
    return envelope->fields[id.value];
  }
  throw std::out_of_range(
      "Tensor::operator[]: no field at position " + std::to_string(id.value) +
      (is_record() ? "; the record has " + std::to_string(envelope->fields.size()) + " fields"
                   : std::string("; the Tensor is not a record")));
}

std::span<const Tensor> Tensor::fields() const noexcept {
  if (!envelope) {
    return {};
  }
  return {envelope->fields.data(), envelope->fields.size()};
}

std::span<const std::string> Tensor::field_names() const noexcept {
  if (!envelope || envelope->fields.empty()) {
    return {};
  }
  return {envelope->names.data(), envelope->names.size()};
}

std::vector<Tensor> Tensor::leaves() const {
  std::vector<Tensor> out;
  append_leaves(*this, out);
  return out;
}

Tensor Tensor::record(std::vector<Tensor> fields) {
  if (fields.empty()) {
    throw std::invalid_argument("Tensor::record: a record needs at least one field");
  }
  auto envelope = std::make_shared<TensorEnvelope>();
  envelope->names.reserve(fields.size());
  for (std::size_t i = 0; i < fields.size(); ++i) {
    envelope->names.push_back(std::to_string(i));
  }
  envelope->fields = std::move(fields);
  Tensor out;
  out.envelope = std::move(envelope);
  return out;
}

} // namespace simaai::neat
