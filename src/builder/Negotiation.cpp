#include "builder/Negotiation.h"

#include "builder/Node.h"

namespace simaai::neat {
namespace {

bool dtype_intersects(const std::vector<simaai::neat::TensorDType>& a,
                      const std::vector<simaai::neat::TensorDType>& b) {
  if (a.empty() || b.empty())
    return true;
  for (auto da : a) {
    for (auto db : b) {
      if (da == db)
        return true;
    }
  }
  return false;
}

bool shape_compatible(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  if (a.empty() || b.empty())
    return true;
  if (a.size() != b.size())
    return true;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] >= 0 && b[i] >= 0 && a[i] != b[i])
      return false;
  }
  return true;
}

bool constraint_compatible(const TensorConstraint& cur, const TensorConstraint& expected) {
  if (!dtype_intersects(cur.dtypes, expected.dtypes))
    return false;
  if (expected.rank >= 0 && cur.rank >= 0 && expected.rank != cur.rank)
    return false;
  if (!shape_compatible(cur.shape, expected.shape))
    return false;
  auto device_equal = [](const Device& a, const Device& b) {
    return a.type == b.type && a.id == b.id;
  };
  auto device_allowed = [&](const Device& dev, const std::vector<Device>& allowed) {
    if (allowed.empty())
      return true;
    for (const auto& candidate : allowed) {
      if (device_equal(dev, candidate))
        return true;
    }
    return false;
  };
  if (expected.device.has_value() && cur.device.has_value()) {
    if (!device_equal(*expected.device, *cur.device))
      return false;
  }
  if (expected.device.has_value() && !cur.device.has_value()) {
    if (!device_allowed(*expected.device, cur.allowed_devices))
      return false;
  }
  if (cur.device.has_value() && !expected.device.has_value()) {
    if (!device_allowed(*cur.device, expected.allowed_devices))
      return false;
  }
  if (!cur.allowed_devices.empty() && !expected.allowed_devices.empty()) {
    bool ok = false;
    for (const auto& a : cur.allowed_devices) {
      if (device_allowed(a, expected.allowed_devices)) {
        ok = true;
        break;
      }
    }
    if (!ok)
      return false;
  }
  if (expected.image_format.has_value() && cur.image_format.has_value()) {
    if (expected.image_format != cur.image_format)
      return false;
  }
  if (!expected.required_segments.empty() && !cur.required_segments.empty()) {
    if (expected.required_segments.size() != cur.required_segments.size())
      return false;
    for (size_t i = 0; i < expected.required_segments.size(); ++i) {
      if (expected.required_segments[i].name != cur.required_segments[i].name)
        return false;
      if (expected.required_segments[i].size_bytes != cur.required_segments[i].size_bytes) {
        return false;
      }
    }
  }
  auto names_in_segments = [](const std::vector<std::string>& names,
                              const std::vector<Segment>& segments) {
    for (const auto& name : names) {
      bool found = false;
      for (const auto& seg : segments) {
        if (seg.name == name) {
          found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
    return true;
  };
  if (!expected.required_segment_names.empty() && !cur.required_segments.empty()) {
    if (!names_in_segments(expected.required_segment_names, cur.required_segments))
      return false;
  }
  if (!cur.required_segment_names.empty() && !expected.required_segments.empty()) {
    if (!names_in_segments(cur.required_segment_names, expected.required_segments))
      return false;
  }
  return true;
}

} // namespace

NegotiationResult validate_tensor_pipeline(const simaai::neat::NodeGroup& group,
                                           const TensorConstraint& input,
                                           ConversionPolicy /*policy*/) {
  NegotiationResult out;
  TensorConstraint current = input;

  for (const auto& node : group.nodes()) {
    if (!node)
      continue;
    const auto* provider = dynamic_cast<const SpecProvider*>(node.get());
    if (!provider)
      continue;

    const TensorConstraint expected = provider->expected_input_spec();
    if (!constraint_compatible(current, expected)) {
      out.ok = false;
      out.error = "Negotiation: constraint mismatch at node " + node->kind();
      return out;
    }
    current = provider->output_spec(current);
  }

  return out;
}

} // namespace simaai::neat
