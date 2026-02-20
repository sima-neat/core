#include "builder/OutputSpec.h"

#include "builder/NodeGroup.h"
#include "builder/Node.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>

namespace simaai::neat {
namespace {

bool outputspec_debug_enabled() {
  const char* env = std::getenv("SIMA_DEBUG_OUTPUTSPEC_LOG");
  return env && env[0] != '\0' && env[0] != '0';
}

std::string spec_to_string(const OutputSpec& spec) {
  std::ostringstream ss;
  ss << "{media=" << (spec.media_type.empty() ? "?" : spec.media_type)
     << " format=" << (spec.format.empty() ? "?" : spec.format) << " w=" << spec.width
     << " h=" << spec.height << " d=" << spec.depth
     << " layout=" << (spec.layout.empty() ? "?" : spec.layout)
     << " dtype=" << (spec.dtype.empty() ? "?" : spec.dtype)
     << " mem=" << (spec.memory.empty() ? "?" : spec.memory) << " bytes=" << spec.byte_size << "}";
  return ss.str();
}

std::size_t bytes_per_element_from_dtype(const std::string& dtype) {
  if (dtype == "UInt8" || dtype == "Int8")
    return 1;
  if (dtype == "UInt16" || dtype == "Int16")
    return 2;
  if (dtype == "Int32" || dtype == "Float32")
    return 4;
  if (dtype == "Float64")
    return 8;
  if (dtype == "BFloat16")
    return 2;
  return 0;
}

std::size_t bytes_per_element_from_format(const std::string& fmt) {
  if (fmt == "FP32" || fmt == "DETESSDEQUANT")
    return 4;
  if (fmt == "DETESS")
    return 2;
  if (fmt == "EVXX_INT8" || fmt == "INT8")
    return 1;
  if (fmt == "EVXX_BFLOAT16" || fmt == "BF16" || fmt == "BFLOAT16")
    return 2;
  return 0;
}

} // namespace

std::size_t expected_byte_size(const OutputSpec& spec) {
  if (spec.byte_size > 0)
    return spec.byte_size;
  if (!spec.has_shape())
    return 0;

  if (spec.media_type == "video/x-raw") {
    if (spec.format == "NV12" || spec.format == "I420") {
      if ((spec.width & 1) || (spec.height & 1))
        return 0;
      return static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height) * 3 / 2;
    }
    if (spec.format == "RGB" || spec.format == "BGR") {
      return static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height) * 3;
    }
    if (spec.format == "GRAY8") {
      return static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height);
    }
  }

  if (spec.media_type == "application/vnd.simaai.tensor") {
    std::size_t elem = bytes_per_element_from_format(spec.format);
    if (elem == 0)
      elem = bytes_per_element_from_dtype(spec.dtype);
    if (elem == 0)
      return 0;
    if (spec.depth <= 0)
      return 0;
    return static_cast<std::size_t>(spec.width) * static_cast<std::size_t>(spec.height) *
           static_cast<std::size_t>(spec.depth) * elem;
  }

  return 0;
}

OutputSpec derive_output_spec(const NodeGroup& group, const OutputSpec& input) {
  OutputSpec spec = input;
  const auto& nodes = group.nodes();
  const bool dbg = outputspec_debug_enabled();
  if (dbg) {
    std::cerr << "[OutputSpec] start " << spec_to_string(spec) << " nodes=" << nodes.size() << "\n";
  }
  for (const auto& n : nodes) {
    if (!n)
      continue;
    auto* provider = dynamic_cast<const OutputSpecProvider*>(n.get());
    if (!provider) {
      if (dbg) {
        std::cerr << "[OutputSpec] node " << n->kind() << " has no OutputSpecProvider\n";
      }
      continue;
    }
    OutputSpec next = provider->output_spec(spec);
    if (dbg) {
      std::cerr << "[OutputSpec] node " << n->kind() << " in=" << spec_to_string(spec)
                << " out=" << spec_to_string(next) << "\n";
    }
    spec = std::move(next);
  }
  if (dbg) {
    std::cerr << "[OutputSpec] end " << spec_to_string(spec) << "\n";
  }
  return spec;
}

} // namespace simaai::neat
