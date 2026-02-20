/**
 * @file
 * @ingroup builder
 * @brief OutputSpec for caps/shape propagation between nodes.
 */
#pragma once

#include <cstddef>
#include <string>

namespace simaai::neat {

enum class SpecCertainty {
  Unknown = 0,
  Hint,
  Derived,
  Authoritative,
};

struct OutputSpec {
  std::string media_type; // e.g. "video/x-raw", "application/vnd.simaai.tensor"
  std::string format;     // e.g. "NV12", "RGB", "FP32"
  int width = -1;
  int height = -1;
  int depth = -1;
  int fps_num = 0;
  int fps_den = 1;
  std::string memory; // e.g. "SystemMemory", "SimaAI", "Unknown"
  std::string layout; // e.g. "HWC", "CHW", "Planar", "Unknown"
  std::string dtype;  // e.g. "UInt8", "Float32"
  std::size_t byte_size = 0;
  SpecCertainty certainty = SpecCertainty::Unknown;
  std::string note;

  bool is_unknown() const {
    return media_type.empty() && format.empty() && width <= 0 && height <= 0 && depth <= 0 &&
           byte_size == 0 && certainty == SpecCertainty::Unknown;
  }

  bool has_shape() const {
    return width > 0 && height > 0;
  }
};

class OutputSpecProvider {
public:
  virtual ~OutputSpecProvider() = default;
  virtual OutputSpec output_spec(const OutputSpec& input) const = 0;
};

std::size_t expected_byte_size(const OutputSpec& spec);
OutputSpec derive_output_spec(const class NodeGroup& group, const OutputSpec& input = {});

} // namespace simaai::neat
