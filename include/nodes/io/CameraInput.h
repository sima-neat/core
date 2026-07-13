/**
 * @file
 * @ingroup nodes_io
 * @brief MIPI/libcamera camera source node wrapper.
 */
#pragma once

#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Options for CameraInput, a live libcamera/MIPI source.
 *
 * The public contract is deliberately camera/frame oriented.  When fallback is
 * enabled, Neat inserts a private adaptive camera memory bridge: EV74 SiMaAI
 * buffers pass through; OS/libcamera buffers are copied into pooled EV74 SiMaAI
 * memory and stamped with GstSimaMeta.  Users do not expose an OsToSima node.
 */
struct CameraInputOptions {
  // Optional libcamera camera-name, e.g. "imx477 5-001a" from `cam -l`.
  // Leave unset to let libcamera select its default camera.
  std::optional<std::string> camera_name;

  std::uint32_t width = 1920;
  std::uint32_t height = 1080;
  std::uint32_t framerate_num = 30;
  std::uint32_t framerate_den = 1;
  std::string format = "NV12";

  // Name used by downstream CVU/MLA configs through GstSimaMeta.
  std::string buffer_name = "camera";

  // Insert a small live-source queue by default to avoid unbounded camera backpressure.
  bool insert_queue = true;
  bool leaky_queue = true;
  std::uint32_t queue_depth = 2;

  // True (default): insert Neat's private adaptive bridge; EV74 SiMaAI buffers
  // pass through, otherwise the bridge copies into EV74 SiMaAI memory.
  // False: require strict camera/device zero-copy and fail if unavailable.
  bool allow_cpu_fallback = true;
};

class CameraInput final : public Node, public OutputSpecProvider {
public:
  explicit CameraInput(CameraInputOptions opt = {});

  std::string kind() const override {
    return "CameraInput";
  }
  std::string user_label() const override;
  InputRole input_role() const override {
    return InputRole::Source;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  MemoryContract memory_contract() const override {
    return MemoryContract::PreferDeviceZeroCopy;
  }

  std::string buffer_name_hint(int node_index) const override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;

  const CameraInputOptions& options() const {
    return opt_;
  }
  std::string caps_string() const;

private:
  CameraInputOptions opt_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> CameraInput(simaai::neat::CameraInputOptions opt = {});
} // namespace simaai::neat::nodes
