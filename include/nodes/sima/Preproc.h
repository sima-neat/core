/**
 * @file
 * @ingroup nodes_sima
 * @brief SimaAI pre-processing node wrapper.
 */
#pragma once

#include "builder/ConfigJsonOverride.h"
#include "builder/ConfigJsonProvider.h"
#include "builder/NextCpuConfigurable.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;

struct PreprocOptions {
  PreprocOptions() = default;
  explicit PreprocOptions(const simaai::neat::Model& model);

  int input_width = 1280;
  int input_height = 720;
  int output_width = 640;
  int output_height = 640;
  int scaled_width = 640;
  int scaled_height = 640;

  int input_channels = 3;
  int output_channels = 3;
  int batch_size = 1;

  bool normalize = true;
  bool aspect_ratio = true;
  bool tessellate = true;
  bool dynamic_input_dims = true;

  int tile_width = 128;
  int tile_height = 32;
  int tile_channels = 3;

  int input_offset = 0;
  int input_stride = 1;
  int output_stride = 1;

  int q_zp = -128;
  double q_scale = 255.06967737092486;

  std::vector<float> channel_mean = {0.0f, 0.0f, 0.0f};
  std::vector<float> channel_stddev = {1.0f, 1.0f, 1.0f};

  std::string input_img_type = "NV12";
  std::string output_img_type = "RGB";
  std::string output_dtype = "EVXX_INT8";
  std::string scaling_type = "BILINEAR";
  std::string padding_type = "CENTER";

  std::string graph_name = "preproc";
  std::string node_name = "preproc";
  std::string element_name;
  std::string cpu = "CVU";
  std::string next_cpu = "CVU";
  std::string debug = "EVXX_DBG_DISABLED";

  std::string upstream_name = "decoder";
  std::string graph_input_name = "input_image";

  std::vector<std::string> output_memory_order;

  int num_buffers = 0;
  int num_buffers_model = 0;
  bool num_buffers_locked = false;

  std::string config_path;
  std::string config_dir;
  bool keep_config = false;
  std::optional<nlohmann::json> config_json;
};

class Preproc final : public Node,
                      public OutputSpecProvider,
                      public ConfigJsonProvider,
                      public ConfigJsonOverride,
                      public NextCpuConfigurable {
public:
  explicit Preproc(PreprocOptions opt = {});

  std::string kind() const override {
    return "Preproc";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  bool has_config_json() const override {
    return true;
  }
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override;
  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;
  OutputSpec output_spec(const OutputSpec& input) const override;
  bool set_next_cpu_if_auto(const std::string& next_cpu) override;
  bool override_config_json(const std::function<void(nlohmann::json&)>& edit,
                            const std::string& tag) override;

  const PreprocOptions& options() const {
    return opt_;
  }
  const std::string& config_path() const {
    return config_path_;
  }
  const std::string& config_snapshot_path() const {
    return config_snapshot_path_;
  }
  const nlohmann::json* config_json() const override;

private:
  struct PreprocConfigHolder;

  PreprocOptions opt_;
  std::shared_ptr<PreprocConfigHolder> config_holder_;
  std::string config_path_;
  std::string config_snapshot_path_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> Preproc(PreprocOptions opt = {});
} // namespace simaai::neat::nodes
