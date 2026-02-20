/**
 * @file
 * @ingroup model
 * @brief ModelPack: load model packs and expose pipeline fragments/stages (internal).
 */
#pragma once

#include "builder/NodeGroup.h"
#include "nodes/io/Input.h"
#include "mpk/PipelineSequence.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace simaai::neat::internal {

enum class PipelineType : std::uint8_t { QuantTess, Preproc, CastTess };

enum class ModelStage { Preprocess, MlaOnly, Postprocess, Full };

struct InferenceTerminalPolicy {
  bool mla_only = false;
  std::optional<std::size_t> last_stage_index;
  std::optional<std::string> last_stage_name;
  std::optional<std::string> last_plugin_id;
  std::optional<std::string> last_processor;
};

struct ModelFragment {
  std::string gst;
  std::vector<std::string> elements;
  std::vector<std::string> config_paths;
};

/**
 * @brief Load model packs and expose stage fragments/NodeGroups.
 */
class ModelPack {
public:
  explicit ModelPack(const std::string& tar_gz);
  ModelPack(const std::string& tar_gz, const std::string& media_type, const std::string& format,
            int width, int height, int depth, int max_width = 0, int max_height = 0,
            int max_depth = 0, bool normalize = false, std::vector<float> mean = {},
            std::vector<float> stddev = {}, const std::string& preproc_next_cpu = {},
            const std::string& upstream_name = "decoder", int num_buffers_cvu = 4,
            int num_buffers_mla = 4, int queue_max_buffers = 0, int64_t queue_max_time_ns = -1,
            const std::string& queue_leaky = {}, const std::string& name_suffix = {},
            const InferenceTerminalPolicy& terminal_policy = {});
#if defined(SIMA_WITH_OPENCV)
  ModelPack(const std::string& tar_gz, const cv::Mat& mat, int max_width = 0, int max_height = 0,
            int max_depth = 0, bool normalize = false, std::vector<float> mean = {},
            std::vector<float> stddev = {}, const std::string& preproc_next_cpu = {},
            const std::string& upstream_name = "decoder", int num_buffers_cvu = 4,
            int num_buffers_mla = 4, int queue_max_buffers = 0, int64_t queue_max_time_ns = -1,
            const std::string& queue_leaky = {}, const std::string& name_suffix = {},
            const InferenceTerminalPolicy& terminal_policy = {});
#endif

  const std::string& etc_dir() const {
    return etc_dir_;
  }
  PipelineType pipeline_type() const {
    return pipeline_type_;
  }
  std::string find_config_path_by_plugin(const std::string& plugin_id) const;
  std::string find_config_path_by_processor(const std::string& processor) const;

  ModelFragment fragment(ModelStage stage) const;
  std::string backend_fragment(ModelStage stage) const;
  simaai::neat::NodeGroup to_node_group(ModelStage stage) const;

  // Infer block (pipeline_sequence.json without user-controlled pre/post).
  simaai::neat::NodeGroup infer_block(const std::string& upstream_name = {}) const;
  simaai::neat::mpk::SequenceSplit split_sequence() const;
  std::string apply_name_suffix(const std::string& base) const;

  int num_buffers_cvu() const {
    return options_.num_buffers_cvu;
  }
  int num_buffers_mla() const {
    return options_.num_buffers_mla;
  }

  simaai::neat::InputOptions input_appsrc_options(bool tensor_mode) const;

  ModelPack clone_with_buffers(int num_buffers_cvu, int num_buffers_mla) const;
  ModelPack clone_with_overrides(const std::string& upstream_name,
                                 const std::string& name_suffix) const;

private:
  struct Config {
    bool normalize = false;
    std::vector<float> mean;
    std::vector<float> stddev;

    int input_width = 0;
    int input_height = 0;
    std::string input_format; // "RGB"/"BGR"/"GRAY"/"NV12"/"IYUV"
    int input_depth = 0;
    int max_input_width = 0;
    int max_input_height = 0;
    int max_input_depth = 0;

    std::string preproc_next_cpu;
    std::string upstream_name = "decoder";

    int num_buffers_cvu = 4;
    int num_buffers_mla = 4;

    int queue_max_buffers = 0;
    int64_t queue_max_time_ns = -1;
    std::string queue_leaky;
    std::string name_suffix;
    InferenceTerminalPolicy terminal_policy;
  };

  void init(const std::string& tar_gz);
  void init_from_config(const std::string& tar_gz, Config cfg);

  std::string etc_dir_;
  Config options_;
  PipelineType pipeline_type_ = PipelineType::Preproc;
};

} // namespace simaai::neat::internal
