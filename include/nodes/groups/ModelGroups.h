/**
 * @file
 * @ingroup nodes_groups
 * @brief Model stage NodeGroups for MPK pipelines.
 */
#pragma once

#include "builder/NodeGroup.h"

#include <string>
#include <vector>
#include <cstdint>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat::nodes::groups {

struct InferOptions {
  int input_width = 0;
  int input_height = 0;
  std::string input_format;
  bool normalize = false;
  std::vector<float> mean;
  std::vector<float> stddev;
  std::string upstream_name;
  std::string preproc_next_cpu;
  int num_buffers_cvu = 4;
  int num_buffers_mla = 4;
  int queue_max_buffers = 0;
  int64_t queue_max_time_ns = -1;
  std::string queue_leaky;
  bool sync_mode = false;
};

simaai::neat::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt = {});
simaai::neat::NodeGroup simple_infer(const std::string& tar_gz);
simaai::neat::NodeGroup postprocessing(const std::string& tar_gz);
simaai::neat::NodeGroup infer(const std::string& tar_gz);
simaai::neat::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt);

simaai::neat::NodeGroup Preprocess(const simaai::neat::Model& model);
simaai::neat::NodeGroup MLA(const simaai::neat::Model& model);
simaai::neat::NodeGroup Postprocess(const simaai::neat::Model& model);
simaai::neat::NodeGroup Infer(const simaai::neat::Model& model);

} // namespace simaai::neat::nodes::groups
