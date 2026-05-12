/**
 * @file
 * @ingroup nodes_groups
 * @brief Model-stage NodeGroups: preprocess, MLA inference, postprocess, and full infer chains.
 *
 * Wraps the four canonical Node sequences a `Model` expands into when added to a
 * Session — preprocess, MLA, postprocess, and the combined `Infer` group that chains
 * them together. The `_tar_gz` overloads build directly from an MPK on disk; the
 * `_Model&` overloads accept an already-parsed `Model` and are what `Model::session()`
 * uses internally.
 *
 * @see Model
 */
#pragma once

#include "builder/NodeGroup.h"
#include "model/PreprocessPlan.h"

#include <string>
#include <vector>
#include <cstdint>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat::nodes::groups {

/**
 * @brief Tunables for the preprocess and combined-infer groups.
 *
 * Selects the input color format expected by the model, optional mean/stddev
 * normalization, and the upstream-side queueing / buffering policy applied at the
 * preprocess and MLA boundaries.
 *
 * @ingroup nodes_groups
 */
struct InferOptions {
  PreprocessColorFormat input_format =
      PreprocessColorFormat::Auto; ///< Color format the model expects.
  bool normalize = false;          ///< If true, apply `(x - mean) / stddev` normalization.
  std::vector<float> mean;         ///< Per-channel mean used when `normalize` is true.
  std::vector<float> stddev;       ///< Per-channel stddev used when `normalize` is true.
  std::string upstream_name;       ///< Name of the upstream Node the preprocess attaches to.
  std::string preproc_next_cpu;    ///< Optional CPU-side next-element override after preprocess.
  int num_buffers_cvu = 4;         ///< CVU-side buffer pool depth for preprocess.
  int num_buffers_mla = 4;         ///< MLA-side buffer pool depth for inference.
  int queue_max_buffers = 0;       ///< Cap on queued buffers (0 = default).
  int64_t queue_max_time_ns = -1;  ///< Cap on queued time in ns (-1 = unset).
  std::string queue_leaky;         ///< Queue leaky policy ("upstream", "downstream", or empty).
  bool sync_mode = false;          ///< If true, run sinks in sync (real-time) mode.
};

/**
 * @brief Build the preprocess NodeGroup for a model, given an MPK tarball path.
 * @param tar_gz Path to the model `.tar.gz`.
 * @param opt Preprocess tunables (color format, normalization, queueing).
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt = {});

/**
 * @brief Build a single-stage MLA-inference NodeGroup from an MPK tarball, with default options.
 * @param tar_gz Path to the model `.tar.gz`.
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup simple_infer(const std::string& tar_gz);

/**
 * @brief Build the postprocess NodeGroup for a model, given an MPK tarball path.
 * @param tar_gz Path to the model `.tar.gz`.
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup postprocessing(const std::string& tar_gz);

/**
 * @brief Build the full preprocess + MLA + postprocess NodeGroup from an MPK tarball.
 * @param tar_gz Path to the model `.tar.gz`.
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup infer(const std::string& tar_gz);

/**
 * @brief Build the full preprocess + MLA + postprocess NodeGroup with explicit options.
 * @param tar_gz Path to the model `.tar.gz`.
 * @param opt Preprocess tunables propagated into the chain.
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt);

/**
 * @brief Build the preprocess NodeGroup from an already-parsed `Model`.
 * @param model The parsed model (used by `Model::session()`).
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup Preprocess(const simaai::neat::Model& model);

/**
 * @brief Build the MLA-inference NodeGroup from an already-parsed `Model`.
 * @param model The parsed model (used by `Model::session()`).
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup MLA(const simaai::neat::Model& model);

/**
 * @brief Build the postprocess NodeGroup from an already-parsed `Model`.
 * @param model The parsed model (used by `Model::session()`).
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup Postprocess(const simaai::neat::Model& model);

/**
 * @brief Build the full preprocess + MLA + postprocess NodeGroup from an already-parsed `Model`.
 * @param model The parsed model (used by `Model::session()`).
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup Infer(const simaai::neat::Model& model);

} // namespace simaai::neat::nodes::groups
