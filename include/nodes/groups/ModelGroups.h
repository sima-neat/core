/**
 * @file
 * @ingroup nodes_groups
 * @brief Model-stage Graph fragments: preprocess, MLA inference, and postprocess helpers.
 *
 * Wraps the canonical Graph fragments a `Model` expands into when added to a
 * Graph — preprocess, MLA inference, and postprocess. The `Infer` fragment is the MLA
 * inference stage only (not pre+infer+post); use `Graph::add(model)` or `model.graph()`
 * for the full model route. The `_tar_gz` overloads build directly from a `.tar.gz` model
 * archive; the `_Model&` overloads accept an already-parsed `Model`.
 *
 * @see Model
 */
#pragma once

#include "model/PreprocessPlan.h"
#include "pipeline/Graph.h"

#include <string>
#include <vector>
#include <cstdint>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat::nodes::groups {

/**
 * @brief Tunables for the preprocess and combined-infer Graph fragments.
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
 * @brief Build the preprocess Graph fragment for a model, given a model archive path.
 * @param tar_gz Path to the `.tar.gz` model archive.
 * @param opt Preprocess tunables (color format, normalization, queueing).
 * @ingroup nodes_groups
 */
simaai::neat::Graph preprocessing(const std::string& tar_gz, const InferOptions& opt = {});

/**
 * @brief Build a single-stage MLA-inference Graph fragment from a model archive, with default
 * options.
 * @param tar_gz Path to the `.tar.gz` model archive.
 * @ingroup nodes_groups
 */
simaai::neat::Graph simple_infer(const std::string& tar_gz);

/**
 * @brief Build the postprocess Graph fragment for a model, given a model archive path.
 * @param tar_gz Path to the `.tar.gz` model archive.
 * @ingroup nodes_groups
 */
simaai::neat::Graph postprocessing(const std::string& tar_gz);

/**
 * @brief Build the current inference-only Graph fragment from a model archive.
 * @param tar_gz Path to the `.tar.gz` model archive.
 * @ingroup nodes_groups
 */
simaai::neat::Graph infer(const std::string& tar_gz);

/**
 * @brief Build the current inference-only Graph fragment with explicit options.
 * @param tar_gz Path to the `.tar.gz` model archive.
 * @param opt Preprocess tunables propagated into the chain.
 * @ingroup nodes_groups
 */
simaai::neat::Graph infer(const std::string& tar_gz, const InferOptions& opt);

/**
 * @brief Build the preprocess Graph fragment from an already-parsed `Model`.
 * @param model The parsed model.
 * @ingroup nodes_groups
 */
simaai::neat::Graph Preprocess(const simaai::neat::Model& model);

/**
 * @brief Build the MLA-inference Graph fragment from an already-parsed `Model`.
 * @param model The parsed model.
 * @ingroup nodes_groups
 */
simaai::neat::Graph MLA(const simaai::neat::Model& model);

/**
 * @brief Build the postprocess Graph fragment from an already-parsed `Model`.
 * @param model The parsed model.
 * @ingroup nodes_groups
 */
simaai::neat::Graph Postprocess(const simaai::neat::Model& model);

/**
 * @brief Build the current inference-only Graph fragment from an already-parsed `Model`.
 * @param model The parsed model.
 * @ingroup nodes_groups
 */
simaai::neat::Graph Infer(const simaai::neat::Model& model);

} // namespace simaai::neat::nodes::groups
