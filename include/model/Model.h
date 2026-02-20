/**
 * @file
 * @ingroup model
 * @brief Model: user-facing model pipeline builder.
 */
#pragma once

#include "builder/NodeGroup.h"
#include "nodes/io/Input.h"
#include "pipeline/Run.h"
#include "pipeline/TensorSpec.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

namespace simaai::neat {

namespace internal {
struct ModelAccess;
} // namespace internal

using TensorSpec = TensorConstraint;

class Model {
public:
  struct PreprocConfig {
    std::optional<int> input_width;
    std::optional<int> input_height;
    std::optional<int> output_width;
    std::optional<int> output_height;
    std::optional<int> scaled_width;
    std::optional<int> scaled_height;
    std::optional<std::string> input_img_type;
    std::optional<std::string> output_img_type;
    std::optional<bool> normalize;
    std::optional<bool> aspect_ratio;
    std::optional<std::array<float, 3>> channel_mean;
    std::optional<std::array<float, 3>> channel_stddev;
    std::optional<std::string> scaling_type;
    std::optional<std::string> padding_type;
  };

  struct InferenceTerminalPolicy {
    bool mla_only = false;
    std::optional<std::size_t> last_stage_index;
    std::optional<std::string> last_stage_name;
    std::optional<std::string> last_plugin_id;
    std::optional<std::string> last_processor;
  };

  struct Options {
    // Input format & dims
    std::string media_type = "video/x-raw";
    std::string format = "RGB";

    // Dynamic input limits for appsrc validation + pool sizing.
    // If <= 0, defaults are applied (HD + depth 3).
    int input_max_width = 0;
    int input_max_height = 0;
    int input_max_depth = 0;

    // Preproc (canonical path)
    PreprocConfig preproc;

    // Postproc / detection
    std::string decode_type;
    float score_threshold = 0.0f;
    float nms_iou_threshold = 0.0f;
    int top_k = 0;
    int original_width = 0;
    int original_height = 0;

    // Naming / wiring
    std::string upstream_name = "decoder";
    std::string name_suffix;

    InferenceTerminalPolicy inference_terminal;
  };

  struct SessionOptions {
    bool include_appsrc = true;
    bool include_appsink = true;
    std::string upstream_name;
    std::string name_suffix;
    std::string buffer_name;
  };

  enum class Stage { Preprocess, Inference, Postprocess, Full };

  explicit Model(const std::string& mpk_path);
  explicit Model(const std::string& mpk_path, const Options& opt);

  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  ~Model();

  // Stage composition
  simaai::neat::NodeGroup preprocess() const;
  simaai::neat::NodeGroup inference() const;
  simaai::neat::NodeGroup postprocess() const;
  simaai::neat::NodeGroup session() const;
  simaai::neat::NodeGroup session(SessionOptions opt) const;

  // Introspection
  TensorSpec input_spec() const;
  TensorSpec output_spec() const;
  std::unordered_map<std::string, std::string> metadata() const;

  // Optional: for graph/advanced usage
  NodeGroup fragment(Stage stage) const;
  std::string backend_fragment(Stage stage) const;

  simaai::neat::InputOptions input_appsrc_options(bool tensor_mode) const;
  std::string find_config_path_by_plugin(const std::string& plugin_id) const;
  std::string find_config_path_by_processor(const std::string& processor) const;
  std::string infer_output_name() const;

  class Runner {
  public:
    Runner() = default;
    explicit Runner(simaai::neat::Run run);

    explicit operator bool() const noexcept;
#if defined(SIMA_WITH_OPENCV)
    bool push(const cv::Mat& input);
#endif
    bool push(const simaai::neat::Tensor& input);
    bool push(const simaai::neat::Sample& input);
    std::optional<simaai::neat::Sample> pull(int timeout_ms = -1);
#if defined(SIMA_WITH_OPENCV)
    simaai::neat::Sample run(const cv::Mat& input, int timeout_ms = -1);
#endif
    simaai::neat::Sample run(const simaai::neat::Tensor& input, int timeout_ms = -1);
    simaai::neat::Sample run(const simaai::neat::Sample& input, int timeout_ms = -1);
    int warmup(const simaai::neat::Tensor& input, int warm = -1, int timeout_ms = -1);
    void close();

  private:
    simaai::neat::Run run_{};
  };

private:
  static const SessionOptions& default_session_options();
  static const simaai::neat::RunOptions& default_run_options();

public:
  Runner build();
  Runner build(const SessionOptions& opt);
  Runner build(const simaai::neat::Tensor& input,
               const SessionOptions& opt = default_session_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
  Runner build(const simaai::neat::Sample& input,
               const SessionOptions& opt = default_session_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
#if defined(SIMA_WITH_OPENCV)
  Runner build(const cv::Mat& input, const SessionOptions& opt = default_session_options(),
               const simaai::neat::RunOptions& run_opt = default_run_options());
#endif

  // Execution
  simaai::neat::Sample run(const simaai::neat::Tensor& input, int timeout_ms = -1);
  simaai::neat::Sample run(const std::vector<Tensor>& inputs, int timeout_ms = -1);
  simaai::neat::Sample run(const simaai::neat::Sample& input, int timeout_ms = -1);
#if defined(SIMA_WITH_OPENCV)
  simaai::neat::Sample run(const cv::Mat& input, int timeout_ms = -1);
#endif

private:
  friend struct internal::ModelAccess;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace simaai::neat
