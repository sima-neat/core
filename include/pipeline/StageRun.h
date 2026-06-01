/**
 * @file
 * @ingroup pipeline
 * @brief Per-stage runtime entry points exposed for advanced/composable pipelines.
 *
 * The `simaai::neat::stages` namespace exposes the individual pipeline stages
 * (preprocessing, MLA inference, EV74 postprocessing, BoxDecode) as standalone
 * functions over `Tensor`/`Sample` lists. Most users should drive the model via
 * `simaai::neat::Run` / `Graph`; these helpers exist for tooling and tests
 * that want to invoke a single stage in isolation.
 *
 * @see Run for the higher-level orchestrator.
 * @see BoxDecodeType for decode-family selection.
 */
#pragma once

#include "pipeline/BoxDecodeType.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"

#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {
class Model;

namespace stages {

/**
 * @brief Options driving a standalone BoxDecode invocation.
 *
 * Bundles the decode family and post-decode filtering knobs (confidence threshold,
 * NMS IoU threshold, top-K cap). The decode type is required at construction; the
 * default constructor is deleted so callers cannot accidentally dispatch with an
 * `Unspecified` decode family.
 *
 * @ingroup pipeline
 * @see BoxDecodeType
 */
struct BoxDecodeOptions {
  // Decode type is explicit by construction; BoxDecode fails fast for Unspecified.
  /// @brief Construct with an explicit decode family.
  explicit BoxDecodeOptions(BoxDecodeType type) : decode_type(type) {}
  /// Default-construction is disabled — `decode_type` must be explicit.
  BoxDecodeOptions() = delete;

  BoxDecodeType decode_type;        ///< Decode family selection (YOLO, DETR, EffDet, ...).
  double detection_threshold = 0.0; ///< Minimum class score to keep a detection.
  double nms_iou_threshold = 0.0;   ///< IoU threshold used by NMS filtering.
  int top_k = 0;                    ///< Maximum detections to retain (0 = backend default).
};

/// @brief Extract the tensor list embedded in a single `Sample`.
TensorList Tensors(const simaai::neat::Sample& sample);
/// @brief Run only the preprocessing stage on raw cv::Mat inputs.
TensorList Preproc(const std::vector<cv::Mat>& inputs, const simaai::neat::Model& model);
/// @brief Run only the inference stage on already-preprocessed tensors.
TensorList Infer(const TensorList& inputs, const simaai::neat::Model& model);
/// @brief Run only the MLA leg on already-prepared tensors.
TensorList MLA(const TensorList& inputs, const simaai::neat::Model& model);
/// @brief Run only the model's postprocessing stage on inference outputs.
TensorList Postprocess(const TensorList& inputs, const simaai::neat::Model& model);
/// @brief Sample-list overload of the preprocessing stage.
Sample Preproc(const Sample& inputs, const simaai::neat::Model& model);
/// @brief Sample-list overload of the inference stage.
Sample Infer(const Sample& inputs, const simaai::neat::Model& model);
/// @brief Sample-list overload of the MLA stage.
Sample MLA(const Sample& inputs, const simaai::neat::Model& model);
/// @brief Sample-list overload of the postprocessing stage.
Sample Postprocess(const Sample& inputs, const simaai::neat::Model& model);
/**
 * @brief Run the BoxDecode stage on inference outputs.
 *
 * @param inputs Inference output samples to decode.
 * @param model Model providing anchors/strides/class metadata.
 * @param opt   Decode-family selection and filtering thresholds.
 * @return Sample list whose payloads carry decoded detections.
 */
Sample BoxDecode(const Sample& inputs, const simaai::neat::Model& model,
                 const BoxDecodeOptions& opt);

/**
 * @brief Run BoxDecode and parse each output into typed bounding boxes.
 *
 * This is the structured-result companion to `BoxDecode(...)`. It preserves the
 * list-only public API rule: pass `Sample{sample}` for a single inference
 * output, and read `results.front()` when only one result is expected.
 */
BoxDecodeResultList BoxDecodeResults(const Sample& inputs, const simaai::neat::Model& model,
                                     const BoxDecodeOptions& opt);

} // namespace stages
} // namespace simaai::neat
