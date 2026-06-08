/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaBoxDecode` Node — postprocess box decode + NMS for object-detection models.
 *
 * Runs on the EV74. Consumes the raw detection-head tensor(s) emitted by the MLA, applies
 * detector-specific box decoding (grid/anchor or raw-distance decode, score activation,
 * thresholding, top-K, and NMS), and returns surviving detections as BoxDecode payloads.
 * Place at the tail of an object-detection pipeline, or use the model-managed Graph route
 * that inserts it from the MPK postprocess contract. The decode family is enumerated in
 * `pipeline/BoxDecodeType.h`; the structured output helpers live in
 * `pipeline/DetectionTypes.h`.
 *
 * @see pipeline/BoxDecodeType.h
 * @see pipeline/DetectionTypes.h
 * @see docs/how-to/boxdecode_decode_types.md
 */
#pragma once

#include "builder/PreprocessMetaRequirement.h"
#include "builder/Node.h"
#include "builder/InputContractConfigurable.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "builder/OutputSpec.h"
#include "model/PreprocessPlan.h"
#include "pipeline/BoxDecodeType.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#endif

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
class Model;
} // namespace simaai::neat

namespace simaai::neat {

struct BoxDecodeOptionsInternal;

/**
 * @brief EV74 postprocess Node that converts detection-head tensors into object boxes.
 *
 * `SimaBoxDecode` is the postprocessing node used by object-detection graphs after MLA
 * inference. It reads the model output tensors, decodes them according to the selected
 * `BoxDecodeType`, applies confidence filtering and NMS, and emits a detection tensor that
 * can be parsed with `decode_bbox_tensor()` or rendered with `SimaRender`.
 *
 * @details
 * **When to use it.**
 *
 * - Use the `Model` constructor for normal model-pack and Graph applications. The model
 *   archive supplies the tensor layout, quantization, class count, and resize information
 *   needed by the decoder.
 * - Use the raw-geometry constructor only when you are wiring detection-head tensors
 *   yourself and can provide the original image size, model input size, and decode family.
 *
 * **Inputs.**
 *
 * The node expects the raw detection output tensors produced by the model. For MPK-backed
 * models, Neat reads the packaged contract and preserves the model's tensor order, physical
 * layout, slices, dtype, and score domain automatically. Application code normally only
 * chooses the decode family (`BoxDecodeType`) and filtering thresholds.
 *
 * **Outputs.**
 *
 * The output is a `BBOX` detection tensor. Use `decode_bbox_tensor()` or
 * `stages::BoxDecodeResults()` to turn it into `Box` records containing class id, score,
 * and coordinates in the source-image space. Use `SimaRender` downstream when you want an
 * annotated video/image stream.
 *
 * **Supported families.**
 *
 * Supported decode families include YOLO, YOLOv5/v7/v8/v9/v10 detection and segmentation
 * variants, YOLOv8 pose, YOLO26 detection/pose/segmentation, YOLOv6, YOLOX, DETR,
 * EfficientDet, RCNN stage 1, and CenterNet. `BoxDecodeType::Unspecified` is only a
 * sentinel and fails before runtime.
 *
 * **Score and layout notes.**
 *
 * Some models output probabilities; others output logits that must be activated before
 * thresholding. Model packs carry this information when available. If you are manually
 * wiring tensors, choose the decode type and decode option that match your exported head
 * format. Do not infer correctness from tensor rank alone: sliced, padded, packed, and
 * dense outputs can have the same logical shape while requiring different handling. The
 * model-aware path handles these details for supported model packs.
 *
 * @see pipeline/BoxDecodeType.h
 * @see pipeline/DetectionTypes.h
 * @see SimaRender
 *
 * @ingroup nodes_sima
 */
class SimaBoxDecode final : public Node,
                            public InputContractConfigurable,
                            public OutputSpecProvider,
                            public PreprocessMetaRequirementProvider,
                            public NodeContractProvider,
                            public NodeContractConfigurable {
public:
  /**
   * @brief Construct from raw geometry — for graphs without a bound `Model`.
   *
   * @param decode_type          Decoder variant (see `BoxDecodeType`).
   * @param detection_threshold  Score threshold; boxes below are dropped (`0.0` = use default).
   * @param nms_iou_threshold    NMS IoU threshold (`0.0` = use default).
   * @param top_k                Max boxes to keep after NMS (`0` = unlimited / variant default).
   * @param element_name         Optional GStreamer element name.
   * @param original_width       Width of the unscaled source frame (used for box rescaling).
   * @param original_height      Height of the unscaled source frame.
   * @param model_width          Input width the model was trained for.
   * @param model_height         Input height the model was trained for.
   * @param decode_type_option   Decoder sub-variant selector (`Auto` to defer to the model).
   */
  explicit SimaBoxDecode(BoxDecodeType decode_type, double detection_threshold = 0.0,
                         double nms_iou_threshold = 0.0, int top_k = 0,
                         const std::string& element_name = "", int original_width = 0,
                         int original_height = 0, int model_width = 0, int model_height = 0,
                         BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
  /**
   * @brief Construct from a bound `Model` — pulls geometry and routing flags from the model.
   *
   * @param model                Source model (provides input shape, routing hints, etc.).
   * @param decode_type          Decoder variant (see `BoxDecodeType`).
   * @param detection_threshold  Score threshold (`0.0` = use default).
   * @param nms_iou_threshold    NMS IoU threshold (`0.0` = use default).
   * @param top_k                Max boxes to keep after NMS.
   * @param element_name         Optional GStreamer element name.
   * @param route_tess_needed    Override: does the route need a tess stage upstream?
   * @param route_quant_needed   Override: does the route need a quant stage upstream?
   * @param original_width       Width of the unscaled source frame.
   * @param original_height      Height of the unscaled source frame.
   * @param model_width          Optional override: model-input width. `0` = derive from the
   *                             model pack (default). When set, the value flows through to the
   *                             decoder kernel's spatial knobs; the model-managed
   *                             `compiled_contract` still drives quant / decode-family / tensor
   *                             layout, so callers can override geometry without falling off the
   *                             model-managed contract path.
   * @param model_height         Optional override: model-input height. Same semantics as
   *                             `model_width`. Both must be set together (or both zero).
   * @param resize_mode_override Optional override: explicit preprocess resize mode
   *                             (`Stretch`/`Letterbox`/`Crop`). Use when running the model
   *                             without an upstream `Preproc` stage and the per-buffer
   *                             `preproc_resize_mode` meta isn't being written by an
   *                             upstream element. When set, the contract drops
   *                             `preproc_resize_mode` from the required-meta list so buffers
   *                             flow through cleanly; otherwise the value is sourced from
   *                             per-buffer `GstSimaaiPreprocessMeta` as before.
   * @param decode_type_option   Decoder sub-variant selector.
   */
  explicit SimaBoxDecode(const simaai::neat::Model& model, BoxDecodeType decode_type,
                         double detection_threshold = 0.0, double nms_iou_threshold = 0.0,
                         int top_k = 0, const std::string& element_name = "",
                         std::optional<bool> route_tess_needed = std::nullopt,
                         std::optional<bool> route_quant_needed = std::nullopt,
                         int original_width = 0, int original_height = 0, int model_width = 0,
                         int model_height = 0,
                         std::optional<ResizeMode> resize_mode_override = std::nullopt,
                         BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
#ifdef SIMA_NEAT_INTERNAL
  /// Internal constructor: takes a pre-extracted static contract (used by the planner).
  explicit SimaBoxDecode(
      const pipeline_internal::sima::BoxDecodeStaticContract& contract, BoxDecodeType decode_type,
      double detection_threshold = 0.0, double nms_iou_threshold = 0.0, int top_k = 0,
      const std::string& element_name = {},
      const std::vector<std::string>& required_preprocess_meta_fields = {},
      std::optional<pipeline_internal::sima::ModelManagedRouteFlags> route_flags = std::nullopt,
      std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> model_semantics =
          std::nullopt,
      std::optional<bool> expect_resize = std::nullopt,
      std::optional<bool> expect_normalize = std::nullopt,
      std::optional<bool> expect_quantize = std::nullopt,
      std::optional<bool> expect_tessellate = std::nullopt, int original_width = 0,
      int original_height = 0, int model_width = 0, int model_height = 0,
      BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
#endif

  /// Type label for this Node kind.
  std::string kind() const override {
    return "SimaBoxDecode";
  }
  /// Whether the Node negotiates static or dynamic caps.
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  /// GStreamer fragment this Node emits.
  std::string backend_fragment(int node_index) const override;
  /// Deterministic element names this Node will create.
  std::vector<std::string> element_names(int node_index) const override;
  /// Negotiated downstream caps produced by this Node.
  OutputSpec output_spec(const OutputSpec& input) const override;
  /// Preprocess metadata fields this Node requires upstream.
  std::optional<PreprocessMetaRequirement> preprocess_meta_requirement() const override;
  /// Structural contract definition for this Node.
  NodeContractDefinition contract_definition() const override;
  /// Compile this Node's contract from the given input.
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* err) const override;
  /// Apply a compiled contract back into this Node.
  void apply_compiled_contract(const CompiledNodeContract& contract, std::string* err) override;
  /// Apply an input contract from upstream.
  void apply_input_contract(const InputContract& contract, std::string* err) override;

#ifdef SIMA_NEAT_INTERNAL
  const std::string& factory_internal() const;
  BoxDecodeType decode_type_internal() const;
  double detection_threshold_internal() const;
  double nms_iou_threshold_internal() const;
  int top_k_internal() const;
  int original_width_internal() const;
  int original_height_internal() const;
  BoxDecodeTypeOption decode_type_option_internal() const;
  const std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics>&
  model_semantics_internal() const;
  const std::optional<pipeline_internal::sima::ModelManagedRouteFlags>&
  model_route_flags_internal() const;
  const std::optional<pipeline_internal::sima::BoxDecodeStaticContract>&
  model_static_contract_internal() const;
  const std::vector<std::string>& required_preprocess_meta_fields_internal() const;
  const std::shared_ptr<const internal::ModelLineageBinding>&
  model_lineage_binding_internal() const;
  internal::RequestedPostRouteKind requested_post_route_internal() const;
#endif

private:
  std::unique_ptr<BoxDecodeOptionsInternal> opt_;
  std::optional<InputContract> input_contract_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for `SimaBoxDecode` from raw geometry — see the class constructor docs.
std::shared_ptr<simaai::neat::Node>
SimaBoxDecode(BoxDecodeType decode_type, double detection_threshold = 0.0,
              double nms_iou_threshold = 0.0, int top_k = 0, const std::string& element_name = "",
              int original_width = 0, int original_height = 0, int model_width = 0,
              int model_height = 0,
              BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
/// Convenience factory for `SimaBoxDecode` from a bound `Model` — see the class constructor docs.
std::shared_ptr<simaai::neat::Node>
SimaBoxDecode(const simaai::neat::Model& model, BoxDecodeType decode_type,
              double detection_threshold = 0.0, double nms_iou_threshold = 0.0, int top_k = 0,
              const std::string& element_name = "",
              std::optional<bool> route_tess_needed = std::nullopt,
              std::optional<bool> route_quant_needed = std::nullopt, int original_width = 0,
              int original_height = 0, int model_width = 0, int model_height = 0,
              std::optional<ResizeMode> resize_mode_override = std::nullopt,
              BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
#ifdef SIMA_NEAT_INTERNAL
/// Internal-only factory used by the route planner with a pre-extracted static contract.
std::shared_ptr<simaai::neat::Node> SimaBoxDecode(
    const pipeline_internal::sima::BoxDecodeStaticContract& contract, BoxDecodeType decode_type,
    double detection_threshold = 0.0, double nms_iou_threshold = 0.0, int top_k = 0,
    const std::string& element_name = {},
    const std::vector<std::string>& required_preprocess_meta_fields = {},
    std::optional<pipeline_internal::sima::ModelManagedRouteFlags> route_flags = std::nullopt,
    std::optional<pipeline_internal::sima::ModelBoxdecodeSemantics> model_semantics = std::nullopt,
    std::optional<bool> expect_resize = std::nullopt,
    std::optional<bool> expect_normalize = std::nullopt,
    std::optional<bool> expect_quantize = std::nullopt,
    std::optional<bool> expect_tessellate = std::nullopt, int original_width = 0,
    int original_height = 0, int model_width = 0, int model_height = 0,
    BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto);
#endif
} // namespace simaai::neat::nodes
