/**
 * @file
 * @ingroup nodes_sima
 * @brief `SimaBoxDecode` Node — postprocess box decode + NMS for object-detection models.
 *
 * Runs on the EV74. Consumes the raw detection-head tensor(s) emitted by the MLA, applies
 * detector-specific box decoding (anchor decode, sigmoid/softmax over class scores), and
 * returns surviving boxes after non-maximum suppression. Place at the tail of an
 * object-detection pipeline; the variant family is enumerated in `pipeline/BoxDecodeType.h`.
 *
 * @see pipeline/BoxDecodeType.h
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
 * @brief EV74 postprocess Node that decodes detection-head tensors into object boxes (with NMS).
 *
 * Pick the constructor that matches what your application has on hand: the bare-parameter
 * form for raw inputs, or the `Model`-aware form when the model carries the geometry and
 * route flags. Decoder variants live in `BoxDecodeType` — pick the one matching your model
 * family (YOLOv5, SSD, RetinaNet, etc.).
 *
 * @see pipeline/BoxDecodeType.h
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
