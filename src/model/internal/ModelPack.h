/**
 * @file
 * @ingroup model
 * @brief ModelPack: load model packs and expose pipeline fragments/stages (internal).
 */
#pragma once

#include "nodes/io/Input.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/RouteGraph.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
}

namespace simaai::neat::internal {

enum class PipelineType : std::uint8_t { Preproc, Quant, Tess, QuantTess, CastTess, Cast };

enum class ModelStage { Preprocess, MlaOnly, Postprocess, Full };

enum class ExecutionStageKind : std::uint8_t {
  Unknown = 0,
  Preproc,
  Quant,
  Tess,
  QuantTess,
  CastTess,
  Mla,
  Detess,
  DetessCast,
  DetessDequant,
  Dequant,
  BoxDecode,
  Cast,
};

struct InferenceTerminalPolicy {
  bool mla_only = false;
  std::optional<std::size_t> last_stage_index;
  std::optional<std::string> last_stage_name;
  std::optional<std::string> last_plugin_id;
  std::optional<std::string> last_processor;
};

struct ExecutionStage {
  std::size_t order_index = 0U;
  std::optional<std::size_t> mpk_plugin_index;
  std::string stage_name;
  std::string factory_name;
  std::string plugin_id;
  std::string processor;
  std::string kernel;
  ExecutionStageKind kind = ExecutionStageKind::Unknown;
};

struct ExecutionPlan {
  std::vector<ExecutionStage> pre;
  std::vector<ExecutionStage> infer;
  std::vector<ExecutionStage> post;
};

struct ModelFragment {
  std::string gst;
  std::vector<std::string> elements;
  struct StageFacts {
    std::string stage_name;
    std::size_t stage_order = 0U;
    std::optional<CompiledProcessCvuContract> processcvu_contract;
    std::optional<bool> processcvu_preproc_single_output_handoff;
    std::optional<CompiledMlaContract> mla_compiled;
    std::optional<CompiledBoxDecodeContract> boxdecode_compiled;
    std::optional<CompiledDequantContract> dequant_compiled;
    std::optional<CompiledTransportContract> transport_compiled;
    std::vector<std::pair<std::string, std::string>> fragment_properties;
  };
  std::vector<StageFacts> stage_facts;
};

/**
 * @brief Load model packs and expose stage fragments as reusable node lists.
 */
class ModelPack {
public:
  explicit ModelPack(const std::string& tar_gz);
  ModelPack(const std::string& tar_gz, const std::string& media_type, const std::string& format,
            int depth, int max_width = 0, int max_height = 0, int max_depth = 0,
            bool normalize = false, std::vector<float> mean = {}, std::vector<float> stddev = {},
            const std::string& preproc_next_cpu = {},
            PipelineType requested_pipeline_type = PipelineType::Preproc,
            const std::string& upstream_name = "decoder", int num_buffers_cvu = 4,
            int num_buffers_mla = 4, int queue_max_buffers = 0, int64_t queue_max_time_ns = -1,
            const std::string& queue_leaky = {}, const std::string& name_suffix = {},
            const InferenceTerminalPolicy& terminal_policy = {},
            bool cleanup_extracted_model_data = true);
#if defined(SIMA_WITH_OPENCV)
  ModelPack(const std::string& tar_gz, const cv::Mat& mat, int max_width = 0, int max_height = 0,
            int max_depth = 0, bool normalize = false, std::vector<float> mean = {},
            std::vector<float> stddev = {}, const std::string& preproc_next_cpu = {},
            PipelineType requested_pipeline_type = PipelineType::Preproc,
            const std::string& upstream_name = "decoder", int num_buffers_cvu = 4,
            int num_buffers_mla = 4, int queue_max_buffers = 0, int64_t queue_max_time_ns = -1,
            const std::string& queue_leaky = {}, const std::string& name_suffix = {},
            const InferenceTerminalPolicy& terminal_policy = {},
            bool cleanup_extracted_model_data = true);
#endif

  const std::string& etc_dir() const {
    return etc_dir_;
  }
  const std::optional<simaai::neat::pipeline_internal::sima::MpkContract>& mpk_contract() const {
    return mpk_contract_;
  }
  const simaai::neat::pipeline_internal::sima::RouteGraph& route_graph() const;
  PipelineType pipeline_type() const {
    return pipeline_type_;
  }
  std::string find_config_path_by_plugin(const std::string& plugin_id) const;
  std::string find_config_path_by_processor(const std::string& processor) const;

  ExecutionPlan execution_plan() const;
  std::vector<ModelFragment::StageFacts> stage_facts_for_model_stage(ModelStage stage) const;
  ModelFragment fragment(ModelStage stage) const;
  std::string backend_fragment(ModelStage stage) const;
  std::vector<std::shared_ptr<simaai::neat::Node>> to_nodes(ModelStage stage) const;

  // Infer block derived from the typed MPK execution plan.
  std::vector<std::shared_ptr<simaai::neat::Node>>
  infer_block(const std::string& upstream_name = {},
              std::shared_ptr<const ModelLineageBinding> model_lineage = nullptr) const;
  std::string apply_name_suffix(const std::string& base) const;
  bool has_terminal_policy() const;

  int num_buffers_cvu() const {
    return options_.num_buffers_cvu;
  }
  int num_buffers_mla() const {
    return options_.num_buffers_mla;
  }
  const std::string& preproc_next_cpu() const {
    return options_.preproc_next_cpu;
  }

  simaai::neat::InputOptions input_appsrc_options(bool tensor_mode) const;

  ModelPack clone_with_buffers(int num_buffers_cvu, int num_buffers_mla) const;
  ModelPack clone_with_overrides(const std::string& upstream_name,
                                 const std::string& name_suffix) const;
  void set_model_managed_stage_facts(
      std::optional<bool> processcvu_preproc_single_output_handoff,
      std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags);

private:
  struct Config {
    bool normalize = false;
    std::vector<float> mean;
    std::vector<float> stddev;

    std::string input_format; // "RGB"/"BGR"/"GRAY"/"NV12"/"IYUV"
    int input_depth = 0;
    int max_input_width = 0;
    int max_input_height = 0;
    int max_input_depth = 0;

    std::string preproc_next_cpu;
    PipelineType requested_pipeline_type = PipelineType::Preproc;
    std::string upstream_name = "decoder";

    int num_buffers_cvu = 4;
    int num_buffers_mla = 4;

    int queue_max_buffers = 0;
    int64_t queue_max_time_ns = -1;
    std::string queue_leaky;
    std::string name_suffix;
    InferenceTerminalPolicy terminal_policy;
    bool cleanup_extracted_model_data = true;
  };

  void init(const std::string& tar_gz);
  void init_from_config(const std::string& tar_gz, Config cfg);
  std::vector<ModelFragment::StageFacts> build_stage_facts(
      const std::vector<ExecutionStage>& stages,
      const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract = std::nullopt,
      ModelStage stage_context = ModelStage::Full) const;

  std::string etc_dir_;
  Config options_;
  PipelineType pipeline_type_ = PipelineType::Preproc;
  std::optional<simaai::neat::pipeline_internal::sima::MpkContract> mpk_contract_;
  mutable std::optional<simaai::neat::pipeline_internal::sima::RouteGraph> route_graph_;
  std::optional<bool> processcvu_preproc_single_output_handoff_;
  std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags_;
};

} // namespace simaai::neat::internal
