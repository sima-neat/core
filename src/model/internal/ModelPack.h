/**
 * @file
 * @ingroup model
 * @brief ModelPack: load model packs and expose pipeline fragments/stages (internal).
 */
#pragma once

#include "nodes/io/Input.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/RouteGraph.h"
#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"
#include "pipeline/internal/DmabufEligibility.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#ifdef SIMA_NEAT_INTERNAL
#include "model/internal/ModelRouteRetarget.h"
#endif

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cv {
class Mat;
}

namespace simaai::neat::internal {

/// NVMe model-store candidates parsed from `/proc/mounts` content, in file order. Separated from
/// the file read so the mount-option filtering is testable without a real NVMe device.
std::vector<std::string> nvme_model_bases_from_mounts(std::istream& mounts);

/// Device class backing `path`, for load diagnostics: "NVMe", "eMMC", or the `/proc/mounts` device
/// name when it is neither. Names the runtime package path, not where inference runs.
std::string modelpack_storage_label(const std::string& path);

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
  HostTvm,
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
  std::optional<pipeline_internal::sima::static_contract::OpId> execution_op_id;
  // A compatibility stage can render several bounded physical submissions as
  // one ProcessCVU element.  The ordered semantic origins author its complete
  // typed member list; command ids retain the exact 32+remainder submission
  // proof for diagnostics and later native executor adoption.
  std::vector<pipeline_internal::sima::static_contract::OpId> execution_op_ids;
  std::vector<pipeline_internal::sima::static_contract::PhysicalCommandId>
      physical_command_ids;
  std::optional<pipeline_internal::sima::static_contract::PhysicalCohortId>
      physical_cohort_id;
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
              std::shared_ptr<const ModelLineageBinding> model_lineage = nullptr,
              bool absorb_model_managed_preproc = false) const;
  CompiledProcessCvuContract
  project_model_managed_preproc_contract(const PreprocOptions& options) const;
  // Resolve and validate the physical DMA-BUF execution contract. Descriptive
  // model APIs intentionally stay semantic-only until this boundary is crossed.
  void prepare_for_execution() const;
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

  const simaai::neat::pipeline_internal::MemoryBackendDecision& memory_backend_decision() const {
    prepare_for_execution();
    return memory_backend_decision_;
  }

  // True when the strict backend owns the complete compiler-authored model
  // command graph. RoutePlanner must not rediscover pre/post adapters around
  // MLA in this mode: those commands already live in the one execution plan.
  bool uses_model_execution_plan() const noexcept {
    return dmabuf_plan_execution_plan_.has_value();
  }

  ModelPack clone_with_buffers(int num_buffers_cvu, int num_buffers_mla) const;
  ModelPack clone_with_overrides(const std::string& upstream_name,
                                 const std::string& name_suffix) const;
  void set_model_managed_stage_facts(
      std::optional<bool> processcvu_preproc_single_output_handoff,
      std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags,
      std::vector<ExecutionStageKind> model_managed_post_kinds = {});

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
  void ensure_dmabuf_execution_plan() const;
  std::vector<ModelFragment::StageFacts> build_stage_facts(
      const std::vector<ExecutionStage>& stages,
      const std::optional<CompiledProcessCvuContract>& upstream_handoff_contract = std::nullopt,
      ModelStage stage_context = ModelStage::Full) const;

  std::string etc_dir_;
  Config options_;
  PipelineType pipeline_type_ = PipelineType::Preproc;
  std::optional<simaai::neat::pipeline_internal::sima::MpkContract> mpk_contract_;
  mutable std::optional<simaai::neat::pipeline_internal::sima::static_contract::ModelExecutionPlan>
      dmabuf_plan_execution_plan_;
  mutable std::optional<simaai::neat::pipeline_internal::sima::static_contract::FrameSlotArenaPlan>
      dmabuf_frame_arena_plan_;
  mutable std::optional<
      simaai::neat::pipeline_internal::sima::static_contract::PhysicalExecutionPlan>
      dmabuf_physical_execution_plan_;
  mutable simaai::neat::pipeline_internal::MemoryBackendDecision memory_backend_decision_;
  mutable std::optional<simaai::neat::pipeline_internal::sima::RouteGraph> route_graph_;
  std::optional<bool> processcvu_preproc_single_output_handoff_;
  std::optional<pipeline_internal::sima::ModelManagedRouteFlags> model_managed_route_flags_;
  std::vector<ExecutionStageKind> model_managed_post_kinds_;
};

} // namespace simaai::neat::internal
