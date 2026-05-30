#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/InputPlanner.h"
#include "model/internal/ModelPack.h"
#include "model/internal/ModelParser.h"
#include "model/internal/ModelRouteRetarget.h"
#include "pipeline/internal/sima/ProcessCvuFamily.h"
#include "pipeline/internal/sima/RouteGraph.h"
#include "model/internal/RoutePlanner.h"

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/Dequant.h"
#include "nodes/sima/Detess.h"
#include "nodes/sima/DetessCast.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/Tess.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/InputRouteProcessor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/BoxDecodeTypeUtils.h"
#include "pipeline/internal/sima/MlaStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/TensorMath.h"

#include <nlohmann/json.hpp>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace simaai::neat {
namespace {

simaai::neat::GraphOptions
route_options_from_model_route_options(const Model::RouteOptions& opt,
                                       const Model::Options* model_opt = nullptr);

struct SliceDims {
  int d = 1, h = 0, w = 0, c = 0;
};
SliceDims dims_from_slice_shape(const std::vector<std::int64_t>& s) {
  SliceDims r;
  if (s.size() == 4) {
    r.d = s[0];
    r.h = s[1];
    r.w = s[2];
    r.c = s[3];
  } else if (s.size() == 3) {
    r.d = 1;
    r.h = s[0];
    r.w = s[1];
    r.c = s[2];
  }
  return r;
}

pipeline_internal::sima::ModelManagedRouteFlags
convert_model_managed_route_flags(const internal::SessionRoutePlan::ModelManagedRouteFlags& src) {
  pipeline_internal::sima::ModelManagedRouteFlags flags;
  flags.quant_needed = src.quant_needed;
  flags.tess_needed = src.tess_needed;
  flags.pre_cast_needed = src.pre_cast_needed;
  flags.quant_contract_required = src.quant_contract_required;
  flags.include_pre_stage = src.include_pre_stage;
  flags.boxdecode_selected = src.boxdecode_selected;
  return flags;
}
using pipeline_internal::env_bool;
using pipeline_internal::env_int;
using pipeline_internal::upper_copy;
namespace rendered_stage_query = pipeline_internal::rendered_stage_query;

struct InputInfo;

internal::InferenceTerminalPolicy
to_internal_terminal_policy(const Model::InferenceTerminalPolicy& policy) {
  internal::InferenceTerminalPolicy out;
  out.mla_only = policy.mla_only;
  out.last_stage_index = policy.last_stage_index;
  out.last_stage_name = policy.last_stage_name;
  out.last_plugin_id = policy.last_plugin_id;
  out.last_processor = policy.last_processor;
  return out;
}

int default_depth_for_format(const std::string& fmt) {
  return pipeline_internal::default_depth_for_image_format(fmt, -1);
}

int clamp_positive_contract_dim(std::int64_t value) {
  if (value <= 0) {
    return 0;
  }
  if (value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

// MPK logical shapes are emitted in canonical geometry order, not as a tensor layout
// promise.  Keep this intentionally narrow: only use it for rank-3/rank-4 logical
// model-input geometry when the contract no longer carries an explicit layout token.
stages::TensorDims dims_from_canonical_mpk_logical_shape(const std::vector<std::int64_t>& shape) {
  stages::TensorDims dims;
  if (shape.size() >= 4U) {
    dims.height = clamp_positive_contract_dim(shape[shape.size() - 3U]);
    dims.width = clamp_positive_contract_dim(shape[shape.size() - 2U]);
    dims.depth = clamp_positive_contract_dim(shape[shape.size() - 1U]);
  } else if (shape.size() == 3U) {
    dims.height = clamp_positive_contract_dim(shape[0]);
    dims.width = clamp_positive_contract_dim(shape[1]);
    dims.depth = clamp_positive_contract_dim(shape[2]);
  }
  return dims;
}

stages::TensorDims dims_from_mla_logical_contract_shape(const std::vector<std::int64_t>& shape,
                                                        TensorLayout layout) {
  if (layout == TensorLayout::Unknown) {
    const stages::TensorDims canonical = dims_from_canonical_mpk_logical_shape(shape);
    if (canonical.width > 0 && canonical.height > 0 && canonical.depth > 0) {
      return canonical;
    }
  }
  return rendered_stage_query::tensor_dims_projection_from_contract_shape(shape, layout);
}

bool preproc_output_dtype_is_quantized(const std::string& raw) {
  const std::string token = upper_copy(raw);
  return token == "INT8" || token == "EVXX_INT8" || token == "UINT8" || token == "U8";
}

std::string route_mla_input_dtype_from_diagnostics(const internal::PreprocessPlannerResult& plan);
const pipeline_internal::sima::MpkPluginIoContract*
find_pre_mla_processcvu_stage(const internal::ModelPack& pack,
                              std::initializer_list<const char*> preferred_families);
PreprocOptions make_preproc_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, const std::string& upstream_name, bool sync);
QuantOptions make_quant_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, bool sync, const internal::OrderedRouteOp* route_op = nullptr,
    bool allow_multi_io_contract = false);
TessOptions make_tess_options_from_typed_adapter(const Model& model,
                                                 const internal::PreprocessPlannerResult& plan,
                                                 const InputInfo* input,
                                                 const std::string& element_name, bool sync,
                                                 const internal::OrderedRouteOp* route_op = nullptr,
                                                 bool allow_multi_io_contract = false);
QuantTessOptions make_quanttess_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, bool sync, const internal::OrderedRouteOp* route_op = nullptr,
    bool allow_multi_io_contract = false);
CastTessOptions make_casttess_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, bool sync, const internal::OrderedRouteOp* route_op = nullptr,
    bool allow_multi_io_contract = false);
CastOptions make_cast_options_from_typed_adapter(const Model* model,
                                                 const std::string& element_name,
                                                 CastDirection direction,
                                                 internal::ModelLineageStageRole stage_role,
                                                 const internal::OrderedRouteOp* route_op = nullptr,
                                                 bool allow_multi_io_contract = false);
DetessCastOptions make_detesscast_options_from_typed_adapter(const Model& model,
                                                             const std::string& element_name,
                                                             bool sync);

void warn_no_warmup_once() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    std::printf(
        "[WARN] Model::Runner::warmup: warm=0; throughput stability may vary without warmup.\n");
  });
}

void emit_model_planner_messages(const VerboseOptions& verbose,
                                 const std::vector<std::string>& warnings) {
  if (warnings.empty()) {
    return;
  }
  if (pipeline_internal::ux::should_emit_topic(verbose,
                                               pipeline_internal::ux::VerboseTopic::Planner)) {
    for (const auto& warn : warnings) {
      std::fprintf(stderr, "[WARN] Model preprocess planner: %s\n", warn.c_str());
    }
    return;
  }
  if (verbose.level == VerbosityLevel::Quiet) {
    return;
  }
  std::fprintf(stderr,
               "[WARN] Model preprocess planner produced %zu advisory message%s; "
               "set Model::Options.verbose.level=Verbose or Model::Options.verbose.planner=true "
               "to inspect details.\n",
               warnings.size(), (warnings.size() == 1U) ? "" : "s");
}

int route_op_exact_stage_score(const pipeline_internal::sima::MpkPluginIoContract& stage,
                               internal::ExecutionStageKind want_kind) {
  const auto graph_kind = pipeline_internal::sima::canonical_route_graph_kernel_kind(
      !stage.kernel.empty() ? stage.kernel : stage.name);
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (want_kind) {
  case internal::ExecutionStageKind::Quant:
    switch (graph_kind) {
    case GraphKind::Quant:
      return 4;
    case GraphKind::QuantTess:
      return 3;
    case GraphKind::Preproc:
      return 1;
    default:
      return -1;
    }
  case internal::ExecutionStageKind::Tess:
    switch (graph_kind) {
    case GraphKind::Tess:
      return 4;
    case GraphKind::QuantTess:
      return 3;
    case GraphKind::Preproc:
      return 1;
    default:
      return -1;
    }
  case internal::ExecutionStageKind::QuantTess:
    switch (graph_kind) {
    case GraphKind::QuantTess:
      return 5;
    case GraphKind::Tess:
      return 4;
    case GraphKind::Quant:
      return 3;
    case GraphKind::Preproc:
      return 1;
    default:
      return -1;
    }
  case internal::ExecutionStageKind::CastTess:
    switch (graph_kind) {
    case GraphKind::CastTess:
      return 5;
    case GraphKind::Tess:
      return 4;
    case GraphKind::Cast:
      return 3;
    case GraphKind::Preproc:
      return 1;
    default:
      return -1;
    }
  case internal::ExecutionStageKind::Cast:
    return graph_kind == GraphKind::Cast ? 4 : -1;
  case internal::ExecutionStageKind::Detess:
    return graph_kind == GraphKind::Detess ? 4 : -1;
  case internal::ExecutionStageKind::DetessCast:
    switch (graph_kind) {
    case GraphKind::DetessCast:
      return 5;
    case GraphKind::Detess:
      return 4;
    case GraphKind::Cast:
      return 3;
    default:
      return -1;
    }
  case internal::ExecutionStageKind::DetessDequant:
    return graph_kind == GraphKind::DetessDequant ? 4 : -1;
  case internal::ExecutionStageKind::Dequant:
    return graph_kind == GraphKind::Dequantize ? 4 : -1;
  case internal::ExecutionStageKind::Preproc:
    return graph_kind == GraphKind::Preproc ? 4 : -1;
  case internal::ExecutionStageKind::Mla:
  case internal::ExecutionStageKind::BoxDecode:
  case internal::ExecutionStageKind::Unknown:
    return -1;
  }
  return -1;
}

std::optional<std::string>
resolve_exact_route_stage_name_or_id(const internal::ModelPack& pack,
                                     internal::ExecutionStageKind want_kind,
                                     const internal::OrderedRouteOp* route_op) {
  if (!route_op || !pack.mpk_contract().has_value()) {
    return std::nullopt;
  }
  auto stage_token = [](const pipeline_internal::sima::MpkPluginIoContract& stage) {
    return !stage.name.empty() ? stage.name : stage.plugin_id;
  };

  const auto& contract = *pack.mpk_contract();
  std::vector<const pipeline_internal::sima::MpkPluginIoContract*> direct_matches;
  for (const std::string& candidate : {route_op->plugin_name, route_op->plugin_id}) {
    if (candidate.empty()) {
      continue;
    }
    if (const auto* stage = pipeline_internal::sima::get_stage_io_contract(contract, candidate)) {
      if (route_op_exact_stage_score(*stage, want_kind) >= 0) {
        direct_matches.push_back(stage);
      }
    }
  }
  if (!direct_matches.empty()) {
    std::sort(direct_matches.begin(), direct_matches.end(), [&](const auto* lhs, const auto* rhs) {
      return route_op_exact_stage_score(*lhs, want_kind) >
             route_op_exact_stage_score(*rhs, want_kind);
    });
    const std::string token = stage_token(*direct_matches.front());
    if (!token.empty()) {
      return token;
    }
  }

  if (route_op->sequence < 0) {
    return std::nullopt;
  }
  std::vector<const pipeline_internal::sima::MpkPluginIoContract*> seq_matches;
  for (const auto& stage : contract.plugins) {
    if (stage.sequence != route_op->sequence) {
      continue;
    }
    if (route_op_exact_stage_score(stage, want_kind) >= 0) {
      seq_matches.push_back(&stage);
    }
  }
  if (seq_matches.empty()) {
    return std::nullopt;
  }
  std::sort(seq_matches.begin(), seq_matches.end(), [&](const auto* lhs, const auto* rhs) {
    return route_op_exact_stage_score(*lhs, want_kind) >
           route_op_exact_stage_score(*rhs, want_kind);
  });
  const std::string token = stage_token(*seq_matches.front());
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(
        stderr,
        "[typed-adapter] resolved exact route stage kind=%d route{name=%s id=%s seq=%d} -> %s\n",
        static_cast<int>(want_kind), route_op->plugin_name.c_str(), route_op->plugin_id.c_str(),
        route_op->sequence, token.empty() ? "<empty>" : token.c_str());
  }
  return token.empty() ? std::nullopt : std::optional<std::string>(token);
}
bool pipeline_requires_tensor_input(const internal::PreprocessPlannerResult& plan) {
  // Route-planned pre-chain is the canonical ingress contract. Adapter-only pre
  // stages (quant/tess/quanttess) require tensor appsrc; preproc-first chains
  // ingest video/x-raw and convert before adapters.
  if (plan.session_route_plan.include_pre_stage && !plan.session_route_plan.pre_chain.empty()) {
    switch (plan.session_route_plan.pre_chain.front()) {
    case internal::SessionPreStageOp::Quant:
    case internal::SessionPreStageOp::Tess:
    case internal::SessionPreStageOp::QuantTess:
    case internal::SessionPreStageOp::Cast:
      return true;
    case internal::SessionPreStageOp::Preproc:
      return false;
    }
  }
  const std::string media = upper_copy(plan.modelpack_media_type);
  return media == "APPLICATION/VND.SIMAAI.TENSOR";
}

internal::PreprocessContractFlags
resolve_preprocess_contract_flags(const internal::PreprocessPlannerResult& plan) {
  internal::PreprocessContractFlags flags;
  flags.quant_needed = plan.session_route_plan.preproc_context.pre_quant_needed;
  flags.tess_needed = plan.session_route_plan.preproc_context.pre_tess_needed;
  for (const auto stage : plan.session_route_plan.pre_chain) {
    switch (stage) {
    case internal::SessionPreStageOp::Quant:
    case internal::SessionPreStageOp::QuantTess:
      flags.quant_needed = true;
      break;
    case internal::SessionPreStageOp::Tess:
    case internal::SessionPreStageOp::CastTess:
      flags.tess_needed = true;
      break;
    case internal::SessionPreStageOp::Preproc:
    case internal::SessionPreStageOp::Cast:
      break;
    }
  }
  return flags;
}

std::string normalize_processcvu_dtype_token(std::string raw, const std::string& fallback) {
  raw = upper_copy(raw);
  if (raw.empty()) {
    return fallback;
  }
  if (raw.find("BFLOAT16") != std::string::npos || raw.find("BF16") != std::string::npos) {
    return "BF16";
  }
  if (raw.find("FLOAT32") != std::string::npos || raw.find("FP32") != std::string::npos) {
    return "FP32";
  }
  if (raw.find("INT8") != std::string::npos) {
    return "INT8";
  }
  if (raw.find("UINT8") != std::string::npos) {
    return "UINT8";
  }
  if (raw.find("INT16") != std::string::npos) {
    return "INT16";
  }
  if (raw.find("UINT16") != std::string::npos) {
    return "UINT16";
  }
  return raw;
}

CompiledProcessCvuContract require_model_managed_preadapter_contract(
    const internal::ModelPack& pack, internal::ExecutionStageKind kind, const char* stage_label,
    const internal::OrderedRouteOp* route_op = nullptr) {
  const auto pre_plan = pack.execution_plan().pre;
  const auto stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
  if (pre_plan.size() != stage_facts.size()) {
    throw std::runtime_error(
        "Model-managed pre-process stage facts are out of sync with execution plan (plan_count=" +
        std::to_string(pre_plan.size()) + ", fact_count=" + std::to_string(stage_facts.size()) +
        ")");
  }

  if (const auto exact_stage = resolve_exact_route_stage_name_or_id(pack, kind, route_op);
      exact_stage.has_value() && pack.mpk_contract().has_value()) {
    try {
      return pipeline_internal::sima::stagesemantics::
          build_processcvu_mpk_preadapter_compiled_contract_for_stage_kind(*pack.mpk_contract(),
                                                                           kind, *exact_stage);
    } catch (const std::exception& ex) {
      if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
        std::fprintf(stderr,
                     "[typed-adapter] exact route stage build failed kind=%d stage=%s error=%s\n",
                     static_cast<int>(kind), exact_stage->c_str(), ex.what());
      }
    }
  }

  auto find_exact_stage_fact = [&]() -> std::optional<CompiledProcessCvuContract> {
    if (!route_op || (!route_op->plugin_name.empty() || !route_op->plugin_id.empty()) == false) {
      return std::nullopt;
    }

    std::unordered_set<std::string> wanted_names;
    const std::array<std::string, 2> exact_candidates = {route_op->plugin_name,
                                                         route_op->plugin_id};
    for (const auto& candidate : exact_candidates) {
      if (candidate.empty()) {
        continue;
      }
      wanted_names.insert(candidate);
      if (pack.mpk_contract().has_value()) {
        if (const auto* mpk_stage =
                pipeline_internal::sima::get_stage_io_contract(*pack.mpk_contract(), candidate);
            mpk_stage != nullptr) {
          if (!mpk_stage->name.empty()) {
            wanted_names.insert(mpk_stage->name);
          }
          if (!mpk_stage->plugin_id.empty()) {
            wanted_names.insert(mpk_stage->plugin_id);
          }
        }
      }
    }
    if (wanted_names.empty()) {
      return std::nullopt;
    }

    std::optional<CompiledProcessCvuContract> matched;
    for (std::size_t index = 0; index < pre_plan.size(); ++index) {
      if (pre_plan[index].kind != kind) {
        continue;
      }
      const auto& fact = stage_facts[index];
      if (!fact.processcvu_contract.has_value()) {
        continue;
      }
      bool sequence_match = false;
      if (route_op->sequence >= 0 && pack.mpk_contract().has_value() &&
          pre_plan[index].mpk_plugin_index.has_value() &&
          *pre_plan[index].mpk_plugin_index < pack.mpk_contract()->plugins.size()) {
        sequence_match = pack.mpk_contract()->plugins[*pre_plan[index].mpk_plugin_index].sequence ==
                         route_op->sequence;
      }
      const bool name_match = wanted_names.count(pre_plan[index].stage_name) > 0U ||
                              wanted_names.count(pre_plan[index].plugin_id) > 0U ||
                              wanted_names.count(fact.stage_name) > 0U;
      if (!(name_match || sequence_match)) {
        continue;
      }
      if (matched.has_value()) {
        throw std::runtime_error(
            std::string("Model-managed ") + (stage_label ? stage_label : "processcvu") +
            " stage resolved multiple canonical compiled contracts for route op '" +
            (!route_op->plugin_name.empty() ? route_op->plugin_name : route_op->plugin_id) +
            "' (matched stage='" + pre_plan[index].stage_name +
            "', index=" + std::to_string(index) + "); ensure only one stage matches the route op");
      }
      matched = *fact.processcvu_contract;
    }
    return matched;
  };

  if (const auto matched = find_exact_stage_fact(); matched.has_value()) {
    return *matched;
  }

  if (route_op && (!route_op->plugin_name.empty() || !route_op->plugin_id.empty())) {
    if (pack.mpk_contract().has_value()) {
      const std::array<std::string, 2> exact_candidates = {route_op->plugin_name,
                                                           route_op->plugin_id};
      for (const auto& candidate : exact_candidates) {
        if (candidate.empty()) {
          continue;
        }
        try {
          return pipeline_internal::sima::stagesemantics::
              build_processcvu_mpk_preadapter_compiled_contract_for_stage_kind(*pack.mpk_contract(),
                                                                               kind, candidate);
        } catch (const std::exception&) {
        }
      }
    }
  }

  const auto* mpk_stage = find_pre_mla_processcvu_stage(
      pack, kind == internal::ExecutionStageKind::Quant
                ? std::initializer_list<const char*>{"quant", "quanttess", "preproc"}
            : kind == internal::ExecutionStageKind::Tess
                ? std::initializer_list<const char*>{"tess", "quanttess", "preproc"}
            : kind == internal::ExecutionStageKind::CastTess
                ? std::initializer_list<const char*>{"casttess", "tess", "preproc"}
            : kind == internal::ExecutionStageKind::Cast
                ? std::initializer_list<const char*>{"cast"}
                : std::initializer_list<const char*>{"quanttess", "preproc"});

  // Locate the (single) plan entry of the requested kind. For fan-in routes
  // the MPK exposes separate per-ingress plugins (e.g. cast_0+tess_0 / cast_1+
  // tess_1) but the route planner fuses them into ONE plan stage of kind
  // CastTess (or QuantTess). In that case `mpk_stage->name` will be a raw
  // per-ingress plugin name that does NOT match the fused fact's stage_name —
  // matching by plan kind instead is unambiguous because the plan has exactly
  // one stage per kind on the pre side.
  std::optional<CompiledProcessCvuContract> matched;
  for (std::size_t i = 0; i < pre_plan.size(); ++i) {
    if (pre_plan[i].kind != kind) {
      continue;
    }
    const auto& fact = stage_facts[i];
    if (!fact.processcvu_contract.has_value()) {
      continue;
    }
    if (mpk_stage && fact.stage_name != mpk_stage->name) {
      // Per the function comment above, the plan has exactly one stage per
      // kind on the pre side, so the kind filter already guarantees
      // uniqueness.  The historic check below was originally added to gate
      // multi-IO fan-in facts through (their stage_name is a fused virtual
      // name that doesn't match any MPK plugin), but the same naming
      // mismatch also happens for single-IO routes where the execution
      // stage_name (e.g. "casttess") differs from the MPK plugin name
      // (e.g. "tessellate_cast_0_MLA_0/...").  Skipping the fact in that
      // case caused BF16 mpk pre-MLA contract resolution to fail.  Now
      // accept the fact regardless of name as long as kind matched and a
      // contract was built.
      (void)fact;
    }
    if (matched.has_value()) {
      throw std::runtime_error(std::string("Model-managed ") +
                               (stage_label ? stage_label : "processcvu") +
                               " stage resolved multiple canonical compiled contracts"
                               " (duplicate at stage='" +
                               fact.stage_name + "'); ensure only one pre-process stage matches");
    }
    matched = *fact.processcvu_contract;
  }
  if (!matched.has_value()) {
    throw std::runtime_error(std::string("Model-managed ") +
                             (stage_label ? stage_label : "processcvu") +
                             " stage requires a canonical compiled contract (searched " +
                             std::to_string(stage_facts.size()) +
                             " stage facts, none had a matching processcvu_contract)."
                             " Ensure the MPK contract includes the required stage.");
  }
  return *matched;
}

CompiledProcessCvuContract
require_model_managed_postprocess_contract(const internal::ModelPack& pack,
                                           internal::ExecutionStageKind kind) {
  const auto post_plan = pack.execution_plan().post;
  const auto post_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
  if (post_plan.size() != post_facts.size()) {
    throw std::runtime_error(
        "Model-managed post-process stage facts are out of sync with execution plan"
        " (plan_count=" +
        std::to_string(post_plan.size()) + ", fact_count=" + std::to_string(post_facts.size()) +
        ")");
  }

  std::vector<std::string> matched_stage_names;
  for (std::size_t index = 0; index < post_plan.size(); ++index) {
    if (post_plan[index].kind != kind) {
      continue;
    }
    const auto& fact = post_facts[index];
    if (!fact.processcvu_contract.has_value()) {
      continue;
    }
    matched_stage_names.push_back(post_plan[index].stage_name);
    if (matched_stage_names.size() > 1U) {
      throw std::runtime_error("Model-managed post-process stage '" + post_plan[index].stage_name +
                               "' has duplicate entries");
    }
    return *fact.processcvu_contract;
  }
  if (matched_stage_names.empty()) {
    const auto stage_name =
        kind == internal::ExecutionStageKind::Detess          ? std::string("detess")
        : kind == internal::ExecutionStageKind::DetessCast    ? std::string("detesscast")
        : kind == internal::ExecutionStageKind::DetessDequant ? std::string("detessdequant")
        : kind == internal::ExecutionStageKind::Dequant       ? std::string("dequant")
                                                              : std::string("processcvu");
    throw std::runtime_error("Model-managed " + stage_name +
                             " stage requires a canonical compiled contract");
  }
  return {};
}

CompiledBoxDecodeContract
require_model_managed_boxdecode_contract(const internal::ModelPack& pack) {
  const auto post_plan = pack.execution_plan().post;
  const auto post_facts = pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
  if (post_plan.size() != post_facts.size()) {
    throw std::runtime_error(
        "Model-managed post-process stage facts are out of sync with execution plan");
  }

  for (std::size_t index = 0; index < post_plan.size(); ++index) {
    if (post_plan[index].kind != internal::ExecutionStageKind::BoxDecode) {
      continue;
    }
    const auto& fact = post_facts[index];
    if (!fact.boxdecode_compiled.has_value()) {
      continue;
    }
    return *fact.boxdecode_compiled;
  }
  throw std::runtime_error("Model-managed boxdecode stage requires a canonical compiled contract");
}

// Legacy processcvu wrappers removed; post-process model-managed stages now use canonical
// processcvu facts and contracts from stage facts directly.

const pipeline_internal::sima::MpkPluginIoContract*
find_pre_mla_processcvu_stage(const internal::ModelPack& pack,
                              std::initializer_list<const char*> preferred_families) {
  const auto& mpk_opt = pack.mpk_contract();
  if (!mpk_opt.has_value()) {
    return nullptr;
  }

  int mla_sequence = std::numeric_limits<int>::max();
  for (const auto& stage : mpk_opt->plugins) {
    if (upper_copy(stage.processor) == "MLA" && stage.sequence >= 0) {
      mla_sequence = std::min(mla_sequence, stage.sequence);
    }
  }

  const auto stage_precedes_mla = [&](const pipeline_internal::sima::MpkPluginIoContract& stage) {
    return mla_sequence == std::numeric_limits<int>::max() || stage.sequence < 0 ||
           stage.sequence < mla_sequence;
  };

  for (const char* preferred : preferred_families) {
    if (!preferred || !*preferred) {
      continue;
    }
    const std::string wanted(preferred);
    const pipeline_internal::sima::MpkPluginIoContract* best = nullptr;
    for (const auto& stage : mpk_opt->plugins) {
      if (!stage_precedes_mla(stage)) {
        continue;
      }
      if (upper_copy(stage.processor) == "MLA") {
        continue;
      }
      if (pipeline_internal::sima::canonical_processcvu_family_from_kernel(stage.kernel) !=
          wanted) {
        continue;
      }
      if (!best ||
          (stage.sequence >= 0 && (best->sequence < 0 || stage.sequence < best->sequence))) {
        best = &stage;
      }
    }
    if (best) {
      return best;
    }
  }

  return nullptr;
}

void populate_effective_model_managed_contract_fields(internal::PreprocessPlannerResult* plan,
                                                      const internal::ModelPack& pack) {
  if (!plan) {
    return;
  }

  const internal::PreprocessContractFlags flags = resolve_preprocess_contract_flags(*plan);
  auto& effective = plan->resolved_plan.effective;

  if (flags.tess_needed) {
    const auto* tess_stage = find_pre_mla_processcvu_stage(pack, {"quanttess", "tess", "preproc"});
    if (tess_stage) {
      const auto sd = dims_from_slice_shape(tess_stage->slice_shape);
      int ch =
          tess_stage->slice_shape.empty() ? 0 : static_cast<int>(tess_stage->slice_shape.back());
      if (ch <= 0 && !tess_stage->input_tensors.empty()) {
        const auto& shape = !tess_stage->input_tensors.front().logical_shape.empty()
                                ? tess_stage->input_tensors.front().logical_shape
                                : tess_stage->input_tensors.front().mpk_shape;
        if (!shape.empty()) {
          ch = static_cast<int>(shape.back());
        }
      }
      std::vector<int> slice_shape = {sd.h, sd.w};
      const int slice_channels = (ch > 0) ? ch : std::max(sd.d, 0);
      if (slice_channels > 0) {
        slice_shape.push_back(slice_channels);
      }
      effective.tessellate.set_slice_shape(std::move(slice_shape));
    }
  } else {
    effective.tessellate.slice_shape.clear();
  }

  if (flags.quant_needed) {
    const auto* quant_stage =
        find_pre_mla_processcvu_stage(pack, {"quanttess", "quant", "preproc"});
    if (quant_stage && quant_stage->quant.has_value()) {
      if (!quant_stage->quant->scales.empty() && quant_stage->quant->scales.front() > 0.0) {
        effective.quantize.scale = quant_stage->quant->scales.front();
      }
      if (!quant_stage->quant->zero_points.empty()) {
        effective.quantize.zero_point = quant_stage->quant->zero_points.front();
      }
    }
  } else {
    effective.quantize.scale = 0.0;
    effective.quantize.zero_point = 0;
  }
}

const char* input_memory_policy_name(InputMemoryPolicy policy) {
  switch (policy) {
  case InputMemoryPolicy::Auto:
    return "auto";
  case InputMemoryPolicy::Ev74:
    return "ev74";
  case InputMemoryPolicy::Dms0:
    return "dms0";
  case InputMemoryPolicy::SystemMemory:
    return "system";
  }
  return "auto";
}

InputMemoryPolicy
resolve_model_ingress_memory_policy(const internal::PreprocessPlannerResult& plan) {
  bool saw_non_cast_pre = false;
  for (const auto op : plan.session_route_plan.pre_chain) {
    if (op == internal::SessionPreStageOp::Cast) {
      continue;
    }
    saw_non_cast_pre = true;
    switch (op) {
    case internal::SessionPreStageOp::Preproc:
    case internal::SessionPreStageOp::Quant:
    case internal::SessionPreStageOp::Tess:
    case internal::SessionPreStageOp::QuantTess:
    case internal::SessionPreStageOp::CastTess:
      return InputMemoryPolicy::Ev74;
    case internal::SessionPreStageOp::Cast:
      break;
    }
  }

  // No effective pre adapter before inference (or Cast-only pre chain): treat as MLA-first.
  if (!plan.session_route_plan.include_pre_stage || !saw_non_cast_pre) {
    return InputMemoryPolicy::Dms0;
  }
  return InputMemoryPolicy::SystemMemory;
}

void apply_model_ingress_memory_policy(InputOptions& opt,
                                       const internal::PreprocessPlannerResult& plan) {
  const InputMemoryPolicy policy = resolve_model_ingress_memory_policy(plan);
  opt.memory_policy = policy;
  if (policy == InputMemoryPolicy::SystemMemory) {
    opt.use_simaai_pool = false;
  } else {
    opt.use_simaai_pool = true;
  }
}

bool runner_debug_enabled() {
  static const bool enabled = env_bool("SIMA_MODEL_RUNNER_DEBUG", false);
  return enabled;
}

std::string resolved_pre_stage_base_name(const internal::SessionRoutePlan& route) {
  if (!route.include_pre_stage || route.pre_chain.empty()) {
    return {};
  }
  switch (route.pre_chain.back()) {
  case internal::SessionPreStageOp::Preproc:
    return "preproc";
  case internal::SessionPreStageOp::Quant:
    return "quant";
  case internal::SessionPreStageOp::Tess:
    return "tess";
  case internal::SessionPreStageOp::QuantTess:
    return "quanttess";
  case internal::SessionPreStageOp::Cast:
    return "cast";
  case internal::SessionPreStageOp::CastTess:
    return "casttess";
  }
  return {};
}

std::string resolved_pre_stage_name(const internal::ModelPack& pack,
                                    const internal::PreprocessPlannerResult& plan) {
  const std::string base = resolved_pre_stage_base_name(plan.session_route_plan);
  return base.empty() ? std::string{} : pack.apply_name_suffix(base);
}

TensorDType dtype_from_format(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up.find("UINT8") != std::string::npos)
    return TensorDType::UInt8;
  if (up.find("INT8") != std::string::npos)
    return TensorDType::Int8;
  if (up.find("UINT16") != std::string::npos)
    return TensorDType::UInt16;
  if (up.find("INT16") != std::string::npos)
    return TensorDType::Int16;
  if (up.find("INT32") != std::string::npos)
    return TensorDType::Int32;
  if (up.find("BF16") != std::string::npos || up.find("BFLOAT16") != std::string::npos ||
      up.find("FP16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (up.find("FP64") != std::string::npos)
    return TensorDType::Float64;
  return TensorDType::Float32;
}

ImageSpec::PixelFormat pixel_format_from_string(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up == "RGB")
    return ImageSpec::PixelFormat::RGB;
  if (up == "BGR")
    return ImageSpec::PixelFormat::BGR;
  if (up == "GRAY" || up == "GRAY8")
    return ImageSpec::PixelFormat::GRAY8;
  if (up == "NV12")
    return ImageSpec::PixelFormat::NV12;
  if (up == "I420")
    return ImageSpec::PixelFormat::I420;
  return ImageSpec::PixelFormat::UNKNOWN;
}

bool is_gray_format(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  return up == "GRAY" || up == "GRAY8";
}

std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

std::string preprocess_color_format_name(PreprocessColorFormat fmt) {
  switch (fmt) {
  case PreprocessColorFormat::RGB:
    return "RGB";
  case PreprocessColorFormat::BGR:
    return "BGR";
  case PreprocessColorFormat::GRAY8:
    return "GRAY8";
  case PreprocessColorFormat::NV12:
    return "NV12";
  case PreprocessColorFormat::I420:
    return "I420";
  case PreprocessColorFormat::Auto:
    break;
  }
  return "";
}

std::string preprocess_graph_family_name(PreprocessGraphFamily family) {
  switch (family) {
  case PreprocessGraphFamily::Disabled:
    return "disabled";
  case PreprocessGraphFamily::Preproc:
    return "preproc";
  case PreprocessGraphFamily::Quant:
    return "quant";
  case PreprocessGraphFamily::Tess:
    return "tess";
  case PreprocessGraphFamily::QuantTess:
    return "quanttess";
  }
  return "preproc";
}

const char* post_route_stage_kind_name(internal::PostRouteStageKind kind) {
  switch (kind) {
  case internal::PostRouteStageKind::None:
    return "none";
  case internal::PostRouteStageKind::Detess:
    return "detess";
  case internal::PostRouteStageKind::DetessDequant:
    return "detessdequant";
  case internal::PostRouteStageKind::Dequantize:
    return "dequantize";
  case internal::PostRouteStageKind::BoxDecode:
    return "boxdecode";
  case internal::PostRouteStageKind::Cast:
    return "cast";
  case internal::PostRouteStageKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

const char* session_pre_stage_op_name(internal::SessionPreStageOp op);

const char* session_post_stage_op_name(internal::SessionPostStageOp op) {
  switch (op) {
  case internal::SessionPostStageOp::Detess:
    return "detess";
  case internal::SessionPostStageOp::DetessCast:
    return "detesscast";
  case internal::SessionPostStageOp::DetessDequant:
    return "detessdequant";
  case internal::SessionPostStageOp::Dequantize:
    return "dequantize";
  case internal::SessionPostStageOp::BoxDecode:
    return "boxdecode";
  case internal::SessionPostStageOp::Cast:
    return "cast";
  }
  return "unknown";
}

const char* stage_node_kind_name(internal::StageNodeKind kind) {
  switch (kind) {
  case internal::StageNodeKind::Preproc:
    return "Preproc";
  case internal::StageNodeKind::Quant:
    return "Quant";
  case internal::StageNodeKind::Tess:
    return "Tess";
  case internal::StageNodeKind::QuantTess:
    return "QuantTess";
  case internal::StageNodeKind::CastTess:
    return "CastTess";
  case internal::StageNodeKind::Detess:
    return "Detess";
  case internal::StageNodeKind::DetessCast:
    return "DetessCast";
  case internal::StageNodeKind::DetessDequant:
    return "DetessDequant";
  case internal::StageNodeKind::Dequant:
    return "Dequant";
  case internal::StageNodeKind::BoxDecode:
    return "BoxDecode";
  }
  return "Unknown";
}

bool route_contains_stage(const internal::SessionRoutePlan& route, internal::StageNodeKind kind) {
  const auto route_contains_post_region =
      [&](const pipeline_internal::sima::RouteGraphKernelKind kernel_kind) {
        return std::any_of(
            route.post_regions.begin(), route.post_regions.end(),
            [&](const internal::RouteRegion& region) { return region.op_kind == kernel_kind; });
      };
  switch (kind) {
  case internal::StageNodeKind::Preproc:
    return std::find(route.pre_chain.begin(), route.pre_chain.end(),
                     internal::SessionPreStageOp::Preproc) != route.pre_chain.end();
  case internal::StageNodeKind::Quant:
    return std::find(route.pre_chain.begin(), route.pre_chain.end(),
                     internal::SessionPreStageOp::Quant) != route.pre_chain.end();
  case internal::StageNodeKind::Tess:
    return std::find(route.pre_chain.begin(), route.pre_chain.end(),
                     internal::SessionPreStageOp::Tess) != route.pre_chain.end();
  case internal::StageNodeKind::QuantTess:
    return std::find(route.pre_chain.begin(), route.pre_chain.end(),
                     internal::SessionPreStageOp::QuantTess) != route.pre_chain.end();
  case internal::StageNodeKind::CastTess:
    return std::find(route.pre_chain.begin(), route.pre_chain.end(),
                     internal::SessionPreStageOp::CastTess) != route.pre_chain.end();
  case internal::StageNodeKind::Detess:
    if (!route.post_regions.empty()) {
      return route_contains_post_region(pipeline_internal::sima::RouteGraphKernelKind::Detess);
    }
    return std::find(route.post_chain.begin(), route.post_chain.end(),
                     internal::SessionPostStageOp::Detess) != route.post_chain.end();
  case internal::StageNodeKind::DetessCast:
    if (!route.post_regions.empty()) {
      return route_contains_post_region(pipeline_internal::sima::RouteGraphKernelKind::DetessCast);
    }
    return std::find(route.post_chain.begin(), route.post_chain.end(),
                     internal::SessionPostStageOp::DetessCast) != route.post_chain.end();
  case internal::StageNodeKind::DetessDequant:
    if (!route.post_regions.empty()) {
      return route_contains_post_region(
          pipeline_internal::sima::RouteGraphKernelKind::DetessDequant);
    }
    return std::find(route.post_chain.begin(), route.post_chain.end(),
                     internal::SessionPostStageOp::DetessDequant) != route.post_chain.end();
  case internal::StageNodeKind::Dequant:
    if (!route.post_regions.empty()) {
      return route_contains_post_region(pipeline_internal::sima::RouteGraphKernelKind::Dequantize);
    }
    return std::find(route.post_chain.begin(), route.post_chain.end(),
                     internal::SessionPostStageOp::Dequantize) != route.post_chain.end();
  case internal::StageNodeKind::BoxDecode:
    if (!route.post_regions.empty()) {
      return route_contains_post_region(pipeline_internal::sima::RouteGraphKernelKind::BoxDecode);
    }
    return std::find(route.post_chain.begin(), route.post_chain.end(),
                     internal::SessionPostStageOp::BoxDecode) != route.post_chain.end();
  }
  return false;
}

bool route_uses_image_ingress(const internal::SessionRoutePlan& route) {
  return route.include_pre_stage && !route.pre_chain.empty() &&
         route.pre_chain.front() == internal::SessionPreStageOp::Preproc;
}

bool should_overlay_appsrc_from_ingress_contract(const internal::SessionRoutePlan& route,
                                                 bool tensor_mode) {
  return tensor_mode || !route_uses_image_ingress(route);
}

bool resolved_preprocess_graph_contains_stage(const internal::PreprocessPlannerResult& plan,
                                              internal::StageNodeKind kind) {
  const std::string kernel = pipeline_internal::lower_copy(plan.resolved_plan.graph_kernel);
  if (!plan.resolved_plan.enabled || kernel.empty() || kernel == "disabled") {
    return false;
  }
  switch (kind) {
  case internal::StageNodeKind::Preproc:
    return kernel == "preproc" || kernel == "preprocess";
  case internal::StageNodeKind::Quant:
    return kernel == "quant";
  case internal::StageNodeKind::Tess:
    return kernel == "tess" || kernel == "tessellate";
  case internal::StageNodeKind::QuantTess:
    return kernel == "quanttess";
  case internal::StageNodeKind::CastTess:
    return kernel == "casttess" || kernel == "casttessellate";
  case internal::StageNodeKind::Detess:
  case internal::StageNodeKind::DetessCast:
  case internal::StageNodeKind::DetessDequant:
  case internal::StageNodeKind::Dequant:
  case internal::StageNodeKind::BoxDecode:
    return false;
  }
  return false;
}

std::string route_stage_summary(const internal::SessionRoutePlan& route) {
  std::ostringstream ss;
  ss << "pre=[";
  for (std::size_t i = 0; i < route.pre_chain.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << session_pre_stage_op_name(route.pre_chain[i]);
  }
  ss << "] post=[";
  for (std::size_t i = 0; i < route.post_chain.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << session_post_stage_op_name(route.post_chain[i]);
  }
  ss << "]";
  return ss.str();
}

std::string preprocess_contract_debug_string(const PreprocessContract& contract) {
  std::ostringstream oss;
  oss << "{media=" << (contract.media_type.empty() ? "<empty>" : contract.media_type)
      << ",format=" << (contract.format.empty() ? "<empty>" : contract.format)
      << ",w=" << contract.width << ",h=" << contract.height << ",d=" << contract.depth
      << ",max_w=" << contract.max_width << ",max_h=" << contract.max_height
      << ",max_d=" << contract.max_depth << "}";
  return oss.str();
}

std::string ingress_tensor_contract_debug_string(const internal::IngressTensorContract& ingress) {
  std::ostringstream oss;
  oss << "{idx=" << ingress.ingress_index
      << ",name=" << (ingress.source_tensor_name.empty() ? "<empty>" : ingress.source_tensor_name)
      << ",media=" << (ingress.media_type.empty() ? "<empty>" : ingress.media_type)
      << ",dtype=" << (ingress.dtype.empty() ? "<empty>" : ingress.dtype)
      << ",layout=" << (ingress.layout.empty() ? "<empty>" : ingress.layout)
      << ",rank=" << ingress.rank << ",batch=" << ingress.batch << ",w=" << ingress.width
      << ",h=" << ingress.height << ",d=" << ingress.depth
      << ",src_stage=" << (ingress.source_stage.empty() ? "<empty>" : ingress.source_stage)
      << ",dst_plugin_index=" << ingress.dst_plugin_index
      << ",dst_input_index=" << ingress.dst_input_index << "}";
  return oss.str();
}

template <typename T, typename Fn>
std::string vector_debug_string(const std::vector<T>& values, Fn&& fn) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) {
      oss << ", ";
    }
    oss << fn(values[i]);
  }
  oss << "]";
  return oss.str();
}

std::string route_diagnostics_debug_string(const std::vector<std::string>& diagnostics) {
  if (diagnostics.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < diagnostics.size(); ++i) {
    if (i != 0U) {
      oss << "; ";
    }
    oss << diagnostics[i];
  }
  oss << "]";
  return oss.str();
}

std::string stage_route_missing_hint(internal::StageNodeKind kind) {
  switch (kind) {
  case internal::StageNodeKind::Preproc:
    return "Pass preprocess options to Model so the planner includes a Preproc stage, "
           "then construct Preproc(Model) again.";
  case internal::StageNodeKind::Quant:
  case internal::StageNodeKind::Tess:
  case internal::StageNodeKind::QuantTess:
  case internal::StageNodeKind::CastTess:
  case internal::StageNodeKind::Detess:
  case internal::StageNodeKind::DetessCast:
  case internal::StageNodeKind::DetessDequant:
  case internal::StageNodeKind::Dequant:
  case internal::StageNodeKind::BoxDecode:
    return "Either configure Model so the route includes that stage, or construct the stage "
           "with explicit standalone options.";
  }
  return {};
}

void maybe_log_model_info_shadow(const internal::PreprocessPlannerResult& plan,
                                 const internal::ModelPack& pack) {
  if (!env_bool("SIMA_MODEL_INFO_SHADOW_DEBUG", false)) {
    return;
  }
  const internal::ParsedModelInfo parsed = internal::parse_model_from_pack(pack);
  const bool cast_symmetry_need_ok = !parsed.needs.pre_cast || parsed.needs.post_cast;
  const bool needs_pre_any =
      parsed.needs.pre_quantization || parsed.needs.pre_tessellation || parsed.needs.pre_cast;
  const bool needs_post_any = parsed.needs.post_detessellation ||
                              parsed.needs.post_dequantization || parsed.needs.post_cast;
  std::fprintf(stderr,
               "[model-info-shadow] model=%s mpk=%s etc=%s "
               "effective{include_pre=%d,include_post=%d,selected_post=%s,infer_only=%d,graph=%s} "
               "route_flags{tess_needed=%d,quant_needed=%d} "
               "pre_needs{q=%d,t=%d,cast=%d} post_needs{detess=%d,dequant=%d,cast=%d} "
               "post_cap{detess=%d,dequant=%d,cast=%d,box=%d} cast_symmetry{needs=%d,effective=%d} "
               "outputs{physical=%zu,logical=%zu,packed=%d}\n",
               parsed.model_name.c_str(), parsed.mpk_json_path.c_str(), pack.etc_dir().c_str(),
               plan.include_preprocess_stage ? 1 : 0, plan.include_postprocess_stage ? 1 : 0,
               plan.route_selected_post_kind.c_str(), plan.infer_only_route ? 1 : 0,
               preprocess_graph_family_name(plan.resolved_plan.graph_family).c_str(),
               parsed.needs.tess_needed ? 1 : 0, parsed.needs.quant_needed ? 1 : 0,
               parsed.needs.pre_quantization ? 1 : 0, parsed.needs.pre_tessellation ? 1 : 0,
               parsed.needs.pre_cast ? 1 : 0, parsed.needs.post_detessellation ? 1 : 0,
               parsed.needs.post_dequantization ? 1 : 0, parsed.needs.post_cast ? 1 : 0,
               parsed.capabilities.has_post_detessellation ? 1 : 0,
               parsed.capabilities.has_post_dequantization ? 1 : 0,
               parsed.capabilities.has_post_cast ? 1 : 0,
               parsed.capabilities.has_post_boxdecode ? 1 : 0, cast_symmetry_need_ok ? 1 : 0,
               plan.route_cast_symmetry_ok ? 1 : 0, parsed.outputs.physical.size(),
               parsed.outputs.logical.size(), parsed.outputs.packed_output ? 1 : 0);

  if (!parsed.pre_kernels.empty()) {
    std::string csv;
    for (std::size_t i = 0; i < parsed.pre_kernels.size(); ++i) {
      if (i > 0U) {
        csv += ",";
      }
      csv += parsed.pre_kernels[i];
    }
    std::fprintf(stderr, "[model-info-shadow] pre_kernels={%s}\n", csv.c_str());
  }
  if (!parsed.post_kernels.empty()) {
    std::string csv;
    for (std::size_t i = 0; i < parsed.post_kernels.size(); ++i) {
      if (i > 0U) {
        csv += ",";
      }
      csv += parsed.post_kernels[i];
    }
    std::fprintf(stderr, "[model-info-shadow] post_kernels={%s}\n", csv.c_str());
  }
  // Emit mismatch warnings only when the effective route violates parsed needs.
  if (needs_pre_any && !plan.include_preprocess_stage) {
    std::fprintf(stderr, "[model-info-shadow][mismatch] effective route disabled pre stage but "
                         "model needs pre processing\n");
  }
  if (needs_post_any && !plan.include_postprocess_stage) {
    std::fprintf(stderr, "[model-info-shadow][mismatch] effective route disabled post stage but "
                         "model needs post processing\n");
  }
  if (!plan.route_cast_symmetry_ok) {
    std::fprintf(stderr, "[model-info-shadow][mismatch] effective route violated cast symmetry "
                         "(pre_cast requires post_cast)\n");
  }
  for (const auto& warn : parsed.warnings) {
    std::fprintf(stderr, "[model-info-shadow][parser] %s\n", warn.c_str());
  }
}

bool format_is_quantized_tensor(std::string fmt) {
  fmt = upper_copy(std::move(fmt));
  if (fmt.empty()) {
    return false;
  }
  if (fmt.find("FP") != std::string::npos || fmt.find("FLOAT") != std::string::npos ||
      fmt.find("BF16") != std::string::npos || fmt.find("BFLOAT16") != std::string::npos) {
    return false;
  }
  return (fmt.find("INT8") != std::string::npos || fmt.find("UINT8") != std::string::npos ||
          fmt.find("INT16") != std::string::npos || fmt.find("UINT16") != std::string::npos ||
          fmt.find("INT32") != std::string::npos || fmt.find("UINT32") != std::string::npos);
}

bool format_is_bf16_tensor(std::string fmt) {
  fmt = upper_copy(std::move(fmt));
  if (fmt.empty()) {
    return false;
  }
  return fmt.find("BF16") != std::string::npos || fmt.find("BFLOAT16") != std::string::npos ||
         fmt.find("EVXX_BFLOAT16") != std::string::npos;
}

std::string route_mla_input_dtype_from_diagnostics(const internal::PreprocessPlannerResult& plan) {
  static constexpr const char* kPrefix = "mla_input_dtype=";
  for (const auto& entry : plan.route_diagnostics) {
    if (entry.rfind(kPrefix, 0) == 0) {
      return entry.substr(std::strlen(kPrefix));
    }
  }
  return {};
}

bool transform_explicit_off(const std::vector<Transform>& transforms, TransformType type) {
  for (const auto& t : transforms) {
    if (t.type != type) {
      continue;
    }
    switch (type) {
    case TransformType::Resize:
      return t.resize.enable == AutoFlag::Off;
    case TransformType::ColorConvert:
      return t.color_convert.enable == AutoFlag::Off;
    case TransformType::LayoutConvert:
      return t.layout_convert.enable == AutoFlag::Off;
    case TransformType::Normalize:
      return t.normalize.enable == AutoFlag::Off;
    case TransformType::Quantize:
      return t.quantize.enable == AutoFlag::Off;
    case TransformType::Tessellate:
      return t.tessellate.enable == AutoFlag::Off;
    }
  }
  return false;
}

bool user_explicit_off(const PreprocessOptions& request, bool transforms_override,
                       TransformType type) {
  if (transforms_override) {
    return transform_explicit_off(request.transforms, type);
  }
  switch (type) {
  case TransformType::Resize:
    return request.resize.enable == AutoFlag::Off;
  case TransformType::ColorConvert:
    return request.color_convert.enable == AutoFlag::Off;
  case TransformType::LayoutConvert:
    return request.layout_convert.enable == AutoFlag::Off;
  case TransformType::Normalize:
    return request.normalize.enable == AutoFlag::Off;
  case TransformType::Quantize:
    return request.quantize.enable == AutoFlag::Off;
  case TransformType::Tessellate:
    return request.tessellate.enable == AutoFlag::Off;
  }
  return false;
}

template <typename ContractT>
bool populate_spec_from_route_contract(const ContractT& contract, TensorSpec* spec) {
  if (!spec || !contract.valid) {
    return false;
  }

  if (!contract.dtype.empty()) {
    spec->dtypes = {dtype_from_format(contract.dtype)};
  }

  // Phase 4a: the contract carries the canonical shape vector from the MPK
  // (e.g. [10,1,1024,2] for a batched MLA input). Copy it verbatim — this is
  // the source of truth for tensor-domain consumers. We deliberately do NOT
  // permute dims based on a layout token: the layout string is advisory
  // metadata only, not a permutation key, and trying to reconstruct a shape
  // vector from (batch,h,w,d) scalars is lossy for rank > 4 and ambiguous
  // when the layout token is missing or unrecognised.
  if (!contract.logical_shape.empty()) {
    spec->shape = contract.logical_shape;
    spec->rank = static_cast<int>(spec->shape.size());
  }
  return true;
}

std::string format_from_tensor(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    const std::string fmt = image_format_name(tensor.semantic.image->format);
    if (!fmt.empty())
      return fmt;
  }
  return "";
}

struct InputInfo {
  enum class FormatSource : std::uint8_t {
    Unspecified = 0,
    Heuristic = 1,
    Explicit = 2,
  };

  std::string media_type;
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
  FormatSource format_source = FormatSource::Unspecified;
};

std::string input_info_debug_string(const InputInfo* input) {
  if (!input) {
    return "<null>";
  }
  std::ostringstream oss;
  oss << "{media=" << (input->media_type.empty() ? "<empty>" : input->media_type)
      << ",format=" << (input->format.empty() ? "<empty>" : input->format) << ",w=" << input->width
      << ",h=" << input->height << ",d=" << input->depth
      << ",format_source=" << static_cast<int>(input->format_source) << "}";
  return oss.str();
}

struct InputKey {
  std::string media_type;
  std::string format;
  int width = 0;
  int height = 0;
  int depth = 0;
};

bool operator==(const InputKey& a, const InputKey& b) {
  return a.media_type == b.media_type && a.format == b.format && a.width == b.width &&
         a.height == b.height && a.depth == b.depth;
}

InputKey input_key_from_appsrc(const InputOptions& opt) {
  InputKey key;
  key.media_type = resolve_input_media_type(opt);
  key.format = upper_copy(opt.format);
  key.width = opt.width;
  key.height = opt.height;
  key.depth = opt.depth;
  return key;
}

#if defined(SIMA_WITH_OPENCV)
InputInfo input_info_from_mat(const cv::Mat& input) {
  InputInfo info;
  info.media_type = "video/x-raw";
  info.width = input.cols;
  info.height = input.rows;
  const int channels = input.channels();
  info.depth = channels;
  info.format = (channels == 1) ? "GRAY8" : "BGR";
  info.format_source = InputInfo::FormatSource::Heuristic;
  return info;
}

cv::Mat maybe_resize_mat_to_tensor_caps(const cv::Mat& input, const InputOptions& src_opt) {
  if (!input.data || input.empty()) {
    return input;
  }
  if (runner_debug_enabled()) {
    const int target_w = (src_opt.width > 0) ? src_opt.width : src_opt.max_width;
    const int target_h = (src_opt.height > 0) ? src_opt.height : src_opt.max_height;
    if (target_w > 0 && target_h > 0 && (input.cols != target_w || input.rows != target_h)) {
      std::fprintf(stderr,
                   "[model-route-debug] tensor_ingress_resize_disabled input=%dx%dx%d "
                   "src_opt{w=%d h=%d max=%dx%d format=%s} requested_target=%dx%d action=keep\n",
                   input.cols, input.rows, input.channels(), src_opt.width, src_opt.height,
                   src_opt.max_width, src_opt.max_height, src_opt.format.str().c_str(), target_w,
                   target_h);
    }
  }
  return input;
}

cv::Mat
maybe_resize_tensor_ingress_mat_for_pre_adapter(const cv::Mat& input,
                                                const internal::PreprocessPlannerResult& plan,
                                                const InputOptions& src_opt) {
  if (!input.data || input.empty()) {
    return input;
  }
  if (!plan.session_route_plan.include_pre_stage || plan.session_route_plan.pre_chain.empty()) {
    // Tensor-only ingress (no explicit pre stage) must still respect model input
    // tensor contract dimensions before appsrc caps negotiation.
    return maybe_resize_mat_to_tensor_caps(input, src_opt);
  }
  if (plan.session_route_plan.pre_chain.front() == internal::SessionPreStageOp::Preproc) {
    return input;
  }
  const char* first_pre = "unknown";
  switch (plan.session_route_plan.pre_chain.front()) {
  case internal::SessionPreStageOp::Preproc:
    first_pre = "preproc";
    break;
  case internal::SessionPreStageOp::Quant:
    first_pre = "quant";
    break;
  case internal::SessionPreStageOp::Tess:
    first_pre = "tess";
    break;
  case internal::SessionPreStageOp::QuantTess:
    first_pre = "quanttess";
    break;
  case internal::SessionPreStageOp::Cast:
    first_pre = "cast";
    break;
  }

  const int target_w = (src_opt.width > 0) ? src_opt.width : src_opt.max_width;
  const int target_h = (src_opt.height > 0) ? src_opt.height : src_opt.max_height;

  if (runner_debug_enabled()) {
    const char* action =
        (target_w > 0 && target_h > 0 && (input.cols != target_w || input.rows != target_h))
            ? "keep_resize_disabled"
            : "keep";
    std::fprintf(stderr,
                 "[model-route-debug] tensor_ingress_resize first_pre=%s input=%dx%dx%d "
                 "src_opt{w=%d h=%d max=%dx%d format=%s} target=%dx%d action=%s\n",
                 first_pre, input.cols, input.rows, input.channels(), src_opt.width, src_opt.height,
                 src_opt.max_width, src_opt.max_height, src_opt.format.str().c_str(), target_w,
                 target_h, action);
  }
  return input;
}
#endif

int64_t shape_dim(const std::vector<int64_t>& shape, size_t index) {
  if (index >= shape.size())
    return -1;
  return shape[index];
}

InputInfo input_info_from_tensor(const simaai::neat::Tensor& tensor, bool image_mode) {
  InputInfo info;
  if (image_mode) {
    info.media_type = "video/x-raw";
    if (tensor.semantic.image.has_value()) {
      info.format = image_format_name(tensor.semantic.image->format);
      info.format_source = InputInfo::FormatSource::Explicit;
    }
  } else {
    info.media_type = "application/vnd.simaai.tensor";
    info.format = format_from_tensor(tensor);
    if (!info.format.empty()) {
      info.format_source = InputInfo::FormatSource::Explicit;
    }
  }

  if (info.format.empty()) {
    if (image_mode) {
      // Do not infer an image color format from shape or dtype. Image tensors
      // are unambiguous only when the caller attached Tensor::semantic.image;
      // otherwise the caller must provide explicit image metadata (for example
      // Python's Tensor.from_numpy(..., image_format=...)).
    } else if (tensor.dtype == TensorDType::Float32) {
      info.format = "FP32";
      info.format_source = InputInfo::FormatSource::Explicit;
    } else if (tensor.dtype == TensorDType::Int8) {
      info.format = "INT8";
      info.format_source = InputInfo::FormatSource::Explicit;
    } else if (tensor.dtype == TensorDType::UInt8) {
      info.format = "UINT8";
      info.format_source = InputInfo::FormatSource::Explicit;
    }
  }

  if (tensor.is_composite() && !tensor.planes.empty()) {
    const int64_t shape_h = shape_dim(tensor.shape, 0);
    const int64_t shape_w = shape_dim(tensor.shape, 1);
    const auto& y = tensor.planes.front();
    if (y.shape.size() >= 2) {
      info.height = (shape_h > 0) ? static_cast<int>(shape_h) : static_cast<int>(y.shape[0]);
      info.width = (shape_w > 0) ? static_cast<int>(shape_w) : static_cast<int>(y.shape[1]);
    }
  } else {
    const TensorCompatDims compat = tensor_compat_dims_from_shape(tensor.shape, tensor.layout);
    info.height = compat.height;
    info.width = compat.width;
    info.depth = compat.depth;
  }
  return info;
}

InputInfo input_info_from_image_sample(const simaai::neat::Sample& sample) {
  const Tensor* tensor = nullptr;
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    tensor = &*sample.tensor;
  } else if (sample.kind == SampleKind::TensorSet && !sample.tensors.empty()) {
    tensor = &sample.tensors.front();
  }
  if (!tensor) {
    return InputInfo{};
  }

  InputInfo info = input_info_from_tensor(*tensor, true);
  const std::string sample_media = sample_media_type(sample);
  if (!sample_media.empty()) {
    info.media_type = sample_media;
  }
  const std::string sample_format =
      !sample.payload_tag.empty() ? sample.payload_tag : sample.format;
  if (!sample_format.empty()) {
    info.format = upper_copy(sample_format);
    info.format_source = InputInfo::FormatSource::Explicit;
  }
  return info;
}

void require_explicit_image_input_info(const InputInfo& info, const char* where) {
  if (upper_copy(info.media_type) != "VIDEO/X-RAW") {
    std::ostringstream oss;
    oss << (where ? where : "Model") << ": image-mode input must be video/x-raw";
    if (!info.media_type.empty()) {
      oss << " (got " << info.media_type << ")";
    }
    throw std::invalid_argument(oss.str());
  }
  if (info.format.empty() || info.format_source != InputInfo::FormatSource::Explicit) {
    throw std::invalid_argument(
        std::string(where ? where : "Model") +
        ": image-mode Tensor input requires explicit image format metadata; pass a Tensor with "
        "ImageSpec/image_format (for Python: Tensor.from_numpy(..., image_format=...)) or a "
        "Sample with video/x-raw format metadata.");
  }
}

InputOptions appsrc_from_info(const InputInfo& info) {
  InputOptions opt;
  opt.payload_type = input_type_from_media_type(info.media_type);
  opt.format = info.format;
  opt.width = info.width;
  opt.height = info.height;
  opt.depth = info.depth;
  return opt;
}

Tensor make_dummy_tensor(const simaai::neat::InputOptions& opt) {
  Tensor t;
  t.device = {DeviceType::CPU, 0};
  t.read_only = true;

  if (resolve_input_media_type(opt) == "application/vnd.simaai.tensor") {
    const int w = (opt.width > 0) ? opt.width : opt.max_width;
    const int h = (opt.height > 0) ? opt.height : opt.max_height;
    const int d = (opt.depth > 0) ? opt.depth : opt.max_depth;
    if (w <= 0 || h <= 0 || d <= 0) {
      throw std::runtime_error(
          "Model::build: missing tensor input shape for dummy input. "
          "Fix: set Model::Options::preprocess.input_max_width/input_max_height/input_max_depth.");
    }
    t.dtype = dtype_from_format(opt.format);
    t.shape = {h, w, d};
    t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
    const std::size_t elem = pipeline_internal::dtype_bytes(t.dtype);
    const std::size_t bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                              static_cast<std::size_t>(d) * elem;
    t.storage = make_cpu_owned_storage(bytes);
    t.strides_bytes = pipeline_internal::contiguous_strides_bytes(t.shape, elem);
    return t;
  }

  const int w = (opt.width > 0) ? opt.width : opt.max_width;
  const int h = (opt.height > 0) ? opt.height : opt.max_height;
  if (w <= 0 || h <= 0) {
    throw std::runtime_error(
        "Model::build: missing image input shape for dummy input. "
        "Fix: set Model::Options::preprocess.input_max_width/input_max_height.");
  }
  std::string fmt = opt.format;
  if (fmt.empty())
    fmt = "RGB";
  const auto pix = pixel_format_from_string(fmt);
  if (pix == ImageSpec::PixelFormat::UNKNOWN) {
    throw std::runtime_error("Model::build: unsupported image format");
  }

  t.dtype = TensorDType::UInt8;
  t.semantic.image = ImageSpec{pix, ""};

  if (pix == ImageSpec::PixelFormat::NV12) {
    const std::size_t y_size = static_cast<std::size_t>(w * h);
    const std::size_t uv_size = static_cast<std::size_t>(w * h / 2);
    t.storage = make_cpu_owned_storage(y_size + uv_size);
    t.shape = {h, w};
    t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};

    Plane y;
    y.role = PlaneRole::Y;
    y.shape = {h, w};
    y.strides_bytes = {w, 1};
    y.byte_offset = 0;

    Plane uv;
    uv.role = PlaneRole::UV;
    uv.shape = {h / 2, w};
    uv.strides_bytes = {w, 1};
    uv.byte_offset = static_cast<int64_t>(y_size);

    t.planes = {y, uv};
    return t;
  }

  if (pix == ImageSpec::PixelFormat::I420) {
    const std::size_t y_size = static_cast<std::size_t>(w * h);
    const std::size_t uv_size = static_cast<std::size_t>(w * h / 4);
    t.storage = make_cpu_owned_storage(y_size + uv_size * 2);
    t.shape = {h, w};
    t.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};

    Plane y;
    y.role = PlaneRole::Y;
    y.shape = {h, w};
    y.strides_bytes = {w, 1};
    y.byte_offset = 0;

    Plane u;
    u.role = PlaneRole::U;
    u.shape = {h / 2, w / 2};
    u.strides_bytes = {w / 2, 1};
    u.byte_offset = static_cast<int64_t>(y_size);

    Plane v;
    v.role = PlaneRole::V;
    v.shape = {h / 2, w / 2};
    v.strides_bytes = {w / 2, 1};
    v.byte_offset = static_cast<int64_t>(y_size + uv_size);

    t.planes = {y, u, v};
    return t;
  }

  const int depth = (pix == ImageSpec::PixelFormat::GRAY8) ? 1 : 3;
  t.shape = (depth == 1) ? std::vector<int64_t>{h, w} : std::vector<int64_t>{h, w, depth};
  t.axis_semantics =
      (depth == 1) ? std::vector<TensorAxisSemantic>{TensorAxisSemantic::H, TensorAxisSemantic::W}
                   : std::vector<TensorAxisSemantic>{TensorAxisSemantic::H, TensorAxisSemantic::W,
                                                     TensorAxisSemantic::C};
  const std::size_t bytes =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(depth);
  t.storage = make_cpu_owned_storage(bytes);
  t.strides_bytes = pipeline_internal::contiguous_strides_bytes(t.shape, 1);
  return t;
}

simaai::neat::Sample make_bundle_from_tensors(const std::vector<Tensor>& inputs) {
  return sample_from_tensors(inputs);
}

std::vector<internal::IngressTensorContract>
normalized_ingress_contracts(const internal::SessionRoutePlan& route) {
  return route.ingress_contracts;
}

const internal::IngressTensorContract* maybe_single_ingress_contract(
    const std::vector<internal::IngressTensorContract>& ingress_contracts) {
  return ingress_contracts.size() == 1U ? &ingress_contracts.front() : nullptr;
}

const PreprocessContract*
maybe_single_preprocess_ingress_contract(const std::vector<PreprocessContract>& ingress_contracts) {
  return ingress_contracts.size() == 1U ? &ingress_contracts.front() : nullptr;
}

std::string
unify_ingress_dtype(const std::vector<internal::IngressTensorContract>& ingress_contracts) {
  std::string dtype;
  for (const auto& ingress : ingress_contracts) {
    if (ingress.dtype.empty()) {
      continue;
    }
    if (dtype.empty()) {
      dtype = ingress.dtype;
      continue;
    }
    if (upper_copy(dtype) != upper_copy(ingress.dtype)) {
      return {};
    }
  }
  return dtype;
}

std::vector<std::string> ingress_names_from_contracts(
    const std::vector<internal::IngressTensorContract>& ingress_contracts) {
  std::vector<std::string> names;
  names.reserve(ingress_contracts.size());
  for (std::size_t i = 0; i < ingress_contracts.size(); ++i) {
    const auto& ingress = ingress_contracts[i];
    if (!ingress.source_tensor_name.empty()) {
      names.push_back(ingress.source_tensor_name);
    } else {
      names.push_back("ifm" + std::to_string(i));
    }
  }
  return names;
}

std::string graph_endpoint_token(std::string_view text, std::string_view fallback) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc)) {
      out.push_back(static_cast<char>(std::tolower(uc)));
    } else if (c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else if (!out.empty() && out.back() != '_') {
      out.push_back('_');
    }
  }
  while (!out.empty() && out.front() == '_') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  if (out.empty()) {
    out = std::string(fallback);
  }
  if (!out.empty() && std::isdigit(static_cast<unsigned char>(out.front()))) {
    out.insert(out.begin(), '_');
  }
  return out;
}

std::vector<std::string> make_unique_endpoint_names(std::vector<std::string> names,
                                                    std::string_view fallback_prefix) {
  std::unordered_set<std::string> used;
  for (std::size_t i = 0; i < names.size(); ++i) {
    std::string base =
        graph_endpoint_token(names[i], std::string(fallback_prefix) + std::to_string(i));
    if (base.empty()) {
      base = std::string(fallback_prefix) + std::to_string(i);
    }
    std::string candidate = base;
    for (std::size_t suffix = 1U; !used.insert(candidate).second; ++suffix) {
      candidate = base + "_" + std::to_string(suffix);
    }
    names[i] = std::move(candidate);
  }
  return names;
}

std::vector<std::string> egress_names_from_contracts(const internal::SessionRoutePlan& route_plan) {
  const std::vector<internal::EgressTensorContract>* contracts = &route_plan.egress_contracts;
  std::vector<internal::EgressTensorContract> single;
  if (contracts->empty() && route_plan.egress_contract.valid) {
    single.push_back(route_plan.egress_contract);
    contracts = &single;
  }

  std::vector<std::string> names;
  names.reserve(contracts->size());
  for (std::size_t i = 0; i < contracts->size(); ++i) {
    const auto& egress = (*contracts)[i];
    if (!egress.source_tensor_name.empty()) {
      names.push_back(egress.source_tensor_name);
    } else if (!egress.source_stage.empty() && contracts->size() == 1U) {
      names.push_back(egress.source_stage);
    } else {
      names.push_back("ofm" + std::to_string(i));
    }
  }
  if (names.empty()) {
    names.push_back("output");
  }
  return make_unique_endpoint_names(std::move(names), "ofm");
}

std::vector<std::string> aggregate_egress_name_for_model_route(const std::string& model_id,
                                                               const Model::RouteOptions& opt) {
  std::string name = !opt.buffer_name.empty() ? opt.buffer_name : model_id;
  if (name.empty()) {
    name = "output";
  }
  return make_unique_endpoint_names({std::move(name)}, "output");
}

bool model_route_exposes_individual_outputs(const internal::SessionRoutePlan& route_plan,
                                            const Model::RouteOptions& opt) {
  // Public Graph model routes default to one aggregate output Sample. Many models expose
  // several logical tensors (for example YOLO bbox/class heads), but optimized kernels pack those
  // tensors into a single physical output buffer. In that common case there is still only one
  // public output endpoint, even when expose_all_outputs=true. The advanced flag only exposes
  // individual endpoints when the model route actually has more than one physical output buffer.
  return opt.expose_all_outputs && route_plan.output_physical_count > 1U;
}

void require_exact_ingress_count(
    std::size_t actual, const std::vector<internal::IngressTensorContract>& ingress_contracts,
    const char* where, const char* kind) {
  const std::size_t expected = ingress_contracts.empty() ? 1U : ingress_contracts.size();
  if (actual == expected) {
    return;
  }
  std::ostringstream oss;
  oss << (where ? where : "Model") << ": expected exactly " << expected << " ingress " << kind;
  if (!ingress_contracts.empty()) {
    oss << " (";
    for (std::size_t i = 0; i < ingress_contracts.size(); ++i) {
      if (i) {
        oss << ",";
      }
      oss << (!ingress_contracts[i].source_tensor_name.empty()
                  ? ingress_contracts[i].source_tensor_name
                  : ("ifm" + std::to_string(i)));
    }
    oss << ")";
  }
  oss << ", got " << actual;
  throw std::invalid_argument(oss.str());
}

void require_single_ingress_api(
    const std::vector<internal::IngressTensorContract>& ingress_contracts, const char* where) {
  if (ingress_contracts.size() <= 1U) {
    return;
  }
  std::ostringstream oss;
  oss << (where ? where : "Model") << ": multi-ingress model requires plural ingress API";
  oss << " (use input_specs()/input_appsrc_options_list())";
  throw std::runtime_error(oss.str());
}

TensorSpec tensor_spec_from_ingress_contract(const internal::IngressTensorContract& ingress) {
  TensorSpec spec;
  if (populate_spec_from_route_contract(ingress, &spec) && !spec.shape.empty()) {
    if (spec.rank < 0) {
      spec.rank = static_cast<int>(spec.shape.size());
    }
  }
  return spec;
}

std::string ingress_expected_format_token(const internal::IngressTensorContract& ingress) {
  return upper_copy(ingress.dtype);
}

void validate_single_tensor_ingress_expectation(const internal::IngressTensorContract& ingress,
                                                const InputInfo& info, const char* where) {
  const std::string expected_media = ingress.valid && !ingress.media_type.empty()
                                         ? upper_copy(ingress.media_type)
                                         : std::string("APPLICATION/VND.SIMAAI.TENSOR");
  const std::string actual_media = upper_copy(info.media_type);
  const std::string expected_format = ingress_expected_format_token(ingress);
  const std::string actual_format = upper_copy(info.format);
  const bool expects_fp32 = expected_format.find("FP32") != std::string::npos ||
                            expected_format.find("FLOAT32") != std::string::npos;
  const bool expects_bf16 = expected_format.find("BF16") != std::string::npos ||
                            expected_format.find("BFLOAT16") != std::string::npos;
  const bool format_ok = expected_format.empty() ||
                         (expects_fp32 && (actual_format.find("FP32") != std::string::npos ||
                                           actual_format.find("FLOAT32") != std::string::npos)) ||
                         (expects_bf16 && (actual_format.find("BF16") != std::string::npos ||
                                           actual_format.find("BFLOAT16") != std::string::npos)) ||
                         (!expects_fp32 && !expects_bf16 && actual_format == expected_format);
  const bool media_ok = actual_media == expected_media;
  const bool width_ok = ingress.width <= 0 || ingress.width == info.width;
  const bool height_ok = ingress.height <= 0 || ingress.height == info.height;
  const bool depth_ok = ingress.depth <= 0 || ingress.depth == info.depth;
  if (media_ok && format_ok && width_ok && height_ok && depth_ok) {
    return;
  }

  std::ostringstream oss;
  oss << (where ? where : "Model") << ": ingress["
      << (ingress.ingress_index >= 0 ? ingress.ingress_index : 0) << "]";
  if (!ingress.source_tensor_name.empty()) {
    oss << " (" << ingress.source_tensor_name << ")";
  }
  oss << " expects media="
      << (ingress.media_type.empty() ? "application/vnd.simaai.tensor" : ingress.media_type);
  if (!ingress.dtype.empty()) {
    oss << " format=" << ingress.dtype;
  }
  if (ingress.width > 0 && ingress.height > 0) {
    oss << " shape=" << ingress.width << "x" << ingress.height;
    if (ingress.depth > 0) {
      oss << "x" << ingress.depth;
    }
  }
  if (!ingress.layout.empty()) {
    oss << " layout=" << ingress.layout;
  }
  oss << ". Received media=" << info.media_type << " format=" << info.format
      << " shape=" << info.width << "x" << info.height << "x" << info.depth;
  throw std::invalid_argument(oss.str());
}

Tensor apply_ingress_tensor_identity(Tensor tensor,
                                     const internal::IngressTensorContract& ingress) {
  const std::string ingress_name =
      !ingress.source_tensor_name.empty()
          ? ingress.source_tensor_name
          : ("ifm" + std::to_string(std::max(ingress.ingress_index, 0)));
  const bool has_explicit_branch_stage = !ingress.branch_ops.empty() &&
                                         !ingress.source_stage.empty() &&
                                         ingress.source_stage != "session_ingress";
  const std::string segment_name = has_explicit_branch_stage ? ingress.source_stage : ingress_name;
  tensor.route.name = ingress_name;
  tensor.route.segment_name = segment_name;
  if (tensor.route.backend_name.empty() || tensor.route.backend_name == "output_tensor") {
    tensor.route.backend_name = segment_name;
  }
  if (ingress.ingress_index >= 0) {
    tensor = internal::remap_tensor_to_consumer_identity(std::move(tensor),
                                                         internal::IngressConsumerTensorIdentity{
                                                             .logical_index = ingress.ingress_index,
                                                             .route_slot = ingress.ingress_index,
                                                         });
  }
  return tensor;
}

TensorList
prepare_ingress_tensors(const TensorList& inputs,
                        const std::vector<internal::IngressTensorContract>& ingress_contracts,
                        const char* where) {
  require_exact_ingress_count(inputs.size(), ingress_contracts, where, "tensor inputs");
  TensorList prepared;
  prepared.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& ingress =
        ingress_contracts.empty() ? internal::IngressTensorContract{} : ingress_contracts[i];
    Tensor tensor = inputs[i];
    InputInfo info = input_info_from_tensor(tensor, false);
    if (!ingress_contracts.empty()) {
      validate_single_tensor_ingress_expectation(ingress, info, where);
      tensor = apply_ingress_tensor_identity(std::move(tensor), ingress);
    }
    prepared.push_back(std::move(tensor));
  }
  return prepared;
}

Sample apply_ingress_sample_identity(Sample sample,
                                     const internal::IngressTensorContract& ingress) {
  const std::string ingress_name =
      !ingress.source_tensor_name.empty()
          ? ingress.source_tensor_name
          : ("ifm" + std::to_string(std::max(ingress.ingress_index, 0)));
  const bool has_explicit_branch_stage = !ingress.branch_ops.empty() &&
                                         !ingress.source_stage.empty() &&
                                         ingress.source_stage != "session_ingress";
  const std::string segment_name = has_explicit_branch_stage ? ingress.source_stage : ingress_name;
  if (sample.stream_label.empty()) {
    sample.stream_label = ingress_name;
  }
  if (sample.segment_name.empty()) {
    sample.segment_name = segment_name;
  }
  if (sample.tensor.has_value()) {
    sample.tensor = apply_ingress_tensor_identity(std::move(*sample.tensor), ingress);
  } else if (sample_has_tensor_list(sample) && sample.tensors.size() == 1U) {
    sample.tensors.front() =
        apply_ingress_tensor_identity(std::move(sample.tensors.front()), ingress);
  }
  return sample;
}

void propagate_common_sample_identity(const Sample& inputs, Sample* out);

Sample bundle_ingress_samples(const Sample& inputs,
                              const std::vector<internal::IngressTensorContract>& ingress_contracts,
                              const char* where) {
  require_exact_ingress_count(inputs.size(), ingress_contracts, where, "sample inputs");
  Sample bundle;
  bundle.kind = SampleKind::Bundle;
  bundle.owned = true;
  bundle.payload_type = PayloadType::Tensor;
  bundle.media_type = "application/vnd.simaai.tensor";
  bundle.fields.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    bundle.fields.push_back(apply_ingress_sample_identity(inputs[i], ingress_contracts[i]));
  }
  propagate_common_sample_identity(inputs, &bundle);
  return bundle;
}

void propagate_common_sample_identity(const Sample& inputs, Sample* out) {
  if (!out || inputs.empty()) {
    return;
  }

  const Sample& first = inputs.front();
  const int64_t frame_id = first.frame_id;
  const int64_t pts_ns = first.pts_ns;
  const int64_t dts_ns = first.dts_ns;
  const int64_t duration_ns = first.duration_ns;
  const std::string stream_id = first.stream_id;

  bool same_frame_id = frame_id >= 0;
  bool same_pts_ns = pts_ns >= 0;
  bool same_dts_ns = dts_ns >= 0;
  bool same_duration_ns = duration_ns >= 0;
  bool same_stream_id = !stream_id.empty();

  for (const auto& sample : inputs) {
    same_frame_id = same_frame_id && sample.frame_id == frame_id;
    same_pts_ns = same_pts_ns && sample.pts_ns == pts_ns;
    same_dts_ns = same_dts_ns && sample.dts_ns == dts_ns;
    same_duration_ns = same_duration_ns && sample.duration_ns == duration_ns;
    same_stream_id = same_stream_id && sample.stream_id == stream_id;
  }

  if (same_frame_id) {
    out->frame_id = frame_id;
  }
  if (same_pts_ns) {
    out->pts_ns = pts_ns;
  }
  if (same_dts_ns) {
    out->dts_ns = dts_ns;
  }
  if (same_duration_ns) {
    out->duration_ns = duration_ns;
  }
  if (same_stream_id) {
    out->stream_id = stream_id;
  }
}

bool sample_represents_multi_ingress_item(const Sample& sample) {
  if (sample.kind == SampleKind::Bundle) {
    return true;
  }
  return sample_has_tensor_list(sample) && sample.tensors.size() > 1U;
}

InputOptions input_options_from_ingress_contract(const internal::IngressTensorContract& ingress) {
  InputOptions opt;
  opt.payload_type = input_type_from_media_type(
      ingress.media_type.empty() ? std::string_view("application/vnd.simaai.tensor")
                                 : std::string_view(ingress.media_type));
  if (!ingress.dtype.empty()) {
    opt.format = ingress.dtype;
  }
  opt.width = ingress.width;
  opt.height = ingress.height;
  opt.depth = ingress.depth;
  opt.max_width = ingress.width;
  opt.max_height = ingress.height;
  opt.max_depth = ingress.depth;
  opt.buffer_name = !ingress.source_tensor_name.empty()
                        ? ingress.source_tensor_name
                        : ("ifm" + std::to_string(std::max(ingress.ingress_index, 0)));
  return opt;
}

InputOptions
overlay_input_options_from_ingress_contract(InputOptions opt,
                                            const internal::IngressTensorContract& ingress) {
  const InputOptions route_opt = input_options_from_ingress_contract(ingress);
  if (route_opt.payload_type != PayloadType::Auto) {
    opt.payload_type = route_opt.payload_type;
  }
  if (!route_opt.format.empty()) {
    opt.format = route_opt.format;
  }
  if (route_opt.width > 0) {
    opt.width = route_opt.width;
  }
  if (route_opt.height > 0) {
    opt.height = route_opt.height;
  }
  if (route_opt.depth > 0) {
    opt.depth = route_opt.depth;
  }
  if (route_opt.max_width > 0) {
    opt.max_width = route_opt.max_width;
  }
  if (route_opt.max_height > 0) {
    opt.max_height = route_opt.max_height;
  }
  if (route_opt.max_depth > 0) {
    opt.max_depth = route_opt.max_depth;
  }
  if (!route_opt.buffer_name.empty()) {
    opt.buffer_name = route_opt.buffer_name;
  }
  return opt;
}

InputOptions override_input_options_from_contract(InputOptions opt,
                                                  const PreprocessContract& contract) {
  if (!contract.media_type.empty()) {
    opt.payload_type = input_type_from_media_type(contract.media_type);
  }
  if (!contract.format.empty()) {
    std::string format = contract.format;
    const std::string media_type = resolve_input_media_type(opt);
    if (pipeline_internal::lower_copy(media_type) == "application/vnd.simaai.tensor") {
      switch (dtype_from_format(format)) {
      case TensorDType::BFloat16:
        format = "EVXX_BFLOAT16";
        break;
      case TensorDType::Float32:
        format = "EVXX_FLOAT32";
        break;
      case TensorDType::Int8:
        format = "EVXX_INT8";
        break;
      case TensorDType::UInt8:
        format = "UINT8";
        break;
      case TensorDType::UInt16:
        format = "UINT16";
        break;
      case TensorDType::Int16:
        format = "INT16";
        break;
      case TensorDType::Int32:
        format = "INT32";
        break;
      case TensorDType::Float64:
        format = "FP64";
        break;
      }
    }
    opt.format = normalize_caps_format_for_media(media_type, format);
  }
  if (contract.width > 0) {
    opt.width = contract.width;
    opt.max_width = contract.width;
  } else if (contract.max_width > 0) {
    opt.max_width = contract.max_width;
  }
  if (contract.height > 0) {
    opt.height = contract.height;
    opt.max_height = contract.height;
  } else if (contract.max_height > 0) {
    opt.max_height = contract.max_height;
  }
  if (contract.depth > 0) {
    opt.depth = contract.depth;
    opt.max_depth = contract.depth;
  } else if (contract.max_depth > 0) {
    opt.max_depth = contract.max_depth;
  }
  return opt;
}

bool session_route_has_ingress_join(const internal::SessionRoutePlan& plan) {
  return std::any_of(plan.ingress_regions.begin(), plan.ingress_regions.end(),
                     [](const internal::RouteRegion& region) {
                       return region.kind == internal::RouteRegionKind::FaninJoin;
                     });
}

// Returns true when the route plan materializes a single multi-IO pre-MLA
// element via a FanoutMap region in pre_regions. This is the signal for the
// bundled-appsrc path: exactly one upstream GstBuffer carrying N tensor
// regions (mirror of the post side's MLA-output buffer that detessdequant
// consumes), instead of N separate appsrcs.
bool session_route_has_multi_io_pre_fan_in(const internal::SessionRoutePlan& plan) {
  return std::any_of(plan.pre_regions.begin(), plan.pre_regions.end(),
                     [](const internal::RouteRegion& region) {
                       return region.kind == internal::RouteRegionKind::FanoutMap;
                     });
}

// True iff the route structurally fans multiple ingress tensors into one
// multi-IO pre-MLA stage. This is always represented as one appsrc sample
// carrying the tensor set into the inline pre stage; it must not externalize
// per-ingress branches.
bool plan_uses_bundled_fan_in(const internal::PreprocessPlannerResult& plan) {
  return session_route_has_multi_io_pre_fan_in(plan.session_route_plan);
}

pipeline_internal::sima::RouteGraphKernelKind
route_graph_kind_from_ordered_op(const internal::OrderedRouteOp& op) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (op.kind) {
  case internal::OrderedRouteOp::Kind::Preproc:
    return GraphKind::Preproc;
  case internal::OrderedRouteOp::Kind::Quant:
    return GraphKind::Quant;
  case internal::OrderedRouteOp::Kind::Tess:
    return GraphKind::Tess;
  case internal::OrderedRouteOp::Kind::QuantTess:
    return GraphKind::QuantTess;
  case internal::OrderedRouteOp::Kind::Cast:
    return GraphKind::Cast;
  case internal::OrderedRouteOp::Kind::CastTess:
    return GraphKind::CastTess;
  case internal::OrderedRouteOp::Kind::Detess:
    return GraphKind::Detess;
  case internal::OrderedRouteOp::Kind::DetessCast:
    return GraphKind::DetessCast;
  case internal::OrderedRouteOp::Kind::DetessDequant:
    return GraphKind::DetessDequant;
  case internal::OrderedRouteOp::Kind::Dequantize:
    return GraphKind::Dequantize;
  case internal::OrderedRouteOp::Kind::BoxDecode:
    return GraphKind::BoxDecode;
  case internal::OrderedRouteOp::Kind::Unpack:
    return GraphKind::Unpack;
  case internal::OrderedRouteOp::Kind::Unknown:
    return GraphKind::Unknown;
  default:
    break;
  }
  return GraphKind::Unknown;
}

std::string ingress_stage_name_from_op(const internal::OrderedRouteOp& op,
                                       std::size_t ingress_index, std::size_t op_index) {
  const auto graph_kind = route_graph_kind_from_ordered_op(op);
  if (graph_kind != pipeline_internal::sima::RouteGraphKernelKind::Unknown) {
    // Route ops intentionally retain source MPK plugin name/id in `op` so exact contract lookup
    // can find graph-authored metadata.  The emitted NEAT stage-id should describe the canonical
    // route operation instead of leaking a component MPK stage name from a fused op.
    return std::string(pipeline_internal::sima::route_graph_kernel_name(graph_kind)) + "_" +
           std::to_string(ingress_index) + "_" + std::to_string(op_index);
  }
  if (!op.plugin_name.empty()) {
    return op.plugin_name;
  }
  if (!op.plugin_id.empty()) {
    return op.plugin_id;
  }
  return "ingress_" + std::to_string(ingress_index) + "_" + std::to_string(op_index);
}

std::shared_ptr<Node> build_ingress_node_from_op(const Model& model,
                                                 const internal::PreprocessPlannerResult& plan,
                                                 const internal::OrderedRouteOp& op,
                                                 const std::string& stage_name,
                                                 const std::string& upstream_name) {
  switch (op.kind) {
  case internal::OrderedRouteOp::Kind::Preproc: {
    PreprocOptions pre_opt = make_preproc_options_from_typed_adapter(
        model, plan, nullptr, stage_name, upstream_name, false);
    return simaai::neat::nodes::Preproc(std::move(pre_opt));
  }
  case internal::OrderedRouteOp::Kind::Quant: {
    QuantOptions q_opt =
        make_quant_options_from_typed_adapter(model, plan, nullptr, stage_name, false, &op);
    return simaai::neat::nodes::Quant(std::move(q_opt));
  }
  case internal::OrderedRouteOp::Kind::Tess: {
    TessOptions t_opt =
        make_tess_options_from_typed_adapter(model, plan, nullptr, stage_name, false, &op);
    return simaai::neat::nodes::Tess(std::move(t_opt));
  }
  case internal::OrderedRouteOp::Kind::QuantTess: {
    QuantTessOptions qt_opt =
        make_quanttess_options_from_typed_adapter(model, plan, nullptr, stage_name, false, &op);
    return simaai::neat::nodes::QuantTess(std::move(qt_opt));
  }
  case internal::OrderedRouteOp::Kind::CastTess: {
    CastTessOptions ct_opt =
        make_casttess_options_from_typed_adapter(model, plan, nullptr, stage_name, false, &op);
    return simaai::neat::nodes::CastTess(std::move(ct_opt));
  }
  case internal::OrderedRouteOp::Kind::Cast: {
    CastOptions cast_opt = make_cast_options_from_typed_adapter(
        &model, stage_name, CastDirection::Fp32ToBf16, internal::ModelLineageStageRole::Preprocess,
        &op, false);
    return simaai::neat::nodes::Cast(std::move(cast_opt));
  }
  case internal::OrderedRouteOp::Kind::Unknown:
  case internal::OrderedRouteOp::Kind::Detess:
  case internal::OrderedRouteOp::Kind::DetessDequant:
  case internal::OrderedRouteOp::Kind::Dequantize:
  case internal::OrderedRouteOp::Kind::BoxDecode:
  case internal::OrderedRouteOp::Kind::Unpack:
    return nullptr;
  }
  return nullptr;
}

Graph build_ingress_branch_graph(const Model& model, const internal::PreprocessPlannerResult& plan,
                                 const internal::IngressTensorContract& ingress,
                                 const Model::RouteOptions& route_opt) {
  Graph branch(route_options_from_model_route_options(route_opt));
  if (ingress.branch_ops.empty()) {
    return branch;
  }

  branch.add(simaai::neat::nodes::Input(input_options_from_ingress_contract(ingress)));
  std::string upstream = input_options_from_ingress_contract(ingress).buffer_name;
  std::size_t emitted = 0U;
  for (std::size_t i = 0; i < ingress.branch_ops.size(); ++i) {
    const auto& op = ingress.branch_ops[i];
    const std::string stage_name = ingress_stage_name_from_op(
        op, static_cast<std::size_t>(std::max(ingress.ingress_index, 0)), i);
    std::shared_ptr<Node> node = build_ingress_node_from_op(model, plan, op, stage_name, upstream);
    if (!node) {
      continue;
    }
    branch.add(std::move(node));
    upstream = stage_name;
    ++emitted;
  }
  if (emitted != 0U) {
    branch.add(simaai::neat::nodes::Output());
  }
  return branch;
}

Sample run_ingress_branch(Graph& branch, Run& runner,
                          const internal::IngressTensorContract& ingress, Sample sample,
                          const char* where) {
  sample = apply_ingress_sample_identity(std::move(sample), ingress);
  if (ingress.branch_ops.empty()) {
    return sample;
  }
  sample = internal::remap_sample_to_consumer_identity(std::move(sample),
                                                       internal::IngressConsumerTensorIdentity{
                                                           .logical_index = 0,
                                                           .physical_index = 0,
                                                           .route_slot = 0,
                                                       });
  if (runner_debug_enabled()) {
    const TensorList branch_inputs = tensors_from_sample(sample, true);
    std::fprintf(stderr, "[model-ingress-debug] branch_input ingress=%d tensors=%zu\n",
                 std::max(ingress.ingress_index, 0), branch_inputs.size());
    for (std::size_t i = 0; i < branch_inputs.size(); ++i) {
      const auto& tensor = branch_inputs[i];
      std::fprintf(stderr,
                   "[model-ingress-debug] branch_input.tensor[%zu] logical=%d physical=%d slot=%d "
                   "memory=%d name=%s segment=%s backend=%s\n",
                   i, tensor.route.logical_index, tensor.route.physical_index,
                   tensor.route.route_slot, tensor.route.memory_index, tensor.route.name.c_str(),
                   tensor.route.segment_name.c_str(), tensor.route.backend_name.c_str());
    }
  }

  RunOptions branch_run_opt;
  branch_run_opt.preset = RunPreset::Reliable;
  branch_run_opt.queue_depth = 1;
  branch_run_opt.overflow_policy = OverflowPolicy::Block;
  branch_run_opt.output_memory = OutputMemory::ZeroCopy;
  if (!runner) {
    runner = branch.build(Sample{sample}, RunMode::Sync, branch_run_opt);
  }
  Sample out = runner.run(Sample{std::move(sample)}, 10000);
  if (out.size() != 1U) {
    std::ostringstream oss;
    oss << (where ? where : "Model ingress") << ": expected exactly one branch output for ingress["
        << std::max(ingress.ingress_index, 0) << "]";
    throw std::runtime_error(oss.str());
  }
  return out.front();
}

Tensor stabilize_ingress_branch_tensor(Tensor tensor) {
  if (!tensor.storage || tensor.storage->kind != StorageKind::GstSample) {
    return tensor;
  }

  const auto original_route = tensor.route;
  const auto original_layout = tensor.layout;
  const auto original_planes = tensor.planes;
  Tensor view = pipeline_internal::tensor_view_from_sample_memory(tensor, tensor.route.memory_index,
                                                                  /*keep_holder=*/true);
  view.route = original_route;
  view.layout = original_layout;
  view.planes = original_planes;
  return view;
}

Sample canonicalize_ingress_branch_output(Sample sample,
                                          const internal::IngressTensorContract& ingress) {
  if (sample.kind == SampleKind::TensorSet || sample.kind == SampleKind::Bundle) {
    TensorList tensors = tensors_from_sample(sample, true);
    for (auto& tensor : tensors) {
      tensor = stabilize_ingress_branch_tensor(std::move(tensor));
      tensor = apply_ingress_tensor_identity(std::move(tensor), ingress);
    }
    Sample out = sample_from_tensors(tensors);
    out.owned = true;
    out.caps_string = sample.caps_string;
    out.stream_label = sample.stream_label;
    out.segment_name = sample.segment_name;
    out.output_index = sample.output_index;
    out.logical_output_index = sample.logical_output_index;
    out.memory_index = sample.memory_index;
    out.route_slot = sample.route_slot;
    out.port_name = sample.port_name;
    out.frame_id = sample.frame_id;
    out.stream_id = sample.stream_id;
    out.input_seq = sample.input_seq;
    out.orig_input_seq = sample.orig_input_seq;
    out.pts_ns = sample.pts_ns;
    out.dts_ns = sample.dts_ns;
    out.duration_ns = sample.duration_ns;
    out.payload_type = sample.payload_type;
    if (!sample.media_type.empty()) {
      out.media_type = sample.media_type;
    }
    if (!sample.format.empty()) {
      out.format = sample.format;
    }
    if (!sample.payload_tag.empty()) {
      out.payload_tag = sample.payload_tag;
    }
    return out;
  }
  if (sample.kind == SampleKind::Tensor && sample.tensor.has_value()) {
    Tensor tensor = stabilize_ingress_branch_tensor(*sample.tensor);
    tensor = apply_ingress_tensor_identity(std::move(tensor), ingress);
    Sample out = sample_from_tensors(TensorList{tensor});
    out.owned = true;
    out.caps_string = sample.caps_string;
    out.stream_label = sample.stream_label;
    out.segment_name = sample.segment_name;
    out.output_index = sample.output_index;
    out.logical_output_index = sample.logical_output_index;
    out.memory_index = sample.memory_index;
    out.route_slot = sample.route_slot;
    out.port_name = sample.port_name;
    out.frame_id = sample.frame_id;
    out.stream_id = sample.stream_id;
    out.input_seq = sample.input_seq;
    out.orig_input_seq = sample.orig_input_seq;
    out.pts_ns = sample.pts_ns;
    out.dts_ns = sample.dts_ns;
    out.duration_ns = sample.duration_ns;
    out.payload_type = sample.payload_type;
    if (!sample.media_type.empty()) {
      out.media_type = sample.media_type;
    }
    if (!sample.format.empty()) {
      out.format = sample.format;
    }
    if (!sample.payload_tag.empty()) {
      out.payload_tag = sample.payload_tag;
    }
    return out;
  }
  return sample;
}

internal::IngressConsumerTensorIdentity joined_tensor_identity_or_fallback(
    const Tensor& tensor, std::size_t index,
    const std::vector<internal::IngressConsumerTensorIdentity>& consumer_identities) {
  internal::IngressConsumerTensorIdentity identity;
  if (index < consumer_identities.size()) {
    identity = consumer_identities[index];
  }
  if (identity.logical_index < 0) {
    identity.logical_index =
        tensor.route.logical_index >= 0 ? tensor.route.logical_index : static_cast<int>(index);
  }
  if (identity.physical_index < 0) {
    identity.physical_index =
        tensor.route.physical_index >= 0 ? tensor.route.physical_index : identity.logical_index;
  }
  if (identity.route_slot < 0) {
    identity.route_slot =
        tensor.route.route_slot >= 0 ? tensor.route.route_slot : identity.logical_index;
  }
  if (identity.memory_index < 0) {
    identity.memory_index = 0;
  }
  return identity;
}

std::vector<internal::IngressConsumerTensorIdentity>
main_route_joined_input_identities(const Model& model) {
  const auto identities_from_static_contract =
      [](const pipeline_internal::sima::MlaStaticContract& contract) {
        std::vector<internal::IngressConsumerTensorIdentity> out;
        int max_local_input_index = -1;
        max_local_input_index =
            contract.inputs.empty() ? -1 : static_cast<int>(contract.inputs.size() - 1U);
        for (const auto& binding : contract.input_bindings) {
          max_local_input_index =
              std::max(max_local_input_index, binding.local_logical_input_index);
        }
        if (max_local_input_index < 0) {
          return out;
        }

        out.resize(static_cast<std::size_t>(max_local_input_index + 1));
        for (std::size_t i = 0; i < out.size(); ++i) {
          auto& identity = out[i];
          identity.logical_index = static_cast<int>(i);
          identity.physical_index = contract.physical_inputs.size() == 1U ? 0 : static_cast<int>(i);
          identity.route_slot = static_cast<int>(i);
        }
        for (const auto& binding : contract.input_bindings) {
          if (binding.local_logical_input_index < 0) {
            continue;
          }
          const std::size_t local_input_index =
              static_cast<std::size_t>(binding.local_logical_input_index);
          if (local_input_index >= out.size()) {
            continue;
          }
          auto& identity = out[local_input_index];
          if (identity.logical_index < 0) {
            identity.logical_index = static_cast<int>(local_input_index);
          }
          if (identity.physical_index < 0) {
            identity.physical_index =
                contract.physical_inputs.size() == 1U ? 0 : static_cast<int>(local_input_index);
          }
          if (identity.route_slot < 0) {
            identity.route_slot = static_cast<int>(local_input_index);
          }
          if (binding.src_physical_output_index >= 0) {
            identity.physical_index = binding.src_physical_output_index;
          }
        }
        for (std::size_t i = 0; i < out.size(); ++i) {
          auto& identity = out[i];
          if (identity.logical_index < 0) {
            identity.logical_index = static_cast<int>(i);
          }
          if (identity.physical_index < 0) {
            identity.physical_index = static_cast<int>(i);
          }
          if (identity.route_slot < 0) {
            identity.route_slot = static_cast<int>(i);
          }
        }
        return out;
      };

  const auto& pack = internal::ModelAccess::pack(model);
  if (const auto& mpk_opt = pack.mpk_contract(); mpk_opt.has_value()) {
    if (const auto* mla_stage = pipeline_internal::sima::get_mla_stage_io_contract(*mpk_opt)) {
      const auto boundary_inputs =
          pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(*mpk_opt);
      const auto published_outputs =
          pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
      const auto logical_outputs =
          pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_opt);
      const auto physical_outputs =
          pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(*mpk_opt);
      auto mla_contract = pipeline_internal::sima::build_mla_static_contract_from_mpk_stage(
          *mla_stage,
          !published_outputs.empty()
              ? published_outputs
              : (logical_outputs.empty() ? mla_stage->output_tensors : logical_outputs),
          physical_outputs.empty() ? mla_stage->output_tensors : physical_outputs,
          !mla_stage->name.empty() ? mla_stage->name : std::string("mla"),
          boundary_inputs.empty() ? nullptr : &boundary_inputs);
      if (mla_stage->input_tensors.size() == 1U && mla_contract.inputs.size() > 1U &&
          mla_contract.physical_inputs.size() > 1U) {
        // The joined ingress transport may pack multiple consumer inputs into one
        // runtime parent buffer, but the consumer-visible identities must still
        // match the MLA contract's logical/physical input selection.
        mla_contract.physical_inputs.resize(1U);
      }
      return identities_from_static_contract(mla_contract);
    }
  }

  std::vector<internal::IngressConsumerTensorIdentity> out;
  const auto infer = internal::ModelAccess::build_infer_nodes(model, false);
  const auto manifest =
      rendered_stage_query::rendered_manifest_from_nodes(infer, "ModelIngressRouteProcessor");
  if (!manifest.has_value()) {
    return out;
  }
  const auto* stage = rendered_stage_query::find_mla_stage(*manifest);
  if (!stage) {
    return out;
  }
  int max_local_input_index = -1;
  for (std::size_t i = 0; i < stage->logical_inputs.size(); ++i) {
    const auto& logical = stage->logical_inputs[i];
    max_local_input_index =
        std::max(max_local_input_index,
                 logical.logical_index >= 0 ? logical.logical_index : static_cast<int>(i));
  }
  for (const auto& binding : stage->input_bindings) {
    max_local_input_index = std::max(max_local_input_index, binding.local_logical_input_index);
  }
  if (max_local_input_index < 0) {
    return out;
  }
  out.resize(static_cast<std::size_t>(max_local_input_index + 1));
  for (std::size_t i = 0; i < stage->logical_inputs.size(); ++i) {
    const auto& logical = stage->logical_inputs[i];
    const int local_input_index =
        logical.logical_index >= 0 ? logical.logical_index : static_cast<int>(i);
    if (local_input_index < 0) {
      continue;
    }
    auto& identity = out[static_cast<std::size_t>(local_input_index)];
    identity.logical_index = logical.logical_index >= 0 ? logical.logical_index : local_input_index;
    identity.physical_index =
        logical.physical_index >= 0 ? logical.physical_index : local_input_index;
    identity.route_slot = logical.logical_index >= 0 ? logical.logical_index : local_input_index;
  }
  for (const auto& binding : stage->input_bindings) {
    if (binding.local_logical_input_index < 0) {
      continue;
    }
    const std::size_t local_input_index =
        static_cast<std::size_t>(binding.local_logical_input_index);
    if (local_input_index >= out.size()) {
      continue;
    }
    auto& identity = out[local_input_index];
    if (binding.src_physical_output_index >= 0) {
      identity.physical_index = binding.src_physical_output_index;
    }
  }
  return out;
}

// True when the model's MLA stage has a native multi-IFM .elf compilation
// (no upstream canonical_op == "pack" producer, N>1 logical inputs). Used by
// ModelIngressRouteProcessor to decide whether to flatten N inputs into a
// single packed parent buffer (monolithic-IFM path) or preserve N distinct
// physical segments (multi-IFM path the firmware needs).
bool main_session_consumer_keeps_distinct_physical_inputs(const Model& model) {
  const auto& pack = internal::ModelAccess::pack(model);
  if (const auto& mpk_opt = pack.mpk_contract(); mpk_opt.has_value()) {
    return pipeline_internal::sima::mla_consumer_keeps_distinct_physical_inputs(*mpk_opt);
  }
  return false;
}

std::string main_route_joined_input_segment_name(const Model& model) {
  const auto& pack = internal::ModelAccess::pack(model);
  if (const auto& mpk_opt = pack.mpk_contract(); mpk_opt.has_value()) {
    const auto boundary_inputs =
        pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(*mpk_opt);
    if (!boundary_inputs.empty()) {
      const auto& tensor = boundary_inputs.front();
      if (!tensor.segment_name.empty()) {
        return tensor.segment_name;
      }
      if (!tensor.name.empty()) {
        return tensor.name;
      }
    }
  }

  const auto infer = internal::ModelAccess::build_infer_nodes(model, false);
  const auto manifest =
      rendered_stage_query::rendered_manifest_from_nodes(infer, "ModelIngressRouteProcessor");
  if (!manifest.has_value()) {
    return {};
  }
  const auto* stage = rendered_stage_query::find_mla_stage(*manifest);
  if (!stage || stage->physical_inputs.empty()) {
    return {};
  }
  return stage->physical_inputs.front().segment_name;
}

bool ingress_consumer_identities_share_one_physical_input(
    const std::vector<internal::IngressConsumerTensorIdentity>& identities,
    std::size_t tensor_count) {
  if (identities.size() != tensor_count || identities.empty()) {
    return false;
  }
  int shared_physical_index = -1;
  for (const auto& identity : identities) {
    if (identity.physical_index < 0) {
      return false;
    }
    if (shared_physical_index < 0) {
      shared_physical_index = identity.physical_index;
      continue;
    }
    if (identity.physical_index != shared_physical_index) {
      return false;
    }
  }
  return true;
}

std::vector<internal::IngressConsumerTensorIdentity> packed_joined_ingress_identities(
    const TensorList& tensors,
    const std::vector<internal::IngressConsumerTensorIdentity>& consumer_identities) {
  std::vector<internal::IngressConsumerTensorIdentity> out;
  out.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto identity = joined_tensor_identity_or_fallback(tensors[i], i, consumer_identities);
    identity.physical_index = 0;
    identity.memory_index = 0;
    if (identity.route_slot < 0) {
      identity.route_slot =
          identity.logical_index >= 0 ? identity.logical_index : static_cast<int>(i);
    }
    out.push_back(identity);
  }
  return out;
}

std::vector<internal::IngressConsumerTensorIdentity> joined_ingress_transport_identities(
    const TensorList& tensors,
    const std::vector<internal::IngressConsumerTensorIdentity>& consumer_identities) {
  std::vector<internal::IngressConsumerTensorIdentity> out;
  out.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto identity = joined_tensor_identity_or_fallback(tensors[i], i, consumer_identities);
    identity.memory_index = static_cast<int>(i);
    if (identity.route_slot < 0) {
      identity.route_slot =
          identity.logical_index >= 0 ? identity.logical_index : static_cast<int>(i);
    }
    out.push_back(identity);
  }
  return out;
}

TensorList materialize_joined_ingress_tensors_impl(
    const TensorList& tensors,
    const std::vector<internal::IngressConsumerTensorIdentity>& consumer_identities,
    const std::string& segment_name, const char* where) {
  if (tensors.empty()) {
    return tensors;
  }

  const auto materialized_tensor_span_bytes = [](const Tensor& tensor) -> std::size_t {
    const std::size_t byte_offset =
        tensor.route.physical_byte_offset >= 0
            ? static_cast<std::size_t>(tensor.route.physical_byte_offset)
            : (tensor.byte_offset >= 0 ? static_cast<std::size_t>(tensor.byte_offset) : 0U);
    if (tensor.storage) {
      const int memory_index = (tensor.route.memory_index >= 0) ? tensor.route.memory_index
                                                                : tensor.route.physical_index;
      if (memory_index >= 0 &&
          static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
        const auto& segment = tensor.storage->sima_segments[static_cast<std::size_t>(memory_index)];
        if (segment.size_bytes > byte_offset) {
          return segment.size_bytes - byte_offset;
        }
      }
      if (tensor.storage->size_bytes > byte_offset) {
        return tensor.storage->size_bytes - byte_offset;
      }
    }
    const Mapping mapping = tensor.map_read();
    if (mapping.data && mapping.size_bytes > 0U) {
      return mapping.size_bytes;
    }
    return tensor.dense_bytes_tight();
  };

  std::size_t total_bytes = 0U;
  std::vector<std::size_t> tensor_bytes;
  tensor_bytes.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    const std::size_t bytes = materialized_tensor_span_bytes(tensor);
    if (bytes == 0U) {
      std::ostringstream oss;
      oss << (where ? where : "Model ingress") << ": joined ingress tensor has unknown byte size";
      throw std::runtime_error(oss.str());
    }
    if (bytes > (std::numeric_limits<std::size_t>::max() - total_bytes)) {
      throw std::runtime_error(std::string(where ? where : "Model ingress") +
                               ": joined ingress byte size overflow");
    }
    tensor_bytes.push_back(bytes);
    total_bytes += bytes;
  }

  auto storage = make_cpu_owned_storage(total_bytes);
  Mapping dst_map = storage->map(MapMode::Write);
  if (!dst_map.data || dst_map.size_bytes < total_bytes) {
    throw std::runtime_error(std::string(where ? where : "Model ingress") +
                             ": joined ingress storage allocation failed");
  }

  TensorList joined;
  joined.reserve(tensors.size());
  std::size_t running_offset = 0U;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const std::size_t bytes = tensor_bytes[i];
    const std::size_t source_offset =
        tensor.route.physical_byte_offset >= 0
            ? static_cast<std::size_t>(tensor.route.physical_byte_offset)
            : (tensor.byte_offset >= 0 ? static_cast<std::size_t>(tensor.byte_offset) : 0U);
    bool copied = false;
    if (tensor.storage) {
      const Mapping storage_map = tensor.storage->map(MapMode::Read);
      if (storage_map.data && storage_map.size_bytes >= source_offset &&
          bytes <= (storage_map.size_bytes - source_offset)) {
        std::memcpy(static_cast<std::uint8_t*>(dst_map.data) + running_offset,
                    static_cast<const std::uint8_t*>(storage_map.data) + source_offset, bytes);
        copied = true;
      }
    }
    if (!copied) {
      const Mapping src_map = tensor.map_read();
      if (src_map.data && src_map.size_bytes >= bytes) {
        std::memcpy(static_cast<std::uint8_t*>(dst_map.data) + running_offset, src_map.data, bytes);
        copied = true;
      }
    }
    if (!copied) {
      std::ostringstream oss;
      oss << (where ? where : "Model ingress")
          << ": failed to map joined ingress tensor payload at index " << i;
      throw std::runtime_error(oss.str());
    }

    Tensor joined_tensor = tensor;
    joined_tensor.storage = storage;
    joined_tensor.byte_offset = static_cast<std::int64_t>(running_offset);
    joined_tensor.route.physical_byte_offset = joined_tensor.byte_offset;
    joined_tensor.route.memory_index = 0;
    const auto identity = joined_tensor_identity_or_fallback(joined_tensor, i, consumer_identities);
    joined_tensor = internal::remap_tensor_to_consumer_identity(std::move(joined_tensor), identity);
    joined_tensor.route.memory_index = 0;
    if (!segment_name.empty()) {
      joined_tensor.route.name = segment_name;
      joined_tensor.route.segment_name = segment_name;
      if (joined_tensor.route.backend_name.empty()) {
        joined_tensor.route.backend_name = segment_name;
      }
    }
    joined.push_back(std::move(joined_tensor));
    running_offset += bytes;
  }
  return joined;
}

bool mla_handoff_contract_is_packed_1d(const internal::PreprocessPlannerResult& plan) {
  const auto& mla = plan.resolved_plan.mla_contract;
  if (pipeline_internal::lower_copy(mla.media_type) != "application/vnd.simaai.tensor") {
    return false;
  }
  const int width = mla.width > 0 ? mla.width : mla.max_width;
  const int height = mla.height > 0 ? mla.height : mla.max_height;
  const int depth = mla.depth > 0 ? mla.depth : mla.max_depth;
  return width > 0 && height == 1 && depth == 1;
}

bool tensor_has_runtime_backing(const Tensor& tensor) {
  return tensor.storage &&
         (tensor.storage->kind == StorageKind::GstSample || !tensor.storage->sima_segments.empty());
}

bool maybe_align_tensor_to_packed_mla_handoff(Tensor* tensor,
                                              const internal::PreprocessPlannerResult& plan) {
  if (!tensor) {
    return false;
  }
  if (session_route_has_ingress_join(plan.session_route_plan)) {
    return false;
  }
  if (plan.session_route_plan.pre_chain.empty()) {
    return false;
  }
  for (const auto op : plan.session_route_plan.pre_chain) {
    switch (op) {
    case internal::SessionPreStageOp::Quant:
    case internal::SessionPreStageOp::Tess:
    case internal::SessionPreStageOp::QuantTess:
    case internal::SessionPreStageOp::Cast:
      break;
    case internal::SessionPreStageOp::Preproc:
      return false;
    }
  }
  if (!mla_handoff_contract_is_packed_1d(plan)) {
    return false;
  }
  // Preserve the semantic tensor descriptor authored by the MPK/runtime contract.
  // Packed MLA transport is a routing/allocation concern, not a semantic shape rewrite.
  return false;
}

std::string ingress_tensor_debug_string(const Tensor& tensor) {
  std::ostringstream oss;
  oss << "dtype=" << static_cast<int>(tensor.dtype) << " layout=" << static_cast<int>(tensor.layout)
      << " shape=[";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << tensor.shape[i];
  }
  oss << "] strides=[";
  for (std::size_t i = 0; i < tensor.strides_bytes.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << tensor.strides_bytes[i];
  }
  oss << "] route{name=" << tensor.route.name << " segment=" << tensor.route.segment_name
      << " logical=" << tensor.route.logical_index << " slot=" << tensor.route.route_slot
      << " memory=" << tensor.route.memory_index << "}";
  if (tensor.storage) {
    oss << " storage{kind=" << static_cast<int>(tensor.storage->kind)
        << " size=" << tensor.storage->size_bytes
        << " holder=" << (tensor.storage->holder ? "1" : "0")
        << " segments=" << tensor.storage->sima_segments.size() << "}";
  } else {
    oss << " storage{kind=-1}";
  }
  return oss.str();
}

class ModelIngressRouteProcessor final : public pipeline_internal::InputRouteProcessor {
public:
  ModelIngressRouteProcessor(Model model, internal::PreprocessPlannerResult plan,
                             Model::RouteOptions route_opt)
      : model_(std::move(model)), plan_(std::move(plan)), route_opt_(std::move(route_opt)),
        ingress_contracts_(normalized_ingress_contracts(plan_.session_route_plan)),
        joined_consumer_identities_(main_route_joined_input_identities(model_)),
        joined_packed_segment_name_(main_route_joined_input_segment_name(model_)),
        consumer_keeps_distinct_physical_inputs_(
            main_session_consumer_keeps_distinct_physical_inputs(model_)),
        is_fan_in_route_(plan_uses_bundled_fan_in(plan_)) {
    if (!is_fan_in_route_) {
      // Legacy multi-ingress: spin up per-ingress branch sessions. The
      // bundled-appsrc fan-in path skips branch_sessions entirely; the
      // user's tensors flow directly through Run::push → InputStream's
      // bundled multi-IO path (the same path Model::graph({include_input=true})
      // uses for fan-in, now reached via Model::build(...) too).
      branch_graphs_.reserve(ingress_contracts_.size());
      for (const auto& ingress : ingress_contracts_) {
        branch_graphs_.push_back(build_ingress_branch_graph(model_, plan_, ingress, route_opt_));
      }
      branch_runners_.resize(ingress_contracts_.size());
    }
  }

  Sample seed_tensors(const TensorList& inputs, const char* where) const override {
    require_exact_ingress_count(inputs.size(), ingress_contracts_, where, "ingress inputs");
    if (is_fan_in_route_) {
      // Bundled fan-in: stamp ingress identity on each user tensor and emit
      // a TensorSet sample. No branch-session run, no clone, no join.
      return build_fan_in_sample(inputs, where);
    }
    if (ingress_contracts_.size() > 1U ||
        session_route_has_ingress_join(plan_.session_route_plan)) {
      // For joined multi-ingress routes, the exact main-session handoff shape is
      // determined by the real ingress-branch outputs. Seed from the real
      // processed sample, then deep-clone it into CPU-owned storage so the
      // build-time seed does not retain live branch-run buffers.
      Sample processed = process_tensors(inputs, where);
      Sample seed = make_owned_seed_sample(processed, where);
      reset_branch_runners();
      return seed;
    }
    return build_seed_sample(where);
  }

  Sample seed_samples(const Sample& inputs, const char* where) const override {
    require_exact_ingress_count(inputs.size(), ingress_contracts_, where, "ingress inputs");
    if (is_fan_in_route_) {
      return build_fan_in_sample_from_samples(inputs, where);
    }
    if (ingress_contracts_.size() > 1U ||
        session_route_has_ingress_join(plan_.session_route_plan)) {
      Sample processed = process_samples(inputs, where);
      Sample seed = make_owned_seed_sample(processed, where);
      reset_branch_runners();
      return seed;
    }
    return build_seed_sample(where);
  }

  Sample process_tensors(const TensorList& inputs, const char* where) const override {
    if (is_fan_in_route_) {
      return build_fan_in_sample(inputs, where);
    }
    TensorList prepared = prepare_ingress_tensors(inputs, ingress_contracts_, where);
    Sample samples;
    samples.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      Sample sample = sample_from_tensors(TensorList{prepared[i]});
      if (!ingress_contracts_.empty()) {
        sample = apply_ingress_sample_identity(std::move(sample), ingress_contracts_[i]);
      }
      samples.push_back(std::move(sample));
    }
    return process_prepared_samples(samples, where);
  }

  Sample process_samples(const Sample& inputs, const char* where) const override {
    if (is_fan_in_route_) {
      if (inputs.size() == 1U && sample_represents_multi_ingress_item(inputs.front()) &&
          ingress_contracts_.size() > 1U) {
        Sample bundle = inputs.front();
        if (bundle.kind == SampleKind::Bundle &&
            bundle.fields.size() == ingress_contracts_.size()) {
          return build_fan_in_sample_from_samples(bundle, where);
        }
      }
      return build_fan_in_sample_from_samples(inputs, where);
    }
    if (inputs.size() == 1U && sample_represents_multi_ingress_item(inputs.front()) &&
        ingress_contracts_.size() > 1U) {
      Sample bundle = inputs.front();
      if (bundle.kind == SampleKind::Bundle && bundle.fields.size() == ingress_contracts_.size()) {
        return process_prepared_samples(bundle, where);
      }
    }
    require_exact_ingress_count(inputs.size(), ingress_contracts_, where, "sample inputs");
    Sample prepared;
    prepared.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      prepared.push_back(apply_ingress_sample_identity(inputs[i], ingress_contracts_[i]));
    }
    return process_prepared_samples(prepared, where);
  }

  // Build a single TensorSet Sample from N user-supplied ingress tensors.
  // Each tensor is stamped with its ingress identity (segment_name etc.) so
  // the bundled multi-IO pre element can route per-binding via the
  // SIMA_TENSOR_SET_META descriptors. No branch processing, no join.
  Sample build_fan_in_sample(const TensorList& inputs, const char* where) const {
    TensorList prepared = prepare_ingress_tensors(inputs, ingress_contracts_, where);
    for (std::size_t i = 0; i < prepared.size() && i < ingress_contracts_.size(); ++i) {
      Sample stamped = sample_from_tensors(TensorList{prepared[i]});
      stamped = apply_ingress_sample_identity(std::move(stamped), ingress_contracts_[i]);
      if (!stamped.tensors.empty()) {
        prepared[i] = stamped.tensors.front();
      }
    }
    stamp_fan_in_packed_parent(&prepared);
    Sample out = sample_from_tensors(prepared);
    if (prepared.size() > 1U) {
      out.segment_name = fan_in_parent_segment_name();
    }
    return out;
  }

  Sample build_fan_in_sample_from_samples(const Sample& inputs, const char* where) const {
    require_exact_ingress_count(inputs.size(), ingress_contracts_, where, "sample inputs");
    TensorList prepared;
    prepared.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      Sample stamped = apply_ingress_sample_identity(inputs[i], ingress_contracts_[i]);
      const TensorList tensors = tensors_from_sample(stamped, true);
      if (tensors.empty()) {
        throw std::runtime_error(std::string(where ? where : "InputRouteProcessor") +
                                 ": ingress sample has no tensors");
      }
      prepared.push_back(tensors.front());
    }
    stamp_fan_in_packed_parent(&prepared);
    Sample out = sample_from_tensors(prepared);
    propagate_common_sample_identity(inputs, &out);
    if (prepared.size() > 1U) {
      out.segment_name = fan_in_parent_segment_name();
    }
    return out;
  }

private:
  std::string fan_in_parent_segment_name() const {
    // ProcessCvu pre-adapter kernels expose one ConfigManager input named
    // "input_tensor". Bundled appsrc fan-in is the appsrc -> first-consumer
    // boundary, so it must describe all logical user inputs as byte-offset views
    // into that single parent even when the final MLA boundary later preserves
    // native distinct physical inputs. The original ingress names remain on
    // route.name/logical_index for TensorSet metadata.
    return "input_tensor";
  }

  void stamp_fan_in_packed_parent(TensorList* tensors) const {
    if (!tensors || tensors->size() <= 1U) {
      return;
    }
    const std::string parent_segment_name = fan_in_parent_segment_name();
    for (std::size_t i = 0; i < tensors->size(); ++i) {
      Tensor& tensor = (*tensors)[i];
      if (tensor.route.logical_index < 0) {
        tensor.route.logical_index = static_cast<int>(i);
      }
      if (tensor.route.route_slot < 0) {
        tensor.route.route_slot = tensor.route.logical_index;
      }
      tensor.route.physical_index = 0;
      tensor.route.memory_index = static_cast<int>(i);
      tensor.route.segment_name = parent_segment_name;
      if (tensor.route.backend_name.empty()) {
        tensor.route.backend_name = parent_segment_name;
      }
    }
  }

  void reset_branch_runners() const {
    for (auto& runner : branch_runners_) {
      runner.close();
      runner = Run{};
    }
  }

  Sample make_owned_seed_sample(const Sample& sample, const char* where) const {
    const TensorList tensors = tensors_from_sample(sample, true);
    TensorList owned;
    owned.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      owned.push_back(tensor.clone());
    }

    Sample out = sample_from_tensors(owned);
    out.payload_type =
        sample.payload_type != PayloadType::Auto ? sample.payload_type : out.payload_type;
    out.media_type = !sample.media_type.empty() ? sample.media_type : out.media_type;
    out.format = sample.format;
    out.payload_tag = sample.payload_tag;
    out.caps_string = sample.caps_string;
    out.port_name = sample.port_name;
    out.segment_name = sample.segment_name;
    out.stream_label = sample.stream_label;
    out.logical_output_index = sample.logical_output_index;
    out.output_index = sample.output_index;
    out.memory_index = sample.memory_index;
    out.route_slot = sample.route_slot;
    out.frame_id = sample.frame_id;
    out.input_seq = sample.input_seq;
    out.orig_input_seq = sample.orig_input_seq;
    if (runner_debug_enabled()) {
      std::fprintf(stderr,
                   "[model-ingress-debug] owned-seed %s tensors=%zu media=%s format=%s payload=%s "
                   "segment=%s\n",
                   where ? where : "InputRouteProcessor::seed", owned.size(),
                   out.media_type.c_str(), out.format.c_str(), out.payload_tag.c_str(),
                   out.segment_name.c_str());
    }
    return out;
  }

  Sample build_seed_sample(const char* where) const {
    // Seed tensors for the joined MLA handoff must not call the singular
    // appsrc API: multi-ingress packs legitimately expose multiple ingress
    // contracts, and the seed is only using the appsrc options as a base
    // before the MLA contract override below becomes authoritative.
    const auto src_opts = model_.input_appsrc_options_list(true);
    InputOptions opt = !src_opts.empty()
                           ? src_opts.front()
                           : internal::ModelAccess::pack(model_).input_appsrc_options(true);
    opt = override_input_options_from_contract(std::move(opt), plan_.resolved_plan.mla_contract);
    Tensor tensor = make_dummy_tensor(opt);
    tensor.route.logical_index = 0;
    tensor.route.physical_index = 0;
    tensor.route.route_slot = 0;
    tensor.route.memory_index = 0;
    const std::string segment_name =
        !joined_packed_segment_name_.empty() ? joined_packed_segment_name_ : std::string("ifm0");
    tensor.route.name = segment_name;
    tensor.route.segment_name = segment_name;
    tensor.route.backend_name = segment_name;
    (void)maybe_align_tensor_to_packed_mla_handoff(&tensor, plan_);
    if (!joined_consumer_identities_.empty()) {
      tensor = internal::remap_tensor_to_consumer_identity(std::move(tensor),
                                                           joined_consumer_identities_.front());
      tensor.route.memory_index = 0;
      if (!segment_name.empty()) {
        tensor.route.name = segment_name;
        tensor.route.segment_name = segment_name;
        tensor.route.backend_name = segment_name;
      }
    }
    if (runner_debug_enabled()) {
      const std::string desc = ingress_tensor_debug_string(tensor);
      std::fprintf(stderr, "[model-ingress-debug] build-seed %s %s\n",
                   where ? where : "InputRouteProcessor::seed", desc.c_str());
    }
    Sample out = sample_from_tensors(TensorList{std::move(tensor)});
    out.owned = true;
    out.payload_type = PayloadType::Tensor;
    out.media_type = "application/vnd.simaai.tensor";
    if (mla_handoff_contract_is_packed_1d(plan_)) {
      out.payload_tag = format_tag_to_string(FormatTag::ByteStream);
      out.format = out.payload_tag;
    }
    return out;
  }

  Sample process_prepared_samples(const Sample& prepared, const char* where) const {
    require_exact_ingress_count(prepared.size(), ingress_contracts_, where, "ingress inputs");
    Sample branch_outputs;
    branch_outputs.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      branch_outputs.push_back(canonicalize_ingress_branch_output(
          run_ingress_branch(branch_graphs_[i], branch_runners_[i], ingress_contracts_[i],
                             prepared[i], where),
          ingress_contracts_[i]));
      if (runner_debug_enabled()) {
        const TensorList branch_tensors = tensors_from_sample(branch_outputs.back(), true);
        std::fprintf(stderr, "[model-ingress-debug] branch[%zu] source=%s tensors=%zu\n", i,
                     ingress_contracts_[i].source_tensor_name.c_str(), branch_tensors.size());
        for (std::size_t tensor_index = 0; tensor_index < branch_tensors.size(); ++tensor_index) {
          const std::string desc = ingress_tensor_debug_string(branch_tensors[tensor_index]);
          std::fprintf(stderr, "[model-ingress-debug] branch[%zu].tensor[%zu] %s\n", i,
                       tensor_index, desc.c_str());
        }
      }
    }
    TensorList joined_tensors;
    std::vector<internal::IngressConsumerTensorIdentity> joined_tensor_identities;
    for (std::size_t i = 0; i < branch_outputs.size(); ++i) {
      const auto& output = branch_outputs[i];
      TensorList tensors = tensors_from_sample(output, true);
      if (tensors.size() == 1U && i < joined_consumer_identities_.size()) {
        joined_tensor_identities.push_back(joined_consumer_identities_[i]);
      }
      joined_tensors.insert(joined_tensors.end(), tensors.begin(), tensors.end());
    }
    if (joined_tensor_identities.size() != joined_tensors.size()) {
      joined_tensor_identities.clear();
      if (joined_consumer_identities_.size() == joined_tensors.size()) {
        joined_tensor_identities = joined_consumer_identities_;
      }
    }
    // Phase 2 gate: do not collapse N>1 ingress tensors into a packed parent
    // segment when the consumer's compiled .elf demands distinct physical
    // input segments (multi-IFM dispatch). The transport-identity branch
    // below will preserve each tensor's per-input memory_index instead.
    const bool packed_joined_handoff =
        joined_tensors.size() > 1U && session_route_has_ingress_join(plan_.session_route_plan) &&
        !joined_packed_segment_name_.empty() && !consumer_keeps_distinct_physical_inputs_;
    if (packed_joined_handoff) {
      const auto transport_identities =
          packed_joined_ingress_identities(joined_tensors, joined_tensor_identities);
      for (std::size_t i = 0; i < joined_tensors.size() && i < transport_identities.size(); ++i) {
        joined_tensors[i] = internal::remap_tensor_to_consumer_identity(
            std::move(joined_tensors[i]), transport_identities[i]);
        joined_tensors[i].route.memory_index = 0;
        joined_tensors[i].route.physical_index = 0;
        if (!joined_packed_segment_name_.empty()) {
          joined_tensors[i].route.name = joined_packed_segment_name_;
          joined_tensors[i].route.segment_name = joined_packed_segment_name_;
          joined_tensors[i].route.backend_name = joined_packed_segment_name_;
        }
      }
    } else if (joined_tensors.size() > 1U && ingress_consumer_identities_share_one_physical_input(
                                                 joined_tensor_identities, joined_tensors.size())) {
      const auto transport_identities =
          joined_ingress_transport_identities(joined_tensors, joined_tensor_identities);
      for (std::size_t i = 0; i < joined_tensors.size() && i < transport_identities.size(); ++i) {
        joined_tensors[i] = internal::remap_tensor_to_consumer_identity(
            std::move(joined_tensors[i]), transport_identities[i]);
      }
    } else if (joined_tensors.size() > 1U) {
      const auto transport_identities =
          joined_ingress_transport_identities(joined_tensors, joined_tensor_identities);
      for (std::size_t i = 0; i < joined_tensors.size() && i < transport_identities.size(); ++i) {
        joined_tensors[i] = internal::remap_tensor_to_consumer_identity(
            std::move(joined_tensors[i]), transport_identities[i]);
      }
    } else if (joined_tensors.size() == 1U) {
      if (mla_handoff_contract_is_packed_1d(plan_) &&
          tensor_has_runtime_backing(joined_tensors.front())) {
        std::vector<internal::IngressConsumerTensorIdentity> materialize_identities;
        if (!joined_consumer_identities_.empty()) {
          materialize_identities.push_back(joined_consumer_identities_.front());
        }
        joined_tensors = materialize_joined_ingress_tensors_impl(
            joined_tensors, materialize_identities, joined_packed_segment_name_, where);
      }
      (void)maybe_align_tensor_to_packed_mla_handoff(&joined_tensors.front(), plan_);
      if (mla_handoff_contract_is_packed_1d(plan_)) {
        joined_tensors.front().semantic.tess.reset();
      }
      if (!joined_consumer_identities_.empty()) {
        joined_tensors.front() = internal::remap_tensor_to_consumer_identity(
            std::move(joined_tensors.front()), joined_consumer_identities_.front());
      }
      if (!joined_packed_segment_name_.empty()) {
        joined_tensors.front().route.name = joined_packed_segment_name_;
        joined_tensors.front().route.segment_name = joined_packed_segment_name_;
        joined_tensors.front().route.backend_name = joined_packed_segment_name_;
      }
    }
    if (runner_debug_enabled()) {
      std::fprintf(stderr, "[model-ingress-debug] joined_tensors=%zu\n", joined_tensors.size());
      for (std::size_t i = 0; i < joined_tensors.size(); ++i) {
        const std::string desc = ingress_tensor_debug_string(joined_tensors[i]);
        std::fprintf(stderr, "[model-ingress-debug] joined.tensor[%zu] %s\n", i, desc.c_str());
      }
    }
    Sample joined = sample_from_tensors(joined_tensors);
    propagate_common_sample_identity(branch_outputs, &joined);
    joined.owned = true;
    joined.payload_type = PayloadType::Tensor;
    joined.media_type = "application/vnd.simaai.tensor";
    if (mla_handoff_contract_is_packed_1d(plan_)) {
      joined.payload_tag = format_tag_to_string(FormatTag::ByteStream);
      joined.format = joined.payload_tag;
    }
    if (packed_joined_handoff && !joined_packed_segment_name_.empty()) {
      joined.segment_name = joined_packed_segment_name_;
    }
    return joined;
  }

  Model model_;
  internal::PreprocessPlannerResult plan_;
  Model::RouteOptions route_opt_;
  std::vector<internal::IngressTensorContract> ingress_contracts_;
  std::vector<internal::IngressConsumerTensorIdentity> joined_consumer_identities_;
  std::string joined_packed_segment_name_;
  // True when the MLA .elf is a multi-IFM compilation (no upstream pack stage).
  // When true, the joined-handoff materialization must preserve each input's
  // memory_index/physical_index/segment_name instead of coalescing them into
  // a packed parent (the firmware addresses each placeholder independently).
  bool consumer_keeps_distinct_physical_inputs_ = false;
  // Whether this route is a multi-IO pre-MLA fan-in handled via the bundled
  // appsrc path (set by ctor, immutable thereafter). When true, ctor skips
  // branch_graphs_ construction and the four virtual overrides emit a
  // single TensorSet Sample directly instead of running per-ingress branches
  // and joining. Gated on SIMA_NEAT_MULTI_IO_BUNDLED_APPSRC.
  bool is_fan_in_route_ = false;
  mutable std::vector<Graph> branch_graphs_;
  mutable std::vector<Run> branch_runners_;
};

struct Range {
  int min = 0;
  int max = 0;
  bool valid = false;
};

struct CapsInfo {
  Range width;
  Range height;
  Range depth;
  std::vector<std::string> formats;
};

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> nums;
  int cur = 0;
  bool in_num = false;
  for (char c : s) {
    if (std::isdigit(static_cast<unsigned char>(c))) {
      cur = cur * 10 + (c - '0');
      in_num = true;
    } else if (in_num) {
      nums.push_back(cur);
      cur = 0;
      in_num = false;
    }
  }
  if (in_num)
    nums.push_back(cur);
  return nums;
}

Range parse_range(const nlohmann::json& values) {
  Range out;
  std::string s;
  if (values.is_string()) {
    s = values.get<std::string>();
  } else if (values.is_number_integer()) {
    out.min = out.max = values.get<int>();
    out.valid = true;
    return out;
  } else {
    return out;
  }
  const std::vector<int> nums = parse_ints(s);
  if (nums.empty())
    return out;
  out.min = nums[0];
  out.max = (nums.size() > 1) ? nums[1] : nums[0];
  if (out.max < out.min)
    std::swap(out.max, out.min);
  out.valid = true;
  return out;
}

std::vector<std::string> parse_value_list(const nlohmann::json& values) {
  std::vector<std::string> out;
  if (values.is_string()) {
    std::string s = values.get<std::string>();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
                                          [](unsigned char ch) { return !std::isspace(ch); }));
      tok.erase(
          std::find_if(tok.rbegin(), tok.rend(), [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          tok.end());
      if (!tok.empty())
        out.push_back(upper_copy(tok));
    }
    return out;
  }
  if (values.is_array()) {
    for (const auto& v : values) {
      if (v.is_string())
        out.push_back(upper_copy(v.get<std::string>()));
    }
  }
  return out;
}

CapsInfo parse_caps_info(const nlohmann::json& cfg) {
  CapsInfo out;
  if (!cfg.contains("caps") || !cfg["caps"].is_object())
    return out;
  const auto& caps = cfg["caps"];
  if (!caps.contains("sink_pads") || !caps["sink_pads"].is_array() || caps["sink_pads"].empty()) {
    return out;
  }
  const auto& pad = caps["sink_pads"][0];
  if (!pad.is_object() || !pad.contains("params") || !pad["params"].is_array()) {
    return out;
  }
  for (const auto& param : pad["params"]) {
    if (!param.is_object())
      continue;
    const std::string name = param.value("name", "");
    if (name.empty())
      continue;
    if (!param.contains("values"))
      continue;
    if (name == "width")
      out.width = parse_range(param["values"]);
    else if (name == "height")
      out.height = parse_range(param["values"]);
    else if (name == "depth")
      out.depth = parse_range(param["values"]);
    else if (name == "format")
      out.formats = parse_value_list(param["values"]);
  }
  return out;
}

void validate_range(const Range& range, int value, const char* label, const char* field) {
  if (!range.valid || value <= 0)
    return;
  if (value < range.min || value > range.max) {
    std::ostringstream ss;
    ss << "Model: " << label << " " << field << " out of bounds: " << value << " (allowed "
       << range.min << "-" << range.max << ")";
    throw std::runtime_error(ss.str());
  }
}

void validate_format(const std::vector<std::string>& formats, const std::string& fmt,
                     const char* label) {
  if (formats.empty() || fmt.empty())
    return;
  const std::string up = upper_copy(fmt);
  for (const auto& allowed : formats) {
    if (allowed == up)
      return;
    if ((allowed == "GRAY" || allowed == "GRAY8") && (up == "GRAY" || up == "GRAY8"))
      return;
  }
  std::ostringstream ss;
  ss << "Model: " << label << " format not allowed: " << fmt;
  throw std::runtime_error(ss.str());
}

void validate_input_against_caps(const InputInfo& info, const nlohmann::json& cfg,
                                 const char* label) {
  CapsInfo caps = parse_caps_info(cfg);
  Range default_range{1, 4096, true};
  if (!caps.width.valid)
    caps.width = default_range;
  if (!caps.height.valid)
    caps.height = default_range;
  if (!caps.depth.valid && info.depth > 0)
    caps.depth = default_range;
  validate_range(caps.width, info.width, label, "width");
  validate_range(caps.height, info.height, label, "height");
  validate_range(caps.depth, info.depth, label, "depth");
  validate_format(caps.formats, info.format, label);
}

internal::PostRouteStageKind parse_selected_post_kind(std::string kind) {
  kind = pipeline_internal::lower_copy(std::move(kind));
  if (kind == "boxdecode" || kind == "boxdecoder") {
    return internal::PostRouteStageKind::BoxDecode;
  }
  if (kind == "detessdequant") {
    return internal::PostRouteStageKind::DetessDequant;
  }
  if (kind == "detessellate" || kind == "detess") {
    return internal::PostRouteStageKind::Detess;
  }
  if (kind == "dequantize" || kind == "dequant") {
    return internal::PostRouteStageKind::Dequantize;
  }
  if (kind == "cast") {
    return internal::PostRouteStageKind::Cast;
  }
  if (kind == "none" || kind.empty()) {
    return internal::PostRouteStageKind::None;
  }
  return internal::PostRouteStageKind::Unknown;
}

internal::PreprocessPlannerResult build_preprocess_plan(const std::string& tar_gz,
                                                        const Model::Options& options) {
  // Build capabilities from the extracted MPK graph templates first, then plan.
  internal::ModelPack capability_pack(tar_gz);
  const internal::PreprocessCapabilities capabilities =
      internal::inspect_preprocess_capabilities(capability_pack);
  internal::PreprocessPlannerResult plan = internal::plan_preprocess(options, capabilities);

  const internal::RouteCapability capability =
      internal::extract_route_capability(capability_pack, plan);
  const internal::ModelSemantics model_semantics = internal::build_model_semantics(capability_pack);
  internal::SessionRoutePlan session_route_plan =
      internal::build_route_plan(options, model_semantics, &capability, &capability_pack);
  const internal::RouteSelection route = internal::plan_route_selection(options, plan, capability);
  if (route.ambiguous) {
    session_route_plan.diagnostics.push_back("route-ambiguous=" + route.ambiguity_reason);
  }
  plan.session_route_plan = session_route_plan;
  populate_effective_model_managed_contract_fields(&plan, capability_pack);

  // Typed route owns ingress contract. Adapter-first pre chains must ingest
  // tensor data at appsrc (not legacy video caps from preproc family probing).
  if (plan.session_route_plan.include_pre_stage && !plan.session_route_plan.pre_chain.empty()) {
    const auto first_pre_stage = plan.session_route_plan.pre_chain.front();
    if (first_pre_stage != internal::SessionPreStageOp::Preproc) {
      const auto route_ingress_contracts = normalized_ingress_contracts(plan.session_route_plan);
      const auto* single_ingress = maybe_single_ingress_contract(route_ingress_contracts);
      const std::string unified_ingress_dtype = unify_ingress_dtype(route_ingress_contracts);
      const auto dtype_is_fp32 = [](const std::string& raw) {
        const std::string token = upper_copy(raw);
        return token.find("FP32") != std::string::npos ||
               token.find("FLOAT32") != std::string::npos;
      };
      const auto dtype_is_bf16 = [](const std::string& raw) {
        const std::string token = upper_copy(raw);
        return token.find("BF16") != std::string::npos ||
               token.find("BFLOAT16") != std::string::npos;
      };

      plan.modelpack_media_type = (single_ingress != nullptr && !single_ingress->media_type.empty())
                                      ? single_ingress->media_type
                                      : "application/vnd.simaai.tensor";
      // Quantized pre-adapter chains ingest FP32 and own quantization in-kernel.
      // Tess-only BF16 chains ingest BF16 directly.
      const bool needs_fp32_ingress =
          first_pre_stage == internal::SessionPreStageOp::Quant ||
          first_pre_stage == internal::SessionPreStageOp::QuantTess ||
          first_pre_stage == internal::SessionPreStageOp::Cast ||
          dtype_is_fp32(unified_ingress_dtype) ||
          plan.session_route_plan.model_managed_route_flags.quant_needed;
      if (needs_fp32_ingress) {
        plan.modelpack_format = "EVXX_FLOAT32";
      } else if (!unified_ingress_dtype.empty() && !dtype_is_bf16(unified_ingress_dtype)) {
        plan.modelpack_format = unified_ingress_dtype;
      } else {
        // BF16 appsrc caps must use EVXX alias tokens for processcvu negotiation.
        plan.modelpack_format = "EVXX_BFLOAT16";
      }
      if (plan.modelpack_input_depth <= 0) {
        if (single_ingress != nullptr && single_ingress->depth > 0) {
          plan.modelpack_input_depth = single_ingress->depth;
        } else if (capability.mla_input_dims.depth > 0) {
          plan.modelpack_input_depth = capability.mla_input_dims.depth;
        } else {
          plan.modelpack_input_depth = 3;
        }
      }
      if (plan.modelpack_max_depth <= 0) {
        if (plan.modelpack_input_depth > 0) {
          plan.modelpack_max_depth = plan.modelpack_input_depth;
        } else {
          plan.modelpack_max_depth = std::max(3, plan.modelpack_input_depth);
        }
      }
      // Keep adapter-first ingress limits dynamic unless the user pinned them.
      // MLA dims are processing target dims, not necessarily ingress max dims.
      int dynamic_max_w =
          (options.preprocess.input_max_width > 0) ? options.preprocess.input_max_width : 1920;
      int dynamic_max_h =
          (options.preprocess.input_max_height > 0) ? options.preprocess.input_max_height : 1080;
      if (first_pre_stage != internal::SessionPreStageOp::Preproc) {
        if (single_ingress != nullptr && single_ingress->width > 0 && single_ingress->height > 0) {
          dynamic_max_w = single_ingress->width;
          dynamic_max_h = single_ingress->height;
        } else if (capability.mla_input_dims.width > 0 && capability.mla_input_dims.height > 0) {
          dynamic_max_w = capability.mla_input_dims.width;
          dynamic_max_h = capability.mla_input_dims.height;
        }
      }
      const int prior_max_w = plan.modelpack_max_width;
      const int prior_max_h = plan.modelpack_max_height;
      bool widened_w = false;
      bool widened_h = false;
      const int ingress_width_for_limits = (single_ingress != nullptr && single_ingress->width > 0)
                                               ? single_ingress->width
                                               : capability.mla_input_dims.width;
      const int ingress_height_for_limits =
          (single_ingress != nullptr && single_ingress->height > 0)
              ? single_ingress->height
              : capability.mla_input_dims.height;
      const bool sane_mla_width = ingress_width_for_limits > 0 && ingress_width_for_limits <= 8192;
      if (plan.modelpack_max_width <= 0 ||
          (sane_mla_width && plan.modelpack_max_width <= ingress_width_for_limits)) {
        plan.modelpack_max_width = dynamic_max_w;
        widened_w = true;
      }
      const bool sane_mla_height =
          ingress_height_for_limits > 0 && ingress_height_for_limits <= 8192;
      if (plan.modelpack_max_height <= 0 ||
          (sane_mla_height && plan.modelpack_max_height <= ingress_height_for_limits)) {
        plan.modelpack_max_height = dynamic_max_h;
        widened_h = true;
      }
      if (runner_debug_enabled()) {
        std::fprintf(stderr,
                     "[model-route-debug] adapter_ingress_limits first_pre=%d quant_needed=%d "
                     "tess_needed=%d cast_needed=%d ingress_dims=%dx%d mla_dims=%dx%d "
                     "prior_max=%dx%d dynamic_max=%dx%d "
                     "resolved_max=%dx%d widened={w:%d,h:%d}\n",
                     static_cast<int>(first_pre_stage),
                     plan.session_route_plan.preproc_context.pre_quant_needed ? 1 : 0,
                     plan.session_route_plan.preproc_context.pre_tess_needed ? 1 : 0,
                     plan.session_route_plan.model_managed_route_flags.pre_cast_needed ? 1 : 0,
                     ingress_width_for_limits, ingress_height_for_limits,
                     capability.mla_input_dims.width, capability.mla_input_dims.height, prior_max_w,
                     prior_max_h, dynamic_max_w, dynamic_max_h, plan.modelpack_max_width,
                     plan.modelpack_max_height, widened_w ? 1 : 0, widened_h ? 1 : 0);
      }
      if (single_ingress != nullptr && single_ingress->depth > 0) {
        plan.modelpack_input_depth = single_ingress->depth;
        plan.modelpack_max_depth = single_ingress->depth;
      } else if (capability.mla_input_dims.depth > 0) {
        plan.modelpack_input_depth = capability.mla_input_dims.depth;
        plan.modelpack_max_depth = capability.mla_input_dims.depth;
      }
    }
  }

  plan.pipeline_type = plan.session_route_plan.pipeline_type;
  plan.include_preprocess_stage = plan.session_route_plan.include_pre_stage;
  plan.include_postprocess_stage = plan.session_route_plan.include_post_stage;
  plan.infer_only_route = plan.session_route_plan.infer_only;
  plan.mla_tessellation = route.effective.mla_tessellation;
  plan.route_selected_post_kind =
      post_route_stage_kind_name(plan.session_route_plan.selected_post_kind);
  plan.route_cast_symmetry_ok = plan.session_route_plan.cast_symmetry_ok;

  if (!plan.session_route_plan.include_pre_stage) {
    if (!capability.mla_input_media_type.empty()) {
      plan.modelpack_media_type = capability.mla_input_media_type;
    }
    if (!capability.mla_input_dtype_raw.empty()) {
      plan.modelpack_format = capability.mla_input_dtype_raw;
    }
    if (capability.mla_input_dims.depth > 0) {
      plan.modelpack_input_depth = capability.mla_input_dims.depth;
      plan.modelpack_max_depth = capability.mla_input_dims.depth;
    }
    if (capability.mla_input_dims.width > 0) {
      plan.modelpack_max_width = capability.mla_input_dims.width;
    }
    if (capability.mla_input_dims.height > 0) {
      plan.modelpack_max_height = capability.mla_input_dims.height;
    }
  }

  if (!plan.include_preprocess_stage) {
    plan.resolved_plan.enabled = false;
    plan.resolved_plan.graph_family = PreprocessGraphFamily::Disabled;
    plan.resolved_plan.graph_kernel = "disabled";
    plan.resolved_plan.graph_config_path.clear();
  }

  plan.resolved_plan.ingress_contracts.clear();
  const auto route_ingress_contracts = normalized_ingress_contracts(plan.session_route_plan);
  if (!route_ingress_contracts.empty()) {
    plan.resolved_plan.ingress_contracts.reserve(route_ingress_contracts.size());
    for (const auto& ingress : route_ingress_contracts) {
      PreprocessContract contract;
      contract.media_type =
          !ingress.media_type.empty() ? ingress.media_type : plan.modelpack_media_type;
      contract.format = !ingress.dtype.empty() ? ingress.dtype : plan.modelpack_format;
      contract.width = ingress.width;
      contract.height = ingress.height;
      contract.depth = ingress.depth;
      contract.max_width = ingress.width > 0 ? ingress.width : plan.modelpack_max_width;
      contract.max_height = ingress.height > 0 ? ingress.height : plan.modelpack_max_height;
      contract.max_depth = ingress.depth > 0 ? ingress.depth : plan.modelpack_max_depth;
      plan.resolved_plan.ingress_contracts.push_back(std::move(contract));
    }
  }

  plan.route_diagnostics = capability.evidence;
  plan.route_diagnostics.push_back(
      std::string("route_plan{use_preproc=") + (plan.session_route_plan.use_preproc ? "1" : "0") +
      ",include_pre=" + (plan.session_route_plan.include_pre_stage ? "1" : "0") +
      ",include_post=" + (plan.session_route_plan.include_post_stage ? "1" : "0") +
      ",boxdecode_selected=" + (plan.session_route_plan.boxdecode_selected ? "1" : "0") +
      ",pre_quant_needed=" +
      (plan.session_route_plan.preproc_context.pre_quant_needed ? "1" : "0") + ",pre_tess_needed=" +
      (plan.session_route_plan.preproc_context.pre_tess_needed ? "1" : "0") + ",pre_cast_needed=" +
      (plan.session_route_plan.model_managed_route_flags.pre_cast_needed ? "1" : "0") + "}");
  plan.route_diagnostics.push_back(std::string("route_plan_pre_chain_size=") +
                                   std::to_string(plan.session_route_plan.pre_chain.size()));
  plan.route_diagnostics.push_back(std::string("route_plan_post_chain_size=") +
                                   std::to_string(plan.session_route_plan.post_chain.size()));
  if (!route_ingress_contracts.empty()) {
    std::ostringstream ingress_oss;
    ingress_oss << "ingress_contracts=[";
    for (std::size_t i = 0; i < route_ingress_contracts.size(); ++i) {
      if (i) {
        ingress_oss << ", ";
      }
      const auto& ingress = route_ingress_contracts[i];
      ingress_oss << ingress.dtype << "/" << ingress.layout << "/" << ingress.height << "x"
                  << ingress.width << "x" << ingress.depth << "@" << ingress.source_stage;
    }
    ingress_oss << "]";
    plan.route_diagnostics.push_back(ingress_oss.str());
  }
  plan.route_diagnostics.insert(plan.route_diagnostics.end(),
                                plan.session_route_plan.diagnostics.begin(),
                                plan.session_route_plan.diagnostics.end());
  if (route.ambiguous) {
    plan.route_diagnostics.push_back(std::string("route_ambiguous=") + route.ambiguity_reason);
  }
  plan.route_diagnostics.push_back(internal::route_selection_debug_string(route));
  plan.route_diagnostics.insert(plan.route_diagnostics.end(), route.diagnostics.begin(),
                                route.diagnostics.end());
  for (const auto& diag : plan.route_diagnostics) {
    plan.resolved_plan.warnings.push_back(std::string("route: ") + diag);
  }

  return plan;
}

internal::PreprocessPlannerResult
build_preprocess_plan_with_verbosity(const std::string& tar_gz, const Model::Options& options) {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(options.verbose);
  auto verbose_guard = pipeline_internal::ux::acquire_runtime_verbosity(options.verbose);
  return build_preprocess_plan(tar_gz, options);
}

} // namespace

struct Model::Impl {
  std::string source_path;
  Options options;
  internal::PreprocessPlannerResult preprocess_plan;
  internal::ModelPack pack;
  mutable std::optional<internal::ModelPack> sync_pack;
  std::string model_id;

  mutable std::mutex sync_mu;
  mutable bool sync_ready = false;
  mutable InputKey sync_key{};
  mutable Runner sync_runner{};

  Impl(const std::string& tar_gz, Options opt)
      : source_path(tar_gz), options(std::move(opt)),
        preprocess_plan(build_preprocess_plan_with_verbosity(tar_gz, options)),
        pack(tar_gz, preprocess_plan.modelpack_media_type, preprocess_plan.modelpack_format,
             preprocess_plan.modelpack_input_depth, preprocess_plan.modelpack_max_width,
             preprocess_plan.modelpack_max_height, preprocess_plan.modelpack_max_depth,
             preprocess_plan.normalize, preprocess_plan.mean, preprocess_plan.stddev,
             /*preproc_next_cpu=*/{}, preprocess_plan.pipeline_type, options.upstream_name,
             /*num_buffers_cvu=*/4,
             /*num_buffers_mla=*/4,
             /*queue_max_buffers=*/0,
             /*queue_max_time_ns=*/-1,
             /*queue_leaky=*/{}, options.name_suffix,
             to_internal_terminal_policy(options.inference_terminal),
             options.cleanup_extracted_model_data) {
    pipeline_internal::ux::emit_line(options.verbose, "Model loaded");
    pipeline_internal::ux::ScopedVerboseContext verbose_ctx(options.verbose);
    auto verbose_guard = pipeline_internal::ux::acquire_runtime_verbosity(options.verbose);
    const auto processcvu_pre_stage_selected = [&]() -> std::optional<bool> {
      const auto& pre_chain = preprocess_plan.session_route_plan.pre_chain;
      if (pre_chain.empty()) {
        return std::nullopt;
      }
      const auto first = pre_chain.front();
      if (first == internal::SessionPreStageOp::Preproc ||
          first == internal::SessionPreStageOp::Quant ||
          first == internal::SessionPreStageOp::Tess ||
          first == internal::SessionPreStageOp::QuantTess) {
        return true;
      }
      return std::nullopt;
    }();
    pack.set_model_managed_stage_facts(
        /*processcvu_preproc_single_output_handoff=*/processcvu_pre_stage_selected,
        convert_model_managed_route_flags(
            preprocess_plan.session_route_plan.model_managed_route_flags));
    auto& rp = preprocess_plan.resolved_plan;
    const std::string pre_name = resolved_pre_stage_name(pack, preprocess_plan);
    const std::string infer_upstream = preprocess_plan.session_route_plan.include_pre_stage
                                           ? (pre_name.empty() ? std::string("decoder") : pre_name)
                                           : std::string("decoder");
    std::string mla_input_dtype;
    TensorLayout mla_input_layout = TensorLayout::Unknown;
    stages::TensorDims mla_dims;
    const auto infer_stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
    for (const auto& fact : infer_stage_facts) {
      if (!fact.mla_compiled.has_value() ||
          fact.mla_compiled->runtime_contract.logical_inputs.empty()) {
        continue;
      }
      const auto& logical_input = fact.mla_compiled->runtime_contract.logical_inputs.front();
      if (!logical_input.dtype.empty()) {
        mla_input_dtype = logical_input.dtype;
      }
      mla_input_layout =
          rendered_stage_query::layout_projection_from_contract_format(logical_input.layout);
      mla_dims = dims_from_mla_logical_contract_shape(logical_input.shape, mla_input_layout);
      break;
    }
    if (mla_dims.width <= 0 || mla_dims.height <= 0 || mla_dims.depth <= 0 ||
        mla_input_dtype.empty()) {
      const auto& mpk_opt = pack.mpk_contract();
      if (!mpk_opt.has_value()) {
        // fall through to rendered infer-block fallback below
      } else if (const auto* mla_stage =
                     pipeline_internal::sima::get_mla_stage_io_contract(*mpk_opt)) {
        const auto boundary_inputs =
            pipeline_internal::sima::get_mla_boundary_physical_inputs_contract(*mpk_opt);
        const auto published_outputs =
            pipeline_internal::sima::get_mla_published_outputs_contract(*mpk_opt);
        const auto logical_outputs =
            pipeline_internal::sima::get_mla_logical_outputs_contract(*mpk_opt);
        const auto physical_outputs =
            pipeline_internal::sima::get_mla_boundary_physical_outputs_contract(*mpk_opt);
        auto mla_contract = pipeline_internal::sima::build_mla_static_contract_from_mpk_stage(
            *mla_stage,
            !published_outputs.empty()
                ? published_outputs
                : (logical_outputs.empty() ? mla_stage->output_tensors : logical_outputs),
            physical_outputs.empty() ? mla_stage->output_tensors : physical_outputs,
            !mla_stage->name.empty() ? mla_stage->name : std::string("mla"),
            boundary_inputs.empty() ? nullptr : &boundary_inputs);
        if (!mla_contract.logical_inputs.empty()) {
          const auto& logical_input = mla_contract.logical_inputs.front();
          mla_input_dtype = logical_input.dtype;
          mla_input_layout =
              rendered_stage_query::layout_projection_from_contract_format(logical_input.layout);
          mla_dims = dims_from_mla_logical_contract_shape(logical_input.shape, mla_input_layout);
        }
      }
    }
    if (mla_dims.width <= 0 || mla_dims.height <= 0 || mla_dims.depth <= 0 ||
        mla_input_dtype.empty()) {
      const auto mla_input_tensor_info =
          rendered_stage_query::mla_input_tensor_info_from_nodes(pack.infer_block(infer_upstream));
      if (mla_input_dtype.empty()) {
        mla_input_dtype = mla_input_tensor_info.logical_dtype;
      }
      if (mla_input_layout == TensorLayout::Unknown) {
        mla_input_layout = mla_input_tensor_info.logical_layout;
      }
      if (mla_dims.width <= 0 || mla_dims.height <= 0 || mla_dims.depth <= 0) {
        mla_dims = dims_from_mla_logical_contract_shape(mla_input_tensor_info.logical_shape,
                                                        mla_input_tensor_info.logical_layout);
      }
    }
    rp.mla_contract.media_type = "application/vnd.simaai.tensor";
    rp.mla_contract.format = mla_input_dtype;
    rp.mla_contract.width = mla_dims.width;
    rp.mla_contract.height = mla_dims.height;
    rp.mla_contract.depth = mla_dims.depth;
    rp.mla_contract.max_width = mla_dims.width;
    rp.mla_contract.max_height = mla_dims.height;
    rp.mla_contract.max_depth = mla_dims.depth;

    if (rp.enabled && rp.effective.resize.enable == AutoFlag::On &&
        (rp.effective.resize.width <= 0 || rp.effective.resize.height <= 0)) {
      if (mla_dims.width <= 0 || mla_dims.height <= 0) {
        std::ostringstream oss;
        oss << "Model preprocess planner hard failure: preprocess.resize is enabled but target "
               "width/height are unresolved. Inference from MLA input contract failed "
               "(infer upstream='"
            << infer_upstream << "', mla width=" << mla_dims.width
            << ", mla height=" << mla_dims.height << "). Selected graph kernel='" << rp.graph_kernel
            << "', config='"
            << (rp.graph_config_path.empty() ? std::string("<none>") : rp.graph_config_path)
            << "'. No host fallback is allowed. Fix by restoring MLA input dims in MPK/infer "
               "wiring or explicitly setting preprocess.resize.width and "
               "preprocess.resize.height.";
        throw std::runtime_error(oss.str());
      }
      if (rp.effective.resize.width <= 0) {
        rp.effective.resize.width = mla_dims.width;
      }
      if (rp.effective.resize.height <= 0) {
        rp.effective.resize.height = mla_dims.height;
      }
      rp.warnings.push_back("preprocess.resize width/height inferred from MLA input contract.");
    }

    auto emit_requirement_warning = [&](internal::RequiredPreprocessOp op,
                                        internal::RequirementSource source, const char* code,
                                        const std::string& reason, const std::string& fix_hint) {
      internal::RequirementIssue issue;
      issue.op = op;
      issue.severity = internal::RequirementSeverity::Warning;
      issue.source = source;
      issue.code = code ? code : "";
      issue.reason = reason;
      issue.fix_hint = fix_hint;
      preprocess_plan.requirement_issues.push_back(issue);
      rp.warnings.push_back(internal::requirement_issue_message(issue));
    };

    const bool normalize_explicit_off =
        user_explicit_off(rp.requested, rp.transforms_override, TransformType::Normalize);
    const bool quantize_explicit_off =
        user_explicit_off(rp.requested, rp.transforms_override, TransformType::Quantize);
    const bool tessellate_explicit_off =
        user_explicit_off(rp.requested, rp.transforms_override, TransformType::Tessellate);
    const bool layout_explicit_off =
        user_explicit_off(rp.requested, rp.transforms_override, TransformType::LayoutConvert);
    const bool resize_explicit_off =
        user_explicit_off(rp.requested, rp.transforms_override, TransformType::Resize);

    if (rp.enabled && normalize_explicit_off && rp.effective.normalize.enable == AutoFlag::Off &&
        (rp.effective.preset == NormalizePreset::ImageNet ||
         rp.effective.preset == NormalizePreset::COCO_YOLO)) {
      emit_requirement_warning(internal::RequiredPreprocessOp::Normalize,
                               internal::RequirementSource::Preset,
                               "PREPROC_REQ_WARN_MODEL_DEFAULT",
                               "normalize was explicitly disabled, but selected preset expects "
                               "normalize to remain enabled.",
                               "Use preprocess.normalize.enable=Auto/On, or set "
                               "preprocess.preset=None for full manual control.");
    }

    const bool mla_quantized_ingress = format_is_quantized_tensor(mla_input_dtype);
    if (rp.enabled && quantize_explicit_off && rp.effective.quantize.enable == AutoFlag::Off &&
        mla_quantized_ingress) {
      emit_requirement_warning(
          internal::RequiredPreprocessOp::Quantize, internal::RequirementSource::MlaContract,
          "PREPROC_REQ_WARN_DYNAMIC_INPUT_UNKNOWN",
          "quantize was explicitly disabled while MLA ingress format appears quantized.",
          "Ensure upstream input tensors already match MLA quantized contract, or set "
          "preprocess.quantize=Auto/On.");
    }

    if (rp.enabled && tessellate_explicit_off && rp.effective.tessellate.enable == AutoFlag::Off &&
        mla_input_layout == TensorLayout::CHW) {
      emit_requirement_warning(
          internal::RequiredPreprocessOp::Tessellate, internal::RequirementSource::MlaContract,
          "PREPROC_REQ_WARN_DYNAMIC_INPUT_UNKNOWN",
          "tessellate was explicitly disabled while MLA input layout is CHW.",
          "If downstream expects tiled tensors, keep preprocess.tessellate=Auto/On.");
    }

    if (rp.enabled && layout_explicit_off && rp.effective.layout_convert.enable == AutoFlag::Off &&
        mla_input_layout != TensorLayout::Unknown) {
      emit_requirement_warning(
          internal::RequiredPreprocessOp::LayoutConvert, internal::RequirementSource::MlaContract,
          "PREPROC_REQ_WARN_DYNAMIC_INPUT_UNKNOWN",
          "layout_convert was explicitly disabled while MLA contract has a fixed input layout.",
          "Keep layout_convert Auto/On when source tensor/image layout may differ from MLA "
          "contract.");
    }

    if (rp.enabled && resize_explicit_off && rp.effective.resize.enable == AutoFlag::Off &&
        mla_dims.width > 0 && mla_dims.height > 0) {
      emit_requirement_warning(
          internal::RequiredPreprocessOp::Resize, internal::RequirementSource::MlaContract,
          "PREPROC_REQ_WARN_DYNAMIC_INPUT_UNKNOWN",
          "resize was explicitly disabled while MLA contract expects fixed spatial dimensions.",
          "Provide already-matched input dimensions at runtime, or set preprocess.resize=Auto/On.");
    }

    emit_model_planner_messages(options.verbose, rp.warnings);
    maybe_log_model_info_shadow(preprocess_plan, pack);

    model_id = pack.etc_dir();
    if (!options.name_suffix.empty()) {
      model_id += "::" + options.name_suffix;
    }
  }

  const internal::ModelPack& pack_for_sync() const {
    if (!sync_pack.has_value()) {
      sync_pack = pack.clone_with_buffers(1, 1);
    }
    return *sync_pack;
  }
};

Model::Model(const std::string& model_path) : Model(model_path, Options{}) {}

Model::Model(const std::string& model_path, const Options& opt) {
  impl_ = std::make_unique<Impl>(model_path, opt);
}

Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

namespace {

std::optional<PreprocessMetaTemplate>
make_preprocess_meta_template(const internal::PreprocessPlannerResult& plan) {
  const auto& rp = plan.resolved_plan;
  if (plan.session_route_plan.boxdecode_selected &&
      rp.graph_family != PreprocessGraphFamily::Preproc) {
    if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
      std::fprintf(stderr,
                   "[typed-adapter] preprocess-meta disabled for non-preproc boxdecode "
                   "graph_family=%s\n",
                   preprocess_graph_family_name(rp.graph_family).c_str());
    }
    return std::nullopt;
  }
  const internal::PreprocessContractFlags contract_flags = resolve_preprocess_contract_flags(plan);
  PreprocessMetaTemplate meta;
  meta.enabled = true;
  meta.quantize = contract_flags.quant_needed;
  meta.tessellate = contract_flags.tess_needed;
  if (!rp.enabled) {
    // Adapter-only ingress (quant/tess/quanttess) may still feed boxdecode and
    // therefore must carry preprocess runtime metadata even without a preproc graph.
    const auto ingress_contracts = normalized_ingress_contracts(plan.session_route_plan);
    const auto* ingress = maybe_single_ingress_contract(ingress_contracts);
    meta.target_width = (ingress != nullptr && ingress->width > 0) ? ingress->width : 0;
    meta.target_height = (ingress != nullptr && ingress->height > 0) ? ingress->height : 0;
    meta.scaled_width = meta.target_width;
    meta.scaled_height = meta.target_height;
    meta.resize_mode = (meta.target_width > 0 && meta.target_height > 0) ? "stretch" : "none";
    meta.color_in = "BGR";
    meta.color_out = "RGB";
    meta.axis_perm.clear();
    meta.normalize = true;
    if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
      std::fprintf(stderr,
                   "[typed-adapter] preprocess-meta source=route_contract rp_enabled=0 "
                   "quantize=%d tessellate=%d\n",
                   meta.quantize ? 1 : 0, meta.tessellate ? 1 : 0);
    }
    return meta;
  }

  meta.normalize = rp.effective.normalize.enable == AutoFlag::On;
  if (rp.effective.resize.enable == AutoFlag::On) {
    meta.target_width = rp.effective.resize.width;
    meta.target_height = rp.effective.resize.height;
    meta.scaled_width = rp.effective.resize.width;
    meta.scaled_height = rp.effective.resize.height;
    switch (rp.effective.resize.mode) {
    case ResizeMode::Stretch:
      meta.resize_mode = "stretch";
      break;
    case ResizeMode::Letterbox:
      meta.resize_mode = "letterbox";
      break;
    case ResizeMode::Crop:
      meta.resize_mode = "crop";
      break;
    }
    meta.pad_value = rp.effective.resize.pad_value;
  } else {
    meta.resize_mode = "none";
  }
  meta.color_in = rp.effective.color_convert.input_format == PreprocessColorFormat::Auto
                      ? plan.modelpack_format
                      : preprocess_color_format_name(rp.effective.color_convert.input_format);
  meta.color_out = rp.effective.color_convert.output_format == PreprocessColorFormat::Auto
                       ? ""
                       : preprocess_color_format_name(rp.effective.color_convert.output_format);
  meta.axis_perm = rp.effective.layout_convert.perm;
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr,
                 "[typed-adapter] preprocess-meta source=route_contract rp_enabled=1 "
                 "quantize=%d tessellate=%d effective_quant=%d effective_tess=%d\n",
                 meta.quantize ? 1 : 0, meta.tessellate ? 1 : 0,
                 rp.effective.quantize.enable == AutoFlag::On ? 1 : 0,
                 rp.effective.tessellate.enable == AutoFlag::On ? 1 : 0);
  }
  return meta;
}

const char* session_pre_adapter_kind_name(internal::SessionPreAdapterKind kind) {
  switch (kind) {
  case internal::SessionPreAdapterKind::None:
    return "none";
  case internal::SessionPreAdapterKind::Quant:
    return "quant";
  case internal::SessionPreAdapterKind::Tess:
    return "tess";
  case internal::SessionPreAdapterKind::QuantTess:
    return "quanttess";
  case internal::SessionPreAdapterKind::CastTess:
    return "casttess";
  }
  return "none";
}

const char* session_pre_stage_op_name(internal::SessionPreStageOp op) {
  switch (op) {
  case internal::SessionPreStageOp::Preproc:
    return "preproc";
  case internal::SessionPreStageOp::Quant:
    return "quant";
  case internal::SessionPreStageOp::Tess:
    return "tess";
  case internal::SessionPreStageOp::QuantTess:
    return "quanttess";
  case internal::SessionPreStageOp::Cast:
    return "cast";
  case internal::SessionPreStageOp::CastTess:
    return "casttess";
  }
  return "unknown";
}

std::optional<pipeline_internal::sima::ModelManagedRouteFlags>
model_route_flags_for_pre_stage(const internal::SessionRoutePlan& route) {
  if (!route.include_pre_stage) {
    return std::nullopt;
  }
  pipeline_internal::sima::ModelManagedRouteFlags flags =
      convert_model_managed_route_flags(route.model_managed_route_flags);
  flags.include_pre_stage = true;
  flags.boxdecode_selected = false;
  return flags;
}

std::optional<pipeline_internal::sima::ModelManagedRouteFlags>
model_route_flags_for_boxdecode_stage(const internal::SessionRoutePlan& route) {
  if (!route.boxdecode_selected) {
    return std::nullopt;
  }
  pipeline_internal::sima::ModelManagedRouteFlags flags;
  flags.quant_needed = route.model_managed_route_flags.quant_needed;
  flags.tess_needed = route.model_managed_route_flags.tess_needed;
  flags.pre_cast_needed = route.model_managed_route_flags.pre_cast_needed;
  flags.quant_contract_required = flags.quant_needed;
  flags.include_pre_stage = route.model_managed_route_flags.include_pre_stage;
  flags.boxdecode_selected = true;
  return flags;
}

std::shared_ptr<const internal::ModelLineageBinding>
make_stage_lineage_binding(const Model& model, internal::ModelLineageStageRole stage_role) {
  return internal::make_model_lineage_binding(model, stage_role,
                                              internal::RequestedPostRouteKind::Auto);
}

template <typename OptionsT> void lock_buffers_for_sync(OptionsT* opt, bool sync) {
  if (!opt || !sync) {
    return;
  }
  opt->num_buffers = 1;
  opt->num_buffers_model = 1;
  opt->num_buffers_locked = true;
}

void init_model_managed_processcvu_buffers(PreprocOptions* opt, const internal::ModelPack& pack,
                                           bool sync) {
  if (!opt) {
    return;
  }
  opt->num_buffers_model = pack.num_buffers_cvu();
  opt->num_buffers = sync ? 1 : opt->num_buffers_model;
  opt->num_buffers_locked = true;
}

template <typename OptionsT>
void init_model_managed_processcvu_buffers(OptionsT* opt, const internal::ModelPack& pack,
                                           bool sync) {
  if (!opt) {
    return;
  }
  opt->config_path.clear();
  opt->num_buffers_model = pack.num_buffers_cvu();
  opt->num_buffers = sync ? 1 : opt->num_buffers_model;
  opt->num_buffers_locked = true;
}

std::string resolve_preproc_input_format(const internal::PreprocessPlannerResult& plan,
                                         const InputInfo* input) {
  if (input && !input->format.empty() &&
      input->format_source == InputInfo::FormatSource::Explicit) {
    return upper_copy(input->format);
  }
  const auto& effective = plan.resolved_plan.effective;
  if (effective.color_convert.input_format != PreprocessColorFormat::Auto) {
    return preprocess_color_format_name(effective.color_convert.input_format);
  }
  if (input && !input->format.empty()) {
    return upper_copy(input->format);
  }
  const auto* ingress =
      maybe_single_preprocess_ingress_contract(plan.resolved_plan.ingress_contracts);
  if (ingress != nullptr && !ingress->format.empty()) {
    const FormatSpec ingress_format{ingress->format};
    if (is_raw_video_format(ingress_format.tag) || plan.modelpack_format.empty()) {
      return ingress->format;
    }
  }
  return plan.modelpack_format;
}

std::string resolve_preproc_output_format(const internal::PreprocessPlannerResult& plan) {
  const auto& effective = plan.resolved_plan.effective;
  if (effective.color_convert.output_format != PreprocessColorFormat::Auto) {
    return preprocess_color_format_name(effective.color_convert.output_format);
  }
  if (effective.color_convert.input_format == PreprocessColorFormat::GRAY8 ||
      effective.input_max_depth == 1) {
    return "GRAY8";
  }
  const auto* ingress =
      maybe_single_ingress_contract(normalized_ingress_contracts(plan.session_route_plan));
  if (ingress != nullptr && ingress->depth == 1) {
    return "GRAY8";
  }
  const auto* resolved_ingress =
      maybe_single_preprocess_ingress_contract(plan.resolved_plan.ingress_contracts);
  if (resolved_ingress != nullptr && resolved_ingress->depth == 1) {
    return "GRAY8";
  }
  // input_format describes the user's ingress memory layout and must not be
  // copied into the model-facing output layout.  In particular, BGR input means
  // "trust the user's bytes are BGR"; model-managed preproc still publishes
  // semantic RGB unless the user explicitly requested an output_format above.
  // Transport formats like NV12/I420 likewise describe ingress memory layout,
  // not the colorized output image that preproc publishes toward
  // normalize/quant/tess.
  return "RGB";
}

std::string resolve_preproc_output_dtype(const internal::PreprocessPlannerResult& plan,
                                         const internal::PreprocessContractFlags& flags) {
  const auto& effective = plan.resolved_plan.effective;
  std::string ingress_dtype = plan.resolved_plan.mla_contract.format;
  if (ingress_dtype.empty()) {
    ingress_dtype = route_mla_input_dtype_from_diagnostics(plan);
  }
  if (ingress_dtype.empty()) {
    ingress_dtype = unify_ingress_dtype(normalized_ingress_contracts(plan.session_route_plan));
  }
  const std::string ingress_up = upper_copy(ingress_dtype);
  if (ingress_up.find("BF16") != std::string::npos ||
      ingress_up.find("BFLOAT16") != std::string::npos) {
    return "EVXX_BFLOAT16";
  }
  if (ingress_up.find("FP32") != std::string::npos ||
      ingress_up.find("FLOAT32") != std::string::npos) {
    return "EVXX_FLOAT32";
  }
  if (ingress_up == "INT8" || ingress_up == "EVXX_INT8" || ingress_up == "UINT8" ||
      ingress_up == "U8") {
    return "INT8";
  }
  if (ingress_up == "INT16" || ingress_up == "EVXX_INT16") {
    return "INT16";
  }
  if (!effective.quantize.output_dtype.empty()) {
    return effective.quantize.output_dtype;
  }
  if (flags.quant_needed) {
    return "INT8";
  }
  // For model-managed routes that fuse an ingress cast into preproc, the public
  // preproc stage publishes the cast result (FP32->BF16) even if downstream
  // transport/MLA diagnostics later describe packed INT8/MLA views. The
  // user-facing typed config should therefore reflect the fused preproc stage's
  // actual published dtype rather than the downstream MLA transport dtype.
  if (ingress_dtype.empty() && plan.session_route_plan.model_managed_route_flags.pre_cast_needed) {
    return "EVXX_BFLOAT16";
  }
  return {};
}

int resolve_preproc_input_width(const internal::PreprocessPlannerResult& plan,
                                const InputInfo* input) {
  if (input && input->width > 0) {
    return input->width;
  }
  const auto* ingress =
      maybe_single_ingress_contract(normalized_ingress_contracts(plan.session_route_plan));
  if (ingress != nullptr && ingress->width > 0) {
    return ingress->width;
  }
  const auto* resolved_ingress =
      maybe_single_preprocess_ingress_contract(plan.resolved_plan.ingress_contracts);
  if (resolved_ingress != nullptr && resolved_ingress->width > 0) {
    return resolved_ingress->width;
  }
  // Model-managed preproc is a user-facing image ingress even when the MPK route
  // fans out internally into multiple tensor ingress contracts (for example, EVO50
  // image_l + image_uv). In that case there is no single route ingress to project
  // here, so fall back to the canonical modelpack image contract instead of
  // treating the split internal ingress tensors as the public ingress.
  if (plan.modelpack_max_width > 0) {
    return plan.modelpack_max_width;
  }
  return (plan.resolved_plan.effective.resize.width > 0) ? plan.resolved_plan.effective.resize.width
                                                         : 0;
}

int resolve_preproc_input_height(const internal::PreprocessPlannerResult& plan,
                                 const InputInfo* input) {
  if (input && input->height > 0) {
    return input->height;
  }
  const auto* ingress =
      maybe_single_ingress_contract(normalized_ingress_contracts(plan.session_route_plan));
  if (ingress != nullptr && ingress->height > 0) {
    return ingress->height;
  }
  const auto* resolved_ingress =
      maybe_single_preprocess_ingress_contract(plan.resolved_plan.ingress_contracts);
  if (resolved_ingress != nullptr && resolved_ingress->height > 0) {
    return resolved_ingress->height;
  }
  // See resolve_preproc_input_width(): multi-ingress internal contracts must not
  // erase the public model-managed image ingress contract.
  if (plan.modelpack_max_height > 0) {
    return plan.modelpack_max_height;
  }
  return (plan.resolved_plan.effective.resize.height > 0)
             ? plan.resolved_plan.effective.resize.height
             : 0;
}

int resolve_preproc_input_depth(const internal::PreprocessPlannerResult& plan,
                                const InputInfo* input, const std::string& input_format) {
  if (input && input->depth > 0) {
    return input->depth;
  }
  const auto* ingress =
      maybe_single_ingress_contract(normalized_ingress_contracts(plan.session_route_plan));
  if (ingress != nullptr && ingress->depth > 0) {
    return ingress->depth;
  }
  const auto* resolved_ingress =
      maybe_single_preprocess_ingress_contract(plan.resolved_plan.ingress_contracts);
  if (resolved_ingress != nullptr && resolved_ingress->depth > 0) {
    return resolved_ingress->depth;
  }
  return pipeline_internal::default_depth_for_image_format(input_format,
                                                           plan.modelpack_input_depth);
}

void populate_model_managed_preproc_options(PreprocOptions* opt,
                                            const internal::PreprocessPlannerResult& plan,
                                            const InputInfo* input) {
  if (!opt) {
    return;
  }

  const internal::PreprocessContractFlags flags = resolve_preprocess_contract_flags(plan);
  const auto& effective = plan.resolved_plan.effective;

  opt->input_img_type = resolve_preproc_input_format(plan, input);
  {
    std::vector<int> input_shape = {resolve_preproc_input_height(plan, input),
                                    resolve_preproc_input_width(plan, input)};
    const int input_depth = resolve_preproc_input_depth(plan, input, opt->input_img_type);
    if (input_depth > 0) {
      input_shape.push_back(input_depth);
    }
    opt->set_input_shape(std::move(input_shape));
  }
  if (!opt->has_input_shape() || opt->input_img_type.empty()) {
    const auto route_ingress = normalized_ingress_contracts(plan.session_route_plan);
    std::ostringstream oss;
    oss << "PreprocOptions(Model): resolved model-managed route is missing input "
           "width/height/format."
        << " resolved={width=" << opt->input_width() << ",height=" << opt->input_height()
        << ",format=" << (opt->input_img_type.empty() ? "<empty>" : opt->input_img_type) << "}"
        << " input=" << input_info_debug_string(input)
        << " route=" << route_stage_summary(plan.session_route_plan) << " route_ingress="
        << vector_debug_string(
               route_ingress,
               [](const auto& ingress) { return ingress_tensor_contract_debug_string(ingress); })
        << " resolved_ingress="
        << vector_debug_string(
               plan.resolved_plan.ingress_contracts,
               [](const auto& ingress) { return preprocess_contract_debug_string(ingress); })
        << " modelpack={format="
        << (plan.modelpack_format.empty() ? "<empty>" : plan.modelpack_format)
        << ",max_w=" << plan.modelpack_max_width << ",max_h=" << plan.modelpack_max_height
        << ",max_d=" << plan.modelpack_max_depth << "}"
        << " route_diagnostics=" << route_diagnostics_debug_string(plan.route_diagnostics);
    throw std::runtime_error(oss.str());
  }

  opt->output_img_type = resolve_preproc_output_format(plan);
  {
    std::vector<int> output_shape = {
        (effective.resize.height > 0) ? effective.resize.height : opt->input_height(),
        (effective.resize.width > 0) ? effective.resize.width : opt->input_width()};
    const int output_depth = pipeline_internal::default_depth_for_image_format(
        opt->output_img_type, opt->input_channels());
    if (output_depth > 0) {
      output_shape.push_back(output_depth);
    }
    opt->set_output_shape(std::move(output_shape));
  }
  opt->scaled_width = opt->output_width();
  opt->scaled_height = opt->output_height();
  if (!opt->has_output_shape()) {
    throw std::runtime_error("PreprocOptions(Model): model-managed preproc stage requires "
                             "output width/height/channels from the resolved route contract.");
  }
  opt->output_dtype = resolve_preproc_output_dtype(plan, flags);
  if (opt->output_dtype.empty()) {
    throw std::runtime_error("PreprocOptions(Model): model-managed preproc stage requires an "
                             "explicit output dtype from the resolved route contract.");
  }
  opt->normalize = plan.normalize;
  opt->channel_mean = plan.mean;
  opt->channel_stddev = plan.stddev;
  opt->tessellate = flags.tess_needed;
  opt->single_output_handoff = true;
  if (effective.resize.enable == AutoFlag::On) {
    opt->scaling_type = effective.resize.scaling_type;
    opt->pad_value = effective.resize.pad_value;
    switch (effective.resize.mode) {
    case ResizeMode::Letterbox:
      opt->aspect_ratio = true;
      opt->padding_type = "CENTER";
      break;
    case ResizeMode::Stretch:
      opt->aspect_ratio = false;
      // The EV74 preproc schema has no "NONE" padding enum.  For stretch,
      // aspect_ratio=false makes the scaled size equal the output size, so no
      // padding is applied; use a schema-valid value that is behavior-neutral.
      opt->padding_type = "CENTER";
      break;
    case ResizeMode::Crop:
      opt->aspect_ratio = true;
      opt->padding_type = "NONE";
      break;
    }
  }

  if (flags.tess_needed) {
    opt->slice_shape = effective.tessellate.slice_shape;
    if (!effective.tessellate.has_slice_shape() || !opt->has_slice_shape()) {
      throw std::runtime_error("PreprocOptions(Model): model-managed preproc stage requires "
                               "slice_shape from the resolved route contract.");
    }
  } else {
    opt->slice_shape.clear();
  }

  if (preproc_output_dtype_is_quantized(opt->output_dtype)) {
    if (!(effective.quantize.scale > 0.0)) {
      throw std::runtime_error("PreprocOptions(Model): model-managed preproc stage requires "
                               "q_scale from the resolved route contract.");
    }
    opt->q_scale = effective.quantize.scale;
    opt->q_zp = static_cast<std::int64_t>(effective.quantize.zero_point);
  } else {
    opt->q_scale.reset();
    opt->q_zp.reset();
  }
}

PreprocOptions make_model_managed_preproc_options_base(const Model& model, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  PreprocOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  opt.model_managed_contract = true;
  opt.next_cpu = pack.preproc_next_cpu();
  return opt;
}

QuantOptions make_model_managed_quant_options_base(const Model& model, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  QuantOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  return opt;
}

TessOptions make_model_managed_tess_options_base(const Model& model, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  TessOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  return opt;
}

QuantTessOptions make_model_managed_quanttess_options_base(const Model& model, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  QuantTessOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  return opt;
}

CastTessOptions make_model_managed_casttess_options_base(const Model& model, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  CastTessOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  return opt;
}

PreprocOptions make_preproc_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, const std::string& upstream_name, bool sync) {
  PreprocOptions opt = make_model_managed_preproc_options_base(model, sync);
#ifdef SIMA_NEAT_INTERNAL
  opt.model_lineage =
      make_stage_lineage_binding(model, internal::ModelLineageStageRole::Preprocess);
#endif
  opt.element_name = element_name;
  opt.node_name = element_name.empty() ? opt.node_name : element_name;
  if (!upstream_name.empty()) {
    opt.upstream_name = upstream_name;
  }
  populate_model_managed_preproc_options(&opt, plan, input);
  return opt;
}

QuantOptions make_quant_options_from_typed_adapter(const Model& model,
                                                   const internal::PreprocessPlannerResult& plan,
                                                   const InputInfo* input,
                                                   const std::string& element_name, bool sync,
                                                   const internal::OrderedRouteOp* route_op,
                                                   bool allow_multi_io_contract) {
  (void)input;
  (void)plan;
  QuantOptions opt = make_model_managed_quant_options_base(model, sync);
#ifdef SIMA_NEAT_INTERNAL
  opt.model_lineage =
      make_stage_lineage_binding(model, internal::ModelLineageStageRole::Preprocess);
#endif
  opt.element_name = element_name;
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model),
          internal::ExecutionStageKind::Quant, "quant",
          allow_multi_io_contract ? nullptr : route_op));
  return opt;
}

TessOptions make_tess_options_from_typed_adapter(const Model& model,
                                                 const internal::PreprocessPlannerResult& plan,
                                                 const InputInfo* input,
                                                 const std::string& element_name, bool sync,
                                                 const internal::OrderedRouteOp* route_op,
                                                 bool allow_multi_io_contract) {
  (void)input;
  (void)plan;
  TessOptions opt = make_model_managed_tess_options_base(model, sync);
#ifdef SIMA_NEAT_INTERNAL
  opt.model_lineage =
      make_stage_lineage_binding(model, internal::ModelLineageStageRole::Preprocess);
#endif
  opt.element_name = element_name;
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr, "[typed-adapter] make_tess_options element_name=%s\n",
                 element_name.c_str());
  }
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model),
          internal::ExecutionStageKind::Tess, "tess",
          allow_multi_io_contract ? nullptr : route_op));
  return opt;
}

QuantTessOptions make_quanttess_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, bool sync, const internal::OrderedRouteOp* route_op,
    bool allow_multi_io_contract) {
  (void)input;
  (void)plan;
  QuantTessOptions opt = make_model_managed_quanttess_options_base(model, sync);
#ifdef SIMA_NEAT_INTERNAL
  opt.model_lineage =
      make_stage_lineage_binding(model, internal::ModelLineageStageRole::Preprocess);
#endif
  opt.element_name = element_name;
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model),
          internal::ExecutionStageKind::QuantTess, "quanttess",
          allow_multi_io_contract ? nullptr : route_op));
  return opt;
}

CastTessOptions make_casttess_options_from_typed_adapter(
    const Model& model, const internal::PreprocessPlannerResult& plan, const InputInfo* input,
    const std::string& element_name, bool sync, const internal::OrderedRouteOp* route_op,
    bool allow_multi_io_contract) {
  (void)input;
  (void)plan;
  CastTessOptions opt = make_model_managed_casttess_options_base(model, sync);
#ifdef SIMA_NEAT_INTERNAL
  opt.model_lineage =
      make_stage_lineage_binding(model, internal::ModelLineageStageRole::Preprocess);
#endif
  opt.element_name = element_name;
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model),
          internal::ExecutionStageKind::CastTess, "casttess",
          allow_multi_io_contract ? nullptr : route_op));
  return opt;
}

DetessOptions make_detess_options_from_typed_adapter(const Model& model,
                                                     const std::string& element_name, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  DetessOptions opt(model);
  if (!element_name.empty()) {
    opt.element_name = element_name;
  }
  auto compiled =
      require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Detess);
  compiled.runtime_contract.plugin_kind = "neatdetess";
  opt.compiled_contract = std::make_shared<const CompiledProcessCvuContract>(std::move(compiled));
  lock_buffers_for_sync(&opt, sync);
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr, "[typed-adapter] post kind=detess include_post=1\n");
    const auto& rc =
        opt.compiled_contract ? opt.compiled_contract->runtime_contract : CompiledRuntimeContract{};
    std::fprintf(stderr,
                 "[typed-adapter] detess plugin_kind=%s logical_outputs=%zu "
                 "logical_inputs=%zu physical_outputs=%zu\n",
                 rc.plugin_kind.c_str(), rc.logical_outputs.size(), rc.logical_inputs.size(),
                 rc.physical_outputs.size());
    for (std::size_t di = 0; di < rc.logical_outputs.size(); ++di) {
      const auto& lo = rc.logical_outputs[di];
      std::fprintf(stderr,
                   "[typed-adapter] detess[%zu] segment=%s offset=%lld "
                   "size=%llu dtype=%s shape=[",
                   di, lo.segment_name.c_str(), static_cast<long long>(lo.byte_offset),
                   static_cast<unsigned long long>(lo.size_bytes), lo.dtype.c_str());
      for (std::size_t si = 0; si < lo.shape.size(); ++si) {
        std::fprintf(stderr, "%s%lld", si ? "," : "", static_cast<long long>(lo.shape[si]));
      }
      std::fprintf(stderr, "]\n");
    }
    if (opt.compiled_contract) {
      const auto& p = opt.compiled_contract->payload;
      std::fprintf(stderr,
                   "[typed-adapter] detess payload graph_family=%s "
                   "canonical=%d input_shapes=%zu output_shapes=%zu\n",
                   p.graph_family.c_str(), p.canonical_contract, p.input_shapes.size(),
                   p.output_shapes.size());
    }
  }
  return opt;
}

DetessCastOptions make_detesscast_options_from_typed_adapter(const Model& model,
                                                             const std::string& element_name,
                                                             bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  DetessCastOptions opt(model);
  if (!element_name.empty()) {
    opt.element_name = element_name;
  }
  opt.compiled_contract = std::make_shared<const CompiledProcessCvuContract>(
      require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::DetessCast));
  lock_buffers_for_sync(&opt, sync);
  return opt;
}

DetessDequantOptions make_detessdequant_options_from_typed_adapter(const Model& model,
                                                                   const std::string& element_name,
                                                                   bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  DetessDequantOptions opt(model);
  if (!element_name.empty()) {
    opt.element_name = element_name;
  }
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_postprocess_contract(
          pack, internal::ExecutionStageKind::DetessDequant));
  lock_buffers_for_sync(&opt, sync);
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr, "[typed-adapter] post kind=detessdequant include_post=1\n");
  }
  return opt;
}

DequantOptions make_dequant_options_from_typed_adapter(const Model& model,
                                                       const std::string& element_name, bool sync) {
  const auto& pack =
      sync ? internal::ModelAccess::pack_for_sync(model) : internal::ModelAccess::pack(model);
  DequantOptions opt(model);
  if (!element_name.empty()) {
    opt.element_name = element_name;
    opt.stage_id = element_name;
  }
  opt.processcvu_compiled_contract = std::make_shared<const CompiledProcessCvuContract>(
      require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Dequant));
  lock_buffers_for_sync(&opt, sync);
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr, "[typed-adapter] post kind=dequant include_post=1\n");
  }
  return opt;
}

CastOptions make_cast_options_from_typed_adapter(const Model* model,
                                                 const std::string& element_name,
                                                 CastDirection direction,
                                                 internal::ModelLineageStageRole stage_role,
                                                 const internal::OrderedRouteOp* route_op,
                                                 bool allow_multi_io_contract) {
  CastOptions opt;
  opt.direction = direction;
  opt.element_name = element_name;
  opt.silent = true;
#ifdef SIMA_NEAT_INTERNAL
  if (model != nullptr) {
    opt.model_lineage = make_stage_lineage_binding(*model, stage_role);
  }
#endif
  if (model != nullptr) {
    const bool is_pre = stage_role == internal::ModelLineageStageRole::Preprocess;
    const auto& pack = internal::ModelAccess::pack(*model);
    opt.compiled_contract = std::make_shared<const CompiledProcessCvuContract>(
        is_pre
            ? require_model_managed_preadapter_contract(
                  pack, internal::ExecutionStageKind::Cast, "cast",
                  allow_multi_io_contract ? nullptr : route_op)
            : require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Cast));
    opt.num_buffers = pack.num_buffers_cvu();
  }
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr, "[typed-adapter] post kind=cast direction=%d element=%s\n",
                 static_cast<int>(direction), element_name.c_str());
  }
  return opt;
}

std::optional<internal::ExecutionStageKind>
execution_stage_kind_from_post_region(const internal::RouteRegion& region) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (region.op_kind) {
  case GraphKind::Detess:
    return internal::ExecutionStageKind::Detess;
  case GraphKind::DetessCast:
    return internal::ExecutionStageKind::DetessCast;
  case GraphKind::DetessDequant:
    return internal::ExecutionStageKind::DetessDequant;
  case GraphKind::Dequantize:
    return internal::ExecutionStageKind::Dequant;
  case GraphKind::Cast:
    return internal::ExecutionStageKind::Cast;
  case GraphKind::BoxDecode:
    return internal::ExecutionStageKind::BoxDecode;
  case GraphKind::Unknown:
  case GraphKind::Preproc:
  case GraphKind::Quant:
  case GraphKind::Tess:
  case GraphKind::QuantTess:
  case GraphKind::CastTess:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    return std::nullopt;
  }
  return std::nullopt;
}

std::string default_post_region_stage_name(const internal::RouteRegion& region,
                                           const std::size_t region_index) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (region.op_kind) {
  case GraphKind::Detess:
    return "post_detess_" + std::to_string(region_index);
  case GraphKind::DetessCast:
    return "post_detesscast_" + std::to_string(region_index);
  case GraphKind::DetessDequant:
    return "post_detessdequant_" + std::to_string(region_index);
  case GraphKind::Dequantize:
    return "post_dequant_" + std::to_string(region_index);
  case GraphKind::Cast:
    return "post_cast_" + std::to_string(region_index);
  case GraphKind::BoxDecode:
    return "boxdecode";
  case GraphKind::Unknown:
  case GraphKind::Preproc:
  case GraphKind::Quant:
  case GraphKind::Tess:
  case GraphKind::QuantTess:
  case GraphKind::CastTess:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    return "post_region_" + std::to_string(region_index);
  }
  return "post_region_" + std::to_string(region_index);
}

std::string stage_name_for_post_region(const internal::ModelPack& pack,
                                       const internal::RouteRegion& region,
                                       const std::size_t region_index) {
  if (region.op_kind == pipeline_internal::sima::RouteGraphKernelKind::Cast) {
    return default_post_region_stage_name(region, region_index);
  }
  if (const auto stage_kind = execution_stage_kind_from_post_region(region);
      stage_kind.has_value()) {
    for (const auto& stage : pack.execution_plan().post) {
      if (stage.kind == *stage_kind && !stage.stage_name.empty()) {
        return stage.stage_name;
      }
    }
  }
  return pack.apply_name_suffix(default_post_region_stage_name(region, region_index));
}

std::size_t model_managed_mla_logical_output_count(const internal::ModelPack& pack) {
  const auto stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
  for (const auto& fact : stage_facts) {
    if (fact.mla_compiled.has_value()) {
      return fact.mla_compiled->runtime_contract.logical_outputs.size();
    }
  }
  return 0U;
}

std::vector<int> logical_indices_from_runtime_contract(const CompiledRuntimeContract& runtime) {
  std::vector<int> indices;
  indices.reserve(runtime.logical_outputs.size());
  for (const auto& logical : runtime.logical_outputs) {
    if (logical.logical_index >= 0) {
      indices.push_back(logical.logical_index);
    }
  }
  return indices;
}

std::vector<int>
unique_logical_indices_from_bindings(const std::vector<internal::RouteTensorBinding>& bindings) {
  std::vector<int> indices;
  std::unordered_set<int> seen;
  indices.reserve(bindings.size());
  for (const auto& binding : bindings) {
    if (binding.logical_index < 0 || !seen.insert(binding.logical_index).second) {
      continue;
    }
    indices.push_back(binding.logical_index);
  }
  return indices;
}

std::vector<int> sorted_logical_index_set(std::vector<int> indices) {
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return indices;
}

std::vector<int> model_managed_mla_logical_output_indices(const internal::ModelPack& pack) {
  const auto stage_facts = pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
  for (const auto& fact : stage_facts) {
    if (fact.mla_compiled.has_value()) {
      return logical_indices_from_runtime_contract(fact.mla_compiled->runtime_contract);
    }
  }
  return {};
}

std::size_t post_region_compiled_logical_output_count(const internal::ModelPack& pack,
                                                      const internal::RouteRegion& region) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (region.op_kind) {
  case GraphKind::Detess:
    return require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Detess)
        .runtime_contract.logical_outputs.size();
  case GraphKind::DetessCast:
    return require_model_managed_postprocess_contract(pack,
                                                      internal::ExecutionStageKind::DetessCast)
        .runtime_contract.logical_outputs.size();
  case GraphKind::DetessDequant:
    return require_model_managed_postprocess_contract(pack,
                                                      internal::ExecutionStageKind::DetessDequant)
        .runtime_contract.logical_outputs.size();
  case GraphKind::Dequantize:
    return require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Dequant)
        .runtime_contract.logical_outputs.size();
  case GraphKind::Cast:
    return region.member_plugin_indices.size();
  case GraphKind::Unknown:
  case GraphKind::Preproc:
  case GraphKind::Quant:
  case GraphKind::Tess:
  case GraphKind::QuantTess:
  case GraphKind::BoxDecode:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    return 0U;
  }
  return 0U;
}

std::vector<int> post_region_compiled_logical_output_indices(const internal::ModelPack& pack,
                                                             const internal::RouteRegion& region) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (region.op_kind) {
  case GraphKind::Detess:
    return logical_indices_from_runtime_contract(
        require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Detess)
            .runtime_contract);
  case GraphKind::DetessCast:
    return logical_indices_from_runtime_contract(
        require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::DetessCast)
            .runtime_contract);
  case GraphKind::DetessDequant:
    return logical_indices_from_runtime_contract(
        require_model_managed_postprocess_contract(pack,
                                                   internal::ExecutionStageKind::DetessDequant)
            .runtime_contract);
  case GraphKind::Dequantize:
    return logical_indices_from_runtime_contract(
        require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Dequant)
            .runtime_contract);
  case GraphKind::Cast:
  case GraphKind::Unknown:
  case GraphKind::Preproc:
  case GraphKind::Quant:
  case GraphKind::Tess:
  case GraphKind::QuantTess:
  case GraphKind::BoxDecode:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    return {};
  }
  return {};
}

std::vector<int>
upstream_logical_indices_for_post_region(const internal::ModelPack& pack,
                                         const std::vector<internal::RouteRegion>& regions,
                                         const std::size_t region_index) {
  if (region_index == 0U) {
    return model_managed_mla_logical_output_indices(pack);
  }

  const auto& upstream_region = regions[region_index - 1U];
  auto indices = post_region_compiled_logical_output_indices(pack, upstream_region);
  if (!indices.empty()) {
    return indices;
  }

  indices = unique_logical_indices_from_bindings(upstream_region.outputs);
  if (!indices.empty()) {
    return indices;
  }
  return unique_logical_indices_from_bindings(upstream_region.inputs);
}

void validate_fanout_region_contract_alignment(const internal::ModelPack& pack,
                                               const std::vector<internal::RouteRegion>& regions,
                                               const std::size_t region_index) {
  const auto& region = regions[region_index];
  if (region.kind != internal::RouteRegionKind::FanoutMap) {
    return;
  }
  const std::size_t expected_outputs = region.member_plugin_indices.size();
  if (expected_outputs == 0U) {
    throw std::runtime_error("Model post region fanout contract is empty.");
  }

  std::size_t upstream_outputs = 0U;
  if (region_index == 0U) {
    upstream_outputs = model_managed_mla_logical_output_count(pack);
  } else {
    upstream_outputs = post_region_compiled_logical_output_count(pack, regions[region_index - 1U]);
  }
  if (upstream_outputs != 0U && upstream_outputs != expected_outputs) {
    throw std::runtime_error(
        "Model post region upstream logical output count mismatch for fanout stage.");
  }

  const std::size_t compiled_outputs = post_region_compiled_logical_output_count(pack, region);
  if (compiled_outputs != 0U && compiled_outputs != expected_outputs) {
    throw std::runtime_error(
        "Model post region compiled logical output count mismatch for fanout stage.");
  }

  const auto expected_input_order = unique_logical_indices_from_bindings(region.inputs);
  const auto upstream_input_order =
      upstream_logical_indices_for_post_region(pack, regions, region_index);
  if (!expected_input_order.empty() && !upstream_input_order.empty()) {
    const bool require_exact_input_order =
        region.op_kind == pipeline_internal::sima::RouteGraphKernelKind::Cast;
    const bool inputs_match = require_exact_input_order
                                  ? (expected_input_order == upstream_input_order)
                                  : (sorted_logical_index_set(expected_input_order) ==
                                     sorted_logical_index_set(upstream_input_order));
    if (!inputs_match) {
      throw std::runtime_error(
          "Model post region upstream logical output ordering mismatch for fanout stage.");
    }
  }

  const auto expected_output_order = unique_logical_indices_from_bindings(region.outputs);
  const auto compiled_output_order = post_region_compiled_logical_output_indices(pack, region);
  if (!expected_output_order.empty() && !compiled_output_order.empty()) {
    const bool require_exact_output_order =
        region.op_kind == pipeline_internal::sima::RouteGraphKernelKind::Cast;
    const bool outputs_match = require_exact_output_order
                                   ? (expected_output_order == compiled_output_order)
                                   : (sorted_logical_index_set(expected_output_order) ==
                                      sorted_logical_index_set(compiled_output_order));
    if (!outputs_match) {
      throw std::runtime_error(
          "Model post region compiled logical output ordering mismatch for fanout stage.");
    }
  }
}

std::shared_ptr<Node> build_postprocess_node_from_region(
    const Model& model, const internal::ModelPack& pack, const Model::Options& opt, bool sync,
    const internal::SessionRoutePlan& route_plan, const internal::RouteRegion& region,
    const std::size_t region_index) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;

  const std::string stage_name = stage_name_for_post_region(pack, region, region_index);
  const auto boxdecode_route_flags = model_route_flags_for_boxdecode_stage(route_plan);
  switch (region.op_kind) {
  case GraphKind::BoxDecode: {
    const std::optional<bool> route_tess_needed =
        boxdecode_route_flags.has_value() ? std::optional<bool>(boxdecode_route_flags->tess_needed)
                                          : std::nullopt;
    const std::optional<bool> route_quant_needed =
        boxdecode_route_flags.has_value() ? std::optional<bool>(boxdecode_route_flags->quant_needed)
                                          : std::nullopt;
    BoxDecodeType decode_type = opt.decode_type;
    float detection_threshold = opt.score_threshold;
    float nms_iou_threshold = opt.nms_iou_threshold;
    int top_k = opt.top_k;
    BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto;
    const ResolvedPreprocessPlan resolved = model.resolved_preprocess_plan();
    int model_width = 0;
    int model_height = 0;
    if (resolved.effective.resize.width > 0 && resolved.effective.resize.height > 0) {
      model_width = resolved.effective.resize.width;
      model_height = resolved.effective.resize.height;
    } else if (resolved.mla_contract.width > 0 && resolved.mla_contract.height > 0) {
      model_width = resolved.mla_contract.width;
      model_height = resolved.mla_contract.height;
    }
    std::optional<ResizeMode> resize_mode_override;
    if (opt.boxdecode_original_width > 0 && opt.boxdecode_original_height > 0 && model_width > 0 &&
        model_height > 0) {
      resize_mode_override = resolved.effective.resize.mode;
    }
    return simaai::neat::nodes::SimaBoxDecode(
        model, decode_type, detection_threshold, nms_iou_threshold, top_k, stage_name,
        route_tess_needed, route_quant_needed, opt.boxdecode_original_width,
        opt.boxdecode_original_height, model_width, model_height, resize_mode_override,
        decode_type_option);
  }
  case GraphKind::Detess: {
    DetessOptions det_opt = make_detess_options_from_typed_adapter(model, stage_name, sync);
    return simaai::neat::nodes::Detess(std::move(det_opt));
  }
  case GraphKind::DetessCast: {
    DetessCastOptions det_opt = make_detesscast_options_from_typed_adapter(model, stage_name, sync);
    return simaai::neat::nodes::DetessCast(std::move(det_opt));
  }
  case GraphKind::DetessDequant: {
    DetessDequantOptions det_opt =
        make_detessdequant_options_from_typed_adapter(model, stage_name, sync);
    return simaai::neat::nodes::DetessDequant(std::move(det_opt));
  }
  case GraphKind::Dequantize: {
    DequantOptions deq_opt = make_dequant_options_from_typed_adapter(model, stage_name, sync);
    return simaai::neat::nodes::Dequant(std::move(deq_opt));
  }
  case GraphKind::Cast: {
    CastOptions cast_opt = make_cast_options_from_typed_adapter(
        &model, stage_name, CastDirection::Bf16ToFp32, internal::ModelLineageStageRole::ManualPost);
    return simaai::neat::nodes::Cast(std::move(cast_opt));
  }
  case GraphKind::Unknown:
  case GraphKind::Preproc:
  case GraphKind::Quant:
  case GraphKind::Tess:
  case GraphKind::QuantTess:
  case GraphKind::CastTess:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    break;
  }
  throw std::runtime_error("Unsupported post region kernel for model materialization.");
}

// Symmetric mirror of build_postprocess_node_from_region. Builds one node per
// pre-region, dispatching on region.op_kind. Region kind metadata is honored by
// the materializer itself: the region's structural fan-in count is already
// baked into the compiled contract (via the renderer's multi-IO branch which
// fires when the MPK has >1 sibling stages), so a Linear region produces a
// single-IO node and a FanoutMap region produces a multi-input node — both via
// the SAME builder call. This keeps materialization symmetric between the pre
// and post sides and avoids any per-stage exact-name plumbing.
std::shared_ptr<Node>
build_preprocess_node_from_region(const Model& model, const internal::ModelPack& /*pack*/,
                                  const internal::PreprocessPlannerResult& plan,
                                  const InputInfo* input, bool sync,
                                  const internal::RouteRegion& region,
                                  const std::string& stage_name, const std::string& upstream_name) {
  using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
  switch (region.op_kind) {
  case GraphKind::Preproc: {
    PreprocOptions pre_opt = make_preproc_options_from_typed_adapter(model, plan, input, stage_name,
                                                                     upstream_name, sync);
    return simaai::neat::nodes::Preproc(std::move(pre_opt));
  }
  case GraphKind::Quant: {
    const bool allow_multi_io_contract = region.kind == internal::RouteRegionKind::FanoutMap;
    QuantOptions q_opt = make_quant_options_from_typed_adapter(model, plan, input, stage_name, sync,
                                                               nullptr, allow_multi_io_contract);
    return simaai::neat::nodes::Quant(std::move(q_opt));
  }
  case GraphKind::Tess: {
    const bool allow_multi_io_contract = region.kind == internal::RouteRegionKind::FanoutMap;
    TessOptions t_opt = make_tess_options_from_typed_adapter(model, plan, input, stage_name, sync,
                                                             nullptr, allow_multi_io_contract);
    return simaai::neat::nodes::Tess(std::move(t_opt));
  }
  case GraphKind::QuantTess: {
    const bool allow_multi_io_contract = region.kind == internal::RouteRegionKind::FanoutMap;
    QuantTessOptions qt_opt = make_quanttess_options_from_typed_adapter(
        model, plan, input, stage_name, sync, nullptr, allow_multi_io_contract);
    return simaai::neat::nodes::QuantTess(std::move(qt_opt));
  }
  case GraphKind::CastTess: {
    const bool allow_multi_io_contract = region.kind == internal::RouteRegionKind::FanoutMap;
    CastTessOptions ct_opt = make_casttess_options_from_typed_adapter(
        model, plan, input, stage_name, sync, nullptr, allow_multi_io_contract);
    return simaai::neat::nodes::CastTess(std::move(ct_opt));
  }
  case GraphKind::Cast: {
    const bool allow_multi_io_contract = region.kind == internal::RouteRegionKind::FanoutMap;
    CastOptions cast_opt = make_cast_options_from_typed_adapter(
        &model, stage_name, CastDirection::Fp32ToBf16, internal::ModelLineageStageRole::Preprocess,
        nullptr, allow_multi_io_contract);
    return simaai::neat::nodes::Cast(std::move(cast_opt));
  }
  case GraphKind::Unknown:
  case GraphKind::Detess:
  case GraphKind::DetessCast:
  case GraphKind::DetessDequant:
  case GraphKind::Dequantize:
  case GraphKind::BoxDecode:
  case GraphKind::Unpack:
  case GraphKind::Slice:
  case GraphKind::PassThrough:
  case GraphKind::Mla:
    break;
  }
  throw std::runtime_error("Unsupported pre region kernel for model materialization.");
}

std::vector<std::shared_ptr<Node>>
build_preprocess_nodes_impl(const Model& model, const internal::ModelPack& pack,
                            const internal::PreprocessPlannerResult& plan, const InputInfo* input,
                            const std::string& element_name, const std::string& upstream_name,
                            bool sync) {
  if (!plan.session_route_plan.include_pre_stage) {
    return {};
  }

  // Walk the structural pre_regions instead of the flat pre_chain. Each region
  // (Linear or FanoutMap) maps to exactly one materialized pre node. This
  // mirrors build_postprocess_nodes_impl which walks post_regions for the
  // symmetric multi-IO post-MLA fan-out (e.g. detessdequant). For FanoutMap
  // regions the underlying compiled contract already carries
  // logical_inputs.size() == N (the renderer's multi-IO branch fires when the
  // MPK has multiple sibling pre-MLA stages of the same family), so the single
  // emitted node is a multi-input element with N sink-pad bindings — the
  // counterpart of the post side's multi-output element.
  const auto& pre_regions = plan.session_route_plan.pre_regions;
  if (pre_regions.empty()) {
    return {};
  }

  std::vector<std::shared_ptr<Node>> nodes;
  nodes.reserve(pre_regions.size());

  auto region_element_name = [&](std::size_t index,
                                 pipeline_internal::sima::RouteGraphKernelKind op_kind) {
    const bool is_last = index + 1U == pre_regions.size();
    if (is_last && !element_name.empty()) {
      return element_name;
    }
    const std::string base = element_name.empty() ? std::string("prestage") : element_name;
    return base + "_" + std::string(pipeline_internal::sima::route_graph_kernel_name(op_kind)) +
           "_" + std::to_string(index);
  };

  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(
        stderr,
        "[typed-adapter] build_preprocess_nodes_impl element_name=%s pre_regions_size=%zu\n",
        element_name.c_str(), pre_regions.size());
  }

  for (std::size_t i = 0; i < pre_regions.size(); ++i) {
    const auto& region = pre_regions[i];
    const std::string stage_name = region_element_name(i, region.op_kind);
    if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
      std::fprintf(stderr,
                   "[typed-adapter] pre_region[%zu] kind=%d op=%d members=%zu stage_name=%s\n", i,
                   static_cast<int>(region.kind), static_cast<int>(region.op_kind),
                   region.member_plugin_indices.size(), stage_name.c_str());
    }
    nodes.push_back(build_preprocess_node_from_region(model, pack, plan, input, sync, region,
                                                      stage_name, upstream_name));
  }

  return nodes;
}

std::vector<std::shared_ptr<Node>>
build_postprocess_nodes_impl(const Model& model, const internal::ModelPack& pack,
                             const Model::Options& opt, bool sync,
                             const internal::SessionRoutePlan& route_plan) {
  if (!route_plan.include_post_stage) {
    return {};
  }
  const auto& post_regions = route_plan.post_regions;
  if (post_regions.empty()) {
    return {};
  }
  std::vector<std::shared_ptr<Node>> nodes;
  nodes.reserve(post_regions.size());
  for (std::size_t i = 0; i < post_regions.size(); ++i) {
    const auto& region = post_regions[i];
    if (region.kind == internal::RouteRegionKind::FaninJoin) {
      throw std::runtime_error(
          "Model postprocess materialization does not support fanin join regions.");
    }
    if (region.kind == internal::RouteRegionKind::FanoutMap) {
      using GraphKind = pipeline_internal::sima::RouteGraphKernelKind;
      switch (region.op_kind) {
      case GraphKind::Detess:
      case GraphKind::DetessCast:
      case GraphKind::DetessDequant:
      case GraphKind::Cast:
      case GraphKind::Dequantize:
        break;
      case GraphKind::Unknown:
      case GraphKind::Preproc:
      case GraphKind::Quant:
      case GraphKind::Tess:
      case GraphKind::QuantTess:
      case GraphKind::CastTess:
      case GraphKind::BoxDecode:
      case GraphKind::Unpack:
      case GraphKind::Slice:
      case GraphKind::PassThrough:
      case GraphKind::Mla:
        throw std::runtime_error(
            "Model postprocess materialization does not support this fanout region kernel.");
      }
      validate_fanout_region_contract_alignment(pack, post_regions, i);
    }
    nodes.push_back(
        build_postprocess_node_from_region(model, pack, opt, sync, route_plan, region, i));
  }
  return nodes;
}

InputInfo require_input_info(const InputInfo& info, bool tensor_mode) {
  if (info.width <= 0 || info.height <= 0) {
    throw std::runtime_error("Model: input missing width/height");
  }
  if (info.format.empty()) {
    throw std::runtime_error("Model: input missing format");
  }
  if (tensor_mode && info.depth <= 0) {
    throw std::runtime_error("Model: tensor input missing depth");
  }
  return info;
}

std::string tensor_appsrc_format_for_plan(const internal::PreprocessPlannerResult& plan,
                                          const InputOptions& src_opt) {
  std::string format =
      normalize_caps_format_for_media("application/vnd.simaai.tensor", src_opt.format.str());
  if (!plan.session_route_plan.pre_chain.empty()) {
    switch (plan.session_route_plan.pre_chain.front()) {
    case internal::SessionPreStageOp::Quant:
    case internal::SessionPreStageOp::QuantTess:
      return "EVXX_FLOAT32";
    case internal::SessionPreStageOp::Tess: {
      const std::string format_up = upper_copy(format);
      if (format_up.find("FP32") != std::string::npos ||
          format_up.find("FLOAT32") != std::string::npos) {
        return "EVXX_FLOAT32";
      }
      if (format_up.empty() || format_up == "BF16" || format_up == "BFLOAT16" ||
          format_up == "EVXX_BFLOAT16") {
        return "EVXX_BFLOAT16";
      }
      break;
    }
    case internal::SessionPreStageOp::CastTess:
      return "EVXX_FLOAT32";
    case internal::SessionPreStageOp::Cast:
      return "EVXX_FLOAT32";
    case internal::SessionPreStageOp::Preproc:
      break;
    }
  }
  return normalize_caps_format_for_media("application/vnd.simaai.tensor", std::move(format));
}

void align_tensor_input_info_with_route_format(InputInfo& info, const InputOptions& src_opt,
                                               const internal::PreprocessPlannerResult& plan,
                                               bool tensor_mode) {
  if (!tensor_mode) {
    return;
  }
  if (upper_copy(info.media_type) != "APPLICATION/VND.SIMAAI.TENSOR") {
    return;
  }
  if (upper_copy(resolve_input_media_type(src_opt)) != "APPLICATION/VND.SIMAAI.TENSOR") {
    return;
  }
  const std::string selected_format = tensor_appsrc_format_for_plan(plan, src_opt);
  if (selected_format.empty()) {
    return;
  }
  // Route/planner-selected tensor format is authoritative for caps negotiation.
  // Tensor dtype compatibility is enforced separately at push time.
  info.format = selected_format;
  info.format_source = InputInfo::FormatSource::Explicit;
}

void validate_pre_adapter_ingress_expectation(const internal::PreprocessPlannerResult& plan,
                                              const InputInfo& info, const char* where) {
  if (!plan.session_route_plan.include_pre_stage || plan.session_route_plan.pre_chain.empty()) {
    return;
  }
  const auto first_pre = plan.session_route_plan.pre_chain.front();
  if (first_pre != internal::SessionPreStageOp::Quant &&
      first_pre != internal::SessionPreStageOp::Tess &&
      first_pre != internal::SessionPreStageOp::QuantTess &&
      first_pre != internal::SessionPreStageOp::CastTess &&
      first_pre != internal::SessionPreStageOp::Cast) {
    return;
  }

  const auto ingress_contracts = normalized_ingress_contracts(plan.session_route_plan);
  const auto* ingress = maybe_single_ingress_contract(ingress_contracts);
  if (ingress == nullptr) {
    return;
  }
  int expected_w = ingress->width;
  int expected_h = ingress->height;
  int expected_d = ingress->depth;
  if (expected_w <= 0 || expected_h <= 0) {
    expected_w = (plan.modelpack_max_width > 0) ? plan.modelpack_max_width : 0;
    expected_h = (plan.modelpack_max_height > 0) ? plan.modelpack_max_height : 0;
  }
  if (expected_d <= 0) {
    expected_d = (plan.modelpack_input_depth > 0) ? plan.modelpack_input_depth : 0;
  }

  const std::string media_up = upper_copy(info.media_type);
  const std::string fmt_up = upper_copy(info.format);
  const bool expects_fp32 =
      first_pre == internal::SessionPreStageOp::Quant ||
      first_pre == internal::SessionPreStageOp::QuantTess ||
      first_pre == internal::SessionPreStageOp::CastTess ||
      first_pre == internal::SessionPreStageOp::Cast ||
      (ingress != nullptr && (upper_copy(ingress->dtype).find("FP32") != std::string::npos ||
                              upper_copy(ingress->dtype).find("FLOAT32") != std::string::npos)) ||
      plan.session_route_plan.model_managed_route_flags.quant_needed;
  const std::string expected_media = ingress != nullptr && !ingress->media_type.empty()
                                         ? upper_copy(ingress->media_type)
                                         : std::string("APPLICATION/VND.SIMAAI.TENSOR");
  const bool media_ok = (media_up == expected_media);
  const bool format_ok = expects_fp32 ? (fmt_up.find("FP32") != std::string::npos ||
                                         fmt_up.find("FLOAT32") != std::string::npos)
                                      : (fmt_up.find("BF16") != std::string::npos ||
                                         fmt_up.find("BFLOAT16") != std::string::npos);
  const bool size_ok = (expected_w <= 0 || expected_h <= 0) ||
                       (info.width == expected_w && info.height == expected_h);
  const bool depth_ok = (expected_d <= 0) || (info.depth == expected_d);

  if (media_ok && format_ok && size_ok && depth_ok) {
    return;
  }

  std::ostringstream oss;
  oss << (where ? where : "Model") << ": pre-adapter ingress expects "
      << (expects_fp32 ? "FP32" : "BF16") << " tensor";
  if (expected_w > 0 && expected_h > 0) {
    oss << " " << expected_w << "x" << expected_h;
  }
  if (expected_d > 0) {
    oss << "x" << expected_d;
  }
  if (ingress != nullptr && !ingress->layout.empty()) {
    oss << " layout=" << ingress->layout;
  }
  if (ingress != nullptr && !ingress->source_stage.empty()) {
    oss << " source_stage=" << ingress->source_stage;
  }
  oss << ". "
      << "Received media=" << info.media_type << " format=" << info.format
      << " shape=" << info.width << "x" << info.height << "x" << info.depth
      << ". Hint: resize to expected spatial size and convert BGR->RGB before calling run/build.";
  throw std::invalid_argument(oss.str());
}

std::vector<std::shared_ptr<Node>>
build_pipeline_nodes(const Model& model, const internal::ModelPack& pack, const Model::Options& opt,
                     const internal::PreprocessPlannerResult& plan, Model::RouteOptions popt,
                     const InputInfo* input, bool sync, bool externalize_preprocess) {
  const bool tensor_mode = externalize_preprocess ? true : pipeline_requires_tensor_input(plan);
  const bool include_preprocess_stage =
      externalize_preprocess ? false : plan.session_route_plan.include_pre_stage;
  const bool include_postprocess_stage =
      plan.session_route_plan.include_post_stage && !pack.has_terminal_policy();
  const std::string pre_name =
      include_preprocess_stage ? resolved_pre_stage_name(pack, plan) : std::string{};

  std::vector<std::shared_ptr<Node>> nodes;

  InputOptions src_opt;
  if (popt.include_input) {
    if (input) {
      src_opt = appsrc_from_info(require_input_info(*input, tensor_mode));
    } else {
      src_opt = pack.input_appsrc_options(tensor_mode);
      const bool ingress_join = session_route_has_ingress_join(plan.session_route_plan);
      const auto ingress_contracts = normalized_ingress_contracts(plan.session_route_plan);
      const bool overlay_from_ingress_contract =
          should_overlay_appsrc_from_ingress_contract(plan.session_route_plan, tensor_mode);
      if (externalize_preprocess) {
        // Externalized pre-stage branches feed the main MLA session at the
        // resolved MLA contract, even when the ingress path fans out and joins
        // multiple adapter branches first.
        src_opt = override_input_options_from_contract(src_opt, plan.resolved_plan.mla_contract);
      } else if (ingress_contracts.size() == 1U && !ingress_join && overlay_from_ingress_contract) {
        src_opt = overlay_input_options_from_ingress_contract(src_opt, ingress_contracts.front());
      }
      // Multi-IO pre-MLA fan-in: the rendered multi-IO processcvu element
      // advertises tensor-set envelope caps on its sink (built by
      // PreparedRuntimeBridge::build_processcvu_sink_caps_local for routes
      // whose input bindings count > 1). The framework-side appsrc must
      // emit the matching envelope form so caps negotiation succeeds and
      // the bundled multi-memory GstBuffer flows through. Single-ingress
      // and non-fan-in routes are unaffected.
      if (plan_uses_bundled_fan_in(plan)) {
        src_opt.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                                "storage=(string)tensorbuffer";
      }
    }
    if (!popt.buffer_name.empty()) {
      src_opt.buffer_name = popt.buffer_name;
    } else if (!popt.upstream_name.empty()) {
      src_opt.buffer_name = popt.upstream_name;
    } else if (!popt.name_suffix.empty()) {
      src_opt.buffer_name = std::string("decoder") + popt.name_suffix;
    }
    apply_model_ingress_memory_policy(src_opt, plan);
    src_opt.preprocess_meta = make_preprocess_meta_template(plan);
    if (runner_debug_enabled()) {
      const InputOptions pack_opt = pack.input_appsrc_options(tensor_mode);
      if (input) {
        std::fprintf(stderr,
                     "[model-runner] appsrc_source=input_info tensor_mode=%d input_whd=%dx%dx%d "
                     "input_fmt=%s pack_opt{media=%s format=%s max=%dx%dx%d}\n",
                     tensor_mode ? 1 : 0, input->width, input->height, input->depth,
                     input->format.c_str(), resolve_input_media_type(pack_opt).c_str(),
                     pack_opt.format.str().c_str(), pack_opt.max_width, pack_opt.max_height,
                     pack_opt.max_depth);
      } else {
        std::fprintf(
            stderr,
            "[model-runner] appsrc_source=modelpack tensor_mode=%d pack_opt{media=%s format=%s "
            "max=%dx%dx%d}\n",
            tensor_mode ? 1 : 0, resolve_input_media_type(pack_opt).c_str(),
            pack_opt.format.str().c_str(), pack_opt.max_width, pack_opt.max_height,
            pack_opt.max_depth);
      }
      const char* first_pre = "none";
      if (!plan.session_route_plan.pre_chain.empty()) {
        switch (plan.session_route_plan.pre_chain.front()) {
        case internal::SessionPreStageOp::Preproc:
          first_pre = "preproc";
          break;
        case internal::SessionPreStageOp::Quant:
          first_pre = "quant";
          break;
        case internal::SessionPreStageOp::Tess:
          first_pre = "tess";
          break;
        case internal::SessionPreStageOp::QuantTess:
          first_pre = "quanttess";
          break;
        case internal::SessionPreStageOp::Cast:
          first_pre = "cast";
          break;
        }
      }
      std::fprintf(stderr,
                   "[model-runner] build_pipeline_nodes sync=%d tensor_mode=%d input_ptr=%d "
                   "src_opt{media=%s format=%s w=%d h=%d d=%d mem_policy=%s use_simaai_pool=%d} "
                   "first_pre=%s include_pre=%d include_post=%d\n",
                   sync ? 1 : 0, tensor_mode ? 1 : 0, input ? 1 : 0,
                   resolve_input_media_type(src_opt).c_str(), src_opt.format.str().c_str(),
                   src_opt.width, src_opt.height, src_opt.depth,
                   input_memory_policy_name(src_opt.memory_policy), src_opt.use_simaai_pool ? 1 : 0,
                   first_pre, include_preprocess_stage ? 1 : 0, include_postprocess_stage ? 1 : 0);
    }
    nodes.push_back(simaai::neat::nodes::Input(src_opt));
  }

  std::string upstream =
      popt.upstream_name.empty() ? (pre_name.empty() ? "decoder" : pre_name) : popt.upstream_name;
  if (!include_preprocess_stage && popt.include_input && popt.upstream_name.empty() &&
      !src_opt.buffer_name.empty()) {
    upstream = src_opt.buffer_name;
  }

  if (include_preprocess_stage) {
    const std::string pre_upstream =
        (popt.include_input && !src_opt.buffer_name.empty()) ? src_opt.buffer_name : upstream;
    auto pre_nodes =
        build_preprocess_nodes_impl(model, pack, plan, input, pre_name, pre_upstream, sync);
    nodes.insert(nodes.end(), pre_nodes.begin(), pre_nodes.end());
  }

  auto infer_nodes = pack.infer_block(
      upstream, make_stage_lineage_binding(model, internal::ModelLineageStageRole::Infer));
  nodes.insert(nodes.end(), infer_nodes.begin(), infer_nodes.end());

  if (include_postprocess_stage) {
    auto post_nodes = build_postprocess_nodes_impl(model, pack, opt, sync, plan.session_route_plan);
    nodes.insert(nodes.end(), post_nodes.begin(), post_nodes.end());
  }

  if (popt.include_output) {
    nodes.push_back(simaai::neat::nodes::Output());
  }

  return nodes;
}

simaai::neat::GraphOptions route_options_from_model_route_options(const Model::RouteOptions& opt,
                                                                  const Model::Options* model_opt) {
  simaai::neat::GraphOptions sess_opt;
  sess_opt.verbose = opt.verbose;
  sess_opt.processcvu_requested_run_target = opt.processcvu_requested_run_target;
  if (model_opt) {
    sess_opt.processcvu = model_opt->processcvu;
    sess_opt.processmla = model_opt->processmla;
    sess_opt.prepared_runner = model_opt->prepared_runner;
    sess_opt.async_queue_depth = model_opt->async_queue_depth;
  }
  if (!opt.processcvu.pre_run_target.empty() &&
      upper_copy(opt.processcvu.pre_run_target) != "AUTO") {
    sess_opt.processcvu.pre_run_target = opt.processcvu.pre_run_target;
  }
  if (!opt.processcvu.post_run_target.empty() &&
      upper_copy(opt.processcvu.post_run_target) != "AUTO") {
    sess_opt.processcvu.post_run_target = opt.processcvu.post_run_target;
  }
  if (opt.processcvu.async) {
    sess_opt.processcvu.async = true;
  }
  if (opt.processmla.async) {
    sess_opt.processmla.async = true;
  }
  if (opt.processmla.output_pool_buffers > 0) {
    sess_opt.processmla.output_pool_buffers = opt.processmla.output_pool_buffers;
  }
  if (opt.processmla.defer_output_invalidate) {
    sess_opt.processmla.defer_output_invalidate = true;
  }
  if (!opt.prepared_runner.mode.empty()) {
    sess_opt.prepared_runner.mode = opt.prepared_runner.mode;
  }
  if (opt.prepared_runner.ring_depth > 0) {
    sess_opt.prepared_runner.ring_depth = opt.prepared_runner.ring_depth;
  }
  if (opt.prepared_runner.profile) {
    sess_opt.prepared_runner.profile = true;
  }
  if (!opt.prepared_runner.dequant_flags.empty()) {
    sess_opt.prepared_runner.dequant_flags = opt.prepared_runner.dequant_flags;
  }
  if (opt.async_queue_depth > 0) {
    sess_opt.async_queue_depth = opt.async_queue_depth;
  }
  return sess_opt;
}

void add_nodes_to_graph(Graph& graph, std::vector<std::shared_ptr<Node>> nodes) {
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
}

Graph graph_from_nodes(std::vector<std::shared_ptr<Node>> nodes, const GraphOptions& opt = {}) {
  Graph graph(opt);
  add_nodes_to_graph(graph, std::move(nodes));
  return graph;
}

const char* public_stage_role_name(Model::Stage stage) {
  switch (stage) {
  case Model::Stage::Preprocess:
    return "preprocess";
  case Model::Stage::Inference:
    return "inference";
  case Model::Stage::Postprocess:
    return "postprocess";
  case Model::Stage::Full:
    return "full";
  }
  return "fragment";
}

template <typename EnumT> int model_options_enum_int(EnumT value) {
  return static_cast<int>(value);
}

nlohmann::ordered_json resize_spec_json(const ResizeSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)},
          {"width", spec.width},
          {"height", spec.height},
          {"mode", model_options_enum_int(spec.mode)},
          {"pad_value", spec.pad_value},
          {"scaling_type", spec.scaling_type}};
}

nlohmann::ordered_json color_convert_spec_json(const ColorConvertSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)},
          {"input_format", model_options_enum_int(spec.input_format)},
          {"output_format", model_options_enum_int(spec.output_format)}};
}

nlohmann::ordered_json layout_convert_spec_json(const LayoutConvertSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)}, {"perm", spec.perm}};
}

nlohmann::ordered_json normalize_spec_json(const NormalizeSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)},
          {"mean", spec.mean},
          {"stddev", spec.stddev},
          {"has_explicit_stats", spec.has_explicit_stats}};
}

nlohmann::ordered_json quantize_spec_json(const QuantizeSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)},
          {"zero_point", spec.zero_point},
          {"scale", spec.scale},
          {"output_dtype", spec.output_dtype}};
}

nlohmann::ordered_json tessellate_spec_json(const TessellateSpec& spec) {
  return {{"enable", model_options_enum_int(spec.enable)}, {"slice_shape", spec.slice_shape}};
}

nlohmann::ordered_json transform_json(const Transform& transform) {
  return {{"type", model_options_enum_int(transform.type)},
          {"resize", resize_spec_json(transform.resize)},
          {"color_convert", color_convert_spec_json(transform.color_convert)},
          {"layout_convert", layout_convert_spec_json(transform.layout_convert)},
          {"normalize", normalize_spec_json(transform.normalize)},
          {"quantize", quantize_spec_json(transform.quantize)},
          {"tessellate", tessellate_spec_json(transform.tessellate)}};
}

nlohmann::ordered_json preprocess_options_json(const PreprocessOptions& opt) {
  nlohmann::ordered_json transforms = nlohmann::ordered_json::array();
  for (const auto& transform : opt.transforms) {
    transforms.push_back(transform_json(transform));
  }
  return {{"kind", model_options_enum_int(opt.kind)},
          {"enable", model_options_enum_int(opt.enable)},
          {"input_max_width", opt.input_max_width},
          {"input_max_height", opt.input_max_height},
          {"input_max_depth", opt.input_max_depth},
          {"resize", resize_spec_json(opt.resize)},
          {"color_convert", color_convert_spec_json(opt.color_convert)},
          {"layout_convert", layout_convert_spec_json(opt.layout_convert)},
          {"normalize", normalize_spec_json(opt.normalize)},
          {"quantize", quantize_spec_json(opt.quantize)},
          {"tessellate", tessellate_spec_json(opt.tessellate)},
          {"transforms", std::move(transforms)},
          {"preset", model_options_enum_int(opt.preset)}};
}

nlohmann::ordered_json processcvu_options_json(const ProcessCvuOptions& opt) {
  return {{"pre_run_target", opt.pre_run_target},
          {"post_run_target", opt.post_run_target},
          {"async", opt.async}};
}

nlohmann::ordered_json processmla_options_json(const ProcessMlaOptions& opt) {
  return {{"async", opt.async},
          {"output_pool_buffers", opt.output_pool_buffers},
          {"defer_output_invalidate", opt.defer_output_invalidate}};
}

nlohmann::ordered_json prepared_runner_options_json(const PreparedRunnerOptions& opt) {
  return {{"mode", opt.mode},
          {"ring_depth", opt.ring_depth},
          {"profile", opt.profile},
          {"dequant_flags", opt.dequant_flags}};
}

std::string model_options_json_for_graph_provenance(const Model::Options& opt) {
  nlohmann::ordered_json terminal;
  terminal["mla_only"] = opt.inference_terminal.mla_only;
  if (opt.inference_terminal.last_stage_index.has_value()) {
    terminal["last_stage_index"] = *opt.inference_terminal.last_stage_index;
  }
  if (opt.inference_terminal.last_stage_name.has_value()) {
    terminal["last_stage_name"] = *opt.inference_terminal.last_stage_name;
  }
  if (opt.inference_terminal.last_plugin_id.has_value()) {
    terminal["last_plugin_id"] = *opt.inference_terminal.last_plugin_id;
  }
  if (opt.inference_terminal.last_processor.has_value()) {
    terminal["last_processor"] = *opt.inference_terminal.last_processor;
  }

  nlohmann::ordered_json out;
  out["preprocess"] = preprocess_options_json(opt.preprocess);
  out["decode_type"] = model_options_enum_int(opt.decode_type);
  out["score_threshold"] = opt.score_threshold;
  out["nms_iou_threshold"] = opt.nms_iou_threshold;
  out["top_k"] = opt.top_k;
  out["boxdecode_original_width"] = opt.boxdecode_original_width;
  out["boxdecode_original_height"] = opt.boxdecode_original_height;
  out["upstream_name"] = opt.upstream_name;
  out["name_suffix"] = opt.name_suffix;
  out["cleanup_extracted_model_data"] = opt.cleanup_extracted_model_data;
  out["inference_terminal"] = std::move(terminal);
  out["processcvu"] = processcvu_options_json(opt.processcvu);
  out["processmla"] = processmla_options_json(opt.processmla);
  out["prepared_runner"] = prepared_runner_options_json(opt.prepared_runner);
  out["async_queue_depth"] = opt.async_queue_depth;
  return out.dump();
}

} // namespace

Graph Model::preprocess() const {
  return internal::ModelAccess::build_stage_graph_fragment(*this, Model::Stage::Preprocess);
}

Graph Model::inference() const {
  return internal::ModelAccess::build_stage_graph_fragment(*this, Model::Stage::Inference);
}

Graph Model::postprocess() const {
  return internal::ModelAccess::build_stage_graph_fragment(*this, Model::Stage::Postprocess);
}

Graph Model::graph() const {
  return graph(Model::RouteOptions{});
}

Graph Model::graph(Model::RouteOptions opt) const {
  return internal::ModelAccess::build_graph_fragment(*this, std::move(opt));
}

TensorSpec Model::input_spec() const {
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  require_single_ingress_api(ingress_contracts, "Model::input_spec");
  const auto specs = input_specs();
  if (!specs.empty()) {
    return specs.front();
  }
  return TensorSpec{};
}

int Model::compiled_batch_size() const {
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  int batch = 1;
  for (const auto& ingress : ingress_contracts) {
    if (ingress.batch > batch) {
      batch = ingress.batch;
    }
  }
  return batch;
}

std::vector<TensorSpec> Model::input_specs() const {
  std::vector<TensorSpec> specs;
  TensorSpec spec;
  const auto& route_plan = impl_->preprocess_plan.session_route_plan;
  const bool tensor_mode = pipeline_requires_tensor_input(impl_->preprocess_plan);
  const bool require_fp32_boundary = route_plan.pre_cast_fp32_to_bf16;
  const auto ingress_contracts = normalized_ingress_contracts(route_plan);
  if (tensor_mode) {
    const auto input_opts = input_appsrc_options_list(true);
    if (!ingress_contracts.empty()) {
      specs.reserve(ingress_contracts.size());
      for (std::size_t i = 0; i < ingress_contracts.size(); ++i) {
        TensorSpec ingress_spec = tensor_spec_from_ingress_contract(ingress_contracts[i]);
        const InputOptions& opt = i < input_opts.size() ? input_opts[i] : input_opts.front();
        if (ingress_spec.dtypes.empty()) {
          ingress_spec.dtypes = {dtype_from_format(opt.format)};
        }
        if (require_fp32_boundary) {
          ingress_spec.dtypes = {TensorDType::Float32};
        }
        if (ingress_spec.shape.empty()) {
          int d = (opt.depth > 0) ? opt.depth : opt.max_depth;
          if (d <= 0) {
            d = 3;
          }
          ingress_spec.shape = {-1, -1, d};
        }
        ingress_spec.rank = static_cast<int>(ingress_spec.shape.size());
        specs.push_back(std::move(ingress_spec));
      }
      return specs;
    }

    const InputOptions opt = impl_->pack.input_appsrc_options(true);
    if (spec.shape.empty()) {
      const auto infer = internal::ModelAccess::build_public_inference_nodes(*this);
      const auto mla_input_tensor_info =
          rendered_stage_query::mla_input_tensor_info_from_nodes(infer);
      const stages::TensorDims mla_dims = dims_from_mla_logical_contract_shape(
          mla_input_tensor_info.logical_shape, mla_input_tensor_info.logical_layout);
      int d = (mla_dims.depth > 0) ? mla_dims.depth : 0;
      if (d <= 0)
        d = (opt.depth > 0) ? opt.depth : opt.max_depth;
      if (d <= 0)
        d = 3;
      if (mla_dims.width > 0 && mla_dims.height > 0 && d > 0) {
        spec.shape = {mla_dims.height, mla_dims.width, d};
      } else {
        spec.shape = {-1, -1, d > 0 ? d : -1};
      }
    }
    if (spec.dtypes.empty()) {
      spec.dtypes = {dtype_from_format(opt.format)};
    }
    if (require_fp32_boundary) {
      spec.dtypes = {TensorDType::Float32};
    }
    if (spec.shape.empty()) {
      int d = (opt.depth > 0) ? opt.depth : opt.max_depth;
      if (d <= 0)
        d = 3;
      spec.shape = {-1, -1, d};
    }
  } else {
    spec.dtypes = {TensorDType::UInt8};
    const InputOptions opt = impl_->pack.input_appsrc_options(false);
    if (is_gray_format(opt.format)) {
      spec.shape = {-1, -1};
    } else {
      int depth = (opt.depth > 0) ? opt.depth : opt.max_depth;
      if (depth <= 0)
        depth = default_depth_for_format(opt.format);
      spec.shape = {-1, -1, depth > 0 ? depth : -1};
    }
    spec.image_format = pixel_format_from_string(opt.format);
  }
  if (!spec.shape.empty()) {
    spec.rank = static_cast<int>(spec.shape.size());
  }
  specs.push_back(std::move(spec));
  return specs;
}

TensorSpec Model::output_spec() const {
  const std::vector<TensorSpec> specs = output_specs();
  if (!specs.empty()) {
    return specs.front();
  }
  return TensorSpec{};
}

std::vector<TensorSpec> Model::output_specs() const {
  std::vector<TensorSpec> specs;
  TensorSpec spec;
  const auto& route_plan = impl_->preprocess_plan.session_route_plan;
  const bool include_post = route_plan.include_post_stage;
  const internal::PostRouteStageKind selected_post_kind = route_plan.selected_post_kind;
  const bool has_box =
      include_post && selected_post_kind == internal::PostRouteStageKind::BoxDecode;
  const bool has_detess =
      include_post && (selected_post_kind == internal::PostRouteStageKind::Detess ||
                       selected_post_kind == internal::PostRouteStageKind::DetessDequant);
  const bool has_dequant =
      include_post && selected_post_kind == internal::PostRouteStageKind::Dequantize;
  const bool has_cast = include_post && selected_post_kind == internal::PostRouteStageKind::Cast;
  const bool implicit_detess_cast = has_detess && route_plan.post_cast_bf16_to_fp32;
  const bool require_fp32_boundary = route_plan.post_cast_bf16_to_fp32;
  const auto push_spec = [&specs](TensorSpec s) {
    if (!s.shape.empty() && s.rank < 0) {
      s.rank = static_cast<int>(s.shape.size());
    }
    specs.push_back(std::move(s));
  };
  const auto append_from_egress_contracts = [&](bool force_float_if_missing_dtype) -> bool {
    if (!include_post) {
      return false;
    }
    std::vector<TensorSpec> candidates;
    if (!route_plan.egress_contracts.empty()) {
      for (const auto& egress : route_plan.egress_contracts) {
        TensorSpec out_spec;
        if (!populate_spec_from_route_contract(egress, &out_spec) || out_spec.shape.empty()) {
          continue;
        }
        if (out_spec.dtypes.empty() && force_float_if_missing_dtype) {
          out_spec.dtypes = {TensorDType::Float32};
        }
        candidates.push_back(std::move(out_spec));
      }
    } else {
      TensorSpec out_spec;
      if (!populate_spec_from_route_contract(route_plan.egress_contract, &out_spec) ||
          out_spec.shape.empty()) {
        return false;
      }
      if (out_spec.dtypes.empty() && force_float_if_missing_dtype) {
        out_spec.dtypes = {TensorDType::Float32};
      }
      candidates.push_back(std::move(out_spec));
    }
    if (candidates.empty()) {
      return false;
    }

    if (require_fp32_boundary) {
      std::vector<TensorSpec> fp32_only;
      fp32_only.reserve(candidates.size());
      for (auto& out_spec : candidates) {
        if (out_spec.dtypes.empty()) {
          out_spec.dtypes = {TensorDType::Float32};
          fp32_only.push_back(std::move(out_spec));
          continue;
        }
        if (out_spec.dtypes.front() == TensorDType::Float32) {
          fp32_only.push_back(std::move(out_spec));
        }
      }
      if (!fp32_only.empty()) {
        candidates = std::move(fp32_only);
      } else {
        for (auto& out_spec : candidates) {
          out_spec.dtypes = {TensorDType::Float32};
        }
      }
    }

    for (auto& out_spec : candidates) {
      push_spec(std::move(out_spec));
    }
    return true;
  };
  const auto append_from_rendered_terminal_stage =
      [&](const std::vector<std::shared_ptr<Node>>& group) -> bool {
    const auto info = rendered_stage_query::terminal_output_info(group, false);
    if (info.outputs.empty()) {
      return false;
    }
    for (const auto& out : info.outputs) {
      TensorSpec out_spec;
      out_spec.dtypes = {info.dtype};
      out_spec.shape = out.shape;
      out_spec.rank = static_cast<int>(out_spec.shape.size());
      push_spec(std::move(out_spec));
    }
    return true;
  };

  if (has_box) {
    spec.dtypes = {TensorDType::UInt8};
    spec.rank = -1;
    push_spec(std::move(spec));
    return specs;
  }
  if (has_detess) {
    auto post = build_postprocess_nodes_impl(*this, impl_->pack, impl_->options, false, route_plan);
    if (append_from_rendered_terminal_stage(post)) {
      return specs;
    }
    const auto info = rendered_stage_query::detessdequant_output_info(post, false);
    if (!info.outputs.empty()) {
      const TensorDType out_dtype = implicit_detess_cast ? TensorDType::Float32 : info.dtype;
      for (const auto& out : info.outputs) {
        TensorSpec out_spec;
        out_spec.dtypes = {out_dtype};
        out_spec.shape = out.shape;
        out_spec.rank = static_cast<int>(out_spec.shape.size());
        push_spec(std::move(out_spec));
      }
      return specs;
    }
    if (append_from_egress_contracts(true)) {
      return specs;
    }
    throw std::runtime_error(
        "Model::output_specs: detess post route requires explicit egress contracts or "
        "rendered detess output contracts.");
  }
  if (has_dequant) {
    auto post = build_postprocess_nodes_impl(*this, impl_->pack, impl_->options, false, route_plan);
    if (append_from_rendered_terminal_stage(post)) {
      return specs;
    }
    const auto info = rendered_stage_query::dequant_output_info(post, false);
    if (!info.outputs.empty()) {
      for (const auto& out : info.outputs) {
        TensorSpec out_spec;
        out_spec.dtypes = {info.dtype};
        out_spec.shape = out.shape;
        out_spec.rank = static_cast<int>(out_spec.shape.size());
        push_spec(std::move(out_spec));
      }
      return specs;
    }
    if (append_from_egress_contracts(true)) {
      return specs;
    }
    throw std::runtime_error(
        "Model::output_specs: dequant post route requires explicit egress contracts or "
        "rendered dequant output contracts.");
  }
  if (has_cast) {
    auto post = build_postprocess_nodes_impl(*this, impl_->pack, impl_->options, false, route_plan);
    if (append_from_rendered_terminal_stage(post)) {
      return specs;
    }
    if (append_from_egress_contracts(true)) {
      return specs;
    }
    throw std::runtime_error(
        "Model::output_specs: cast post route requires explicit egress contracts.");
  }
  if (append_from_egress_contracts(has_detess || has_dequant || has_cast)) {
    return specs;
  }
  if (append_from_egress_contracts(true)) {
    return specs;
  }
  const auto infer = internal::ModelAccess::build_public_inference_nodes(*this);
  const auto mla_outputs = rendered_stage_query::mla_output_tensors_from_nodes(infer);
  if (mla_outputs.empty()) {
    throw std::runtime_error(
        "Model::output_specs: inference group is missing rendered MLA output contracts.");
  }
  for (const auto& output : mla_outputs) {
    TensorSpec out_spec;
    out_spec.dtypes = {dtype_from_format(output.data_type)};
    out_spec.shape = output.shape;
    out_spec.rank = static_cast<int>(out_spec.shape.size());
    push_spec(std::move(out_spec));
  }
  return specs;
}

ResolvedPreprocessPlan Model::resolved_preprocess_plan() const {
  return impl_->preprocess_plan.resolved_plan;
}

Model::PreprocessRequirements Model::preprocess_requirements() const {
  PreprocessRequirements out;
  const auto& route = impl_->preprocess_plan.session_route_plan;
  const auto flags = resolve_preprocess_contract_flags(impl_->preprocess_plan);
  out.has_preproc_stage = resolved_preprocess_graph_contains_stage(
      impl_->preprocess_plan, internal::StageNodeKind::Preproc);
  out.quant_needed = flags.quant_needed;
  out.tess_needed = flags.tess_needed;

  const auto ingress_contracts = normalized_ingress_contracts(route);
  require_single_ingress_api(ingress_contracts, "Model::preprocess_requirements");
  const auto* ingress = maybe_single_ingress_contract(ingress_contracts);
  if (ingress != nullptr) {
    out.input_media_type = ingress->media_type;
    out.input_format =
        !ingress->dtype.empty() ? ingress->dtype : impl_->preprocess_plan.modelpack_format;
  } else {
    out.input_media_type = impl_->preprocess_plan.modelpack_media_type;
    out.input_format = impl_->preprocess_plan.modelpack_format;
  }

  const auto& effective = impl_->preprocess_plan.resolved_plan.effective;
  if (effective.color_convert.output_format != PreprocessColorFormat::Auto) {
    out.output_format = preprocess_color_format_name(effective.color_convert.output_format);
  } else if (effective.color_convert.input_format != PreprocessColorFormat::Auto) {
    out.output_format = preprocess_color_format_name(effective.color_convert.input_format);
  } else {
    out.output_format = "RGB";
  }
  out.output_dtype = resolve_preproc_output_dtype(impl_->preprocess_plan, flags);
  if (out.output_dtype.empty()) {
    const std::string ingress_dtype = (ingress != nullptr && !ingress->dtype.empty())
                                          ? ingress->dtype
                                          : impl_->preprocess_plan.modelpack_format;
    out.output_dtype = format_is_bf16_tensor(ingress_dtype) ? "EVXX_BFLOAT16" : "INT16";
  }
  const int output_depth = pipeline_internal::default_depth_for_image_format(
      out.output_format,
      ingress != nullptr ? ingress->depth : impl_->preprocess_plan.modelpack_input_depth);
  if (effective.resize.height > 0 && effective.resize.width > 0 && output_depth > 0) {
    out.output_shape = {effective.resize.height, effective.resize.width, output_depth};
  }
  out.axis_perm = effective.layout_convert.perm;
  if (effective.tessellate.has_slice_shape()) {
    out.slice_shape = effective.tessellate.slice_shape;
  }
  if (effective.quantize.scale > 0.0) {
    out.q_scale = effective.quantize.scale;
  }
  out.q_zp = static_cast<std::int64_t>(effective.quantize.zero_point);
  return out;
}

Model::ModelInfo Model::info() const {
  ModelInfo out;
  const internal::ParsedModelInfo parsed = internal::parse_model_from_pack(impl_->pack);

  out.mpk_json_path = parsed.mpk_json_path;
  out.model_name = parsed.model_name;

  out.needs.pre_quantization = parsed.needs.pre_quantization;
  out.needs.pre_tessellation = parsed.needs.pre_tessellation;
  out.needs.pre_cast = parsed.needs.pre_cast;
  out.needs.post_detessellation = parsed.needs.post_detessellation;
  out.needs.post_dequantization = parsed.needs.post_dequantization;
  out.needs.post_cast = parsed.needs.post_cast;

  out.capabilities.has_pre_quantization = parsed.capabilities.has_pre_quantization;
  out.capabilities.has_pre_tessellation = parsed.capabilities.has_pre_tessellation;
  out.capabilities.has_pre_cast = parsed.capabilities.has_pre_cast;
  out.capabilities.has_post_detessellation = parsed.capabilities.has_post_detessellation;
  out.capabilities.has_post_dequantization = parsed.capabilities.has_post_dequantization;
  out.capabilities.has_post_cast = parsed.capabilities.has_post_cast;
  out.capabilities.has_post_boxdecode = parsed.capabilities.has_post_boxdecode;

  out.selection.include_preprocess_stage =
      impl_->preprocess_plan.session_route_plan.include_pre_stage;
  out.selection.include_postprocess_stage =
      impl_->preprocess_plan.session_route_plan.include_post_stage;
  out.selection.infer_only = impl_->preprocess_plan.infer_only_route;
  out.selection.preprocess_graph =
      preprocess_graph_family_name(impl_->preprocess_plan.resolved_plan.graph_family);
  out.selection.selected_post_kind =
      post_route_stage_kind_name(impl_->preprocess_plan.session_route_plan.selected_post_kind);

  out.output_topology.physical_outputs = parsed.outputs.physical.size();
  out.output_topology.logical_outputs = parsed.outputs.logical.size();
  out.output_topology.packed_outputs = parsed.outputs.packed_output;

  out.pre_kernels = parsed.pre_kernels;
  out.post_kernels = parsed.post_kernels;
  out.warnings = parsed.warnings;
  return out;
}

std::unordered_map<std::string, std::string> Model::metadata() const {
  std::unordered_map<std::string, std::string> out;
  const std::string path = impl_->pack.etc_dir() + "/metadata.json";
  std::ifstream in(path);
  if (!in.is_open())
    return out;
  nlohmann::json j;
  in >> j;
  if (!j.is_object())
    return out;
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.value().is_string()) {
      out[it.key()] = it.value().get<std::string>();
    } else {
      out[it.key()] = it.value().dump();
    }
  }
  return out;
}

Graph Model::fragment(Stage stage) const {
  return internal::ModelAccess::build_stage_graph_fragment(*this, stage);
}

std::string Model::backend_fragment(Stage stage) const {
  internal::ModelStage s = internal::ModelStage::Full;
  switch (stage) {
  case Stage::Preprocess:
    s = internal::ModelStage::Preprocess;
    break;
  case Stage::Inference:
    s = internal::ModelStage::MlaOnly;
    break;
  case Stage::Postprocess:
    s = internal::ModelStage::Postprocess;
    break;
  case Stage::Full:
    s = internal::ModelStage::Full;
    break;
  }
  return impl_->pack.backend_fragment(s);
}

InputOptions Model::input_appsrc_options(bool tensor_mode) const {
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  require_single_ingress_api(ingress_contracts, "Model::input_appsrc_options");
  const auto opts = input_appsrc_options_list(tensor_mode);
  if (!opts.empty()) {
    return opts.front();
  }
  // Route contract takes precedence over caller hint: adapter-only pre routes
  // (quant/tess/quanttess ingress) require tensor appsrc for valid caps.
  const bool route_requires_tensor = pipeline_requires_tensor_input(impl_->preprocess_plan);
  const bool effective_tensor_mode = tensor_mode || route_requires_tensor;
  InputOptions opt = impl_->pack.input_appsrc_options(effective_tensor_mode);
  apply_model_ingress_memory_policy(opt, impl_->preprocess_plan);
  return opt;
}

std::vector<InputOptions> Model::input_appsrc_options_list(bool tensor_mode) const {
  const bool route_requires_tensor = pipeline_requires_tensor_input(impl_->preprocess_plan);
  const bool effective_tensor_mode = tensor_mode || route_requires_tensor;
  InputOptions base = impl_->pack.input_appsrc_options(effective_tensor_mode);
  apply_model_ingress_memory_policy(base, impl_->preprocess_plan);

  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  if (ingress_contracts.empty()) {
    return {base};
  }
  if (!should_overlay_appsrc_from_ingress_contract(impl_->preprocess_plan.session_route_plan,
                                                   effective_tensor_mode)) {
    return std::vector<InputOptions>(ingress_contracts.size(), base);
  }

  std::vector<InputOptions> out;
  out.reserve(ingress_contracts.size());
  for (const auto& ingress : ingress_contracts) {
    InputOptions opt = overlay_input_options_from_ingress_contract(base, ingress);
    out.push_back(std::move(opt));
  }
  return out;
}

std::string Model::find_config_path_by_plugin(const std::string& plugin_id) const {
  return impl_->pack.find_config_path_by_plugin(plugin_id);
}

std::string Model::find_config_path_by_processor(const std::string& processor) const {
  return impl_->pack.find_config_path_by_processor(processor);
}

std::string Model::infer_output_name() const {
  const auto frag = impl_->pack.fragment(internal::ModelStage::MlaOnly);
  if (frag.elements.empty())
    return {};
  return frag.elements.back();
}

const Model::RouteOptions& Model::default_route_options() {
  static const Model::RouteOptions opt{};
  return opt;
}

Model::RouteOptions route_options_for_model_runner(Model::RouteOptions opt) {
  opt.include_input = true;
  opt.include_output = true;
  return opt;
}

const simaai::neat::RunOptions& Model::default_run_options() {
  static const simaai::neat::RunOptions opt{};
  return opt;
}

namespace {} // namespace

Model::Runner::Runner(simaai::neat::Run run) : run_(std::move(run)), ingress_names_({"ifm0"}) {}

Model::Runner::Runner(simaai::neat::Run run, const simaai::neat::InputOptions& tensor_input_opt)
    : run_(std::move(run)), tensor_input_opt_for_cv_(tensor_input_opt), ingress_names_({"ifm0"}) {}

Model::Runner::Runner(simaai::neat::Run run, const simaai::neat::InputOptions& tensor_input_opt,
                      std::vector<std::string> ingress_names)
    : run_(std::move(run)), tensor_input_opt_for_cv_(tensor_input_opt),
      ingress_names_(std::move(ingress_names)) {
  if (ingress_names_.empty()) {
    ingress_names_.push_back("ifm0");
  }
}

Model::Runner::Runner(simaai::neat::Run run, std::vector<std::string> ingress_names)
    : run_(std::move(run)), ingress_names_(std::move(ingress_names)) {
  if (ingress_names_.empty()) {
    ingress_names_.push_back("ifm0");
  }
}

Model::Runner::operator bool() const noexcept {
  return static_cast<bool>(run_);
}

#if defined(SIMA_WITH_OPENCV)
bool Model::Runner::push(const std::vector<cv::Mat>& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::Runner::push: empty image list");
  }
  if (tensor_input_opt_for_cv_.has_value()) {
    std::vector<internal::IngressTensorContract> ingress_contracts;
    ingress_contracts.reserve(ingress_names_.size());
    for (std::size_t i = 0; i < ingress_names_.size(); ++i) {
      internal::IngressTensorContract ingress;
      ingress.ingress_index = static_cast<int>(i);
      ingress.source_tensor_name = ingress_names_[i];
      ingress_contracts.push_back(std::move(ingress));
    }
    require_exact_ingress_count(inputs.size(), ingress_contracts, "Model::Runner::push",
                                "image inputs");
    TensorList tensors;
    tensors.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      if (runner_debug_enabled()) {
        std::fprintf(
            stderr, "[model-runner] push(cv::Mat) -> tensor media=%s format=%s w=%d h=%d d=%d\n",
            resolve_input_media_type(*tensor_input_opt_for_cv_).c_str(),
            tensor_input_opt_for_cv_->format.str().c_str(), tensor_input_opt_for_cv_->width,
            tensor_input_opt_for_cv_->height, tensor_input_opt_for_cv_->depth);
      }
      const cv::Mat prepared = maybe_resize_mat_to_tensor_caps(input, *tensor_input_opt_for_cv_);
      Tensor tensor =
          tensor_from_cv_mat(prepared, *tensor_input_opt_for_cv_, "Model::Runner::push");
      tensor.route.name = ingress_names_[i];
      tensor.route.segment_name = ingress_names_[i];
      tensor.route.backend_name = ingress_names_[i];
      tensors.emplace_back(std::move(tensor));
    }
    return run_.push(tensors);
  }
  return run_.push(inputs);
}
#endif

bool Model::Runner::push(const simaai::neat::TensorList& inputs) {
  return run_.push(inputs);
}

bool Model::Runner::push(const simaai::neat::Sample& inputs) {
  return run_.push(inputs);
}

simaai::neat::Sample Model::Runner::pull(int timeout_ms) {
  return run_.pull_samples(timeout_ms);
}

#if defined(SIMA_WITH_OPENCV)
simaai::neat::TensorList Model::Runner::run(const std::vector<cv::Mat>& inputs, int timeout_ms) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::Runner::run: empty image list");
  }
  if (tensor_input_opt_for_cv_.has_value()) {
    std::vector<internal::IngressTensorContract> ingress_contracts;
    ingress_contracts.reserve(ingress_names_.size());
    for (std::size_t i = 0; i < ingress_names_.size(); ++i) {
      internal::IngressTensorContract ingress;
      ingress.ingress_index = static_cast<int>(i);
      ingress.source_tensor_name = ingress_names_[i];
      ingress_contracts.push_back(std::move(ingress));
    }
    require_exact_ingress_count(inputs.size(), ingress_contracts, "Model::Runner::run",
                                "image inputs");
    TensorList tensors;
    tensors.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      if (runner_debug_enabled()) {
        std::fprintf(
            stderr, "[model-runner] run(cv::Mat) -> tensor media=%s format=%s w=%d h=%d d=%d\n",
            resolve_input_media_type(*tensor_input_opt_for_cv_).c_str(),
            tensor_input_opt_for_cv_->format.str().c_str(), tensor_input_opt_for_cv_->width,
            tensor_input_opt_for_cv_->height, tensor_input_opt_for_cv_->depth);
      }
      const cv::Mat prepared = maybe_resize_mat_to_tensor_caps(input, *tensor_input_opt_for_cv_);
      Tensor tensor = tensor_from_cv_mat(prepared, *tensor_input_opt_for_cv_, "Model::Runner::run");
      tensor.route.name = ingress_names_[i];
      tensor.route.segment_name = ingress_names_[i];
      tensor.route.backend_name = ingress_names_[i];
      tensors.emplace_back(std::move(tensor));
    }
    return run_.run(tensors, timeout_ms);
  }
  return run_.run(inputs, timeout_ms);
}
#endif

simaai::neat::TensorList Model::Runner::run(const simaai::neat::TensorList& inputs,
                                            int timeout_ms) {
  return run_.run(inputs, timeout_ms);
}

simaai::neat::Sample Model::Runner::run(const simaai::neat::Sample& inputs, int timeout_ms) {
  return run_.run(inputs, timeout_ms);
}

simaai::neat::MeasureScope
Model::Runner::start_measurement(const simaai::neat::MeasureOptions& opt) {
  return run_.start_measurement(opt);
}

int Model::Runner::warmup(const simaai::neat::TensorList& inputs, int warm, int timeout_ms) {
  if (warm < 0) {
    warm = env_int("SIMA_ASYNC_WARMUP", 0);
  }
  if (warm <= 0) {
    warn_no_warmup_once();
    return 0;
  }
  for (int i = 0; i < warm; ++i) {
    (void)run(inputs, timeout_ms);
  }
  return warm;
}

void Model::Runner::close() {
  run_.close();
}

simaai::neat::RunStats Model::Runner::stats() const {
  return run_.stats();
}

simaai::neat::RunMeasurementSummary Model::Runner::measurement_summary() const {
  return run_.measurement_summary();
}

simaai::neat::RuntimeMetrics
Model::Runner::metrics(const simaai::neat::RuntimeMetricsOptions& opt) const {
  simaai::neat::RuntimeMetrics out = run_.metrics(opt);
  out.source_kind = "model";
  return out;
}

std::string Model::Runner::metrics_report(const simaai::neat::RuntimeMetricsOptions& opt,
                                          simaai::neat::RuntimeMetricsFormat format) const {
  return simaai::neat::format_runtime_metrics(metrics(opt), format);
}

std::string Model::Runner::metrics_report(simaai::neat::RuntimeMetricsFormat format) const {
  return metrics_report(simaai::neat::RuntimeMetricsOptions{}, format);
}

simaai::neat::RunDiagSnapshot Model::Runner::diag_snapshot() const {
  return run_.diag_snapshot();
}

std::string Model::Runner::report(const simaai::neat::RunReportOptions& opt) const {
  return run_.report(opt);
}

void Model::Runner::close_input() {
  run_.close_input();
}

Model::Runner Model::build() {
  return build(Model::RouteOptions{}, simaai::neat::RunOptions{});
}

Model::Runner Model::build(const Model::RouteOptions& opt) {
  return build(opt, simaai::neat::RunOptions{});
}

Model::Runner Model::build(const simaai::neat::RunOptions& run_opt) {
  return build(Model::RouteOptions{}, run_opt);
}

Model::Runner Model::build(const Model::RouteOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  const Model::RouteOptions build_opt = route_options_for_model_runner(opt);
  const bool tensor_mode = pipeline_requires_tensor_input(impl_->preprocess_plan);
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  if (!tensor_mode && ingress_contracts.size() > 1U) {
    throw std::runtime_error(
        "Model::build(): multi-ingress image convenience is unsupported; use build(TensorList)");
  }
  const auto src_opts = input_appsrc_options_list(tensor_mode);
  TensorList dummy_inputs;
  dummy_inputs.reserve(src_opts.size());
  for (const auto& src_opt : src_opts) {
    dummy_inputs.push_back(make_dummy_tensor(src_opt));
  }
  internal::ModelPack pack = impl_->pack;
  if (!build_opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, build_opt.name_suffix);
  }
  const bool use_input_route_processor = plan_uses_bundled_fan_in(impl_->preprocess_plan);
  const bool externalize_preprocess = false;
  auto nodes = build_pipeline_nodes(*this, pack, impl_->options, impl_->preprocess_plan, build_opt,
                                    nullptr, false, externalize_preprocess);
  Graph p(route_options_from_model_route_options(build_opt, &impl_->options));
  add_nodes_to_graph(p, std::move(nodes));
  if (use_input_route_processor) {
    internal::ModelAccess::configure_session_input_route(p, *this, build_opt);
  }
  Run run = p.build(dummy_inputs, RunMode::Async, run_opt);
  const auto ingress_names = ingress_names_from_contracts(ingress_contracts);
  if (tensor_mode) {
    const auto src_opts2 = input_appsrc_options_list(true);
    const InputOptions src_opt =
        !src_opts2.empty() ? src_opts2.front() : pack.input_appsrc_options(true);
    return Runner(std::move(run), src_opt, ingress_names);
  }
  return Runner(std::move(run), ingress_names);
}

Model::Runner Model::build(const simaai::neat::TensorList& inputs, const Model::RouteOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  const Model::RouteOptions build_opt = route_options_for_model_runner(opt);
  if (inputs.empty()) {
    throw std::runtime_error("Model::build: empty tensor list");
  }
  internal::ModelPack pack = impl_->pack;
  if (!build_opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, build_opt.name_suffix);
  }
  const bool tensor_mode = pipeline_requires_tensor_input(impl_->preprocess_plan);
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  if (tensor_mode) {
    require_exact_ingress_count(inputs.size(), ingress_contracts, "Model::build", "tensor inputs");
  } else if (inputs.size() != 1U) {
    throw std::runtime_error(
        "Model::build(TensorList): image-mode convenience expects exactly one image tensor; use "
        "Graph for explicit multi-input pipelines.");
  } else if (ingress_contracts.size() > 1U) {
    throw std::runtime_error(
        "Model::build(TensorList): multi-ingress image convenience is unsupported; use "
        "Model::build(Sample)");
  }
  const bool use_input_route_processor = plan_uses_bundled_fan_in(impl_->preprocess_plan);
  const bool externalize_preprocess = false;
  std::optional<InputInfo> image_input_info;
  if (!tensor_mode) {
    image_input_info = input_info_from_tensor(inputs.front(), true);
    require_explicit_image_input_info(*image_input_info, "Model::build(TensorList)");
  }
  auto nodes = build_pipeline_nodes(*this, pack, impl_->options, impl_->preprocess_plan, build_opt,
                                    image_input_info ? &*image_input_info : nullptr, false,
                                    externalize_preprocess);
  Graph p(route_options_from_model_route_options(build_opt, &impl_->options));
  add_nodes_to_graph(p, std::move(nodes));
  if (use_input_route_processor) {
    internal::ModelAccess::configure_session_input_route(p, *this, build_opt);
  }
  Run run = p.build(inputs, RunMode::Async, run_opt);
  const auto ingress_names = ingress_names_from_contracts(ingress_contracts);
  if (!tensor_mode) {
    return Runner(std::move(run), ingress_names);
  }
  const auto src_opts = input_appsrc_options_list(true);
  const InputOptions src_opt =
      !src_opts.empty() ? src_opts.front() : pack.input_appsrc_options(true);
  if (runner_debug_enabled()) {
    std::fprintf(stderr,
                 "[model-runner] build(tensors) tensor_mode=1 media=%s format=%s w=%d h=%d d=%d\n",
                 resolve_input_media_type(src_opt).c_str(), src_opt.format.str().c_str(),
                 src_opt.width, src_opt.height, src_opt.depth);
  }
  return Runner(std::move(run), src_opt, ingress_names);
}

Model::Runner Model::build(const simaai::neat::Sample& inputs, const Model::RouteOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  const Model::RouteOptions build_opt = route_options_for_model_runner(opt);
  if (inputs.empty()) {
    throw std::runtime_error("Model::build: empty sample list");
  }
  internal::ModelPack pack = impl_->pack;
  if (!build_opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, build_opt.name_suffix);
  }
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  const auto ingress_names = ingress_names_from_contracts(ingress_contracts);
  const bool use_input_route_processor = plan_uses_bundled_fan_in(impl_->preprocess_plan);
  const bool externalize_preprocess = false;
  const bool tensor_mode = pipeline_requires_tensor_input(impl_->preprocess_plan);
  std::optional<InputInfo> image_input_info;
  if (!tensor_mode) {
    if (inputs.size() != 1U) {
      throw std::runtime_error(
          "Model::build(Sample): image-mode convenience expects exactly one image sample; "
          "use Graph for explicit multi-input pipelines.");
    }
    image_input_info = input_info_from_image_sample(inputs.front());
    require_explicit_image_input_info(*image_input_info, "Model::build(Sample)");
  }
  auto nodes = build_pipeline_nodes(*this, pack, impl_->options, impl_->preprocess_plan, build_opt,
                                    image_input_info ? &*image_input_info : nullptr, false,
                                    externalize_preprocess);
  Graph p(route_options_from_model_route_options(build_opt, &impl_->options));
  add_nodes_to_graph(p, std::move(nodes));
  if (use_input_route_processor) {
    internal::ModelAccess::configure_session_input_route(p, *this, build_opt);
  }
  Run run = p.build(inputs, RunMode::Async, run_opt);
  return Runner(std::move(run), ingress_names);
}

#if defined(SIMA_WITH_OPENCV)
Model::Runner Model::build(const std::vector<cv::Mat>& inputs, const Model::RouteOptions& opt,
                           const simaai::neat::RunOptions& run_opt) {
  const Model::RouteOptions build_opt = route_options_for_model_runner(opt);
  if (inputs.empty()) {
    throw std::runtime_error("Model::build: empty image list");
  }
  const auto ingress_contracts =
      normalized_ingress_contracts(impl_->preprocess_plan.session_route_plan);
  if (ingress_contracts.size() > 1U) {
    throw std::runtime_error(
        "Model::build(cv::Mat): multi-ingress convenience is unsupported; use build(TensorList)");
  }
  const bool tensor_mode = pipeline_requires_tensor_input(impl_->preprocess_plan);
  if (tensor_mode) {
    internal::ModelPack pack = impl_->pack;
    if (!build_opt.name_suffix.empty()) {
      pack = pack.clone_with_overrides(std::string{}, build_opt.name_suffix);
    }
    const InputOptions src_opt = pack.input_appsrc_options(true);
    TensorList tensors;
    tensors.reserve(inputs.size());
    for (const auto& input : inputs) {
      const cv::Mat prepared_input =
          maybe_resize_tensor_ingress_mat_for_pre_adapter(input, impl_->preprocess_plan, src_opt);
      tensors.emplace_back(
          simaai::neat::tensor_from_cv_mat(prepared_input, src_opt, "Model::build"));
    }
    return build(tensors, build_opt, run_opt);
  }
  internal::ModelPack pack = impl_->pack;
  if (!build_opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, build_opt.name_suffix);
  }
  InputInfo info = input_info_from_mat(inputs.front());
  if (!tensor_mode) {
    const std::string preproc_input_format =
        resolve_preproc_input_format(impl_->preprocess_plan, &info);
    if (!preproc_input_format.empty()) {
      info.format = preproc_input_format;
      info.format_source = InputInfo::FormatSource::Explicit;
    }
  }
  auto nodes = build_pipeline_nodes(*this, pack, impl_->options, impl_->preprocess_plan, build_opt,
                                    &info, false, false);
  Graph p(route_options_from_model_route_options(build_opt, &impl_->options));
  add_nodes_to_graph(p, std::move(nodes));
  Run run = p.build(inputs, RunMode::Async, run_opt);
  return Runner(std::move(run));
}
#endif

simaai::neat::TensorList Model::run(const std::vector<Tensor>& inputs, int timeout_ms) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::run: empty input list");
  }
  Runner runner = build(inputs);
  return runner.run(inputs, timeout_ms);
}

simaai::neat::Sample Model::run(const simaai::neat::Sample& inputs, int timeout_ms) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::run: empty sample list");
  }
  Runner runner = build(inputs);
  return runner.run(inputs, timeout_ms);
}

#if defined(SIMA_WITH_OPENCV)
simaai::neat::TensorList Model::run(const std::vector<cv::Mat>& inputs, int timeout_ms) {
  if (inputs.empty()) {
    throw std::runtime_error("Model::run: empty image list");
  }
  Runner runner = build(inputs);
  return runner.run(inputs, timeout_ms);
}
#endif

namespace internal {
Tensor remap_tensor_to_consumer_identity(Tensor tensor,
                                         const IngressConsumerTensorIdentity& identity) {
  if (identity.logical_index >= 0) {
    tensor.route.logical_index = identity.logical_index;
  }
  if (identity.physical_index >= 0) {
    tensor.route.physical_index = identity.physical_index;
  }
  if (identity.route_slot >= 0) {
    tensor.route.route_slot = identity.route_slot;
  }
  if (identity.memory_index >= 0) {
    tensor.route.memory_index = identity.memory_index;
  }
  return tensor;
}

Sample remap_sample_to_consumer_identity(Sample sample,
                                         const IngressConsumerTensorIdentity& identity) {
  if (sample.tensor.has_value()) {
    sample.tensor = remap_tensor_to_consumer_identity(std::move(*sample.tensor), identity);
  } else if (sample_has_tensor_list(sample) && sample.tensors.size() == 1U) {
    sample.tensors.front() =
        remap_tensor_to_consumer_identity(std::move(sample.tensors.front()), identity);
  }
  if (identity.logical_index >= 0) {
    sample.logical_output_index = identity.logical_index;
    sample.output_index = identity.logical_index;
  }
  if (identity.route_slot >= 0) {
    sample.route_slot = identity.route_slot;
  }
  if (identity.memory_index >= 0) {
    sample.memory_index = identity.memory_index;
  }
  return sample;
}

TensorList materialize_joined_ingress_tensors(
    const TensorList& tensors,
    const std::vector<IngressConsumerTensorIdentity>& consumer_identities,
    const std::string& segment_name, const char* where) {
  return materialize_joined_ingress_tensors_impl(tensors, consumer_identities, segment_name, where);
}

const ModelPack& ModelAccess::pack(const Model& model) {
  return model.impl_->pack;
}

const ModelPack& ModelAccess::pack_for_sync(const Model& model) {
  return model.impl_->pack_for_sync();
}

std::string ModelAccess::model_id(const Model& model) {
  return model.impl_->model_id;
}

std::string ModelAccess::source_path(const Model& model) {
  return model.impl_->source_path;
}

Model::Options ModelAccess::options(const Model& model) {
  return model.impl_->options;
}

Model ModelAccess::clone_with_options(const Model& model, const Model::Options& opt) {
  return Model(model.impl_->source_path, opt);
}

PostRouteStageKind ModelAccess::resolved_post_kind(const Model& model) {
  return model.impl_->preprocess_plan.session_route_plan.selected_post_kind;
}

PreprocessContractFlags ModelAccess::preprocess_contract_flags(const Model& model) {
  return resolve_preprocess_contract_flags(model.impl_->preprocess_plan);
}

pipeline_internal::sima::ModelManagedRouteFlags
ModelAccess::model_managed_route_flags(const Model& model) {
  return convert_model_managed_route_flags(
      model.impl_->preprocess_plan.session_route_plan.model_managed_route_flags);
}

bool ModelAccess::has_model_managed_stage(const Model& model, StageNodeKind kind) {
  switch (kind) {
  case StageNodeKind::Preproc:
  case StageNodeKind::Quant:
  case StageNodeKind::Tess:
  case StageNodeKind::QuantTess:
  case StageNodeKind::CastTess:
    return resolved_preprocess_graph_contains_stage(model.impl_->preprocess_plan, kind) ||
           route_contains_stage(model.impl_->preprocess_plan.session_route_plan, kind);
  case StageNodeKind::Detess:
  case StageNodeKind::DetessCast:
  case StageNodeKind::DetessDequant:
  case StageNodeKind::Dequant:
  case StageNodeKind::BoxDecode:
    return route_contains_stage(model.impl_->preprocess_plan.session_route_plan, kind);
  }
  return false;
}

void ModelAccess::require_model_managed_stage(const Model& model, StageNodeKind kind,
                                              const char* caller) {
  if (has_model_managed_stage(model, kind)) {
    return;
  }
  std::ostringstream oss;
  oss << (caller ? caller : "Model stage") << ": resolved model route does not contain "
      << stage_node_kind_name(kind) << ". "
      << "Route summary: " << route_stage_summary(model.impl_->preprocess_plan.session_route_plan)
      << ". " << stage_route_missing_hint(kind);
  throw std::runtime_error(oss.str());
}

PreprocOptions ModelAccess::build_preprocess_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::Preproc, "PreprocOptions(Model)");
  PreprocOptions opt = make_model_managed_preproc_options_base(model, sync);
  populate_model_managed_preproc_options(&opt, model.impl_->preprocess_plan, nullptr);
  return opt;
}

QuantOptions ModelAccess::build_quant_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::Quant, "QuantOptions(Model)");
  QuantOptions opt = make_model_managed_quant_options_base(model, sync);
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? model.impl_->pack_for_sync() : model.impl_->pack,
          internal::ExecutionStageKind::Quant, "quant"));
  return opt;
}

TessOptions ModelAccess::build_tess_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::Tess, "TessOptions(Model)");
  TessOptions opt = make_model_managed_tess_options_base(model, sync);
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? model.impl_->pack_for_sync() : model.impl_->pack,
          internal::ExecutionStageKind::Tess, "tess"));
  return opt;
}

QuantTessOptions ModelAccess::build_quanttess_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::QuantTess, "QuantTessOptions(Model)");
  QuantTessOptions opt = make_model_managed_quanttess_options_base(model, sync);
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? model.impl_->pack_for_sync() : model.impl_->pack,
          internal::ExecutionStageKind::QuantTess, "quanttess"));
  return opt;
}

CastTessOptions ModelAccess::build_casttess_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::CastTess, "CastTessOptions(Model)");
  CastTessOptions opt = make_model_managed_casttess_options_base(model, sync);
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_preadapter_contract(
          sync ? model.impl_->pack_for_sync() : model.impl_->pack,
          internal::ExecutionStageKind::CastTess, "casttess"));
  return opt;
}

DetessOptions ModelAccess::build_detess_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::Detess, "DetessOptions(Model)");
  const auto& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  DetessOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  opt.no_json_path = true;
  if (std::getenv("SIMA_KEEP_DETESS_CONFIG")) {
    opt.keep_config = true;
  }
  auto compiled =
      require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::Detess);
  compiled.runtime_contract.plugin_kind = "neatdetess";
  opt.compiled_contract = std::make_shared<const CompiledProcessCvuContract>(std::move(compiled));
  const auto plan = pack.execution_plan();
  if (!plan.infer.empty()) {
    opt.upstream_name = plan.infer.back().stage_name;
  }
  for (const auto& stage : plan.post) {
    if (stage.kind == internal::ExecutionStageKind::Detess) {
      opt.element_name = stage.stage_name;
      break;
    }
  }
  return opt;
}

DetessCastOptions ModelAccess::build_detesscast_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::DetessCast, "DetessCastOptions(Model)");
  const auto& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  DetessCastOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  opt.compiled_contract = std::make_shared<const CompiledProcessCvuContract>(
      require_model_managed_postprocess_contract(pack, internal::ExecutionStageKind::DetessCast));
  const auto plan = pack.execution_plan();
  if (!plan.infer.empty()) {
    opt.upstream_name = plan.infer.back().stage_name;
  }
  for (const auto& stage : plan.post) {
    if (stage.kind == internal::ExecutionStageKind::DetessCast) {
      opt.element_name = stage.stage_name;
      break;
    }
  }
  return opt;
}

DetessDequantOptions ModelAccess::build_detessdequant_stage_options(const Model& model, bool sync) {
  require_model_managed_stage(model, StageNodeKind::DetessDequant, "DetessDequantOptions(Model)");
  const auto& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  DetessDequantOptions opt;
  init_model_managed_processcvu_buffers(&opt, pack, sync);
  opt.compiled_contract =
      std::make_shared<const CompiledProcessCvuContract>(require_model_managed_postprocess_contract(
          pack, internal::ExecutionStageKind::DetessDequant));
  const auto plan = pack.execution_plan();
  if (!plan.infer.empty()) {
    opt.upstream_name = plan.infer.back().stage_name;
  }
  for (const auto& stage : plan.post) {
    if (stage.kind == internal::ExecutionStageKind::DetessDequant) {
      opt.element_name = stage.stage_name;
      break;
    }
  }
  return opt;
}

CompiledBoxDecodeContract ModelAccess::build_boxdecode_stage_contract(const Model& model,
                                                                      bool sync) {
  require_model_managed_stage(model, StageNodeKind::BoxDecode, "SimaBoxDecode(Model)");
  const auto& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  return require_model_managed_boxdecode_contract(pack);
}

void ModelAccess::configure_session_input_route(simaai::neat::Graph& session, const Model& model,
                                                const Model::RouteOptions& opt) {
  Model owned_model = ModelAccess::clone_with_options(model, ModelAccess::options(model));
  session.input_route_processor_ = std::make_shared<ModelIngressRouteProcessor>(
      std::move(owned_model), model.impl_->preprocess_plan, opt);
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_public_preprocess_nodes(const Model& model) {
  if (!model.impl_->preprocess_plan.session_route_plan.include_pre_stage) {
    return {};
  }
  const auto& pack = model.impl_->pack;
  const std::string pre_name = resolved_pre_stage_name(pack, model.impl_->preprocess_plan);
  return build_preprocess_nodes_impl(model, pack, model.impl_->preprocess_plan, nullptr, pre_name,
                                     "decoder", false);
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_public_inference_nodes(const Model& model) {
  const auto& pack = model.impl_->pack;
  const std::string pre_name = resolved_pre_stage_name(pack, model.impl_->preprocess_plan);
  const std::string upstream = model.impl_->preprocess_plan.session_route_plan.include_pre_stage
                                   ? (pre_name.empty() ? std::string("decoder") : pre_name)
                                   : std::string("decoder");
  return pack.infer_block(
      upstream, make_stage_lineage_binding(model, internal::ModelLineageStageRole::Infer));
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_public_postprocess_nodes(const Model& model) {
  const auto& pack = model.impl_->pack;
  return build_postprocess_nodes_impl(model, pack, model.impl_->options, false,
                                      model.impl_->preprocess_plan.session_route_plan);
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_public_route_nodes(const Model& model,
                                                                         Model::RouteOptions opt) {
  internal::ModelPack pack = model.impl_->pack;
  if (!opt.name_suffix.empty()) {
    pack = pack.clone_with_overrides(std::string{}, opt.name_suffix);
  }
  return build_pipeline_nodes(model, pack, model.impl_->options, model.impl_->preprocess_plan,
                              std::move(opt), nullptr, false, false);
}

std::vector<std::shared_ptr<Node>>
ModelAccess::build_public_stage_fragment_nodes(const Model& model, Model::Stage stage) {
  switch (stage) {
  case Model::Stage::Preprocess:
    return ModelAccess::build_public_preprocess_nodes(model);
  case Model::Stage::Inference:
    return ModelAccess::build_public_inference_nodes(model);
  case Model::Stage::Postprocess:
    return ModelAccess::build_public_postprocess_nodes(model);
  case Model::Stage::Full:
    return ModelAccess::build_public_route_nodes(model, Model::RouteOptions{});
  }
  return {};
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_preprocess_nodes(const Model& model,
                                                                       bool sync) {
  require_model_managed_stage(model, StageNodeKind::Preproc,
                              "Model::preprocess()/stages::Preproc(Model)");
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  return build_preprocess_nodes_impl(model, pack, model.impl_->preprocess_plan, nullptr,
                                     std::string{}, std::string{}, sync);
}

std::vector<std::shared_ptr<Node>>
ModelAccess::build_preprocess_nodes_for_input(const Model& model, const InputOptions& input,
                                              bool sync) {
  require_model_managed_stage(model, StageNodeKind::Preproc, "stages::Preproc(input, Model)");
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  InputInfo info;
  info.media_type = resolve_input_media_type(input);
  info.format = upper_copy(input.format);
  info.width = input.width;
  info.height = input.height;
  info.depth = input.depth;
  if (!info.format.empty()) {
    info.format_source = InputInfo::FormatSource::Explicit;
  }
  return build_preprocess_nodes_impl(model, pack, model.impl_->preprocess_plan, &info,
                                     std::string{}, std::string{}, sync);
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_infer_nodes(const Model& model, bool sync) {
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  std::string upstream = "decoder";
  if (model.impl_->preprocess_plan.session_route_plan.include_pre_stage) {
    const std::string pre = resolved_pre_stage_name(pack, model.impl_->preprocess_plan);
    upstream = pre.empty() ? std::string("decoder") : pre;
  }
  return pack.infer_block(
      upstream, make_stage_lineage_binding(model, internal::ModelLineageStageRole::Infer));
}

std::vector<std::shared_ptr<Node>> ModelAccess::build_postprocess_nodes(const Model& model,
                                                                        bool sync) {
  const ModelPack& pack = sync ? model.impl_->pack_for_sync() : model.impl_->pack;
  return build_postprocess_nodes_impl(model, pack, model.impl_->options, sync,
                                      model.impl_->preprocess_plan.session_route_plan);
}

Graph ModelAccess::build_stage_graph_fragment(const Model& model, Model::Stage stage) {
  Graph graph = graph_from_nodes(ModelAccess::build_public_stage_fragment_nodes(model, stage));
  const auto range_end =
      graph.linear_nodes_snapshot("ModelAccess::build_stage_graph_fragment").size();
  if (range_end == 0U) {
    return graph;
  }

  runtime::FragmentBoundaryHints hints;
  runtime::Provenance provenance;
  provenance.model_id = ModelAccess::model_id(model);
  provenance.model_source_path = ModelAccess::source_path(model);
  provenance.model_stage_role = public_stage_role_name(stage);
  provenance.model_options_json =
      model_options_json_for_graph_provenance(ModelAccess::options(model));
  graph.attach_fragment_boundary_hints_(0, range_end, std::move(hints), std::move(provenance));
  return graph;
}

Graph ModelAccess::build_graph_fragment(const Model& model, Model::RouteOptions opt,
                                        runtime::FragmentBoundaryHints* hints) {
  Graph graph(route_options_from_model_route_options(opt, &model.impl_->options));
  add_nodes_to_graph(graph, ModelAccess::build_public_route_nodes(model, opt));

  runtime::FragmentBoundaryHints fragment_hints;
  const bool tensor_mode = pipeline_requires_tensor_input(model.impl_->preprocess_plan);
  const bool bundled_fan_in = plan_uses_bundled_fan_in(model.impl_->preprocess_plan);
  fragment_hints.tensor_mode = tensor_mode;
  fragment_hints.bundled_fan_in = bundled_fan_in;
  fragment_hints.ingress_inputs = model.input_appsrc_options_list(tensor_mode);
  fragment_hints.ingress_endpoint_names =
      make_unique_endpoint_names(ingress_names_from_contracts(normalized_ingress_contracts(
                                     model.impl_->preprocess_plan.session_route_plan)),
                                 "ifm");
  if (fragment_hints.ingress_endpoint_names.empty()) {
    for (std::size_t i = 0; i < fragment_hints.ingress_inputs.size(); ++i) {
      fragment_hints.ingress_endpoint_names.push_back(
          !fragment_hints.ingress_inputs[i].buffer_name.empty()
              ? fragment_hints.ingress_inputs[i].buffer_name
              : ("ifm" + std::to_string(i)));
    }
    fragment_hints.ingress_endpoint_names =
        make_unique_endpoint_names(std::move(fragment_hints.ingress_endpoint_names), "ifm");
  }
  const auto& route_plan = model.impl_->preprocess_plan.session_route_plan;
  fragment_hints.egress_endpoint_names =
      model_route_exposes_individual_outputs(route_plan, opt)
          ? egress_names_from_contracts(route_plan)
          : aggregate_egress_name_for_model_route(ModelAccess::model_id(model), opt);
  if (bundled_fan_in) {
    for (auto& ingress : fragment_hints.ingress_inputs) {
      ingress.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                              "storage=(string)tensorbuffer";
    }
  }

  if (bundled_fan_in) {
    ModelAccess::configure_session_input_route(graph, model, opt);
    fragment_hints.input_route_processor = graph.input_route_processor_;
  }

  if (hints) {
    *hints = fragment_hints;
  }
  const auto range_end = graph.linear_nodes_snapshot("ModelAccess::build_graph_fragment").size();
  runtime::Provenance provenance;
  provenance.model_id = ModelAccess::model_id(model);
  provenance.model_source_path = ModelAccess::source_path(model);
  provenance.model_stage_role = "route";
  provenance.model_options_json =
      model_options_json_for_graph_provenance(ModelAccess::options(model));
  provenance.model_route.present = true;
  provenance.model_route.upstream_name = opt.upstream_name;
  provenance.model_route.name_suffix = opt.name_suffix;
  provenance.model_route.buffer_name = opt.buffer_name;
  provenance.model_route.processcvu_requested_run_target = opt.processcvu_requested_run_target;
  provenance.model_route.processcvu = opt.processcvu;
  provenance.model_route.processmla = opt.processmla;
  provenance.model_route.prepared_runner = opt.prepared_runner;
  provenance.model_route.async_queue_depth = opt.async_queue_depth;
  provenance.model_route.expose_all_outputs = opt.expose_all_outputs;
  graph.attach_fragment_boundary_hints_(0, range_end, std::move(fragment_hints),
                                        std::move(provenance));

  return graph;
}
} // namespace internal

} // namespace simaai::neat
