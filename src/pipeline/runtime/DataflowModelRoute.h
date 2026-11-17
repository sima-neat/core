#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// A model route on the Dataflow backend (H1): one Input, the stages of a Model between it and one
// Output. GStreamer builds it as appsrc ! [neatkernelrunner !] neatprocessmla [! neatkernelrunner]
// ! appsink; here it is Feed -> [soc::KernelOp ->] soc::MlaOp [-> soc::KernelOp] -> Fetch, one
// operator per stage:
//   - Model::inference() of an MLA-only model: the MLA stage alone;
//   - Model::preprocess(), inference(), postprocess() of a model package whose pre and post stages
//     are canonical transforms (quanttess, detessdequant, ...) placed on the A65
//     (GraphOptions::processcvu.pre_run_target / post_run_target, ProcessCvuRunTargetPolicy.cpp).
// Here: what the route's stages are, read from what Core compiles and lowers for the same nodes on
// GStreamer (the rendered stage manifest and its checked kernel recipes), and what the input's
// port does with them: load the model and make the kernels, check the input's binding against
// the first stage's input, add the operators to the Net. The pulled tensor is described here
// once, as GStreamer's pull describes the last stage's published output (InputStreamPull.cpp
// apply_tensor_set_descriptor, StageRecipePreparation.cpp
// build_publish_contract_from_manifest_stage_local, finalize_processcvu_publication).
//
// Only this file's .cpp names dataflow::soc, and only in a build that has it
// (SIMANEAT_WITH_DATAFLOW_SOC: the Cortex-A65 with the Runner's MLA package;
// SIMANEAT_WITH_DATAFLOW_SOC_KERNELS: with the Runner's Executor and the Kernels package too).
// Without it the compiler does not recognise the route and nothing here is called.

#include "builder/Node.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <simaai/scheduler/client/parameters.h>
#include <simaai/scheduler/client/tensor.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

struct DataflowInputBinding;

// The tensor of a pulled message on a route whose Fetch does not carry the Feed's spec: the last
// stage's one published output, a view of `size_bytes` at `byte_offset` of that stage's output
// buffer.
struct DataflowPulledTensor {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  TensorRouteMeta route; ///< as GStreamer's tensor-set descriptor fills it
  std::optional<QuantSpec> quant;
  std::string format;    ///< the caps format token of the output ("EVXX_INT8", "EVXX_FLOAT32")
  std::string stage_key; ///< the stage's name
  /// Sample::stream_label and segment_name: the stage's name for an MLA stage, the stage's
  /// published output name for a kernel stage (TensorStagePublication::output_name).
  std::string sample_label;
};

// One stage of the route: a model on the MLA, or one kernel of the Kernels package on the A65.
struct DataflowModelStage {
  enum class Kind { Mla, Kernel };
  Kind kind = Kind::Mla;
  std::string key;                ///< the stage's name (messages, the operator's name)
  std::uint64_t input_bytes = 0;  ///< its one input buffer
  std::uint64_t output_bytes = 0; ///< its one output buffer
  // Mla: the admitted MLA ELF, and how many jobs are out at once (MlaOp::Options::depth).
  std::string model_path;
  std::uint32_t depth = 2;
  // Kernel: the request Core's lowering gave the Kernels package for this stage (the recipe's,
  // KernelStageRecipe::Request): the operation's name, its two tensors, its parameters.
  std::string operation;
  ::simaai::scheduler::client::TensorSpec input, output;
  std::vector<::simaai::scheduler::client::Parameter> parameters;
};

// The route's stages in execution order, and its two ends.
struct DataflowModelRoute {
  std::vector<DataflowModelStage> stages;
  std::uint64_t input_bytes = 0; ///< the first stage's input buffer
  TensorDType input_dtype = TensorDType::UInt8;
  std::vector<std::int64_t> input_shape; ///< the first stage's logical input
  /// The first stage is an A65 kernel: it reads its input through the CPU's address, so a pushed
  /// device tensor is wrapped with one, and a tensor that differs from `input_shape` by leading
  /// unit axes only is pushed as `input_shape` (GStreamer takes both).
  bool kernel_input = false;
  DataflowPulledTensor output;
  std::string text; ///< one line, for the [NEAT] trace and messages
};

// True when this build can run a model route at all (dataflow::soc is in it).
bool dataflow_has_model_route() noexcept;

// The stages of `route` (the nodes of GStreamer's segment: the Input, the model's stage nodes, the
// Output), compiled with `options` as GStreamer's build compiles them, when every stage is one
// this backend can carry; null with `why` otherwise.
std::shared_ptr<const DataflowModelRoute>
plan_dataflow_model_route(const std::vector<std::shared_ptr<Node>>& route,
                          const GraphOptions& options, std::string* why);

// What one Run's operators are made with: the loaded model (soc::MlaModel) of an MLA stage and
// the checked kernel (soc::Kernel) of a kernel stage, one per stage. Throws NeatError when the
// MLA refuses the model or the Kernels package the request.
std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute& route, const char* where);

// Brings the input's `binding` to what the first stage takes: its shape fixed at its bound,
// device memory, the stage's dtype and exactly its input bytes; and gives it the route's pulled
// tensor. "" when it fits; otherwise why it does not.
std::string bind_dataflow_model_input(const DataflowModelRoute& route,
                                      DataflowInputBinding& binding);

// `tensor` as the route's first stage takes it: `tensor` itself, or, on a route whose first stage
// is a kernel, a view of the same storage with the stage's shape when the tensor's differs from it
// by leading unit axes only (a dense, contiguous tensor); `view` holds it.
const Tensor& dataflow_model_input_view(const DataflowModelRoute& route, const Tensor& tensor,
                                        Tensor& view);

// Adds the route's operators to `net` for an input port of `input` (the Feed's spec), one per
// stage. Returns their indices in stage order (each has one input and one output) and the spec of
// the last one's output port in `output`. The caller connects them.
std::vector<std::uint32_t> add_dataflow_model_ops(pipeline_internal::df::Net& net,
                                                  const DataflowModelRoute& route,
                                                  const std::shared_ptr<void>& loaded,
                                                  const pipeline_internal::df::PortSpec& input,
                                                  pipeline_internal::df::PortSpec& output);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
