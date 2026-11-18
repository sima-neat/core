#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// A model route on the Dataflow backend (H1): one Input, the stages of a Model between it and one
// Output. GStreamer builds it as appsrc ! [neatkernelrunner !] neatprocessmla [! neatkernelrunner]
// ! appsink; here it is Feed -> [soc::KernelOp ->] soc::MlaOp [-> soc::KernelOp] -> Fetch, one
// operator per stage:
//   - Model::inference() of an MLA-only model: the MLA stage alone. A model writes one output
//     buffer. The tensors it publishes in that buffer may be several, each at its byte offset (the
//     heads of a detection model): then the MLA stage is the route's last, and a pull returns
//     them as GStreamer's does, a TensorList of views of the one pulled message. A model of
//     several output buffers, and a stage behind an MLA stage of several tensors, are declined
//     at the plan and stay on GStreamer;
//   - Model::preprocess(), inference(), postprocess() of a model package whose pre and post stages
//     are canonical transforms (quanttess, detessdequant, ...), each on the A65 or on the EV74 as
//     the Graph places it (GraphOptions::processcvu.pre_run_target / post_run_target,
//     ProcessCvuRunTargetPolicy.cpp). A stage on the EV74 is taken only where the EV74 runs the
//     Kernels firmware; on a board with the stock firmware the route is declined at the plan.
//   - Model::preprocess() of a model whose pre stage is the image preproc (the Kernels package's
//     "preproc", EV74 only: resize, colour, normalize, quantize and tessellate a packed RGB or BGR
//     frame), alone or in front of the stages above. GStreamer runs it through a frame adapter
//     that gives the kernel each frame's geometry; here it is one kernel at one frame size, the
//     size the input has at the build (its seed, or its declaration), made from the same checked
//     policy (PreprocFrameAdapter::checked_fixed_frame). A frame of another size is refused at
//     the push; a build that does not say the size, an NV12 or other planar input, an image
//     frontend and a batched (ROI) stage are declined at the plan and stay on GStreamer.
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
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <simaai/scheduler/client/parameters.h>
#include <simaai/scheduler/client/tensor.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

struct DataflowInputBinding;

// A tensor of a pulled message on a route whose Fetch does not carry the Feed's spec: a published
// output of the last stage, a view of `size_bytes` at `byte_offset` of that stage's one output
// buffer. The route's DataflowModelRoute::output is the stage's first published tensor; a stage
// that publishes several in its buffer (an MLA model with several heads) lists the others in
// `packed`.
struct DataflowPulledTensor {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  TensorRouteMeta route; ///< as GStreamer's tensor-set descriptor fills it
  std::optional<QuantSpec> quant;
  std::string format; ///< the caps format token of the output ("EVXX_INT8", "EVXX_FLOAT32")
  /// A tiled output published as its packed storage (the image preproc's tensor in front of the
  /// MLA): the checked storage description GStreamer's publication gives it (the tile shape, the
  /// axes, the storage bytes), with `format` empty and no strides. Unset: a dense output, whose
  /// TessSpec is `format` alone.
  std::optional<TessSpec> tess;
  /// What the route's image preproc publishes with every frame (Semantic::preprocess of the
  /// pulled tensor), a constant of the frame size the route was built for. Unset: no such stage.
  std::optional<PreprocessRuntimeMeta> preprocess;
  /// `tess` or `preprocess` is set: the pull calls apply_dataflow_frame_semantics.
  bool frame_semantics = false;
  /// The stage's further published tensors, in its logical order, each one a view at its own
  /// `byte_offset` of the same output buffer: the pull returns this tensor and then these, all
  /// over the one pulled message (one storage, which every one of them keeps alive). Without a
  /// `format`, as GStreamer's generic tensor-set caps leave them: no TessSpec on the tensors, no
  /// tag on the Sample. Empty: the stage publishes one tensor. (`packed` of a listed tensor is
  /// not read.)
  std::vector<DataflowPulledTensor> packed;
  /// `frame_semantics`, or `packed` is not empty: the pull builds its tensors out of line
  /// (fill_dataflow_stage_tensors). The one flag a pull reads on its way to the one dense tensor
  /// of every other route; set once, when the route is planned.
  bool beyond_leaf = false;
  std::string stage_key; ///< the stage's name
  /// Sample::stream_label and segment_name: the stage's name for an MLA stage, the stage's
  /// published output name for a kernel stage (TensorStagePublication::output_name).
  std::string sample_label;
};

// One stage of the route: a model on the MLA, or one kernel of the Kernels package on the A65 or
// on the EV74.
struct DataflowModelStage {
  enum class Kind { Mla, Kernel };
  Kind kind = Kind::Mla;
  std::string key;                ///< the stage's name (messages, the operator's name)
  std::uint64_t input_bytes = 0;  ///< its one input buffer
  std::uint64_t output_bytes = 0; ///< its one output buffer
  // Mla: the admitted MLA ELF, and how many jobs are out at once (MlaOp::Options::depth).
  std::string model_path;
  std::uint32_t depth = 2;
  // Kernel: where it runs (the checked recipe's target), and the request Core's lowering gave the
  // Kernels package for this stage (the recipe's, KernelStageRecipe::Request): the operation's
  // name, its two tensors, its parameters.
  bool ev74 = false;
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
  /// The first stage is a kernel: a tensor that differs from `input_shape` by leading unit axes
  /// only is pushed as `input_shape` (GStreamer takes both). On the A65 (`ev74_input` false) it
  /// reads its input through the CPU's address, so a pushed device tensor is wrapped with one.
  bool kernel_input = false;
  /// The first stage is a kernel on the EV74: it reads by descriptor and reaches linux,cma only.
  /// The input's pool is Ev74 memory whatever was declared; a pushed TensorMemory::EV74 tensor is
  /// wrapped by its descriptor alone, and a TensorMemory::MLA tensor (the DMS heap) is copied into
  /// a slot, never handed on under the port's kind.
  bool ev74_input = false;
  /// The first stage is the image preproc, planned for one frame size: `input_shape` is that
  /// frame as the kernel takes it ([1, height, width, channels]), which the input's port is
  /// fixed at whatever its declaration's bounds allow; a pushed frame of another size, or with
  /// ROI metadata, is refused.
  bool frame_input = false;
  DataflowPulledTensor output;
  std::string text; ///< one line, for the [NEAT] trace and messages
};

// True when this build can run a model route at all (dataflow::soc is in it).
bool dataflow_has_model_route() noexcept;

// The stages of `route` (the nodes of GStreamer's segment: the Input, the model's stage nodes, the
// Output), compiled with `options` as GStreamer's build compiles them, when every stage is one
// this backend can carry; null with `why` otherwise. `declared` (the Input's options) and `seed`
// (the build's seed, or empty) are what says the frame size of an image preproc stage.
std::shared_ptr<const DataflowModelRoute>
plan_dataflow_model_route(const std::vector<std::shared_ptr<Node>>& route,
                          const GraphOptions& options, const InputOptions& declared,
                          std::span<const Tensor> seed, std::string* why);

// What one Run's operators are made with: the loaded model (soc::MlaModel) of an MLA stage and
// the checked kernel (soc::Kernel) of a kernel stage, one per stage. Throws NeatError when the
// MLA refuses the model or the Kernels package the request.
std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute& route, const char* where);

// Brings the input's `binding` to what the first stage takes: its shape fixed at its bound (at
// the route's frame, in front of an image preproc), device memory, the stage's dtype and exactly
// its input bytes; and gives it the route's pulled tensor. "" when it fits; otherwise why it does
// not.
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

// The pulled tensor's Semantic behind an image preproc stage (`pulled.frame_semantics`): the tiled
// output's storage description in Semantic::tess and the frame's facts in Semantic::preprocess.
void apply_dataflow_frame_semantics(const DataflowPulledTensor& pulled, Tensor& tensor);

// The tensors of a pulled message on a route whose output is `pulled.beyond_leaf`, appended to
// `out`: the stage's first published tensor and then each of `pulled.packed`, in the stage's
// order, every one a view at its own byte offset of the same `storage` (the pulled message's
// output buffer from `byte_offset`: its slot's DMA-BUF, or with OutputMemory::Owned one CPU copy
// of it), with its frame semantics. The tensors share `storage`, and so the message: the slot
// returns to its pool when the last of them, or of their copies, is dropped. `envelope` is the
// list's one envelope (null in a Sample). The list is reserved once, for all of them.
void fill_dataflow_stage_tensors(const DataflowPulledTensor& pulled,
                                 const std::shared_ptr<TensorBuffer>& storage,
                                 std::int64_t byte_offset,
                                 const std::shared_ptr<const TensorEnvelope>& envelope,
                                 TensorList& out);

// Before a push into a route whose first stage is the image preproc (`route.frame_input`) of a
// tensor that is not the validated frame: throws NeatError (misconfig.input_shape) for a frame of
// another size than the route was built for, and for one that carries ROI metadata, each with what
// to do instead; returns for anything else (the binding's own check says what is wrong with it).
void refuse_dataflow_frame(const DataflowModelRoute& route, const Tensor& tensor,
                           const char* where);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
