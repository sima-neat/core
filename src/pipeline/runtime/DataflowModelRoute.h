#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// A model route on the Dataflow backend: one Input, the stages of a Model between it and one
// Output. GStreamer builds it as appsrc ! [neatkernelrunner !] neatprocessmla [! neatkernelrunner]
// ! appsink; here it is Feed -> [soc::KernelOp ->] soc::MlaOp [-> soc::KernelOp] -> Fetch, one
// operator per stage:
//   - Model::inference() of an MLA-only model: the MLA stage alone. A model writes one output
//     buffer. The tensors it publishes in that buffer may be several, each at its byte offset (the
//     heads of a detection model): when the MLA stage is the route's last, a pull returns them
//     as GStreamer's does, a TensorList of views of the one pulled message. A model of several
//     output buffers, and a stage other than BoxDecode behind an MLA stage of several tensors,
//     are not carried: the plan says so and the build fails;
//   - Model::preprocess(), inference(), postprocess() of a model package whose pre and post stages
//     are canonical transforms (quanttess, detessdequant, ...), each on the A65 or on the EV74 as
//     the Graph places it (GraphOptions::processcvu.pre_run_target / post_run_target,
//     ProcessCvuRunTargetPolicy.cpp). A stage on the EV74 is taken only where the EV74 runs the
//     Kernels firmware; on a board with another firmware the route is not carried, and the plan
//     says so as the board's fact.
//   - Model::preprocess() of a model whose pre stage is the image preproc (the Kernels package's
//     "preproc", EV74 only: resize, colour, normalize, quantize and tessellate an RGB or BGR
//     frame, or an NV12 one), alone or in front of the stages above. GStreamer runs it
//     through a frame adapter that gives the kernel each frame's geometry; here the request for
//     a frame size is made from the same checked policy (PreprocFrameAdapter::
//     checked_fixed_frame), in one of two ways:
//       - the Input declares the frame's size (InputOptions::shape with every axis fixed, or
//         width and height): one kernel at that size, the port the frame. A frame of another
//         size is refused at the push;
//       - it does not: the port is bounded by the Input's capacity as GStreamer resolves it
//         (InputOptions::max_width / max_height, else the seed's size; the stage's capacity
//         envelope when the build says nothing, and never above it), the
//         operator keeps a kernel per frame size (soc::KernelOp::Options::Variants) and every
//         frame is run by its own (DataflowFrames, below). An RGB or BGR frame in EV74 memory is
//         wrapped whatever its size (the runtime takes a wrap shorter than a bounded port's
//         bound). A frame above the bound is refused at the push, with GStreamer's capacity
//         error below the envelope.
//     An NV12 frame is one port of bytes, the two planes views of it at fixed offsets
//     (soc::KernelOp::Options::Packing): a frame of the port's size, its chroma plane behind its
//     luma plane, is wrapped; a smaller one is copied to those offsets. I420 and the other planar
//     formats, an image frontend and a batched (ROI) stage are not carried (the plan says so);
//     a frame with ROI metadata is refused at the push, and an upstream stage's preprocess
//     metadata without an ROI is not read (as GStreamer's stage does not read it on a frame that
//     is not the build's seed).
//   - Model::postprocess() of a detection model (BoxDecode) behind the MLA stage, at the route's
//     end: the Kernels package's "boxdecode" on the A65 as one kernel operator whose inputs are
//     the heads, views at their byte offsets of the MLA's one output buffer (one port, no join),
//     and whose outputs (boxes, scores, classes, count) are views of one small buffer; behind it
//     a Core operator on the Worker writes the BBOX wire from them with the stage's own
//     serializer (BoxDecodePublication.cpp), which GStreamer runs as the stage's host task. The
//     kernel is made for the frames the route's image preproc is planned for
//     (boxdecode_request_for_frames: what GStreamer's per-frame resolver selects for such a
//     frame): once when the frame size is declared; otherwise the operator runs each frame size
//     with the Run's kernel's "image_size" variant (soc::Kernel::with), selected by the stamp's
//     user word (DataflowFrames). Carried: YOLOv8-style detection (profile yolo_dfl_v1) with
//     C-blocked heads. The other decode families, pose, segmentation and oriented boxes, tiled
//     heads (detess members), and a decode that takes each frame's size without the image
//     preproc in front are not carried (the plan says so).
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
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/DataflowRuntime.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/runtime/DataflowRoutePlan.h"

#include <simaai/scheduler/client/host_task.h>
#include <simaai/scheduler/client/parameters.h>
#include <simaai/scheduler/client/tensor.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

struct DataflowInputBinding;

// The frames of a route whose image preproc takes a size that changes between pushes: Core's
// side table from the push to the stages and to the pull. The push states a frame's geometry in
// its stamp's user word (dataflow_frame_word), which the runtime carries through every operator
// untouched; the table, keyed by that word, holds what Core derives from the geometry: the
// preproc and BoxDecode kernels for it and the preprocess facts the pull publishes. Defined in
// DataflowModelRoute.cpp.
struct DataflowFrames;

// A tensor of a pulled message on a route whose Fetch does not carry the Feed's spec: a published
// output of the last stage, a view of `size_bytes` at `byte_offset` of that stage's one output
// buffer. The route's DataflowModelRoute::output is the stage's first published tensor; a stage
// that publishes several in its buffer (an MLA model with several heads) lists the others in
// `more`.
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
  /// pulled tensor): the declared frame size's, or, where the frames change size, the port's
  /// bound's (the pull publishes each frame's own then: `frames`). Unset: no such stage.
  std::optional<PreprocessRuntimeMeta> preprocess;
  /// A detection stage's wire (BoxDecode): Semantic::detection of the pulled tensor, beside the
  /// legacy TessSpec that carries the same `format`. Unset: no such stage.
  std::optional<DetectionSpec> detection;
  /// `tess`, `preprocess` or `detection` is set: the pull calls apply_dataflow_frame_semantics.
  bool frame_semantics = false;
  /// The route's frames have a size that changes between pushes: `preprocess` above is the
  /// port's bound's, and the pull publishes each frame's own from this table by the pulled
  /// message's stamp. The route owns it. Null: one frame size.
  const DataflowFrames* frames = nullptr;
  /// The stage's further published tensors, in its logical order, each one a view at its own
  /// `byte_offset` of the same output buffer: the pull returns this tensor and then these, all
  /// over the one pulled message (one storage, which every one of them keeps alive). Without a
  /// `format`, as GStreamer's generic tensor-set caps leave them: no TessSpec on the tensors, no
  /// tag on the Sample. Empty: the stage publishes one tensor. (`more` of a listed tensor is not
  /// read.)
  std::vector<DataflowPulledTensor> more;
  /// `frame_semantics`, or `more` is not empty: the pull builds its tensors out of line
  /// (fill_dataflow_stage_tensors). The one flag a pull reads on its way to the one dense tensor
  /// of every other route; set once, when the route is planned.
  bool beyond_leaf = false;
  std::string stage_key; ///< the stage's name
  /// Sample::stream_label and segment_name: the stage's name for an MLA stage, the stage's
  /// published output name for a kernel stage (TensorStagePublication::output_name).
  std::string sample_label;
};

// One stage of the route, one operator: a model on the MLA, one kernel of the Kernels package on
// the A65 or on the EV74, or Core's BoxDecode wire serializer behind the BoxDecode kernel.
struct DataflowModelStage {
  // A tensor of a side whose tensors are views of the port's one buffer
  // (soc::KernelOp::Options::Packing): the tensor, and where it lies in the port's bytes.
  struct View {
    ::simaai::scheduler::client::TensorSpec tensor;
    std::size_t offset = 0, bytes = 0;
  };
  // A model on the MLA: the admitted MLA ELF, and how many jobs are out at once
  // (soc::MlaOp::Options::depth).
  struct Mla {
    std::string model_path;
    std::uint32_t depth = 2;
  };
  // A kernel of the Kernels package: where it runs (the checked recipe's target), and the request
  // Core's lowering gave the Kernels package for this stage (KernelStageRecipe::Request): the
  // operation's name, its tensors, its parameters. A packed side lists every tensor of the side
  // in the kernel's order, each a view of the port's `input_bytes` or `output_bytes`; `input` or
  // `output` is then not read. An empty list: the side is its one tensor.
  struct Kernel {
    // Which kernel runs a message (soc::KernelOp::Options::Variants): the stage's one kernel
    // (None); or, on a route whose frames change size, the image stage's or the BoxDecode kernel
    // of the frame's geometry (DataflowFrames), found by the message's extent (RGB or BGR frames
    // on the bounded port) or by its stamp's user word where the port's bytes do not tell the
    // geometry (an NV12 frame's byte port, the MLA's buffer that BoxDecode reads).
    enum class Variants : std::uint8_t { None, ImageByExtent, ImageByWord, DecodeByWord };
    bool ev74 = false;
    std::string operation;
    ::simaai::scheduler::client::TensorSpec input, output;
    std::vector<::simaai::scheduler::client::Parameter> parameters;
    std::vector<View> packed_inputs, packed_outputs;
    Variants variants = Variants::None;
  };
  // Core's BoxDecode wire serializer behind the BoxDecode kernel (KernelStageRecipe::
  // BoxDecodeProgram): `inputs` are the kernel's outputs in the stage's input buffer (offset and
  // bytes), from which `write` writes the `output_bytes` of the wire.
  struct Wire {
    std::vector<View> inputs;
    std::shared_ptr<const void> program;
    ::simaai::scheduler::client::HostStatus (*write)(
        const void* program, std::span<const ::simaai::scheduler::client::ConstBytes> inputs,
        ::simaai::scheduler::client::MutableBytes output) = nullptr;
  };
  std::string key;                ///< the stage's name (messages, the operator's name)
  std::uint64_t input_bytes = 0;  ///< its one input buffer
  std::uint64_t output_bytes = 0; ///< its one output buffer
  std::variant<Mla, Kernel, Wire> kind;
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
  /// `frame_input`, and the build declares no frame size: the frames' table. `frame_shape` is
  /// then the port's bound ([1, max height, max width, channels]; an NV12 route: the byte port,
  /// [bytes]) and `input_shape` is EMPTY, so that every push of one tensor leaves the device
  /// push's fast path where it leaves it for a tensor without the stage's batch axis
  /// (dataflow_device::push), and takes the frames' own push. Null: one frame size.
  std::shared_ptr<DataflowFrames> frames;
  std::vector<std::int64_t> frame_shape;
  /// `frames`, and the frames are NV12: the port is one buffer of bytes with the planes at
  /// fixed offsets, and a pushed frame is brought to it by dataflow_planar_frame.
  bool planar_input = false;
  DataflowPulledTensor output;
  std::string text; ///< one line, for the [NEAT] trace and messages
};

// True when this build can run a model route at all (dataflow::soc is in it).
bool dataflow_has_model_route() noexcept;

// The stages of `route` when every stage is one this backend can carry; null with `why`
// otherwise: plan_dataflow_route_stages of its manifest (dataflow_model_manifest,
// DataflowRoutePlan.h), with the board's EV74 asked for a route with a kernel stage on the EV74
// (soc::ev74_runs_kernels_firmware).
std::shared_ptr<const DataflowModelRoute>
plan_dataflow_model_route(const std::vector<std::shared_ptr<Node>>& route,
                          const GraphOptions& options, const InputOptions& declared,
                          std::span<const Tensor> seed, pipeline_internal::DataflowNotCarried* why);

// What one Run's operators are made with: the loaded model (soc::MlaModel) of an MLA stage and
// the checked kernel (soc::Kernel) of a kernel stage, one per such stage. Throws NeatError when the
// MLA refuses the model or the Kernels package the request.
std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute& route, const char* where);

// Brings the input's `binding` to what the first stage takes: its shape fixed at its bound (at
// the route's frame, in front of an image preproc), device memory, the stage's dtype and exactly
// its input bytes; and gives it the route's pulled tensor. "" when it fits; otherwise why it does
// not.
std::string bind_dataflow_model_input(const DataflowModelRoute& route,
                                      DataflowInputBinding& binding);

// What the route's first stage takes, as an Input's declaration that binds the input without a
// seed: an image of the route's frame format bounded by its port's bound, or, in front of a
// kernel stage (which takes a pushed tensor with leading unit axes as its own), that stage's
// tensor (its shape and dtype), in `declared`'s memory. `declared` itself otherwise: an image
// stage planned for a declared frame size, an MLA stage first (a pushed tensor binds it as it is).
InputOptions dataflow_model_input_contract(const DataflowModelRoute& route,
                                           const InputOptions& declared);

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

// The pulled tensor's Semantic behind an image preproc stage or a BoxDecode stage
// (`pulled.frame_semantics`): the tiled output's storage description in Semantic::tess, the
// frame's facts in Semantic::preprocess, the wire's format in Semantic::detection.
void apply_dataflow_frame_semantics(const DataflowPulledTensor& pulled, Tensor& tensor);

// The tensors of a pulled message on a route whose output is `pulled.beyond_leaf`, appended to
// `out`: the stage's first published tensor and then each of `pulled.more`, in the stage's
// order, every one a view at its own byte offset of the same `storage` (the output buffer of
// the pulled `message`, from its first byte: a model route's last stage writes a slot of its
// own pool, so its slot's DMA-BUF, or with OutputMemory::Owned one CPU copy of it), with its
// frame semantics: the route's, and on a route whose frames change size (`pulled.frames`) the
// preprocess facts of the frame the message's stamp names. The tensors share `storage`, and so
// the message: the slot returns to its pool when the last of them, or of their copies, is
// dropped. `envelope` is the list's one envelope (null in a Sample). The list is reserved once,
// for all of them.
void fill_dataflow_stage_tensors(const DataflowPulledTensor& pulled,
                                 const std::shared_ptr<TensorBuffer>& storage,
                                 const pipeline_internal::df::Message& message,
                                 const std::shared_ptr<const TensorEnvelope>& envelope,
                                 TensorList& out);

// Before a push into a route whose first stage is the image preproc at a declared frame size
// (`route.frame_input` without `route.frames`) of a tensor that is not the validated frame:
// throws NeatError (misconfig.input_shape) for a frame of another size than the declared one
// (with what to do instead), and the not-carried error for one that carries ROI metadata; returns
// for anything else
// (the binding's own check says what is wrong with it), and on a route whose frames change size
// (dataflow_frame_word checks those).
void refuse_dataflow_frame(const DataflowModelRoute& route, const Tensor& tensor,
                           const char* where);

// A route whose frames change size (`route.frames`), before a push of `tensor`: the stamp's
// user word that states the frame's geometry to the route's stages and to the pull. Throws
// NeatError (misconfig.input_shape), each with what to do instead, for a frame the route cannot
// run: one above the port's bound, of another format or layout than the route's frames (an image
// spec of another format: GStreamer's "image format mismatch"), or one whose kernel request the
// Kernels package refuses; and the not-carried error for a frame with ROI metadata. An upstream
// stage's preprocess metadata without an ROI is not read. So that nothing the Worker is later
// asked for can fail, the first frame of a size makes that size's kernels here, on the pushing
// thread (cold: the image stage's kernel is checked and made; the BoxDecode kernel is a variant of
// the Run's); a frame of a size the table holds costs a compare, and one of the port's bound's size
// nothing more at its first push (the table starts with it). Never 0.
std::uint64_t dataflow_frame_word(const DataflowModelRoute& route, const Tensor& tensor,
                                  const char* where);

// A route of NV12 frames (`route.planar_input`): whether `tensor` (a frame dataflow_frame_word
// passed) is the port's bytes as it is: its size the port's, the luma plane at the tensor's
// first byte, the chroma plane behind the luma's rows, rows not padded. Then the push backs the
// slot with the tensor's own DMA-BUF; otherwise it copies the planes to the port's offsets
// (copy_dataflow_planar_frame).
bool dataflow_planar_frame_wraps(const DataflowModelRoute& route, const Tensor& tensor);

// The planes of the NV12 `tensor`, rows tight, into `slot` (a slot of the route's byte port) at
// the port's plane offsets. False when the tensor's planes cannot be read.
bool copy_dataflow_planar_frame(const DataflowModelRoute& route, const Tensor& tensor,
                                std::span<std::byte> slot);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
