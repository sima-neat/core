#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The MLA-only model route on the Dataflow backend (H1): Input -> Model::inference() -> Output,
// which GStreamer builds as appsrc ! neatprocessmla ! appsink, is Feed -> soc::MlaOp -> Fetch.
// Here: what the route's one MLA stage is, read from the contracts Core compiles for the same
// nodes on GStreamer (the rendered stage manifest), and the three things the input's port does
// with it: load the model, check the input's binding against the MLA's input, add the operator
// to the Net. The pulled tensor is described here once, as GStreamer's pull describes the MLA
// stage's published output (InputStreamPull.cpp apply_tensor_set_descriptor,
// StageRecipePreparation.cpp build_publish_contract_from_manifest_stage_local).
//
// Only this file's .cpp names dataflow::soc, and only in a build that has it
// (SIMANEAT_WITH_DATAFLOW_SOC: the Cortex-A65 with the Runner's MLA package). Without it the
// compiler does not recognise the route and nothing here is called.

#include "builder/Node.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowRuntime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

#pragma GCC visibility push(hidden)

struct DataflowInputBinding;

// The tensor of a pulled message on a route whose Fetch does not carry the Feed's spec: the MLA
// stage's one published output, a view of `size_bytes` at `byte_offset` of the MLA's output buffer.
struct DataflowPulledTensor {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides_bytes;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  TensorRouteMeta route; ///< as GStreamer's tensor-set descriptor fills it
  std::optional<QuantSpec> quant;
  std::string format;    ///< the caps format token of the output ("EVXX_INT8")
  std::string stage_key; ///< Sample::stream_label and segment_name
};

// The one MLA stage of the route.
struct DataflowMlaStage {
  std::string model_path;        ///< the admitted MLA ELF
  std::uint64_t input_bytes = 0; ///< the MLA's one input buffer
  TensorDType input_dtype = TensorDType::UInt8;
  std::vector<std::int64_t> input_shape; ///< the logical input, for messages
  std::uint64_t output_bytes = 0;        ///< the MLA's one output buffer
  std::uint32_t depth = 2;               ///< jobs out at once (MlaOp::Options::depth)
  DataflowPulledTensor output;
  std::string text; ///< one line, for the [NEAT] trace and messages
};

// True when this build can run the route (dataflow::soc is in it).
bool dataflow_has_mla_route() noexcept;

// The stage of `route` (the nodes of GStreamer's segment: the Input, the model fragment, the
// Output), when the fragment is one MLA stage with one input buffer and one published output and
// this backend can carry it; null with `why` otherwise.
std::shared_ptr<const DataflowMlaStage>
plan_dataflow_mla_stage(const std::vector<std::shared_ptr<Node>>& route, std::string* why);

// The loaded model (soc::MlaModel) of `stage`, for one Run. Throws NeatError when the MLA refuses
// it.
std::shared_ptr<void> load_dataflow_mla_model(const DataflowMlaStage& stage, const char* where);

// Brings the input's `binding` to what the MLA takes: its shape fixed at its bound, device memory,
// the MLA's dtype and exactly the MLA's input bytes; and gives it the stage's pulled tensor.
// "" when it fits; otherwise why it does not.
std::string bind_dataflow_mla_input(const DataflowMlaStage& stage, DataflowInputBinding& binding);

// Adds the stage's operator to `net` for an input port of `input` (the Feed's spec). Returns the
// operator's index (its input 0 takes the Feed's link, its output 0 feeds the Fetch) and the spec
// of its output port in `output`.
std::uint32_t add_dataflow_mla_op(pipeline_internal::df::Net& net, const DataflowMlaStage& stage,
                                  const std::shared_ptr<void>& model,
                                  const pipeline_internal::df::PortSpec& input,
                                  pipeline_internal::df::PortSpec& output);

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
