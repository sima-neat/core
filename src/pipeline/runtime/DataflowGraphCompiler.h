#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Dataflow backend's compiler (H0 plan 3.6): which runtime a Graph::build() takes, whether
// the graph qualifies for Dataflow, and the Net a qualifying Input -> Output graph becomes.
// Graph::build's four entries call DataflowGraphAccess::build before gst_init_once().

#include "pipeline/GraphOptions.h"
#include "pipeline/Run.h"
#include "pipeline/TensorCore.h"
#include "pipeline/runtime/DataflowRun.h"

#if defined(SIMANEAT_WITH_DATAFLOW)
#include "pipeline/internal/DataflowRuntime.h"
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {
class Graph;
} // namespace simaai::neat

namespace simaai::neat::runtime {

struct RunCore;
struct DataflowGraphView; // DataflowGraphCompiler.cpp: what the compiler reads of a Graph

// Friend of Graph (Graph.h): reads its composition view. Hidden explicitly, since Graph.h's forward
// declaration fixed the class's visibility before the pragma below.
struct __attribute__((visibility("hidden"))) DataflowGraphAccess {
  // The Graph's composition as the compiler needs it, read as compile_public_graph reads it.
  static DataflowGraphView view(const Graph& graph);

  // nullptr: build this Run on GStreamer. Otherwise the RunCore of a Run on Dataflow. Throws
  // NeatError when RunOptions::backend is Dataflow and this build or this graph cannot honor it.
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const TensorList& seed, const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const Sample& seed, const char* where);
  static std::shared_ptr<RunCore> build(const Graph& graph, const RunOptions& opt, RunMode mode,
                                        const std::vector<cv::Mat>& seed, const char* where);
};

#pragma GCC visibility push(hidden)

#if defined(SIMANEAT_WITH_DATAFLOW)

// One field of a record binding: the Core metadata of a dense field, and where it lives in the
// record's slot (port_spec.h's field_offset and field_extent_offset, resolved once at binding).
struct DataflowFieldBinding {
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<TensorAxisSemantic> axis_semantics;
  Semantic semantic{};
  std::size_t rank = 0;
  std::size_t offset = 0;        ///< the field's bytes, from the slot start
  std::size_t extent_offset = 0; ///< the field's Extent in the record header
};

// Core's side of the bound input: the port spec a pushed tensor must fit, and the Core metadata
// (dtype, layout, semantic, planes) the runtime does not carry, restored on every pulled tensor.
// A leaf binds a TensorSpec; a record (an item of several tensors: TensorList, TensorSet, a
// positional Tensor::record) a RecordSchema with positional field names "0", "1", ... and one
// DataflowFieldBinding per field.
struct DataflowInputBinding {
  enum class Planes { None, Nv12, I420 };
  pipeline_internal::df::PortSpec spec;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<TensorAxisSemantic> axis_semantics;
  Semantic semantic{};
  Planes planes = Planes::None;
  std::size_t rank = 0; ///< Core rank: the spec's for a dense tensor, 2 ({H, W}) for planes
  std::vector<DataflowFieldBinding> fields; ///< a record's fields; empty for a leaf
  std::string text;                         ///< the spec as one line, for messages
  bool is_record() const noexcept {
    return !fields.empty();
  }
};

// The input's binding from its declaration and, when given, a tensor (a seed or the first push).
// nullopt with `why` set when this backend cannot carry it. A declared shape (InputOptions::shape)
// binds from the declaration alone; each of its dynamic axes needs a bound (max_shape).
std::optional<DataflowInputBinding> bind_dataflow_input(const InputOptions& declared,
                                                        const Tensor* tensor, std::string* why);

// A record binding from the tensors of one item (2 or more): every field dense, bound by its
// tensor (dtype and shape fixed). nullopt with `why` set when this backend cannot carry it.
std::optional<DataflowInputBinding> bind_dataflow_record(const InputOptions& declared,
                                                         std::span<const Tensor> fields,
                                                         std::string* why);

// "" when `tensor` fits `binding`; otherwise why it does not.
std::string dataflow_binding_mismatch(const DataflowInputBinding& binding, const Tensor& tensor);

// "" when the item's `fields` fit the record `binding`; otherwise why they do not.
std::string dataflow_record_mismatch(const DataflowInputBinding& binding,
                                     std::span<const Tensor> fields);

// The one-Feed, one-Fetch Net for `binding`, compiled. Throws NeatError on a compile error.
struct DataflowNet {
  pipeline_internal::df::Plan plan;
  std::uint32_t feed = 0;  ///< the Feed's operator index
  std::uint32_t fetch = 0; ///< the Fetch's operator index
};
DataflowNet compile_dataflow_net(const DataflowRunConfig& config,
                                 const DataflowInputBinding& binding, std::uint32_t stream);

#endif // SIMANEAT_WITH_DATAFLOW

#pragma GCC visibility pop

} // namespace simaai::neat::runtime
