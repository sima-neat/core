// Graph::input and Graph::output (H0 plan 3.2): named application endpoints by value. An input is
// a public Input vertex appended without a link to the previous Node; an output connects a value's
// producer to a public Output, through the same composition connect() uses. Nothing here is
// backend-specific: the GStreamer compiler and the Dataflow compiler read the resulting vertices
// as they read nodes::Input and nodes::Output.

#include "pipeline/Graph.h"

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/ErrorUtil.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

namespace {

namespace error_util = pipeline_internal::error_util;

std::string dims_text(const std::vector<std::int64_t>& dims) {
  std::string out = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
  }
  return out + "]";
}

void require_name(const std::string& name, const char* where, const char* code) {
  if (name.empty()) {
    error_util::throw_session_error(code, std::string(where) + ": the name is empty");
  }
}

// The declaration's own consistency; what a push must then fit is checked at push.
void check_declared_shape(const std::string& name, const std::vector<std::int64_t>& shape,
                          const std::vector<std::int64_t>& max_shape) {
  const std::string where = "Graph::input(\"" + name + "\", shape)";
  if (shape.empty()) {
    error_util::throw_session_error(error_codes::kInputShape,
                                    where + ": the shape is empty; use Graph::input(name) to "
                                            "declare an input whose shape the first push sets");
  }
  if (!max_shape.empty() && max_shape.size() != shape.size()) {
    error_util::throw_session_error(error_codes::kInputShape,
                                    where + ": max_shape " + dims_text(max_shape) +
                                        " has another rank than shape " + dims_text(shape));
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const std::int64_t size = shape[i];
    if (size == 0 || size < -1) {
      error_util::throw_session_error(error_codes::kInputShape,
                                      where + ": axis " + std::to_string(i) + " of shape " +
                                          dims_text(shape) +
                                          " is neither a size (>= 1) nor dynamic (-1)");
    }
    if (max_shape.empty()) {
      continue;
    }
    const std::int64_t bound = max_shape[i];
    if (size > 0 && bound != size && bound != -1) {
      error_util::throw_session_error(error_codes::kInputShape,
                                      where + ": axis " + std::to_string(i) + " is fixed at " +
                                          std::to_string(size) + " but max_shape gives " +
                                          std::to_string(bound));
    }
    if (size == -1 && bound != -1 && bound < 1) {
      error_util::throw_session_error(error_codes::kInputShape,
                                      where + ": the bound " + std::to_string(bound) +
                                          " of dynamic axis " + std::to_string(i) +
                                          " is not a size (>= 1, or -1 for none)");
    }
  }
}

FormatTag tensor_format_tag(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return FormatTag::UINT8;
  case TensorDType::Int8:
    return FormatTag::INT8;
  case TensorDType::BFloat16:
    return FormatTag::BF16;
  case TensorDType::Float32:
    return FormatTag::FP32;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::Int32:
  case TensorDType::Float64:
    break;
  }
  return FormatTag::Auto;
}

} // namespace

Graph::Handle Graph::input(std::string name) {
  require_name(name, "Graph::input", error_codes::kPipelineShape);
  auto node = std::make_shared<Input>(name);
  append_pipeline_vertex_for_internal_graph_(node);
  Handle out;
  out.node_ = std::move(node);
  out.graph_id_ = graph_id_;
  return out;
}

Graph::Handle Graph::input(std::string name, std::vector<std::int64_t> shape, TensorDType dtype,
                           std::vector<std::int64_t> max_shape) {
  require_name(name, "Graph::input", error_codes::kPipelineShape);
  check_declared_shape(name, shape, max_shape);
  InputOptions options;
  // A declared shape declares a tensor input: tensor media, its format the dtype's token where
  // one exists (GStreamer's tensor caps), and the shape that a push must fit.
  options.payload_type = PayloadType::Tensor;
  options.format = tensor_format_tag(dtype);
  options.shape = std::move(shape);
  options.max_shape = std::move(max_shape);
  options.dtype = dtype;
  auto node = std::make_shared<Input>(name, std::move(options));
  append_pipeline_vertex_for_internal_graph_(node);
  Handle out;
  out.node_ = std::move(node);
  out.graph_id_ = graph_id_;
  return out;
}

void Graph::output(std::string name, Handle value) {
  require_name(name, "Graph::output", error_codes::kPipelineShape);
  if (!value.node_) {
    error_util::throw_session_error(error_codes::kPipelineShape,
                                    "Graph::output(\"" + name + "\"): the Handle is empty");
  }
  if (value.graph_id_ != graph_id_) {
    error_util::throw_session_error(error_codes::kPipelineShape,
                                    "Graph::output(\"" + name +
                                        "\"): the Handle belongs to another Graph");
  }
  connect(value.node_, std::make_shared<Output>(std::move(name)));
}

} // namespace simaai::neat
