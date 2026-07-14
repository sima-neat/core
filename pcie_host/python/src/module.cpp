#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <simaai/neat/pcie/Model.h>

#include <Python.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
namespace pcie = simaai::neat::pcie;

namespace {

struct PythonObjectHolder {
  explicit PythonObjectHolder(PyObject* object) : object(object) {
    Py_XINCREF(object);
  }

  ~PythonObjectHolder() {
    if (!object) {
      return;
    }
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(object);
    PyGILState_Release(state);
  }

  PyObject* object = nullptr;
};

std::size_t dtype_bytes(const pcie::TensorDType dtype) {
  switch (dtype) {
  case pcie::TensorDType::UInt8:
  case pcie::TensorDType::Int8:
    return 1;
  case pcie::TensorDType::UInt16:
  case pcie::TensorDType::Int16:
  case pcie::TensorDType::BFloat16:
    return 2;
  case pcie::TensorDType::Int32:
  case pcie::TensorDType::Float32:
    return 4;
  case pcie::TensorDType::Float64:
    return 8;
  }
  return 0;
}

const char* numpy_dtype_name(const pcie::TensorDType dtype) {
  switch (dtype) {
  case pcie::TensorDType::UInt8:
    return "uint8";
  case pcie::TensorDType::Int8:
    return "int8";
  case pcie::TensorDType::UInt16:
  case pcie::TensorDType::BFloat16:
    return "uint16";
  case pcie::TensorDType::Int16:
    return "int16";
  case pcie::TensorDType::Int32:
    return "int32";
  case pcie::TensorDType::Float32:
    return "float32";
  case pcie::TensorDType::Float64:
    return "float64";
  }
  return "uint8";
}

pcie::TensorDType dtype_from_numpy_name(const std::string& dtype_name) {
  if (dtype_name == "uint8") {
    return pcie::TensorDType::UInt8;
  }
  if (dtype_name == "int8") {
    return pcie::TensorDType::Int8;
  }
  if (dtype_name == "uint16") {
    return pcie::TensorDType::UInt16;
  }
  if (dtype_name == "int16") {
    return pcie::TensorDType::Int16;
  }
  if (dtype_name == "int32") {
    return pcie::TensorDType::Int32;
  }
  if (dtype_name == "float32") {
    return pcie::TensorDType::Float32;
  }
  if (dtype_name == "float64") {
    return pcie::TensorDType::Float64;
  }
  if (dtype_name == "bfloat16") {
    return pcie::TensorDType::BFloat16;
  }
  throw std::runtime_error("unsupported NumPy dtype for PCIe tensor: " + dtype_name);
}

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape,
                                             const std::size_t elem_size) {
  std::vector<std::int64_t> strides(shape.size(), 0);
  std::int64_t stride = static_cast<std::int64_t>(elem_size);
  for (std::size_t index = shape.size(); index > 0; --index) {
    const std::size_t dim = index - 1;
    strides[dim] = stride;
    stride *= shape[dim];
  }
  return strides;
}

pcie::TensorLayout infer_layout(const std::vector<std::int64_t>& shape) {
  if (shape.size() == 2) {
    return pcie::TensorLayout::HW;
  }
  if (shape.size() == 3) {
    return pcie::TensorLayout::HWC;
  }
  if (shape.size() == 4) {
    return pcie::TensorLayout::NHWC;
  }
  return pcie::TensorLayout::Unknown;
}

std::vector<std::int64_t> tuple_to_i64_vector(const nb::object& object) {
  std::vector<std::int64_t> out;
  nb::tuple tuple = nb::cast<nb::tuple>(object);
  out.reserve(tuple.size());
  for (nb::handle item : tuple) {
    out.push_back(nb::cast<std::int64_t>(item));
  }
  return out;
}

pcie::Tensor tensor_from_owned_bytes(std::vector<std::uint8_t> data, pcie::TensorDType dtype,
                                     std::vector<std::int64_t> shape,
                                     std::vector<std::int64_t> strides_bytes,
                                     pcie::PixelFormat image_format, std::string route_name) {
  if (dtype_bytes(dtype) == 0) {
    throw std::runtime_error("unsupported tensor dtype");
  }
  if (strides_bytes.empty()) {
    strides_bytes = contiguous_strides(shape, dtype_bytes(dtype));
  }

  auto owner = std::make_shared<std::vector<std::uint8_t>>(std::move(data));

  pcie::Tensor tensor;
  tensor.dtype = dtype;
  tensor.layout = infer_layout(shape);
  tensor.shape = std::move(shape);
  tensor.strides_bytes = std::move(strides_bytes);
  tensor.owner = owner;
  tensor.data = owner->empty() ? nullptr : owner->data();
  tensor.size_bytes = owner->size();
  tensor.image_format = image_format;
  if (image_format != pcie::PixelFormat::Unknown) {
    tensor.image = pcie::ImageSpec{image_format, ""};
  }
  tensor.route.name = std::move(route_name);
  return tensor;
}

pcie::Tensor tensor_from_numpy_copy(nb::object array, const pcie::PixelFormat image_format,
                                    const std::string& route_name) {
  nb::module_ np = nb::module_::import_("numpy");
  nb::object contiguous = np.attr("ascontiguousarray")(array);
  const auto dtype_name = nb::cast<std::string>(nb::str(contiguous.attr("dtype").attr("name")));
  const pcie::TensorDType dtype = dtype_from_numpy_name(dtype_name);
  const std::vector<std::int64_t> shape = tuple_to_i64_vector(contiguous.attr("shape"));
  const std::vector<std::int64_t> strides = tuple_to_i64_vector(contiguous.attr("strides"));

  Py_buffer view{};
  if (PyObject_GetBuffer(contiguous.ptr(), &view, PyBUF_SIMPLE) != 0) {
    throw nb::python_error();
  }
  std::vector<std::uint8_t> data;
  try {
    if (view.len < 0) {
      throw std::runtime_error("NumPy buffer reported a negative byte length");
    }
    data.resize(static_cast<std::size_t>(view.len));
    if (view.len > 0 && view.buf) {
      std::memcpy(data.data(), view.buf, static_cast<std::size_t>(view.len));
    }
  } catch (...) {
    PyBuffer_Release(&view);
    throw;
  }
  PyBuffer_Release(&view);

  return tensor_from_owned_bytes(std::move(data), dtype, shape, strides, image_format, route_name);
}

pcie::Tensor tensor_from_numpy_zero_copy(nb::object array, const pcie::PixelFormat image_format,
                                         const std::string& route_name) {
  nb::module_ np = nb::module_::import_("numpy");
  nb::object view_array = np.attr("asarray")(array);
  if (!nb::cast<bool>(view_array.attr("flags").attr("c_contiguous"))) {
    throw std::runtime_error("Tensor.from_numpy(copy=False) requires a C-contiguous NumPy array");
  }

  const auto dtype_name = nb::cast<std::string>(nb::str(view_array.attr("dtype").attr("name")));
  const pcie::TensorDType dtype = dtype_from_numpy_name(dtype_name);
  const std::vector<std::int64_t> shape = tuple_to_i64_vector(view_array.attr("shape"));
  const std::vector<std::int64_t> strides = tuple_to_i64_vector(view_array.attr("strides"));

  Py_buffer view{};
  if (PyObject_GetBuffer(view_array.ptr(), &view, PyBUF_SIMPLE) != 0) {
    throw nb::python_error();
  }

  try {
    if (view.len < 0) {
      throw std::runtime_error("NumPy buffer reported a negative byte length");
    }
    auto owner = std::make_shared<PythonObjectHolder>(view_array.ptr());
    pcie::Tensor tensor;
    tensor.dtype = dtype;
    tensor.layout = infer_layout(shape);
    tensor.shape = shape;
    tensor.strides_bytes =
        strides.empty() ? contiguous_strides(shape, dtype_bytes(dtype)) : strides;
    tensor.owner = std::move(owner);
    tensor.data = view.buf;
    tensor.size_bytes = static_cast<std::size_t>(view.len);
    tensor.byte_offset = 0;
    tensor.image_format = image_format;
    if (image_format != pcie::PixelFormat::Unknown) {
      tensor.image = pcie::ImageSpec{image_format, ""};
    }
    tensor.route.name = route_name;
    tensor.read_only = true;
    PyBuffer_Release(&view);
    return tensor;
  } catch (...) {
    PyBuffer_Release(&view);
    throw;
  }
}

pcie::Tensor tensor_from_numpy(nb::object array, const bool copy,
                               const pcie::PixelFormat image_format,
                               const std::string& route_name) {
  if (copy) {
    return tensor_from_numpy_copy(std::move(array), image_format, route_name);
  }
  return tensor_from_numpy_zero_copy(std::move(array), image_format, route_name);
}

nb::bytes tensor_to_bytes(const pcie::Tensor& tensor) {
  if (!tensor.data && tensor.size_bytes != 0) {
    throw std::runtime_error("tensor has no data pointer");
  }
  if (tensor.byte_offset < 0 || static_cast<std::size_t>(tensor.byte_offset) > tensor.size_bytes) {
    throw std::runtime_error("tensor byte offset is outside tensor payload");
  }
  const auto offset = static_cast<std::size_t>(tensor.byte_offset);
  const auto* data = static_cast<const char*>(tensor.data) + offset;
  const auto size = tensor.size_bytes - offset;
  return nb::bytes(data, size);
}

nb::object tensor_to_numpy(const pcie::Tensor& tensor) {
  nb::module_ np = nb::module_::import_("numpy");
  nb::object dtype = np.attr("dtype")(numpy_dtype_name(tensor.dtype));
  nb::object array = np.attr("frombuffer")(tensor_to_bytes(tensor), "dtype"_a = dtype);
  if (!tensor.shape.empty()) {
    nb::list shape;
    for (const auto dim : tensor.shape) {
      shape.append(nb::int_(dim));
    }
    try {
      array = array.attr("reshape")(shape);
    } catch (const nb::python_error&) {
      PyErr_Clear();
    }
  }
  return array;
}

pcie::Tensor tensor_from_bytes(nb::bytes payload, const pcie::TensorDType dtype,
                               const std::vector<std::int64_t>& shape,
                               const std::vector<std::int64_t>& strides_bytes,
                               const pcie::PixelFormat image_format,
                               const std::string& route_name) {
  char* bytes = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(payload.ptr(), &bytes, &size) != 0) {
    throw nb::python_error();
  }
  if (size < 0) {
    throw std::runtime_error("bytes object reported a negative size");
  }
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  if (size > 0 && bytes) {
    std::memcpy(data.data(), bytes, static_cast<std::size_t>(size));
  }
  return tensor_from_owned_bytes(std::move(data), dtype, shape, strides_bytes, image_format,
                                 route_name);
}

pcie::TensorList
tensor_list_from_python(nb::handle object,
                        const pcie::PixelFormat image_format = pcie::PixelFormat::Unknown,
                        const std::string& route_name = "") {
  if (nb::isinstance<pcie::Tensor>(object)) {
    return {nb::cast<pcie::Tensor>(object)};
  }

  if (PyList_Check(object.ptr()) || PyTuple_Check(object.ptr())) {
    pcie::TensorList tensors;
    nb::sequence sequence = nb::borrow<nb::sequence>(object);
    const Py_ssize_t size = PySequence_Size(object.ptr());
    if (size > 0) {
      tensors.reserve(static_cast<std::size_t>(size));
    }
    for (nb::handle item : sequence) {
      if (nb::isinstance<pcie::Tensor>(item)) {
        tensors.push_back(nb::cast<pcie::Tensor>(item));
      } else {
        tensors.push_back(
            tensor_from_numpy(nb::borrow<nb::object>(item), false, image_format, route_name));
      }
    }
    return tensors;
  }

  return {tensor_from_numpy(nb::borrow<nb::object>(object), false, image_format, route_name)};
}

nb::object optional_tensors_to_python(std::optional<pcie::TensorList> tensors) {
  if (!tensors) {
    return nb::none();
  }
  return nb::cast(std::move(*tensors));
}

nb::list tensors_to_numpy_list(const pcie::TensorList& tensors) {
  nb::list out;
  for (const auto& tensor : tensors) {
    out.append(tensor_to_numpy(tensor));
  }
  return out;
}

} // namespace

NB_MODULE(_pyneatpcie_core, m) {
  m.doc() = "Python bindings for the SiMa NEAT PCIe host co-processor API";

#if defined(PYNEATPCIE_VERSION)
  m.attr("__version__") = PYNEATPCIE_VERSION;
#else
  m.attr("__version__") = "0.0.0";
#endif

  nb::enum_<pcie::PipelineState>(m, "PipelineState")
      .value("Uninitialized", pcie::PipelineState::Uninitialized)
      .value("Starting", pcie::PipelineState::Starting)
      .value("Ready", pcie::PipelineState::Ready)
      .value("Failed", pcie::PipelineState::Failed)
      .value("Stopping", pcie::PipelineState::Stopping)
      .value("Exited", pcie::PipelineState::Exited);

  nb::enum_<pcie::InputKind>(m, "InputKind")
      .value("Tensor", pcie::InputKind::Tensor)
      .value("Image", pcie::InputKind::Image);

  nb::enum_<pcie::AutoFlag>(m, "AutoFlag")
      .value("Auto", pcie::AutoFlag::Auto)
      .value("On", pcie::AutoFlag::On)
      .value("Off", pcie::AutoFlag::Off);

  nb::enum_<pcie::ResizeMode>(m, "ResizeMode")
      .value("Stretch", pcie::ResizeMode::Stretch)
      .value("Letterbox", pcie::ResizeMode::Letterbox)
      .value("Crop", pcie::ResizeMode::Crop);

  nb::enum_<pcie::ColorFormat>(m, "ColorFormat")
      .value("Auto", pcie::ColorFormat::Auto)
      .value("RGB", pcie::ColorFormat::RGB)
      .value("BGR", pcie::ColorFormat::BGR)
      .value("GRAY8", pcie::ColorFormat::GRAY8)
      .value("NV12", pcie::ColorFormat::NV12)
      .value("I420", pcie::ColorFormat::I420);

  nb::enum_<pcie::NormalizePreset>(m, "NormalizePreset")
      .value("None", pcie::NormalizePreset::None)
      .value("ImageNet", pcie::NormalizePreset::ImageNet)
      .value("COCO_YOLO", pcie::NormalizePreset::COCO_YOLO);

  nb::enum_<pcie::BoxDecodeType>(m, "BoxDecodeType")
      .value("Unspecified", pcie::BoxDecodeType::Unspecified)
      .value("YoloV5", pcie::BoxDecodeType::YoloV5)
      .value("YoloV6", pcie::BoxDecodeType::YoloV6)
      .value("YoloV7", pcie::BoxDecodeType::YoloV7)
      .value("YoloV8", pcie::BoxDecodeType::YoloV8)
      .value("YoloV26", pcie::BoxDecodeType::YoloV26)
      .value("YoloX", pcie::BoxDecodeType::YoloX);

  nb::enum_<pcie::BoxDecodeTypeOption>(m, "BoxDecodeTypeOption")
      .value("Auto", pcie::BoxDecodeTypeOption::Auto)
      .value("Ultralytics", pcie::BoxDecodeTypeOption::Ultralytics)
      .value("EfficientNMS", pcie::BoxDecodeTypeOption::EfficientNMS);

  nb::enum_<pcie::TensorDType>(m, "TensorDType")
      .value("UInt8", pcie::TensorDType::UInt8)
      .value("Int8", pcie::TensorDType::Int8)
      .value("UInt16", pcie::TensorDType::UInt16)
      .value("Int16", pcie::TensorDType::Int16)
      .value("Int32", pcie::TensorDType::Int32)
      .value("BFloat16", pcie::TensorDType::BFloat16)
      .value("Float32", pcie::TensorDType::Float32)
      .value("Float64", pcie::TensorDType::Float64);

  nb::enum_<pcie::TensorLayout>(m, "TensorLayout")
      .value("Unknown", pcie::TensorLayout::Unknown)
      .value("HW", pcie::TensorLayout::HW)
      .value("HWC", pcie::TensorLayout::HWC)
      .value("NHWC", pcie::TensorLayout::NHWC);

  nb::enum_<pcie::PixelFormat>(m, "PixelFormat")
      .value("Unknown", pcie::PixelFormat::Unknown)
      .value("RGB", pcie::PixelFormat::RGB)
      .value("BGR", pcie::PixelFormat::BGR)
      .value("GRAY8", pcie::PixelFormat::GRAY8)
      .value("NV12", pcie::PixelFormat::NV12)
      .value("I420", pcie::PixelFormat::I420);

  nb::class_<pcie::ConnectionOptions>(m, "ConnectionOptions")
      .def(nb::init<>())
      .def_rw("card_host", &pcie::ConnectionOptions::card_host)
      .def_rw("card_id", &pcie::ConnectionOptions::card_id)
      .def_rw("user", &pcie::ConnectionOptions::user)
      .def_rw("queue", &pcie::ConnectionOptions::queue)
      .def_rw("max_inflight", &pcie::ConnectionOptions::max_inflight)
      .def_rw("card_env", &pcie::ConnectionOptions::card_env)
      .def_rw("card_gst_debug", &pcie::ConnectionOptions::card_gst_debug)
      .def_rw("card_gst_debug_file", &pcie::ConnectionOptions::card_gst_debug_file)
      .def(
          "__init__",
          [](pcie::ConnectionOptions* self, std::string card_host, int card_id, std::string user,
             int queue, int max_inflight) {
            new (self) pcie::ConnectionOptions();
            self->card_host = std::move(card_host);
            self->card_id = card_id;
            self->user = std::move(user);
            self->queue = queue;
            self->max_inflight = max_inflight;
          },
          "card_host"_a = "", "card_id"_a = 0, "user"_a = "sima", "queue"_a = 0,
          "max_inflight"_a = 10);

  nb::class_<pcie::ModelOptions::Preprocess::Resize>(m, "PreprocessResizeOptions")
      .def(nb::init<>())
      .def_rw("enable", &pcie::ModelOptions::Preprocess::Resize::enable)
      .def_rw("width", &pcie::ModelOptions::Preprocess::Resize::width)
      .def_rw("height", &pcie::ModelOptions::Preprocess::Resize::height)
      .def_rw("mode", &pcie::ModelOptions::Preprocess::Resize::mode)
      .def_rw("pad_value", &pcie::ModelOptions::Preprocess::Resize::pad_value)
      .def_rw("scaling_type", &pcie::ModelOptions::Preprocess::Resize::scaling_type);

  nb::class_<pcie::ModelOptions::Preprocess::ColorConvert>(m, "PreprocessColorConvertOptions")
      .def(nb::init<>())
      .def_rw("enable", &pcie::ModelOptions::Preprocess::ColorConvert::enable)
      .def_rw("input_format", &pcie::ModelOptions::Preprocess::ColorConvert::input_format)
      .def_rw("output_format", &pcie::ModelOptions::Preprocess::ColorConvert::output_format);

  nb::class_<pcie::ModelOptions::Preprocess::Normalize>(m, "PreprocessNormalizeOptions")
      .def(nb::init<>())
      .def_rw("enable", &pcie::ModelOptions::Preprocess::Normalize::enable)
      .def_rw("preset", &pcie::ModelOptions::Preprocess::Normalize::preset)
      .def_rw("has_explicit_stats", &pcie::ModelOptions::Preprocess::Normalize::has_explicit_stats)
      .def_rw("mean", &pcie::ModelOptions::Preprocess::Normalize::mean)
      .def_rw("stddev", &pcie::ModelOptions::Preprocess::Normalize::stddev);

  nb::class_<pcie::ModelOptions::Preprocess>(m, "PreprocessOptions")
      .def(nb::init<>())
      .def_rw("kind", &pcie::ModelOptions::Preprocess::kind)
      .def_rw("enable", &pcie::ModelOptions::Preprocess::enable)
      .def_rw("input_max_width", &pcie::ModelOptions::Preprocess::input_max_width)
      .def_rw("input_max_height", &pcie::ModelOptions::Preprocess::input_max_height)
      .def_rw("input_max_depth", &pcie::ModelOptions::Preprocess::input_max_depth)
      .def_rw("resize", &pcie::ModelOptions::Preprocess::resize)
      .def_rw("color_convert", &pcie::ModelOptions::Preprocess::color_convert)
      .def_rw("normalize", &pcie::ModelOptions::Preprocess::normalize);

  nb::class_<pcie::ModelOptions>(m, "ModelOptions")
      .def(nb::init<>())
      .def_rw("preprocess", &pcie::ModelOptions::preprocess)
      .def_rw("decode_type", &pcie::ModelOptions::decode_type)
      .def_rw("decode_type_option", &pcie::ModelOptions::decode_type_option)
      .def_rw("score_threshold", &pcie::ModelOptions::score_threshold)
      .def_rw("nms_iou_threshold", &pcie::ModelOptions::nms_iou_threshold)
      .def_rw("top_k", &pcie::ModelOptions::top_k)
      .def_rw("num_classes", &pcie::ModelOptions::num_classes);

  nb::class_<pcie::TensorRoute>(m, "TensorRoute")
      .def(nb::init<>())
      .def_rw("name", &pcie::TensorRoute::name)
      .def_rw("logical_index", &pcie::TensorRoute::logical_index)
      .def_rw("backend_output_index", &pcie::TensorRoute::backend_output_index)
      .def_rw("physical_index", &pcie::TensorRoute::physical_index)
      .def_rw("route_slot", &pcie::TensorRoute::route_slot)
      .def_rw("memory_index", &pcie::TensorRoute::memory_index)
      .def_rw("physical_byte_offset", &pcie::TensorRoute::physical_byte_offset);

  nb::class_<pcie::Tensor>(m, "Tensor")
      .def(nb::init<>())
      .def_rw("dtype", &pcie::Tensor::dtype)
      .def_rw("layout", &pcie::Tensor::layout)
      .def_rw("shape", &pcie::Tensor::shape)
      .def_rw("strides_bytes", &pcie::Tensor::strides_bytes)
      .def_rw("size_bytes", &pcie::Tensor::size_bytes)
      .def_rw("byte_offset", &pcie::Tensor::byte_offset)
      .def_rw("image_format", &pcie::Tensor::image_format)
      .def_rw("route", &pcie::Tensor::route)
      .def_rw("read_only", &pcie::Tensor::read_only)
      .def_static("from_bytes", &tensor_from_bytes, "payload"_a, "dtype"_a, "shape"_a,
                  "strides_bytes"_a = std::vector<std::int64_t>{},
                  "image_format"_a = pcie::PixelFormat::Unknown, "route_name"_a = "")
      .def_static("from_numpy", &tensor_from_numpy, "array"_a, "copy"_a = false,
                  "image_format"_a = pcie::PixelFormat::Unknown, "route_name"_a = "")
      .def("to_bytes", &tensor_to_bytes)
      .def("to_numpy", &tensor_to_numpy);

  nb::class_<pcie::TensorInfo>(m, "TensorInfo")
      .def(nb::init<>())
      .def_rw("name", &pcie::TensorInfo::name)
      .def_rw("dtype", &pcie::TensorInfo::dtype)
      .def_rw("shape", &pcie::TensorInfo::shape)
      .def_rw("size_bytes", &pcie::TensorInfo::size_bytes);

  nb::class_<pcie::ModelInfo>(m, "ModelInfo")
      .def(nb::init<>())
      .def_rw("inputs", &pcie::ModelInfo::inputs)
      .def_rw("outputs", &pcie::ModelInfo::outputs)
      .def_rw("has_preprocess", &pcie::ModelInfo::has_preprocess)
      .def_rw("has_boxdecode", &pcie::ModelInfo::has_boxdecode);

  nb::class_<pcie::Model>(m, "Model")
      .def(nb::init<std::string, pcie::ModelOptions, pcie::ConnectionOptions>(), "model_path"_a,
           "options"_a = pcie::ModelOptions{}, "connection"_a = pcie::ConnectionOptions{})
      .def("info", &pcie::Model::info)
      .def("input_specs", &pcie::Model::input_specs)
      .def("output_specs", &pcie::Model::output_specs)
      .def(
          "build",
          [](pcie::Model& model, const int readiness_timeout_ms) {
            nb::gil_scoped_release release;
            model.build(readiness_timeout_ms);
          },
          "readiness_timeout_ms"_a = 180000)
      .def("running", &pcie::Model::running)
      .def("close", &pcie::Model::close)
      .def(
          "__enter__",
          [](pcie::Model& model) -> pcie::Model& {
            return model;
          },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](pcie::Model& model, nb::handle, nb::handle, nb::handle) {
            model.close();
            return false;
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none())
      .def(
          "push",
          [](pcie::Model& model, nb::object tensors) {
            pcie::TensorList list = tensor_list_from_python(tensors);
            nb::gil_scoped_release release;
            return model.push(list);
          },
          "tensors"_a)
      .def(
          "push_numpy",
          [](pcie::Model& model, nb::object arrays) {
            pcie::TensorList list = tensor_list_from_python(arrays);
            nb::gil_scoped_release release;
            return model.push(list);
          },
          "arrays"_a)
      .def(
          "pull",
          [](pcie::Model& model, const int timeout_ms) {
            std::optional<pcie::TensorList> result;
            {
              nb::gil_scoped_release release;
              result = model.pull(timeout_ms);
            }
            return optional_tensors_to_python(std::move(result));
          },
          "timeout_ms"_a = -1)
      .def(
          "run",
          [](pcie::Model& model, nb::object tensors, const int timeout_ms) {
            pcie::TensorList list = tensor_list_from_python(tensors);
            nb::gil_scoped_release release;
            return model.run(list, timeout_ms);
          },
          "tensors"_a, "timeout_ms"_a = -1)
      .def(
          "run_numpy",
          [](pcie::Model& model, nb::object arrays, const int timeout_ms) {
            pcie::TensorList list = tensor_list_from_python(arrays);
            pcie::TensorList result;
            {
              nb::gil_scoped_release release;
              result = model.run(list, timeout_ms);
            }
            return tensors_to_numpy_list(result);
          },
          "arrays"_a, "timeout_ms"_a = -1)
      .def(
          "run_image",
          [](pcie::Model& model, nb::object image, const int timeout_ms,
             const pcie::PixelFormat format) {
            pcie::TensorList list = tensor_list_from_python(image, format, "input_image");
            pcie::TensorList result;
            {
              nb::gil_scoped_release release;
              result = model.run(list, timeout_ms);
            }
            return result;
          },
          "image"_a, "timeout_ms"_a = -1, "format"_a = pcie::PixelFormat::BGR);
}
