// The format contract of one public Input on the Dataflow backend (DataflowInputBinding.h:
// InputOptions and the seed give shape, bounds and format): the input's port spec, the texts of a
// tensor that does not fit it, and the validated-geometry cache.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowInputBinding.h"

#include "pipeline/internal/SampleProjection.h"
#include "pipeline/runtime/DataflowDeviceMemory.h"
#include "pipeline/runtime/DataflowModelRoute.h"

#include <algorithm>
#include <array>
#include <variant>

namespace simaai::neat::runtime {

namespace {

namespace df = pipeline_internal::df;

std::optional<df::DType> dataflow_dtype(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return df::DType::UInt8;
  case TensorDType::Int8:
    return df::DType::Int8;
  case TensorDType::UInt16:
    return df::DType::UInt16;
  case TensorDType::Int16:
    return df::DType::Int16;
  case TensorDType::Int32:
    return df::DType::Int32;
  case TensorDType::BFloat16:
    return df::DType::BFloat16;
  case TensorDType::Float32:
    return df::DType::Float32;
  case TensorDType::Float64:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<ImageSpec::PixelFormat> pixel_format(FormatTag tag) {
  switch (tag) {
  case FormatTag::RGB:
    return ImageSpec::PixelFormat::RGB;
  case FormatTag::BGR:
    return ImageSpec::PixelFormat::BGR;
  case FormatTag::GRAY8:
    return ImageSpec::PixelFormat::GRAY8;
  case FormatTag::NV12:
    return ImageSpec::PixelFormat::NV12;
  case FormatTag::I420:
    return ImageSpec::PixelFormat::I420;
  case FormatTag::NV16:
    return ImageSpec::PixelFormat::NV16;
  default:
    return std::nullopt;
  }
}

// The declared format first, then the tensor's own, as describe_tensor_geometry_or_throw reads
// them.
std::optional<ImageSpec::PixelFormat> image_format(const InputOptions& declared,
                                                   const Tensor* tensor) {
  if (auto fmt = pixel_format(declared.format.tag)) {
    return fmt;
  }
  if (tensor && tensor->semantic.image.has_value() &&
      tensor->semantic.image->format != ImageSpec::PixelFormat::UNKNOWN) {
    return tensor->semantic.image->format;
  }
  return std::nullopt;
}

// One axis: max_* bounds it (min 1), a declared size fixes it, otherwise the tensor's size does.
std::optional<df::Dim> axis_dim(int declared, int max, std::int64_t actual, std::string* why,
                                const char* axis) {
  if (max > 0) {
    if (actual > max) {
      *why = std::string("its ") + axis + " " + std::to_string(actual) +
             " exceeds InputOptions::max_" + axis + " " + std::to_string(max);
      return std::nullopt;
    }
    return df::Dim(1, max);
  }
  if (declared > 0) {
    return df::Dim(declared);
  }
  if (actual > 0) {
    return df::Dim(actual);
  }
  return std::nullopt;
}

// Which axes are height, width and channels, as tensor_compat_dims_from_shape_local reads a
// shape. -1: none.
struct AxisRoles {
  int height = -1;
  int width = -1;
  int depth = -1;
};

AxisRoles axis_roles(std::size_t rank, TensorLayout layout) {
  const int r = static_cast<int>(rank);
  AxisRoles roles;
  if (layout == TensorLayout::HW) {
    roles.height = r >= 2 ? r - 2 : -1;
    roles.width = r - 1;
  } else if (layout == TensorLayout::CHW) {
    roles.depth = r >= 3 ? r - 3 : -1;
    roles.height = r >= 2 ? r - 2 : -1;
    roles.width = r - 1;
  } else if (r >= 3) {
    roles.height = r - 3;
    roles.width = r - 2;
    roles.depth = r - 1;
  } else if (r == 2) {
    roles.height = 0;
    roles.width = 1;
  } else {
    roles.width = 0;
  }
  return roles;
}

// Planar NV12 or I420 at the bound: planes packed back to back at the bound's pitches.
df::Planar planar_layout(df::PixelFormat format, std::int64_t max_h, std::int64_t max_w) {
  df::Planar planar;
  planar.format = format;
  const std::int64_t half_w = (max_w + 1) / 2;
  const std::int64_t half_h = (max_h + 1) / 2;
  planar.plane[0] = {0, max_w};
  if (format == df::PixelFormat::NV12) {
    planar.planes = 2;
    planar.plane[1] = {max_w * max_h, half_w * 2};
  } else {
    planar.planes = 3;
    planar.plane[1] = {max_w * max_h, half_w};
    planar.plane[2] = {max_w * max_h + half_h * half_w, half_w};
  }
  return planar;
}

// Where the input's slots live: DataflowGraphCompiler resolved InputOptions::memory_policy (never
// Auto here; a caller's own options, a cv::Mat view's, say SystemMemory).
df::MemoryKind memory_kind(InputMemoryPolicy policy) {
  switch (policy) {
  case InputMemoryPolicy::Ev74:
    return df::MemoryKind::Ev74;
  case InputMemoryPolicy::Dms0:
    return df::MemoryKind::Dms0;
  default:
    return df::MemoryKind::Host;
  }
}

std::string describe_spec(const df::PortSpec& spec) {
  // A record's description is one TensorSpec per field: up to kMaxFields of them.
  std::vector<char> buffer(spec.is_record() ? df::kDescribeMax * (df::kMaxFields + 1U)
                                            : df::kDescribeMax);
  df::Writer writer{std::span<char>(buffer)};
  df::describe(spec, writer);
  return std::string(writer.view());
}

// InputOptions::shape (Graph::input(name, shape, dtype)): the spec is the declaration, whatever
// the seed or the first push (validate_public_input checks a push against it). A dynamic axis is
// Dim{1, max_shape[i]}; without a bound it cannot be sized, so the Run declines.
std::optional<DataflowInputBinding> bind_declared_shape(const InputOptions& declared,
                                                        std::string* why) {
  const auto dtype = dataflow_dtype(declared.dtype);
  if (!dtype) {
    *why = "its declared dtype is Float64";
    return std::nullopt;
  }
  if (declared.shape.size() > df::kMaxRank) {
    *why = "its declared shape has rank " + std::to_string(declared.shape.size()) +
           ", where a tensor has rank 1 to 6";
    return std::nullopt;
  }
  df::TensorSpec spec;
  spec.dtype = *dtype;
  spec.layout = df::Dense{};
  spec.shape.rank = static_cast<std::uint8_t>(declared.shape.size());
  for (std::size_t i = 0; i < declared.shape.size(); ++i) {
    const std::int64_t size = declared.shape[i];
    const std::int64_t bound = declared.max_shape.empty() ? -1 : declared.max_shape[i];
    if (size > 0) {
      spec.shape.dims[i] = df::Dim(size);
    } else if (bound > 0) {
      spec.shape.dims[i] = df::Dim(1, bound);
    } else {
      *why = "axis " + std::to_string(i) +
             " of its declared shape is dynamic without a bound, and the input's pool is sized "
             "at build (Graph::input's max_shape declares the bound)";
      return std::nullopt;
    }
  }
  DataflowInputBinding binding;
  binding.dtype = declared.dtype;
  binding.rank = declared.shape.size();
  binding.spec.payload = spec;
  binding.spec.memory = memory_kind(declared.memory_policy);
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("its declared shape is not valid: ") + status->message;
    return std::nullopt;
  }
  binding.text = describe_spec(binding.spec);
  return binding;
}

} // namespace

std::optional<DataflowInputBinding> bind_dataflow_input(const InputOptions& declared,
                                                        const Tensor* tensor, std::string* why) {
  why->clear();
#if !defined(SIMANEAT_WITH_DATAFLOW_DEVICE_MEMORY)
  if (memory_kind(declared.memory_policy) != df::MemoryKind::Host) {
    *why = "it is in device memory (InputOptions::memory_policy Ev74 or Dms0), and this build "
           "of the Dataflow backend was configured without the Runner's memory package";
    return std::nullopt;
  }
#endif
  if (declared.payload_type == PayloadType::Encoded || declared.format.tag == FormatTag::H264 ||
      declared.format.tag == FormatTag::H265 || declared.format.tag == FormatTag::ENCODED) {
    *why = "it is an encoded input (an H.264, H.265 or other encoded payload)";
    return std::nullopt;
  }
  if (!declared.shape.empty()) {
    return bind_declared_shape(declared, why);
  }
  const std::optional<ImageSpec::PixelFormat> fmt = image_format(declared, tensor);
  if (fmt == ImageSpec::PixelFormat::NV16) {
    *why = "it is an NV16 input, where an image input is NV12, I420 or a packed format";
    return std::nullopt;
  }
  const bool nv12 = fmt == ImageSpec::PixelFormat::NV12;
  const bool i420 = fmt == ImageSpec::PixelFormat::I420;

  DataflowInputBinding binding;
  df::TensorSpec spec;
  if (tensor) {
    if (tensor->has_tiled_storage()) {
      *why = "it is a tiled (MLA) tensor";
      return std::nullopt;
    }
    if (tensor->semantic.encoded.has_value() || tensor->semantic.byte_stream.has_value()) {
      *why = "it is an encoded or byte-stream tensor";
      return std::nullopt;
    }
    const auto dtype = dataflow_dtype(tensor->dtype);
    if (!dtype) {
      *why = "it is a Float64 tensor";
      return std::nullopt;
    }
    if (tensor->is_composite() != (nv12 || i420)) {
      *why = tensor->is_composite() ? "it is a composite tensor other than NV12 or I420"
                                    : "it is an NV12 or I420 tensor without plane records";
      return std::nullopt;
    }
    binding.dtype = tensor->dtype;
    binding.layout = tensor->layout;
    binding.axis_semantics = tensor->axis_semantics;
    binding.semantic = tensor->semantic;
    binding.semantic.preprocess.reset(); // per-frame runtime metadata, not part of the spec
    spec.dtype = *dtype;
    if (nv12 || i420) {
      const int h = tensor->height();
      const int w = tensor->width();
      const auto hd = axis_dim(declared.height, declared.max_height, h, why, "height");
      const auto wd = axis_dim(declared.width, declared.max_width, w, why, "width");
      if (!hd || !wd) {
        if (why->empty())
          *why = "it has no width or height";
        return std::nullopt;
      }
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.rank = 2;
    } else {
      const std::size_t rank = tensor->shape.size();
      if (rank == 0U || rank > df::kMaxRank) {
        *why =
            "it is a tensor of rank " + std::to_string(rank) + ", where a tensor has rank 1 to 6";
        return std::nullopt;
      }
      const AxisRoles roles = axis_roles(rank, tensor->layout);
      spec.shape.rank = static_cast<std::uint8_t>(rank);
      for (std::size_t i = 0; i < rank; ++i) {
        const int axis = static_cast<int>(i);
        const std::int64_t actual = tensor->shape[i];
        std::optional<df::Dim> dim;
        if (axis == roles.height) {
          dim = axis_dim(declared.height, declared.max_height, actual, why, "height");
        } else if (axis == roles.width) {
          dim = axis_dim(declared.width, declared.max_width, actual, why, "width");
        } else if (axis == roles.depth && !fmt) {
          dim = axis_dim(declared.depth, declared.max_depth, actual, why, "depth");
        } else if (actual > 0) {
          dim = df::Dim(actual); // channels of an image format, batch and other axes: fixed
        }
        if (!dim) {
          if (why->empty())
            *why = "its axis " + std::to_string(i) + " is empty";
          return std::nullopt;
        }
        spec.shape.dims[i] = *dim;
      }
      binding.rank = rank;
    }
  } else {
    // Declaration only: an image with its format and both sizes (or bounds) declared.
    if (!fmt || (declared.payload_type != PayloadType::Image &&
                 declared.payload_type != PayloadType::Auto)) {
      return std::nullopt; // bound by the first push
    }
    std::string unused;
    const auto hd = axis_dim(declared.height, declared.max_height, 0, &unused, "height");
    const auto wd = axis_dim(declared.width, declared.max_width, 0, &unused, "width");
    if (!hd || !wd) {
      return std::nullopt;
    }
    spec.dtype = df::DType::UInt8;
    binding.dtype = TensorDType::UInt8;
    binding.semantic.image = ImageSpec{*fmt, ""};
    if (nv12 || i420) {
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.layout = TensorLayout::HW;
      binding.rank = 2;
    } else if (*fmt == ImageSpec::PixelFormat::GRAY8) {
      spec.shape = df::Shape::of({*hd, *wd});
      binding.layout = TensorLayout::HW;
      binding.rank = 2;
    } else {
      spec.shape = df::Shape::of({*hd, *wd, df::Dim(3)});
      binding.layout = TensorLayout::HWC;
      binding.rank = 3;
    }
  }
  if (fmt && !binding.semantic.image.has_value()) {
    binding.semantic.image = ImageSpec{*fmt, ""};
  } else if (fmt) {
    binding.semantic.image->format = *fmt;
  }
  if (nv12 || i420) {
    binding.planes = nv12 ? DataflowInputBinding::Planes::Nv12 : DataflowInputBinding::Planes::I420;
    spec.dtype = df::DType::UInt8;
    spec.layout = planar_layout(nv12 ? df::PixelFormat::NV12 : df::PixelFormat::I420,
                                spec.shape.dims[0].max, spec.shape.dims[1].max);
  } else {
    spec.layout = df::Dense{};
  }
  binding.spec.payload = spec;
  binding.spec.memory = memory_kind(declared.memory_policy);
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("its spec is not valid: ") + status->message;
    return std::nullopt;
  }
  binding.text = describe_spec(binding.spec);
  return binding;
}

std::string fix_dataflow_binding_at_bound(DataflowInputBinding& binding) {
  auto* spec = std::get_if<df::TensorSpec>(&binding.spec.payload);
  if (!spec || spec->shape.fixed()) {
    return {};
  }
  for (std::size_t i = 0; i < spec->shape.rank; ++i) {
    spec->shape.dims[i] = df::Dim(spec->shape.dims[i].max);
  }
  if (const df::Status status = df::validate(binding.spec)) {
    return std::string("the input's spec is not valid: ") + status->message;
  }
  binding.text = describe_spec(binding.spec);
  return {};
}

std::string dataflow_binding_mismatch(const DataflowInputBinding& binding, const Tensor& tensor) {
  const auto& spec = std::get<df::TensorSpec>(binding.spec.payload);
  const auto mismatch = [&](const std::string& what) {
    return what + "; the input is bound to " + binding.text +
           " (Dataflow backend: by its declaration, its seed or its first push)";
  };
  if (tensor.dtype != binding.dtype) {
    return mismatch("dtype differs");
  }
  if (binding.semantic.image.has_value() && tensor.semantic.image.has_value() &&
      tensor.semantic.image->format != ImageSpec::PixelFormat::UNKNOWN &&
      tensor.semantic.image->format != binding.semantic.image->format) {
    return mismatch(std::string("image format ") +
                    pipeline_internal::pixel_format_token(tensor.semantic.image->format) +
                    " differs");
  }
  if (binding.planes != DataflowInputBinding::Planes::None) {
    if (!tensor.is_composite()) {
      return mismatch("a dense tensor where planes are bound");
    }
    const std::int64_t h = tensor.height();
    const std::int64_t w = tensor.width();
    if (!spec.shape.dims[0].contains(h) || !spec.shape.dims[1].contains(w)) {
      return mismatch("size " + std::to_string(w) + "x" + std::to_string(h) + " is out of bounds");
    }
    if ((h % 2) != 0 || (w % 2) != 0) {
      return mismatch("odd size " + std::to_string(w) + "x" + std::to_string(h));
    }
    return {};
  }
  if (tensor.is_composite()) {
    return mismatch("a composite tensor where a dense one is bound");
  }
  if (tensor.shape.size() != spec.shape.rank) {
    return mismatch("rank " + std::to_string(tensor.shape.size()) + " differs");
  }
  for (std::size_t i = 0; i < spec.shape.rank; ++i) {
    if (!spec.shape.dims[i].contains(tensor.shape[i])) {
      return mismatch("dimension " + std::to_string(i) + " = " + std::to_string(tensor.shape[i]) +
                      " is out of bounds");
    }
  }
  return {};
}

std::optional<DataflowInputBinding> bind_dataflow_record(const InputOptions& declared,
                                                         std::span<const Tensor> fields,
                                                         std::string* why) {
  why->clear();
  if (!declared.shape.empty()) {
    *why = "it is an item of " + std::to_string(fields.size()) +
           " tensors where one tensor of a declared shape is expected";
    return std::nullopt;
  }
  if (memory_kind(declared.memory_policy) != df::MemoryKind::Host) {
    // The runtime's device ports carry one tensor (block.h); GStreamer's device pool is one too.
    *why = "it is an item of " + std::to_string(fields.size()) +
           " tensors (a record) into an input in device memory (InputOptions::memory_policy Ev74 "
           "or Dms0), where a device input carries one tensor a push";
    return std::nullopt;
  }
  if (fields.size() > df::kMaxFields) {
    *why = "it is a record of " + std::to_string(fields.size()) +
           " tensors, where a record has at most " + std::to_string(df::kMaxFields);
    return std::nullopt;
  }
  // Each field binds as an undeclared tensor of its own: the InputOptions describe one tensor (an
  // image's format and size), so only their pool settings apply to a field (a record's memory is
  // the host's, above).
  InputOptions field_declared;
  field_declared.memory_policy = InputMemoryPolicy::SystemMemory;
  field_declared.use_simaai_pool = declared.use_simaai_pool;
  field_declared.pool_max_buffers = declared.pool_max_buffers;
  DataflowInputBinding binding;
  df::RecordSchema schema;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    std::optional<DataflowInputBinding> field =
        bind_dataflow_input(field_declared, &fields[i], why);
    if (!field) {
      *why = "field " + std::to_string(i) +
             " of the record: " + (why->empty() ? std::string("it binds no spec") : *why);
      return std::nullopt;
    }
    if (field->planes != DataflowInputBinding::Planes::None) {
      *why = "field " + std::to_string(i) +
             " of the record is an NV12 or I420 image, where a record's fields are dense tensors";
      return std::nullopt;
    }
    // Dense fields, fixed by the tensor: a field has no declaration of its own.
    df::TensorSpec spec = std::get<df::TensorSpec>(field->spec.payload);
    for (std::size_t d = 0; d < spec.shape.rank; ++d) {
      spec.shape.dims[d] = df::Dim(fields[i].shape[d]);
    }
    schema.fields.push_back(
        df::Field{df::FieldId{static_cast<std::uint16_t>(i)}, std::to_string(i), spec});
    DataflowFieldBinding meta;
    meta.dtype = field->dtype;
    meta.layout = field->layout;
    meta.axis_semantics = std::move(field->axis_semantics);
    meta.semantic = std::move(field->semantic);
    meta.rank = field->rank;
    binding.fields.push_back(std::move(meta));
  }
  binding.spec.payload = std::move(schema);
  binding.spec.capacity_hint =
      static_cast<std::uint32_t>(std::clamp(declared.pool_max_buffers, 0, 4096));
  if (const df::Status status = df::validate(binding.spec)) {
    *why = std::string("its record spec is not valid: ") + status->message;
    return std::nullopt;
  }
  const auto& bound = std::get<df::RecordSchema>(binding.spec.payload);
  for (std::size_t i = 0; i < binding.fields.size(); ++i) {
    binding.fields[i].offset = df::field_offset(bound, i);
    binding.fields[i].extent_offset = df::field_extent_offset(bound, i);
  }
  binding.text = describe_spec(binding.spec);
  return binding;
}

std::string dataflow_record_mismatch(const DataflowInputBinding& binding,
                                     std::span<const Tensor> fields) {
  const auto mismatch = [&](const std::string& what) {
    return what + "; the input is bound to the record " + binding.text +
           " (Dataflow backend: by its seed or its first push)";
  };
  if (!binding.is_record()) {
    return "an item of " + std::to_string(fields.size()) +
           " tensors where one tensor is bound; the input is bound to " + binding.text +
           " (Dataflow backend: by its declaration, its seed or its first push)";
  }
  if (fields.size() != binding.fields.size()) {
    return mismatch("an item of " + std::to_string(fields.size()) + " tensors");
  }
  const auto& schema = std::get<df::RecordSchema>(binding.spec.payload);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const Tensor& tensor = fields[i];
    const df::TensorSpec& spec = schema.fields[i].spec;
    const auto field = [&mismatch, i](const std::string& what) {
      return mismatch("field " + std::to_string(i) + ": " + what);
    };
    if (tensor.dtype != binding.fields[i].dtype) {
      return field("dtype differs");
    }
    if (tensor.is_composite() || tensor.has_tiled_storage()) {
      return field("a composite or tiled tensor where a dense one is bound");
    }
    if (tensor.shape.size() != spec.shape.rank) {
      return field("rank " + std::to_string(tensor.shape.size()) + " differs");
    }
    for (std::size_t d = 0; d < spec.shape.rank; ++d) {
      if (!spec.shape.dims[d].contains(tensor.shape[d])) {
        return field("dimension " + std::to_string(d) + " = " + std::to_string(tensor.shape[d]) +
                     " is out of bounds");
      }
    }
  }
  return {};
}

// ---- the validated geometry of a leaf, and whether it can be wrapped
// ------------------------------

namespace {

// No Semantic beyond an image spec.
bool plain_semantic(const Semantic& s) noexcept {
  return !s.audio && !s.tokens && !s.text && !s.byte_stream && !s.tess && !s.encoded && !s.quant &&
         !s.detection && !s.feature && !s.preprocess;
}

} // namespace

bool DataflowLeafGeometry::keepable(const Tensor& t) noexcept {
  return t.storage &&
         (t.storage->kind == StorageKind::CpuOwned ||
          t.storage->kind == StorageKind::CpuExternal) &&
         plain_semantic(t.semantic);
}

bool DataflowLeafGeometry::keepable_device(const Tensor& t) noexcept {
  return t.storage && dataflow_device::device_storage_kind(t.storage->kind) &&
         plain_semantic(t.semantic);
}

// matches() for a device tensor: the same fields. Its own copy of the comparison, out of line, so
// that the inline one, which every push into a host input runs, stays as it is.
bool DataflowLeafGeometry::matches_device(const Tensor& t) const noexcept {
  if (!keepable_device(t) || t.storage->kind != kind || t.storage->size_bytes != storage_bytes ||
      t.dtype != dtype || t.layout != layout || t.byte_offset != byte_offset ||
      t.device.type != device.type || t.device.id != device.id || t.shape != shape ||
      t.strides_bytes != strides_bytes || t.axis_semantics != axis_semantics ||
      t.planes.size() != planes.size() || t.semantic.image.has_value() != image.has_value()) {
    return false;
  }
  if (image && (t.semantic.image->format != image->format ||
                t.semantic.image->color_space != image->color_space)) {
    return false;
  }
  for (std::size_t i = 0; i < planes.size(); ++i) {
    const Plane& a = t.planes[i];
    const Plane& b = planes[i];
    if (a.role != b.role || a.byte_offset != b.byte_offset || a.shape != b.shape ||
        a.strides_bytes != b.strides_bytes) {
      return false;
    }
  }
  return true;
}

const DataflowLeafGeometry*
DataflowValidatedLeaf::find_device(std::span<const Tensor> item) const noexcept {
  const DataflowLeafGeometry* known =
      item.size() == 1U ? leaf_.load(std::memory_order_acquire) : nullptr;
  return known && known->matches_device(item.front()) ? known : nullptr;
}

DataflowLeafGeometry DataflowLeafGeometry::of(const Tensor& t) {
  DataflowLeafGeometry g;
  g.kind = t.storage->kind;
  g.storage_bytes = t.storage->size_bytes;
  g.dtype = t.dtype;
  g.layout = t.layout;
  g.byte_offset = t.byte_offset;
  g.device = t.device;
  g.shape = t.shape;
  g.strides_bytes = t.strides_bytes;
  g.axis_semantics = t.axis_semantics;
  g.planes = t.planes;
  g.image = t.semantic.image;
  return g;
}

void DataflowValidatedLeaf::remember(const Tensor& tensor, bool wrappable) {
  if (!storage_) {
    auto geometry = std::make_unique<DataflowLeafGeometry>(DataflowLeafGeometry::of(tensor));
    geometry->wrappable = wrappable;
    storage_ = std::move(geometry);
    leaf_.store(storage_.get(), std::memory_order_release);
  }
}

// Whether a leaf's bytes are exactly a whole slot of `binding`'s port, so that the slot can be
// backed by the tensor's own storage. The runtime takes a wrap only of exactly slot_bytes (block.h:
// push then checks no extent against it), so the tensor's own bytes must be that size: every frame
// of a fixed-shape port, a frame at the bound of a bounded one (InputOptions::max_*); a smaller
// frame is copied. CPU storage that holds them from the tensor's first byte; a dense port's tensor
// dense and contiguous (a slot holds it tight); a planar port's (NV12, I420) frame at the bound,
// its planes at the offsets and pitches of the port's layout, which planar_layout packs back to
// back. Everything read here is part of DataflowLeafGeometry, so a repeated geometry reuses the
// answer; wrap_block (DataflowTensorCodec.h) checks the rest per push.
// A device port (memory Ev74 or Dms0) takes device memory only: the tensor's storage is device
// storage (dataflow_device::device_storage_kind: a TensorMemory::EV74 or MLA tensor;
// dataflow_device::wrap_tensor checks that it is one DMA-BUF of the port's kind), laid out by the
// same rule. A CPU tensor is copied into the slot.
// A model route whose frames change size (the binding's pulled tensor has the frames' table)
// has a bounded device port that takes every frame as it is: a dense, contiguous frame
// below the bound, held in memory of its own size, is wrapped too (the runtime's wrap shorter
// than a bounded port's bound, storage/block.h).
bool dataflow_wrap_layout(const DataflowInputBinding& binding, std::size_t slot_bytes,
                          const Tensor& t) {
  if (binding.is_record() || slot_bytes == 0U || !t.storage || t.byte_offset < 0 ||
      t.has_tiled_storage()) {
    return false;
  }
  if (binding.spec.memory != df::MemoryKind::Host
          ? !dataflow_device::device_storage_kind(t.storage->kind)
          : t.storage->kind != StorageKind::CpuOwned &&
                t.storage->kind != StorageKind::CpuExternal) {
    return false;
  }
  const auto offset = static_cast<std::size_t>(t.byte_offset);
  if (binding.pulled && binding.pulled->frames &&
      binding.planes == DataflowInputBinding::Planes::None) {
    if (!t.is_dense() || t.shape.empty() ||
        !(t.strides_bytes.empty() || t.strides_bytes.size() == t.shape.size()) ||
        !t.is_contiguous()) {
      return false;
    }
    const std::size_t bytes = t.dense_bytes_tight();
    return bytes != 0U && bytes <= slot_bytes && offset <= t.storage->size_bytes &&
           t.storage->size_bytes - offset >= bytes;
  }
  if (offset > t.storage->size_bytes || t.storage->size_bytes - offset < slot_bytes) {
    return false;
  }
  if (binding.planes == DataflowInputBinding::Planes::None) {
    return t.is_dense() && !t.shape.empty() &&
           (t.strides_bytes.empty() || t.strides_bytes.size() == t.shape.size()) &&
           t.is_contiguous() && t.dense_bytes_tight() == slot_bytes;
  }
  const auto& spec = std::get<df::TensorSpec>(binding.spec.payload);
  const auto& planar = std::get<df::Planar>(spec.layout);
  const bool nv12 = binding.planes == DataflowInputBinding::Planes::Nv12;
  if (t.dtype != TensorDType::UInt8 || !(nv12 ? t.is_nv12() : t.is_i420()) ||
      t.planes.size() != planar.planes) {
    return false;
  }
  const Plane* y = t.try_plane(PlaneRole::Y);
  if (!y || y->shape.size() < 2U) {
    return false;
  }
  const std::int64_t rows = y->shape[0];
  const std::int64_t cols = y->shape[1];
  if (rows != spec.shape.dims[0].max || cols != spec.shape.dims[1].max || rows % 2 != 0 ||
      cols % 2 != 0) {
    return false;
  }
  const auto at = [&](PlaneRole role, std::size_t p, std::int64_t r, std::int64_t c) {
    const Plane* plane = t.try_plane(role);
    return plane && plane->shape.size() >= 2U && plane->shape[0] == r && plane->shape[1] == c &&
           plane->byte_offset == planar.plane[p].offset_bytes &&
           (plane->strides_bytes.empty() ? c : plane->strides_bytes[0]) ==
               planar.plane[p].stride_bytes &&
           (plane->strides_bytes.size() < 2U || plane->strides_bytes[1] == 1);
  };
  if (!at(PlaneRole::Y, 0, rows, cols)) {
    return false;
  }
  return nv12 ? at(PlaneRole::UV, 1, rows / 2, cols)
              : at(PlaneRole::U, 1, rows / 2, cols / 2) && at(PlaneRole::V, 2, rows / 2, cols / 2);
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
