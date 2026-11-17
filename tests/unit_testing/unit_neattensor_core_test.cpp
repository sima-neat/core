#include "pipeline/TensorCore.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/InputStreamUtil.h"

#include <gst/gst.h>
#include <array>

#include "test_utils.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace simaai::neat;
namespace pi = simaai::neat::pipeline_internal;

template <class Action> void require_scalar_rejected(Action action, const std::string& why) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, why);
}

void test_scalar_copies_and_bounds() {
  std::array<std::int32_t, 3> values{991, 37, 992};
  Tensor scalar;
  scalar.storage = make_cpu_external_storage(values.data(), sizeof(values));
  scalar.dtype = TensorDType::Int32;
  scalar.byte_offset = sizeof(std::int32_t);
  const std::vector<uint8_t> expected(reinterpret_cast<const uint8_t*>(&values[1]),
                                      reinterpret_cast<const uint8_t*>(&values[1]) +
                                          sizeof(values[1]));
  require(scalar.shape.empty() && scalar.dense_bytes_tight() == sizeof(values[1]),
          "bound numerical rank-zero Tensor must contain one element");
  require(scalar.map_read().size_bytes == 2 * sizeof(values[1]),
          "public scalar mapping lost its complete accessible carrier extent");
  require(pi::tensor_bytes_tight(scalar) == scalar.dense_bytes_tight(),
          "scalar transport sizing disagrees with public dense sizing");
  require(scalar.copy_dense_bytes_tight() == expected && scalar.copy_payload_bytes() == expected,
          "scalar copy included padding or missed its nonzero offset");
  std::array<uint8_t, 8> destination{};
  destination.fill(0xAB);
  require(scalar.copy_dense_bytes_tight_to(destination.data(), destination.size()) &&
              std::memcmp(destination.data(), expected.data(), expected.size()) == 0 &&
              destination[4] == 0xAB,
          "scalar bounded copy wrote beyond one element");
  require(!scalar.copy_dense_bytes_tight_to(destination.data(), 3),
          "scalar copy accepted a short destination");
  const auto cloned = scalar.clone();
  require(cloned.storage != scalar.storage && cloned.byte_offset == 0 && cloned.shape.empty() &&
              cloned.strides_bytes.empty() && cloned.copy_dense_bytes_tight() == expected,
          "scalar clone lost rank/value/independent ownership");
  scalar.semantic.quant = QuantSpec{};
  require(scalar.dense_bytes_tight() == sizeof(values[1]) &&
              scalar.clone().copy_dense_bytes_tight() == expected,
          "quantized scalar metadata was mistaken for an opaque carrier");
  scalar.semantic.quant.reset();
  const auto created = Tensor::from_vector(std::vector<std::int32_t>{37}, {}, TensorMemory::CPU);
  require(created.shape.empty() && created.copy_dense_bytes_tight() == expected,
          "explicit rank-zero from_vector did not retain its one element");
  for (const std::vector<std::int32_t> input : {std::vector<std::int32_t>{}, {1, 2}}) {
    bool rejected = false;
    try {
      (void)Tensor::from_vector(input, {}, TensorMemory::CPU);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "rank-zero from_vector accepted zero or multiple elements");
  }

  // Malformed numerical scalars must not fall back to whole-carrier payloads.
  for (int defect = 0; defect < 5; ++defect) {
    Tensor invalid = scalar;
    if (defect < 2) {
      invalid.storage = make_cpu_owned_storage(defect == 0 ? 0 : 3);
      invalid.byte_offset = 0;
    } else if (defect == 2) {
      invalid.byte_offset = -1;
    } else if (defect == 3) {
      invalid.byte_offset = sizeof(values) - 3;
    } else {
      invalid.strides_bytes = {sizeof(std::int32_t)};
    }
    require(invalid.dense_bytes_tight() == sizeof(std::int32_t),
            "invalid scalar geometry fell back to opaque sizing");
    require(!invalid.copy_dense_bytes_tight_to(destination.data(), destination.size()),
            "invalid scalar copy accepted insufficient storage or malformed geometry");
    require_scalar_rejected([&] { (void)invalid.copy_dense_bytes_tight(); },
                            "invalid scalar tight copy was accepted");
    require_scalar_rejected([&] { (void)invalid.copy_payload_bytes(); },
                            "invalid scalar payload copy fell back to its carrier");
    require_scalar_rejected([&] { (void)invalid.clone(); }, "invalid scalar clone was accepted");
    Sample owned;
    owned.kind = SampleKind::TensorSet;
    owned.tensors = {invalid};
    pi::reset_tensor_io_stats();
    bool rejected_before_copy = false;
    try {
      pi::materialize_owned_output(owned);
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      rejected_before_copy =
          message.find("Owned output: tensor view exceeds") != std::string::npos ||
          message.find("Owned output: tensor stride rank") != std::string::npos;
    }
    require(rejected_before_copy && pi::snapshot_tensor_io_stats().tensor_copy_count == 0,
            "Owned scalar extent was not rejected before carrier allocation/copy");
  }

  for (int kind = 0; kind < 4; ++kind) {
    Tensor opaque = scalar;
    opaque.dtype = TensorDType::UInt8;
    if (kind == 0)
      opaque.semantic.encoded = EncodedSpec{};
    else if (kind == 1)
      opaque.semantic.byte_stream = ByteStreamSpec{};
    else if (kind == 2)
      opaque.semantic.tess = TessSpec{};
    else
      opaque.semantic.text = TextSpec{};
    require(opaque.dense_bytes_tight() == 0 && pi::tensor_bytes_tight(opaque) == 0 &&
                opaque.copy_payload_bytes().size() == sizeof(values) - sizeof(values[0]),
            "explicit opaque rank-unknown carrier was truncated to a numeric scalar");
  }
  Tensor unbound;
  require(unbound.dense_bytes_tight() == 0 && !unbound.map_read().data,
          "default Tensor fabricated scalar storage");
  require_scalar_rejected([&] { (void)unbound.clone(); }, "unbound clone fabricated storage");
  require_scalar_rejected([&] { (void)unbound.copy_dense_bytes_tight(); },
                          "unbound Tensor copied fabricated scalar bytes");
}

void test_scalar_callback_lifetime() {
  auto values =
      std::make_shared<std::array<std::int32_t, 3>>(std::array<std::int32_t, 3>{991, 37, 992});
  auto storage = std::make_shared<Storage>();
  storage->size_bytes = sizeof(*values);
  unsigned calls = 0, ends = 0;
  std::size_t exposed = sizeof(*values);
  storage->map_fn = [&, values](MapMode mode) {
    require(mode == MapMode::Read, "scalar reader requested write access");
    ++calls;
    Mapping mapped;
    mapped.data = values->data();
    mapped.size_bytes = exposed;
    mapped.unmap = [&] { ++ends; };
    return mapped;
  };
  Tensor scalar;
  scalar.storage = storage;
  scalar.dtype = TensorDType::Int32;
  scalar.byte_offset = sizeof(std::int32_t);
  exposed = 7;
  require(!pi::map_dense_tensor_read(scalar).data && calls == 1 && ends == 1,
          "short custom scalar map was accepted or lost its cleanup");
  exposed = sizeof(*values);
  auto mapped = pi::map_dense_tensor_read(scalar);
  require(mapped.data == values->data() + 1 && mapped.size_bytes == sizeof(std::int32_t) &&
              calls == 2 && ends == 1,
          "scalar reader changed callback authority, offset, or logical extent");
  std::weak_ptr<Storage> owner = storage;
  std::weak_ptr<std::array<std::int32_t, 3>> backing = values;
  scalar = {};
  storage.reset();
  values.reset();
  require(!owner.expired() && !backing.expired() && ends == 1 &&
              *static_cast<const std::int32_t*>(mapped.data) == 37,
          "scalar mapping did not retain its source until CPU access ended");
  mapped = {};
  require(owner.expired() && backing.expired() && ends == 2,
          "scalar mapping did not release its authority/cleanup exactly once");
}

void test_scalar_sample_packing() {
  auto values =
      std::make_shared<std::array<std::int32_t, 3>>(std::array<std::int32_t, 3>{991, 37, 992});
  Tensor scalar;
  scalar.storage = make_cpu_external_storage(values->data(), sizeof(*values), values);
  scalar.dtype = TensorDType::Int32;
  scalar.byte_offset = sizeof(std::int32_t);
  SampleSpec spec;
  spec.kind = SampleMediaKind::Tensor;
  spec.dtype = scalar.dtype;
  spec.tensor_view_bytes_actual = spec.required_bytes_actual = pi::tensor_bytes_tight(scalar);
  std::string error;
  auto buffer = std::shared_ptr<GstBuffer>(
      pi::build_gst_buffer_from_tensor(scalar, spec, {}, &error), gst_buffer_unref);
  require(buffer != nullptr, "scalar buffer packing failed: " + error);
  GstMapInfo mapped{};
  require(gst_buffer_map(buffer.get(), &mapped, GST_MAP_READ), "scalar buffer map failed");
  require(mapped.size == sizeof(std::int32_t) &&
              mapped.data == reinterpret_cast<const uint8_t*>(values->data() + 1) &&
              *reinterpret_cast<const std::int32_t*>(mapped.data) == 37,
          "scalar transport copied its zero-copy view or exposed carrier padding");
  gst_buffer_unmap(buffer.get(), &mapped);
  // Exercise the adapter's explicit-copy alternative, with the same logical extent.
  pi::GstBufferBuildPolicy policy;
  policy.allow_zero_copy = false;
  auto copied = std::shared_ptr<GstBuffer>(
      pi::build_gst_buffer_from_tensor(scalar, spec, policy, &error), gst_buffer_unref);
  require(copied != nullptr && gst_buffer_get_size(copied.get()) == sizeof(std::int32_t),
          "scalar transport copy lost its logical extent: " + error);
  std::int32_t actual = 0;
  require(gst_buffer_extract(copied.get(), 0, &actual, sizeof(actual)) == sizeof(actual) &&
              actual == 37,
          "scalar transport copy missed the selected offset");
  std::weak_ptr<std::array<std::int32_t, 3>> owner = values;
  scalar = {};
  values.reset();
  require(!owner.expired(), "scalar zero-copy buffer lost the source owner");
  buffer.reset();
  require(owner.expired(), "scalar zero-copy buffer retained the owner after release");
}

void test_scalar_selected_memory() {
  gst_init(nullptr, nullptr);
  auto* buffer = gst_buffer_new();
  for (const std::array<std::int32_t, 3> values :
       {std::array<std::int32_t, 3>{101, 102, 103}, {991, 37, 992}}) {
    auto* memory = gst_allocator_alloc(nullptr, sizeof(values), nullptr);
    GstMapInfo map{};
    require(memory && gst_memory_map(memory, &map, GST_MAP_WRITE), "scalar memory fixture map");
    std::memcpy(map.data, values.data(), sizeof(values));
    gst_memory_unmap(memory, &map);
    gst_buffer_append_memory(buffer, memory);
  }
  auto* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
  gst_buffer_unref(buffer);
  Tensor scalar;
  scalar.storage = pi::make_gst_sample_storage(sample);
  // Public TensorSet decoding binds the selected CPU memory's mapper; a route
  // number alone must not replace the whole-storage mapping authority.
  scalar = pi::tensor_view_from_sample_memory(scalar, 1);
  gst_sample_unref(sample);
  scalar.dtype = TensorDType::Int32;
  scalar.byte_offset = sizeof(std::int32_t);
  scalar.route.memory_index = 1;
  const auto copy = scalar.clone();
  require(scalar.map_read().size_bytes == 2 * sizeof(std::int32_t) &&
              copy.copy_dense_bytes_tight().size() == sizeof(std::int32_t) &&
              *copy.data_ptr<std::int32_t>() == 37,
          "scalar copy lost selected memory authority or public carrier extent");
}
} // namespace

int main() {
  try {
    test_scalar_copies_and_bounds();
    test_scalar_callback_lifetime();
    test_scalar_selected_memory();
    test_scalar_sample_packing();
    auto storage = simaai::neat::make_cpu_owned_storage(16);
    require(storage && storage->size_bytes == 16, "storage size mismatch");

    simaai::neat::Tensor t;
    t.storage = storage;
    t.dtype = simaai::neat::TensorDType::UInt8;
    t.shape = {4, 4};
    t.strides_bytes = {4, 1};
    t.device = {simaai::neat::DeviceType::CPU, 0};
    t.read_only = false;

    require(t.is_dense(), "expected dense tensor");
    require(t.is_contiguous(), "expected contiguous tensor");

    {
      auto map = t.map(simaai::neat::MapMode::ReadWrite);
      require(map.data != nullptr, "map failed");
      std::memset(map.data, 0xAB, map.size_bytes);
    }

    simaai::neat::Tensor copy = t.clone();
    require(copy.is_contiguous(), "clone not contiguous");
    require(copy.storage && copy.storage->size_bytes == 16, "clone size mismatch");
    {
      auto map = copy.map(simaai::neat::MapMode::Read);
      require(map.data != nullptr, "clone map failed");
      const uint8_t* bytes = static_cast<const uint8_t*>(map.data);
      require(bytes[0] == 0xAB, "clone data mismatch");
    }

    simaai::neat::Tensor non;
    auto storage_nc = simaai::neat::make_cpu_owned_storage(8);
    {
      auto map = storage_nc->map(simaai::neat::MapMode::Write);
      require(map.data != nullptr, "non-contig map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      buf[0] = 1;
      buf[1] = 2;
      buf[2] = 0;
      buf[3] = 0;
      buf[4] = 3;
      buf[5] = 4;
      buf[6] = 0;
      buf[7] = 0;
    }
    non.storage = storage_nc;
    non.dtype = simaai::neat::TensorDType::UInt8;
    non.shape = {2, 2};
    non.strides_bytes = {4, 1};
    non.device = {simaai::neat::DeviceType::CPU, 0};
    non.read_only = false;
    require(!non.is_contiguous(), "expected non-contiguous");
    {
      simaai::neat::Tensor non_copy = non.clone();
      require(non_copy.is_contiguous(), "non-contiguous clone not contiguous");
      auto map = non_copy.map(simaai::neat::MapMode::Read);
      require(map.data != nullptr, "non-contig clone map failed");
      const uint8_t* buf = static_cast<const uint8_t*>(map.data);
      require(buf[0] == 1 && buf[1] == 2 && buf[2] == 3 && buf[3] == 4,
              "non-contig clone data mismatch");
    }

    simaai::neat::Tensor empty;
    empty.storage = simaai::neat::make_cpu_owned_storage(0);
    empty.dtype = simaai::neat::TensorDType::Float32;
    empty.shape = {0, 256};
    empty.strides_bytes = {1024, 4};
    empty.device = {simaai::neat::DeviceType::CPU, 0};
    require(empty.copy_dense_bytes_tight().empty(), "empty tensor copy failed");

    {
      const int w = 4;
      const int h = 2;
      auto storage_i420 = simaai::neat::make_cpu_owned_storage(12);
      auto map = storage_i420->map(simaai::neat::MapMode::Write);
      require(map.data != nullptr, "i420 map failed");
      uint8_t* buf = static_cast<uint8_t*>(map.data);
      for (int i = 0; i < 8; ++i)
        buf[i] = static_cast<uint8_t>(i);
      buf[8] = 100;
      buf[9] = 101;
      buf[10] = 200;
      buf[11] = 201;

      simaai::neat::Tensor i420;
      i420.storage = storage_i420;
      i420.dtype = simaai::neat::TensorDType::UInt8;
      i420.layout = simaai::neat::TensorLayout::HW;
      i420.shape = {h, w};
      i420.device = {simaai::neat::DeviceType::CPU, 0};
      i420.read_only = true;
      i420.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::I420, ""};
      simaai::neat::Plane y;
      y.role = simaai::neat::PlaneRole::Y;
      y.shape = {h, w};
      y.strides_bytes = {w, 1};
      y.byte_offset = 0;
      simaai::neat::Plane u;
      u.role = simaai::neat::PlaneRole::U;
      u.shape = {h / 2, w / 2};
      u.strides_bytes = {w / 2, 1};
      u.byte_offset = 8;
      simaai::neat::Plane v;
      v.role = simaai::neat::PlaneRole::V;
      v.shape = {h / 2, w / 2};
      v.strides_bytes = {w / 2, 1};
      v.byte_offset = 10;
      i420.planes = {y, u, v};

      require(i420.i420_required_bytes() == 12, "i420 required bytes mismatch");
      auto mapped = i420.map_i420_read();
      require(mapped.has_value(), "i420 map_i420_read failed");
      require(mapped->view.width == w && mapped->view.height == h, "i420 dims mismatch");

      std::vector<uint8_t> out = i420.copy_i420_contiguous();
      require(out.size() == 12, "i420 copy size mismatch");
      require(out[0] == 0 && out[7] == 7, "i420 Y data mismatch");
      require(out[8] == 100 && out[9] == 101, "i420 U data mismatch");
      require(out[10] == 200 && out[11] == 201, "i420 V data mismatch");
      std::string err;
      require(i420.validate(&err), "i420 validate failed: " + err);
    }

    std::cout << "[OK] unit_tensortensor_core_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
