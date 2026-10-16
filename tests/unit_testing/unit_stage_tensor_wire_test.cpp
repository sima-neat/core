#include "pipeline/internal/StageTensorWire.h"
#include "test_main.h"

#include <memory>
#include <stdexcept>
#include <string>

RUN_TEST(
    "unit_stage_tensor_wire_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal;

      int map_count = 0;
      auto storage = std::make_shared<TensorBuffer>();
      storage->kind = StorageKind::CpuExternal;
      storage->size_bytes = 512U;
      storage->map_fn = [&](MapMode) -> Mapping {
        ++map_count;
        throw std::runtime_error("wire projection must not map data");
      };
      Tensor source;
      source.storage = storage;
      source.dtype = TensorDType::BFloat16;
      source.layout = TensorLayout::CHW;
      source.shape = {1, 3, 4, 5};
      source.strides_bytes = {192, 64, 16, 2};
      source.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::C, TensorAxisSemantic::H,
                               TensorAxisSemantic::W};
      source.byte_offset = 32;
      source.route.name = "logical_head";
      source.route.segment_name = "MLA_0";
      source.route.physical_byte_offset = 32;
      source.semantic.quant = QuantSpec{};
      source.semantic.quant->scale = 0.25F;

      const auto dims = stage_tensor_wire_dims(source);
      require(dims.width == 5 && dims.height == 4 && dims.depth == 3,
              "rank-four CHW input must use the common layout-aware caps projection");
      const Tensor wire = stage_source_tensor_view(source);
      require(wire.shape == source.shape && wire.strides_bytes == source.strides_bytes &&
                  wire.axis_semantics == source.axis_semantics && wire.layout == source.layout &&
                  wire.dtype == source.dtype && wire.byte_offset == source.byte_offset &&
                  wire.route.physical_byte_offset == source.route.physical_byte_offset &&
                  wire.route.name == source.route.name &&
                  wire.route.segment_name == source.route.segment_name &&
                  wire.storage == source.storage,
              "wire negotiation must preserve the complete padded source descriptor");
      require(wire.semantic.quant.has_value() && wire.semantic.quant->scale == 0.25F &&
                  !wire.semantic.tess.has_value() && !wire.semantic.byte_stream.has_value() &&
                  !wire.semantic.image.has_value(),
              "wire projection must preserve quantization without inventing tensor semantics");

      Tensor byte_stream;
      byte_stream.storage = storage;
      byte_stream.dtype = TensorDType::UInt8;
      byte_stream.shape = {128};
      byte_stream.strides_bytes = {1};
      byte_stream.byte_offset = 8;
      byte_stream.semantic.byte_stream = ByteStreamSpec{};
      byte_stream.semantic.byte_stream->description = "opaque packed input";
      const auto byte_dims = stage_tensor_wire_dims(byte_stream);
      const Tensor byte_wire = stage_source_tensor_view(byte_stream);
      require(byte_dims.width == 128 && byte_dims.height == 1 && byte_dims.depth == 1,
              "one-dimensional opaque input must have valid negotiation dimensions");
      require(byte_wire.shape == byte_stream.shape &&
                  byte_wire.strides_bytes == byte_stream.strides_bytes &&
                  byte_wire.byte_offset == byte_stream.byte_offset &&
                  byte_wire.layout == TensorLayout::Unknown &&
                  byte_wire.semantic.byte_stream.has_value() &&
                  byte_wire.semantic.byte_stream->description == "opaque packed input" &&
                  !byte_wire.semantic.tess.has_value(),
              "wire projection must not reshape or retag opaque input bytes");

      Tensor image = source;
      image.dtype = TensorDType::UInt8;
      image.layout = TensorLayout::HWC;
      image.shape = {4, 5, 3};
      image.strides_bytes = {20, 3, 1};
      image.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
      image.semantic.image = ImageSpec{};
      image.semantic.image->format = ImageSpec::PixelFormat::RGB;
      const Tensor image_wire = stage_source_tensor_view(image);
      require(image_wire.strides_bytes == image.strides_bytes &&
                  image_wire.semantic.image->format == ImageSpec::PixelFormat::RGB &&
                  !image_wire.semantic.tess.has_value(),
              "image negotiation must retain padded rows and not add tessellated semantics");
      require(map_count == 0, "source preservation and caps projection must never map payloads");
    }));
