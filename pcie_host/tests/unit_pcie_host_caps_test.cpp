#include "HostPcieChannel.h"
#include "HostPcieTensorSetMeta.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie_internal = simaai::neat::pcie::internal;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    const std::string tensor_caps = pcie_internal::HostPcieChannel::tensor_set_caps();
    require(tensor_caps == "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                           "storage=(string)tensorbuffer",
            "unexpected tensor-set caps: " + tensor_caps);

    simaai::neat::pcie::Tensor tensor;
    tensor.shape = {1, 2, 3};
    const std::string concrete_tensor_caps =
        pcie_internal::HostPcieChannel::caps_for_tensors({tensor});
    require(concrete_tensor_caps ==
                "application/vnd.simaai.tensor, format=(string)EVXX_UINT8, "
                "dtype=(string)EVXX_UINT8, rank=(int)3, dim0=(int)1, dim1=(int)2, "
                "dim2=(int)3, shape=(string)\"1,2,3\", "
                "representation=(string)tensor-set, storage=(string)tensorbuffer",
            "plain tensor must use concrete tensor-set caps: " + concrete_tensor_caps);

    bool rejected_empty_payload = false;
    try {
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({});
    } catch (const std::runtime_error&) {
      rejected_empty_payload = true;
    }
    require(rejected_empty_payload, "empty tensor payload must be rejected");

    simaai::neat::pcie::Tensor image;
    image.dtype = simaai::neat::pcie::TensorDType::UInt8;
    image.layout = simaai::neat::pcie::TensorLayout::HWC;
    image.shape = {480, 640, 3};
    image.image =
        simaai::neat::pcie::ImageSpec{.format = simaai::neat::pcie::ImageSpec::PixelFormat::BGR};
    const std::string image_caps = pcie_internal::HostPcieChannel::caps_for_tensors({image});
    require(image_caps == "video/x-raw,format=(string)BGR,width=(int)640,height=(int)480",
            "unexpected image caps: " + image_caps);

    simaai::neat::pcie::Tensor invalid_image = image;
    invalid_image.dtype = simaai::neat::pcie::TensorDType::Float32;
    bool rejected_non_uint8_image = false;
    try {
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({invalid_image});
    } catch (const std::runtime_error&) {
      rejected_non_uint8_image = true;
    }
    require(rejected_non_uint8_image, "raw image caps must reject non-UInt8 storage");

    invalid_image = image;
    invalid_image.shape[2] = 1;
    bool rejected_wrong_channels = false;
    try {
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({invalid_image});
    } catch (const std::runtime_error&) {
      rejected_wrong_channels = true;
    }
    require(rejected_wrong_channels, "BGR caps must reject non-three-channel storage");

    simaai::neat::pcie::Tensor singleton_image = image;
    singleton_image.layout = simaai::neat::pcie::TensorLayout::NHWC;
    singleton_image.shape = {1, 480, 640, 3};
    const std::string singleton_image_caps =
        pcie_internal::HostPcieChannel::caps_for_tensors({singleton_image});
    require(singleton_image_caps == "video/x-raw,format=(string)BGR,width=(int)640,height=(int)480",
            "unexpected singleton NHWC image caps: " + singleton_image_caps);

    bool rejected_image_batch = false;
    try {
      singleton_image.shape[0] = 2;
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({singleton_image});
    } catch (const std::runtime_error&) {
      rejected_image_batch = true;
    }
    require(rejected_image_batch, "batched NHWC image payload must be rejected");

    simaai::neat::pcie::Tensor nv12;
    nv12.dtype = simaai::neat::pcie::TensorDType::UInt8;
    nv12.layout = simaai::neat::pcie::TensorLayout::HW;
    nv12.shape = {720, 640};
    nv12.image =
        simaai::neat::pcie::ImageSpec{.format = simaai::neat::pcie::ImageSpec::PixelFormat::NV12};
    const std::string nv12_caps = pcie_internal::HostPcieChannel::caps_for_tensors({nv12});
    require(nv12_caps == "video/x-raw,format=(string)NV12,width=(int)640,height=(int)480",
            "unexpected packed NV12 caps: " + nv12_caps);

    simaai::neat::pcie::Tensor i420 = nv12;
    i420.image =
        simaai::neat::pcie::ImageSpec{.format = simaai::neat::pcie::ImageSpec::PixelFormat::I420};
    const std::string i420_caps = pcie_internal::HostPcieChannel::caps_for_tensors({i420});
    require(i420_caps == "video/x-raw,format=(string)I420,width=(int)640,height=(int)480",
            "unexpected packed I420 caps: " + i420_caps);

    bool rejected_invalid_planar_height = false;
    try {
      nv12.shape = {721, 640};
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({nv12});
    } catch (const std::runtime_error&) {
      rejected_invalid_planar_height = true;
    }
    require(rejected_invalid_planar_height, "invalid packed NV12 height must be rejected");

    bool rejected_odd_planar_width = false;
    try {
      nv12.shape = {720, 641};
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({nv12});
    } catch (const std::runtime_error&) {
      rejected_odd_planar_width = true;
    }
    require(rejected_odd_planar_width, "odd packed NV12/I420 width must be rejected");

    simaai::neat::pcie::Tensor legacy_image = image;
    legacy_image.image.reset();
    legacy_image.image_format = simaai::neat::pcie::PixelFormat::RGB;
    const std::string legacy_caps =
        pcie_internal::HostPcieChannel::caps_for_tensors({legacy_image});
    require(legacy_caps == "video/x-raw,format=(string)RGB,width=(int)640,height=(int)480",
            "unexpected legacy image caps: " + legacy_caps);

    bool rejected_mixed_payload = false;
    try {
      (void)pcie_internal::HostPcieChannel::caps_for_tensors({image, tensor});
    } catch (const std::runtime_error&) {
      rejected_mixed_payload = true;
    }
    require(rejected_mixed_payload, "mixed raw image/tensor payload must be rejected");

    pcie_internal::HostPcieChannel::validate_output_payload_size(4096, 4096);
    pcie_internal::HostPcieChannel::validate_output_payload_size(8192, 4096);
    bool rejected_truncated_output = false;
    try {
      pcie_internal::HostPcieChannel::validate_output_payload_size(4095, 4096);
    } catch (const std::runtime_error&) {
      rejected_truncated_output = true;
    }
    require(rejected_truncated_output, "truncated PCIe output must be rejected");

    constexpr std::size_t kLogicalInt8Bytes = 2400;
    constexpr std::size_t kPackedCarrierBytes = 4096;
    constexpr std::size_t kDefaultRouteFp32Bytes = kLogicalInt8Bytes * 4;
    pcie_internal::HostPcieChannel::validate_output_payload_size(kPackedCarrierBytes,
                                                                 kPackedCarrierBytes, true);
    bool rejected_default_route_payload = false;
    try {
      pcie_internal::HostPcieChannel::validate_output_payload_size(kDefaultRouteFp32Bytes,
                                                                   kPackedCarrierBytes, true);
    } catch (const std::runtime_error&) {
      rejected_default_route_payload = true;
    }
    require(rejected_default_route_payload,
            "a compacted route must reject the larger payload the default route would return");
    bool rejected_short_carrier = false;
    try {
      pcie_internal::HostPcieChannel::validate_output_payload_size(kLogicalInt8Bytes,
                                                                   kPackedCarrierBytes, true);
    } catch (const std::runtime_error&) {
      rejected_short_carrier = true;
    }
    require(rejected_short_carrier, "a compacted route must reject a short carrier");

    simaai::neat::pcie::Tensor int8_tensor;
    int8_tensor.dtype = simaai::neat::pcie::TensorDType::Int8;
    int8_tensor.shape = {1, 2, 3};
    require(pcie_internal::HostPcieChannel::caps_for_tensors({int8_tensor}) ==
                "application/vnd.simaai.tensor, format=(string)EVXX_INT8, "
                "dtype=(string)EVXX_INT8, rank=(int)3, dim0=(int)1, dim1=(int)2, "
                "dim2=(int)3, shape=(string)\"1,2,3\", "
                "representation=(string)tensor-set, storage=(string)tensorbuffer",
            "mla_only INT8 input must negotiate EVXX_INT8 tensor-set caps");

    require(pcie_internal::HostPcieChannel::required_transport_buffer_size(1024, 2048, 4096) ==
                512U * 1024U,
            "transport buffer must retain the 512 KiB minimum");
    require(pcie_internal::HostPcieChannel::required_transport_buffer_size(
                1024U * 1024U, 4U * 1024U * 1024U, 6U * 1024U * 1024U) == 6U * 1024U * 1024U,
            "transport buffer must include the first submitted payload");
    bool rejected_oversized_transport = false;
    try {
      (void)pcie_internal::HostPcieChannel::required_transport_buffer_size(
          1024, 2048, 128U * 1024U * 1024U + 1U);
    } catch (const std::runtime_error&) {
      rejected_oversized_transport = true;
    }
    require(rejected_oversized_transport, "transport payload above 128 MiB must be rejected");

    pcie_internal::HostPcieChannel channel;
    GstBuffer* buffer = gst_buffer_new();
    require(buffer != nullptr, "failed to allocate request-ID test buffer");
    constexpr std::int32_t request_id = -123456789;
    pcie_internal::HostPcieChannel::attach_request_id(buffer, request_id);
    const auto restored = pcie_internal::HostPcieChannel::request_id_from_buffer(buffer);
    gst_buffer_unref(buffer);
    require(restored.has_value(), "request ID metadata must be readable");
    require(*restored == request_id, "request ID metadata must preserve the signed 32-bit value");

    channel.configure({}, 0, 0, 1, false);
    channel.request_stop();
    bool stopped_try_push_rejected = false;
    try {
      (void)channel.try_push(1, {tensor});
    } catch (const std::runtime_error& error) {
      stopped_try_push_rejected = std::string(error.what()).find("stopped") != std::string::npos;
    }
    require(stopped_try_push_rejected, "try_push must reject a stopped channel");

    bool stopped_push_rejected = false;
    try {
      (void)channel.push({tensor});
    } catch (const std::runtime_error& error) {
      stopped_push_rejected = std::string(error.what()).find("stopped") != std::string::npos;
    }
    require(stopped_push_rejected, "push must reject a stopped channel");

    {
      std::vector<std::uint8_t> blob(160, 0xEE);
      for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
          for (std::size_t k = 0; k < 2; ++k) {
            blob[r * 48 + c * 16 + k] = static_cast<std::uint8_t>(1 + (r * 3 + c) * 2 + k);
          }
        }
      }
      for (std::size_t i = 0; i < 64; ++i) {
        blob[96 + i] = static_cast<std::uint8_t>(100 + i);
      }
      std::vector<std::uint8_t> expected(76);
      for (std::size_t i = 0; i < 12; ++i) {
        expected[i] = static_cast<std::uint8_t>(1 + i);
      }
      for (std::size_t i = 0; i < 64; ++i) {
        expected[12 + i] = static_cast<std::uint8_t>(100 + i);
      }

      pcie_internal::PcieModelFacts facts;
      facts.outputs.resize(2);
      facts.outputs[0].name = "head_0";
      facts.outputs[0].dtype = "INT8";
      facts.outputs[0].shape = {2, 3, 2};
      facts.outputs[0].size_bytes = 12;
      facts.outputs[0].transport_strides_bytes = {48, 16, 1};
      facts.outputs[1].name = "head_1";
      facts.outputs[1].dtype = "INT8";
      facts.outputs[1].shape = {1, 4, 16};
      facts.outputs[1].size_bytes = 64;
      facts.outputs[1].payload_offset = 96;
      facts.outputs[1].transport_strides_bytes = {64, 16, 1};
      facts.outputs[1].dense_offset = 12;
      facts.packed_output_bytes = 160;
      facts.dense_output_bytes = 76;

      auto owner = std::make_shared<pcie_internal::MappedSample>();
      owner->buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_READONLY, blob.data(),
                                                  blob.size(), 0, blob.size(), nullptr, nullptr);
      owner->sample = gst_sample_new(owner->buffer, nullptr, nullptr, nullptr);
      gst_buffer_unref(owner->buffer);
      require(gst_buffer_map(owner->buffer, &owner->map, GST_MAP_READ),
              "failed to map the synthetic output buffer");
      owner->mapped = true;

      const auto heads = pcie_internal::HostPcieChannel::tensors_from_output_payload(owner, facts);
      require(heads.size() == 2U, "packed MLA output must yield one tensor per head");
      require(heads[0].owner && heads[0].owner == heads[1].owner && heads[0].owner != owner,
              "compacted heads must share one dense owner instead of the mapped sample");
      require(owner.use_count() == 1, "compacted heads must not retain the mapped sample");
      require(heads[0].dtype == simaai::neat::pcie::TensorDType::Int8 &&
                  heads[0].shape == std::vector<std::int64_t>({2, 3, 2}) &&
                  heads[0].size_bytes == 12U &&
                  heads[0].strides_bytes == std::vector<std::int64_t>({6, 2, 1}),
              "compacted head must publish contiguous logical geometry");
      const auto* dense = static_cast<const std::uint8_t*>(heads[0].data);
      require(static_cast<const std::uint8_t*>(heads[1].data) == dense + 12,
              "second head must follow the first in the dense block");
      require(std::vector<std::uint8_t>(dense, dense + 76) == expected,
              "transport padding must be compacted out of every head");
      require(heads[0].route.name == "head_0" && heads[1].route.name == "head_1",
              "compacted heads must keep their public names");

      facts.dense_output_bytes = 0;
      const auto views = pcie_internal::HostPcieChannel::tensors_from_output_payload(owner, facts);
      require(views.size() == 2U && views[1].owner == owner &&
                  views[1].data == static_cast<std::uint8_t*>(owner->map.data) + 96,
              "outputs without a dense block must stay zero-copy views of the mapped sample");
    }

    std::cout << "[PASS] host channel caps\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
