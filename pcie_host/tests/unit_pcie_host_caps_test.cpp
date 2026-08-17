#include "HostPcieChannel.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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

    std::cout << "[PASS] host channel caps\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
