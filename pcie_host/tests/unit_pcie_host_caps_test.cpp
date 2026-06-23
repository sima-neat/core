#include "HostPcieChannel.h"

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
    require(tensor_caps == "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer",
            "unexpected tensor-set caps: " + tensor_caps);

    simaai::neat::pcie::Tensor tensor;
    tensor.shape = {1, 2, 3};
    require(pcie_internal::HostPcieChannel::caps_for_tensors({tensor}) == tensor_caps,
            "plain tensor must use tensor-set caps");

    simaai::neat::pcie::Tensor image;
    image.dtype = simaai::neat::pcie::TensorDType::UInt8;
    image.layout = simaai::neat::pcie::TensorLayout::HWC;
    image.shape = {480, 640, 3};
    image.image = simaai::neat::pcie::ImageSpec{
        .format = simaai::neat::pcie::ImageSpec::PixelFormat::BGR};
    const std::string image_caps = pcie_internal::HostPcieChannel::caps_for_tensors({image});
    require(image_caps == "video/x-raw,format=(string)BGR,width=(int)640,height=(int)480",
            "unexpected image caps: " + image_caps);

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

    std::cout << "[PASS] host channel caps\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
