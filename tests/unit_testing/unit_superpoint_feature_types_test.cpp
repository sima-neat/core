#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/FeatureTypes.h"
#include "pipeline/GraphOptions.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

simaai::neat::Tensor byte_tensor(std::vector<std::uint8_t> bytes, const char* feature_format) {
  auto storage = simaai::neat::make_cpu_owned_storage(bytes.size());
  if (!bytes.empty()) {
    auto mapping = storage->map(simaai::neat::MapMode::Write);
    std::memcpy(mapping.data, bytes.data(), bytes.size());
  }
  simaai::neat::Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.shape = {static_cast<std::int64_t>(bytes.size())};
  tensor.strides_bytes = {1};
  if (feature_format) {
    simaai::neat::tag_feature_format(tensor, feature_format);
  }
  return tensor;
}
} // namespace

int main() {
  try {
    using namespace simaai::neat;
    require(static_cast<int>(BoxDecodeType::SuperPoint) == 23,
            "SuperPoint enum value must be append-only 23");
    require(std::string(box_decode_type_token(BoxDecodeType::SuperPoint)) == "superpoint",
            "SuperPoint token mismatch");

    FeaturePointsHeaderV1 header;
    header.count = 2;
    header.capacity = 2;
    header.descriptor_dim = 4;
    header.keypoints_offset = sizeof(header);
    header.keypoints_stride = 8;
    header.scores_offset = header.keypoints_offset + 16;
    header.scores_stride = 4;
    header.descriptors_offset = header.scores_offset + 8;
    header.descriptor_stride = 16;
    header.total_bytes = header.descriptors_offset + 32;
    std::vector<std::uint8_t> payload(header.total_bytes, 0U);
    std::memcpy(payload.data(), &header, sizeof(header));
    const float keypoints[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float scores[] = {0.75F, 0.5F};
    const float descriptors[] = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
    std::memcpy(payload.data() + header.keypoints_offset, keypoints, sizeof(keypoints));
    std::memcpy(payload.data() + header.scores_offset, scores, sizeof(scores));
    std::memcpy(payload.data() + header.descriptors_offset, descriptors, sizeof(descriptors));

    auto decoded = decode_superpoint_tensor(byte_tensor(std::move(payload), nullptr));
    require(decoded.keypoints.shape == std::vector<std::int64_t>({2, 2}),
            "V1 keypoint shape mismatch");
    require(decoded.scores.shape == std::vector<std::int64_t>({2}), "V1 score shape mismatch");
    require(decoded.descriptors.shape == std::vector<std::int64_t>({2, 4}),
            "V1 descriptor shape mismatch");
    require(decoded.descriptors.dtype == TensorDType::Float32, "V1 descriptor dtype mismatch");

    // Exercise the section-span arithmetic with maximally hostile 32-bit header fields. On
    // 32-bit hosts the multiplication/addition guard trips; on 64-bit hosts the resulting span
    // is still rejected against total_bytes. It must never wrap into an apparently valid section.
    FeaturePointsHeaderV1 extreme_header;
    extreme_header.count = 0;
    extreme_header.capacity = std::numeric_limits<std::uint32_t>::max();
    extreme_header.descriptor_dim = std::numeric_limits<std::uint16_t>::max();
    extreme_header.keypoints_offset = sizeof(extreme_header);
    extreme_header.keypoints_stride = std::numeric_limits<std::uint32_t>::max() - 3U;
    extreme_header.scores_offset = sizeof(extreme_header);
    extreme_header.scores_stride = std::numeric_limits<std::uint32_t>::max() - 3U;
    extreme_header.descriptors_offset = sizeof(extreme_header);
    extreme_header.descriptor_stride = std::numeric_limits<std::uint32_t>::max() - 3U;
    extreme_header.total_bytes = sizeof(extreme_header);
    std::vector<std::uint8_t> extreme_payload(sizeof(extreme_header), 0U);
    std::memcpy(extreme_payload.data(), &extreme_header, sizeof(extreme_header));
    bool extreme_rejected = false;
    try {
      (void)decode_superpoint_tensor(byte_tensor(std::move(extreme_payload), nullptr));
    } catch (const std::runtime_error& e) {
      const std::string message = e.what();
      extreme_rejected = message.find("overflows") != std::string::npos ||
                         message.find("out of bounds") != std::string::npos;
    }
    require(extreme_rejected,
            "V1 parser must reject section span multiplication/addition overflow");

    constexpr std::size_t kLegacyStride = 264U;
    std::vector<std::uint8_t> legacy(sizeof(std::int32_t) + kLegacyStride, 0U);
    const std::int32_t count = 1;
    const std::uint16_t x = 12;
    const std::uint16_t y = 34;
    const float score = 0.9F;
    std::memcpy(legacy.data(), &count, sizeof(count));
    std::memcpy(legacy.data() + 4, &x, sizeof(x));
    std::memcpy(legacy.data() + 6, &y, sizeof(y));
    std::memcpy(legacy.data() + 8, &score, sizeof(score));
    auto legacy_decoded =
        decode_superpoint_tensor(byte_tensor(std::move(legacy), kFeatureFormatLegacyA65V0));
    require(legacy_decoded.descriptors.shape == std::vector<std::int64_t>({1, 256}),
            "legacy descriptor shape mismatch");
    require(legacy_decoded.descriptors.dtype == TensorDType::Int8,
            "legacy descriptor dtype mismatch");

    // With boxdecode_raw_outputs the TensorSet holds the decoded legacy tensor, then the raw heads:
    // the first tensor keeps the sample's legacy tag and the raw heads stay untagged.
    std::vector<std::uint8_t> raw_legacy(sizeof(std::int32_t) + kLegacyStride, 0U);
    std::memcpy(raw_legacy.data(), &count, sizeof(count));
    Sample raw_sample;
    raw_sample.kind = SampleKind::TensorSet;
    raw_sample.payload_tag = kFeatureFormatLegacyA65V0;
    raw_sample.tensors.push_back(byte_tensor(std::move(raw_legacy), nullptr));
    raw_sample.tensors.push_back(byte_tensor(std::vector<std::uint8_t>(16, 0U), nullptr));
    tag_feature_format_in_sample(raw_sample);
    require(read_feature_format(raw_sample.tensors.front()) == kFeatureFormatLegacyA65V0 &&
                read_feature_format(raw_sample.tensors.back()).empty(),
            "raw-output TensorSet must tag only its decoded legacy feature tensor");
    require(decode_superpoint_tensor(raw_sample.tensors.front()).descriptors.shape ==
                std::vector<std::int64_t>({1, 256}),
            "legacy SuperPoint tensor with raw outputs must decode");

    BoxDecodeOptions options{BoxDecodeType::SuperPoint};
    options.superpoint.profile = SuperPointProfile::LightGlueV1;
    options.top_k = 600;
    auto node_base =
        nodes::SimaBoxDecode(options, "", 640, 480, 640, 480, BoxDecodeTypeOption::Auto,
                             BoxDecodeSourceStorage::DenseHwc);
    auto node = std::dynamic_pointer_cast<SimaBoxDecode>(node_base);
    require(static_cast<bool>(node), "SuperPoint node factory type mismatch");
    OutputSpec input;
    input.memory = "SystemMemory";
    require(node->output_spec(input).format == kFeatureFormatPointsV1,
            "SuperPoint output spec must be FEATURE_POINTS_V1");

    BoxDecodeOptions invalid_spatial{BoxDecodeType::SuperPoint};
    invalid_spatial.superpoint.profile = SuperPointProfile::LightGlueV1;
    invalid_spatial.superpoint.nms_radius = -2;
    bool invalid_spatial_rejected = false;
    try {
      (void)nodes::SimaBoxDecode(invalid_spatial, "", 640, 480, 640, 480, BoxDecodeTypeOption::Auto,
                                 BoxDecodeSourceStorage::DenseHwc);
    } catch (const std::invalid_argument& e) {
      invalid_spatial_rejected = std::string(e.what()).find("zero is valid") != std::string::npos;
    }
    require(invalid_spatial_rejected,
            "SuperPoint spatial values below the -1 sentinel must fail at construction");

    std::cout << "unit_superpoint_feature_types_test: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "unit_superpoint_feature_types_test: FAIL: " << e.what() << "\n";
    return 1;
  }
}
