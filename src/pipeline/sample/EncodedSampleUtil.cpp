// src/pipeline/EncodedSampleUtil.cpp
#include "pipeline/EncodedSampleUtil.h"

#include "pipeline/Tensor.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace simaai::neat {
using pipeline_internal::lower_copy;
namespace {

const char* short_tag_for_codec(simaai::neat::EncodedSpec::Codec codec) {
  switch (codec) {
  case simaai::neat::EncodedSpec::Codec::H264:
    return "H264";
  case simaai::neat::EncodedSpec::Codec::H265:
    return "H265";
  case simaai::neat::EncodedSpec::Codec::RTP_H264:
    return "RTP_H264";
  case simaai::neat::EncodedSpec::Codec::RTP_H265:
    return "RTP_H265";
  case simaai::neat::EncodedSpec::Codec::JPEG:
    return "JPEG";
  case simaai::neat::EncodedSpec::Codec::UNKNOWN:
  default:
    return "";
  }
}

} // namespace

simaai::neat::EncodedSpec::Codec caps_to_codec(const std::string& caps_string) {
  const std::string caps = lower_copy(caps_string);
  const bool is_rtp = (caps.find("application/x-rtp") != std::string::npos);
  if (is_rtp && caps.find("encoding-name=h264") != std::string::npos) {
    return simaai::neat::EncodedSpec::Codec::RTP_H264;
  }
  if (is_rtp && caps.find("encoding-name=h265") != std::string::npos) {
    return simaai::neat::EncodedSpec::Codec::RTP_H265;
  }
  if (caps.find("video/x-h264") != std::string::npos) {
    return simaai::neat::EncodedSpec::Codec::H264;
  }
  if (caps.find("video/x-h265") != std::string::npos) {
    return simaai::neat::EncodedSpec::Codec::H265;
  }
  if (caps.find("image/jpeg") != std::string::npos) {
    return simaai::neat::EncodedSpec::Codec::JPEG;
  }
  return simaai::neat::EncodedSpec::Codec::UNKNOWN;
}

Sample make_encoded_sample(std::vector<uint8_t> bytes, std::string caps_string, int64_t pts_ns,
                           int64_t dts_ns, int64_t duration_ns) {
  if (bytes.empty()) {
    throw std::invalid_argument("make_encoded_sample: bytes are empty");
  }
  if (caps_string.empty()) {
    throw std::invalid_argument("make_encoded_sample: caps_string is empty");
  }

  Sample out;
  out.kind = SampleKind::TensorSet;
  out.caps_string = std::move(caps_string);
  out.pts_ns = pts_ns;
  out.dts_ns = dts_ns;
  out.duration_ns = duration_ns;

  auto holder = std::make_shared<std::vector<uint8_t>>(std::move(bytes));

  simaai::neat::Tensor tensor;
  tensor.storage =
      simaai::neat::make_cpu_external_storage(holder->data(), holder->size(), holder, true);
  tensor.device = {simaai::neat::DeviceType::CPU, 0};
  tensor.read_only = true;
  tensor.byte_offset = 0;
  tensor.dtype = TensorDType::UInt8;
  tensor.layout = TensorLayout::Unknown;
  tensor.shape = {static_cast<int64_t>(holder->size())};
  tensor.strides_bytes = {1};
  tensor.semantic.encoded = simaai::neat::EncodedSpec{};
  tensor.semantic.encoded->codec = caps_to_codec(out.caps_string);

  const char* tag = short_tag_for_codec(tensor.semantic.encoded->codec);
  if (tag && *tag) {
    out.payload_tag = tag;
  }

  out.tensors = TensorList{std::move(tensor)};
  return out;
}

} // namespace simaai::neat
