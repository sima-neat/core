#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct GstSampleUnref {
  void operator()(GstSample* sample) const {
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};

using GstSamplePtr = std::unique_ptr<GstSample, GstSampleUnref>;

GstSamplePtr make_gst_encoded_sample(const std::string& caps_string,
                                     const std::vector<uint8_t>& bytes) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes.size(), nullptr);
  require(buffer != nullptr, "failed to allocate encoded GstBuffer");

  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map encoded GstBuffer");
  std::memcpy(map.data, bytes.data(), bytes.size());
  gst_buffer_unmap(buffer, &map);

  GstCaps* caps = gst_caps_from_string(caps_string.c_str());
  require(caps != nullptr, "failed to parse encoded caps");

  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(sample != nullptr, "failed to allocate encoded GstSample");
  return GstSamplePtr(sample);
}

simaai::neat::Sample sample_from_gst_encoded(GstSample* gst_sample, const std::string& caps_string,
                                             std::string media_type = "video/x-h264") {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::TensorSet;
  sample.payload_type = simaai::neat::PayloadType::Encoded;
  sample.media_type = std::move(media_type);
  sample.caps_string = caps_string;
  sample.tensors = simaai::neat::TensorList{simaai::neat::from_gst_sample(gst_sample)};
  return sample;
}

void require_encoded_holder_sample(const simaai::neat::Sample& sample,
                                   simaai::neat::EncodedSpec::Codec codec, const char* where) {
  require(sample.tensors.size() == 1U, std::string(where) + ": expected one tensor");
  const auto& tensor = sample.tensors.front();
  require(tensor.semantic.encoded.has_value(), std::string(where) + ": missing encoded semantic");
  require(tensor.semantic.encoded->codec == codec, std::string(where) + ": codec mismatch");
  require(tensor.storage != nullptr, std::string(where) + ": missing storage");
  require(tensor.storage->kind == simaai::neat::StorageKind::GstSample,
          std::string(where) + ": expected GstSample storage");
  require(tensor.storage->holder != nullptr, std::string(where) + ": missing holder");
}

} // namespace

int main() {
  try {
    using namespace simaai::neat;
    simaai::neat::gst_init_once();

    Sample sample =
        make_encoded_sample(std::vector<uint8_t>(10, 0xAB), "video/x-h264", 100, 200, 300);
    require(sample.tensors.size() == 1U,
            "encoded Sample should expose one tensor in the TensorList");
    const auto& enc_tensor = sample.tensors.front();
    require(enc_tensor.semantic.encoded.has_value(), "encoded Sample missing semantic");
    require(enc_tensor.semantic.encoded->codec == simaai::neat::EncodedSpec::Codec::H264,
            "encoded Sample codec mismatch");
    require(enc_tensor.dtype == TensorDType::UInt8, "encoded Sample dtype mismatch");
    require(enc_tensor.shape.size() == 1 && enc_tensor.shape[0] == 10,
            "encoded Sample shape mismatch");
    require(sample.pts_ns == 100 && sample.dts_ns == 200 && sample.duration_ns == 300,
            "encoded Sample timestamps mismatch");

    const std::string h264_caps = "video/x-h264,stream-format=(string)byte-stream,"
                                  "alignment=(string)au,parsed=(boolean)true";
    const std::string jpeg_caps = "image/jpeg,width=(int)16,height=(int)16";
    Sample h264_sample =
        make_encoded_sample(std::vector<uint8_t>{0x00, 0x00, 0x01, 0x67}, h264_caps);

    GstSamplePtr h264_gst = make_gst_encoded_sample(h264_caps, {0x00, 0x00, 0x01, 0x67});
    Tensor h264_tensor = from_gst_sample(h264_gst.get());
    require(h264_tensor.semantic.encoded.has_value(), "H264 GstSample missing encoded semantic");
    require(h264_tensor.semantic.encoded->codec == EncodedSpec::Codec::H264,
            "H264 GstSample codec mismatch");
    require(h264_tensor.storage != nullptr && h264_tensor.storage->kind == StorageKind::GstSample,
            "H264 GstSample should keep GstSample storage");

    GstSamplePtr jpeg_gst = make_gst_encoded_sample(jpeg_caps, {0xFF, 0xD8, 0xFF, 0xD9});
    Tensor jpeg_tensor = from_gst_sample(jpeg_gst.get());
    require(jpeg_tensor.semantic.encoded.has_value(), "JPEG GstSample missing encoded semantic");
    require(jpeg_tensor.semantic.encoded->codec == EncodedSpec::Codec::JPEG,
            "JPEG GstSample codec mismatch");
    require(jpeg_tensor.storage != nullptr && jpeg_tensor.storage->kind == StorageKind::GstSample,
            "JPEG GstSample should keep GstSample storage");

    {
      Graph roundtrip;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Encoded;
      src_opt.format = simaai::neat::FormatTag::H264;
      src_opt.caps_override = h264_caps;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      roundtrip.add(nodes::Input(src_opt));
      roundtrip.add(nodes::Output(OutputOptions::Latest()));

      RunOptions zero_copy_opt;
      zero_copy_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
      Run zero_copy_run = roundtrip.build(Sample{h264_sample}, zero_copy_opt);
      require(zero_copy_run.push(Sample{h264_sample}), "encoded zero-copy roundtrip push failed");
      std::optional<Sample> pulled = zero_copy_run.pull(1000);
      require(pulled.has_value(), "encoded zero-copy roundtrip pull timed out");
      require_encoded_holder_sample(*pulled, EncodedSpec::Codec::H264,
                                    "encoded zero-copy roundtrip");
    }

    {
      Graph roundtrip;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Encoded;
      src_opt.format = simaai::neat::FormatTag::H264;
      src_opt.caps_override = h264_caps;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      roundtrip.add(nodes::Input(src_opt));
      roundtrip.add(nodes::Output(OutputOptions::Latest()));

      RunOptions owned_opt;
      owned_opt.output_memory = simaai::neat::OutputMemory::Owned;
      Run owned_run = roundtrip.build(Sample{h264_sample}, owned_opt);
      require(owned_run.push(Sample{h264_sample}), "encoded owned roundtrip push failed");
      std::optional<Sample> pulled = owned_run.pull(1000);
      require(pulled.has_value(), "encoded owned roundtrip pull timed out");
      require(pulled->tensors.size() == 1U, "encoded owned output should expose one tensor");
      require(pulled->tensors.front().storage != nullptr, "encoded owned output missing storage");
      require(pulled->tensors.front().storage->kind != StorageKind::GstSample,
              "encoded owned output should materialize bytes");
    }

    {
      Sample holder_sample = sample_from_gst_encoded(h264_gst.get(), h264_caps);
      require_encoded_holder_sample(holder_sample, EncodedSpec::Codec::H264,
                                    "holder-backed encoded input");

      Graph p;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Encoded;
      src_opt.format = simaai::neat::FormatTag::H264;
      src_opt.caps_override = h264_caps;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      p.add(nodes::Input(src_opt));
      p.custom("fakesink name=encoded_holder_sink sync=false");

      RunOptions run_opt;
      Run run = p.build(Sample{holder_sample}, run_opt);
      require(run.push(Sample{holder_sample}), "holder-backed encoded Sample push failed");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      require(run.last_error().empty(), "unexpected error after holder-backed encoded push");
    }

    Graph p;
    InputOptions src_opt;
    src_opt.payload_type = simaai::neat::PayloadType::Auto;
    src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
    p.add(nodes::Input(src_opt));
    p.custom("fakesink name=encoded_sink sync=false");

    RunOptions run_opt;
    Run run = p.build(Sample{sample}, run_opt);

    require(run.push(Sample{sample}), "encoded Sample push failed");
    Sample sample2 = make_encoded_sample(std::vector<uint8_t>(7, 0xCD), "video/x-h264");
    require(run.push(Sample{sample2}), "encoded Sample push2 failed");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    require(run.last_error().empty(), "unexpected error after encoded pushes");

    Sample bad = make_encoded_sample(std::vector<uint8_t>(5, 0xEF), "video/x-h265");
    require(run.push(Sample{bad}), "encoded Sample push3 failed");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!run.last_error().empty())
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(!run.last_error().empty(), "expected caps change error");
    require_contains(run.last_error(), "caps change", "caps change error missing detail");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
