#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
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
      GstCaps* caps = gst_caps_from_string(h264_caps.c_str());
      require(caps != nullptr, "failed to parse pooled encoded caps");

      GstBufferPool* pool = gst_buffer_pool_new();
      require(pool != nullptr, "failed to create encoded source buffer pool");
      GstStructure* config = gst_buffer_pool_get_config(pool);
      gst_buffer_pool_config_set_params(config, caps, 4U, 1U, 1U);
      require(gst_buffer_pool_set_config(pool, config),
              "failed to configure encoded source buffer pool");
      require(gst_buffer_pool_set_active(pool, TRUE),
              "failed to activate encoded source buffer pool");

      GstBuffer* source_buffer = nullptr;
      require(gst_buffer_pool_acquire_buffer(pool, &source_buffer, nullptr) == GST_FLOW_OK,
              "failed to acquire encoded source buffer");
      GstBuffer* expected_parent = source_buffer;
      GstMapInfo map{};
      require(gst_buffer_map(source_buffer, &map, GST_MAP_WRITE),
              "failed to map encoded source buffer");
      const uint8_t payload[] = {0x00, 0x00, 0x01, 0x67};
      std::memcpy(map.data, payload, sizeof(payload));
      gst_buffer_unmap(source_buffer, &map);
      require(gst_buffer_add_custom_meta(source_buffer, "GstSimaMeta") != nullptr,
              "failed to attach source GstSimaMeta");
      require(
          update_simaai_meta_fields(source_buffer, std::nullopt, 0, 0, std::nullopt, std::nullopt),
          "failed to initialize source GstSimaMeta");
      require(write_sample_timing_to_gst_buffer(source_buffer, SampleTimingOverrides{}),
              "failed to initialize source timing metadata");

      GstSample* gst_sample = gst_sample_new(source_buffer, caps, nullptr, nullptr);
      gst_buffer_unref(source_buffer);
      require(gst_sample != nullptr, "failed to create pooled encoded GstSample");
      Sample holder_sample = sample_from_gst_encoded(gst_sample, h264_caps);
      gst_sample_unref(gst_sample);
      holder_sample.tensors.front().device.type = DeviceType::SIMA_CVU;
      holder_sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
      // Differ from the pooled buffer metadata so the push path must create a writable clone.
      holder_sample.input_seq = 1;
      holder_sample.orig_input_seq = 1;

      auto gate = std::make_shared<pipeline_internal::HolderLoanGate>(1);
      std::string loan_error;
      require(pipeline_internal::attach_zero_copy_loan_to_sample(holder_sample, gate, &loan_error),
              loan_error.empty() ? "failed to attach zero-copy holder loan" : loan_error.c_str());
      require(gate->inflight() == 1, "device-backed fixture should acquire one holder loan");

      InputOptions src_opt;
      src_opt.payload_type = PayloadType::Encoded;
      src_opt.format = FormatTag::H264;
      src_opt.caps_override = h264_caps;
      src_opt.memory_policy = InputMemoryPolicy::Ev74;
      src_opt.block = true;

      GstElement* pipeline = gst_pipeline_new("zero-copy-parent-lifetime");
      GstElement* appsrc = gst_element_factory_make("appsrc", "mysrc");
      GstElement* appsink = gst_element_factory_make("appsink", "mysink");
      require(pipeline != nullptr && appsrc != nullptr && appsink != nullptr,
              "failed to create holder transfer test pipeline");
      g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", TRUE, nullptr);
      g_object_set(appsink, "sync", FALSE, "max-buffers", 1U, "drop", FALSE, "enable-last-sample",
                   FALSE, nullptr);
      gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
      gst_bin_add_many(GST_BIN(pipeline), appsrc, appsink, nullptr);
      require(gst_element_link(appsrc, appsink), "failed to link holder transfer test pipeline");
      appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
      appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
      require(appsrc != nullptr && appsink != nullptr,
              "failed to retain holder transfer test elements");
      require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
              "failed to start holder transfer test pipeline");

      InputStreamOptions stream_opt;
      stream_opt.copy_input = false;
      stream_opt.require_device_visible_input = true;
      InputStream stream =
          InputStream::create(pipeline, appsrc, appsink, derive_sample_spec_or_throw(holder_sample),
                              src_opt, stream_opt, {}, nullptr);
      require(stream.try_push_message(holder_sample), "pooled holder-backed encoded push failed");
      holder_sample = Sample{};

      GstSample* pulled = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), GST_SECOND);
      require(pulled != nullptr, "pooled holder-backed encoded pull timed out");
      require(gate->inflight() == 1,
              "downstream transfer buffer should retain the input holder loan");
      GstBuffer* transfer_buffer = gst_sample_get_buffer(pulled);
      GstCustomMeta* transfer_meta = gst_buffer_get_custom_meta(transfer_buffer, "GstSimaMeta");
      GstStructure* transfer_structure =
          transfer_meta ? gst_custom_meta_get_structure(transfer_meta) : nullptr;
      gint64 transfer_input_seq = -1;
      require(transfer_structure &&
                  gst_structure_get_int64(transfer_structure, "input-seq", &transfer_input_seq) &&
                  transfer_input_seq == 1,
              "zero-copy transfer should contain the updated metadata from the writable clone");
      GstParentBufferMeta* parent_meta = gst_buffer_get_parent_buffer_meta(transfer_buffer);
      require(parent_meta != nullptr && parent_meta->buffer == expected_parent,
              "zero-copy transfer should retain the original pooled buffer as its parent");

      GstBufferPoolAcquireParams acquire_params{};
      acquire_params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
      GstBuffer* blocked_buffer = nullptr;
      const GstFlowReturn blocked_flow =
          gst_buffer_pool_acquire_buffer(pool, &blocked_buffer, &acquire_params);
      if (blocked_buffer) {
        gst_buffer_unref(blocked_buffer);
      }
      require(blocked_flow != GST_FLOW_OK,
              "source buffer pool must not reacquire a buffer retained by a zero-copy transfer");

      gst_sample_unref(pulled);
      stream.close();
      GstBuffer* reacquired_buffer = nullptr;
      GstFlowReturn reacquire_flow = GST_FLOW_EOS;
      for (int attempt = 0; attempt < 100 && reacquire_flow != GST_FLOW_OK; ++attempt) {
        reacquire_flow = gst_buffer_pool_acquire_buffer(pool, &reacquired_buffer, &acquire_params);
        if (reacquire_flow != GST_FLOW_OK) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      require(reacquire_flow == GST_FLOW_OK,
              "source buffer should return to its pool after transfer release");
      require(gate->inflight() == 0,
              "input holder loan should release with the downstream transfer buffer");

      GstCustomMeta* legacy_meta = gst_buffer_add_custom_meta(reacquired_buffer, "GstSimaMeta");
      GstStructure* legacy_structure =
          legacy_meta ? gst_custom_meta_get_structure(legacy_meta) : nullptr;
      require(legacy_structure != nullptr, "failed to attach legacy GstSimaMeta");
      require(update_simaai_meta_fields(reacquired_buffer, std::nullopt, 0, 0, std::nullopt,
                                        std::nullopt),
              "failed to initialize legacy GstSimaMeta");
      GstBuffer* expected_legacy_buffer = reacquired_buffer;
      GstSample* legacy_gst_sample = gst_sample_new(reacquired_buffer, caps, nullptr, nullptr);
      gst_buffer_unref(reacquired_buffer);
      require(legacy_gst_sample != nullptr, "failed to create legacy metadata GstSample");
      Sample legacy_holder_sample = sample_from_gst_encoded(legacy_gst_sample, h264_caps);
      gst_sample_unref(legacy_gst_sample);
      legacy_holder_sample.tensors.front().device.type = DeviceType::SIMA_CVU;
      legacy_holder_sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
      require(sample_timing_overrides_from_sample(legacy_holder_sample).empty(),
              "legacy holder fixture must not carry a timing override");

      GstElement* legacy_pipeline = gst_pipeline_new("legacy-meta-zero-copy");
      GstElement* legacy_appsrc = gst_element_factory_make("appsrc", "legacy-src");
      GstElement* legacy_appsink = gst_element_factory_make("appsink", "legacy-sink");
      require(legacy_pipeline != nullptr && legacy_appsrc != nullptr && legacy_appsink != nullptr,
              "failed to create legacy metadata holder pipeline");
      g_object_set(legacy_appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", TRUE,
                   nullptr);
      g_object_set(legacy_appsink, "sync", FALSE, "max-buffers", 1U, "drop", FALSE,
                   "enable-last-sample", FALSE, nullptr);
      gst_app_src_set_caps(GST_APP_SRC(legacy_appsrc), caps);
      gst_bin_add_many(GST_BIN(legacy_pipeline), legacy_appsrc, legacy_appsink, nullptr);
      require(gst_element_link(legacy_appsrc, legacy_appsink),
              "failed to link legacy metadata holder pipeline");
      legacy_appsrc = gst_bin_get_by_name(GST_BIN(legacy_pipeline), "legacy-src");
      legacy_appsink = gst_bin_get_by_name(GST_BIN(legacy_pipeline), "legacy-sink");
      require(legacy_appsrc != nullptr && legacy_appsink != nullptr,
              "failed to retain legacy metadata holder elements");
      require(gst_element_set_state(legacy_pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
              "failed to start legacy metadata holder pipeline");
      InputStream legacy_stream = InputStream::create(
          legacy_pipeline, legacy_appsrc, legacy_appsink,
          derive_sample_spec_or_throw(legacy_holder_sample), src_opt, stream_opt, {}, nullptr);
      require(legacy_stream.try_push_message(legacy_holder_sample),
              "legacy metadata holder push should not require a writable envelope");
      legacy_holder_sample = Sample{};

      GstSample* legacy_pulled =
          gst_app_sink_try_pull_sample(GST_APP_SINK(legacy_appsink), GST_SECOND);
      require(legacy_pulled != nullptr, "legacy metadata holder pull timed out");
      require(gst_sample_get_buffer(legacy_pulled) == expected_legacy_buffer,
              "empty timing overrides must preserve the original shared holder buffer");
      gst_sample_unref(legacy_pulled);
      legacy_stream.close();

      reacquired_buffer = nullptr;
      reacquire_flow = GST_FLOW_EOS;
      for (int attempt = 0; attempt < 100 && reacquire_flow != GST_FLOW_OK; ++attempt) {
        reacquire_flow = gst_buffer_pool_acquire_buffer(pool, &reacquired_buffer, &acquire_params);
        if (reacquire_flow != GST_FLOW_OK) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      require(reacquire_flow == GST_FLOW_OK,
              "legacy metadata source buffer should return to its pool after transfer release");
      gst_buffer_unref(reacquired_buffer);
      require(gst_buffer_pool_set_active(pool, FALSE),
              "failed to deactivate encoded source buffer pool");
      gst_object_unref(pool);
      gst_caps_unref(caps);
    }

    {
      Sample holder_sample = sample_from_gst_encoded(h264_gst.get(), h264_caps);
      require_encoded_holder_sample(holder_sample, EncodedSpec::Codec::H264,
                                    "holder-backed encoded input");

      Graph p;
      InputOptions src_opt;
      src_opt.payload_type = PayloadType::Encoded;
      src_opt.format = FormatTag::H264;
      src_opt.caps_override = h264_caps;
      src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
      p.add(nodes::Input(src_opt));
      p.custom("fakesink name=encoded_holder_sink sync=false");

      RunOptions run_opt;
      Run run = p.build(Sample{holder_sample}, run_opt);
      require(run.push(Sample{holder_sample}), "holder-backed encoded Sample push failed");
      require(run.try_push_holder(holder_sample.tensors.front().storage->holder),
              "holder-only encoded push should reuse the saved full encoded spec");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      require(run.last_error().empty(), "unexpected error after holder-backed encoded push");
    }

    {
      Graph p;
      InputOptions src_opt;
      src_opt.payload_type = simaai::neat::PayloadType::Auto;
      src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
      p.add(nodes::Input(src_opt));
      p.custom("fakesink name=encoded_seed_caps_sink sync=false");

      RunOptions run_opt;
      Run run = p.build(Sample{h264_sample}, run_opt);
      const std::string pipeline = p.last_pipeline();
      require_contains(pipeline, "video/x-h264",
                       "encoded seed build should preserve H264 caps media");
      require_contains(pipeline, "stream-format=(string)byte-stream",
                       "encoded seed build should preserve H264 stream-format caps");
      require(pipeline.find("video/x-raw,format=ENCODED") == std::string::npos,
              "encoded seed build must not infer raw ENCODED caps");
      require(run.push(Sample{h264_sample}), "encoded seed caps push failed");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      require(run.last_error().empty(), "unexpected error after encoded seed caps push");
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
