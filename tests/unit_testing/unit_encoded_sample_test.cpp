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

#include <atomic>
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
    Sample h264_holder = sample_from_gst_encoded(h264_gst.get(), h264_caps);
    require(!h264_holder.tensors.front().storage->sima_segments.empty(),
            "system-memory GstSample fixture should retain physical memory views");
    require(!pipeline_internal::sample_has_device_gstsample_holder(h264_holder),
            "system-memory encoded GstSample must not consume device-output loan credits");

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
      GstMemory* expected_memory = gst_buffer_peek_memory(source_buffer, 0);
      require(expected_memory != nullptr, "pooled encoded buffer should carry memory");
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
      require(gst_buffer_peek_memory(transfer_buffer, 0) == expected_memory,
              "zero-copy transfer should preserve the original pooled memory");

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

      GstBuffer* downstream_copy = gst_buffer_new();
      require(downstream_copy != nullptr, "failed to allocate downstream zero-copy view");
      const GstBufferCopyFlags copy_flags =
          static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS |
                                          GST_BUFFER_COPY_META | GST_BUFFER_COPY_MEMORY);
      require(gst_buffer_copy_into(downstream_copy, transfer_buffer, copy_flags, 0, -1),
              "failed to copy downstream zero-copy view");
      gst_sample_unref(pulled);
      stream.close();

      blocked_buffer = nullptr;
      const GstFlowReturn copied_view_blocked_flow =
          gst_buffer_pool_acquire_buffer(pool, &blocked_buffer, &acquire_params);
      if (blocked_buffer) {
        gst_buffer_unref(blocked_buffer);
      }
      require(copied_view_blocked_flow != GST_FLOW_OK,
              "source pool must remain blocked while a downstream zero-copy view is alive");
      gst_buffer_unref(downstream_copy);

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
      require(reacquired_buffer == expected_parent,
              "source pool should recycle the original buffer after transfer release");
      require(gate->inflight() == 0,
              "input holder loan should release with the downstream transfer buffer");

      GstCustomMeta* legacy_meta = gst_buffer_add_custom_meta(reacquired_buffer, "GstSimaMeta");
      GstStructure* legacy_structure =
          legacy_meta ? gst_custom_meta_get_structure(legacy_meta) : nullptr;
      require(legacy_structure != nullptr, "failed to attach legacy GstSimaMeta");
      require(update_simaai_meta_fields(reacquired_buffer, std::nullopt, 2, 2, std::nullopt,
                                        std::nullopt),
              "failed to initialize legacy GstSimaMeta");
      GstBuffer* legacy_view = gst_buffer_new();
      require(legacy_view != nullptr, "failed to allocate legacy zero-copy view");
      require(gst_buffer_copy_into(legacy_view, reacquired_buffer, copy_flags, 0, -1),
              "failed to create legacy zero-copy view");
      require(gst_buffer_add_parent_buffer_meta(legacy_view, reacquired_buffer) != nullptr,
              "failed to retain legacy pooled parent");
      GstBuffer* nested_view = gst_buffer_new();
      require(nested_view != nullptr, "failed to allocate nested zero-copy view");
      require(gst_buffer_copy_into(nested_view, legacy_view, copy_flags, 0, -1),
              "failed to create nested zero-copy view");
      require(gst_buffer_add_parent_buffer_meta(nested_view, legacy_view) != nullptr,
              "failed to retain intermediate zero-copy view");
      GstSample* legacy_gst_sample = gst_sample_new(nested_view, caps, nullptr, nullptr);
      gst_buffer_unref(nested_view);
      gst_buffer_unref(legacy_view);
      gst_buffer_unref(reacquired_buffer);
      require(legacy_gst_sample != nullptr, "failed to create legacy metadata GstSample");
      Sample legacy_holder_sample = sample_from_gst_encoded(legacy_gst_sample, h264_caps);
      gst_sample_unref(legacy_gst_sample);
      legacy_holder_sample.tensors.front().device.type = DeviceType::SIMA_CVU;
      legacy_holder_sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
      legacy_holder_sample.input_seq = 3;
      legacy_holder_sample.orig_input_seq = 3;
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
              "legacy metadata holder push should create a writable metadata view");
      legacy_holder_sample = Sample{};

      GstSample* legacy_pulled =
          gst_app_sink_try_pull_sample(GST_APP_SINK(legacy_appsink), GST_SECOND);
      require(legacy_pulled != nullptr, "legacy metadata holder pull timed out");
      GstBuffer* legacy_transfer_buffer = gst_sample_get_buffer(legacy_pulled);
      require(legacy_transfer_buffer != expected_parent,
              "metadata update should use a zero-copy transfer envelope");
      GstCustomMeta* legacy_transfer_meta =
          gst_buffer_get_custom_meta(legacy_transfer_buffer, "GstSimaMeta");
      GstStructure* legacy_transfer_structure =
          legacy_transfer_meta ? gst_custom_meta_get_structure(legacy_transfer_meta) : nullptr;
      gint64 legacy_transfer_input_seq = -1;
      require(legacy_transfer_structure &&
                  gst_structure_get_int64(legacy_transfer_structure, "input-seq",
                                          &legacy_transfer_input_seq) &&
                  legacy_transfer_input_seq == 3,
              "writable metadata view should contain the updated input sequence");
      require(gst_buffer_peek_memory(legacy_transfer_buffer, 0) == expected_memory,
              "no-loan transfer should preserve the original pooled memory");

      Sample replay_sample = sample_from_gst_encoded(legacy_pulled, h264_caps);
      replay_sample.tensors.front().device.type = DeviceType::SIMA_CVU;
      replay_sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
      replay_sample.input_seq = 4;
      replay_sample.orig_input_seq = 4;
      gst_sample_unref(legacy_pulled);
      require(legacy_stream.try_push_message(replay_sample),
              "replayed transfer should create another writable metadata view");
      replay_sample = Sample{};

      legacy_pulled = gst_app_sink_try_pull_sample(GST_APP_SINK(legacy_appsink), GST_SECOND);
      require(legacy_pulled != nullptr, "replayed metadata holder pull timed out");
      legacy_transfer_buffer = gst_sample_get_buffer(legacy_pulled);
      legacy_transfer_meta = gst_buffer_get_custom_meta(legacy_transfer_buffer, "GstSimaMeta");
      legacy_transfer_structure =
          legacy_transfer_meta ? gst_custom_meta_get_structure(legacy_transfer_meta) : nullptr;
      legacy_transfer_input_seq = -1;
      require(legacy_transfer_structure &&
                  gst_structure_get_int64(legacy_transfer_structure, "input-seq",
                                          &legacy_transfer_input_seq) &&
                  legacy_transfer_input_seq == 4,
              "replayed writable view should contain the updated input sequence");
      require(gst_buffer_peek_memory(legacy_transfer_buffer, 0) == expected_memory,
              "replayed transfer should preserve the original pooled memory");

      blocked_buffer = nullptr;
      const GstFlowReturn legacy_blocked_flow =
          gst_buffer_pool_acquire_buffer(pool, &blocked_buffer, &acquire_params);
      if (blocked_buffer) {
        gst_buffer_unref(blocked_buffer);
      }
      require(legacy_blocked_flow != GST_FLOW_OK,
              "source pool must not reacquire a no-loan holder retained downstream");
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
              "nested metadata source buffer should return to its pool after transfer release");
      require(reacquired_buffer == expected_parent,
              "nested metadata source pool should recycle the original buffer");

      // Reuse the same one-buffer pool root across two independent appsrc
      // consumers. This mirrors a graph fan-out where encoder and detector
      // branches both create a zero-copy transfer proxy for one camera frame.
      GstBuffer* fanout_parent = reacquired_buffer;
      GstMemory* fanout_memory = gst_buffer_peek_memory(reacquired_buffer, 0);
      require(fanout_memory == expected_memory,
              "fan-out fixture should reuse the original pooled memory");
      GstSample* fanout_gst_sample = gst_sample_new(reacquired_buffer, caps, nullptr, nullptr);
      gst_buffer_unref(reacquired_buffer);
      reacquired_buffer = nullptr;
      require(fanout_gst_sample != nullptr, "failed to create fan-out pooled GstSample");
      Sample fanout_holder_sample = sample_from_gst_encoded(fanout_gst_sample, h264_caps);
      gst_sample_unref(fanout_gst_sample);
      fanout_holder_sample.tensors.front().device.type = DeviceType::SIMA_CVU;
      fanout_holder_sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
      fanout_holder_sample.input_seq = 5;
      fanout_holder_sample.orig_input_seq = 5;

      const auto make_fanout_stream = [&](const char* pipeline_name, const char* appsrc_name,
                                          const char* appsink_name,
                                          GstElement** appsink_out) -> InputStream {
        GstElement* fanout_pipeline = gst_pipeline_new(pipeline_name);
        GstElement* fanout_appsrc = gst_element_factory_make("appsrc", appsrc_name);
        GstElement* fanout_appsink = gst_element_factory_make("appsink", appsink_name);
        require(fanout_pipeline != nullptr && fanout_appsrc != nullptr && fanout_appsink != nullptr,
                "failed to create fan-out holder transfer pipeline");
        g_object_set(fanout_appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", TRUE,
                     nullptr);
        g_object_set(fanout_appsink, "sync", FALSE, "max-buffers", 1U, "drop", FALSE,
                     "enable-last-sample", FALSE, nullptr);
        gst_app_src_set_caps(GST_APP_SRC(fanout_appsrc), caps);
        gst_bin_add_many(GST_BIN(fanout_pipeline), fanout_appsrc, fanout_appsink, nullptr);
        require(gst_element_link(fanout_appsrc, fanout_appsink),
                "failed to link fan-out holder transfer pipeline");
        fanout_appsrc = gst_bin_get_by_name(GST_BIN(fanout_pipeline), appsrc_name);
        fanout_appsink = gst_bin_get_by_name(GST_BIN(fanout_pipeline), appsink_name);
        require(fanout_appsrc != nullptr && fanout_appsink != nullptr,
                "failed to retain fan-out holder transfer elements");
        require(gst_element_set_state(fanout_pipeline, GST_STATE_PLAYING) !=
                    GST_STATE_CHANGE_FAILURE,
                "failed to start fan-out holder transfer pipeline");
        *appsink_out = fanout_appsink;
        return InputStream::create(fanout_pipeline, fanout_appsrc, fanout_appsink,
                                   derive_sample_spec_or_throw(fanout_holder_sample), src_opt,
                                   stream_opt, {}, nullptr);
      };

      GstElement* fanout_appsink_a = nullptr;
      GstElement* fanout_appsink_b = nullptr;
      InputStream fanout_stream_a = make_fanout_stream("zero-copy-fanout-a", "fanout-src-a",
                                                       "fanout-sink-a", &fanout_appsink_a);
      InputStream fanout_stream_b = make_fanout_stream("zero-copy-fanout-b", "fanout-src-b",
                                                       "fanout-sink-b", &fanout_appsink_b);
      require(fanout_stream_a.try_push_message(fanout_holder_sample),
              "first fan-out holder push failed");
      require(fanout_stream_b.try_push_message(fanout_holder_sample),
              "second fan-out holder push failed");

      GstSample* fanout_pulled_a =
          gst_app_sink_try_pull_sample(GST_APP_SINK(fanout_appsink_a), GST_SECOND);
      GstSample* fanout_pulled_b =
          gst_app_sink_try_pull_sample(GST_APP_SINK(fanout_appsink_b), GST_SECOND);
      require(fanout_pulled_a != nullptr && fanout_pulled_b != nullptr,
              "fan-out holder pull timed out");
      GstBuffer* fanout_transfer_a = gst_sample_get_buffer(fanout_pulled_a);
      GstBuffer* fanout_transfer_b = gst_sample_get_buffer(fanout_pulled_b);
      require(fanout_transfer_a != fanout_transfer_b,
              "fan-out branches should own distinct transfer buffers");
      require(gst_buffer_peek_memory(fanout_transfer_a, 0) == fanout_memory &&
                  gst_buffer_peek_memory(fanout_transfer_b, 0) == fanout_memory,
              "fan-out transfers should share the original pooled memory");
      GstParentBufferMeta* fanout_parent_meta_a =
          gst_buffer_get_parent_buffer_meta(fanout_transfer_a);
      GstParentBufferMeta* fanout_parent_meta_b =
          gst_buffer_get_parent_buffer_meta(fanout_transfer_b);
      require(fanout_parent_meta_a != nullptr && fanout_parent_meta_a->buffer != nullptr &&
                  fanout_parent_meta_b != nullptr && fanout_parent_meta_b->buffer != nullptr,
              "each fan-out transfer should retain a deferred parent proxy");
      require(fanout_parent_meta_a->buffer != fanout_parent_meta_b->buffer,
              "fan-out branches should own distinct deferred parent proxies");

      std::atomic<int> fanout_proxy_a_finalized{0};
      std::atomic<int> fanout_proxy_b_finalized{0};
      const auto count_proxy_finalization = [](gpointer data, GstMiniObject*) {
        static_cast<std::atomic<int>*>(data)->fetch_add(1, std::memory_order_relaxed);
      };
      gst_mini_object_weak_ref(GST_MINI_OBJECT_CAST(fanout_parent_meta_a->buffer),
                               count_proxy_finalization, &fanout_proxy_a_finalized);
      gst_mini_object_weak_ref(GST_MINI_OBJECT_CAST(fanout_parent_meta_b->buffer),
                               count_proxy_finalization, &fanout_proxy_b_finalized);
      const auto wait_for_proxy_finalization = [](const std::atomic<int>& count) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
        while (count.load(std::memory_order_relaxed) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return count.load(std::memory_order_relaxed) == 1;
      };

      fanout_holder_sample = Sample{};
      fanout_stream_a.close();
      fanout_stream_b.close();

      gst_sample_unref(fanout_pulled_a);
      require(wait_for_proxy_finalization(fanout_proxy_a_finalized),
              "first fan-out proxy should finalize with its branch");
      require(fanout_proxy_b_finalized.load(std::memory_order_relaxed) == 0,
              "second fan-out proxy must remain alive with its branch");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      blocked_buffer = nullptr;
      const GstFlowReturn one_branch_blocked_flow =
          gst_buffer_pool_acquire_buffer(pool, &blocked_buffer, &acquire_params);
      if (blocked_buffer) {
        gst_buffer_unref(blocked_buffer);
      }
      require(one_branch_blocked_flow != GST_FLOW_OK,
              "source pool must remain blocked while one fan-out branch is alive");

      gst_sample_unref(fanout_pulled_b);
      require(wait_for_proxy_finalization(fanout_proxy_b_finalized),
              "second fan-out proxy should finalize with its branch");
      GstBuffer* fanout_reacquired_buffer = nullptr;
      GstFlowReturn fanout_reacquire_flow = GST_FLOW_EOS;
      for (int attempt = 0; attempt < 100 && fanout_reacquire_flow != GST_FLOW_OK; ++attempt) {
        fanout_reacquire_flow =
            gst_buffer_pool_acquire_buffer(pool, &fanout_reacquired_buffer, &acquire_params);
        if (fanout_reacquire_flow != GST_FLOW_OK) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      require(fanout_reacquire_flow == GST_FLOW_OK,
              "source buffer should return to its pool after every fan-out branch releases it");
      require(fanout_reacquired_buffer == fanout_parent,
              "fan-out source pool should recycle the original buffer");
      require(gst_buffer_peek_memory(fanout_reacquired_buffer, 0) == fanout_memory,
              "fan-out source pool should recycle the original memory");
      gst_buffer_unref(fanout_reacquired_buffer);
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
