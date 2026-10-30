#include "pipeline/graph/GraphDetail.h"
#include "test_main.h"

#include <gstsimaaitensorbuffer_meta.h>

#include <limits>

namespace {
using namespace simaai::neat;
using namespace simaai::gst;
using Buffer = std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>;

Buffer buffer() {
  return Buffer(gst_buffer_new_allocate(nullptr, 4, nullptr), gst_buffer_unref);
}

GstStructure* legacy_header(GstBuffer* buffer, gint64 frame, const char* stream) {
  auto* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  require(meta != nullptr, "legacy timing header");
  auto* structure = gst_custom_meta_get_structure(meta);
  gst_structure_set(structure, "frame-id", G_TYPE_INT64, frame, nullptr);
  if (stream)
    gst_structure_set(structure, "stream-id", G_TYPE_STRING, stream, nullptr);
  // These fields have never overridden the timing observer's frame/stream key.
  gst_structure_set(structure, "input-seq", G_TYPE_INT64, gint64(8), "orig-input-seq", G_TYPE_INT64,
                    gint64(9), "orig-stream-id", G_TYPE_STRING, "other-stream", "sample-frame-id",
                    G_TYPE_INT64, gint64(10), nullptr);
  return structure;
}

TensorBufferPublicationOwner publication() {
  TensorBufferPublishContract contract;
  contract.stage_key = "timing-parity";
  contract.physical_outputs.push_back({0, 4, "arena"});
  TensorBufferPublishLogicalOutput tensor;
  tensor.logical_index = tensor.physical_index = tensor.memory_index = 0;
  tensor.backend_output_index = tensor.route_slot = 0;
  tensor.logical_name = tensor.backend_name = "output";
  tensor.segment_name = "arena";
  tensor.size_bytes = 4;
  tensor.dtype = SIMA_TENSOR_SET_DTYPE_UINT8_V1;
  tensor.layout = SIMA_TENSOR_SET_LAYOUT_HW_V1;
  tensor.shape = {2, 2};
  tensor.stride_bytes = {2, 1};
  contract.logical_outputs.push_back(tensor);
  TensorBufferPreparedMetaTemplate prepared;
  std::string error;
  require(tensor_buffer_prepare_meta_template_from_contract(contract, &prepared, &error), error);
  auto owner = tensor_buffer_freeze_publication(prepared, &error);
  require(bool(owner), error);
  return owner;
}
} // namespace

RUN_TEST("unit_element_timing_metadata_test", ([] {
           gst_init(nullptr, nullptr);
           tensor_buffer_register_meta();
           if (!gst_meta_get_info("GstSimaMeta")) {
             static const gchar* tags[] = {nullptr};
             gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr);
           }
           const auto owner = publication();
           for (const auto frame : {gint64(0), gint64(300), std::numeric_limits<gint64>::max()}) {
             for (const char* stream : {"camera", ""}) {
               auto legacy = buffer();
               legacy_header(legacy.get(), frame, stream);
               pipeline_internal::ElementTimingKey expected;
               require(extract_element_timing_key(legacy.get(), expected), "legacy key missing");
               require(expected.frame_id == frame &&
                           expected.stream_hash == uint32_t(std::hash<std::string>{}(stream)),
                       "legacy key precedence changed");

               auto compact = buffer();
               std::string error;
               require(tensor_buffer_attach_publication(compact.get(), owner, &error), error);
               legacy_header(compact.get(), frame, stream);
               pipeline_internal::ElementTimingKey actual;
               require(extract_element_timing_key(compact.get(), actual) &&
                           pipeline_internal::ElementTimingKeyEq{}(expected, actual),
                       "V1 timing header must use legacy fallback");

               TensorBufferFrameMetadata current;
               current.frame_id = frame;
               current.stream_id = stream;
               current.input_sequence = 8;
               current.original_sequence = 9;
               require(tensor_buffer_attach_frame_metadata(compact.get(), current, &error, true),
                       error);
               require(extract_element_timing_key(compact.get(), actual) &&
                           pipeline_internal::ElementTimingKeyEq{}(expected, actual),
                       "V2 timing key differs from legacy/V1");
               require(!gst_buffer_get_custom_meta(compact.get(), "GstSimaMeta") &&
                           tensor_buffer_frame_metadata(compact.get(), &error)->frame_id == frame,
                       "timing observer changed compact representation");

               legacy_header(compact.get(), frame + (frame != G_MAXINT64), "conflicting");
               actual = {71, 72};
               require(!extract_element_timing_key(compact.get(), actual) &&
                           actual.frame_id == 71 && actual.stream_hash == 72,
                       "mixed authorities must not produce a guessed timing key");
             }
           }
           auto missing = buffer();
           pipeline_internal::ElementTimingKey key;
           require(!extract_element_timing_key(nullptr, key) &&
                       !extract_element_timing_key(missing.get(), key),
                   "missing header produced a key");
           std::string error;
           require(tensor_buffer_attach_publication(missing.get(), owner, &error), error);
           require(!extract_element_timing_key(missing.get(), key), "headerless V1 produced a key");
           auto* structure = legacy_header(missing.get(), -1, nullptr);
           require(extract_element_timing_key(missing.get(), key) && key.frame_id == -1 &&
                       key.stream_hash == 0,
                   "legacy absent stream or negative-id behavior changed");
           gst_structure_set(structure, "frame-id", G_TYPE_UINT64, guint64(1), nullptr);
           require(!extract_element_timing_key(missing.get(), key), "wrong frame-id type accepted");
         }));
