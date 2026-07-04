#include "gst/GstInit.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kLoanValidField = "neat-latest-mux-loan-valid";
constexpr const char* kLoanNamespaceField = "neat-latest-mux-loan-namespace";
constexpr const char* kLoanStreamIdField = "neat-latest-mux-loan-stream-id";
constexpr const char* kLoanFrameIdField = "neat-latest-mux-loan-frame-id";

struct MuxLoanKey {
  std::uint64_t namespace_id = 0;
  std::string stream_id;
  std::int64_t frame_id = -1;
};

simaai::neat::Sample make_gst_sample_backed_sample(std::optional<MuxLoanKey> key) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 16U, nullptr);
  require(buffer != nullptr, "failed to allocate GstBuffer");

  if (key.has_value()) {
    GstCustomMeta* meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
    require(meta != nullptr, "failed to attach GstSimaMeta");
    GstStructure* s = gst_custom_meta_get_structure(meta);
    require(s != nullptr, "failed to access GstSimaMeta structure");
    gst_structure_set(s, kLoanValidField, G_TYPE_BOOLEAN, TRUE, kLoanNamespaceField,
                      G_TYPE_UINT64, static_cast<guint64>(key->namespace_id),
                      kLoanStreamIdField, G_TYPE_STRING, key->stream_id.c_str(),
                      kLoanFrameIdField, G_TYPE_INT64, static_cast<gint64>(key->frame_id),
                      nullptr);
  }

  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", "width",
                                      G_TYPE_INT, 4, "height", G_TYPE_INT, 4, nullptr);
  require(caps != nullptr, "failed to allocate GstCaps");
  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(gst_sample != nullptr, "failed to allocate GstSample");

  simaai::neat::Tensor tensor;
  tensor.storage = simaai::neat::pipeline_internal::make_gst_sample_storage(gst_sample);
  tensor.dtype = simaai::neat::TensorDType::UInt8;
  tensor.shape = {16};
  gst_sample_unref(gst_sample);

  simaai::neat::Sample sample = simaai::neat::sample_from_tensors({tensor});
  sample.stream_id = "fallback-stream";
  sample.frame_id = 7;
  return sample;
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    const simaai::neat::Sample namespaced =
        make_gst_sample_backed_sample(MuxLoanKey{1234, "mux-stream", 42});
    std::vector<simaai::neat::pipeline_internal::RealtimeFrameCredit> credits =
        simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(namespaced);
    require(credits.size() == 1U, "expected one namespaced realtime credit");
    require(credits[0].namespace_id == 1234U, "credit should preserve mux namespace");
    require(credits[0].stream_id == "mux-stream", "credit should use stamped mux stream id");
    require(credits[0].frame_id == 42, "credit should use stamped mux frame id");

    const simaai::neat::Sample fallback = make_gst_sample_backed_sample(std::nullopt);
    credits = simaai::neat::pipeline_internal::realtime_frame_credits_for_sample(fallback);
    require(credits.size() == 1U, "expected one fallback realtime credit");
    require(credits[0].namespace_id == 0U, "fallback credit should remain unqualified");
    require(credits[0].stream_id == "fallback-stream", "fallback should use sample stream id");
    require(credits[0].frame_id == 7, "fallback should use sample frame id");

    std::cout << "[OK] unit_latest_mux_credit_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
