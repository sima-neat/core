// Core graph/Sample policy adapter; Runtime owns the Gst mux and loan registry.
#include "pipeline/internal/RealtimeFrameCredit.h"
#include "pipeline/internal/TensorUtil.h"

#include <algorithm>
#include <gst/gst.h>
#include <simaai/runtime/gstreamer/elements.h>

namespace simaai::neat::pipeline_internal {

std::vector<RealtimeFrameCredit> realtime_frame_credits_for_sample(const Sample& sample) {
  std::vector<RealtimeFrameCredit> credits;
  auto add_credit = [&credits](const RealtimeFrameCredit& credit) {
    if ((credit.frame_id < 0 || credit.stream_id.empty()) && credit.input_seq < 0 &&
        credit.orig_input_seq < 0) {
      return;
    }
    const auto found =
        std::find_if(credits.begin(), credits.end(), [&](const RealtimeFrameCredit& existing) {
          return existing.namespace_id == credit.namespace_id &&
                 existing.frame_id == credit.frame_id && existing.stream_id == credit.stream_id &&
                 existing.input_seq == credit.input_seq &&
                 existing.orig_input_seq == credit.orig_input_seq &&
                 existing.graph_private == credit.graph_private;
        });
    if (found == credits.end()) {
      credits.push_back(credit);
    }
  };
  for (const auto& credit : attached_realtime_frame_credits_from_sample(sample)) {
    add_credit(credit);
  }
  auto add_tensor_key = [&](const Tensor& tensor) {
    // Latest-by-stream mux credits are registered under the mux instance namespace stamped on
    // GstSimaMeta. Prefer that exact key so drops/stops can release credits unambiguously when
    // multiple muxes in the process reuse the same stream/frame ids.
    if (!tensor.storage || tensor.storage->kind != StorageKind::GstSample) {
      return false;
    }
    const std::shared_ptr<void> holder = holder_from_tensor(tensor);
    GstBuffer* buffer = holder ? buffer_from_tensor_holder(holder) : nullptr;
    if (!buffer) {
      return false;
    }
    simaai::runtime::gstreamer::MuxLoanKey key;
    const bool found = simaai::runtime::gstreamer::read_mux_loan_key(buffer, &key);
    gst_buffer_unref(buffer);
    if (found) {
      add_credit(RealtimeFrameCredit{key.namespace_id, key.stream_id, key.frame_id, key.input_seq,
                                     key.orig_input_seq});
    }
    return found;
  };
  auto collect_stamped_keys = [&](auto&& self, const Sample& s) -> bool {
    bool found_stamped_key = false;
    if (s.tensor) {
      found_stamped_key = add_tensor_key(*s.tensor) || found_stamped_key;
    }
    for (const auto& tensor : s.tensors) {
      found_stamped_key = add_tensor_key(tensor) || found_stamped_key;
    }
    for (const auto& field : s.fields) {
      found_stamped_key = self(self, field) || found_stamped_key;
    }
    return found_stamped_key;
  };
  const bool has_stamped_latest_mux_key = collect_stamped_keys(collect_stamped_keys, sample);
  auto add_fallback_key = [&](const Sample& s) {
    add_credit(RealtimeFrameCredit{0, s.stream_id, s.frame_id, s.input_seq, s.orig_input_seq});
  };
  auto walk = [&](auto&& self, const Sample& s) -> void {
    add_fallback_key(s);
    for (const auto& field : s.fields) {
      self(self, field);
    }
  };
  if (!has_stamped_latest_mux_key) {
    walk(walk, sample);
  }
  return credits;
}

bool retain_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                   std::size_t extra_refs, const char* mode) {
  bool retained_any = retain_registered_realtime_frame_credits(credits, extra_refs, mode);
  if (credits.empty() || extra_refs == 0U) {
    return retained_any;
  }
  for (const auto& credit : credits) {
    if (credit.graph_private || credit.namespace_id == 0) {
      continue;
    }
    retained_any = simaai::runtime::gstreamer::retain_mux_loan(
                       simaai::runtime::gstreamer::MuxLoanKey{credit.namespace_id, credit.stream_id,
                                                              credit.frame_id, credit.input_seq,
                                                              credit.orig_input_seq},
                       extra_refs, mode) ||
                   retained_any;
  }
  return retained_any;
}

void release_realtime_frame_credits_impl(const std::vector<RealtimeFrameCredit>& credits,
                                         const char* mode, bool by_output) {
  for (const auto& credit : credits) {
    if (credit.graph_private) {
      (void)release_registered_realtime_frame_credit(credit, mode, by_output);
      continue;
    }
    if (credit.namespace_id != 0) {
      /*
       * Non-zero keys collected from GstSimaMeta are latest-by-stream mux loan
       * keys, not C++ realtime-credit registry namespaces. Keep those domains
       * separate: the graph credit, when present, is carried as an exact
       * graph-private sidecar credit and was handled above.
       */
      (void)simaai::runtime::gstreamer::release_mux_loan(
          simaai::runtime::gstreamer::MuxLoanKey{credit.namespace_id, credit.stream_id,
                                                 credit.frame_id, credit.input_seq,
                                                 credit.orig_input_seq},
          mode);
      continue;
    }
    if (release_registered_realtime_frame_credit(credit, mode, by_output)) {
      continue;
    }
    if (!credit.stream_id.empty() && credit.frame_id >= 0) {
      (void)simaai::runtime::gstreamer::release_mux_loan(
          simaai::runtime::gstreamer::MuxLoanKey{credit.namespace_id, credit.stream_id,
                                                 credit.frame_id, credit.input_seq,
                                                 credit.orig_input_seq},
          mode);
    } else if (!credit.stream_id.empty() && (credit.input_seq >= 0 || credit.orig_input_seq >= 0)) {
      (void)simaai::runtime::gstreamer::release_mux_loan(
          simaai::runtime::gstreamer::MuxLoanKey{credit.namespace_id, credit.stream_id,
                                                 credit.frame_id, credit.input_seq,
                                                 credit.orig_input_seq},
          mode);
    }
  }
}

void release_realtime_frame_credits(const std::vector<RealtimeFrameCredit>& credits,
                                    const char* mode) {
  release_realtime_frame_credits_impl(credits, mode, /*by_output=*/true);
}

void release_realtime_frame_credits_without_output(const std::vector<RealtimeFrameCredit>& credits,
                                                   const char* mode) {
  release_realtime_frame_credits_impl(credits, mode, /*by_output=*/false);
}

void release_realtime_frame_credits_for_sample(const Sample& sample, const char* mode) {
  release_realtime_frame_credits(realtime_frame_credits_for_sample(sample), mode);
}

} // namespace simaai::neat::pipeline_internal
