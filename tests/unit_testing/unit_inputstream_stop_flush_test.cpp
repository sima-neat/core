#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/GstTeardownBudget.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key) {
    if (const char* current = std::getenv(key_)) {
      had_value_ = true;
      old_value_ = current;
    }
    ::setenv(key_, value, 1);
  }

  ~EnvVarGuard() {
    if (had_value_) {
      ::setenv(key_, old_value_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_value_ = false;
  std::string old_value_;
};

struct FlushCounts {
  std::atomic<int> starts{0};
  std::atomic<int> stops{0};
};

GstPadProbeReturn count_flush_events(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* counts = static_cast<FlushCounts*>(user_data);
  GstEvent* event = GST_PAD_PROBE_INFO_EVENT(info);
  if (!counts || !event) {
    return GST_PAD_PROBE_OK;
  }
  if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_START) {
    counts->starts.fetch_add(1, std::memory_order_relaxed);
  } else if (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP) {
    counts->stops.fetch_add(1, std::memory_order_relaxed);
  }
  return GST_PAD_PROBE_OK;
}

simaai::neat::SampleSpec make_encoded_spec() {
  simaai::neat::SampleSpec spec;
  spec.kind = simaai::neat::SampleMediaKind::Encoded;
  spec.caps_string = "application/octet-stream";
  spec.caps_key = simaai::neat::capkey_from_spec(spec);
  return spec;
}

} // namespace

RUN_TEST(
    "unit_inputstream_stop_flush_test", ([] {
      using namespace simaai::neat;

      gst_init_once();
      EnvVarGuard flush_timeout("SIMA_INPUTSTREAM_STOP_FLUSH_TIMEOUT_MS", "0");

      using simaai::neat::pipeline_internal::synchronous_live_teardown_budget_ms;
      constexpr std::uint64_t kDefaultRtspTeardownNs = 100'000'000ULL;
      require(synchronous_live_teardown_budget_ms(2000, 0, 0) == 2000,
              "non-RTSP teardown must retain the existing 2s budget");
      const int streams24_budget =
          synchronous_live_teardown_budget_ms(2000, 24ULL * kDefaultRtspTeardownNs, 24);
      require(streams24_budget == 4650, "24 RTSP sources must add 2.4s plus the scheduling margin");
      const int streams48_budget =
          synchronous_live_teardown_budget_ms(2000, 48ULL * kDefaultRtspTeardownNs, 48);
      require(streams48_budget == 7050, "48 RTSP sources must add 4.8s plus the scheduling margin");
      require(synchronous_live_teardown_budget_ms(2000, 1, 1) == 2251,
              "sub-millisecond RTSP timeouts must round up rather than under-budget");
      require(synchronous_live_teardown_budget_ms(2000, std::numeric_limits<std::uint64_t>::max(),
                                                  1) == 30000,
              "pathological RTSP teardown budgets must be capped at 30s");

      const auto pipeline_budget = [](std::size_t rtsp_sources) {
        GstElement* pipeline = gst_pipeline_new(nullptr);
        require(pipeline != nullptr, "expected teardown-budget test pipeline");
        for (std::size_t i = 0; i < rtsp_sources; ++i) {
          GstElement* source = gst_element_factory_make("rtspsrc", nullptr);
          require(source != nullptr, "rtspsrc is required for teardown-budget test");
          // Keep this explicit so the test follows the effective property
          // rather than depending on the host GStreamer package default.
          g_object_set(source, "teardown-timeout", kDefaultRtspTeardownNs, nullptr);
          require(gst_bin_add(GST_BIN(pipeline), source),
                  "failed to add rtspsrc to teardown-budget test pipeline");
        }
        const int budget =
            simaai::neat::pipeline_internal::effective_synchronous_teardown_timeout_ms(pipeline,
                                                                                       2000);
        gst_object_unref(pipeline);
        return budget;
      };
      require(pipeline_budget(0) == 2000,
              "a normal graph without RTSP sources must keep the 2s budget");
      require(pipeline_budget(24) == streams24_budget,
              "24-source graph must include every rtspsrc TEARDOWN timeout");
      require(pipeline_budget(48) == streams48_budget,
              "48-source graph must include every rtspsrc TEARDOWN timeout");

      const auto run_stop = [](bool prefer_synchronous_teardown) {
        GError* error = nullptr;
        GstElement* pipeline = gst_parse_launch(
            "appsrc name=mysrc is-live=true format=time "
            "caps=application/octet-stream ! identity name=probe ! fakesink sync=false",
            &error);
        if (error) {
          const std::string message =
              error->message ? error->message : "failed to build stop-flush test pipeline";
          g_error_free(error);
          throw std::runtime_error(message);
        }
        require(pipeline != nullptr, "expected a valid stop-flush test pipeline");

        GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
        GstElement* identity = gst_bin_get_by_name(GST_BIN(pipeline), "probe");
        require(appsrc != nullptr && identity != nullptr,
                "stop-flush test pipeline elements are missing");

        FlushCounts counts;
        GstPad* sink_pad = gst_element_get_static_pad(identity, "sink");
        require(sink_pad != nullptr, "identity sink pad is missing");
        gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_EVENT_FLUSH, count_flush_events, &counts,
                          nullptr);
        gst_object_unref(sink_pad);
        gst_object_unref(identity);

        require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                "failed to start stop-flush test pipeline");

        InputOptions input_options;
        InputStreamOptions stream_options;
        stream_options.prefer_synchronous_teardown = prefer_synchronous_teardown;
        InputStream stream = InputStream::create(pipeline, appsrc, nullptr, make_encoded_spec(),
                                                 input_options, stream_options, {}, nullptr);
        stream.stop();

        const int starts = counts.starts.load(std::memory_order_relaxed);
        const int stops = counts.stops.load(std::memory_order_relaxed);
        stream.close();
        return std::pair{starts, stops};
      };

      const auto legacy = run_stop(false);
      require(legacy.first >= 1, "legacy InputStream::stop must send FLUSH_START");
      require(legacy.second == 0,
              "legacy InputStream::stop must keep the pipeline flushing until NULL");

      const auto synchronous = run_stop(true);
      require(synchronous.first == 0,
              "synchronous live InputStream::stop must not send pre-NULL FLUSH_START");
      require(synchronous.second == 0,
              "synchronous live InputStream::stop must not send FLUSH_STOP");
    }));
