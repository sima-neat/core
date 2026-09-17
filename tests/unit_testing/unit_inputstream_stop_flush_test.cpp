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
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>
#include <utility>

namespace {

using TestClock = std::chrono::steady_clock;
using namespace std::chrono_literals;

typedef struct _GstTestDelayedNull {
  GstElement parent;
  gint delay_ready_to_null;
} GstTestDelayedNull;

typedef struct _GstTestDelayedNullClass {
  GstElementClass parent_class;
} GstTestDelayedNullClass;

G_DEFINE_TYPE(GstTestDelayedNull, gst_test_delayed_null, GST_TYPE_ELEMENT)

GstStateChangeReturn gst_test_delayed_null_change_state(GstElement* element,
                                                        GstStateChange transition) {
  auto* self = reinterpret_cast<GstTestDelayedNull*>(element);
  if (transition == GST_STATE_CHANGE_READY_TO_NULL &&
      g_atomic_int_get(&self->delay_ready_to_null) != 0) {
    return GST_STATE_CHANGE_FAILURE;
  }
  return GST_ELEMENT_CLASS(gst_test_delayed_null_parent_class)->change_state(element, transition);
}

void gst_test_delayed_null_class_init(GstTestDelayedNullClass* klass) {
  GST_ELEMENT_CLASS(klass)->change_state = gst_test_delayed_null_change_state;
}

void gst_test_delayed_null_init(GstTestDelayedNull* self) {
  g_atomic_int_set(&self->delay_ready_to_null, 1);
}

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
      using pipeline_internal::InputStreamTeardownPolicy;

      gst_init_once();
      EnvVarGuard flush_timeout("SIMA_INPUTSTREAM_STOP_FLUSH_TIMEOUT_MS", "0");
      EnvVarGuard synchronous_no_defer("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0");

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

      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! identity name=cpu ! appsink name=mysink") ==
                  InputStreamTeardownPolicy::Deferred,
              "plain CPU appsrc pipelines must keep deferred teardown");
      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! neatprocessmla name=mla async=true ! appsink name=mysink") ==
                  InputStreamTeardownPolicy::MustReachNull,
              "ProcessMLA appsrc pipelines must use synchronous teardown");
      require(inputstream_pipeline_teardown_policy(
                  "( appsrc name=mysrc ! neatprocesscvu name=cvu async=true ) ! fakesink") ==
                  InputStreamTeardownPolicy::MustReachNull,
              "ProcessCVU appsrc pipelines must use synchronous teardown");
      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! h264parse ! neatdecoder name=decoder ! "
                  "neatencoder name=encoder ! fakesink") ==
                  InputStreamTeardownPolicy::MustReachNull,
              "direct codec pipelines must use synchronous teardown");
      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! neatencoder name=encoder ! fakesink") ==
                  InputStreamTeardownPolicy::MustReachNull,
              "direct encoder pipelines must use synchronous teardown");
      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! identity name=neatprocessmla_probe ! fakesink") ==
                  InputStreamTeardownPolicy::Deferred,
              "an element name containing a driver factory must not change teardown policy");
      require(inputstream_pipeline_teardown_policy(
                  "appsrc name=mysrc ! neatprocessmla_legacy name=mla ! fakesink") ==
                  InputStreamTeardownPolicy::Deferred,
              "a factory prefix match must not change teardown policy");

      const auto run_stop = [](InputStreamTeardownPolicy teardown_policy) {
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
        GstElement* observed_pipeline = GST_ELEMENT(gst_object_ref(pipeline));

        InputOptions input_options;
        InputStreamOptions stream_options;
        stream_options.teardown_policy = teardown_policy;
        InputStream stream = InputStream::create(pipeline, appsrc, nullptr, make_encoded_spec(),
                                                 input_options, stream_options, {}, nullptr);
        stream.stop();

        const int starts = counts.starts.load(std::memory_order_relaxed);
        const int stops = counts.stops.load(std::memory_order_relaxed);
        GstState current = GST_STATE_VOID_PENDING;
        GstState pending = GST_STATE_VOID_PENDING;
        (void)gst_element_get_state(observed_pipeline, &current, &pending, 0);
        gst_object_unref(observed_pipeline);
        stream.close();
        return std::tuple{starts, stops, current, pending};
      };

      const auto legacy = run_stop(InputStreamTeardownPolicy::Deferred);
      require(std::get<0>(legacy) >= 1, "legacy InputStream::stop must send FLUSH_START");
      require(std::get<1>(legacy) == 0,
              "legacy InputStream::stop must keep the pipeline flushing until NULL");

      const auto synchronous = run_stop(InputStreamTeardownPolicy::BoundedPreferred);
      require(std::get<0>(synchronous) == 0,
              "synchronous live InputStream::stop must not send pre-NULL FLUSH_START");
      require(std::get<1>(synchronous) == 0,
              "synchronous live InputStream::stop must not send FLUSH_STOP");
      require(std::get<2>(synchronous) == GST_STATE_NULL,
              "synchronous InputStream::stop must reach NULL before returning");
      require(std::get<3>(synchronous) == GST_STATE_VOID_PENDING ||
                  std::get<3>(synchronous) == GST_STATE_NULL,
              "synchronous InputStream::stop must not leave a non-NULL state transition pending");

      {
        EnvVarGuard force_deferred("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "1");
        const auto mandatory = run_stop(InputStreamTeardownPolicy::MustReachNull);
        require(std::get<2>(mandatory) == GST_STATE_NULL,
                "mandatory driver teardown must ignore the deferred-teardown override");
        require(std::get<3>(mandatory) == GST_STATE_VOID_PENDING ||
                    std::get<3>(mandatory) == GST_STATE_NULL,
                "mandatory driver teardown must not leave a pending non-NULL transition");
      }

      {
        EnvVarGuard short_teardown_timeout("SIMA_GST_TEARDOWN_TIMEOUT_MS", "20");
        EnvVarGuard force_deferred("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "1");
        auto* delayed = GST_ELEMENT(g_object_new(gst_test_delayed_null_get_type(), nullptr));
        require(delayed != nullptr, "failed to allocate delayed-NULL test element");
        require(gst_element_set_state(delayed, GST_STATE_READY) != GST_STATE_CHANGE_FAILURE,
                "failed to put delayed-NULL test element in READY");
        GstElement* observed = GST_ELEMENT(gst_object_ref(delayed));
        GstElement* teardown_owned = delayed;
        std::atomic<bool> teardown_returned{false};
        std::thread teardown_thread([&]() {
          pipeline_internal::stop_and_unref_no_flush(
              teardown_owned, InputStreamTeardownPolicy::MustReachNull);
          teardown_returned.store(true, std::memory_order_release);
        });

        const auto deadline = TestClock::now() + 100ms;
        while (!teardown_returned.load(std::memory_order_acquire) &&
               TestClock::now() < deadline) {
          std::this_thread::sleep_for(1ms);
        }
        const bool returned_while_non_null =
            teardown_returned.load(std::memory_order_acquire);

        auto* delayed_state = reinterpret_cast<GstTestDelayedNull*>(observed);
        g_atomic_int_set(&delayed_state->delay_ready_to_null, 0);
        (void)gst_element_continue_state(observed, GST_STATE_CHANGE_SUCCESS);
        teardown_thread.join();

        GstState current = GST_STATE_VOID_PENDING;
        GstState pending = GST_STATE_VOID_PENDING;
        (void)gst_element_get_state(observed, &current, &pending, 0);
        gst_object_unref(observed);
        require(!returned_while_non_null,
                "mandatory driver teardown must retain ownership while NULL transition fails");
        require(teardown_owned == nullptr,
                "mandatory driver teardown must consume the caller's pipeline reference");
        require(current == GST_STATE_NULL,
                "mandatory driver teardown must return only after reaching NULL");
        require(pending == GST_STATE_VOID_PENDING || pending == GST_STATE_NULL,
                "mandatory driver teardown must not return with a non-NULL state pending");
      }
    }));
