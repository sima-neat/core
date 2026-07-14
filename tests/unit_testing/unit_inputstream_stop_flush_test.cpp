#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

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

RUN_TEST("unit_inputstream_stop_flush_test", ([] {
           using namespace simaai::neat;

           gst_init_once();
           EnvVarGuard flush_timeout("SIMA_INPUTSTREAM_STOP_FLUSH_TIMEOUT_MS", "0");

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
           InputStream stream = InputStream::create(pipeline, appsrc, nullptr, make_encoded_spec(),
                                                    input_options, stream_options, {}, nullptr);
           stream.stop();

           require(counts.starts.load(std::memory_order_relaxed) >= 1,
                   "InputStream::stop must send FLUSH_START");
           require(counts.stops.load(std::memory_order_relaxed) == 0,
                   "InputStream::stop must keep the pipeline flushing until NULL");

           stream.close();
         }));
