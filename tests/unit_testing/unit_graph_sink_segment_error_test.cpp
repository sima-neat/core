// A graph branch that ends in its own sink (as VideoSender ends in udpsink) has no Output to
// pull. When an element in that branch fails, the graph must stop with that element's error
// instead of running on with the branch silently dead. The branch's bus watch also races graph
// stop, which releases the branch pipeline before the watch thread is joined.
#include "gst/GstInit.h"
#include "internal/InputStream.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <atomic>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;

const std::string kCaps = "image/jpeg,width=(int)16,height=(int)16,framerate=(fraction)30/1";

InputOptions encoded_input_options() {
  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Encoded;
  src_opt.format = FormatTag::ENCODED;
  src_opt.caps_override = kCaps;
  src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
  return src_opt;
}

Sample encoded_frame(int frame) {
  return make_encoded_sample(std::vector<uint8_t>{0xff, 0xd8, 0xff, 0xd9}, kCaps,
                             frame * 33333333LL, -1, 33333333LL);
}

// The watch may only touch a pipeline reference it owns: stop() must not free a pipeline the
// watch acquired, and a stopped stream must not hand out its released pipeline.
void check_acquired_pipeline_ref_outlives_stop() {
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(
      "appsrc name=src format=time caps=application/octet-stream ! fakesink sync=false", &error);
  if (error) {
    const std::string message = error->message ? error->message : "pipeline parse failed";
    g_error_free(error);
    throw std::runtime_error(message);
  }
  require(pipeline != nullptr, "expected a valid pipeline");
  GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "src");
  require(appsrc != nullptr, "appsrc is missing");
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "failed to start the pipeline");

  SampleSpec spec;
  spec.kind = SampleMediaKind::Encoded;
  spec.caps_string = "application/octet-stream";
  spec.caps_key = capkey_from_spec(spec);
  InputStreamOptions stream_opt;
  stream_opt.teardown_policy = pipeline_internal::InputStreamTeardownPolicy::MustReachNull;
  InputStream stream =
      InputStream::create(pipeline, appsrc, nullptr, spec, InputOptions{}, stream_opt, {}, nullptr);

  GstElement* owned = stream.acquire_pipeline_ref();
  require(owned == pipeline, "a running stream must hand out its pipeline");
  static std::atomic<bool> finalized{false};
  g_object_weak_ref(G_OBJECT(owned), [](gpointer, GObject*) { finalized.store(true); }, nullptr);
  stream.stop();
  require(!finalized.load(), "stop() must not free a pipeline the watch acquired");
  GstState state = GST_STATE_VOID_PENDING;
  (void)gst_element_get_state(owned, &state, nullptr, 0);
  require(state == GST_STATE_NULL, "stop() must still tear the pipeline down");
  gst_object_unref(owned);
  require(stream.acquire_pipeline_ref() == nullptr,
          "a stopped stream must not hand out its released pipeline");
  stream.close();
  require(stream.acquire_pipeline_ref() == nullptr, "a closed stream has no pipeline");
}

// Stop while the sink-only branch's watch is starting: graph stop releases the branch pipeline
// before it joins the watch, so a watch holding a borrowed pointer reads freed memory.
void race_graph_stop_with_sink_segment_watch() {
  for (int iter = 0; iter < 200; ++iter) {
    Graph source("sink_race_source");
    source.add(nodes::Input(encoded_input_options()));
    Graph pulled("sink_race_output");
    pulled.add(nodes::Output());
    Graph sink_only("sink_race_branch");
    sink_only.add(nodes::Custom("identity ! fakesink sync=false async=false"));
    Graph split("sink_race_split");
    split.connect(source, pulled);
    split.connect(source, sink_only);
    Run run = split.build();
    if (iter % 2 == 0) {
      require(run.push(encoded_frame(0)), "the first push must be accepted");
    }
    run.stop();
  }
}

} // namespace

int main() {
  try {
    gst_init_once();
    check_acquired_pipeline_ref_outlives_stop();
    race_graph_stop_with_sink_segment_watch();
    const InputOptions src_opt = encoded_input_options();
    Graph source("sink_error_source");
    source.add(nodes::Input(src_opt));
    Graph pulled("sink_error_output");
    pulled.add(nodes::Output());
    Graph failing("sink_error_branch");
    failing.add(nodes::Custom("identity error-after=3 ! fakesink sync=false async=false"));
    Graph split("sink_error_split");
    split.connect(source, pulled);
    split.connect(source, failing);
    Run run = split.build();

    bool stopped = false;
    for (int frame = 0; frame < 200 && !stopped; ++frame) {
      try {
        if (!run.push(encoded_frame(frame)) || !run.pull(1000).has_value()) {
          stopped = true;
        }
      } catch (const std::exception&) {
        stopped = true;
      }
    }
    const std::string error = run.last_error();
    std::cout << "[INFO] last_error: " << error << "\n";
    require(stopped, "the graph kept running after its sink-only branch failed");
    require(error.find("identity") != std::string::npos,
            "the graph error should name the failing sink-only branch element");
    run.stop();
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_graph_sink_segment_error_test\n";
  return 0;
}
