// No RTSP credentials in what the Dataflow backend says about a graph (owner's decision 38): a
// graph whose source's URL holds `user:password@` is refused at Graph::build on Dataflow in the one
// not-carried form, and neither the error's text nor its report (every field, as to_json renders
// it) holds the user or the password; the source is named `rtsp://<redacted>@host...`, as
// GStreamer's own errors print a source. Nothing connects: the refusal comes before anything is
// opened, so no server is needed.
//
// Graph::describe() (DECISIONS.md 43) names the source the same way on both backends. On Dataflow,
// Graph::describe_backend() (what GStreamer's build of the Graph would run) holds no credentials
// either and is made without starting GStreamer (support/gstreamer_footprint.h).
//
// Registered with --backend=dataflow and --backend=gstreamer: the refusal and the
// describe_backend() rows on Dataflow; on GStreamer, describe() redacted and describe_backend() its
// own launch string as it is (the URL in it unchanged).

#include "nodes/common/Output.h"
#include "nodes/io/RTSPInput.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "support/backend_selection.h"
#include "support/gstreamer_footprint.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

using namespace simaai::neat;

constexpr const char* kUser = "camuser";
constexpr const char* kPassword = "secret";
constexpr const char* kUrl = "rtsp://camuser:secret@127.0.0.1:18554/cam";
constexpr const char* kRedacted = "rtsp://<redacted>@127.0.0.1:18554/cam";

void refusal_names_the_source_without_credentials() {
  Graph g;
  g.add(nodes::RTSPInput(kUrl));
  g.add(nodes::Output());
  bool refused = false;
  std::string what;
  std::string report;
  try {
    (void)g.build(RunOptions{});
  } catch (const NeatError& e) {
    refused = true;
    what = e.what();
    report = e.report().to_json();
  }
  require(refused, "a graph with an RTSP source is refused on Dataflow");
  require_contains(what, "is not carried by the Dataflow backend", "the one not-carried form");
  require_contains(what, kRedacted, "the source named with its credentials removed");
  for (const std::string& text : {what, report}) {
    require(text.find(kPassword) == std::string::npos, "the password appears in: " + text);
    require(text.find(kUser) == std::string::npos, "the user appears in: " + text);
  }
}

Graph rtsp_graph() {
  Graph g;
  g.add(nodes::RTSPInput(kUrl));
  g.add(nodes::Output());
  return g;
}

void require_no_credentials(const std::string& text, const std::string& what) {
  require(text.find(kPassword) == std::string::npos, what + ": the password appears in: " + text);
  require(text.find(kUser) == std::string::npos, what + ": the user appears in: " + text);
}

// Graph::describe() on either backend: the source named without its credentials.
void describe_names_the_source_without_credentials() {
  const std::string text = rtsp_graph().describe();
  require_contains(text, "RTSPInput", "describe() lists the source");
  require_no_credentials(text, "describe()");
}

// Graph::describe_backend() on Dataflow: GStreamer's launch string for the Graph, its source's
// location redacted, and GStreamer never started for it.
void describe_backend_on_dataflow_without_credentials() {
  const std::string text = rtsp_graph().describe_backend();
  require_contains(text, "rtspsrc", "describe_backend() is the launch string");
  require_contains(text, kRedacted, "the location redacted");
  require_no_credentials(text, "describe_backend()");
  const sima_test::GStreamerFootprint gst = sima_test::gstreamer_footprint();
  require(gst.empty(), "describe_backend() on Dataflow left GStreamer " + gst.text());
}

// Graph::describe_backend() on GStreamer: its own launch string, unchanged.
void describe_backend_on_gstreamer_as_it_is() {
  require_contains(rtsp_graph().describe_backend(), kUrl, "the location as given");
}

} // namespace

RUN_BACKEND_TEST("unit_dataflow_credentials_test", ([] {
                   describe_names_the_source_without_credentials();
                   if (!sima_test::on_dataflow()) {
                     describe_backend_on_gstreamer_as_it_is();
                     return;
                   }
                   describe_backend_on_dataflow_without_credentials();
                   refusal_names_the_source_without_credentials();
                 }));
