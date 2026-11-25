// No RTSP credentials in what the Dataflow backend says about a graph (owner's decision 38): a
// graph whose source's URL holds `user:password@` is refused at Graph::build on Dataflow in the one
// not-carried form, and neither the error's text nor its report (every field, as to_json renders
// it) holds the user or the password; the source is named `rtsp://<redacted>@host...`, as
// GStreamer's own errors print a source. Nothing connects: the refusal comes before anything is
// opened, so no server is needed.
//
// Registered with --backend=dataflow only: GStreamer's texts are its own
// (Graph::describe_backend() and Graph::last_pipeline() keep the launch string as it is).

#include "nodes/common/Output.h"
#include "nodes/io/RTSPInput.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "support/backend_selection.h"
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

} // namespace

RUN_BACKEND_TEST("unit_dataflow_credentials_test", ([] {
                   if (!sima_test::on_dataflow()) {
                     return;
                   }
                   refusal_names_the_source_without_credentials();
                 }));
