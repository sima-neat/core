// Empirical proof for the "rvalue move left undeleted => segfault?" debate.
//
// Reproduces the disputed snippet shape with a REAL source-mode pipeline::Graph
// (videotestsrc, no model needed) and answers two questions on the board:
//
//   Scenario A  (Manuel's exact snippet):  build(); Graph c = (Graph&&)g; c.run();
//               run() is called on the MOVE TARGET c, which owns every resource.
//               Expectation: works, produces a frame, no segfault. The move is
//               transparent because c is a faithful transfer of g's state.
//
//   Scenario B  (the genuinely-correct use-after-move): move g2 -> c2, then call
//               methods on the MOVED-FROM g2. Expectation: clean C++ exception
//               (e.g. "cannot compile a moved-from composition"), NOT a SIGSEGV,
//               because Graph hardens its moved-from state the same way Run does.
//
// The program returns 0 only if it reaches the end WITHOUT crashing. A segfault
// would terminate with signal 11 (exit 139) and never print "[DONE]".

#include "pipeline/Graph.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

// NOTE: do NOT call gst_init() here. Graph::run()/build() call
// simaai::neat::gst_init_once() internally; a manual gst_init() trips the
// framework's "already initialized" guard and would mask the real behavior.

using namespace simaai::neat::nodes;

namespace {

// Build a fresh source-mode Graph that emits a single test frame and stops.
simaai::neat::Graph make_source_graph(bool& got_frame) {
  simaai::neat::Graph g;
  g.custom("videotestsrc num-buffers=1", simaai::neat::InputRole::Source);
  g.add(VideoConvert());
  g.add(CapsNV12SysMem(64, 64, 30));
  g.add(Output());
  g.set_tensor_callback([&got_frame](const simaai::neat::Tensor&) {
    got_frame = true;
    return false; // stop after the first frame
  });
  return g; // returned by value -> exercises move construction itself
}

} // namespace

int main() {
  int failures = 0;

  // ---------------------------------------------------------------------------
  // SCENARIO A — the exact snippet: run() on the MOVE TARGET.
  // ---------------------------------------------------------------------------
  try {
    bool got_frame = false;
    simaai::neat::Graph g = make_source_graph(got_frame);

    // Matches the snippet: build() returns a Run by value; it is discarded here.
    (void)g.build();

    // "Allowed cause of rvalue move semantics."
    simaai::neat::Graph c = (simaai::neat::Graph&&)g;

    // run() on c — the object that RECEIVED every resource. Should just work.
    c.run();

    std::cout << "[A] c.run() on move-TARGET returned normally; got_frame="
              << (got_frame ? "true" : "false") << "\n";
    if (!got_frame) {
      std::cout << "[A] (note) no frame captured, but importantly: no crash\n";
    }
  } catch (const std::exception& e) {
    // A clean throw here is still "not a segfault"; report it.
    std::cout << "[A] c.run() threw (clean, not a crash): " << e.what() << "\n";
  }

  // ---------------------------------------------------------------------------
  // SCENARIO B — the genuinely-correct use-after-move: use the MOVED-FROM object.
  // ---------------------------------------------------------------------------
  {
    bool got_frame2 = false;
    simaai::neat::Graph g2 = make_source_graph(got_frame2);

    // Move g2 away. g2 is now a moved-from husk (composition_ == nullptr).
    simaai::neat::Graph c2 = (simaai::neat::Graph&&)g2;

    // (B1) Call build() on the MOVED-FROM g2. This is the real use-after-move.
    bool b1_threw = false;
    std::string b1_msg;
    try {
      (void)g2.build();
    } catch (const std::exception& e) {
      b1_threw = true;
      b1_msg = e.what();
    }
    std::cout << "[B1] moved-from g2.build() threw=" << (b1_threw ? "true" : "false") << " msg=\""
              << b1_msg << "\"\n";
    if (!b1_threw) {
      std::cout << "[B1] FAIL: expected a clean throw from moved-from build()\n";
      ++failures;
    }

    // (B2) Call run() on the MOVED-FROM g2 as well.
    bool b2_threw = false;
    std::string b2_msg;
    try {
      g2.run();
    } catch (const std::exception& e) {
      b2_threw = true;
      b2_msg = e.what();
    }
    std::cout << "[B2] moved-from g2.run() threw=" << (b2_threw ? "true" : "false") << " msg=\""
              << b2_msg << "\"\n";
    if (!b2_threw) {
      std::cout << "[B2] FAIL: expected a clean throw from moved-from run()\n";
      ++failures;
    }

    // (B3) The move TARGET c2 must still be fully usable.
    try {
      c2.run();
      std::cout << "[B3] move-TARGET c2.run() returned normally; got_frame="
                << (got_frame2 ? "true" : "false") << "\n";
    } catch (const std::exception& e) {
      std::cout << "[B3] c2.run() threw (clean, not a crash): " << e.what() << "\n";
    }
  }

  // Reaching here at all proves there was no SIGSEGV anywhere above.
  std::cout << "[DONE] reached end of program WITHOUT a segfault (exit " << (failures == 0 ? 0 : 1)
            << ")\n";
  return failures == 0 ? 0 : 1;
}
