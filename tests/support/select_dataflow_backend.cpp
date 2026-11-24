// Linked into the second executable of a test whose sources stay unchanged (tests/CMakeLists.txt,
// "the H0-scope tests a second time"): the process runs on the Dataflow backend, selected before
// main() as an application's first call would select it (neat/runtime.h, set_backend).

#include "neat/runtime.h"

namespace {

[[maybe_unused]] const bool selected = [] {
  simaai::neat::set_backend(simaai::neat::RunBackend::Dataflow);
  return true;
}();

} // namespace
