#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The one form of the error a customer gets for what the Dataflow backend does not carry:
//
//   [<code>] <where>: <subject> is not carried by the Dataflow backend: <fact>.
//   Hint: <action>
//
//   where    the customer's call (Graph::build, Run::push, ...), never an internal function;
//   subject  who is not carried, in the customer's terms: a stage as the route lists it with its
//            kind and target ("stage 2 'detess_dequant_0' (kernel detessdequant on the EV74) of
//            model yolo11n"), a node ("node Custom 'overlay'"), an option, "this frame", "this
//            board";
//   fact     what it is or has, in the present tense, as a clause that reads after "it" or stands
//            alone ("it runs 6 kernel invocations per frame"): no promise, no name of a plan step,
//            no runtime type, no reason phrased as something the runtime lacks;
//   action   dataflow_backend_action(): how to run the program on the GStreamer backend.
//
// The route planner and the graph compiler return a DataflowNotCarried; every thrower goes through
// throw_dataflow_not_carried, so a text cannot leave the form. A reader recognises the outcome by
// the error code and the fixed words "is not carried by the Dataflow backend". Compiled in every
// build of the library, with or without the Dataflow runtime.

#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

// What follows "Hint: ".
enum class DataflowRemedy : unsigned char {
  /// The program selects the GStreamer backend (neat/runtime.h: set_backend). A process runs on
  /// one backend, so this is the action for an error raised on a Run that exists as well.
  Backend,
  /// The board's EV74 does not run the Kernels firmware: the GStreamer backend, or that firmware.
  /// The sentence reads "... is not carried by the Dataflow backend on this board: ...".
  BackendOrFirmware,
};

// What the Dataflow backend does not carry. Empty `subject`: nothing, the answer of a planner that
// carries the route.
struct DataflowNotCarried {
  const char* code = nullptr; ///< an error_codes:: family; null: misconfig.pipeline_shape
  std::string subject;
  std::string fact;
  DataflowRemedy remedy = DataflowRemedy::Backend;

  explicit operator bool() const noexcept {
    return !subject.empty();
  }
};

// "<subject> is not carried by the Dataflow backend: <fact>" (no final period): the sentence of
// the form without its call, for a log line and for a test that reads a planner's answer.
__attribute__((visibility("default")))
std::string dataflow_not_carried_sentence(const DataflowNotCarried& what);

// The action of `remedy`: the one sentence that names how the GStreamer backend is selected, and
// what the remedy adds to it.
__attribute__((visibility("default"))) std::string dataflow_backend_action(DataflowRemedy remedy);

// Throws the NeatError of the form above for `what`, raised by the customer's call `where`.
[[noreturn]] __attribute__((visibility("default")))
void throw_dataflow_not_carried(std::string_view where, const DataflowNotCarried& what);

} // namespace simaai::neat::pipeline_internal
