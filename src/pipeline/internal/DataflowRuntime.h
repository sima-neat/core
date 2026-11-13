#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#if !defined(SIMANEAT_WITH_DATAFLOW)
#error "DataflowRuntime.h needs a build configured with SIMANEAT_DATAFLOW_SOURCE_DIR."
#endif

// The one place Core includes the Neat Dataflow runtime (SIMANEAT_DATAFLOW_SOURCE_DIR). Every Core
// file reaches the runtime through this header and no other.
//
// libsima_neat.so embeds the runtime's objects with hidden visibility. A Core TU is compiled with
// default visibility, so without the pragma below the runtime's inline functions and the templates
// a Core TU instantiates over runtime types would be emitted as WEAK DEFAULT symbols and exported
// from libsima_neat.so (interposable across DSOs, and a second definition beside the hidden one).
// Under the pragma they are WEAK HIDDEN. The standard headers the runtime's headers use are
// included first, outside the pragma, so no libc or libstdc++ declaration is first seen hidden.

#include <array>
#include <cassert>
#include <chrono>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <atomic>

#pragma GCC visibility push(hidden)
#include "dataflow/dataflow.h"
#if defined(SIMANEAT_WITH_DATAFLOW_DEVICE_MEMORY)
// H0b: the Runner-backed source of device pools (dataflow::runner_memory). It names no Runner type.
#include "dataflow/runner/block_source.h"
#endif
#pragma GCC visibility pop

namespace simaai::neat::pipeline_internal {
namespace df = ::simaai::dataflow;
} // namespace simaai::neat::pipeline_internal
