#pragma once

#include "builder/OutputSpec.h"
#include "pipeline/TensorCore.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::debug {

struct DebugOptions {
  int timeout_ms = 10000;
  bool strict = false;
  bool force_system_memory = true;
};

struct DebugOutput {
  OutputSpec expected;
  OutputSpec observed;
  std::string caps_string;
  bool tensorizable = false;
  bool unknown = false;
  std::vector<std::string> warnings;
  std::optional<simaai::neat::Tensor> tensor; // owning tensor (when available)
  std::vector<uint8_t> bytes;                 // raw bytes for non-tensorizable outputs
};

struct DebugStream {
  std::function<std::optional<DebugOutput>(int)> next;
  std::function<void()> close;
  OutputSpec expected;
  OutputSpec observed;
  std::string caps_string;
  bool tensorizable = false;
  bool unknown = false;
  std::shared_ptr<void> state; // keeps underlying pipeline/stream alive

  explicit operator bool() const {
    return static_cast<bool>(next);
  }
};

struct OutputTag {};
struct StreamTag {};

inline constexpr OutputTag output{};
inline constexpr StreamTag stream{};

} // namespace simaai::neat::debug
