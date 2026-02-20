/**
 * @file
 * @ingroup tensors
 * @brief Conversion vocabulary, policies, and tracing for Tensor.
 */
#pragma once

#include "pipeline/TensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

enum class ConversionKind {
  Reinterpret = 0,
  View,
  Pack,
  Convert,
  Transfer,
};

enum class ConversionPolicy {
  Strict = 0,
  AllowWithTrace,
  AllowSilent,
};

struct ConversionCost {
  std::uint64_t bytes_copied = 0;
  int compute_class = 0; // 0=low, 1=med, 2=high
};

struct ConversionTrace {
  std::string stage;
  ConversionKind kind = ConversionKind::Reinterpret;
  std::string src_desc;
  std::string dst_desc;
  std::uint64_t bytes_copied = 0;
  std::uint64_t elapsed_us = 0;
  ConversionPolicy policy = ConversionPolicy::Strict;
};

struct ConversionTraceCollector {
  std::vector<ConversionTrace> traces;

  void add(ConversionTrace trace) {
    traces.push_back(std::move(trace));
  }
  void clear() {
    traces.clear();
  }
};

ConversionCost estimate_conversion_cost(ConversionKind kind, std::uint64_t bytes_copied);
bool conversion_allowed(ConversionPolicy policy, ConversionKind kind);

} // namespace simaai::neat
