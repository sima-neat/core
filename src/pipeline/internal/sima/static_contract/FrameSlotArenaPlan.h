#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct PhysicalExecutionPlan;

enum class ArenaStorageDomain : std::uint8_t {
  Unknown = 0,
  Cma = 1,
  Dms = 2,
  HostCpu = 3,
  Mixed = 4,
