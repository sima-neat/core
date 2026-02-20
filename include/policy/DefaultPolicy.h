/**
 * @file
 * @ingroup policy
 * @brief Default policy bundle.
 */
#pragma once

#include "policy/DecoderPolicy.h"
#include "policy/EncoderPolicy.h"
#include "policy/MemoryPolicy.h"
#include "policy/RtspPolicy.h"

namespace simaai::neat::policy {

struct DefaultPolicy {
  DecoderPolicy decoder{};
  EncoderPolicy encoder{};
  MemoryPolicy memory{};
  RtspPolicy rtsp{};
};

DefaultPolicy make_default_policy();

} // namespace simaai::neat::policy
