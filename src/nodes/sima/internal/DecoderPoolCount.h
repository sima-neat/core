#pragma once

#include <stdexcept>
#include <string>

namespace simaai::neat::internal {

// simaaidecoder pool range: gstsimaaidecoder.cpp:1679-1693 (pspecs), :2163-2172 (READY check).
inline void require_decoder_pool_count(const char* field, int value) {
  if (value > 0 && (value < 2 || value > 64)) {
    throw std::invalid_argument(std::string(field ? field : "<unknown>") +
                                " must be in 2..64 or <= 0 (element default), got " +
                                std::to_string(value));
  }
}

} // namespace simaai::neat::internal
