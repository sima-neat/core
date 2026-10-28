#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <simaai/scheduler/client/kernel.h>

namespace simaai::neat::pipeline_internal::sima {

// A compact one-byte, full-channel MLA cell tile is ordinary NHWC storage
// when each cell satisfies tile alignment. A whole-volume tile is equivalent
// too. Other tile orders/padding, and BF16 low/high byte planes, are not dense.
inline bool
kernel_tensor_requires_detessellation(const ::simaai::scheduler::client::TensorSpec& tensor) {
  namespace rt = ::simaai::scheduler::client;
  const auto* tiled = std::get_if<rt::MlaTiled>(&tensor.layout);
  if (!tiled)
    return std::holds_alternative<rt::MlaCBlocked>(tensor.layout);
  if ((tensor.dtype != rt::DType::Int8 && tensor.dtype != rt::DType::UInt8) ||
      tiled->channels != rt::MlaTiled::Channels::Compact || tensor.shape.size() != 4 ||
      tiled->axes != std::array<std::int8_t, 4>{0, 1, 2, 3} ||
      tiled->tile_shape[3] != tensor.shape[3])
    return true;
  bool cell = true;
  bool whole = true;
  for (std::size_t i = 0; i < 3; ++i) {
    cell = cell && tiled->tile_shape[i] == 1;
    whole = whole && tiled->tile_shape[i] >= tensor.shape[i];
  }
  return !whole && !(cell && tensor.shape[3] % rt::MlaTiled::tile_alignment_bytes == 0);
}

} // namespace simaai::neat::pipeline_internal::sima
