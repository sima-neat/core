/**
 * @file
 * @ingroup graph_runtime
 * @brief Single inbox per stage node.
 */
#pragma once

#include "graph/StageExecutor.h"
#include "graph/runtime/BlockingQueue.h"

#include <cstddef>

namespace simaai::neat::graph::runtime {

struct StageMailbox {
  explicit StageMailbox(std::size_t capacity = 0) : inbox(capacity) {}
  BlockingQueue<StageMsg> inbox;
};

} // namespace simaai::neat::graph::runtime
