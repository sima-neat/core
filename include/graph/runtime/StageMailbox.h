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

/**
 * @brief Single thread-safe inbox the runtime delivers messages into for one stage.
 *
 * Wraps a `BlockingQueue<StageMsg>` sized at construction. The runtime pushes incoming
 * messages into the mailbox; the stage's worker thread pops and dispatches them.
 *
 * @see BlockingQueue
 * @ingroup graph
 */
struct StageMailbox {
  /// Construct a mailbox with the given inbox capacity (0 = unbounded).
  explicit StageMailbox(std::size_t capacity = 0) : inbox(capacity) {}
  BlockingQueue<StageMsg> inbox; ///< Bounded blocking queue holding pending `StageMsg`s.
};

} // namespace simaai::neat::graph::runtime
