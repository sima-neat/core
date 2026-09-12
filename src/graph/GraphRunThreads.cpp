#include "internal/GraphRunState.h"

namespace simaai::neat::graph {
void GraphRun::stop() {
  if (!state_ || !state_->core) {
    return;
  }
  state_->core->stop();
}
} // namespace simaai::neat::graph
