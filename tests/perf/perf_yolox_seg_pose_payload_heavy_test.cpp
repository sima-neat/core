#include "yolox_seg_pose_payload_perf_common.h"

// Worst case at the documented cap: top_k 100 fully populated, ~2.5 MB of masks.
int main() {
  return run_payload_scenario_main({"yolox_seg_pose_payload_decode_heavy", 100, 100});
}
