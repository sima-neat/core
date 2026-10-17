#include "yolox_seg_pose_payload_perf_common.h"

// Shipped-config workload: top_k 24 with 8 detections.
int main() {
  return run_payload_scenario_main({"yolox_seg_pose_payload_decode_representative", 8, 24});
}
