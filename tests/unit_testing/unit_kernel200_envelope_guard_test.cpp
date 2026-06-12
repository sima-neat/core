#include "pipeline/internal/sima/Kernel200EnvelopeGuard.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_kernel200_envelope_guard_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           Kernel200EnvelopeLimits limits;
           limits.max_w = 1280;
           limits.max_h = 720;
           limits.max_stride = 2048;
           limits.allocated_bytes = 3 * 1280 * 720;

           Kernel200EnvelopeActual ok;
           ok.actual_w = 640;
           ok.actual_h = 360;
           ok.actual_stride = 640;
           ok.required_bytes = 3 * 640 * 360;

           Kernel200EnvelopeViolation violation;
           require(validate_kernel200_envelope(limits, ok, &violation),
                   "envelope guard should pass for valid actual dims");
           require(violation.code.empty(), "violation code must be empty on success");

           Kernel200EnvelopeActual bad_w = ok;
           bad_w.actual_w = 1600;
           require(!validate_kernel200_envelope(limits, bad_w, &violation),
                   "envelope guard should fail when actual width exceeds max");
           require(violation.code == "actual_gt_max_w",
                   "violation code mismatch for actual width overflow");
           require_contains(violation.message, "exceeds max",
                            "violation message should describe max overflow");

           Kernel200EnvelopeActual bad_bytes = ok;
           bad_bytes.required_bytes = limits.allocated_bytes + 1;
           require(!validate_kernel200_envelope(limits, bad_bytes, &violation),
                   "envelope guard should fail when required bytes exceed allocation");
           require(violation.code == "required_gt_allocated_bytes",
                   "violation code mismatch for allocation overflow");

           Kernel200EnvelopeActual bad_dims{};
           bad_dims.actual_w = 0;
           bad_dims.actual_h = 360;
           bad_dims.actual_stride = 640;
           bad_dims.required_bytes = 100;
           require(!validate_kernel200_envelope(limits, bad_dims, &violation),
                   "envelope guard should fail for invalid actual dims");
           require(violation.code == "invalid_actual_dims",
                   "violation code mismatch for invalid actual dims");
         }));
