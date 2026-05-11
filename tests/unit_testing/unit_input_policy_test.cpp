#include "mpk_test_utils.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstddef>
#include <string>

RUN_TEST("unit_input_policy_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::pipeline_internal;

           // Model defaults: unresolved max uses 1920x1080 and format-derived depth.
           {
             ModelInputPolicyRequest req;
             req.format = "RGB";
             const ModelInputPolicyResult out = resolve_for_model(req);

             require(out.resolved_input_format == "RGB", "model policy: input format mismatch");
             require(out.resolved_max_input_width == 1920,
                     "model policy: default max width must be 1920");
             require(out.resolved_max_input_height == 1080,
                     "model policy: default max height must be 1080");
             require(out.resolved_max_input_depth == 3,
                     "model policy: default RGB max depth must be 3");
           }

           // Model defaults are independent from dynamic runtime ingress dimensions.
           {
             ModelInputPolicyRequest req;
             req.format = "GRAY";
             const ModelInputPolicyResult out = resolve_for_model(req);

             require(out.resolved_input_format == "GRAY", "model policy: format mismatch");
             require(out.resolved_max_input_width == 1920,
                     "model policy: default max width must remain 1920");
             require(out.resolved_max_input_height == 1080,
                     "model policy: default max height must remain 1080");
             require(out.resolved_max_input_depth == 1,
                     "model policy: GRAY max depth should resolve to 1");
           }

           // Model explicit max overrides always win.
           {
             ModelInputPolicyRequest req;
             req.format = "BGR";
             req.input_max_width = 1024;
             req.input_max_height = 768;
             req.input_max_depth = 4;
             const ModelInputPolicyResult out = resolve_for_model(req);

             require(out.resolved_max_input_width == 1024,
                     "model policy: explicit max width must win");
             require(out.resolved_max_input_height == 768,
                     "model policy: explicit max height must win");
             require(out.resolved_max_input_depth == 4,
                     "model policy: explicit max depth must win");
           }

           // Session bounded mode: explicit seed/max should be preserved with origins.
           {
             InputOptions opt;
             opt.width = 320;
             opt.height = 240;
             opt.depth = 3;
             opt.max_width = 640;
             opt.max_height = 480;
             opt.max_depth = 3;

             SampleSpec seed;
             seed.width = 300;
             seed.height = 200;
             seed.depth = 3;

             const std::size_t bounded_estimate = 12345;
             const SessionInputPolicyResult out =
                 resolve_for_session(opt, seed, /*requested_max_input_bytes=*/0, bounded_estimate);

             require(out.shape_policy == InputStreamOptions::ShapePolicy::BoundedDynamic,
                     "session policy: expected bounded dynamic mode");
             require(out.shape_limits.seed_width == 320,
                     "session policy: user seed width should win");
             require(out.shape_limits.seed_height == 240,
                     "session policy: user seed height should win");
             require(out.shape_limits.seed_depth == 3,
                     "session policy: user seed depth should win");
             require(out.shape_limits.seed_width_origin == InputStreamOptions::LimitOrigin::UserSeed,
                     "session policy: seed width origin should be user_seed");
             require(out.shape_limits.max_width == 640,
                     "session policy: explicit max width should be preserved");
             require(out.shape_limits.max_height == 480,
                     "session policy: explicit max height should be preserved");
             require(out.shape_limits.max_depth == 3,
                     "session policy: explicit max depth should be preserved");
             require(out.shape_limits.max_width_origin == InputStreamOptions::LimitOrigin::UserMax,
                     "session policy: max width origin should be user_max");
             require(out.max_input_bytes_guard == bounded_estimate,
                     "session policy: bounded mode should use bounded estimate guard");
             require(out.byte_guard_origin ==
                         InputStreamOptions::ByteGuardOrigin::DerivedBoundedEstimate,
                     "session policy: bounded mode guard origin should be derived estimate");
           }

           // Session elastic mode: default guard should come from env-derived elastic limit.
           {
             sima_test::ScopedEnvVar elastic_guard_mb("SIMA_INPUTSTREAM_ELASTIC_MAX_MB", "2");

             InputOptions opt;
             SampleSpec seed;
             seed.width = 64;
             seed.height = 64;
             seed.depth = 3;

             const SessionInputPolicyResult out =
                 resolve_for_session(opt, seed, /*requested_max_input_bytes=*/0,
                                     /*bounded_estimate_bytes=*/7777);

             require(out.shape_policy == InputStreamOptions::ShapePolicy::ElasticDynamic,
                     "session policy: expected elastic mode when no explicit bounds exist");
             require(out.max_input_bytes_guard == (2ULL * 1024ULL * 1024ULL),
                     "session policy: elastic guard should use env MB default");
             require(out.byte_guard_origin ==
                         InputStreamOptions::ByteGuardOrigin::DerivedElasticDefault,
                     "session policy: elastic guard origin should be derived elastic default");
           }

           // Seed > max validation should reject deterministically.
           {
             InputOptions opt;
             opt.width = 640;
             opt.height = 480;
             opt.depth = 3;
             opt.max_width = 320;
             opt.max_height = 480;
             opt.max_depth = 3;

             SampleSpec seed;
             const auto limits = resolve_shape_limits(opt, seed);
             const auto err = validate_shape_limits(limits);
             require(err.has_value(), "session policy: expected validation error for seed > max");
             require_contains(*err, "width", "session policy: expected width error");
           }

           // Model-vs-session parity where rules are equivalent (seed-as-max behavior).
           {
             ModelInputPolicyRequest mreq;
             mreq.format = "RGB";
             mreq.input_max_width = 640;
             mreq.input_max_height = 480;
             mreq.input_max_depth = 3;
             const ModelInputPolicyResult mout = resolve_for_model(mreq);

             InputOptions sopt;
             sopt.width = 640;
             sopt.height = 480;
             sopt.depth = 3;
             SampleSpec seed;
             seed.width = 640;
             seed.height = 480;
             seed.depth = 3;
             const SessionInputPolicyResult sout =
                 resolve_for_session(sopt, seed, /*requested_max_input_bytes=*/4096,
                                     /*bounded_estimate_bytes=*/1024);

             require(mout.resolved_max_input_width == sout.shape_limits.max_width,
                     "policy parity: max width mismatch");
             require(mout.resolved_max_input_height == sout.shape_limits.max_height,
                     "policy parity: max height mismatch");
             require(mout.resolved_max_input_depth == sout.shape_limits.max_depth,
                     "policy parity: max depth mismatch");
           }
         }));
