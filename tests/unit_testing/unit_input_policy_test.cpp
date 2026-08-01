#include "model_archive_test_utils.h"
#include "nodes/io/Input.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstddef>
#include <string>

RUN_TEST(
    "unit_input_policy_test", ([] {
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
        require(out.resolved_max_input_depth == 3, "model policy: default RGB max depth must be 3");
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

        require(out.resolved_max_input_width == 1024, "model policy: explicit max width must win");
        require(out.resolved_max_input_height == 768, "model policy: explicit max height must win");
        require(out.resolved_max_input_depth == 4, "model policy: explicit max depth must win");
      }

      // Capacity diagnostics must report the actual value, configured limit, and public fix.
      {
        const std::string message =
            shape_limit_exceeded_message("Graph::build(input)", "width", 2048, 1920);
        const std::string hint = shape_limit_fix_hint("width", 2048);
        require_contains(message, "input width 2048 exceeds configured capacity 1920",
                         "shape diagnostic should include actual and configured width");
        require_contains(hint, "Model::Options::preprocess.input_max_width",
                         "shape diagnostic should name the public model option");
        require_contains(hint, "at least 2048",
                         "shape diagnostic should include the required capacity");
      }

      // Graph bounded mode: explicit seed/max should be preserved with origins.
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
        const GraphInputPolicyResult out =
            resolve_for_graph(opt, seed, /*requested_max_input_bytes=*/0, bounded_estimate);

        require(out.shape_policy == InputStreamOptions::ShapePolicy::BoundedDynamic,
                "graph policy: expected bounded dynamic mode");
        require(out.shape_limits.seed_width == 320, "graph policy: user seed width should win");
        require(out.shape_limits.seed_height == 240, "graph policy: user seed height should win");
        require(out.shape_limits.seed_depth == 3, "graph policy: user seed depth should win");
        require(out.shape_limits.seed_width_origin == InputStreamOptions::LimitOrigin::UserSeed,
                "graph policy: seed width origin should be user_seed");
        require(out.shape_limits.max_width == 640,
                "graph policy: explicit max width should be preserved");
        require(out.shape_limits.max_height == 480,
                "graph policy: explicit max height should be preserved");
        require(out.shape_limits.max_depth == 3,
                "graph policy: explicit max depth should be preserved");
        require(out.shape_limits.max_width_origin == InputStreamOptions::LimitOrigin::UserMax,
                "graph policy: max width origin should be user_max");
        require(out.max_input_bytes_guard == bounded_estimate,
                "graph policy: bounded mode should use bounded estimate guard");
        require(out.byte_guard_origin ==
                    InputStreamOptions::ByteGuardOrigin::DerivedBoundedEstimate,
                "graph policy: bounded mode guard origin should be derived estimate");
      }

      // Graph elastic mode: default guard should come from env-derived elastic limit.
      {
        sima_test::ScopedEnvVar elastic_guard_mb("SIMA_INPUTSTREAM_ELASTIC_MAX_MB", "2");

        InputOptions opt;
        SampleSpec seed;
        seed.width = 64;
        seed.height = 64;
        seed.depth = 3;

        const GraphInputPolicyResult out =
            resolve_for_graph(opt, seed, /*requested_max_input_bytes=*/0,
                              /*bounded_estimate_bytes=*/7777);

        require(out.shape_policy == InputStreamOptions::ShapePolicy::ElasticDynamic,
                "graph policy: expected elastic mode when no explicit bounds exist");
        require(out.max_input_bytes_guard == (2ULL * 1024ULL * 1024ULL),
                "graph policy: elastic guard should use env MB default");
        require(out.byte_guard_origin == InputStreamOptions::ByteGuardOrigin::DerivedElasticDefault,
                "graph policy: elastic guard origin should be derived elastic default");
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
        require(err.has_value(), "graph policy: expected validation error for seed > max");
        require_contains(*err, "width", "graph policy: expected width error");
      }

      // Encoded seed samples carry a full caps contract that cannot be reconstructed from
      // format/shape fields alone.
      {
        const std::string caps =
            "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
            "parsed=(boolean)true";
        InputOptions opt;
        SampleSpec seed;
        seed.kind = SampleMediaKind::Encoded;
        seed.media_type = "video/x-h264";
        seed.format = "ENCODED";
        seed.caps_string = caps;

        const InputOptions out = complete_input_options_from_seed_spec(opt, seed);
        require(out.payload_type == PayloadType::Encoded,
                "encoded seed should resolve Input payload_type");
        require(out.caps_override == caps, "encoded seed should preserve exact caps string");
      }

      // Explicit caps are user-owned and should not be replaced by seed inference.
      {
        InputOptions opt;
        opt.payload_type = PayloadType::Encoded;
        opt.caps_override = "video/x-h264,stream-format=(string)avc,alignment=(string)au";

        SampleSpec seed;
        seed.kind = SampleMediaKind::Encoded;
        seed.media_type = "video/x-h264";
        seed.format = "ENCODED";
        seed.caps_string = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
                           "parsed=(boolean)true";

        const InputOptions out = complete_input_options_from_seed_spec(opt, seed);
        require(out.caps_override == opt.caps_override,
                "explicit caps_override should win over encoded seed caps");
      }

      // Model-vs-Graph parity where rules are equivalent (seed-as-max behavior).
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
        const GraphInputPolicyResult sout =
            resolve_for_graph(sopt, seed, /*requested_max_input_bytes=*/4096,
                              /*bounded_estimate_bytes=*/1024);

        require(mout.resolved_max_input_width == sout.shape_limits.max_width,
                "policy parity: max width mismatch");
        require(mout.resolved_max_input_height == sout.shape_limits.max_height,
                "policy parity: max height mismatch");
        require(mout.resolved_max_input_depth == sout.shape_limits.max_depth,
                "policy parity: max depth mismatch");
      }
    }));
