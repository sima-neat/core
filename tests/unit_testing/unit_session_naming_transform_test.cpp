#include "pipeline/Session.h"
#include "pipeline/internal/PipelineBuild.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>
#include <vector>

RUN_TEST("unit_session_naming_transform_test", ([] {
           using namespace simaai::neat;

           NameTransform t;
           t.prefix = "pre_";
           t.suffix = "_suf";

           require(name_transform_enabled(t),
                   "name_transform_enabled should be true for non-empty transform");
           require(apply_name_transform(t, "decoder") == "pre_decoder_suf",
                   "apply_name_transform failed basic prefix/suffix transform");
           require(apply_name_transform(t, "pre_decoder_suf") == "pre_decoder_suf",
                   "apply_name_transform should be idempotent");

           // Session naming keeps RTSP payloader names unchanged.
           require(apply_name_transform(t, "pay0") == "pay0",
                   "RTSP payloader name should not be transformed");

           // PipelineBuild transform does not special-case payN.
           require(apply_name_transform_once("pay0", t) == "pre_pay0_suf",
                   "apply_name_transform_once should transform payloader names");

           const std::vector<std::string> names = {"a", "pre_b_suf"};
           const auto transformed = apply_name_transform(t, names);
           require(transformed.size() == 2, "vector transform size mismatch");
           require(transformed[0] == "pre_a_suf", "vector transform first element mismatch");
           require(transformed[1] == "pre_b_suf",
                   "vector transform should preserve already transformed names");

           SessionOptions opt;
           opt.element_name_prefix = "x_";
           opt.element_name_suffix = "_y";

           Session session(opt);
           session.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
           session.custom(
               "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

           const std::string backend = session.describe_backend(false);
           require_contains(backend, "x_mysink_y",
                            "describe_backend should include transformed appsink element name");
         }));
