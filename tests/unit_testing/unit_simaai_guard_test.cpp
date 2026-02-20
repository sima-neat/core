#include "pipeline/internal/SimaaiGuard.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST(
    "unit_simaai_guard_test", ([] {
      using namespace simaai::neat::pipeline_internal;

      reset_simaai_guard_for_test();

      require(pipeline_uses_simaai("neatprocesscvu name=n0"),
              "pipeline_uses_simaai should detect simaai plugin token");
      require(!pipeline_uses_simaai("videotestsrc ! fakesink"),
              "pipeline_uses_simaai false-positive on non-simaai pipeline");

      require(pipeline_uses_mla("neatprocessmla name=n1"),
              "pipeline_uses_mla should detect MLA plugin token");
      require(!pipeline_uses_mla("neatprocesscvu name=n2"),
              "pipeline_uses_mla false-positive on non-MLA simaai plugin");

      std::string err;
      auto no_guard =
          acquire_simaai_guard("unit_simaai_guard_test", "videotestsrc ! fakesink", false, &err);
      require(
          !no_guard,
          "acquire_simaai_guard should not allocate guard when force=false and no simaai pipeline");
      require(err.empty(), "acquire_simaai_guard should not set error for no-op path");

      auto forced_guard =
          acquire_simaai_guard("unit_simaai_guard_test", "videotestsrc ! fakesink", true, &err);
      require(static_cast<bool>(forced_guard),
              "acquire_simaai_guard(force=true) should return a guard object");

      // Guard enforcement is currently disabled; this should be a no-op.
      enforce_single_mla_pipeline("unit_simaai_guard_test", "neatprocessmla ! fakesink", nullptr,
                                  "test");
    }));
