#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Session.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <optional>
#include <string>

namespace {

struct ScopedEnv {
  explicit ScopedEnv(const char* key, const char* value) : key_(key) {
    const char* prev = std::getenv(key_);
    if (prev) {
      prev_ = std::string(prev);
    }
    if (value) {
      ::setenv(key_, value, 1);
    } else {
      ::unsetenv(key_);
    }
  }

  ~ScopedEnv() {
    if (prev_.has_value()) {
      ::setenv(key_, prev_->c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  std::optional<std::string> prev_;
};

} // namespace

RUN_TEST(
    "session_validate_report_regression_test", ([] {
      using namespace simaai::neat;

      {
        Session empty;
        ValidateOptions opt;
        opt.parse_launch = false;

        const SessionReport rep = empty.validate(opt);
        require(rep.error_code == error_codes::kPipelineShape,
                "validate(parse_launch=false): expected misconfig.pipeline_shape error code");
        require_contains(
            rep.repro_note, "contract checks failed",
            "validate(parse_launch=false) should still report contract failures on empty pipeline");
      }

      {
        Session source_only;
        source_only.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
        source_only.add(nodes::Output(OutputOptions::Latest()));

        ValidateOptions opt;
        opt.parse_launch = false;
        const SessionReport rep = source_only.validate(opt);

        require(rep.error_code.empty(),
                "validate(parse_launch=false): skipped parse path should not set error_code");
        require(rep.pipeline_string == "<parse_launch disabled>",
                "validate(parse_launch=false) should set parse-launch-disabled pipeline sentinel");
        require_contains(rep.repro_note, "skipped",
                         "validate(parse_launch=false) should report skipped parse-launch path");
      }

      {
        Session with_input;
        InputOptions src_opt;
        src_opt.media_type = "video/x-raw";
        src_opt.format = "RGB";
        src_opt.use_simaai_pool = false;
        with_input.add(nodes::Input(src_opt));
        with_input.add(nodes::Output(OutputOptions::Latest()));

        ValidateOptions opt;
        opt.parse_launch = false;
        const SessionReport rep = with_input.validate(opt);
        require(rep.error_code == error_codes::kPipelineShape,
                "validate() without input should set misconfig.pipeline_shape");
        require(rep.pipeline_string == "<input required>",
                "validate() without input frame should reject Input appsrc pipelines");
        require_contains(rep.repro_note, "Input() is present",
                         "validate() missing-input diagnostic mismatch");
      }

      // Parse/launch failure path should include deterministic repro diagnostics.
      {
        Session broken_parse;
        broken_parse.custom("thiselementdoesnotexist definitely=true", InputRole::Source);
        broken_parse.custom(
            "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

        ValidateOptions opt;
        opt.parse_launch = true;
        opt.enforce_names = false;
        const SessionReport rep = broken_parse.validate(opt);

        require(rep.error_code == error_codes::kParseLaunch,
                "validate(parse failure): expected build.parse_launch error code");
        require(!rep.pipeline_string.empty(),
                "validate(parse failure): expected non-empty pipeline string");
        require(!rep.repro_note.empty(),
                "validate(parse failure): missing parse-launch failure diagnostic");
        require_contains(rep.repro_note, "gst_parse_launch failed",
                         "validate(parse failure): missing parse-launch failure diagnostic");
        require_contains(rep.repro_gst_launch, "gst-launch-1.0 -v",
                         "validate(parse failure): missing gst-launch reproduction command");
      }

      // Live source validate() should time out in PAUSED and report deterministically.
      {
        Session live_source;
        live_source.custom("videotestsrc is-live=true pattern=black", InputRole::Source);
        live_source.custom(
            "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

        ScopedEnv timeout_env("SIMA_GST_VALIDATE_TIMEOUT_MS", "120");

        ValidateOptions opt;
        opt.parse_launch = true;
        const SessionReport rep = live_source.validate(opt);

        require(rep.error_code == error_codes::kRuntimePull,
                "validate(live timeout): expected runtime.pull error code");
        require(!rep.pipeline_string.empty(),
                "validate(live timeout): expected non-empty pipeline string");
        require_contains(rep.repro_note, "preroll timed out",
                         "validate(live timeout): expected timeout diagnostic");
      }

      // Finite source validate() should populate report fields on parse path.
      {
        Session finite_source;
        finite_source.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
        finite_source.custom(
            "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

        ValidateOptions opt;
        opt.parse_launch = true;
        const SessionReport rep = finite_source.validate(opt);

        require(rep.error_code.empty(),
                "validate(finite): successful preroll should not set error_code");
        require(!rep.pipeline_string.empty(),
                "validate(finite): pipeline string should be populated");
        require(!rep.repro_note.empty(), "validate(finite): repro_note should be populated");
        require_contains(rep.repro_note, "validate:",
                         "validate(finite): repro_note should include validate status");
      }
    }));
