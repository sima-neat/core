#include "pipeline/internal/EnvUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <string>

namespace {

class EnvVarGuard {
public:
  EnvVarGuard(const char* key, const char* value) : key_(key), had_(false) {
    const char* cur = std::getenv(key_);
    if (cur) {
      had_ = true;
      old_ = cur;
    }
    if (value) {
      ::setenv(key_, value, 1);
    } else {
      ::unsetenv(key_);
    }
  }

  ~EnvVarGuard() {
    if (had_) {
      ::setenv(key_, old_.c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

private:
  const char* key_;
  bool had_;
  std::string old_;
};

} // namespace

RUN_TEST("unit_env_util_profile_test", ([] {
           using simaai::neat::pipeline_internal::env_bool;

           EnvVarGuard profile("SIMA_DEBUG_PROFILE", "pipeline");
           EnvVarGuard level("SIMA_DEBUG_LEVEL", "1");
           EnvVarGuard diag_unset("SIMA_ASYNC_TPUT_DIAG", nullptr);
           require(env_bool("SIMA_ASYNC_TPUT_DIAG", false),
                   "pipeline profile should enable SIMA_ASYNC_TPUT_DIAG at level=1");

           EnvVarGuard pool_unset("SIMA_PULL_TIMEOUT_POOL_DIAG", nullptr);
           require(!env_bool("SIMA_PULL_TIMEOUT_POOL_DIAG", false),
                   "pipeline profile level=1 should not enable pool diag");

           {
             EnvVarGuard level2("SIMA_DEBUG_LEVEL", "2");
             require(env_bool("SIMA_PULL_TIMEOUT_POOL_DIAG", false),
                     "pipeline profile level=2 should enable pool diag");
           }

           {
             EnvVarGuard explicit_off("SIMA_ASYNC_TPUT_DIAG", "0");
             require(!env_bool("SIMA_ASYNC_TPUT_DIAG", false),
                     "explicit env value must override debug profile fallback");
           }

           {
             EnvVarGuard gst_profile("SIMA_DEBUG_PROFILE", "gst");
             EnvVarGuard gst_level("SIMA_DEBUG_LEVEL", "1");
             EnvVarGuard gst_flow_unset("SIMA_GST_FLOW_DEBUG", nullptr);
             require(env_bool("SIMA_GST_FLOW_DEBUG", false),
                     "gst profile should enable SIMA_GST_FLOW_DEBUG");
           }

           {
             EnvVarGuard sample_profile("SIMA_DEBUG_PROFILE", "sample");
             EnvVarGuard sample_level("SIMA_DEBUG_LEVEL", "1");
             EnvVarGuard rtsp_unset("SIMA_RTSP_DEBUG", nullptr);
             require(!env_bool("SIMA_RTSP_DEBUG", false),
                     "unrelated profile component should not enable RTSP debug");
           }
         }));
