#include "support/core_render_retirement.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <array>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

namespace {
using Policy = simaai::neat::pipeline_internal::InputStreamTeardownPolicy;

template <typename F> void rejected(F&& call, std::string_view diagnostic) {
  try {
    call();
  } catch (const std::runtime_error& error) {
    require(std::string_view(error.what()).find(diagnostic) != std::string_view::npos,
            "unexpected retirement admission diagnostic");
    return;
  }
  throw std::runtime_error("malformed retirement admission was accepted");
}

class EnvGuard {
public:
  explicit EnvGuard(const char* name) : name_(name) {
    if (const char* value = std::getenv(name)) {
      present_ = true;
      value_ = value;
    }
  }
  ~EnvGuard() {
    if (present_)
      ::setenv(name_, value_.c_str(), 1);
    else
      ::unsetenv(name_);
  }

private:
  const char* name_;
  bool present_ = false;
  std::string value_;
};

simaai::neat::InputStream make_stream_state(const simaai::neat::InputStreamOptions& options) {
  simaai::neat::SampleSpec spec;
  spec.kind = simaai::neat::SampleMediaKind::Encoded;
  spec.caps_string = "application/octet-stream";
  spec.caps_key = simaai::neat::capkey_from_spec(spec);
  // Real create()/State ownership and caps lifetime, but no Gst pipeline,
  // worker, DMA allocation or accelerator. This tests observation, not NULL.
  return simaai::neat::InputStream::create(nullptr, nullptr, nullptr, spec,
                                           simaai::neat::InputOptions{}, options, {}, nullptr);
}

void actual_stream_policy_contract() {
  using namespace simaai::neat;
  using core_render_test::is_driver_stream;
  constexpr std::string_view driver_pipeline = "appsrc ! neatprocessmla ! appsink";
  gst_init(nullptr, nullptr);

  InputStream empty;
  require(!empty.effective_teardown_policy(), "empty stream must not invent a policy");
  rejected([&] { (void)is_driver_stream(driver_pipeline, empty); }, "finalized teardown policy");

  for (const auto policy : {Policy::Deferred, Policy::BoundedPreferred, Policy::MustReachNull}) {
    InputStreamOptions options;
    options.teardown_policy = policy;
    auto stream = make_stream_state(options);
    options.teardown_policy =
        policy == Policy::MustReachNull ? Policy::Deferred : Policy::MustReachNull;
    require(stream.effective_teardown_policy() == policy,
            "query must read the immutable created State, not caller options");
    if (policy == Policy::MustReachNull)
      require(is_driver_stream(driver_pipeline, stream), "actual Must policy was not admitted");
    else
      rejected([&] { (void)is_driver_stream(driver_pipeline, stream); }, "MustReachNull");

    InputStream moved(std::move(stream));
    require(!stream.effective_teardown_policy(), "moved-from stream retained a policy");
    require(moved.effective_teardown_policy() == policy, "move lost the created State policy");
    stream = std::move(moved);
    require(!moved.effective_teardown_policy(), "move assignment retained source policy");
    require(stream.effective_teardown_policy() == policy, "move assignment changed policy");
    stream.stop();
    require(stream.effective_teardown_policy() == policy,
            "policy observation must not masquerade as a completion flag");
    stream.close();
    require(!stream.effective_teardown_policy(), "closed stream still exposes released State");
  }

  InputStreamOptions requested;
  require(requested.teardown_policy == Policy::Deferred, "unexpected default build policy");
  auto finalized = requested;
  finalized.teardown_policy = Policy::MustReachNull;
  // Exercise the real RunCore initializer that preserves the requested options
  // separately from the finalized InputStream. No private State is fabricated.
  auto core = runtime::RunCore::start_single_pipeline(make_stream_state(finalized), RunOptions{},
                                                      requested, RunMode::Sync);
  require(core->pipeline.stream_opt.teardown_policy == Policy::Deferred,
          "regression requires an unchanged requested-options snapshot");
  require(core->pipeline.stream.effective_teardown_policy() == Policy::MustReachNull,
          "RunCore did not retain the actual finalized stream policy");
  require(is_driver_stream(driver_pipeline, core->pipeline.stream),
          "stale requested options must not reject the actual Must stream");
  core->close();
  require(!core->pipeline.stream.effective_teardown_policy(),
          "RunCore close did not release the observed State");
}
} // namespace

RUN_TEST("unit_core_render_retirement_contract_test", ([] {
           using core_render_test::is_driver_stream;
           using core_render_test::require_synchronous_teardown_environment;

           actual_stream_policy_contract();

           // Real Core's classifier, not a duplicate test factory catalogue.
           // A split graph contains ordinary Gst ingress/egress as well as
           // numerical leaves; push/pull support does not identify a driver.
           struct Leaf {
             std::string_view pipeline;
             Policy policy;
             bool driver;
           };
           const std::array<Leaf, 4> topology{{
               {"appsrc name=input ! queue ! appsink name=ingress", Policy::Deferred, false},
               {"appsrc ! neatkernelrunner name=preproc ! appsink", Policy::MustReachNull, true},
               {"appsrc ! neatprocessmla name=inference ! appsink", Policy::MustReachNull, true},
               {"appsrc name=egress ! appsink name=output", Policy::Deferred, false},
           }};
           std::size_t streams = 0;
           std::size_t drivers = 0;
           for (const auto& leaf : topology) {
             const bool driver = is_driver_stream(leaf.pipeline, leaf.policy);
             require(driver == leaf.driver, "compiled leaf classification mismatch");
             ++streams;
             drivers += driver;
           }
           require(streams == 4 && drivers == 2, "plain transport leaves must not be discarded");
           for (const auto factory : {"neatkernelrunner", "neatprocessmla", "neatprocesscvu",
                                      "neatdecoder", "neatencoder"}) {
             const auto pipeline = std::string("( appsrc ! ") + factory + " ) ! appsink";
             require(is_driver_stream(pipeline, Policy::MustReachNull),
                     "hardware leaf not counted");
             for (const auto policy : {Policy::Deferred, Policy::BoundedPreferred})
               rejected([&] { (void)is_driver_stream(pipeline, policy); }, "MustReachNull");
           }
           require(!is_driver_stream("appsrc ! identity name=neatprocessmla_probe ! appsink",
                                     Policy::Deferred),
                   "an element name is not a hardware factory");
           require(!is_driver_stream("appsrc ! neatprocessmla_legacy ! appsink", Policy::Deferred),
                   "a factory prefix is not a hardware factory");
           require(!is_driver_stream(topology.front().pipeline, Policy::BoundedPreferred),
                   "stronger ordinary transport teardown is permitted");
           require(!is_driver_stream(topology.front().pipeline, Policy::MustReachNull),
                   "policy alone must not invent a hardware factory");
           rejected([&] { (void)is_driver_stream("", Policy::MustReachNull); },
                    "compiled pipeline");

           EnvGuard defer_guard("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH");
           EnvGuard async_guard("SIMA_GST_TEARDOWN_ASYNC");
           for (const auto name : {"SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "SIMA_GST_TEARDOWN_ASYNC"}) {
             ::setenv("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", "0", 1);
             ::setenv("SIMA_GST_TEARDOWN_ASYNC", "0", 1);
             require_synchronous_teardown_environment();
             ::unsetenv(name);
             rejected(require_synchronous_teardown_environment, name);
             for (const auto value : {"", "1", "false", "00"}) {
               ::setenv(name, value, 1);
               rejected(require_synchronous_teardown_environment, name);
             }
           }
         }));
