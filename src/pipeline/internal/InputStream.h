#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/InputStreamStats.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"
#include "pipeline/Run.h"
#include "pipeline/internal/OutputTensorOverride.h"

#include <gst/gst.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {

struct InputOptions;
struct Sample;
struct SampleSpec;
struct SampleTimingOverrides;

namespace pipeline_internal {
struct DiagCtx;
} // namespace pipeline_internal

struct InputStreamOptions {
  enum class DynamicCapability {
    StaticOnly = 0,
    IngressDynamicCvuOnly = 1,
    FullyDynamic = 2,
  };

  enum class ShapePolicy {
    BoundedDynamic = 0,
    ElasticDynamic = 1,
    LockedByCapsOverride = 2,
  };

  enum class LimitOrigin {
    Unset = 0,
    SeedInput = 1,
    UserSeed = 2,
    UserMax = 3,
  };

  enum class ByteGuardOrigin {
    Unset = 0,
    User = 1,
    DerivedElasticDefault = 2,
    DerivedBoundedEstimate = 3,
  };

  struct ResolvedShapeLimits {
    int seed_width = -1;
    int seed_height = -1;
    int seed_depth = -1;
    LimitOrigin seed_width_origin = LimitOrigin::Unset;
    LimitOrigin seed_height_origin = LimitOrigin::Unset;
    LimitOrigin seed_depth_origin = LimitOrigin::Unset;
    int max_width = -1;
    int max_height = -1;
    int max_depth = -1;
    LimitOrigin max_width_origin = LimitOrigin::Unset;
    LimitOrigin max_height_origin = LimitOrigin::Unset;
    LimitOrigin max_depth_origin = LimitOrigin::Unset;
    bool max_width_explicit = false;
    bool max_height_explicit = false;
    bool max_depth_explicit = false;
  };

  int timeout_ms = 10000;
  int worker_poll_ms = 0;
  bool appsink_sync = true;
  bool appsink_drop = false;
  int appsink_max_buffers = 4;
  bool enable_timings = false;
  bool startup_preflight = false;
  int stability_frames = 2;
  std::size_t max_input_bytes = 0;
  bool copy_output = true;
  bool copy_input = false;
  bool prepare_output_cpu_visible = false;
  int holder_loan_credits = 0;
  bool holder_loan_credits_auto = false;
  int holder_loan_sample_window = 1;
  int holder_loan_per_sample_arity = 1;
  // When true, CPU-backed Tensor inputs are rejected before the slow
  // InputStream memcpy fallback.  Device-first routes should receive tensors
  // constructed in the required memory placement; set
  // SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1 for legacy compatibility.
  bool require_device_visible_input = false;
  bool reuse_input_buffer = false;
  // True for user-visible Output/appsink endpoints.  False for graph-internal
  // transport appsinks, where downstream edge/view contracts must be preserved
  // instead of being rewritten as public terminal outputs.
  bool public_output_contract = true;
  DynamicCapability dynamic_capability = DynamicCapability::StaticOnly;
  ShapePolicy shape_policy = ShapePolicy::BoundedDynamic;
  ResolvedShapeLimits shape_limits{};
  ByteGuardOrigin byte_guard_origin = ByteGuardOrigin::Unset;
  bool allow_ingress_cvu_format_renegotiation = false;
  std::optional<OutputTensorOverride> output_override;
  std::function<void(const InputDropInfo&)> on_input_drop;
};

class InputStream {
public:
  struct State;
  InputStream() = default;
  InputStream(const InputStream&) = delete;
  InputStream& operator=(const InputStream&) = delete;

  InputStream(InputStream&&) noexcept;
  InputStream& operator=(InputStream&&) noexcept;
  ~InputStream();

  explicit operator bool() const noexcept;
  bool can_push() const noexcept;
  bool can_pull() const noexcept;

  static InputStream create(GstElement* pipeline, GstElement* appsrc, GstElement* appsink,
                            const SampleSpec& spec, const InputOptions& src_opt,
                            const InputStreamOptions& opt,
                            std::shared_ptr<pipeline_internal::DiagCtx> diag,
                            std::shared_ptr<void> guard);

  Sample push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms = -1);
  void push(const cv::Mat& input);
  bool try_push(const cv::Mat& input);
  void push(const simaai::neat::Tensor& input);
  bool try_push(const simaai::neat::Tensor& input);
  void push_message(const Sample& msg);
  bool try_push_message(const Sample& msg);
  void push_holder(const std::shared_ptr<void>& holder);
  bool try_push_holder(const std::shared_ptr<void>& holder);
  Sample pull(int timeout_ms = -1);
  void pull_and_discard(int timeout_ms = -1);
  void signal_eos();
  void drain_before_teardown(int timeout_ms);

  void start(std::function<void(Sample)> on_output);
  void stop();
  void stop_async();
  bool running() const;
  std::string last_error() const;
  InputStreamStats stats() const;
  std::string diagnostics_summary() const;
  std::shared_ptr<pipeline_internal::DiagCtx> diag_ctx() const;
  GstElement* pipeline_handle() const;
  void set_on_caps_change(std::function<void(const SampleSpec&, const SampleSpec&)> cb);

  void close();

private:
  std::shared_ptr<State> state_;

  explicit InputStream(std::shared_ptr<State> state);
  bool push_with_fill(const char* where, const std::function<size_t(uint8_t*, size_t)>& fill,
                      size_t required_bytes, const std::optional<int64_t>& frame_id_override,
                      const std::optional<int64_t>& input_seq_override,
                      const std::optional<int64_t>& orig_input_seq_override,
                      const std::optional<std::string>& stream_id_override,
                      const std::optional<std::string>& buffer_name_override,
                      const SampleTimingOverrides& timing_override,
                      const std::function<void(GstBuffer**)>& prepare = {}, int input_width = -1,
                      int input_height = -1);
  friend class Graph;
  friend class Run;
};

} // namespace simaai::neat
