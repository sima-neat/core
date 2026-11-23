#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <gst/video/video.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <atomic>
#include <mutex>
#include <string_view>
#include <simaai/scheduler/client/frame_adapter.h>
namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
// Scheduler names are qualified (rt::, CanonicalKernelLowering.h), never brought in by a
// using-directive: unqualified, Result, Status and TensorSpec would name simaai::neat's own types
// in a TU that also includes pipeline/Result.h, InputStreamUtil.h or model/Model.h.
namespace gstreamer = ::simaai::scheduler::client;
// Core-owned image geometry, caps and metadata policy. Numerical validation
// and ROI filtering remain in kernel-owned generic preparation.
class PreprocFrameAdapter : public gstreamer::TensorStageFrameAdapter {
public:
  // Immutable numerical/image policy; contains no admission, session or cache.
  struct CheckedPolicy;
  static rt::Result<std::shared_ptr<const CheckedPolicy>>
      check_policy(std::shared_ptr<const ProcessCvuStagePayload>);
  static rt::Result<std::shared_ptr<const CheckedPolicy>>
  check_policy(const ProcessCvuStagePayload&);
  static rt::Result<std::shared_ptr<PreprocFrameAdapter>>
      instantiate(std::shared_ptr<const CheckedPolicy>);
  static const rt::KernelExecutable& checked_executable(const CheckedPolicy&);
  static std::shared_ptr<const gstreamer::TensorStageImagePublication>
  checked_image_publication(const CheckedPolicy&);
  // Initial immutable facts from the same checked policy as runtime publication.
  // This cold query opens no session and retains no frame or adapter state.
  static rt::Result<simaai::gst::TensorBufferPreprocessFactsOwner> checked_initial_preprocess_facts(
      std::shared_ptr<const CheckedPolicy>,
      const simaai::gst::TensorBufferPreprocessFacts* upstream = nullptr);
  // The stage as one kernel request at one packed frame geometry, for a backend that runs it
  // without a frame adapter (Dataflow, at a size fixed when the input binds): the request the
  // geometry variant gives the Kernels package for such a frame (bind's cold path with no ROI
  // metadata and no video meta: the same input_specs, static_parameters, parameters and
  // prepare_publication), checked, with the facts published with every frame of it. `row_stride`
  // is the frame's row pitch in bytes: of the one plane of a packed format, or of both planes of
  // an NV12 frame whose chroma plane follows its luma plane at once (the frame as one buffer of
  // row_stride x height x 3 / 2 bytes). Cold; opens no session and retains no frame or adapter
  // state.
  struct FixedFrame {
    std::vector<rt::TensorSpec> inputs, outputs;
    std::vector<rt::Parameter> parameters;
    rt::KernelExecutable executable;
    simaai::gst::TensorBufferPreprocessFactsOwner facts;
  };
  static rt::Result<FixedFrame> checked_fixed_frame(std::shared_ptr<const CheckedPolicy>,
                                                    std::size_t width, std::size_t height,
                                                    std::size_t row_stride);
  // The capacity envelope of the checked policy: the largest frame the stage takes (width,
  // height), which input_specs holds every frame against.
  static std::pair<std::size_t, std::size_t> checked_envelope(const CheckedPolicy&);
  // Check the complete initial image recipe before opening the graph's device.
  static rt::Result<std::shared_ptr<PreprocFrameAdapter>>
      check(std::shared_ptr<const ProcessCvuStagePayload>);
  const rt::KernelExecutable& initial_executable() const;
  rt::Status admitted(const rt::VariantPreparer&) override;
  rt::Status check_caps(const GstCaps*) const;
  // Negotiation queries validate without changing the active frame geometry.
  rt::Status accepts_caps(const GstCaps*) const;
  // Fields replaced (or cleared) by this publisher, never extension metadata.
  static bool owns_publication_field(GQuark) noexcept;
  bool replaces_publication_field(GQuark field) const noexcept override {
    return owns_publication_field(field);
  }
  // Scheduler supplies a borrowed writer over the existing frame binding and
  // current producer loans. Geometry caching retains neither writer nor buffer.
  rt::Status
  bind(GstBuffer*, rt::BindingWriter&, GstStructure* output_metadata, bool fresh_metadata = false,
       simaai::gst::TensorBufferPreprocessFactsOwner* compact_facts = nullptr) const override;

private:
  struct Geometry {
    GstVideoFormat format = GST_VIDEO_FORMAT_UNKNOWN;
    std::size_t width = 0, height = 0, batch = 1, frame_bytes = 0;
    std::array<std::size_t, 3> offsets{}, strides{};
    bool operator==(const Geometry&) const = default;
  };
  explicit PreprocFrameAdapter(std::shared_ptr<const CheckedPolicy>);
  struct IntegerList;
  static rt::Result<IntegerList> integer_list(const GstStructure*, const char*, std::size_t,
                                              bool optional = false);
  enum class PaddingPlacement { Center, TopLeft, TopRight, BottomLeft, BottomRight };
  static rt::Result<Geometry> describe_geometry(Geometry, std::size_t width, std::size_t height);
  rt::Result<std::array<std::int64_t, 4>> destination(std::int64_t source_width,
                                                      std::int64_t source_height,
                                                      std::int64_t width,
                                                      std::int64_t height) const;
  rt::Result<GstVideoInfo> parse_caps(const GstCaps*) const;
  rt::Result<std::vector<rt::TensorSpec>> input_specs(const Geometry&) const;
  // Slot order belongs to the bound geometry capability, not the frame metadata.
  enum GeometryField {
    SourceRoi,
    SourceRois,
    Destination,
    Padding,
    ResizeWindow,
    GeometryFieldCount
  };
  using GeometryParameters = std::array<std::optional<rt::ParameterValue>, GeometryFieldCount>;
  rt::Result<std::vector<rt::Parameter>> static_parameters() const;
  rt::Result<GeometryParameters> parameters(const Geometry&, const GstStructure*,
                                            const IntegerList* checked_indices = nullptr) const;
  std::shared_ptr<const CheckedPolicy> policy_;
  rt::VariantPreparer preparer_;
  std::atomic<bool> preparation_started_{false};
  // Canonical packed geometry changes on CAPS, not on each frame. VideoMeta
  // and ROI overrides are applied to a local copy and never poison this base.
  mutable Geometry default_geometry_;
  mutable std::mutex mutex_;
  struct StructureFree {
    void operator()(GstStructure* value) const noexcept {
      gst_structure_free(value);
    }
  };
  using Publication = simaai::gst::TensorBufferPreprocessFactsOwner;
  // Pure metadata composition shared by cold seeding and runtime cache misses.
  static rt::Result<Publication> compose_publication(const Publication&,
                                                     const std::array<double, 6>& upstream,
                                                     const std::array<int, 4>& upstream_roi,
                                                     int original_width, int original_height,
                                                     bool has_roi, std::string_view color_in);
  rt::Result<Publication> prepare_publication(const Geometry&, const rt::KernelExecutable&,
                                              const GstStructure*) const;
  struct Variant {
    Geometry geometry;
    std::unique_ptr<GstStructure, StructureFree> signature;
    rt::PreparedSelection operation;
    Publication publication;
  };
  // A current/previous pair prevents alternating image/ROI signatures from
  // rebuilding every frame. Immutable snapshots only, never frame storage or
  // producer loans. Unique signatures still take the fully checked path.
  mutable std::array<Variant, 2> variants_;
  struct ComposedPublication {
    Publication base, result;
    std::array<double, 6> upstream{};
    std::array<int, 4> roi{};
    int original_width = 0, original_height = 0;
    bool has_roi = false;
    std::string color_in;
  };
  // One immutable composition per local variant. No frame identity or loan is retained.
  mutable std::array<ComposedPublication, 2> composed_;
  mutable std::size_t current_variant_ = 0;
};
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
