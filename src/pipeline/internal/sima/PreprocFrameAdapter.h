#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <gst/video/video.h>
#include <gstsimaaitensorbuffer.h>
#include <atomic>
#include <mutex>
#include <simaai/runtime/kernel_preparation.h>
namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
using namespace ::simaai::runtime;
// Core-owned image geometry, caps and metadata policy. Numerical validation
// and ROI filtering remain in kernel-owned generic preparation.
class PreprocFrameAdapter : public gstreamer::TensorStageFrameAdapter {
public:
  // Immutable numerical/image policy; contains no admission, session or cache.
  struct CheckedPolicy;
  static Result<std::shared_ptr<const CheckedPolicy>>
      check_policy(std::shared_ptr<const ProcessCvuStagePayload>);
  static Result<std::shared_ptr<const CheckedPolicy>> check_policy(const ProcessCvuStagePayload&);
  static Result<std::shared_ptr<PreprocFrameAdapter>>
      instantiate(std::shared_ptr<const CheckedPolicy>);
  static const KernelExecutable& checked_executable(const CheckedPolicy&);
  static Result<std::shared_ptr<PreprocFrameAdapter>>
      create(std::shared_ptr<const ProcessCvuStagePayload>, KernelSessionOptions);
  // Check the complete initial image recipe before opening the graph's device.
  static Result<std::shared_ptr<PreprocFrameAdapter>>
      check(std::shared_ptr<const ProcessCvuStagePayload>);
  const KernelExecutable& initial_executable() const;
  Status prepare(KernelSessionOptions);
  const KernelSession& session() const {
    return session_;
  }
  Status check_caps(const GstCaps*) const;
  // Negotiation queries validate without changing the active frame geometry.
  Status accepts_caps(const GstCaps*) const;
  // Fields replaced (or cleared) by this publisher, never extension metadata.
  static bool owns_publication_field(GQuark) noexcept;
  bool replaces_publication_field(GQuark field) const noexcept override {
    return owns_publication_field(field);
  }
  // Called synchronously for each checked input-plane span. The stage supplies
  // its common allocation admission/current-loan path; this callback and its
  // context are never retained. The appender retains each current read loan in
  // its frame or binding; cached geometry never substitutes for that loan. The
  // logical plane index avoids reconstructing port associations from view values.
  using PlaneResolver = gstreamer::TensorStageFrameAdapter::PlaneResolver;
  // fresh_metadata is a local proof: the destination was just created and
  // no fields owned by this publisher were copied into it. Never cache it.
  Status bind(GstBuffer*, PlaneResolver, gstreamer::Binding&, GstStructure* output_metadata,
              bool fresh_metadata = false,
              simaai::gst::TensorBufferPreprocessFactsOwner* compact_facts = nullptr) const;
  // Ordinary imports remain available for providers without reusable admission.
  Status bind(GstBuffer*, const gstreamer::Storage&, gstreamer::Binding&,
              GstStructure* output_metadata, bool fresh_metadata = false,
              simaai::gst::TensorBufferPreprocessFactsOwner* compact_facts = nullptr) const;

private:
  struct Geometry {
    GstVideoFormat format = GST_VIDEO_FORMAT_UNKNOWN;
    std::size_t width = 0, height = 0, batch = 1, frame_bytes = 0;
    std::array<std::size_t, 3> offsets{}, strides{};
    bool operator==(const Geometry&) const = default;
  };
  explicit PreprocFrameAdapter(std::shared_ptr<const CheckedPolicy>);
  struct IntegerList;
  static Result<IntegerList> integer_list(const GstStructure*, const char*, std::size_t,
                                          bool optional = false);
  enum class PaddingPlacement { Center, TopLeft, TopRight, BottomLeft, BottomRight };
  static Result<Geometry> describe_geometry(Geometry, std::size_t width, std::size_t height);
  Result<std::array<std::int64_t, 4>> destination(std::int64_t source_width,
                                                  std::int64_t source_height, std::int64_t width,
                                                  std::int64_t height) const;
  Result<GstVideoInfo> parse_caps(const GstCaps*) const;
  Result<std::vector<TensorSpec>> input_specs(const Geometry&) const;
  // Slot order belongs to the bound geometry capability, not the frame metadata.
  enum GeometryField { SourceRoi, SourceRois, Destination, Padding, GeometryFieldCount };
  using GeometryParameters = std::array<std::optional<ParameterValue>, GeometryFieldCount>;
  Result<std::vector<Parameter>> static_parameters() const;
  Result<GeometryParameters> parameters(const Geometry&, const GstStructure*,
                                        const IntegerList* checked_indices = nullptr) const;
  std::shared_ptr<const CheckedPolicy> policy_;
  KernelSession session_;
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
  Result<Publication> prepare_publication(const Geometry&, const KernelExecutable&,
                                          const GstStructure*) const;
  struct Variant {
    Geometry geometry;
    std::unique_ptr<GstStructure, StructureFree> signature;
    std::shared_ptr<const PreparedOperation> operation;
    Publication publication;
  };
  // A current/previous pair prevents alternating image/ROI signatures from
  // rebuilding every frame. Immutable snapshots only, never frame storage or
  // producer loans. Unique signatures still take the fully checked path.
  mutable std::array<Variant, 2> variants_;
  mutable std::size_t current_variant_ = 0;
};
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
