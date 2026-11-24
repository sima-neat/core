#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <simaai/scheduler/client/host_task.h>
#include <simaai/scheduler/client/stage.h>
#include <simaai/scheduler/client/metrics.h>
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

namespace simaai::neat::pipeline_internal::sima {
// Retained only by the in-process graph. The manifest's stable transport ABI
// carries placement and metadata; checked numerical recipes never serialize.
struct KernelStageRecipe {
  ::simaai::scheduler::client::Hardware target = ::simaai::scheduler::client::Hardware::A65;
  std::vector<::simaai::scheduler::client::TensorStageMember> members;
  // The request Core's lowering gave the Kernels package (check_kernel) for each member's
  // executable: the operation's name, its tensors and its parameters. Parallel to `members`
  // where the lowering records it (the canonical transforms), empty otherwise. Cold and
  // inspection only: an executable does not give its request back, and a backend that makes its
  // own kernel object for the stage (Dataflow: soc::Kernel) makes it from this same request.
  struct Request {
    std::string operation;
    std::vector<::simaai::scheduler::client::TensorSpec> inputs, outputs;
    std::vector<::simaai::scheduler::client::Parameter> parameters;
  };
  std::vector<Request> requests;
  using Resolver = ::simaai::scheduler::client::VariantResolver;
  struct ResolverFactory {
    std::size_t member = 0;
    std::function<Resolver()> instantiate;
  };
  // Factories capture immutable checked policy only, never an adapter/session,
  // admitted operation or current producer loan. Member descriptors have no
  // live resolver. Each cold request instantiates fresh state exactly once.
  std::vector<ResolverFactory> resolver_factories;
  std::function<::simaai::scheduler::client::Result<
      std::shared_ptr<::simaai::scheduler::client::TensorStageFrameAdapter>>()>
      frame_adapter_factory;
  std::vector<std::byte> constant_data;
  // Producer-owned wire identity, applied after final arena layout is known.
  // Empty for ordinary numerical tensors; never inferred from their bytes.
  std::string output_format;
  std::shared_ptr<const ::simaai::scheduler::client::TensorStageImagePublication> image_publication;
  // Internal tensor regions share the stage arena but are not graph outputs.
  // Cold preparation makes these available to dependent members, then removes
  // their metadata routes from the published tensor set.
  std::vector<LogicalTensorStaticSpec> private_outputs;
  std::vector<PhysicalBufferStaticSpec> private_buffers;
  // Optional cold semantic boundaries. No numerical ABI or per-frame metadata is changed.
  std::shared_ptr<const ::simaai::scheduler::client::StageMetrics> metrics;
  // Inspection only; not part of the Scheduler program or numerical request.
  struct BoxDecodeSettings {
    BoxDecodeType decode_type = BoxDecodeType::Unspecified;
    int num_classes = 0;
    double detection_threshold = 0;
    double nms_iou_threshold = 0;
    int topk = 0;
  };
  std::optional<BoxDecodeSettings> boxdecode_settings;
  // BoxDecode only: the stage for a backend that does not hand `members` to the Scheduler as a
  // program (Dataflow: its own kernel object, made once for the frames its input is bound to, and
  // Core's wire serializer as an operator behind it). Cold and inspection only, as `requests`
  // (which stays empty for this stage: its last member is Core's host task, not a kernel).
  struct BoxDecodeProgram {
    std::size_t member = 0; ///< the numerical member in `members`
    /// Its check_kernel request, at the cold seed's geometry (prepare_boxdecode_kernel).
    Request request;
    std::string profile, task; ///< the request's decode profile and task
    // Who says the image geometry, as attach_boxdecode_geometry_resolver installed it: the
    // original size comes with each frame's preprocess facts (the "image_size" variant), the
    // frame must state its resize mode, the profile takes each frame's affine as well.
    bool size_from_frame = false, resize_required = false, complete_geometry = false;
    std::string resize; ///< the checked decode geometry's resize mode, as preprocess facts name it
    // What the resolver holds besides: the model canvas the frames are resized onto; the cold
    // seed's image size and model-to-original transform (scale x, scale y, offset x, offset y),
    // which are the request's; whether a frame's two scales must be equal (oriented boxes) and
    // whether each frame must bring its transform.
    int model_width = 0, model_height = 0;
    int seed_width = 0, seed_height = 0;
    std::optional<std::array<float, 4>> seed_transform;
    bool uniform_transform = false, frame_transform_required = false;
    // Core's wire serializer (BoxDecodePublication.cpp), the function the stage's host task
    // calls: `inputs` are the numerical member's outputs in its order, `output` the wire of
    // `wire_bytes`. Null where the wire is not written from all of the member's outputs.
    std::shared_ptr<const void> wire;
    ::simaai::scheduler::client::HostStatus (*write)(
        const void* wire, std::span<const ::simaai::scheduler::client::ConstBytes> inputs,
        ::simaai::scheduler::client::MutableBytes output) = nullptr;
    std::size_t wire_bytes = 0;
  };
  std::optional<BoxDecodeProgram> boxdecode_program;
};
// Transfer immutable factories into a fresh request. Scheduler instantiates them
// only after validating the entire stopped cohort and resource budgets.
[[nodiscard]] ::simaai::scheduler::client::Status
wire_kernel_stage_factories(const KernelStageRecipe&, ::simaai::scheduler::client::KernelProgram&);
} // namespace simaai::neat::pipeline_internal::sima
