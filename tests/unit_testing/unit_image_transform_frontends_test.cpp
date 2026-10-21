#include <neat.h>
#include "nodes/sima/Preproc.h"
#include "nodes/sima/internal/ImageTransformPolicy.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/TensorAdapters.h"
#include "test_main.h"
#include <gst/video/video.h>
#include <memory>

namespace {
using namespace simaai::neat;
namespace lower = pipeline_internal::sima::kernel_lowering;
namespace rt = simaai::scheduler::client;
OutputSpec input(const char* format = "RGB", int width = 18, int height = 10) {
  OutputSpec spec;
  spec.media_type = "video/x-raw";
  spec.format = format;
  spec.dtype = "UInt8";
  spec.width = width;
  spec.height = height;
  spec.depth = std::string_view(format) == "GRAY8" ? 1 : 3;
  return spec;
}
Preproc& preproc(const std::shared_ptr<Node>& node) {
  auto* p = dynamic_cast<Preproc*>(node.get());
  require(p && p->kind() == "Preproc", "frontend must retain the existing Preproc holder");
  require(internal::ImageTransformAccess::policy(*p) != nullptr, "immutable policy missing");
  return *p;
}
template <class F> void rejects(F fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "invalid image frontend request accepted");
}
void contracts() {
  for (const auto format : {ImageFormat::RGB, ImageFormat::BGR, ImageFormat::GRAY8,
                            ImageFormat::NV12, ImageFormat::I420, ImageFormat::NV16}) {
    ColorConvertOptions options;
    options.output_format = format;
    auto node = nodes::ColorConvert(options);
    auto& p = preproc(node);
    const auto out = p.output_spec(input(format == ImageFormat::GRAY8 ? "GRAY8" : "RGB"));
    require(out.media_type == "video/x-raw" && out.dtype == "UInt8",
            "color output is not a raw image");
    const bool planar =
        format == ImageFormat::NV12 || format == ImageFormat::NV16 || format == ImageFormat::I420;
    require(out.layout == (planar                         ? "Planar"
                           : format == ImageFormat::GRAY8 ? "HW"
                                                          : "HWC"),
            "color output must project the raw-image plane layout");
    CompiledNodeContract compiled;
    std::string error;
    require(p.compile_node_contract({}, &compiled, &error), "color compilation: " + error);
    const auto& c = *compiled.processcvu;
    const auto count = format == ImageFormat::I420
                           ? 3U
                           : (format == ImageFormat::NV12 || format == ImageFormat::NV16 ? 2U : 1U);
    require(c.runtime_contract.physical_outputs.size() == 1, "image must have one physical parent");
    require(c.runtime_contract.logical_outputs.size() == count, "checked numerical planes lost");
    auto checked = lower::PreprocFrameAdapter::check_policy(c.payload);
    require(!std::holds_alternative<rt::Error>(checked), "adapter rejected compiled image policy");
    auto policy =
        std::get<std::shared_ptr<const lower::PreprocFrameAdapter::CheckedPolicy>>(checked);
    const auto image = lower::PreprocFrameAdapter::checked_image_publication(*policy);
    require(image && image->planes.size() == count && image->width == 18,
            "raw publication missing");
    require(c.runtime_contract.physical_outputs[0].size_bytes >=
                image->planes.back().byte_offset +
                    image->planes.back().stride * image->planes.back().visible_rows,
            "physical image does not reserve the full pitched surface");
  }
  {
    ColorConvertOptions o;
    o.output_format = ImageFormat::NV12;
    o.surface = ImageSurfaceLayout::EncoderNv12;
    auto node = nodes::ColorConvert(o);
    auto& p = preproc(node);
    p.output_spec(input());
    CompiledNodeContract compiled;
    std::string error;
    require(p.compile_node_contract({}, &compiled, &error),
            "encoder surface compilation: " + error);
    const auto& c = compiled.processcvu->runtime_contract;
    require(c.logical_outputs[0].stride_bytes[1] == 64 &&
                c.logical_outputs[1].byte_offset == 64 * 16 &&
                c.physical_outputs[0].size_bytes == 64 * 24,
            "encoder NV12 layout must use 64-byte pitch and eight-row luma storage");
  }
  ResizeOptions resize;
  resize.width = 8;
  resize.height = 8;
  resize.max_input_width = 32;
  resize.max_input_height = 24;
  auto r = nodes::Resize(resize);
  require(preproc(r).output_spec(input()).width == 8, "resize must fix output geometry");
  InputContract too_large;
  too_large.media_type = "video/x-raw";
  too_large.format = "RGB";
  too_large.width = 34;
  too_large.height = 10;
  std::string error;
  preproc(r).apply_input_contract(too_large, &error);
  require(!error.empty(), "bounded resize accepted an oversized input");
  for (const auto* format : {"RGB", "GRAY8", "NV12", "I420", "NV16"}) {
    auto resized = nodes::Resize(resize);
    auto& p = preproc(resized);
    const auto resized_spec = p.output_spec(input(format));
    const std::string token(format);
    const bool planar = token == "NV12" || token == "NV16" || token == "I420";
    require(resized_spec.layout == (planar             ? "Planar"
                                    : token == "GRAY8" ? "HW"
                                                       : "HWC"),
            "resize output must retain the raw-image plane layout");
    CompiledNodeContract compiled;
    require(p.compile_node_contract({}, &compiled, &error), "resize compilation: " + error);
    auto checked = lower::PreprocFrameAdapter::check_policy(compiled.processcvu->payload);
    require(!std::holds_alternative<rt::Error>(checked), "resize checked policy failed");
  }
  auto n = nodes::Normalization();
  const auto out = preproc(n).output_spec(input("BGR"));
  require(out.media_type == "application/vnd.simaai.tensor" && out.dtype == "BFloat16" &&
              out.format == "BF16" && out.layout == "HWC",
          "normalization must publish BF16 tensor caps, not fake raw video");
  CompiledNodeContract normalized;
  require(preproc(n).compile_node_contract({}, &normalized, &error),
          "normalization compilation: " + error);
  auto normalized_policy = lower::PreprocFrameAdapter::check_policy(normalized.processcvu->payload);
  require(!std::holds_alternative<rt::Error>(normalized_policy),
          "normalization checked policy failed");
  auto normalized_checked =
      std::get<std::shared_ptr<const lower::PreprocFrameAdapter::CheckedPolicy>>(normalized_policy);
  require(!lower::PreprocFrameAdapter::checked_image_publication(*normalized_checked),
          "BF16 tensor advertised raw image publication");
  auto inspection = simaai::scheduler::client::inspect_kernel(
      lower::PreprocFrameAdapter::checked_executable(*normalized_checked));
  require(!std::holds_alternative<rt::Error>(inspection), "normalization inspection failed");
  const auto& output_tensor = std::get<rt::KernelInspection>(inspection).outputs()[0];
  require(output_tensor.dtype == rt::DType::BFloat16 &&
              std::holds_alternative<simaai::scheduler::client::Contiguous>(output_tensor.layout),
          "normalization output must be dense BF16");
  require(preproc(n).contract_definition().outputs[0].media_type == out.media_type,
          "normalization node and boundary media disagree");
  rejects([] {
    auto n = nodes::Normalization();
    preproc(n).output_spec(input("NV12"));
  });
  rejects([] {
    NormalizationOptions o;
    o.stddev[0] = 0;
    nodes::Normalization(o);
  });
  rejects([] { nodes::Resize({}); });
  rejects([] {
    ResizeOptions o;
    o.width = 8;
    o.height = 8;
    o.pad_value = 1;
    auto n = nodes::Resize(o);
    preproc(n).output_spec(input("NV12"));
  });
  rejects([] {
    ColorConvertOptions o;
    o.color.range = ImageYuvRange::Limited;
    auto n = nodes::ColorConvert(o);
    preproc(n).output_spec(input("NV12"));
  });
}
void planar_image_boundary_chain() {
  InputOptions options;
  options.payload_type = PayloadType::Image;
  options.format = FormatTag::NV16;
  options.width = 18;
  options.height = 10;
  ResizeOptions resize;
  resize.width = 8;
  resize.height = 6;
  ColorConvertOptions convert;
  convert.output_format = ImageFormat::NV16;
  const std::vector<std::shared_ptr<Node>> chain{
      nodes::Input("nv16", options), nodes::Resize(resize), nodes::ColorConvert(convert)};
  const auto boundary = derive_output_spec(chain);
  require(boundary.format == "NV16" && boundary.layout == "Planar" &&
              boundary.byte_size == 8U * 6U * 2U,
          "NV16 image chain lost its planar boundary");
  // Use the real graph contract walk and renderer. Boundary Planar labels
  // must not reinterpret legacy internal numerical HWC port descriptors.
  ContractCompileInput compile_input;
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled = compile_node_contracts(chain, compile_input, &diagnostics);
  require(compiled.fully_renderable && diagnostics.errors.empty(),
          "NV16 planar boundary did not compose through contract compilation");
  const auto manifest =
      render_manifest_from_compiled_contracts(compiled, compile_input, &diagnostics);
  require(manifest && manifest->stages.size() == 2 && diagnostics.errors.empty(),
          "NV16 image chain failed canonical rendering");
}

void yuv_video_meta_envelopes() {
  for (const auto* format : {"NV12", "I420", "NV16"}) {
    const bool nv16 = std::string_view(format) == "NV16";
    const bool planar = std::string_view(format) == "I420";
    SampleSpec spec;
    spec.kind = SampleMediaKind::RawVideo;
    spec.media_type = "video/x-raw";
    spec.format = format;
    spec.width = 8;
    spec.height = nv16 ? 5 : 6;
    spec.required_bytes_actual = 256;
    for (unsigned i = 0; i < (planar ? 3U : 2U); ++i) {
      PlaneInfo plane;
      plane.role = i == 0   ? PlaneRole::Y
                   : planar ? (i == 1 ? PlaneRole::U : PlaneRole::V)
                            : PlaneRole::UV;
      plane.width = i && planar ? 4 : 8;
      plane.height = i && !nv16 ? spec.height / 2 : spec.height;
      plane.stride_bytes = i && planar ? 8 : 16;
      plane.offset_bytes = i == 0 ? 0 : i == 1 ? 128 : 192;
      plane.size_bytes = plane.height * plane.stride_bytes;
      spec.planes.push_back(plane);
    }
    std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> original(
        gst_buffer_new_allocate(nullptr, spec.required_bytes_actual, nullptr), gst_buffer_unref);
    auto verify = [&](GstBuffer* buffer) {
      const auto* meta = gst_buffer_get_video_meta(buffer);
      require(meta && meta->format == gst_video_format_from_string(format) && meta->width == 8 &&
                  meta->height == static_cast<guint>(spec.height) &&
                  meta->n_planes == spec.planes.size(),
              std::string(format) + " metadata format/geometry mismatch");
      for (unsigned i = 0; i < meta->n_planes; ++i)
        require(meta->offset[i] == static_cast<gsize>(spec.planes[i].offset_bytes) &&
                    meta->stride[i] == spec.planes[i].stride_bytes,
                std::string(format) + " metadata lost plane offset/pitch");
    };
    std::string error;
    auto* raw = original.get();
    require(pipeline_internal::attach_video_meta(&raw, spec, &error),
            std::string(format) + " missing VideoMeta creation: " + error);
    require(raw == original.get(), "fresh writable envelope was replaced");
    verify(raw);
    raw = gst_buffer_ref(original.get());
    const bool writable =
        pipeline_internal::ensure_writable_for_meta(&raw, spec, "YUV identity envelope", &error);
    std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> view(raw, gst_buffer_unref);
    require(writable && raw != original.get() && gst_buffer_is_writable(raw),
            std::string(format) + " writable identity envelope: " + error);
    require(gst_buffer_n_memory(raw) == 1 &&
                gst_buffer_peek_memory(raw, 0) == gst_buffer_peek_memory(original.get(), 0),
            std::string(format) + " identity envelope copied payload");
    const auto* parent = gst_buffer_get_parent_buffer_meta(raw);
    require(parent && parent->buffer == original.get(), "identity envelope lost parent owner");
    require(pipeline_internal::attach_video_meta(&raw, spec, &error),
            std::string(format) + " identity envelope VideoMeta: " + error);
    require(raw == view.get(), "writable identity envelope was replaced");
    verify(raw);
    SampleSpec wrong = spec;
    wrong.planes[1].height = nv16 ? spec.height / 2 : spec.height;
    std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> invalid(
        gst_buffer_new_allocate(nullptr, spec.required_bytes_actual, nullptr), gst_buffer_unref);
    raw = invalid.get();
    require(!pipeline_internal::attach_video_meta(&raw, wrong, &error),
            std::string(format) + " accepted invalid chroma plane height");
    require(raw == invalid.get(), "invalid metadata request unexpectedly replaced the buffer");
  }
}
void nv16_sample() {
  // One padded image, not two unrelated tensors; no payload map is needed to import it.
  auto* buffer = gst_buffer_new_allocate(nullptr, 256, nullptr);
  gsize offsets[GST_VIDEO_MAX_PLANES]{0, 128};
  gint strides[GST_VIDEO_MAX_PLANES]{16, 16};
  require(gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV16,
                                         8, 5, 2, offsets, strides),
          "video meta creation failed");
  auto* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV16", "width",
                                   G_TYPE_INT, 8, "height", G_TYPE_INT, 5, nullptr);
  auto* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  auto tensor = from_gst_sample(sample);
  gst_sample_unref(sample);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(tensor.semantic.image && tensor.semantic.image->format == ImageType::NV16,
          "NV16 semantic format lost");
  require(tensor.planes.size() == 2 && tensor.planes[1].shape == std::vector<int64_t>({5, 8}) &&
              tensor.planes[1].byte_offset == 128 && tensor.planes[1].strides_bytes[0] == 16,
          "NV16 full-height chroma or authoritative plane offsets lost");
  std::string error;
  require(tensor.validate(&error), "NV16 validation: " + error);
  const auto spec = describe_tensor_spec_or_throw(tensor, {}, "NV16 round trip");
  require(spec.format == "NV16" && spec.width == 8 && spec.height == 5 && spec.planes.size() == 2 &&
              spec.planes[1].height == 5 && spec.planes[1].stride_bytes == 16 &&
              spec.planes[1].offset_bytes == 128 && spec.required_bytes_actual == 256,
          "ingress description lost NV16 padded storage or full-height UV");
  pipeline_internal::GstBufferBuildPolicy policy;
  policy.require_video_meta = true;
  auto* round_buffer =
      pipeline_internal::build_gst_buffer_from_tensor(tensor, spec, policy, &error);
  require(round_buffer, "NV16 buffer round trip: " + error);
  auto* original_sample = static_cast<GstSample*>(tensor.storage->holder.get());
  auto* original_buffer = gst_sample_get_buffer(original_sample);
  require(gst_buffer_n_memory(round_buffer) == 1 &&
              gst_buffer_peek_memory(round_buffer, 0) == gst_buffer_peek_memory(original_buffer, 0),
          "NV16 round trip copied or replaced the payload allocation");
  auto* round_caps = caps_from_spec(spec);
  auto* round_sample = gst_sample_new(round_buffer, round_caps, nullptr, nullptr);
  auto round_tensor = from_gst_sample(round_sample);
  gst_sample_unref(round_sample);
  gst_caps_unref(round_caps);
  gst_buffer_unref(round_buffer);
  require(round_tensor.validate(&error) && round_tensor.planes.size() == 2 &&
              round_tensor.planes[1].shape == std::vector<int64_t>({5, 8}) &&
              round_tensor.planes[1].byte_offset == 128 &&
              round_tensor.planes[1].strides_bytes == std::vector<int64_t>({16, 1}),
          "NV16 round-trip sample lost authoritative plane geometry");
  // Appsrc stamps frame identity on a borrowed holder. Its writable metadata
  // envelope shares payload memory but must recreate VideoMeta afterwards.
  auto* writable = gst_buffer_ref(original_buffer);
  const bool made_writable = pipeline_internal::ensure_writable_for_meta(
      &writable, spec, "NV16 identity envelope", &error);
  std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> writable_owner(writable,
                                                                         gst_buffer_unref);
  require(made_writable && writable != original_buffer && gst_buffer_is_writable(writable),
          "NV16 writable identity envelope: " + error);
  require(gst_buffer_n_memory(writable) == 1 &&
              gst_buffer_peek_memory(writable, 0) == gst_buffer_peek_memory(original_buffer, 0),
          "NV16 identity envelope copied or replaced the payload allocation");
  const bool attached = pipeline_internal::attach_video_meta(&writable, spec, &error);
  require(writable == writable_owner.get(), "writable envelope was unexpectedly replaced");
  require(attached, "NV16 identity envelope VideoMeta: " + error);
  const auto* video_meta = gst_buffer_get_video_meta(writable);
  require(video_meta && video_meta->format == GST_VIDEO_FORMAT_NV16 && video_meta->width == 8 &&
              video_meta->height == 5 && video_meta->n_planes == 2 &&
              video_meta->offset[1] == 128 && video_meta->stride[0] == 16 &&
              video_meta->stride[1] == 16,
          "NV16 identity envelope lost padded full-height UV layout");
  tensor.planes[1].shape[0] = 2;
  require(!tensor.validate(&error), "NV16 accepted half-height chroma");
}
} // namespace
RUN_TEST("image transform frontends", [] {
  gst_init(nullptr, nullptr);
  contracts();
  planar_image_boundary_chain();
  nv16_sample();
  yuv_video_meta_envelopes();
})
