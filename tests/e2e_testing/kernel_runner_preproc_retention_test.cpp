// Core-owned dynamic geometry policy through Scheduler's generic carrier engine.
// Migrated from Scheduler's legacy stage test without changing pixel/fact assertions.
#include "gst/GstInit.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include <gst/SimaPluginStaticManifestAbi.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/client/dma_memory.h>
#include <simaai/scheduler/testing/probe.h>
#include <simaai/scheduler/client/tensor_buffer.h>
#include <gst/video/video.h>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace rt = simaai::scheduler::client;
namespace probe = simaai::scheduler::testing;
namespace tr = simaai::gst;
namespace gs = simaai::scheduler::client;
namespace dma = simaai::scheduler::client::dma;
namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T take(rt::Result<T> result) {
  if (auto* error = std::get_if<rt::Error>(&result))
    throw std::runtime_error(error->field + ": " + error->message);
  return std::get<T>(std::move(result));
}
void okay(rt::Status status) {
  if (status)
    throw std::runtime_error(status->field + ": " + status->message);
}
sima_ev_tensor_desc descriptor(bool input) {
  sima_ev_tensor_desc result{};
  result.dtype = input ? SIMA_EV_DTYPE_FP32 : SIMA_EV_DTYPE_BF16;
  result.shape.rank = 3;
  const int bytes = input ? 4 : 2;
  for (unsigned i = 0; i < 3; ++i) {
    result.shape.sizes[i] = i + 2;
    result.shape.axis_semantics[i] = SIMA_EV_AXIS_H + i;
  }
  result.layout_kind = SIMA_EV_LAYOUT_STRIDED;
  result.layout.strided.strides_bytes[0] = 12 * bytes;
  result.layout.strided.strides_bytes[1] = 4 * bytes;
  result.layout.strided.strides_bytes[2] = bytes;
  return result;
}
void actual_preproc(bool compact = false, bool full = false, bool preflight = false) {
  namespace ps = simaai::neat::pipeline_internal::sima;
  ps::StageStaticSpec authored;
  authored.element_name = authored.logical_stage_id = "preproc-retention";
  authored.frame_arena_role = ps::FrameArenaRole::Allocate;
  authored.frame_arena_size_bytes = 4096;
  authored.frame_arena_provenance = ps::static_contract::ArenaAllocationProvenance::CoreAllocated;
  authored.payload_kind = ps::StagePayloadKind::ProcessCvu;
  auto& cfg = authored.processcvu;
  cfg.canonical_contract = true;
  cfg.graph_id = 200;
  cfg.graph_name = cfg.graph_family = "preproc";
  cfg.resolved_exec_backend = "EV74";
  cfg.requested_run_target = cfg.run_target = "EV74";
  cfg.batch_size = 1;
  cfg.num_in_tensor = 1;
  cfg.input_img_type = cfg.output_img_type = "RGB";
  cfg.scaling_type = "BILINEAR";
  cfg.padding_type = "CENTER";
  cfg.aspect_ratio = 1;
  cfg.normalize = 1;
  cfg.input_dtype = "INT8";
  cfg.output_dtype = "BF16";
  cfg.channel_mean = {0.0, 0.0, 0.0};
  cfg.channel_stddev = {1.0, 1.0, 1.0};
  for (unsigned output = 0; output < 2; ++output) {
    auto d = descriptor(!output);
    if (!output)
      d.dtype = SIMA_EV_DTYPE_INT8;
    d.shape.sizes[0] = 8;
    d.shape.sizes[1] = 8;
    d.shape.sizes[2] = 3;
    d.layout.strided.strides_bytes[0] = 24 * (output ? 2 : 1);
    d.layout.strided.strides_bytes[1] = 3 * (output ? 2 : 1);
    d.layout.strided.strides_bytes[2] = output ? 2 : 1;
    // Raw RGB pixels are unsigned bytes; the output is numerical BF16.
    auto& tensors = output ? cfg.output_tensors : cfg.input_tensors;
    tensors.emplace_back(d, output ? "BF16" : "UINT8");
  }
  cfg.default_input_name = "source";
  cfg.default_output_names = {"result"};
  cfg.primary_output_name = "result";
  ps::LogicalInputStaticSpec logical_input;
  logical_input.logical_index = logical_input.backend_input_index = logical_input.physical_index =
      0;
  logical_input.shape = {8, 8, 3};
  logical_input.stride_bytes = {24, 3, 1};
  logical_input.size_bytes = 192;
  logical_input.dtype = "UINT8";
  logical_input.layout = "HWC";
  logical_input.logical_name = logical_input.backend_name = logical_input.segment_name = "source";
  authored.logical_inputs = {logical_input};
  ps::LogicalTensorStaticSpec logical_output;
  logical_output.logical_index = 23;
  logical_output.backend_output_index = logical_output.physical_index = logical_output.output_slot =
      logical_output.tensor_index = 0;
  logical_output.shape = {8, 8, 3};
  logical_output.stride_bytes = {48, 6, 2};
  logical_output.size_bytes = 384;
  logical_output.dtype = "BF16";
  logical_output.layout = "HWC";
  logical_output.logical_name = logical_output.backend_name = logical_output.segment_name =
      "result";
  authored.logical_outputs = {logical_output};
  ps::PhysicalBufferStaticSpec physical;
  physical.physical_index = physical.allocator_index = 0;
  physical.size_bytes = 384;
  physical.source_byte_offset = 128;
  physical.segment_name = "result";
  authored.physical_outputs = {physical};
  authored.output_order = {{0, 23, 0, "result", "result"}};
  ps::SimaPluginStaticManifest manifest;
  manifest.stages.push_back(std::move(authored));
  // Cold regression: a selected output's backend identity is authoritative,
  // including tessellated handoffs whose selected output is not backend zero.
  auto selected = manifest;
  selected.stages.front().logical_outputs.front().backend_output_index = 1;
  okay(ps::prepare_kernel_stage_recipes(&selected, std::nullopt, {}, {},
                                        simaai::neat::NameTransform{}));
  ps::finalize_pipeline_storage(selected);
  okay(ps::finalize_stage_publications(&selected));
  const auto& selected_recipe = *selected.stages.front().native_kernel;
  const auto& selected_publication = selected.stages.front().tensor_publication;
  require(selected_recipe.members.front().outputs.front().tensor == 1 && selected_publication &&
              selected_publication->physical_views.size() == 1 &&
              selected_publication->physical_views.front().backend_output_index == 1,
          "Preproc canonical route and physical publication must retain selected backend ID");
  okay(ps::prepare_kernel_stage_recipes(&manifest, std::nullopt, {}, {},
                                        simaai::neat::NameTransform{}));
  // Independent graph builds share checked immutable policy; only admission
  // requests acquire fresh geometry/session state, never the source recipe.
  const auto original_recipe = manifest.stages.front().native_kernel;
  const auto original_executable = original_recipe->members.front().executable;
  auto independent = manifest;
  okay(ps::prepare_kernel_stage_recipes(&independent, std::nullopt, {}, {},
                                        simaai::neat::NameTransform{}));
  require(independent.stages.front().native_kernel == original_recipe &&
              original_recipe->members.front().executable == original_executable,
          "independent graph rechecked immutable numerical policy");
  auto transformed = independent;
  transformed.stages.front().logical_stage_id = "reused-preproc-instance";
  transformed.stages.front().element_name = "reused-preproc-element";
  okay(ps::prepare_kernel_stage_recipes(&transformed, std::nullopt, {}, {},
                                        simaai::neat::NameTransform{}));
  ps::finalize_pipeline_storage(manifest);
  okay(ps::finalize_stage_publications(&manifest));
  ps::finalize_pipeline_storage(transformed);
  okay(ps::finalize_stage_publications(&transformed));
  gs::TensorStagePublication original_publication, transformed_publication;
  std::string publication_error;
  require(ps::build_kernel_stage_publication(manifest.stages.front(), &original_publication,
                                             &publication_error),
          publication_error.c_str());
  require(ps::build_kernel_stage_publication(transformed.stages.front(), &transformed_publication,
                                             &publication_error),
          publication_error.c_str());
  require(
      transformed.stages.front().native_kernel->members.front().executable == original_executable &&
          transformed_publication.contract->stage_key == "reused-preproc-instance" &&
          original_publication.contract == manifest.stages.front().tensor_publication->contract &&
          original_publication.contract->stage_key != transformed_publication.contract->stage_key &&
          transformed_publication.contract->logical_outputs.front().backend_output_index ==
              original_publication.contract->logical_outputs.front().backend_output_index &&
          transformed_publication.physical_views.front().backend_output_index ==
              original_publication.physical_views.front().backend_output_index &&
          gst_caps_is_equal(transformed_publication.input_caps.get(),
                            original_publication.input_caps.get()) &&
          gst_caps_is_equal(transformed_publication.output_caps.get(),
                            original_publication.output_caps.get()),
      "transformed graph lost instance publication identity or mutated checked recipe");
  gs::KernelProgram first_instance, second_instance;
  first_instance.members = original_recipe->members;
  second_instance.members = original_recipe->members;
  okay(ps::wire_kernel_stage_factories(*original_recipe, first_instance));
  okay(ps::wire_kernel_stage_factories(*original_recipe, second_instance));
  require(first_instance.frame_adapter_factory && second_instance.frame_adapter_factory,
          "independent requests lost immutable image factories");
  auto first_adapter = take(first_instance.frame_adapter_factory());
  auto second_adapter = take(second_instance.frame_adapter_factory());
  require(first_adapter != second_adapter, "independent requests shared mutable image state");
  std::weak_ptr<gs::TensorStageFrameAdapter> released = first_adapter;
  first_adapter.reset();
  second_adapter.reset();
  first_instance = {};
  second_instance = {};
  require(released.expired(), "retained recipe owns an instantiated image adapter");
  if (preflight) {
    std::cout << "Core Preproc canonical route and independent geometry policy: PASS\n";
    return;
  }
  const auto& recipe = *manifest.stages.front().native_kernel;
  gs::StageRequest request;
  gs::KernelProgram program;
  program.target = recipe.target;
  program.members = recipe.members;
  okay(ps::wire_kernel_stage_factories(recipe, program));
  program.constant_data = recipe.constant_data;
  request.program = std::move(program);
  request.layout = manifest.stages.front().frame_layout;
  request.publication = original_publication;
  request.options.capacity = 4;
  auto stage = take(probe::StageProbe::create(std::move(request), compact));
  // Three simultaneous outputs outlive the two-entry geometry cache. The
  // fourth output then returns to A, forcing a real preparation after eviction.
  struct Case {
    int width, height, value, roi_height;
  };
  // roi_height -1 means ordinary image mode, zero means explicit empty ROI.
  const std::array cases{Case{8, 8, 255, -1}, Case{8, 4, 128, -1}, Case{4, 8, 64, -1},
                         Case{8, 8, 192, -1}, Case{8, 8, 128, 4},  Case{8, 8, 255, 0},
                         Case{8, 8, 255, 8},  Case{8, 8, 64, -1}};
  const auto list = [](GstStructure* s, const char* name, std::initializer_list<int> entries) {
    GValue values = G_VALUE_INIT;
    g_value_init(&values, GST_TYPE_LIST);
    for (int entry : entries) {
      GValue value = G_VALUE_INIT;
      g_value_init(&value, G_TYPE_INT);
      g_value_set_int(&value, entry);
      gst_value_list_append_value(&values, &value);
      g_value_unset(&value);
    }
    gst_structure_set_value(s, name, &values);
    g_value_unset(&values);
  };
  tr::TensorBufferPublicationOwner publication_owner;
  const auto check = [&](GstBuffer* output, std::size_t index) {
    const auto& expected = cases[index];
    std::string view_error;
    rt::TensorBufferView view;
    require(rt::tensor_buffer_create_view(output, nullptr, &view, &view_error), view_error.c_str());
    require(view.tensors().size() == 1, "Preproc common reader tensor count");
    const auto& tensor = view.tensors().front();
    require(tensor.shape == std::vector<std::int64_t>({8, 8, 3}) &&
                tensor.stride_bytes == std::vector<std::int64_t>({48, 6, 2}) &&
                tensor.size_bytes == 384 && tensor.physical_span_bytes == 384 &&
                tensor.byte_offset == 128 && tensor.memory_index == 0 &&
                tensor.dtype == SIMA_TENSOR_SET_DTYPE_BF16_V1,
            "dynamic Preproc changed fixed output shape/valid extent");
    auto current = tr::tensor_buffer_publication(output, &view_error);
    require(view_error.empty() && bool(current) == compact,
            "Preproc selected wrong TensorSet representation");
    if (compact) {
      require(!gst_buffer_get_custom_meta(output, SIMA_TENSOR_SET_META_NAME),
              "compact Preproc duplicated static metadata");
      if (!publication_owner)
        publication_owner = current;
      require(current == publication_owner, "dynamic facts rebuilt frozen TensorSet owner");
    }
    const auto* current_frame = tr::tensor_buffer_frame_metadata(output, &view_error);
    const bool expected_full = full && index != 4;
    require(view_error.empty() && bool(current_frame) == expected_full,
            "Preproc full-frame/extension fallback selection");
    probe::GstBufferOwner legacy_projection;
    if (current_frame) {
      require(!gst_buffer_get_custom_meta(output, "GstSimaMeta") && current_frame->preprocess,
              "full Preproc serialized legacy facts or lost typed owner");
      legacy_projection.reset(tr::tensor_buffer_clone_envelope(output, &view_error));
      require(legacy_projection && tr::tensor_buffer_materialize_frame_metadata(
                                       legacy_projection.get(), &view_error),
              view_error.c_str());
    }
    auto* meta = gst_buffer_get_custom_meta(legacy_projection ? legacy_projection.get() : output,
                                            "GstSimaMeta");
    require(meta, "Preproc dynamic facts missing");
    auto* facts = gst_custom_meta_get_structure(meta);
    const auto integer = [&](const char* name, int wanted) {
      gint value = -1;
      require(gst_structure_get_int(facts, name, &value) && value == wanted, name);
    };
    integer("preproc_original_width", expected.width);
    integer("preproc_original_height", expected.height);
    const bool empty = expected.roi_height == 0;
    const int scaled_width = expected.width;
    const int scaled_height = expected.roi_height > 0 ? expected.roi_height : expected.height;
    if (!empty) {
      integer("preproc_resized_width", 8);
      integer("preproc_resized_height", 8);
      integer("preproc_scaled_width", scaled_width);
      integer("preproc_scaled_height", scaled_height);
      integer("preproc_pad_left", (8 - scaled_width) / 2);
      integer("preproc_pad_top", (8 - scaled_height) / 2);
    }
    if (expected.roi_height >= 0) {
      integer("preproc_roi_count", empty ? 0 : 1);
      const auto* rects = gst_structure_get_value(facts, "preproc_roi_rects");
      const auto* affines = gst_structure_get_value(facts, "preproc_roi_affines");
      require(rects && GST_VALUE_HOLDS_LIST(rects) &&
                  gst_value_list_get_size(rects) == (empty ? 0U : 4U) && affines &&
                  GST_VALUE_HOLDS_LIST(affines) &&
                  gst_value_list_get_size(affines) == (empty ? 0U : 6U),
              "Preproc ROI transition retained stale geometry records");
    } else {
      require(!gst_structure_has_field(facts, "preproc_roi_count"),
              "ordinary image publication retained ROI mode");
    }
    gint64 frame_id = -1;
    guint64 timestamp = 0;
    require(gst_structure_get_int64(facts, "frame-id", &frame_id) &&
                frame_id == gint64(33 + index) &&
                gst_structure_get_uint64(facts, "timestamp", &timestamp) &&
                timestamp == guint64(500 + index),
            "Preproc stale correlation");
    require(gst_structure_has_field(facts, "input-seq") == (index % 2 == 0),
            "Preproc optional correlation presence was not current");
    const auto extension = "retained-" + std::to_string(index);
    require(g_strcmp0(gst_structure_get_string(facts, "preproc_extension"),
                      (!full || index == 4) ? extension.c_str() : nullptr) == 0,
            "Preproc unknown extension changed or stale fallback extension survived");
    if (index == 4) {
      guint64 pcie = 0;
      require(gst_structure_get_uint64(facts, "pcie-buffer-id", &pcie) && pcie == 1234,
              "Preproc fallback discarded PCIe extension");
    } else {
      require(!gst_structure_has_field(facts, "pcie-buffer-id"), "stale PCIe fallback value");
    }
    // Empty ROI publishes zero valid regions; its capacity storage is not an
    // extra logical image, so do not infer pixel validity from stale tail bytes.
    if (empty)
      return;
    dma::Error error;
    auto result =
        dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(output, 0),
                                       std::size_t(tensor.byte_offset), tensor.size_bytes, &error);
    auto mapped = result ? result->map(dma::CpuAccess::Read, &error) : std::nullopt;
    require(bool(mapped), error.message().c_str());
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        const int left = (8 - scaled_width) / 2, top = (8 - scaled_height) / 2;
        const bool image =
            x >= left && x < left + scaled_width && y >= top && y < top + scaled_height;
        const float wanted = image ? float(expected.value) / 255.0F : 0.0F;
        for (int c = 0; c < 3; ++c) {
          std::uint16_t bits;
          std::memcpy(&bits,
                      static_cast<const std::byte*>(mapped->data()) + 2 * ((y * 8 + x) * 3 + c), 2);
          const float value = std::bit_cast<float>(std::uint32_t(bits) << 16);
          require(std::abs(value - wanted) < 0.01F,
                  "dynamic Preproc normalized/letterbox pixel mismatch");
        }
      }
    require(mapped->finish(&error), error.message().c_str());
  };
  std::vector<probe::GstBufferOwner> retained;
  std::shared_ptr<const void> first_operation;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& current = cases[index];
    auto caps = std::shared_ptr<GstCaps>(
        gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", "width", G_TYPE_INT,
                            current.width, "height", G_TYPE_INT, current.height, "framerate",
                            GST_TYPE_FRACTION, 30, 1, nullptr),
        gst_caps_unref);
    okay(stage.check_caps(0, caps.get()));
    dma::Error error;
    probe::GstBufferOwner input(dma::allocateDmaBufBuffer(dma::HeapKind::Cma, 4096, {}, &error));
    require(bool(input), error.message().c_str());
    auto image =
        dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(input.get(), 0), 0,
                                       std::size_t(current.width * current.height * 3), &error);
    auto map = image ? image->map(dma::CpuAccess::Write, &error) : std::nullopt;
    require(bool(map), error.message().c_str());
    std::memset(map->data(), current.value, current.width * current.height * 3);
    require(map->finish(&error), error.message().c_str());
    map.reset();
    require(gst_buffer_add_video_meta(input.get(), GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_RGB,
                                      current.width, current.height),
            "dynamic video meta");
    auto* meta = gst_buffer_add_custom_meta(input.get(), "GstSimaMeta");
    require(meta, "Preproc input correlation metadata");
    auto* facts = gst_custom_meta_get_structure(meta);
    const auto extension = "retained-" + std::to_string(index);
    gst_structure_set(facts, "frame-id", G_TYPE_INT64, gint64(33 + index), "timestamp",
                      G_TYPE_UINT64, guint64(500 + index), "preproc_resized_width", G_TYPE_INT, 999,
                      nullptr);
    if (!full || index == 4)
      gst_structure_set(facts, "preproc_extension", G_TYPE_STRING, extension.c_str(), nullptr);
    if (index == 4)
      gst_structure_set(facts, "pcie-buffer-id", G_TYPE_UINT64, guint64(1234), nullptr);
    if (index % 2 == 0)
      gst_structure_set(facts, "input-seq", G_TYPE_INT64, gint64(index + 100), nullptr);
    if (current.roi_height >= 0) {
      gst_structure_set(facts, "preproc_roi_count", G_TYPE_INT, current.roi_height == 0 ? 0 : 1,
                        nullptr);
      if (current.roi_height == 0) {
        list(facts, "preproc_roi_rects", {});
        list(facts, "preproc_roi_batch_indices", {});
      } else {
        list(facts, "preproc_roi_rects", {0, 0, 8, current.roi_height});
        list(facts, "preproc_roi_batch_indices", {0});
      }
    }
    const std::array inputs{input.get()};
    auto frame = take(stage.bind(inputs, full      ? probe::PublicationMode::CompactFrameV2
                                         : compact ? probe::PublicationMode::CompactTensorSetV1
                                                   : probe::PublicationMode::Legacy));
    require(frame.admitted(0), "Preproc did not use admitted storage");
    const auto operation = frame.operation_identity(0);
    require(bool(operation), "dynamic Preproc operation missing");
    if (index == 0)
      first_operation = operation;
    if (index > 0 && index <= 3)
      require(operation != first_operation,
              "Preproc geometry changed or evicted without a new preparation");
    if (index == 3)
      first_operation.reset();
    okay(frame.execute());
    okay(frame.finish());
    check(frame.output(), index);
    if (index < 3)
      retained.push_back(frame.take_output());
    for (std::size_t previous = 0; previous < retained.size(); ++previous)
      check(retained[previous].get(), previous);
  }
  stage.reset();
  for (std::size_t previous = 0; previous < retained.size(); ++previous)
    check(retained[previous].get(), previous);
  std::cout << "EV74 production Preproc "
            << (full      ? "full-compact-v2"
                : compact ? "compact-v1"
                          : "legacy")
            << ": A/B/C/A, ROI 1/0/1, retained pixels/facts/common views PASS\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    simaai::neat::gst_init_once();
    require(argc == 2, "expected --legacy, --compact, --full, or --preflight-only");
    const bool compact = std::strcmp(argv[1], "--compact") == 0;
    const bool full = std::strcmp(argv[1], "--full") == 0;
    const bool preflight = std::strcmp(argv[1], "--preflight-only") == 0;
    require(compact || full || preflight || std::strcmp(argv[1], "--legacy") == 0, "unknown mode");
    require(tr::tensor_buffer_register_meta(), "metadata registration failed");
    actual_preproc(compact || full, full, preflight);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
