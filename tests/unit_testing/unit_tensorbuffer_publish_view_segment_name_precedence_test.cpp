#include <gstsimaaitensorbuffer.h>

#include "gst/SimaTensorSetMetaAbi.h"
#include "test_main.h"

#include <gst/gst.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

void ensure_tensor_set_meta_registered() {
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    const gchar* tags[] = {nullptr};
    (void)gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr);
  }
}

using GstBufferPtr = std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)>;

GstBufferPtr make_source_buffer(gsize size, guint8 fill) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
  require(buffer != nullptr, "failed to allocate source buffer");
  GstMapInfo map{};
  require(gst_buffer_map(buffer, &map, GST_MAP_WRITE), "failed to map source buffer");
  std::memset(map.data, fill, map.size);
  gst_buffer_unmap(buffer, &map);
  return GstBufferPtr(buffer, &gst_buffer_unref);
}

struct PublishStageFixture {
  std::array<gint64, 2> shape = {4, 4};
  std::array<gint64, 2> stride_bytes = {4, 1};
  std::array<SimaPluginPhysicalBuffer, 2> physical_outputs{};
  std::array<SimaPluginLogicalTensor, 1> logical_outputs{};
  std::array<SimaPluginOutputRoute, 1> output_order{};
  SimaPluginStageSpec stage{};

  PublishStageFixture() {
    physical_outputs[0].physical_index = 1;
    physical_outputs[0].size_bytes = 16;
    physical_outputs[0].segment_name = "output_rgb_image";

    physical_outputs[1].physical_index = 2;
    physical_outputs[1].size_bytes = 16;
    physical_outputs[1].segment_name = "output_tessellated_image";

    logical_outputs[0].logical_index = 1;
    logical_outputs[0].backend_output_index = 1;
    logical_outputs[0].physical_index = 1;
    logical_outputs[0].output_slot = 0;
    logical_outputs[0].tensor_index = 0;
    logical_outputs[0].byte_offset = 0;
    logical_outputs[0].size_bytes = 16;
    logical_outputs[0].shape = shape.data();
    logical_outputs[0].shape_len = static_cast<guint>(shape.size());
    logical_outputs[0].stride_bytes = stride_bytes.data();
    logical_outputs[0].stride_bytes_len = static_cast<guint>(stride_bytes.size());
    logical_outputs[0].dtype = "INT8";
    logical_outputs[0].logical_name = "output_tessellated_image";
    logical_outputs[0].backend_name = "output_tessellated_image";
    logical_outputs[0].segment_name = "output_tessellated_image";

    output_order[0].output_slot = 0;
    output_order[0].cm_output_name = "output_tessellated_image";
    output_order[0].segment_name = "output_tessellated_image";

    stage.element_name = "neatprocesscvu";
    stage.logical_stage_id = "preproc_1";
    stage.plugin_kind = "processcvu";
    stage.kernel_kind = "preproc";
    stage.physical_outputs = physical_outputs.data();
    stage.physical_outputs_len = static_cast<guint>(physical_outputs.size());
    stage.logical_outputs = logical_outputs.data();
    stage.logical_outputs_len = static_cast<guint>(logical_outputs.size());
    stage.output_order = output_order.data();
    stage.output_order_len = static_cast<guint>(output_order.size());
  }
};

} // namespace

RUN_TEST("unit_tensorbuffer_publish_view_segment_name_precedence_test", ([] {
  ensure_tensor_set_meta_registered();

  auto parent = make_source_buffer(48U, 0x11);
  auto rgb = make_source_buffer(16U, 0x22);
  auto tess = make_source_buffer(16U, 0x33);

  std::vector<simaai::gst::TensorBufferBuildSegment> segments = {
      {"__processcvu_parent__", parent.get(), 48U},
      {"output_rgb_image", rgb.get(), 16U},
      {"output_tessellated_image", tess.get(), 16U},
  };

  GstBuffer* raw_buffer = nullptr;
  std::string err;
  require(simaai::gst::tensor_buffer_build_segmented_buffer(segments, &raw_buffer, &err),
          std::string("failed to build segmented buffer: ") + err);
  GstBufferPtr buffer(raw_buffer, &gst_buffer_unref);

  PublishStageFixture fixture;
  simaai::gst::TensorBufferView publish_view;
  err.clear();
  require(simaai::gst::tensor_buffer_build_publish_view(
              buffer.get(), fixture.stage,
              simaai::gst::TensorBufferProducerKind::ProcessCvu, &publish_view, &err),
          std::string("failed to build publish view: ") + err);
  require(publish_view.segments.size() == 3U, "publish view should preserve parent/rgb/tess segments");
  require(publish_view.tensors.size() == 1U, "publish view should expose one logical tensor");
  require(publish_view.segments[1].name == "output_rgb_image",
          "publish view should preserve rgb runtime segment name");
  require(publish_view.segments[2].name == "output_tessellated_image",
          "publish view should preserve tess runtime segment name");
  require(publish_view.tensors.front().memory_index == 2,
          "publish view should resolve exposed tess handoff by runtime segment name before physical index");
  require(publish_view.tensors.front().segment_name == "output_tessellated_image",
          "publish view should preserve tess handoff segment name");

  err.clear();
  require(simaai::gst::tensor_buffer_attach_meta(buffer.get(), publish_view, &err),
          std::string("failed to attach tensorbuffer meta: ") + err);

  simaai::gst::TensorBufferView roundtrip;
  err.clear();
  require(simaai::gst::tensor_buffer_create_view(buffer.get(), nullptr, &roundtrip, &err),
          std::string("failed to recreate tensorbuffer view: ") + err);
  require(roundtrip.segments.size() == 3U, "roundtrip view should preserve runtime segment count");
  require(roundtrip.segments[1].name == "output_rgb_image",
          "roundtrip view should not rename rgb segment to tess handoff");
  require(roundtrip.segments[2].name == "output_tessellated_image",
          "roundtrip view should preserve tess runtime segment name");
  require(roundtrip.tensors.size() == 1U, "roundtrip view should expose one logical tensor");
  require(roundtrip.tensors.front().memory_index == 2,
          "roundtrip view should keep the tess logical tensor bound to the tess runtime segment");
  require(roundtrip.tensors.front().segment_name == "output_tessellated_image",
          "roundtrip view should preserve tess logical segment name");
  require(roundtrip.tensors.front().memory_index >= 0 &&
              static_cast<std::size_t>(roundtrip.tensors.front().memory_index) < roundtrip.segments.size() &&
              roundtrip.segments[static_cast<std::size_t>(roundtrip.tensors.front().memory_index)].name ==
                  roundtrip.tensors.front().segment_name,
          "roundtrip view should keep memory index and segment name aligned");
}));
