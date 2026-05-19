#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "test_main.h"

#include <gst/gst.h>

using namespace simaai::neat;

namespace {

void ensure_gst_ready() {
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
  const gchar* tags[] = {nullptr};
  if (gst_meta_get_info("GstSimaSampleMeta") == nullptr) {
    (void)gst_meta_register_custom("GstSimaSampleMeta", tags, nullptr, nullptr, nullptr);
  }
  if (gst_meta_get_info(SIMA_TENSOR_SET_META_NAME) == nullptr) {
    (void)gst_meta_register_custom(SIMA_TENSOR_SET_META_NAME, tags, nullptr, nullptr, nullptr);
  }
}

Tensor make_tensor(int logical_index, std::size_t bytes, const std::string& name) {
  Tensor t;
  t.dtype = TensorDType::BFloat16;
  t.layout = TensorLayout::HWC;
  t.shape = {1, static_cast<int64_t>(bytes / 2), 1};
  t.strides_bytes = {static_cast<int64_t>(bytes), 2, 2};
  t.storage = make_cpu_owned_storage(bytes);
  t.route.logical_index = logical_index;
  t.route.physical_index = logical_index;
  t.route.route_slot = logical_index;
  t.route.name = name;
  t.route.backend_name = name;
  t.route.segment_name = name;
  t.route.stage_key = "mla_join_test";
  return t;
}

} // namespace

RUN_TEST(
    "unit_sample_tensor_set_materialized_parent_offsets_test", ([] {
      ensure_gst_ready();

      Tensor y = make_tensor(0, 16U, "cast_0");
      Tensor uv = make_tensor(1, 8U, "cast_1");
      const Sample sample = sample_from_tensors(TensorList{y, uv});

      std::string err;
      auto holder = pipeline_internal::make_sample_holder_from_bundle(sample, &err);
      require(holder != nullptr, std::string("failed to materialize tensor-set holder: ") + err);

      auto* gst_sample = static_cast<GstSample*>(holder.get());
      require(gst_sample != nullptr, "materialized holder should be a GstSample");

      simaai::neat::pipeline_internal::TensorBufferView view;
      err.clear();
      require(simaai::neat::pipeline_internal::tensor_buffer_descriptor_from_sample(gst_sample,
                                                                                    &view, &err),
              std::string("failed to extract tensorbuffer descriptor: ") + err);
      require(view.tensors.size() == 2U, "materialized tensor-set should expose two tensors");
      require(view.tensors[0].physical_index == 0 && view.tensors[1].physical_index == 1,
              "materialized tensor-set should preserve logical physical indices");
      require(view.tensors[0].byte_offset == 0U, "first tensor should start at offset zero");
      if (view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 0) {
        require(view.tensors[1].byte_offset == 16U,
                "single-parent materialized tensor-set should use cumulative offsets");
        require(view.tensors[0].segment_name == view.tensors[1].segment_name,
                "single-parent materialized tensor-set should use one canonical segment name");
      } else {
        require(view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 1,
                "fallback materialized tensor-set should preserve one memory per tensor");
        require(view.tensors[1].byte_offset == 0U,
                "fallback materialized tensor-set should keep child-relative offsets");
      }

      auto shared_storage = make_cpu_owned_storage(24U);
      require(shared_storage != nullptr, "failed to allocate shared parent storage");

      Tensor joined_y;
      joined_y.dtype = TensorDType::BFloat16;
      joined_y.layout = TensorLayout::HWC;
      joined_y.shape = {1, 8, 1};
      joined_y.strides_bytes = {16, 2, 2};
      joined_y.storage = shared_storage;
      joined_y.byte_offset = 0;
      joined_y.route.logical_index = 0;
      joined_y.route.physical_index = 0;
      joined_y.route.route_slot = 0;
      joined_y.route.name = "cast_0";
      joined_y.route.backend_name = "cast_0";
      joined_y.route.segment_name = "joined_ifm";
      joined_y.route.stage_key = "mla_join_test";

      Tensor joined_uv = joined_y;
      joined_uv.shape = {1, 4, 1};
      joined_uv.strides_bytes = {8, 2, 2};
      joined_uv.byte_offset = 16;
      joined_uv.route.logical_index = 1;
      joined_uv.route.physical_index = 1;
      joined_uv.route.route_slot = 1;
      joined_uv.route.name = "cast_1";
      joined_uv.route.backend_name = "cast_1";

      const Sample joined_sample = sample_from_tensors(TensorList{joined_y, joined_uv});
      err.clear();
      auto joined_holder = pipeline_internal::make_sample_holder_from_bundle(joined_sample, &err);
      require(joined_holder != nullptr,
              std::string("failed to materialize joined-parent tensor-set holder: ") + err);

      auto* joined_gst_sample = static_cast<GstSample*>(joined_holder.get());
      require(joined_gst_sample != nullptr, "joined-parent holder should be a GstSample");
      GstCaps* joined_caps = gst_sample_get_caps(joined_gst_sample);
      require(joined_caps != nullptr, "joined-parent holder should carry GstCaps");
      GstStructure* joined_caps_struct = gst_caps_get_structure(joined_caps, 0);
      require(joined_caps_struct != nullptr, "joined-parent holder caps should have a structure");
      const gchar* media = gst_structure_get_name(joined_caps_struct);
      require(std::string(media ? media : "") == "application/vnd.simaai.tensor",
              "joined-parent holder should preserve tensor media type");
      // Joined tensor-set caps no longer carry explicit width/height
      // ints (the packed span is described via tensor metadata fields
      // instead). Smoke-check the caps structure exists.
      (void)joined_caps_struct;
    }));
