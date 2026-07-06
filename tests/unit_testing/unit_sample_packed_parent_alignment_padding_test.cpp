#include "pipeline/Tensor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "test_main.h"

#include <gst/gst.h>

// Regression for issue #538 (device-tensor path): a pushed multi-input tensor
// list whose FIRST tensor is a device-backed buffer padded for alignment
// (runtime segment > logical-tight bytes, not tessellated) must be laid out in
// the packed parent by its LOGICAL-TIGHT span, not its padded transport span.
// Otherwise every subsequent tensor's byte offset drifts past what the pre-MLA
// casttess/quanttess consumer reads (it reads at logical-tight offsets), which
// on RF-DETR fed the gather bytes from feature padding -> scrambled IFM.
//
// Here the first tensor has 16 logical bytes but 24 bytes of backing storage
// (8 bytes of alignment padding). The second tensor must therefore start at
// offset 16 (tight), not 24 (padded).

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

// Logical shape yields `logical_bytes`; storage is allocated `storage_bytes`
// (>= logical) to model device alignment padding.
Tensor make_padded_tensor(int logical_index, std::size_t logical_bytes, std::size_t storage_bytes,
                          const std::string& name) {
  Tensor t;
  t.dtype = TensorDType::BFloat16;
  t.layout = TensorLayout::HWC;
  t.shape = {1, static_cast<int64_t>(logical_bytes / 2), 1};
  t.strides_bytes = {static_cast<int64_t>(logical_bytes), 2, 2};
  t.storage = make_cpu_owned_storage(storage_bytes);
  t.route.logical_index = logical_index;
  t.route.physical_index = logical_index;
  t.route.route_slot = logical_index;
  t.route.name = name;
  t.route.backend_name = name;
  t.route.segment_name = name;
  t.route.stage_key = "mla_padded_join_test";
  return t;
}

} // namespace

RUN_TEST("unit_sample_packed_parent_alignment_padding_test", ([] {
           ensure_gst_ready();

           // feature: 16 logical bytes, 24 bytes backing (8 pad); gather: 8 bytes tight.
           Tensor feature = make_padded_tensor(0, 16U, 24U, "cast_0");
           Tensor gather = make_padded_tensor(1, 8U, 8U, "cast_1");
           const Sample sample = sample_from_tensors(TensorList{feature, gather});

           std::string err;
           auto holder = pipeline_internal::make_sample_holder_from_bundle(sample, &err);
           require(holder != nullptr,
                   std::string("failed to materialize tensor-set holder: ") + err);

           auto* gst_sample = static_cast<GstSample*>(holder.get());
           require(gst_sample != nullptr, "materialized holder should be a GstSample");

           simaai::neat::pipeline_internal::TensorBufferView view;
           err.clear();
           require(simaai::neat::pipeline_internal::tensor_buffer_descriptor_from_sample(
                       gst_sample, &view, &err),
                   std::string("failed to extract tensorbuffer descriptor: ") + err);
           require(view.tensors.size() == 2U, "materialized tensor-set should expose two tensors");
           require(view.tensors[0].byte_offset == 0U, "first tensor should start at offset zero");

           // Only the single-parent (cumulative-offset) materialization is affected by
           // the packed layout; the fallback (one memory per tensor) uses child-relative
           // offsets where padding is irrelevant.
           if (view.tensors[0].memory_index == 0 && view.tensors[1].memory_index == 0) {
             require(
                 view.tensors[1].byte_offset == 16U,
                 "packed parent must lay out the second tensor at the first tensor's LOGICAL-TIGHT "
                 "offset (16), dropping the 8 bytes of alignment padding — got " +
                     std::to_string(view.tensors[1].byte_offset));
           }
         }));
