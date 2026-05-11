#include "pipeline/SessionOptions.h"
#include "pipeline/Tensor.h"
#include "test_main.h"

using namespace simaai::neat;

namespace {

Tensor make_test_tensor(int logical_index, const std::string& name) {
  Tensor t;
  t.dtype = TensorDType::Float32;
  t.layout = TensorLayout::HWC;
  t.shape = {2, 2, 3};
  t.strides_bytes = {24, 12, 4};
  t.storage = make_cpu_owned_storage(48);
  t.route.logical_index = logical_index;
  t.route.physical_index = logical_index + 10;
  t.route.memory_index = logical_index + 20;
  t.route.physical_byte_offset = logical_index * 16;
  t.route.name = name;
  t.route.segment_name = name + "_segment";
  return t;
}

} // namespace

RUN_TEST("unit_tensor_list_api_test", ([] {
           Tensor a = make_test_tensor(0, "a");
           Tensor b = make_test_tensor(1, "b");

           Tensor a_clone = a.clone();
           require(a_clone.route.logical_index == a.route.logical_index,
                   "clone should preserve tensor logical index");
           require(a_clone.route.segment_name == a.route.segment_name,
                   "clone should preserve tensor segment name");

           const TensorList tensors = {a, b};
           const Sample sample = sample_from_tensors(tensors);
           require(sample.kind == SampleKind::TensorSet,
                   "sample_from_tensors should create TensorSet for multi-output");
           require(sample.tensors.size() == 2, "TensorSet sample should contain two tensors");
           require(sample.fields.empty(),
                   "TensorSet sample should not mirror tensors through legacy fields");

           const TensorList roundtrip = tensors_from_sample(sample);
           require(roundtrip.size() == 2, "tensors_from_sample should recover all tensors");
           require(roundtrip[0].route.logical_index == 0,
                   "first tensor logical index should roundtrip");
           require(roundtrip[0].route.name == "a", "first tensor name should roundtrip");
           require(roundtrip[1].route.logical_index == 1,
                   "second tensor logical index should roundtrip");
           require(roundtrip[1].route.segment_name == "b_segment",
                   "second tensor segment name should roundtrip");

           const Sample single_sample = sample_from_tensors(TensorList{a});
           require(single_sample.kind == SampleKind::TensorSet,
                   "sample_from_tensors should keep single tensor on TensorSet transport");
           require(single_sample.tensors.size() == 1U,
                   "single tensor TensorSet should contain exactly one tensor");
           const TensorList single_roundtrip = tensors_from_sample(single_sample);
           require(single_roundtrip.size() == 1, "single tensor roundtrip should have size 1");
           require(single_roundtrip[0].route.memory_index == a.route.memory_index,
                   "single tensor memory index should roundtrip");
         }));
