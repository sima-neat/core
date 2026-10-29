#include "pipeline/internal/StageSourceSegmentName.h"
#include "test_main.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

RUN_TEST("unit_stage_source_segment_name_test", ([] {
           using namespace simaai::neat;
           using namespace simaai::neat::pipeline_internal;

           Tensor unnamed;
           require(tensor_source_segment_name(unnamed).empty(),
                   "an unnamed tensor must have no invented source label");
           require(tensor_wire_source_segment_name(unnamed, "").empty(),
                   "an unnamed wire must not invent the decoder label");
           require(tensor_wire_source_segment_name(unnamed, "authored_ingress") ==
                       "authored_ingress",
                   "the supplied wire name must remain the fallback for an unnamed tensor");

           int maps = 0;
           auto holder = std::make_shared<int>(0);
           auto storage = std::make_shared<TensorBuffer>();
           storage->holder = holder;
           storage->size_bytes = 1209600U;
           storage->sima_segments = {{"MLA_0", 1209600U}};
           storage->map_fn = [&](MapMode) -> Mapping {
             ++maps;
             throw std::runtime_error("wire naming must not map payload data");
           };

           constexpr std::array<std::int64_t, 6> offsets = {
               0, 409600, 512000, 537600, 1049600, 1177600,
           };
           Sample heads;
           heads.kind = SampleKind::TensorSet;
           heads.segment_name = "aggregate_label";
           for (std::size_t i = 0; i < offsets.size(); ++i) {
             Tensor tensor;
             tensor.storage = storage;
             tensor.dtype = TensorDType::Int8;
             tensor.layout = TensorLayout::HWC;
             tensor.shape = {1, 1, 1};
             tensor.strides_bytes = {1, 1, 1};
             tensor.byte_offset = offsets[i];
             tensor.route.logical_index = static_cast<int>(i);
             tensor.route.memory_index = 0;
             tensor.route.physical_index = 0;
             tensor.route.name = "head_" + std::to_string(i);
             tensor.route.segment_name = "MLA_0";
             heads.tensors.push_back(std::move(tensor));
           }
           const long storage_refs = storage.use_count();
           const long holder_refs = holder.use_count();
           require(sample_is_multi_output(heads), "the fixture must exercise multi-output input");
           require(sample_source_segment_name(heads) == "MLA_0",
                   "actual tensor carrier name must take precedence over an aggregate label");
           require(tensor_wire_source_segment_name(heads.tensors.front(), "consumer_input") ==
                       "MLA_0",
                   "appsrc naming must prefer the actual producer carrier");
           for (std::size_t i = 0; i < heads.tensors.size(); ++i) {
             const Tensor& tensor = heads.tensors[i];
             require(tensor.route.name == "head_" + std::to_string(i) &&
                         tensor.route.logical_index == static_cast<int>(i) &&
                         tensor.byte_offset == offsets[i] && tensor.storage == storage,
                     "source naming must not rewrite logical heads, offsets or ownership");
           }
           require(storage.use_count() == storage_refs && holder.use_count() == holder_refs,
                   "source naming must not retain input storage or its producer holder");

           Tensor storage_named = heads.tensors.front();
           storage_named.route.segment_name.clear();
           require(tensor_source_segment_name(storage_named) == "MLA_0",
                   "named storage metadata must remain available without a route label");
           storage_named.storage = std::make_shared<TensorBuffer>();
           storage_named.storage->sima_segments = {{"parent", 8U}, {"selected", 8U}};
           storage_named.route.memory_index = 1;
           require(tensor_source_segment_name(storage_named) == "selected",
                   "a supplied memory index must select its actual storage segment");
           storage_named.route.memory_index = -1;
           storage_named.route.physical_index = 1;
           require(tensor_source_segment_name(storage_named) == "selected",
                   "the existing physical-index fallback must remain unchanged");

           Sample sample_named;
           sample_named.kind = SampleKind::TensorSet;
           sample_named.tensors = {unnamed};
           sample_named.segment_name = "sample_carrier";
           require(sample_source_segment_name(sample_named) == "sample_carrier",
                   "Sample metadata must supply the source label when a tensor is unnamed");
           require(tensor_wire_source_segment_name(sample_named.tensors.front(),
                                                   sample_source_segment_name(sample_named)) ==
                       "sample_carrier",
                   "the appsrc name must receive the selected Sample fallback");
           sample_named.tensors.push_back(unnamed);
           require(sample_is_multi_output(sample_named) &&
                       sample_source_segment_name(sample_named) == "sample_carrier",
                   "multi-output samples must use the same optional Sample naming policy");
           sample_named.segment_name.clear();
           require(sample_source_segment_name(sample_named).empty(),
                   "a completely unnamed multi-output sample must stay unnamed");

           Sample bundle;
           bundle.kind = SampleKind::Bundle;
           bundle.fields = {heads};
           require(sample_source_segment_name(bundle) == "MLA_0",
                   "the existing first-field Bundle naming policy must remain unchanged");
           bundle.segment_name = "bundle_carrier";
           require(sample_source_segment_name(bundle) == "bundle_carrier",
                   "an explicit Bundle source label must precede first-field fallback");
           require(maps == 0, "all optional naming paths must inspect metadata only");
         }));
