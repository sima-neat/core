#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/RouteGraph.h"
#include "test_main.h"

namespace {

using simaai::neat::pipeline_internal::sima::MpkContract;
using simaai::neat::pipeline_internal::sima::MpkContractEdge;
using simaai::neat::pipeline_internal::sima::MpkPluginIoContract;
using simaai::neat::pipeline_internal::sima::MpkShapeSemantics;
using simaai::neat::pipeline_internal::sima::MpkTensorContract;

MpkTensorContract make_tensor(const char* name) {
  MpkTensorContract tensor;
  tensor.name = name;
  tensor.dtype = "BF16";
  tensor.mpk_shape = {1, 1, 1, 1};
  tensor.logical_shape = tensor.mpk_shape;
  tensor.shape_semantics = MpkShapeSemantics::Geometry;
  tensor.size_bytes = 2U;
  return tensor;
}

} // namespace

RUN_TEST("unit_route_graph_slice_kernel_test", ([] {
           using simaai::neat::pipeline_internal::sima::RouteGraphKernelKind;

           MpkContract contract;

           MpkPluginIoContract mla;
           mla.name = "MLA_0";
           mla.processor = "MLA";
           mla.kernel = "mla";
           mla.output_tensors.push_back(make_tensor("MLA_0"));
           contract.plugins.push_back(mla);

           MpkPluginIoContract unpack;
           unpack.name = "MLA_0_ofm_unpack_transform";
           unpack.kernel = "unpack_transform";
           unpack.input_tensors.push_back(make_tensor("MLA_0"));
           unpack.output_tensors.push_back(make_tensor("MLA_0_ofm_unpack_transform_0"));
           contract.plugins.push_back(unpack);

           MpkPluginIoContract slice;
           slice.name = "slice_MLA_0/tuple_get_item_0_slice_transform";
           slice.kernel = "slice_transform";
           slice.input_tensors.push_back(make_tensor("MLA_0_ofm_unpack_transform_0"));
           slice.output_tensors.push_back(make_tensor("slice_MLA_0/tuple_get_item_0_slice_transform"));
           contract.plugins.push_back(slice);

           MpkPluginIoContract cast;
           cast.name = "cast_0";
           cast.kernel = "cast_transform";
           cast.input_tensors.push_back(make_tensor("slice_MLA_0/tuple_get_item_0_slice_transform"));
           cast.output_tensors.push_back(make_tensor("cast_0_out"));
           contract.plugins.push_back(cast);

           contract.edges.push_back(MpkContractEdge{0U, 0, 1U, 0, "MLA_0",
                                                    "MLA_0_ofm_unpack_transform", "MLA_0"});
           contract.edges.push_back(MpkContractEdge{1U, 0, 2U, 0, "MLA_0_ofm_unpack_transform",
                                                    "slice_MLA_0/tuple_get_item_0_slice_transform",
                                                    "MLA_0_ofm_unpack_transform_0"});
           contract.edges.push_back(MpkContractEdge{2U, 0, 3U, 0,
                                                    "slice_MLA_0/tuple_get_item_0_slice_transform",
                                                    "cast_0",
                                                    "slice_MLA_0/tuple_get_item_0_slice_transform"});

           const auto graph = simaai::neat::pipeline_internal::sima::build_route_graph(contract);
           require(graph.nodes.size() == 4U, "route graph should contain all four nodes");
           require(graph.nodes[0].kind == RouteGraphKernelKind::Mla,
                   "MLA node should classify as MLA");
           require(graph.nodes[1].kind == RouteGraphKernelKind::Unpack,
                   "unpack node should classify as unpack");
           require(graph.nodes[2].kind == RouteGraphKernelKind::Slice,
                   "slice_transform should classify as slice in the route graph");
           require(graph.nodes[3].kind == RouteGraphKernelKind::Cast,
                   "cast node should classify as cast");
         }));
