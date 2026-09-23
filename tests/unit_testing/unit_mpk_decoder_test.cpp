#define SIMA_NEAT_INTERNAL 1
#include "gst/SimaPluginStaticManifestAbi.h"
#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/KernelRegistry.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::static_contract;

void check(const bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

MlaElfIoTopology monolithic_topology(const std::uint64_t ifm_extent = 8U,
                                     const std::uint64_t ofm_extent = 8U) {
  MlaElfIoTopology topology;
  topology.valid = true;
  topology.monolithic_ifm = true;
  topology.monolithic_ofm = true;
  topology.monolithic_ifm_extent_bytes = ifm_extent;
  topology.monolithic_ofm_extent_bytes = ofm_extent;
  topology.source_path = "synthetic.elf";
  return topology;
}

const std::string& valid_manifest() {
  static const std::string manifest = R"json({
    "name":"synthetic",
    "model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {
        "name":"cast0","sequence":1,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"cast_transform","params":{"out_dtype":"bfloat16",
            "input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
        "input_nodes":[{"name":"input","size":16}],
        "output_nodes":[{"name":"cast0","size":8}]
      },
      {
        "name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "number_of_quads_to_user":4},
        "input_nodes":[{"name":"cast0","size":8}],
        "output_nodes":[{"name":"mla0","size":8}],
        "resources":{"executable":"synthetic_mla.elf"}
      },
      {
        "name":"cast1","sequence":3,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"cast_transform","params":{"out_dtype":"float32",
            "input_shapes":[[1,4]],"output_shapes":[[1,4]]}},
        "input_nodes":[{"name":"mla0","size":8}],
        "output_nodes":[{"name":"decorated/model/output:0","size":16}]
      },
      {
        "name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"decorated/model/output:0","size":16}],
        "output_nodes":[{"name":"pass_through_out_0","size":16}]
      }
    ]
  })json";
  return manifest;
}

const std::string& packed_read_manifest() {
  static const std::string manifest = R"json({
    "name":"packed-read-synthetic",
    "model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"image0","size":64},{"name":"image1","size":64}],
    "plugins":[
      {"name":"quant1","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"quantization_transform","params":{
                          "channel_params":[[0.25,0]],"num_bits":8,
                          "rounding":"TONEAREST","output_data_type":"int8",
                          "input_shapes":[[1,1,1,16]],"output_shapes":[[1,1,1,16]]}},
       "input_nodes":[{"name":"image1","size":64}],
       "output_nodes":[{"name":"input1","size":16}]},
      {"name":"quant0","sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"quantization_transform","params":{
                          "channel_params":[[0.25,0]],"num_bits":8,
                          "rounding":"TONEAREST","output_data_type":"int8",
                          "input_shapes":[[1,1,1,16]],"output_shapes":[[1,1,1,16]]}},
       "input_nodes":[{"name":"image0","size":64}],
       "output_nodes":[{"name":"input0","size":16}]},
      {
        "name":"pack","sequence":3,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pack_transform","params":{"input_shapes":[[1,16],[1,16]],
            "output_shapes":[[1,32]]}},
        "input_nodes":[{"name":"input0","size":16},{"name":"input1","size":16}],
        "output_nodes":[{"name":"packed_ifm","size":32}]
      },
      {
        "name":"MLA_0","sequence":4,"processor":"MLA","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "number_of_quads_to_user":4},
        "input_nodes":[{"name":"packed_ifm","size":32}],
        "output_nodes":[{"name":"packed_ofm","size":32}],
        "resources":{"executable":"synthetic_mla.elf"}
      },
      {
        "name":"unpack","sequence":5,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"unpack_transform","params":{
            "tensor_types":["int8","int8"],
            "tensor_shapes":[[1,2,2,4],[1,2,2,4]],
            "input_shapes":[[1,32]],
            "output_shapes":[[1,2,2,4],[1,2,2,4]]}},
        "input_nodes":[{"name":"packed_ofm","size":32}],
        "output_nodes":[{"name":"unpacked0","size":16},{"name":"unpacked1","size":16}]
      },
      {
        "name":"slice0","sequence":6,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,0],"end":[1,2,2,1],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked0","size":16}],
        "output_nodes":[{"name":"slice0_out","size":4}]
      },
      {
        "name":"slice1","sequence":7,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,1],"end":[1,2,2,2],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked1","size":16}],
        "output_nodes":[{"name":"slice1_out","size":4}]
      },
      {
        "name":"publish","sequence":8,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"slice0_out","size":4},{"name":"slice1_out","size":4}],
        "output_nodes":[{"name":"pass_through_out_0","size":4},
          {"name":"pass_through_out_1","size":4}]
      }
    ]
  })json";
  return manifest;
}

const std::string& two_mla_manifest() {
  static const std::string manifest = R"json({
    "name":"two-mla-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"encoded","size":32}],
       "resources":{"executable":"encoder.so"}},
      {"name":"MLA_decoder","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"encoded","size":32}],
       "output_nodes":[{"name":"decoded","size":8}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":8}],
       "output_nodes":[{"name":"output","size":8}]}
    ]
  })json";
  return manifest;
}

const std::string& two_mla_with_a65_module_manifest() {
  static const std::string manifest = R"json({
    "name":"two-mla-a65-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_encoder","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"encoded","size":32}],
       "resources":{"executable":"encoder.elf"}},
      {"name":"APU_module","sequence":2,"processor":"A65","type":"sgpProcess",
       "config_params":{"input_names":["arm_3_i0"],
                        "input_types":[{"scalar":"float32","shape":[1,8]}],
                        "output_types":[{"scalar":"float32","shape":[1,8]}]},
       "input_nodes":[{"name":"encoded","size":32}],
       "output_nodes":[{"name":"transformed","size":32}],
       "resources":{"executable":"middle.so"}},
      {"name":"MLA_decoder","sequence":3,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"transformed","size":32}],
       "output_nodes":[{"name":"decoded","size":8}],
       "resources":{"executable":"decoder.elf"}},
      {"name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"decoded","size":8}],
       "output_nodes":[{"name":"output","size":8}]}
    ]
  })json";
  return manifest;
}

const std::string& reshape_manifest() {
  static const std::string manifest = R"json({
    "name":"reshape-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"reshape","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"reshape_transform","params":{
                          "newshape":[1,1,4],"input_shapes":[[1,4]],
                          "output_shapes":[[1,1,4]]}},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"reshaped","size":16}]},
      {"name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"reshaped","size":16}],
       "output_nodes":[{"name":"mla_out","size":8}],
       "resources":{"executable":"synthetic_mla.elf"}},
      {"name":"publish","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"mla_out","size":8}],
       "output_nodes":[{"name":"output","size":8}]}
    ]
  })json";
  return manifest;
}

const std::string& detess_dequant_manifest() {
  static const std::string manifest = R"json({
    "name":"detess-layout-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"mla_out","size":16}],
       "resources":{"executable":"synthetic_mla.elf"}},
      {"name":"detess","sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"detessellation_transform","params":{
                          "slice_shape":[1,1,4],"frame_shape":[1,2,2,4],
                          "align_c16":false,"cblock":false,"frame_type":"int8",
                          "input_shapes":[[1,16]],"output_shapes":[[1,2,2,4]]}},
       "input_nodes":[{"name":"mla_out","size":16}],
       "output_nodes":[{"name":"detess_out","size":16}]},
      {"name":"dequant","sequence":3,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"dequantization_transform","params":{
                          "channel_params":[[1.0,0]],
                          "input_data_type":"int8",
                          "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,4]]}},
       "input_nodes":[{"name":"detess_out","size":16}],
       "output_nodes":[{"name":"dense_out","size":64}]},
      {"name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"dense_out","size":64}],
       "output_nodes":[{"name":"public_out","size":64}]}
    ]
  })json";
  return manifest;
}

const std::string& resnet_batch_flatten_manifest() {
  // Exact shapes and byte extents from resnet_50_mpk.json in the production
  // AFE 2.1.0 package. Batch flatten changes only the logical rank between
  // detessellation and dequantization.
  static const std::string manifest = R"json({
    "name":"resnet-batch-flatten","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"MLA_0","size":1008}],
       "resources":{"executable":"synthetic_mla.elf"}},
      {"name":"detessellate_MLA_0_detessellation_transform","sequence":2,
       "processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"detessellation_transform","params":{
                          "slice_shape":[1,1,1000],"frame_shape":[1,1,1,1000],
                          "align_c16":true,"cblock":true,"frame_type":"int8",
                          "input_shapes":[[1,1008]],
                          "output_shapes":[[1,1,1,1000]]}},
       "input_nodes":[{"name":"MLA_0","size":1008}],
       "output_nodes":[{"name":"detessellate_MLA_0_detessellation_transform",
                         "size":1000}]},
      {"name":"EV_1/batch_flatten_0","sequence":3,"processor":"EV74",
       "type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"batch_flatten_transform","params":{
                          "input_shapes":[[1,1,1,1000]],
                          "output_shapes":[[1,1000]]}},
       "input_nodes":[{"name":"detessellate_MLA_0_detessellation_transform",
                        "size":1000}],
       "output_nodes":[{"name":"EV_1/batch_flatten_0","size":1000}]},
      {"name":"dequantize_1","sequence":4,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"dequantization_transform","params":{
                          "channel_params":[[7.26249308476883,-67]],
                          "input_data_type":"int8","input_shapes":[[1,1000]],
                          "output_shapes":[[1,1000]]}},
       "input_nodes":[{"name":"EV_1/batch_flatten_0","size":1000}],
       "output_nodes":[{"name":"dequantize_1/resnetv17_dense0_fwd","size":4000}]},
      {"name":"publish","sequence":5,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"pass_through","params":{}},
       "input_nodes":[{"name":"dequantize_1/resnetv17_dense0_fwd","size":4000}],
       "output_nodes":[{"name":"output","size":4000}]}
    ]
  })json";
  return manifest;
}

const std::string& cast_tess_manifest() {
  static const std::string manifest = R"json({
    "name":"cast-tess-layout-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"image","size":256}],
    "plugins":[
      {"name":"cast","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast_transform","params":{
                          "out_dtype":"bfloat16",
                          "input_shapes":[[1,2,2,16]],
                          "output_shapes":[[1,2,2,16]]}},
       "input_nodes":[{"name":"image","size":256}],
       "output_nodes":[{"name":"cast_out","size":128}]},
      {"name":"tess","sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"tessellation_transform","params":{
                          "slice_shape":[2,1,16],"align_c16":false,"cblock":false,
                          "frame_type":"bfloat16",
                          "input_shapes":[[1,2,2,16]],"output_shapes":[[1,128]]}},
       "input_nodes":[{"name":"cast_out","size":128}],
       "output_nodes":[{"name":"tess_out","size":128}]},
      {"name":"MLA_0","sequence":3,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"tess_out","size":128}],
       "output_nodes":[{"name":"mla_out","size":16}],
       "resources":{"executable":"synthetic_mla.elf"}}
    ]
  })json";
  return manifest;
}

const std::string& quant_tess_manifest() {
  static const std::string manifest = R"json({
    "name":"quant-tess-layout-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"image","size":256}],
    "plugins":[
      {"name":"quant","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"quantization_transform","params":{
                          "channel_params":[[0.25,0]],"num_bits":8,
                          "rounding":"TONEAREST","output_data_type":"int8",
                          "input_shapes":[[1,2,2,16]],
                          "output_shapes":[[1,2,2,16]]}},
       "input_nodes":[{"name":"image","size":256}],
       "output_nodes":[{"name":"quant_out","size":64}]},
      {"name":"tess","sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"tessellation_transform","params":{
                          "slice_shape":[2,1,16],"align_c16":false,"cblock":false,
                          "frame_type":"int8",
                          "input_shapes":[[1,2,2,16]],"output_shapes":[[1,64]]}},
       "input_nodes":[{"name":"quant_out","size":64}],
       "output_nodes":[{"name":"tess_out","size":64}]},
      {"name":"MLA_0","sequence":3,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"tess_out","size":64}],
       "output_nodes":[{"name":"mla_out","size":16}],
       "resources":{"executable":"synthetic_mla.elf"}}
    ]
  })json";
  return manifest;
}

const std::string& yolov8_quant_tess_ingress_manifest() {
  static const std::string manifest = R"json({
    "name":"yolov8-quant-tess-ingress","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"images","size":4915200}],
    "plugins":[
      {"name":"quantize_0","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"quantization_transform","params":{
                          "channel_params":[[255.06967737092486,-128]],"num_bits":8,
                          "rounding":"TONEAREST","output_data_type":"int8",
                          "input_shapes":[[1,640,640,3]],
                          "output_shapes":[[1,640,640,3]]}},
       "input_nodes":[{"name":"images","size":4915200}],
       "output_nodes":[{"name":"quantize_0","size":1228800}]},
      {"name":"tessellate_quantize_0_MLA_0/placeholder_0_0_tessellation_transform",
       "sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"tessellation_transform","params":{
                          "slice_shape":[640,32,3],"align_c16":false,"cblock":false,
                          "frame_type":"int8","input_shapes":[[1,640,640,3]],
                          "output_shapes":[[1,1228800]]}},
       "input_nodes":[{"name":"quantize_0","size":1228800}],
       "output_nodes":[{
         "name":"tessellate_quantize_0_MLA_0/placeholder_0_0_tessellation_transform",
         "size":1228800}]},
      {"name":"MLA_0","sequence":3,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{
         "name":"tessellate_quantize_0_MLA_0/placeholder_0_0_tessellation_transform",
         "size":1228800}],
       "output_nodes":[{"name":"mla_out","size":16}],
       "resources":{"executable":"synthetic_mla.elf"}}
    ]
  })json";
  return manifest;
}

const std::string& standalone_quant_mla_manifest() {
  static const std::string manifest = R"json({
    "name":"standalone-quant-layout-synthetic","model_sdk_version":"2.0.0",
    "input_nodes":[{"name":"image","size":256}],
    "plugins":[
      {"name":"quant","sequence":1,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"quantization_transform","params":{
                          "channel_params":[[0.25,0]],"num_bits":8,
                          "rounding":"TONEAREST","output_data_type":"int8",
                          "input_shapes":[[1,2,2,16]],
                          "output_shapes":[[1,2,2,16]]}},
       "input_nodes":[{"name":"image","size":256}],
       "output_nodes":[{"name":"quant_out","size":64}]},
      {"name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4},
       "input_nodes":[{"name":"quant_out","size":64}],
       "output_nodes":[{"name":"mla_out","size":16}],
       "resources":{"executable":"synthetic_mla.elf"}}
    ]
  })json";
  return manifest;
}

const std::string& qmla_padded_output_manifest() {
  static const std::string manifest = R"json({
    "name":"qmla-padded","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_171","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"float32","shape":[1,4]}],
                        "output_types":[{"scalar":"float32","shape":[1,300,91]}]},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"pred_logits","size":109200}],
       "resources":{"executable":"MLA_171.elf"}}
    ]
  })json";
  return manifest;
}

MlaElfIoTopology qmla_padded_topology(const std::uint64_t ofm_extent = 110400U) {
  MlaElfIoTopology topology;
  topology.valid = true;
  topology.ifm_symbol_names = {"data.ifm.persistent.afe_direct_input_0.b0"};
  topology.ifm_extent_bytes = {16U};
  topology.ofm_symbol_names = {"data.ofm.persistent.afe_mla_output_0.b0"};
  topology.ofm_extent_bytes = {ofm_extent};
  topology.source_path = "MLA_171.elf";
  return topology;
}

std::string replace_once(std::string value, const std::string& before, const std::string& after) {
  const auto position = value.find(before);
  check(position != std::string::npos, "test mutation token exists");
  value.replace(position, before.size(), after);
  return value;
}

void expect_error(const std::string& manifest, const MlaElfIoTopology& topology,
                  const MpkDecodeErrorCode expected, const char* message) {
  const auto result = MpkDecoder{}.decode_json(manifest, topology, "synthetic.json");
  check(!result, message);
  check(result.error.has_value() && result.error->code == expected, message);
  check(!result.error->json_path.empty() && !result.error->detail.empty(),
        "failure has stable path and detail");
}

void test_exact_registry() {
  struct ExpectedKernel {
    const char* processor;
    const char* kernel;
    OpKind kind;
    std::size_t inputs;
    std::size_t outputs;
  };
  constexpr std::array<ExpectedKernel, 14> capabilities = {{
      {"EV74", "cast_transform", OpKind::Cast, 1, 1},
      {"EV74", "cast", OpKind::Cast, 1, 1},
      {"EV74", "quantization_transform", OpKind::Quantize, 1, 1},
      {"EV74", "tessellation_transform", OpKind::Tessellate, 1, 1},
      {"EV74", "pack_transform", OpKind::Pack, 2, 1},
      {"MLA", "", OpKind::Mla, 2, 3},
      {"EV74", "unpack_transform", OpKind::Unpack, 1, 6},
      {"EV74", "slice_transform", OpKind::Slice, 1, 1},
      {"EV74", "reshape_transform", OpKind::Reshape, 1, 1},
      {"EV74", "batch_flatten_transform", OpKind::Reshape, 1, 1},
      {"EV74", "detessellation_transform", OpKind::Detessellate, 1, 1},
      {"EV74", "dequantization_transform", OpKind::Dequantize, 1, 1},
      {"A65", "", OpKind::HostTvm, 2, 3},
      {"EV74", "pass_through", OpKind::PassThrough, 6, 6},
  }};
  for (const auto& expected : capabilities) {
    const auto descriptor = lookup_exact_kernel(expected.processor, expected.kernel);
    check(descriptor && descriptor->kind == expected.kind,
          "supported processor/kernel pair resolves exactly");
    check(exact_kernel_arity_is_valid(*descriptor, expected.inputs, expected.outputs),
          "supported operation arity is accepted");
    check(!exact_kernel_arity_is_valid(*descriptor, 0U, expected.outputs) &&
              !exact_kernel_arity_is_valid(*descriptor, expected.inputs, 0U),
          "empty operation ports are rejected");
  }
  check(!lookup_exact_kernel("EV74", "prefix_cast_transform"),
        "registry rejects substring matches");
  check(!lookup_exact_kernel("ev74", "cast_transform"), "registry rejects processor case folding");
  check(!lookup_exact_kernel("EV74", "CAST"), "registry rejects kernel case folding");
  check(!lookup_exact_kernel("unknown", ""), "registry rejects unknown processors");
  check(!exact_kernel_arity_is_valid(*lookup_exact_kernel("EV74", "pack_transform"), 1U, 1U),
        "pack requires multiple inputs");
  check(!exact_kernel_arity_is_valid(*lookup_exact_kernel("EV74", "unpack_transform"), 2U, 1U),
        "unpack requires one input");
  check(!exact_kernel_arity_is_valid(*lookup_exact_kernel("EV74", "pass_through"), 2U, 1U),
        "pass-through requires equal port counts");
}

void test_explicit_cast_input_dtype() {
  const auto topology = monolithic_topology();
  const auto legacy = MpkDecoder{}.decode_json(valid_manifest(), topology);
  check(static_cast<bool>(legacy), "legacy casts without in_dtype remain admitted");
  auto manifest = nlohmann::json::parse(valid_manifest());
  manifest["plugins"][0]["config_params"]["params"]["in_dtype"] = "float32";
  manifest["plugins"][2]["config_params"]["params"]["in_dtype"] = "bfloat16";
  const auto explicit_casts = MpkDecoder{}.decode_json(manifest.dump(), topology);
  check(static_cast<bool>(explicit_casts),
        "explicit FP32/BF16 casts in both directions are admitted");
  check(explicit_casts.plan->values().size() == legacy.plan->values().size(),
        "explicit source dtype does not change the admitted values");
  for (std::size_t index = 0; index < legacy.plan->values().size(); ++index) {
    check(explicit_casts.plan->values()[index].logical_dtype ==
              legacy.plan->values()[index].logical_dtype,
          "explicit source dtype preserves the inferred value dtypes");
  }
  for (const auto plugin_index : {0, 2}) {
    for (const auto& bad_dtype : {"int8", "float16", "unknown"}) {
      auto invalid = manifest;
      invalid["plugins"][plugin_index]["config_params"]["params"]["in_dtype"] = bad_dtype;
      expect_error(invalid.dump(), topology, MpkDecodeErrorCode::ConfigurationMismatch,
                   "unsupported explicit source dtype is rejected");
    }
    auto contradictory = manifest;
    auto& params = contradictory["plugins"][plugin_index]["config_params"]["params"];
    params["in_dtype"] = params["out_dtype"];
    expect_error(contradictory.dump(), topology, MpkDecodeErrorCode::ConfigurationMismatch,
                 "same-dtype cast contradicts the registered conversion");
    for (const auto& bad_type : {nlohmann::json(nullptr), nlohmann::json(1), nlohmann::json(true),
                                 nlohmann::json::array()}) {
      auto invalid = manifest;
      invalid["plugins"][plugin_index]["config_params"]["params"]["in_dtype"] = bad_type;
      expect_error(invalid.dump(), topology, MpkDecodeErrorCode::InvalidField,
                   "explicit source dtype must be a string");
    }
    auto extra = manifest;
    extra["plugins"][plugin_index]["config_params"]["params"]["ignored"] = 1;
    expect_error(extra.dump(), topology, MpkDecodeErrorCode::InvalidField,
                 "explicit source dtype does not permit unrelated config fields");
  }
}

void test_generic_cast_does_not_invent_image_layout() {
  const std::array<TensorShape, 5> shapes{{{1, 7}, {7}, {2, 3, 7}, {1, 2, 3, 7}, {1, 2, 3, 5, 7}}};
  for (const auto& shape : shapes) {
    std::uint64_t elements = 1U;
    for (const auto dim : shape) {
      elements *= static_cast<std::uint64_t>(dim);
    }
    const auto bf16_bytes = elements * 2U;
    const auto fp32_bytes = elements * 4U;
    const bool padded_rank2 = shape.size() == 2U;
    const auto ofm_bytes = padded_rank2 ? 16U : bf16_bytes;
    auto manifest = nlohmann::json::parse(valid_manifest());
    manifest["input_nodes"][0]["size"] = fp32_bytes;
    auto& plugins = manifest["plugins"];
    for (const auto index : {0, 2}) {
      auto& params = plugins[index]["config_params"]["params"];
      params["in_dtype"] = index == 0 ? "float32" : "bfloat16";
      params["input_shapes"] = nlohmann::json::array({shape});
      params["output_shapes"] = nlohmann::json::array({shape});
      plugins[index]["input_nodes"][0]["size"] = index == 0 ? fp32_bytes : bf16_bytes;
      plugins[index]["output_nodes"][0]["size"] = index == 0 ? bf16_bytes : fp32_bytes;
    }
    plugins[1]["input_nodes"][0]["size"] = bf16_bytes;
    plugins[1]["output_nodes"][0]["size"] = ofm_bytes;
    plugins[3]["input_nodes"][0]["size"] = fp32_bytes;
    plugins[3]["output_nodes"][0]["size"] = fp32_bytes;
    if (padded_rank2) {
      plugins[2]["input_nodes"][0]["name"] = "sliced";
      plugins[2]["sequence"] = 4;
      plugins[3]["sequence"] = 5;
      const nlohmann::json slice = {{"name", "slice"},
                                    {"sequence", 3},
                                    {"processor", "EV74"},
                                    {"type", "sgpProcess"},
                                    {"config_params",
                                     {{"desired_batch_size", 1},
                                      {"actual_batch_size", 1},
                                      {"kernel", "slice_transform"},
                                      {"params",
                                       {{"begin", {0, 0}},
                                        {"end", {1, 7}},
                                        {"input_shape", {1, 8}},
                                        {"output_shape", shape},
                                        {"input_shapes", {{1, 8}}},
                                        {"output_shapes", nlohmann::json::array({shape})}}}}},
                                    {"input_nodes", {{{"name", "mla0"}, {"size", ofm_bytes}}}},
                                    {"output_nodes", {{{"name", "sliced"}, {"size", bf16_bytes}}}}};
      plugins.insert(plugins.begin() + 2, slice);
    }
    const auto decoded =
        MpkDecoder{}.decode_json(manifest.dump(), monolithic_topology(bf16_bytes, ofm_bytes));
    if (!decoded && decoded.error) {
      std::cerr << decoded.error->json_path << ": " << decoded.error->detail << '\n';
    }
    check(static_cast<bool>(decoded),
          "generic Cast chain preserves authored ranks one through five");
    const auto& plan = *decoded.plan;
    for (const auto& value : plan.values()) {
      check(!value.logical_layout.has_value(),
            "Cast must not invent image layout on generic tensors");
    }
    const auto* published = plan.value(plan.model_outputs().front().value_id);
    check(published && published->logical_shape == shape && published->logical_dtype == "float32" &&
              published->required_bytes == fp32_bytes,
          "generic Cast publication preserves exact shape, dtype and byte extent");
    if (padded_rank2) {
      const auto& slice = plan.ops()[2];
      check(slice.kind == OpKind::Slice &&
                plan.value(slice.inputs.front())->required_bytes == 16U &&
                plan.value(slice.outputs.front())->required_bytes == 14U &&
                plan.value(slice.outputs.front())->read_expression.has_value(),
            "rank-two slice preserves the padded carrier and exact seven-element BF16 view");
    }

    std::string error;
    const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
    check(physical.has_value(), "generic Cast chain lowers physically");
    const auto arena =
        FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                    kLegacyEvoCmaRegionAlignmentBytes, &error);
    check(arena.has_value(), "generic Cast frame arena compiles");
    std::size_t cast_commands = 0U;
    for (const auto& command : physical->commands) {
      if (command.engine != PhysicalEngine::Cvu) {
        continue;
      }
      ++cast_commands;
      const std::array<std::uint32_t, 1> command_ids{command.id};
      const auto contract = build_dmabuf_plan_processcvu_command_contract(
          plan, *physical, command_ids, *arena, &error);
      if (!contract) {
        std::cerr << error << '\n';
      }
      check(contract && contract->payload.input_tensors.size() == 1U &&
                contract->payload.output_tensors.size() == 1U,
            "generic Cast projection authors one input/output descriptor pair");
      const auto& input = contract->payload.input_tensors.front();
      const auto& output = contract->payload.output_tensors.front();
      check(input.shape.rank == shape.size() && output.shape.rank == shape.size(),
            "Cast descriptors preserve the authored rank");
      check((input.dtype == SIMA_EV_DTYPE_FP32 && output.dtype == SIMA_EV_DTYPE_BF16) ||
                (input.dtype == SIMA_EV_DTYPE_BF16 && output.dtype == SIMA_EV_DTYPE_FP32),
            "Cast descriptors preserve the exact scalar conversion");
      check(input.storage.nbytes == (input.dtype == SIMA_EV_DTYPE_FP32 ? fp32_bytes : bf16_bytes) &&
                output.storage.nbytes ==
                    (output.dtype == SIMA_EV_DTYPE_FP32 ? fp32_bytes : bf16_bytes),
            "Cast descriptor byte extents remain independent of layout evidence");
      std::int64_t input_stride = input.dtype == SIMA_EV_DTYPE_FP32 ? 4 : 2;
      std::int64_t output_stride = output.dtype == SIMA_EV_DTYPE_FP32 ? 4 : 2;
      for (std::size_t reverse = shape.size(); reverse > 0U; --reverse) {
        const auto axis = reverse - 1U;
        check(input.shape.sizes[axis] == shape[axis] && output.shape.sizes[axis] == shape[axis] &&
                  input.shape.axis_semantics[axis] == SIMA_EV_AXIS_UNKNOWN &&
                  output.shape.axis_semantics[axis] == SIMA_EV_AXIS_UNKNOWN,
              "generic Cast descriptors retain authored shape without invented axes");
        // The sliced [1,7] BF16 view retains its [1,8] parent row stride.
        const auto expected_input_stride =
            padded_rank2 && input.dtype == SIMA_EV_DTYPE_BF16 && axis == 0U ? 16 : input_stride;
        check(input.layout.strided.strides_bytes[axis] == expected_input_stride &&
                  output.layout.strided.strides_bytes[axis] == output_stride,
              "generic Cast descriptors retain exact dense or sliced-parent strides");
        input_stride *= shape[axis];
        output_stride *= shape[axis];
      }
      check(contract->runtime_contract.logical_outputs.size() == 1U &&
                contract->runtime_contract.logical_outputs.front().layout.empty() &&
                contract->payload.runtime_output_logical_layout_list ==
                    std::vector<std::string>{""},
            "generic Cast publication retains unknown layout through runtime projection");
    }
    check(cast_commands == 2U, "both FP32-to-BF16 and BF16-to-FP32 Cast descriptors are checked");
  }
}

MlaStaticContract project_single_mla(const ModelExecutionPlan& plan) {
  MlaStaticContract contract;
  std::vector<PhysicalPortSource> sources;
  for (const auto& port : plan.backend_ports()) {
    const auto* value = plan.value(port.value_id);
    PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(port.port_index);
    physical.size_bytes = value->required_bytes;
    physical.segment_name = value->name;
    if (port.direction == BackendPortDirection::Input) {
      contract.physical_inputs.push_back(physical);
      TensorStaticSpec logical;
      logical.tensor_index = static_cast<int>(port.port_index);
      contract.logical_inputs.push_back(logical);
      sources.push_back({value->id, static_cast<int>(port.port_index)});
    } else {
      contract.dispatcher_physical_outputs.push_back(physical);
    }
  }
  std::string error;
  const bool projected = apply_dmabuf_plan_contract_projection(plan, &contract, sources, &error);
  if (!projected) {
    std::cerr << error << "\n";
  }
  check(projected, "decoded exact contracts project through the frame arena");
  return contract;
}

void test_detess_byte_carrier_keeps_logical_frame() {
  auto manifest = nlohmann::json::parse(detess_dequant_manifest());
  const nlohmann::json frame = {1, 3, 5, 7};
  auto& plugins = manifest["plugins"];
  plugins[0]["output_nodes"][0]["size"] = 480;
  plugins[1]["input_nodes"][0]["size"] = 480;
  plugins[1]["output_nodes"][0]["size"] = 210;
  auto& params = plugins[1]["config_params"]["params"];
  params["frame_type"] = "bfloat16";
  params["frame_shape"] = frame;
  params["slice_shape"] = {3, 5, 7};
  params["align_c16"] = true;
  params["cblock"] = true;
  params["input_shapes"] = {{1, 480}};
  params["output_shapes"] = nlohmann::json::array({frame});
  plugins[2]["config_params"]["kernel"] = "cast_transform";
  plugins[2]["config_params"]["params"] = {{"in_dtype", "bfloat16"},
                                           {"out_dtype", "float32"},
                                           {"input_shapes", nlohmann::json::array({frame})},
                                           {"output_shapes", nlohmann::json::array({frame})}};
  plugins[2]["input_nodes"][0]["size"] = 210;
  plugins[2]["output_nodes"][0]["size"] = 420;
  plugins[3]["input_nodes"][0]["size"] = 420;
  plugins[3]["output_nodes"][0]["size"] = 420;
  const auto decoded = MpkDecoder{}.decode_json(manifest.dump(), monolithic_topology(16U, 480U));
  check(static_cast<bool>(decoded), "BF16 detess byte-carrier contract is admitted");
  const auto& plan = *decoded.plan;
  const auto* value = plan.value(plan.ops()[1].inputs.front());
  check(value->logical_shape == TensorShape({1, 3, 5, 7}) && value->logical_dtype == "bfloat16" &&
            value->required_bytes == 480U &&
            plan.ops()[1].input_shapes == std::vector<TensorShape>{{1, 480}},
        "detess preserves frame geometry separately from its authored 480-byte carrier");
  const auto& cast = plan.ops()[2];
  check(cast.kind == OpKind::Cast && plan.value(cast.inputs.front())->logical_layout == "HWC" &&
            plan.value(cast.outputs.front())->logical_layout == "HWC" &&
            plan.value(plan.model_outputs().front().value_id)->logical_layout == "HWC",
        "Cast and publication preserve the exact layout established by Detess");
  const auto contract = project_single_mla(plan);
  check(contract.logical_outputs.front().shape == TensorShape({1, 3, 5, 7}) &&
            contract.physical_outputs.front().size_bytes == 480U,
        "MLA projects 210 logical BF16 bytes over the exact 480-byte tiled carrier");
  params["input_shapes"] = {{1, 240}};
  expect_error(manifest.dump(), monolithic_topology(16U, 480U),
               MpkDecodeErrorCode::ConfigurationMismatch,
               "detess cannot reinterpret a contradictory carrier shape as BF16 element counts");
}

void test_unpack_tiled_carriers_keep_storage_geometry() {
  auto manifest = nlohmann::json::parse(valid_manifest());
  auto& plugins = manifest["plugins"];
  plugins.erase(plugins.begin() + 2, plugins.end());
  plugins[1]["output_nodes"] = {{{"name", "packed_ofm"}, {"size", 576}}};
  auto unpack = nlohmann::json::parse(packed_read_manifest())["plugins"][4];
  unpack["sequence"] = 3;
  unpack["input_nodes"] = {{{"name", "packed_ofm"}, {"size", 576}}};
  unpack["output_nodes"] = {{{"name", "tiled0"}, {"size", 480}},
                            {{"name", "tiled1"}, {"size", 96}}};
  auto& unpack_params = unpack["config_params"]["params"];
  unpack_params["tensor_shapes"] = {{1, 480}, {1, 96}};
  unpack_params["input_shapes"] = {{1, 576}};
  unpack_params["output_shapes"] = {{1, 480}, {1, 96}};
  plugins.push_back(unpack);

  const std::array<TensorShape, 2> frames{{{1, 3, 5, 7}, {1, 1, 3, 14}}};
  const std::array<std::uint64_t, 2> stored_bytes{480U, 96U};
  const std::array<std::uint64_t, 2> dense_bytes{210U, 84U};
  const auto detess_template = nlohmann::json::parse(detess_dequant_manifest())["plugins"][1];
  const auto cast_template = nlohmann::json::parse(valid_manifest())["plugins"][2];
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto suffix = std::to_string(index);
    auto detess = detess_template;
    detess["name"] = "detess" + suffix;
    detess["sequence"] = 4U + index * 2U;
    detess["input_nodes"] = {{{"name", "tiled" + suffix}, {"size", stored_bytes[index]}}};
    detess["output_nodes"] = {{{"name", "detess" + suffix}, {"size", dense_bytes[index]}}};
    auto& params = detess["config_params"]["params"];
    params["frame_type"] = "bfloat16";
    params["frame_shape"] = frames[index];
    params["slice_shape"] = TensorShape(frames[index].begin() + 1, frames[index].end());
    params["align_c16"] = true;
    params["cblock"] = true;
    params["input_shapes"] = {{1, stored_bytes[index]}};
    params["output_shapes"] = nlohmann::json::array({frames[index]});
    plugins.push_back(detess);
    auto cast = cast_template;
    cast["name"] = "cast_out" + suffix;
    cast["sequence"] = 5U + index * 2U;
    cast["input_nodes"] = {{{"name", "detess" + suffix}, {"size", dense_bytes[index]}}};
    cast["output_nodes"] = {{{"name", "output" + suffix}, {"size", dense_bytes[index] * 2U}}};
    cast["config_params"]["params"] = {{"in_dtype", "bfloat16"},
                                       {"out_dtype", "float32"},
                                       {"input_shapes", nlohmann::json::array({frames[index]})},
                                       {"output_shapes", nlohmann::json::array({frames[index]})}};
    plugins.push_back(cast);
  }

  auto publish = nlohmann::json::parse(valid_manifest())["plugins"][3];
  publish["sequence"] = 8;
  publish["input_nodes"] = {{{"name", "output0"}, {"size", 420}},
                            {{"name", "output1"}, {"size", 168}}};
  publish["output_nodes"] = {{{"name", "public0"}, {"size", 420}},
                             {{"name", "public1"}, {"size", 168}}};
  plugins.push_back(publish);

  const auto decoded = MpkDecoder{}.decode_json(manifest.dump(), monolithic_topology(8U, 576U));
  if (!decoded && decoded.error) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded),
        "two BF16 tiled Unpack carriers decode independently of frames");
  const auto& plan = *decoded.plan;
  const auto& unpack_op = plan.ops()[2];
  check(unpack_op.kind == OpKind::Unpack && unpack_op.outputs.size() == 2U,
        "two tiled outputs retain their compiler-authored Unpack order");
  const auto contract = project_single_mla(plan);
  check(contract.physical_outputs.size() == 1U &&
            contract.physical_outputs.front().size_bytes == 576U &&
            contract.logical_outputs.size() == 2U,
        "MLA publishes two carrier views over one exact 576-byte OFM");
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto* value = plan.value(unpack_op.outputs[index]);
    const TensorShape storage_shape{1, static_cast<std::int64_t>(stored_bytes[index])};
    const std::vector<std::int64_t> strides{static_cast<std::int64_t>(stored_bytes[index]), 1};
    const std::uint64_t offset = index == 0U ? 0U : 480U;
    check(value && value->logical_shape == frames[index] && value->logical_dtype == "bfloat16" &&
              value->required_bytes == stored_bytes[index],
          "Unpack preserves semantic BF16 frame separately from tiled storage bytes");
    check(value->read_expression && value->read_expression->storage_shape == storage_shape &&
              value->read_expression->stride_bytes == strides &&
              value->read_expression->source_value_id == unpack_op.inputs.front() &&
              value->read_expression->byte_offset == offset,
          "Unpack read expression retains explicit byte-carrier shape, strides, and offset");
    check(value->storage_binding && value->storage_binding->physical_span == stored_bytes[index] &&
              value->storage_binding->byte_offset == offset &&
              value->storage_binding->carrier_id ==
                  plan.value(unpack_op.inputs.front())->storage_binding->carrier_id,
          "both tiled views bind their exact disjoint spans in the shared MLA carrier");
    const auto& logical = contract.logical_outputs[index];
    check(logical.shape == storage_shape && logical.dtype == "int8" &&
              logical.stride_bytes == strides && logical.size_bytes == stored_bytes[index] &&
              logical.byte_offset == static_cast<std::int64_t>(offset) && logical.layout.empty(),
          "MLA projection publishes byte-carrier geometry without semantic BF16/HWC leakage");
  }

  std::string error;
  const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
  check(physical.has_value(), "tiled Unpack branches lower to physical commands");
  std::vector<PhysicalCommandId> detesscast_commands;
  for (const auto& command : physical->commands) {
    if (command.engine == PhysicalEngine::Cvu && command.graph_id == 225U) {
      detesscast_commands.push_back(command.id);
    }
  }
  check(!detesscast_commands.empty(), "BF16 tiled branches select graph225 Detess+Cast");
  const auto arena =
      FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                  kLegacyEvoCmaRegionAlignmentBytes, &error);
  check(arena.has_value(), "tiled Unpack branches compile their shared frame arena");
  const auto cvu = build_dmabuf_plan_processcvu_command_contract(
      plan, *physical, detesscast_commands, *arena, &error);
  if (!cvu) {
    std::cerr << error << "\n";
  }
  check(cvu && cvu->payload.input_tensors.size() == frames.size() &&
            cvu->payload.output_tensors.size() == frames.size() &&
            cvu->runtime_contract.logical_inputs.size() == frames.size() &&
            cvu->runtime_contract.physical_inputs.size() == frames.size(),
        "graph225 projection retains both tiled input and dense output contracts");
  const auto* parent_region = arena->region(unpack_op.inputs.front());
  check(parent_region != nullptr, "packed MLA output owns one arena region");
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const auto& input = cvu->payload.input_tensors[index];
    const auto& output = cvu->payload.output_tensors[index];
    check(input.dtype == SIMA_EV_DTYPE_BF16 && input.shape.rank == frames[index].size() &&
              input.storage.nbytes == stored_bytes[index] && output.dtype == SIMA_EV_DTYPE_FP32 &&
              output.shape.rank == frames[index].size() &&
              output.storage.nbytes == dense_bytes[index] * 2U,
          "CVU payload retains semantic precision and exact tiled/dense storage sizes");
    for (std::size_t axis = 0; axis < frames[index].size(); ++axis) {
      check(input.shape.sizes[axis] == frames[index][axis] &&
                output.shape.sizes[axis] == frames[index][axis],
            "CVU descriptors retain frame geometry instead of byte-carrier geometry");
    }
    const auto& logical = cvu->runtime_contract.logical_inputs[index];
    const auto& binding = cvu->runtime_contract.physical_inputs[index];
    const auto offset = index == 0U ? 0U : 480U;
    check(logical.shape == TensorShape({1, static_cast<std::int64_t>(stored_bytes[index])}) &&
              logical.dtype == "int8" && logical.layout.empty() &&
              logical.stride_bytes ==
                  std::vector<std::int64_t>({static_cast<std::int64_t>(stored_bytes[index]), 1}) &&
              logical.size_bytes == stored_bytes[index] && logical.byte_offset == 0 &&
              binding.size_bytes == stored_bytes[index] &&
              binding.address_source == PhysicalAddressSource::FrameArenaSpan &&
              binding.source_byte_offset ==
                  static_cast<std::int64_t>(parent_region->byte_offset + offset),
          "CVU runtime inputs use exact byte-carrier metadata and shared-arena offsets");
  }

  auto incompatible_consumer = manifest;
  auto& consumer_plugins = incompatible_consumer["plugins"];
  auto tess = consumer_plugins[3];
  tess["name"] = "tess_on_tiled_carrier";
  tess["sequence"] = 8;
  tess["config_params"]["kernel"] = "tessellation_transform";
  auto& tess_params = tess["config_params"]["params"];
  tess_params.erase("frame_shape");
  tess_params["input_shapes"] = nlohmann::json::array({frames.front()});
  tess_params["output_shapes"] = {{1, 480}};
  tess["output_nodes"] = {{{"name", "retiled"}, {"size", 480}}};
  consumer_plugins.back()["sequence"] = 9;
  consumer_plugins.back()["input_nodes"].push_back({{"name", "retiled"}, {"size", 480}});
  consumer_plugins.back()["output_nodes"].push_back({{"name", "public2"}, {"size", 480}});
  consumer_plugins.insert(consumer_plugins.end() - 1, tess);
  const auto rejected_consumer =
      MpkDecoder{}.decode_json(incompatible_consumer.dump(), monolithic_topology(8U, 576U));
  check(!rejected_consumer && rejected_consumer.error &&
            rejected_consumer.error->code == MpkDecodeErrorCode::ConfigurationMismatch &&
            rejected_consumer.error->detail.find(
                "storage view 'tiled0' has incompatible consumer 'tess_on_tiled_carrier'") !=
                std::string::npos,
        "contiguous carrier bytes cannot authorize a shared dense Tessellate consumer");

  auto incompatible = manifest;
  auto& incompatible_params = incompatible["plugins"][2]["config_params"]["params"];
  incompatible_params["tensor_shapes"][1] = {1, 95};
  incompatible_params["output_shapes"][1] = {1, 95};
  expect_error(incompatible.dump(), monolithic_topology(8U, 576U),
               MpkDecodeErrorCode::ValueSizeMismatch,
               "Unpack rejects a carrier shape inconsistent with its declared byte length");
}

void test_dense_ifm_tail_padding() {
  auto manifest = nlohmann::json::parse(valid_manifest());
  manifest["input_nodes"][0]["size"] = 28;
  auto& cast = manifest["plugins"][0];
  cast["input_nodes"][0]["size"] = 28;
  cast["output_nodes"][0]["size"] = 14;
  cast["config_params"]["params"]["input_shapes"] = {{1, 7}};
  cast["config_params"]["params"]["output_shapes"] = {{1, 7}};
  manifest["plugins"][1]["input_nodes"][0]["size"] = 14;
  auto topology = monolithic_topology(16U, 8U);
  topology.monolithic_ifm = false;
  topology.monolithic_ifm_extent_bytes = 0U;
  topology.ifm_symbol_names = {"data.ifm.persistent.MLA_0/placeholder_0_0.b0"};
  topology.ifm_extent_bytes = {16U};
  const auto decoded = MpkDecoder{}.decode_json(manifest.dump(), topology);
  check(static_cast<bool>(decoded), "exact dense 14-byte BF16 input admits 16-byte ELF extent");
  const auto& plan = *decoded.plan;
  const auto& port = plan.backend_ports(0U, BackendPortDirection::Input).front();
  const auto* value = plan.value(port.value_id);
  check(value->required_bytes == 14U && value->logical_shape == TensorShape({1, 7}) &&
            port.physical_extent_bytes == 14U,
        "IFM allocation tail does not enlarge the MPK transfer");
  const auto contract = project_single_mla(plan);
  check(contract.physical_inputs.front().size_bytes == 14U &&
            contract.input_bindings.front().src_physical_size_bytes == 14U,
        "IFM arena projection binds exactly the MPK bytes");
  for (const auto bad_extent : {13U, 15U, 17U, 32U}) {
    topology.ifm_extent_bytes = {bad_extent};
    expect_error(manifest.dump(), topology, MpkDecodeErrorCode::ValueSizeMismatch,
                 "undersized, unaligned, and excessive IFM extents remain rejected");
  }
  const auto monolithic = MpkDecoder{}.decode_json(manifest.dump(), monolithic_topology(16U, 8U));
  check(static_cast<bool>(monolithic), "monolithic allocation tail preserves MPK transfer length");
  check(project_single_mla(*monolithic.plan).physical_inputs.front().size_bytes == 14U,
        "monolithic transfer is not rounded to the ELF reservation");
  auto spatial = manifest;
  spatial["input_nodes"][0]["size"] = 420;
  spatial["plugins"][0]["input_nodes"][0]["size"] = 420;
  spatial["plugins"][0]["output_nodes"][0]["size"] = 210;
  spatial["plugins"][0]["config_params"]["params"]["input_shapes"] = {{1, 3, 5, 7}};
  spatial["plugins"][0]["config_params"]["params"]["output_shapes"] = {{1, 3, 5, 7}};
  spatial["plugins"][1]["input_nodes"][0]["size"] = 210;
  const auto c7 = MpkDecoder{}.decode_json(spatial.dump(), monolithic_topology(224U, 8U));
  check(static_cast<bool>(c7), "C7 BF16 HWC input admits 210-byte MPK transfer with 224-byte ELF");
  const auto c7_contract = project_single_mla(*c7.plan);
  check(c7_contract.physical_inputs.front().size_bytes == 210U &&
            c7_contract.input_bindings.front().src_physical_size_bytes == 210U,
        "C7 BF16 HWC projection retains exactly 210 bytes");
  for (const auto bad_extent : {209U, 211U, 223U, 225U, 240U}) {
    expect_error(spatial.dump(), monolithic_topology(bad_extent, 8U),
                 MpkDecodeErrorCode::ValueSizeMismatch,
                 "monolithic transfer rejects inconsistent ELF allocation extents");
  }
  topology.ifm_extent_bytes = {16U};
  auto batched = manifest;
  batched["plugins"][1]["config_params"]["actual_batch_size"] = 2;
  batched["plugins"][1]["config_params"]["desired_batch_size"] = 2;
  expect_error(batched.dump(), topology, MpkDecodeErrorCode::ValueSizeMismatch,
               "batch-row padding requires a separate contract and cannot use the batch-one rule");
  cast["config_params"]["params"]["input_shapes"] = {{7, 1}};
  cast["config_params"]["params"]["output_shapes"] = {{7, 1}};
  expect_error(manifest.dump(), topology, MpkDecodeErrorCode::ValueSizeMismatch,
               "logical leading dimension must also prove batch one");
}

void test_compiler_version_does_not_restrict_admission() {
  const std::vector<nlohmann::json> versions = {nullptr,
                                                true,
                                                42,
                                                nlohmann::json::array(),
                                                nlohmann::json::object(),
                                                "",
                                                "2.0.0",
                                                "2.1.0",
                                                "2.1.3",
                                                "3.0.0",
                                                "3.0.1",
                                                "99.0.0"};
  struct Fixture {
    const std::string& manifest;
    MlaElfIoTopology topology;
  };
  const std::array<Fixture, 3> fixtures = {{
      {valid_manifest(), monolithic_topology()},
      {yolov8_quant_tess_ingress_manifest(), monolithic_topology(1228800U, 16U)},
      {resnet_batch_flatten_manifest(), monolithic_topology(16U, 1008U)},
  }};
  for (const auto& fixture : fixtures) {
    auto manifest = nlohmann::json::parse(fixture.manifest);
    for (std::size_t index = 0; index <= versions.size(); ++index) {
      if (index == versions.size()) {
        manifest.erase("model_sdk_version");
      } else {
        manifest["model_sdk_version"] = versions[index];
      }
      const auto result = MpkDecoder{}.decode_json(manifest.dump(), fixture.topology);
      check(static_cast<bool>(result), "compiler version metadata does not restrict admission");
      const auto expected_version = index < versions.size() && versions[index].is_string()
                                        ? versions[index].get<std::string>()
                                        : std::string{};
      check(result.plan->contract_version() == expected_version,
            "string version metadata remains available as provenance");
    }
  }
}

void test_success_and_immutable_contract() {
  const auto result =
      MpkDecoder{}.decode_json(valid_manifest(), monolithic_topology(), "synthetic.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "valid manifest decodes");
  const auto& plan = *result.plan;
  check(plan.contract_version() == "2.0.0", "contract version preserved exactly");
  check(plan.model_inputs().size() == 1U && plan.model_outputs().size() == 1U,
        "public input/output arity preserved");
  check(plan.ops().size() == 3U && plan.values().size() == 4U,
        "publication metadata is absent from the executable graph and value table");
  check(plan.model_outputs()[0].name == "decorated/model/output:0",
        "ordered PassThrough input is published under its real producer name");
  check(plan.value(plan.model_outputs()[0].value_id)->logical_dtype == "float32",
        "typed inverse preserves published dtype");
  check(plan.backend_ports().size() == 2U, "monolithic ELF is exactly 1 IFM / 1 OFM");
  for (const auto& port : plan.backend_ports()) {
    check(port.required_alignment_bytes == kLegacyEvoCmaRegionAlignmentBytes,
          "legacy port uses documented conservative alignment");
    check(port.alignment_authority == BackendPortAlignmentAuthority::LegacyPolicy,
          "legacy alignment provenance is policy, not MPK/ELF");
  }
  check(plan.backend_ports()[0].elf_symbol == "data.ifm.b0" &&
            plan.backend_ports()[1].elf_symbol == "data.ofm.b0",
        "exact monolithic symbols retained");
  check(!result.proof.empty(), "deterministic proof report emitted");
}

void test_unpack_and_slice_are_read_expressions() {
  const auto result = MpkDecoder{}.decode_json(
      packed_read_manifest(), monolithic_topology(32U, 32U), "packed-read-synthetic.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "packed read-expression manifest decodes");
  const auto& plan = *result.plan;
  check(plan.backend_ports().size() == 2U, "packed route remains one IFM and one OFM");
  check(plan.model_outputs().size() == 2U, "both logical reads are published");
  const auto* pack = std::get_if<PackOpConfig>(&plan.ops()[2].config);
  check(pack != nullptr && pack->components.size() == 2U &&
            pack->components[0].parent_offset == 0U && pack->components[0].stored_bytes == 16U &&
            pack->components[1].parent_offset == 16U && pack->components[1].stored_bytes == 16U,
        "Pack carries exact ordered parent placement");

  // Value order is: two public inputs, two producers, Pack, MLA, two unpack reads, two
  // slice reads.  Every read remains rooted in the
  // single materialized MLA OFM; neither Unpack nor Slice schedules work.
  const auto require_read = [&](const ValueId id, const std::uint64_t offset) {
    const auto* value = plan.value(id);
    check(value != nullptr && value->read_expression.has_value(),
          "logical value carries a compiled read expression");
    check(value->read_expression->source_value_id == 5U,
          "logical read is composed to the physical MLA carrier");
    check(value->read_expression->byte_offset == offset,
          "logical read preserves its exact carrier-relative offset");
    check(value->read_expression->stride_bytes == std::vector<std::int64_t>({16, 8, 4, 1}),
          "logical read preserves the exact inherited byte strides");
  };
  require_read(6U, 0U);
  require_read(7U, 16U);
  require_read(8U, 0U);
  require_read(9U, 17U);
  check(plan.ops().size() == 7U && plan.values().size() == 10U,
        "PassThrough creates neither an executable op nor duplicate output values");
  check(plan.model_outputs()[0].value_id == 8U && plan.model_outputs()[1].value_id == 9U,
        "publication points directly at the two compiled Slice views");

  const auto& parent = *plan.value(4U)->storage_binding;
  check(plan.value(2U)->storage_binding->kind == StorageBindingKind::Root &&
            plan.value(3U)->storage_binding->kind == StorageBindingKind::Root &&
            plan.value(2U)->storage_binding->carrier_id == parent.carrier_id &&
            plan.value(3U)->storage_binding->carrier_id == parent.carrier_id &&
            plan.value(2U)->storage_binding->byte_offset == 16U &&
            plan.value(3U)->storage_binding->byte_offset == 0U && plan.carriers().size() == 4U,
        "decoded Pack coalesces producer roots by identity, not producer order");
  std::string error;
  const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
  check(physical.has_value(), "decoded direct Pack lowers without a Pack command");
  const auto arena =
      FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                  kLegacyEvoCmaRegionAlignmentBytes, &error);
  check(arena && arena->region(2U) == arena->region(4U) && arena->region(3U) == arena->region(4U),
        "producer writes and MLA input allocate one canonical carrier region");
  std::vector<PhysicalCommandId> commands;
  for (const auto& command : physical->commands) {
    if (command.engine == PhysicalEngine::Cvu) {
      commands.push_back(command.id);
    }
  }
  const auto contract =
      build_dmabuf_plan_processcvu_command_contract(plan, *physical, commands, *arena, &error);
  if (!contract) {
    std::cerr << error << '\n';
  }
  check(contract && contract->runtime_contract.physical_outputs.size() == 2U,
        "real decoder-to-generic-projection path emits both producer-direct writes");
  for (std::size_t index = 0U; index < contract->runtime_contract.logical_outputs.size(); ++index) {
    const auto& logical = contract->runtime_contract.logical_outputs[index];
    const auto& output = contract->runtime_contract.physical_outputs[index];
    const auto offset = logical.logical_name == "input0" ? 0U : 16U;
    check(output.source_byte_offset ==
                  static_cast<std::int64_t>(arena->region(4U)->byte_offset + offset) &&
              output.size_bytes == 16U &&
              output.source_byte_offset % output.required_alignment_bytes == 0U,
          "each exact producer range equals its MLA parent component without 4KiB member padding");
  }

  std::size_t read_proofs = 0U;
  for (const auto& fact : result.proof) {
    if (fact.subject.rfind("read[", 0U) == 0U &&
        fact.evidence.find("no runtime operation is scheduled") != std::string::npos) {
      ++read_proofs;
    }
  }
  check(read_proofs == 4U, "decoder proves both unpack and both slice reads are not jobs");
}

void test_reshape_is_an_exact_read_expression() {
  const auto result =
      MpkDecoder{}.decode_json(reshape_manifest(), monolithic_topology(16U, 8U), "reshape.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "exact byte-preserving reshape is accepted");
  const auto& reshape = result.plan->ops().front();
  const auto& output = result.plan->values().at(reshape.outputs.front());
  check(reshape.kind == OpKind::Reshape && output.read_expression.has_value() &&
            output.read_expression->source_value_id == reshape.inputs.front() &&
            output.required_bytes ==
                result.plan->values().at(reshape.inputs.front()).required_bytes,
        "reshape lowers to one zero-offset root view without materialization");

  const auto mismatch =
      replace_once(reshape_manifest(), "\"output_nodes\":[{\"name\":\"reshaped\",\"size\":16}]",
                   "\"output_nodes\":[{\"name\":\"reshaped\",\"size\":12}]");
  const auto rejected =
      MpkDecoder{}.decode_json(mismatch, monolithic_topology(16U, 8U), "reshape-mismatch.json");
  check(!rejected && rejected.error->code == MpkDecodeErrorCode::ValueSizeMismatch,
        "reshape that changes the byte extent fails closed");
}

void test_registered_detess_layout_is_preserved_through_dequant() {
  const auto result = MpkDecoder{}.decode_json(detess_dequant_manifest(),
                                               monolithic_topology(16U, 16U), "detess-layout.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "detess/dequant manifest decodes");
  const auto& plan = *result.plan;
  const auto& detess = plan.ops().at(1);
  const auto& dequant = plan.ops().at(2);
  check(plan.value(detess.outputs.front())->logical_layout == "HWC",
        "registered graph 3 authors canonical HWC output axes");
  check(plan.value(dequant.outputs.front())->logical_layout == "HWC",
        "layout-preserving graph 223 retains graph 3 output axes");
  check(plan.value(plan.model_outputs().front().value_id)->logical_layout == "HWC",
        "publication retains the exact transform layout");
}

void test_resnet_batch_flatten_is_transparent_to_fused_graph227() {
  const auto result =
      MpkDecoder{}.decode_json(resnet_batch_flatten_manifest(), monolithic_topology(16U, 1008U),
                               "resnet-batch-flatten.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "exact ResNet batch-flatten chain decodes");
  const auto foreign_reshape_grammar =
      replace_once(resnet_batch_flatten_manifest(),
                   "\"kernel\":\"batch_flatten_transform\",\"params\":{\n"
                   "                          \"input_shapes\"",
                   "\"kernel\":\"batch_flatten_transform\",\"params\":{\n"
                   "                          \"newshape\":[1,1000],\"input_shapes\"");
  const auto grammar_rejected =
      MpkDecoder{}.decode_json(foreign_reshape_grammar, monolithic_topology(16U, 1008U),
                               "resnet-batch-flatten-foreign-grammar.json");
  check(!grammar_rejected && grammar_rejected.error->code == MpkDecodeErrorCode::InvalidField,
        "batch flatten accepts only its exact two-shape-list grammar");
  const auto& plan = *result.plan;
  check(plan.ops().size() == 4U && plan.ops()[1].kind == OpKind::Detessellate &&
            plan.ops()[2].kind == OpKind::Reshape &&
            plan.ops()[2].kernel == "batch_flatten_transform" &&
            plan.ops()[3].kind == OpKind::Dequantize,
        "batch flatten retains exact compiler identity as the existing Reshape relation");
  const auto* flattened = plan.value(plan.ops()[2].outputs.front());
  check(flattened && flattened->required_bytes == 1000U &&
            flattened->logical_shape == TensorShape({1, 1000}) && flattened->read_expression &&
            flattened->read_expression->source_value_id == plan.ops()[1].outputs.front() &&
            flattened->read_expression->byte_offset == 0U &&
            flattened->read_expression->stride_bytes == std::vector<std::int64_t>({1000, 1}),
        "batch flatten is one zero-offset same-byte ordered read expression");

  std::vector<ValueId> relation_values;
  check(resolve_exact_private_ordered_relation_path(plan, 1U, 3U, &relation_values) &&
            relation_values == std::vector<ValueId>(
                                   {plan.ops()[1].outputs.front(), plan.ops()[2].outputs.front()}),
        "shared physical proof sees exactly the private detess-to-dequant view path");

  std::string error;
  const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
  check(physical.has_value(), "ResNet batch-flatten chain lowers physically");
  std::vector<PhysicalCommandId> graph227_commands;
  std::size_t cvu_commands = 0U;
  for (const auto& command : physical->commands) {
    if (command.engine != PhysicalEngine::Cvu) {
      continue;
    }
    ++cvu_commands;
    if (command.graph_id == 227U) {
      graph227_commands.push_back(command.id);
    }
  }
  check(cvu_commands == 1U && graph227_commands.size() == 1U &&
            physical->commands[graph227_commands.front()].members.size() == 1U &&
            physical->commands[graph227_commands.front()].members.front().semantic_chain ==
                std::vector<OpId>({1U, 3U}) &&
            !physical->command_for_semantic_op[2U].has_value(),
        "detess/view/dequant is one graph227 submission and the view schedules no command");

  const auto arena =
      FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                  kLegacyEvoCmaRegionAlignmentBytes, &error);
  check(arena.has_value(), "relation-transparent graph227 frame arena compiles");
  const auto contract = build_dmabuf_plan_processcvu_command_contract(
      plan, *physical, graph227_commands, *arena, &error);
  check(contract && contract->payload.graph_id == 227 &&
            contract->payload.input_tensors.size() == 1U &&
            contract->payload.output_tensors.size() == 1U,
        "projection accepts the same proved relation path and authors one graph227 outer pair");
  if (contract) {
    const auto& input = contract->payload.input_tensors.front();
    const auto& output = contract->payload.output_tensors.front();
    const std::array<int, 4U> expected_shape{1, 1, 1, 1000};
    const std::array<std::uint8_t, 4U> expected_axes{SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W,
                                                     SIMA_EV_AXIS_C};
    check(input.shape.rank == expected_shape.size() && output.shape.rank == expected_shape.size(),
          "graph227 physical endpoints retain the registered Detess frame rank");
    for (std::size_t axis = 0U; axis < expected_shape.size(); ++axis) {
      check(input.shape.sizes[axis] == expected_shape[axis] &&
                output.shape.sizes[axis] == expected_shape[axis] &&
                input.shape.axis_semantics[axis] == expected_axes[axis] &&
                output.shape.axis_semantics[axis] == expected_axes[axis],
            "graph227 physical endpoints share exact N/H/W/C frame geometry");
    }
    check(input.dtype == SIMA_EV_DTYPE_INT8 && output.dtype == SIMA_EV_DTYPE_FP32 &&
              input.storage.nbytes == 1008U && output.storage.nbytes == 4000U,
          "graph227 physical descriptors retain the exact packed input and dense FP32 output");
    check(contract->payload.output_shapes == std::vector<std::vector<int>>{{1, 1000}} &&
              contract->payload.runtime_output_logical_shapes ==
                  std::vector<std::vector<int>>{{1, 1000}} &&
              contract->payload.runtime_output_logical_layout_list == std::vector<std::string>{""},
          "graph227 publication retains the post-view flattened logical contract");
    check(contract->payload.typed_output_layout_token() == "HWC" &&
              contract->payload.logical_output_layout_token().empty() &&
              contract->runtime_contract.logical_outputs.size() == 1U &&
              contract->runtime_contract.logical_outputs.front().layout.empty(),
          "unknown post-view layout must not inherit the physical EV image axes");
    auto layout_payload = contract->payload;
    layout_payload.runtime_output_logical_layout_list = {"HWC", ""};
    check(layout_payload.logical_output_layout_token(0U) == "HWC" &&
              layout_payload.logical_output_layout_token(1U).empty(),
          "each authored logical layout is authoritative, including an unknown member");
    layout_payload.runtime_output_logical_layout_list.clear();
    check(layout_payload.logical_output_layout_token() == "HWC",
          "payloads without authored logical layout retain physical layout fallback");
  }

  auto two_views =
      replace_once(resnet_batch_flatten_manifest(),
                   "\"input_nodes\":[{\"name\":\"EV_1/batch_flatten_0\",\"size\":1000}],\n"
                   "       \"output_nodes\":[{\"name\":\"dequantize_1/resnetv17_dense0_fwd\","
                   "\"size\":4000}]",
                   "\"input_nodes\":[{\"name\":\"second_view\",\"size\":1000}],\n"
                   "       \"output_nodes\":[{\"name\":\"dequantize_1/resnetv17_dense0_fwd\","
                   "\"size\":4000}]");
  two_views = replace_once(std::move(two_views),
                           "\"input_data_type\":\"int8\",\"input_shapes\":[[1,1000]],\n"
                           "                          \"output_shapes\":[[1,1000]]",
                           "\"input_data_type\":\"int8\",\"input_shapes\":[[1,1,1000]],\n"
                           "                          \"output_shapes\":[[1,1,1000]]");
  two_views =
      replace_once(std::move(two_views), "{\"name\":\"dequantize_1\",\"sequence\":4",
                   "{\"name\":\"second_view\",\"sequence\":4,\"processor\":\"EV74\","
                   "\"type\":\"sgpProcess\",\n"
                   "       \"config_params\":{\"desired_batch_size\":1,\"actual_batch_size\":1,\n"
                   "                        \"kernel\":\"reshape_transform\",\"params\":{\n"
                   "                          \"newshape\":[1,1,1000],"
                   "\"input_shapes\":[[1,1000]],\n"
                   "                          \"output_shapes\":[[1,1,1000]]}},\n"
                   "       \"input_nodes\":[{\"name\":\"EV_1/batch_flatten_0\",\"size\":1000}],\n"
                   "       \"output_nodes\":[{\"name\":\"second_view\",\"size\":1000}]},\n"
                   "      {\"name\":\"dequantize_1\",\"sequence\":5");
  two_views = replace_once(std::move(two_views), "{\"name\":\"publish\",\"sequence\":5",
                           "{\"name\":\"publish\",\"sequence\":6");
  const auto twice_decoded =
      MpkDecoder{}.decode_json(two_views, monolithic_topology(16U, 1008U), "resnet-two-views.json");
  check(static_cast<bool>(twice_decoded) && twice_decoded.plan->ops().size() == 5U,
        "two consecutive exact views decode without inventing work");
  std::vector<ValueId> twice_values;
  check(resolve_exact_private_ordered_relation_path(*twice_decoded.plan, 1U, 4U, &twice_values) &&
            twice_values.size() == 3U,
        "shared relation proof composes two private dense order-preserving views");
  const auto twice_physical = PhysicalExecutionLowerer::lower(*twice_decoded.plan, &error);
  std::size_t twice_graph227 = 0U;
  if (twice_physical) {
    for (const auto& command : twice_physical->commands) {
      twice_graph227 += command.graph_id == 227U ? 1U : 0U;
    }
  }
  check(twice_physical && twice_graph227 == 1U && !twice_physical->command_for_semantic_op[2U] &&
            !twice_physical->command_for_semantic_op[3U],
        "two consecutive views remain transparent to one graph227 submission");

  const auto observed_relation = replace_once(
      resnet_batch_flatten_manifest(),
      "\"input_nodes\":[{\"name\":\"dequantize_1/resnetv17_dense0_fwd\",\"size\":4000}],\n"
      "       \"output_nodes\":[{\"name\":\"output\",\"size\":4000}]",
      "\"input_nodes\":[{\"name\":\"EV_1/batch_flatten_0\",\"size\":1000},"
      "{\"name\":\"dequantize_1/resnetv17_dense0_fwd\",\"size\":4000}],\n"
      "       \"output_nodes\":[{\"name\":\"observed_flatten\",\"size\":1000},"
      "{\"name\":\"output\",\"size\":4000}]");
  const auto observed = MpkDecoder{}.decode_json(observed_relation, monolithic_topology(16U, 1008U),
                                                 "resnet-observed-batch-flatten.json");
  check(static_cast<bool>(observed) && observed.plan->model_outputs().size() == 2U &&
            !resolve_exact_private_ordered_relation_path(*observed.plan, 1U, 3U),
        "a branched/public batch-flatten value is not relation-transparent");
  const auto split = PhysicalExecutionLowerer::lower(*observed.plan, &error);
  std::size_t graph3 = 0U;
  std::size_t graph223 = 0U;
  std::size_t graph227 = 0U;
  if (split) {
    for (const auto& command : split->commands) {
      graph3 += command.graph_id == 3U ? 1U : 0U;
      graph223 += command.graph_id == 223U ? 1U : 0U;
      graph227 += command.graph_id == 227U ? 1U : 0U;
    }
  }
  check(split && graph3 == 1U && graph223 == 1U && graph227 == 0U,
        "observed relation splits to exact standalone graphs without speculative fusion");
}

void test_fused_ingress_layout_evidence_authors_exact_descriptor_axes() {
  struct Case {
    const std::string& manifest;
    std::uint32_t graph_id;
    std::uint64_t mla_ifm_bytes;
    const char* label;
  };
  const std::array cases{
      Case{cast_tess_manifest(), 224U, 128U, "Cast+Tess"},
      Case{quant_tess_manifest(), 226U, 64U, "Quantize+Tess"},
  };
  const std::array<std::uint8_t, 4U> expected_axes{SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W,
                                                   SIMA_EV_AXIS_C};

  for (const auto& test_case : cases) {
    const auto decoded = MpkDecoder{}.decode_json(test_case.manifest,
                                                  monolithic_topology(test_case.mla_ifm_bytes, 16U),
                                                  std::string(test_case.label) + ".json");
    if (!decoded && decoded.error.has_value()) {
      std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
    }
    check(static_cast<bool>(decoded), "raw fused-ingress AFE manifest decodes");
    const auto& plan = *decoded.plan;
    for (const auto& op : plan.ops()) {
      if (op.kind == OpKind::Cast) {
        check(plan.value(op.inputs.front())->logical_layout == "HWC" &&
                  plan.value(op.outputs.front())->logical_layout == "HWC",
              "Cast preserves downstream Tess layout evidence at both exact endpoints");
      }
    }
    const auto* outer_input = plan.value(plan.model_inputs().front());
    check(outer_input && outer_input->logical_shape == TensorShape({1, 2, 2, 16}) &&
              outer_input->logical_layout == "HWC",
          "downstream Tess HWC evidence reaches the exact equal-shape outer dense input");

    std::string error;
    const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
    check(physical.has_value(), "raw fused-ingress plan lowers physically");
    std::vector<PhysicalCommandId> commands;
    for (const auto& command : physical->commands) {
      if (command.engine == PhysicalEngine::Cvu && command.graph_id == test_case.graph_id) {
        commands.push_back(command.id);
      }
    }
    check(commands.size() == 1U, "raw AFE pair selects one exact fused graph command");
    const auto arena =
        FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                    kLegacyEvoCmaRegionAlignmentBytes, &error);
    check(arena.has_value(), "raw fused-ingress frame arena compiles");
    const auto contract =
        build_dmabuf_plan_processcvu_command_contract(plan, *physical, commands, *arena, &error);
    check(contract.has_value(), "raw fused-ingress ProcessCVU descriptor contract builds");
    check(contract->payload.input_tensors.size() == 1U &&
              contract->payload.output_tensors.size() == 1U,
          "fused ingress descriptor binds exactly one outer pair");
    const auto& input = contract->payload.input_tensors.front();
    const auto& output = contract->payload.output_tensors.front();
    check(input.shape.rank == expected_axes.size() && output.shape.rank == expected_axes.size(),
          "fused ingress descriptor retains semantic rank rather than flattened wire rank");
    for (std::size_t axis = 0; axis < expected_axes.size(); ++axis) {
      check(input.shape.axis_semantics[axis] == expected_axes[axis] &&
                output.shape.axis_semantics[axis] == expected_axes[axis],
            "fused ingress descriptor outer pair has identical canonical N/H/W/C axes");
    }
  }
}

void test_tessellate_keeps_yolov8_semantic_shape_separate_from_packed_carrier() {
  const auto decoded = MpkDecoder{}.decode_json(yolov8_quant_tess_ingress_manifest(),
                                                monolithic_topology(1228800U, 16U),
                                                "yolov8-quant-tess-ingress.json");
  if (!decoded && decoded.error.has_value()) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded), "YOLOv8-sized quant+tess ingress manifest decodes");
  const auto& plan = *decoded.plan;
  const auto& tess = plan.ops().at(1U);
  const auto* target = plan.value(tess.outputs.front());
  check(tess.kind == OpKind::Tessellate && target &&
            tess.input_shapes == std::vector<TensorShape>{{1, 640, 640, 3}} &&
            tess.output_shapes == std::vector<TensorShape>{{1, 1228800}} &&
            target->logical_shape == TensorShape({1, 640, 640, 3}),
        "Tessellate ValueSpec must use the exact semantic input frame rather than its "
        "flattened packed output shape");
  check(target->required_bytes == 1228800U &&
            target->representation == ValueRepresentation::Tessellated &&
            target->logical_dtype == "int8" && target->logical_layout == "HWC" &&
            target->storage_binding.has_value() &&
            target->storage_binding->physical_span == 1228800U,
        "YOLOv8 tessellated MLA ingress must retain independent exact physical carrier facts");
  const auto mla_inputs = plan.backend_ports(0U, BackendPortDirection::Input);
  check(mla_inputs.size() == 1U && mla_inputs.front().value_id == target->id &&
            mla_inputs.front().physical_extent_bytes == 1228800U,
        "first MLA port must bind the semantically-shaped tessellated carrier exactly");
}

void test_standalone_quantize_authors_exact_graph222_layout() {
  const auto decoded = MpkDecoder{}.decode_json(
      standalone_quant_mla_manifest(), monolithic_topology(64U, 16U), "standalone-quant.json");
  if (!decoded && decoded.error.has_value()) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded), "standalone Quantize-to-MLA manifest decodes");

  const auto& plan = *decoded.plan;
  const auto& quant = plan.ops().front();
  const auto* input = plan.value(quant.inputs.front());
  const auto* output = plan.value(quant.outputs.front());
  check(quant.kind == OpKind::Quantize && input && output && input->logical_layout == "HWC" &&
            output->logical_layout == "HWC",
        "registered graph 222 authors HWC axes on both standalone Quantize endpoints");

  std::string error;
  const auto physical = PhysicalExecutionLowerer::lower(plan, &error);
  check(physical.has_value(), "standalone Quantize plan lowers physically");
  std::vector<PhysicalCommandId> commands;
  for (const auto& command : physical->commands) {
    if (command.engine == PhysicalEngine::Cvu && command.graph_id == 222U) {
      commands.push_back(command.id);
    }
  }
  check(commands.size() == 1U, "standalone Quantize selects one exact graph 222 command");
  const auto arena =
      FrameSlotArenaPlan::compile(plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
                                  kLegacyEvoCmaRegionAlignmentBytes, &error);
  check(arena.has_value(), "standalone Quantize frame arena compiles");
  const auto contract =
      build_dmabuf_plan_processcvu_command_contract(plan, *physical, commands, *arena, &error);
  check(contract.has_value(), "standalone graph 222 descriptor contract builds");

  const auto& payload = contract->payload;
  check(payload.graph_id == 222 && payload.input_tensors.size() == 1U &&
            payload.output_tensors.size() == 1U && payload.round_off == 1 && payload.has_q_scale &&
            payload.q_scale == 0.25 && payload.has_q_zp && payload.q_zp == 0 &&
            payload.q_scale_list == std::vector<double>{0.25} &&
            payload.q_zp_list == std::vector<int>{0},
        "graph 222 retains its exact typed qparams and TONEAREST mode");
  const auto& input_desc = payload.input_tensors.front();
  const auto& output_desc = payload.output_tensors.front();
  check(input_desc.dtype == SIMA_EV_DTYPE_FP32 && output_desc.dtype == SIMA_EV_DTYPE_INT8 &&
            input_desc.storage.nbytes == 256U && output_desc.storage.nbytes == 64U &&
            sima_ev_infer_dense_tensor_format(&input_desc) == SIMA_EV_DENSE_FORMAT_NDHWC &&
            sima_ev_infer_dense_tensor_format(&output_desc) == SIMA_EV_DENSE_FORMAT_NDHWC,
        "graph 222 emits exact dense NDHWC FP32-to-INT8 endpoint descriptors");
  const std::array<std::uint8_t, 4U> expected_axes{SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W,
                                                   SIMA_EV_AXIS_C};
  for (std::size_t axis = 0; axis < expected_axes.size(); ++axis) {
    check(input_desc.shape.axis_semantics[axis] == expected_axes[axis] &&
              output_desc.shape.axis_semantics[axis] == expected_axes[axis],
          "graph 222 endpoint descriptors preserve identical N/H/W/C axes");
  }

  const auto contradictory =
      replace_once(standalone_quant_mla_manifest(), "\"output_shapes\":[[1,2,2,16]]",
                   "\"output_shapes\":[[1,2,1,32]]");
  const auto rejected = MpkDecoder{}.decode_json(contradictory, monolithic_topology(64U, 16U),
                                                 "standalone-quant-contradictory-shape.json");
  check(!rejected && rejected.error.has_value() &&
            rejected.error->code == MpkDecodeErrorCode::ConfigurationMismatch &&
            rejected.error->detail.find("contradictory exact endpoint shapes") != std::string::npos,
        "standalone graph 222 rejects contradictory exact endpoint geometry");
}

void test_qmla_output_physical_extent_and_row_pitch() {
  const auto decoded = MpkDecoder{}.decode_json(qmla_padded_output_manifest(),
                                                qmla_padded_topology(), "qmla-padded.json");
  if (!decoded && decoded.error) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded), "typed QMLA padded output decodes");
  const auto& plan = *decoded.plan;
  const auto& port = plan.backend_ports(0U, BackendPortDirection::Output).front();
  const auto* value = plan.value(port.value_id);
  const auto* carrier =
      value && value->storage_binding ? plan.carrier(value->storage_binding->carrier_id) : nullptr;
  check(port.physical_extent_bytes == 110400U && value && value->required_bytes == 109200U &&
            value->storage_binding && value->storage_binding->physical_span == 110396U &&
            value->storage_binding->stride_bytes == std::vector<std::int64_t>({110400, 368, 4}) &&
            carrier && carrier->required_bytes == 110400U,
        "QMLA physical carrier remains separate from logical addressed logits");

  const auto contradictory = MpkDecoder{}.decode_json(
      qmla_padded_output_manifest(), qmla_padded_topology(110416U), "qmla-contradictory.json");
  check(!contradictory && contradictory.error &&
            contradictory.error->code == MpkDecodeErrorCode::ConfigurationMismatch,
        "unregistered larger QMLA extent fails closed instead of guessing padding");

  auto missing = qmla_padded_topology();
  missing.ofm_extent_bytes.clear();
  const auto missing_result =
      MpkDecoder{}.decode_json(qmla_padded_output_manifest(), missing, "qmla-missing-extent.json");
  check(!missing_result && missing_result.error &&
            missing_result.error->code == MpkDecodeErrorCode::ElfTopologyInvalid,
        "missing QMLA extent evidence fails closed");
}

void test_fail_closed_cases() {
  const auto topology = monolithic_topology();
  const auto model_sdk_3_manifest =
      replace_once(yolov8_quant_tess_ingress_manifest(), "\"2.1.0\"", "\"3.0.0\"");
  expect_error(
      replace_once(model_sdk_3_manifest, "quantization_transform", "quantization_transform_suffix"),
      monolithic_topology(1228800U, 16U), MpkDecodeErrorCode::UnsupportedKernel,
      "Model Compiler 3 kernel substring is not an alias");
  expect_error(replace_once(model_sdk_3_manifest,
                            "\"output_nodes\":[{\"name\":\"quantize_0\",\"size\":1228800}]",
                            "\"output_nodes\":[{\"name\":\"quantize_0\",\"size\":1228800},"
                            "{\"name\":\"extra\",\"size\":1}]"),
               monolithic_topology(1228800U, 16U), MpkDecodeErrorCode::InvalidKernelArity,
               "quantization rejects multiple outputs");
  expect_error(replace_once(valid_manifest(), "cast_transform", "cast_transform_suffix"), topology,
               MpkDecodeErrorCode::UnsupportedKernel, "kernel substring is not an alias");
  expect_error(
      replace_once(valid_manifest(), "\"name\":\"cast0\",\"size\":8}],\n        \"output_nodes\"",
                   "\"name\":\"missing\",\"size\":8}],\n        \"output_nodes\""),
      topology, MpkDecodeErrorCode::MissingProducer, "missing full-name producer fails closed");
  expect_error(replace_once(valid_manifest(), "\"params\":{\"out_dtype\":\"bfloat16\"",
                            "\"params\":{\"ignored\":1,\"out_dtype\":\"bfloat16\""),
               topology, MpkDecodeErrorCode::InvalidField,
               "untyped extra operation config is not ignored");
  expect_error(replace_once(valid_manifest(), "\"input_shapes\":[[1,4]],\"output_shapes\":[[1,4]]",
                            "\"input_shapes\":[[1,4]],\"output_shapes\":[[2,2]]"),
               topology, MpkDecodeErrorCode::ConfigurationMismatch,
               "shape-preserving transform rejects contradictory exact endpoint shapes");

  auto two_ifm = topology;
  two_ifm.monolithic_ifm = false;
  two_ifm.monolithic_ifm_extent_bytes = 0U;
  two_ifm.ifm_symbol_names = {"data.ifm.persistent.qmla_ifm_0.b0",
                              "data.ifm.persistent.qmla_ifm_1.b0"};
  two_ifm.ifm_extent_bytes = {8U, 8U};
  expect_error(valid_manifest(), two_ifm, MpkDecodeErrorCode::ElfTopologyMismatch,
               "MPK/ELF port arity mismatch fails before plan creation");

  auto conflict = topology;
  conflict.ifm_layout_conflict = true;
  expect_error(valid_manifest(), conflict, MpkDecodeErrorCode::ElfTopologyInvalid,
               "ambiguous ELF layout fails closed");
}

void test_direct_publication_without_passthrough() {
  const auto direct = replace_once(valid_manifest(),
                                   R"json(,
      {
        "name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"decorated/model/output:0","size":16}],
        "output_nodes":[{"name":"pass_through_out_0","size":16}]
      })json",
                                   "");
  const auto result = MpkDecoder{}.decode_json(direct, monolithic_topology(), "direct-output.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "one terminal produced leaf is a complete direct publication");
  check(result.plan->model_outputs().size() == 1U &&
            result.plan->model_outputs().front().name == "decorated/model/output:0" &&
            result.plan->ops().size() == 3U,
        "direct publication preserves the exact terminal producer identity");

  const auto ambiguous = replace_once(
      direct, R"json("output_nodes":[{"name":"decorated/model/output:0","size":16}])json",
      R"json("output_nodes":[{"name":"decorated/model/output:0","size":16},{"name":"other","size":16}])json");
  const auto rejected =
      MpkDecoder{}.decode_json(ambiguous, monolithic_topology(), "ambiguous-output.json");
  check(!rejected && rejected.error->code == MpkDecodeErrorCode::InvalidKernelArity,
        "an invalid arity is rejected before publication inference");

  const auto extra_authority =
      replace_once(direct, R"json("name":"synthetic",)json",
                   R"json("name":"synthetic","execution_contract":{},)json");
  const auto authority_rejected =
      MpkDecoder{}.decode_json(extra_authority, monolithic_topology(), "second-authority.json");
  check(!authority_rejected && authority_rejected.error->code == MpkDecodeErrorCode::InvalidField &&
            authority_rejected.error->json_path == "$.execution_contract",
        "a superseded second execution authority is rejected rather than ignored");
}

void test_exact_multi_mla_evidence() {
  const auto encoder_topology = monolithic_topology(16U, 32U);
  const auto decoder_topology = monolithic_topology(32U, 8U);
  const std::vector<MlaStageExecutableEvidence> evidence{
      {"MLA_decoder", "decoder.elf", decoder_topology},
      {"MLA_encoder", "encoder.so", encoder_topology},
  };
  const auto result = MpkDecoder{}.decode_json(two_mla_manifest(), evidence, "two-mla.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "two MLA stages join evidence by identity, not list order");
  const auto& plan = *result.plan;
  check(plan.mla_stage_count() == 2U, "two immutable MLA stage keys are indexed");
  check(plan.mla_stage(0)->key.logical_stage_id == "MLA_encoder" &&
            plan.mla_stage(0)->key.executable == "encoder.so" &&
            plan.mla_stage(1)->key.logical_stage_id == "MLA_decoder" &&
            plan.mla_stage(1)->key.executable == "decoder.elf",
        "stage keys preserve MPK graph order and exact executable tokens");
  check(plan.mla_stage_for_identity("MLA_encoder", "encoder.so") == plan.mla_stage(0) &&
            plan.mla_stage_for_identity("MLA_encoder", "decoder.elf") == nullptr,
        "stage identity lookup requires both the MPK logical id and executable token");
  check(plan.backend_ports(0, BackendPortDirection::Input).size() == 1U &&
            plan.backend_ports(0, BackendPortDirection::Output).front().physical_extent_bytes ==
                32U &&
            plan.backend_ports(1, BackendPortDirection::Input).front().physical_extent_bytes ==
                32U &&
            plan.backend_ports(1, BackendPortDirection::Output).front().physical_extent_bytes == 8U,
        "each stage retains an independent dense ordered port span");

  auto missing = evidence;
  missing.pop_back();
  const auto missing_result = MpkDecoder{}.decode_json(two_mla_manifest(), missing, "two-mla.json");
  check(!missing_result &&
            missing_result.error->code == MpkDecodeErrorCode::MissingMlaExecutableEvidence,
        "missing exact stage evidence fails before plan creation");

  auto wrong = evidence;
  wrong[1].executable = "decoder.elf";
  const auto wrong_result = MpkDecoder{}.decode_json(two_mla_manifest(), wrong, "two-mla.json");
  check(!wrong_result &&
            wrong_result.error->code == MpkDecodeErrorCode::MissingMlaExecutableEvidence,
        "swapped executable identity cannot bind by topology position");

  const auto ambiguous_single =
      MpkDecoder{}.decode_json(two_mla_manifest(), encoder_topology, "two-mla.json");
  check(!ambiguous_single && ambiguous_single.error->code == MpkDecodeErrorCode::MultipleMlaStages,
        "single-topology compatibility API rejects a multi-stage manifest");

  const auto typed_manifest = two_mla_with_a65_module_manifest();
  const std::vector<HostTvmExecutableEvidence> host_evidence{{
      "APU_module",
      "middle.so",
      {"arm_3_i0"},
      {{"float32", {1, 8}}},
      {{"float32", {1, 8}}},
      {0},
  }};
  auto typed_mla_evidence = evidence;
  typed_mla_evidence[1].executable = "encoder.elf";
  const auto typed_result = MpkDecoder{}.decode_json(typed_manifest, typed_mla_evidence,
                                                     host_evidence, "two-mla-a65-typed.json");
  if (!typed_result && typed_result.error.has_value()) {
    std::cerr << typed_result.error->json_path << ": " << typed_result.error->detail << "\n";
  }
  check(static_cast<bool>(typed_result), "typed A65 stage joins exact structural module evidence");
  const auto& host_op = typed_result.plan->ops().at(1);
  check(host_op.kind == OpKind::HostTvm,
        "processor A65 lowers to the explicit host TVM operation kind");
  const auto& host_config = std::get<HostTvmOpConfig>(host_op.config);
  check(host_config.executable == "middle.so" &&
            host_config.output_alias_input == std::vector<std::int32_t>{0},
        "host executable identity and structurally proven alias are immutable");
  const auto& host_output = typed_result.plan->values().at(host_op.outputs.front());
  check(host_output.read_expression.has_value() &&
            host_output.read_expression->source_value_id == host_op.inputs.front(),
        "exact TVM __nop is an address view rather than a materialized allocation");

  for (const auto* version : {"2.1.0", "2.1.3", "3.0.0", "99.0.0"}) {
    const auto manifest = replace_once(typed_manifest, "2.0.0", version);
    const auto result = MpkDecoder{}.decode_json(manifest, typed_mla_evidence, host_evidence);
    check(static_cast<bool>(result), "typed A65 capability is independent of compiler version");
    const auto missing_host = MpkDecoder{}.decode_json(manifest, typed_mla_evidence);
    check(!missing_host && missing_host.error &&
              missing_host.error->code == MpkDecodeErrorCode::UnsupportedHostModule,
          "A65 still requires exact host executable evidence");
  }

  auto linked_parameter_evidence = host_evidence;
  linked_parameter_evidence.front().input_names = {"arm_3_i0", "linked_weight"};
  linked_parameter_evidence.front().input_types = {{"float32", {1, 8}}, {"float32", {8, 8}}};
  linked_parameter_evidence.front().argument_names = linked_parameter_evidence.front().input_names;
  linked_parameter_evidence.front().argument_types = linked_parameter_evidence.front().input_types;
  const auto linked_parameter_result =
      MpkDecoder{}.decode_json(typed_manifest, typed_mla_evidence, linked_parameter_evidence,
                               "two-mla-a65-linked-parameter.json");
  check(static_cast<bool>(linked_parameter_result),
        "GraphExecutor linked parameters are separated from external A65 inputs");
  const auto& linked_host =
      std::get<HostTvmOpConfig>(linked_parameter_result.plan->ops().at(1).config);
  check(linked_host.input_names == std::vector<std::string>{"arm_3_i0"} &&
            linked_host.linked_parameter_names == std::vector<std::string>{"linked_weight"},
        "A65 evidence preserves the disjoint external and linked argument sets");

  auto int64_manifest = replace_once(typed_manifest, "\"scalar\":\"float32\",\"shape\":[1,8]",
                                     "\"scalar\":\"int64\",\"shape\":[1,4]");
  int64_manifest = replace_once(int64_manifest, "\"scalar\":\"float32\",\"shape\":[1,8]",
                                "\"scalar\":\"int64\",\"shape\":[1,4]");
  auto int64_evidence = host_evidence;
  int64_evidence.front().input_types = {{"int64", {1, 4}}};
  int64_evidence.front().output_types = {{"int64", {1, 4}}};
  const auto int64_result = MpkDecoder{}.decode_json(int64_manifest, typed_mla_evidence,
                                                     int64_evidence, "two-mla-a65-int64.json");
  check(static_cast<bool>(int64_result),
        "typed A65 INT64 ports use the exact registered 8-byte DLPack mapping");

  auto wrong_host_evidence = host_evidence;
  wrong_host_evidence.front().input_names.front() = "guessed_input";
  const auto wrong_host = MpkDecoder{}.decode_json(typed_manifest, typed_mla_evidence,
                                                   wrong_host_evidence, "two-mla-a65-typed.json");
  check(!wrong_host && wrong_host.error->code == MpkDecodeErrorCode::ConfigurationMismatch,
        "MPK and embedded GraphExecutor ports must agree exactly");
}

int validate_explicit_pair(const char* manifest_path, const char* elf_path) {
  MlaElfIoTopology topology;
  if (!read_mla_elf_io_topology(elf_path, &topology)) {
    std::cerr << topology.error << "\n";
    return 2;
  }
  const auto result = MpkDecoder{}.decode_file(manifest_path, topology);
  if (!result) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
    return 1;
  }
  std::size_t ifm_count = 0U;
  std::size_t ofm_count = 0U;
  for (const auto& port : result.plan->backend_ports()) {
    if (port.direction == BackendPortDirection::Input) {
      ++ifm_count;
    } else {
      ++ofm_count;
    }
  }
  std::cout << "inputs=" << result.plan->model_inputs().size()
            << " outputs=" << result.plan->model_outputs().size() << " ifm=" << ifm_count
            << " ofm=" << ofm_count << "\n";
  return 0;
}

} // namespace

int main(const int argc, char** argv) {
  if (argc == 3) {
    return validate_explicit_pair(argv[1], argv[2]);
  }
  check(argc == 1, "usage: unit_mpk_decoder_test [manifest elf]");
  test_exact_registry();
  test_compiler_version_does_not_restrict_admission();
  test_success_and_immutable_contract();
  test_explicit_cast_input_dtype();
  test_generic_cast_does_not_invent_image_layout();
  test_detess_byte_carrier_keeps_logical_frame();
  test_unpack_tiled_carriers_keep_storage_geometry();
  test_dense_ifm_tail_padding();
  test_unpack_and_slice_are_read_expressions();
  test_reshape_is_an_exact_read_expression();
  test_registered_detess_layout_is_preserved_through_dequant();
  test_resnet_batch_flatten_is_transparent_to_fused_graph227();
  test_fused_ingress_layout_evidence_authors_exact_descriptor_axes();
  test_tessellate_keeps_yolov8_semantic_shape_separate_from_packed_carrier();
  test_standalone_quantize_authors_exact_graph222_layout();
  test_qmla_output_physical_extent_and_row_pitch();
  test_exact_multi_mla_evidence();
  test_direct_publication_without_passthrough();
  test_fail_closed_cases();
  std::cout << "unit_mpk_decoder_test: PASS\n";
  return 0;
}
