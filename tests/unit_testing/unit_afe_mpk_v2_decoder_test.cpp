#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/AfeMpkV2Decoder.h"
#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/KernelRegistry.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

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
    "input_nodes":[{"name":"input0","size":16},{"name":"input1","size":16}],
    "plugins":[
      {
        "name":"pack","sequence":1,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pack_transform","params":{"input_shapes":[[1,16],[1,16]],
            "output_shapes":[[1,32]]}},
        "input_nodes":[{"name":"input0","size":16},{"name":"input1","size":16}],
        "output_nodes":[{"name":"packed_ifm","size":32}]
      },
      {
        "name":"MLA_0","sequence":2,"processor":"MLA","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "number_of_quads_to_user":4},
        "input_nodes":[{"name":"packed_ifm","size":32}],
        "output_nodes":[{"name":"packed_ofm","size":32}],
        "resources":{"executable":"synthetic_mla.elf"}
      },
      {
        "name":"unpack","sequence":3,"processor":"EV74","type":"sgpProcess",
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
        "name":"slice0","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,0],"end":[1,2,2,1],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked0","size":16}],
        "output_nodes":[{"name":"slice0_out","size":4}]
      },
      {
        "name":"slice1","sequence":5,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"slice_transform","params":{
            "begin":[0,0,0,1],"end":[1,2,2,2],
            "input_shape":[1,2,2,4],"output_shape":[1,2,2,1],
            "input_shapes":[[1,2,2,4]],"output_shapes":[[1,2,2,1]]}},
        "input_nodes":[{"name":"unpacked1","size":16}],
        "output_nodes":[{"name":"slice1_out","size":4}]
      },
      {
        "name":"publish","sequence":6,"processor":"EV74","type":"sgpProcess",
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

const std::string& qmla_padded_cast_output_manifest() {
  static const std::string manifest = R"json({
    "name":"qmla-padded-cast","model_sdk_version":"2.1.0",
    "input_nodes":[{"name":"input","size":16}],
    "plugins":[
      {"name":"MLA_0","sequence":1,"processor":"MLA","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "number_of_quads_to_user":4,
                        "input_types":[{"scalar":"float32","shape":[1,4]}],
                        "output_types":[{"scalar":"bfloat16","shape":[1,1225,4]}]},
       "input_nodes":[{"name":"input","size":16}],
       "output_nodes":[{"name":"mla_out","size":9800}],
       "resources":{"executable":"MLA_0.elf"}},
      {"name":"cast_0","sequence":2,"processor":"EV74","type":"sgpProcess",
       "config_params":{"desired_batch_size":1,"actual_batch_size":1,
                        "kernel":"cast","params":{"out_dtype":"float32",
                          "input_shapes":[[1,1225,4]],
                          "output_shapes":[[1,1225,4]]}},
       "input_nodes":[{"name":"mla_out","size":9800}],
       "output_nodes":[{"name":"cast_out","size":19600}]}
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
                  const AfeMpkV2DecodeErrorCode expected, const char* message) {
  const auto result = AfeMpkV2Decoder{}.decode_json(manifest, topology, "synthetic.json");
  check(!result, message);
  check(result.error.has_value() && result.error->code == expected, message);
  check(!result.error->json_path.empty() && !result.error->detail.empty(),
        "failure has stable path and detail");
}

void test_exact_registry() {
  const auto legacy = lookup_exact_kernel("2.0.0", "EV74", "cast_transform");
  const auto modern_alias = lookup_exact_kernel("2.0.0", "EV74", "cast");
  check(legacy.has_value() && legacy->kind == OpKind::Cast, "legacy cast token has an exact entry");
  check(modern_alias.has_value() && modern_alias->kind == OpKind::Cast,
        "new cast spelling has its own exact entry");
  check(!lookup_exact_kernel("2.0.0", "EV74", "prefix_cast_transform").has_value(),
        "registry rejects substring matches");
  check(!lookup_exact_kernel("2.0.0", "ev74", "cast_transform").has_value(),
        "registry rejects processor case folding");
  check(!lookup_exact_kernel("2.0.1", "EV74", "cast_transform").has_value(),
        "registry rejects version fallback");
  const auto batch_flatten =
      lookup_exact_kernel("2.1.0", "EV74", "batch_flatten_transform");
  check(batch_flatten && batch_flatten->kind == OpKind::Reshape,
        "AFE 2.1 batch flatten has one exact address-view registry entry");
  check(!lookup_exact_kernel("2.0.0", "EV74", "batch_flatten_transform"),
        "batch flatten does not acquire a version fallback");
}

void test_success_and_immutable_contract() {
  const auto result =
      AfeMpkV2Decoder{}.decode_json(valid_manifest(), monolithic_topology(), "synthetic.json");
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
  const auto result = AfeMpkV2Decoder{}.decode_json(packed_read_manifest(),
                                                    monolithic_topology(32U, 32U),
                                                    "packed-read-synthetic.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "packed read-expression manifest decodes");
  const auto& plan = *result.plan;
  check(plan.backend_ports().size() == 2U, "packed route remains one IFM and one OFM");
  check(plan.model_outputs().size() == 2U, "both logical reads are published");
  const auto* pack = std::get_if<PackOpConfig>(&plan.ops().front().config);
  check(pack != nullptr && pack->components.size() == 2U &&
            pack->components[0].parent_offset == 0U && pack->components[0].stored_bytes == 16U &&
            pack->components[1].parent_offset == 16U && pack->components[1].stored_bytes == 16U,
        "Pack carries exact ordered parent placement");

  // Value order is: two public inputs, pack, MLA, two unpack reads, two
  // slice reads.  Every read remains rooted in the
  // single materialized MLA OFM; neither Unpack nor Slice schedules work.
  const auto require_read = [&](const ValueId id, const std::uint64_t offset) {
    const auto* value = plan.value(id);
    check(value != nullptr && value->read_expression.has_value(),
          "logical value carries a compiled read expression");
    check(value->read_expression->source_value_id == 3U,
          "logical read is composed to the physical MLA carrier");
    check(value->read_expression->byte_offset == offset,
          "logical read preserves its exact carrier-relative offset");
    check(value->read_expression->stride_bytes == std::vector<std::int64_t>({16, 8, 4, 1}),
          "logical read preserves the exact inherited byte strides");
  };
  require_read(4U, 0U);
  require_read(5U, 16U);
  require_read(6U, 0U);
  require_read(7U, 17U);
  check(plan.ops().size() == 5U && plan.values().size() == 8U,
        "PassThrough creates neither an executable op nor duplicate output values");
  check(plan.model_outputs()[0].value_id == 6U && plan.model_outputs()[1].value_id == 7U,
        "publication points directly at the two compiled Slice views");

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
      AfeMpkV2Decoder{}.decode_json(reshape_manifest(), monolithic_topology(16U, 8U),
                                    "reshape.json");
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
      AfeMpkV2Decoder{}.decode_json(mismatch, monolithic_topology(16U, 8U),
                                    "reshape-mismatch.json");
  check(!rejected && rejected.error->code == AfeMpkV2DecodeErrorCode::ValueSizeMismatch,
        "reshape that changes the byte extent fails closed");
}

void test_registered_detess_layout_is_preserved_through_dequant() {
  const auto result = AfeMpkV2Decoder{}.decode_json(
      detess_dequant_manifest(), monolithic_topology(16U, 16U), "detess-layout.json");
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
  const auto result = AfeMpkV2Decoder{}.decode_json(
      resnet_batch_flatten_manifest(), monolithic_topology(16U, 1008U),
      "resnet-batch-flatten.json");
  if (!result && result.error.has_value()) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "exact ResNet batch-flatten chain decodes");
  const auto foreign_reshape_grammar = replace_once(
      resnet_batch_flatten_manifest(),
      "\"kernel\":\"batch_flatten_transform\",\"params\":{\n"
      "                          \"input_shapes\"",
      "\"kernel\":\"batch_flatten_transform\",\"params\":{\n"
      "                          \"newshape\":[1,1000],\"input_shapes\"");
  const auto grammar_rejected = AfeMpkV2Decoder{}.decode_json(
      foreign_reshape_grammar, monolithic_topology(16U, 1008U),
      "resnet-batch-flatten-foreign-grammar.json");
  check(!grammar_rejected &&
            grammar_rejected.error->code == AfeMpkV2DecodeErrorCode::InvalidField,
        "batch flatten accepts only its exact two-shape-list grammar");
  const auto& plan = *result.plan;
  check(plan.ops().size() == 4U && plan.ops()[1].kind == OpKind::Detessellate &&
            plan.ops()[2].kind == OpKind::Reshape &&
            plan.ops()[2].kernel == "batch_flatten_transform" &&
            plan.ops()[3].kind == OpKind::Dequantize,
        "batch flatten retains exact compiler identity as the existing Reshape relation");
  const auto* flattened = plan.value(plan.ops()[2].outputs.front());
  check(flattened && flattened->required_bytes == 1000U &&
            flattened->logical_shape == TensorShape({1, 1000}) &&
            flattened->read_expression &&
            flattened->read_expression->source_value_id == plan.ops()[1].outputs.front() &&
            flattened->read_expression->byte_offset == 0U &&
            flattened->read_expression->stride_bytes == std::vector<std::int64_t>({1000, 1}),
        "batch flatten is one zero-offset same-byte ordered read expression");

  std::vector<ValueId> relation_values;
  check(resolve_exact_private_ordered_relation_path(plan, 1U, 3U, &relation_values) &&
            relation_values ==
                std::vector<ValueId>({plan.ops()[1].outputs.front(),
                                      plan.ops()[2].outputs.front()}),
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

  const auto arena = FrameSlotArenaPlan::compile(
      plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
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
    const std::array<std::uint8_t, 4U> expected_axes{
        SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W, SIMA_EV_AXIS_C};
    check(input.shape.rank == expected_shape.size() &&
              output.shape.rank == expected_shape.size(),
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
              contract->payload.runtime_output_logical_layout_list ==
                  std::vector<std::string>{""},
          "graph227 publication retains the post-view flattened logical contract");
  }

  auto two_views = replace_once(
      resnet_batch_flatten_manifest(),
      "\"input_nodes\":[{\"name\":\"EV_1/batch_flatten_0\",\"size\":1000}],\n"
      "       \"output_nodes\":[{\"name\":\"dequantize_1/resnetv17_dense0_fwd\","
      "\"size\":4000}]",
      "\"input_nodes\":[{\"name\":\"second_view\",\"size\":1000}],\n"
      "       \"output_nodes\":[{\"name\":\"dequantize_1/resnetv17_dense0_fwd\","
      "\"size\":4000}]");
  two_views = replace_once(
      std::move(two_views),
      "\"input_data_type\":\"int8\",\"input_shapes\":[[1,1000]],\n"
      "                          \"output_shapes\":[[1,1000]]",
      "\"input_data_type\":\"int8\",\"input_shapes\":[[1,1,1000]],\n"
      "                          \"output_shapes\":[[1,1,1000]]");
  two_views = replace_once(
      std::move(two_views), "{\"name\":\"dequantize_1\",\"sequence\":4",
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
  two_views = replace_once(std::move(two_views),
                           "{\"name\":\"publish\",\"sequence\":5",
                           "{\"name\":\"publish\",\"sequence\":6");
  const auto twice_decoded = AfeMpkV2Decoder{}.decode_json(
      two_views, monolithic_topology(16U, 1008U), "resnet-two-views.json");
  check(static_cast<bool>(twice_decoded) && twice_decoded.plan->ops().size() == 5U,
        "two consecutive exact views decode without inventing work");
  std::vector<ValueId> twice_values;
  check(resolve_exact_private_ordered_relation_path(
            *twice_decoded.plan, 1U, 4U, &twice_values) && twice_values.size() == 3U,
        "shared relation proof composes two private dense order-preserving views");
  const auto twice_physical = PhysicalExecutionLowerer::lower(*twice_decoded.plan, &error);
  std::size_t twice_graph227 = 0U;
  if (twice_physical) {
    for (const auto& command : twice_physical->commands) {
      twice_graph227 += command.graph_id == 227U ? 1U : 0U;
    }
  }
  check(twice_physical && twice_graph227 == 1U &&
            !twice_physical->command_for_semantic_op[2U] &&
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
  const auto observed = AfeMpkV2Decoder{}.decode_json(
      observed_relation, monolithic_topology(16U, 1008U),
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
  const std::array<std::uint8_t, 4U> expected_axes{
      SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W, SIMA_EV_AXIS_C};

  for (const auto& test_case : cases) {
    const auto decoded = AfeMpkV2Decoder{}.decode_json(
        test_case.manifest, monolithic_topology(test_case.mla_ifm_bytes, 16U),
        std::string(test_case.label) + ".json");
    if (!decoded && decoded.error.has_value()) {
      std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
    }
    check(static_cast<bool>(decoded), "raw fused-ingress AFE manifest decodes");
    const auto& plan = *decoded.plan;
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
    const auto arena = FrameSlotArenaPlan::compile(
        plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
        kLegacyEvoCmaRegionAlignmentBytes, &error);
    check(arena.has_value(), "raw fused-ingress frame arena compiles");
    const auto contract = build_dmabuf_plan_processcvu_command_contract(
        plan, *physical, commands, *arena, &error);
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
  const auto decoded = AfeMpkV2Decoder{}.decode_json(
      yolov8_quant_tess_ingress_manifest(), monolithic_topology(1228800U, 16U),
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
  const auto decoded = AfeMpkV2Decoder{}.decode_json(
      standalone_quant_mla_manifest(), monolithic_topology(64U, 16U),
      "standalone-quant.json");
  if (!decoded && decoded.error.has_value()) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded), "standalone Quantize-to-MLA manifest decodes");

  const auto& plan = *decoded.plan;
  const auto& quant = plan.ops().front();
  const auto* input = plan.value(quant.inputs.front());
  const auto* output = plan.value(quant.outputs.front());
  check(quant.kind == OpKind::Quantize && input && output &&
            input->logical_layout == "HWC" && output->logical_layout == "HWC",
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
  const auto arena = FrameSlotArenaPlan::compile(
      plan, *physical, FrameSlotArenaReuse::DisjointLifetimes,
      kLegacyEvoCmaRegionAlignmentBytes, &error);
  check(arena.has_value(), "standalone Quantize frame arena compiles");
  const auto contract = build_dmabuf_plan_processcvu_command_contract(
      plan, *physical, commands, *arena, &error);
  check(contract.has_value(), "standalone graph 222 descriptor contract builds");

  const auto& payload = contract->payload;
  check(payload.graph_id == 222 && payload.input_tensors.size() == 1U &&
            payload.output_tensors.size() == 1U && payload.round_off == 1 &&
            payload.has_q_scale && payload.q_scale == 0.25 && payload.has_q_zp &&
            payload.q_zp == 0 &&
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
  const std::array<std::uint8_t, 4U> expected_axes{
      SIMA_EV_AXIS_N, SIMA_EV_AXIS_H, SIMA_EV_AXIS_W, SIMA_EV_AXIS_C};
  for (std::size_t axis = 0; axis < expected_axes.size(); ++axis) {
    check(input_desc.shape.axis_semantics[axis] == expected_axes[axis] &&
              output_desc.shape.axis_semantics[axis] == expected_axes[axis],
          "graph 222 endpoint descriptors preserve identical N/H/W/C axes");
  }

  const auto contradictory = replace_once(
      standalone_quant_mla_manifest(), "\"output_shapes\":[[1,2,2,16]]",
      "\"output_shapes\":[[1,2,1,32]]");
  const auto rejected = AfeMpkV2Decoder{}.decode_json(
      contradictory, monolithic_topology(64U, 16U),
      "standalone-quant-contradictory-shape.json");
  check(!rejected && rejected.error.has_value() &&
            rejected.error->code == AfeMpkV2DecodeErrorCode::ConfigurationMismatch &&
            rejected.error->detail.find("contradictory exact endpoint shapes") !=
                std::string::npos,
        "standalone graph 222 rejects contradictory exact endpoint geometry");
}

void test_qmla_output_physical_extent_and_row_pitch() {
  const auto decoded = AfeMpkV2Decoder{}.decode_json(
      qmla_padded_output_manifest(), qmla_padded_topology(), "qmla-padded.json");
  if (!decoded && decoded.error) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded), "typed QMLA padded output decodes");
  const auto& plan = *decoded.plan;
  const auto& port = plan.backend_ports(0U, BackendPortDirection::Output).front();
  const auto* value = plan.value(port.value_id);
  const auto* carrier = value && value->storage_binding
                            ? plan.carrier(value->storage_binding->carrier_id)
                            : nullptr;
  check(port.physical_extent_bytes == 110400U && value && value->required_bytes == 109200U &&
            value->storage_binding && value->storage_binding->physical_span == 110396U &&
            value->storage_binding->stride_bytes ==
                std::vector<std::int64_t>({110400, 368, 4}) &&
            carrier && carrier->required_bytes == 110400U,
        "QMLA physical carrier remains separate from logical addressed logits");

  const auto contradictory = AfeMpkV2Decoder{}.decode_json(
      qmla_padded_output_manifest(), qmla_padded_topology(110416U),
      "qmla-contradictory.json");
  check(!contradictory && contradictory.error &&
            contradictory.error->code == AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
        "unregistered larger QMLA extent fails closed instead of guessing padding");

  auto missing = qmla_padded_topology();
  missing.ofm_extent_bytes.clear();
  const auto missing_result = AfeMpkV2Decoder{}.decode_json(
      qmla_padded_output_manifest(), missing, "qmla-missing-extent.json");
  check(!missing_result && missing_result.error &&
            missing_result.error->code == AfeMpkV2DecodeErrorCode::ElfTopologyInvalid,
        "missing QMLA extent evidence fails closed");
}

void test_cast_preserves_qmla_layout_authority() {
  const auto decoded = AfeMpkV2Decoder{}.decode_json(
      qmla_padded_cast_output_manifest(), qmla_padded_topology(19600U),
      "qmla-padded-cast.json");
  if (!decoded && decoded.error) {
    std::cerr << decoded.error->json_path << ": " << decoded.error->detail << "\n";
  }
  check(static_cast<bool>(decoded),
        "Cast after a row-padded QMLA output preserves the QMLA layout authority");
  const auto& cast = decoded.plan->ops().at(1U);
  const auto* input = decoded.plan->value(cast.inputs.front());
  const auto* output = decoded.plan->value(cast.outputs.front());
  check(input && output && input->logical_layout == "normal" &&
            output->logical_layout == input->logical_layout,
        "Cast propagates its proven input layout instead of inventing HWC axes");
}

void test_fail_closed_cases() {
  const auto topology = monolithic_topology();
  expect_error(replace_once(valid_manifest(), "2.0.0", "2.0.1"), topology,
               AfeMpkV2DecodeErrorCode::UnsupportedContractVersion,
               "unsupported version fails closed");
  expect_error(replace_once(valid_manifest(), "cast_transform", "cast_transform_suffix"), topology,
               AfeMpkV2DecodeErrorCode::UnsupportedKernel, "kernel substring is not an alias");
  expect_error(replace_once(valid_manifest(),
                            "\"name\":\"cast0\",\"size\":8}],\n        \"output_nodes\"",
                            "\"name\":\"missing\",\"size\":8}],\n        \"output_nodes\""),
               topology, AfeMpkV2DecodeErrorCode::MissingProducer,
               "missing full-name producer fails closed");
  expect_error(replace_once(valid_manifest(), "\"params\":{\"out_dtype\":\"bfloat16\"",
                            "\"params\":{\"ignored\":1,\"out_dtype\":\"bfloat16\""),
               topology, AfeMpkV2DecodeErrorCode::InvalidField,
               "untyped extra operation config is not ignored");
  expect_error(replace_once(valid_manifest(),
                            "\"input_shapes\":[[1,4]],\"output_shapes\":[[1,4]]",
                            "\"input_shapes\":[[1,4]],\"output_shapes\":[[2,2]]"),
               topology, AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
               "shape-preserving transform rejects contradictory exact endpoint shapes");

  auto two_ifm = topology;
  two_ifm.monolithic_ifm = false;
  two_ifm.monolithic_ifm_extent_bytes = 0U;
  two_ifm.ifm_symbol_names = {"data.ifm.persistent.qmla_ifm_0.b0",
                              "data.ifm.persistent.qmla_ifm_1.b0"};
  two_ifm.ifm_extent_bytes = {8U, 8U};
  expect_error(valid_manifest(), two_ifm, AfeMpkV2DecodeErrorCode::ElfTopologyMismatch,
               "MPK/ELF port arity mismatch fails before plan creation");

  auto conflict = topology;
  conflict.ifm_layout_conflict = true;
  expect_error(valid_manifest(), conflict, AfeMpkV2DecodeErrorCode::ElfTopologyInvalid,
               "ambiguous ELF layout fails closed");
}

void test_direct_publication_without_passthrough() {
  const auto direct = replace_once(
      valid_manifest(),
      R"json(,
      {
        "name":"publish","sequence":4,"processor":"EV74","type":"sgpProcess",
        "config_params":{"desired_batch_size":1,"actual_batch_size":1,
          "kernel":"pass_through","params":{}},
        "input_nodes":[{"name":"decorated/model/output:0","size":16}],
        "output_nodes":[{"name":"pass_through_out_0","size":16}]
      })json",
      "");
  const auto result =
      AfeMpkV2Decoder{}.decode_json(direct, monolithic_topology(), "direct-output.json");
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
      AfeMpkV2Decoder{}.decode_json(ambiguous, monolithic_topology(), "ambiguous-output.json");
  check(!rejected &&
            rejected.error->code == AfeMpkV2DecodeErrorCode::InvalidKernelArity,
        "an invalid arity is rejected before publication inference");

  const auto extra_authority =
      replace_once(direct, R"json("name":"synthetic",)json",
                   R"json("name":"synthetic","execution_contract":{},)json");
  const auto authority_rejected = AfeMpkV2Decoder{}.decode_json(
      extra_authority, monolithic_topology(), "second-authority.json");
  check(!authority_rejected &&
            authority_rejected.error->code == AfeMpkV2DecodeErrorCode::InvalidField &&
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
  const auto result = AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), evidence, "two-mla.json");
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
            plan.backend_ports(1, BackendPortDirection::Output).front().physical_extent_bytes ==
                8U,
        "each stage retains an independent dense ordered port span");

  auto missing = evidence;
  missing.pop_back();
  const auto missing_result =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), missing, "two-mla.json");
  check(!missing_result &&
            missing_result.error->code == AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence,
        "missing exact stage evidence fails before plan creation");

  auto wrong = evidence;
  wrong[1].executable = "decoder.elf";
  const auto wrong_result =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), wrong, "two-mla.json");
  check(!wrong_result &&
            wrong_result.error->code == AfeMpkV2DecodeErrorCode::MissingMlaExecutableEvidence,
        "swapped executable identity cannot bind by topology position");

  const auto ambiguous_single =
      AfeMpkV2Decoder{}.decode_json(two_mla_manifest(), encoder_topology, "two-mla.json");
  check(!ambiguous_single &&
            ambiguous_single.error->code == AfeMpkV2DecodeErrorCode::MultipleMlaStages,
        "single-topology compatibility API rejects a multi-stage manifest");

  const auto a65_result = AfeMpkV2Decoder{}.decode_json(two_mla_with_a65_module_manifest(),
                                                        evidence, "two-mla-a65.json");
  check(!a65_result && a65_result.error->code == AfeMpkV2DecodeErrorCode::UnsupportedHostModule &&
            a65_result.error->json_path.find("config_params") != std::string::npos,
        "the frozen 2.0 contract cannot acquire A65 meaning from a filename suffix");

  const auto typed_manifest =
      replace_once(two_mla_with_a65_module_manifest(), "\"2.0.0\"", "\"2.1.0\"");
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
  const auto typed_result = AfeMpkV2Decoder{}.decode_json(typed_manifest, typed_mla_evidence,
                                                          host_evidence, "two-mla-a65-typed.json");
  if (!typed_result && typed_result.error.has_value()) {
    std::cerr << typed_result.error->json_path << ": " << typed_result.error->detail << "\n";
  }
  check(static_cast<bool>(typed_result),
        "typed 2.1 A65 stage joins exact structural module evidence");
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

  auto int64_manifest = replace_once(typed_manifest,
                                     "\"scalar\":\"float32\",\"shape\":[1,8]",
                                     "\"scalar\":\"int64\",\"shape\":[1,4]");
  int64_manifest = replace_once(int64_manifest,
                                "\"scalar\":\"float32\",\"shape\":[1,8]",
                                "\"scalar\":\"int64\",\"shape\":[1,4]");
  auto int64_evidence = host_evidence;
  int64_evidence.front().input_types = {{"int64", {1, 4}}};
  int64_evidence.front().output_types = {{"int64", {1, 4}}};
  const auto int64_result = AfeMpkV2Decoder{}.decode_json(
      int64_manifest, typed_mla_evidence, int64_evidence, "two-mla-a65-int64.json");
  check(static_cast<bool>(int64_result),
        "typed A65 INT64 ports use the exact registered 8-byte DLPack mapping");

  auto wrong_host_evidence = host_evidence;
  wrong_host_evidence.front().input_names.front() = "guessed_input";
  const auto wrong_host = AfeMpkV2Decoder{}.decode_json(
      typed_manifest, typed_mla_evidence, wrong_host_evidence, "two-mla-a65-typed.json");
  check(!wrong_host && wrong_host.error->code == AfeMpkV2DecodeErrorCode::ConfigurationMismatch,
        "MPK and embedded GraphExecutor ports must agree exactly");
}

int validate_explicit_pair(const char* manifest_path, const char* elf_path) {
  MlaElfIoTopology topology;
  if (!read_mla_elf_io_topology(elf_path, &topology)) {
    std::cerr << topology.error << "\n";
    return 2;
  }
  const auto result = AfeMpkV2Decoder{}.decode_file(manifest_path, topology);
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
  check(argc == 1, "usage: unit_afe_mpk_v2_decoder_test [manifest elf]");
  test_exact_registry();
  test_success_and_immutable_contract();
  test_unpack_and_slice_are_read_expressions();
  test_reshape_is_an_exact_read_expression();
  test_registered_detess_layout_is_preserved_through_dequant();
  test_resnet_batch_flatten_is_transparent_to_fused_graph227();
  test_fused_ingress_layout_evidence_authors_exact_descriptor_axes();
  test_tessellate_keeps_yolov8_semantic_shape_separate_from_packed_carrier();
  test_standalone_quantize_authors_exact_graph222_layout();
  test_qmla_output_physical_extent_and_row_pitch();
  test_cast_preserves_qmla_layout_authority();
  test_exact_multi_mla_evidence();
  test_direct_publication_without_passthrough();
  test_fail_closed_cases();
  std::cout << "unit_afe_mpk_v2_decoder_test: PASS\n";
  return 0;
}
