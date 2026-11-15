#pragma once
// Segment Anything prompt-decoder MPK manifests for unit tests: the real
// MobileSAM decoder manifest, and EdgeTAM and EfficientSAM decoders
// synthesized from the sima-neat/models interfaces with the plugin topology of
// the real manifests. No ELF or model weights are involved.
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sam_test {
// MobileSAM prompt decoder manifest (models repository build A, mixed W8A8),
// kept to its plugin graph. Mask logits [1,256,256,16] and IoU [1,16] share
// one MLA buffer behind unpack, slice and dequantize plugins.
inline constexpr const char* kMobileSamDecoderMpk = R"json({
 "name": "mobile_sam_w8a8_mixed_A_prompt_decoder", "sequence": 1, "model_sdk_version": "3.0.0",
 "input_nodes": [
  {"name": "prompt_decoder_image_embeddings", "type": "buffer", "size": 4194304, "input_range": [-0.6191582679748535, 0.5514378547668457]},
  {"name": "prompt_decoder_point_encoding", "type": "buffer", "size": 6144, "input_range": [-0.9999993443489075, 1.0]},
  {"name": "prompt_decoder_point_types", "type": "buffer", "size": 120, "input_range": [0.0, 1.0]}
 ],
 "plugins": [
  {"name": "quantize_0", "sequence": 1, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "quantization_transform", "params": {"channel_params": [[127.56229292873613, 0]], "num_bits": 8, "rounding": "TONEAREST", "output_data_type": "int8", "input_shapes": [[1, 1, 6, 256]], "output_shapes": [[1, 1, 6, 256]]}},
   "input_nodes": [{"name": "prompt_decoder_point_encoding", "size": 6144}],
   "output_nodes": [{"name": "quantize_0", "type": "buffer", "size": 1536}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "quantize_1", "sequence": 2, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "quantization_transform", "params": {"channel_params": [[255.12458585747225, -128]], "num_bits": 8, "rounding": "TONEAREST", "output_data_type": "int8", "input_shapes": [[1, 1, 6, 5]], "output_shapes": [[1, 1, 6, 5]]}},
   "input_nodes": [{"name": "prompt_decoder_point_types", "size": 120}],
   "output_nodes": [{"name": "quantize_1", "type": "buffer", "size": 30}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "quantize_2", "sequence": 3, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "quantization_transform", "params": {"channel_params": [[216.899465246583, -2]], "num_bits": 8, "rounding": "TONEAREST", "output_data_type": "int8", "input_shapes": [[1, 64, 64, 256]], "output_shapes": [[1, 64, 64, 256]]}},
   "input_nodes": [{"name": "prompt_decoder_image_embeddings", "size": 4194304}],
   "output_nodes": [{"name": "quantize_2", "type": "buffer", "size": 1048576}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "MLA_0", "sequence": 4, "processor": "MLA", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "number_of_quads_to_user": 4},
   "input_nodes": [{"name": "quantize_0", "size": 1536}, {"name": "quantize_1", "size": 30}, {"name": "quantize_2", "size": 1048576}],
   "output_nodes": [{"name": "MLA_0", "type": "buffer", "size": 1048592}],
   "resources": {"executable": "mobile_sam_w8a8_mixed_A_prompt_decoder_stage1_mla.elf"}},
  {"name": "MLA_0_ofm_unpack_transform", "sequence": 5, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "unpack_transform", "params": {"tensor_types": ["int8", "int8"], "tensor_shapes": [[1, 256, 256, 16], [1, 16]], "input_shapes": [[1, 1048592]], "output_shapes": [[1, 256, 256, 16], [1, 16]]}},
   "input_nodes": [{"name": "MLA_0", "size": 1048592}],
   "output_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "type": "buffer", "size": 1048576}, {"name": "MLA_0_ofm_unpack_transform_1", "type": "buffer", "size": 16}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "slice_MLA_0/tuple_get_item_0_slice_transform", "sequence": 6, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "slice_transform", "params": {"begin": [0, 0, 0, 0], "end": [1, 256, 256, 4], "input_shape": [1, 256, 256, 16], "output_shape": [1, 256, 256, 4], "input_shapes": [[1, 256, 256, 16]], "output_shapes": [[1, 256, 256, 4]]}},
   "input_nodes": [{"name": "MLA_0_ofm_unpack_transform_0", "size": 1048576}],
   "output_nodes": [{"name": "slice_MLA_0/tuple_get_item_0_slice_transform", "type": "buffer", "size": 262144}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "dequantize_94", "sequence": 7, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "dequantization_transform", "params": {"channel_params": [[2.1094489005250545, 70]], "input_data_type": "int8", "input_shapes": [[1, 256, 256, 4]], "output_shapes": [[1, 256, 256, 4]]}},
   "input_nodes": [{"name": "slice_MLA_0/tuple_get_item_0_slice_transform", "size": 262144}],
   "output_nodes": [{"name": "dequantize_94/prompt_decoder_masks", "type": "buffer", "size": 1048576}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "slice_MLA_0/tuple_get_item_1_slice_transform", "sequence": 8, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "slice_transform", "params": {"begin": [0, 0], "end": [1, 4], "input_shape": [1, 16], "output_shape": [1, 4], "input_shapes": [[1, 16]], "output_shapes": [[1, 4]]}},
   "input_nodes": [{"name": "MLA_0_ofm_unpack_transform_1", "size": 16}],
   "output_nodes": [{"name": "slice_MLA_0/tuple_get_item_1_slice_transform", "type": "buffer", "size": 4}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "dequantize_95", "sequence": 9, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "dequantization_transform", "params": {"channel_params": [[251.50381076556937, -128]], "input_data_type": "int8", "input_shapes": [[1, 4]], "output_shapes": [[1, 4]]}},
   "input_nodes": [{"name": "slice_MLA_0/tuple_get_item_1_slice_transform", "size": 4}],
   "output_nodes": [{"name": "dequantize_95/prompt_decoder_iou_predictions", "type": "buffer", "size": 16}],
   "resources": {"executable": "kernel_name_tbd"}},
  {"name": "PassThrough", "sequence": 10, "processor": "EV74", "type": "sgpProcess",
   "config_params": {"desired_batch_size": 1, "actual_batch_size": 1, "kernel": "pass_through", "params": {}},
   "input_nodes": [{"name": "dequantize_94/prompt_decoder_masks", "size": 1048576}, {"name": "dequantize_95/prompt_decoder_iou_predictions", "size": 16}],
   "output_nodes": [{"name": "pass_through_out_0", "type": "buffer", "size": 1048576}, {"name": "pass_through_out_1", "type": "buffer", "size": 16}],
   "resources": null}
 ]
})json";

// Prompt-decoder manifests for family members without a compiled archive here,
// synthesized from the models repository interfaces with the plugin topology of
// the real manifests (MobileSAM W8A8 above; the EfficientSAM3 BF16 seed).
inline nlohmann::json sam_plugin(const std::string& name, int sequence, const std::string& kernel,
                                 nlohmann::json params,
                                 const std::vector<std::pair<std::string, std::int64_t>>& inputs,
                                 const std::vector<std::pair<std::string, std::int64_t>>& outputs) {
  nlohmann::json plugin = {{"name", name},
                           {"sequence", sequence},
                           {"processor", kernel.empty() ? "MLA" : "EV74"},
                           {"type", "sgpProcess"},
                           {"resources", {{"executable", "kernel_name_tbd"}}}};
  plugin["config_params"] = {{"desired_batch_size", 1}, {"actual_batch_size", 1}};
  if (kernel.empty())
    plugin["config_params"]["number_of_quads_to_user"] = 4;
  else
    plugin["config_params"].update({{"kernel", kernel}, {"params", std::move(params)}});
  plugin["input_nodes"] = nlohmann::json::array();
  for (const auto& [node, size] : inputs)
    plugin["input_nodes"].push_back({{"name", node}, {"size", size}});
  plugin["output_nodes"] = nlohmann::json::array();
  for (const auto& [node, size] : outputs)
    plugin["output_nodes"].push_back({{"name", node}, {"type", "buffer"}, {"size", size}});
  return plugin;
}

inline std::int64_t sam_numel(const std::vector<std::int64_t>& shape) {
  std::int64_t n = 1;
  for (const auto v : shape)
    n *= v;
  return n;
}

// EdgeTAM mask_decoder (models/edgetam surgery.py INTERFACES, W16A16), with the
// plugin graph of the CI-built manifest: one packed BF16 MLA output unpacked to
// mask_logits [1,256,256,8], iou_scores [1,8], object_score_logits [1,8] and
// object_pointers [1,4,256], sliced to [1,256,256,4], [1,4] and [1,1] and cast
// to the float32 interface compile.py audits.
inline std::string edgetam_decoder_mpk() {
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> inputs = {
      {"image_embeddings", {1, 64, 64, 256}},
      {"high_res_features_0", {1, 256, 256, 32}},
      {"high_res_features_1", {1, 128, 128, 64}},
      {"point_coords", {1, 1, 3, 2}},
      {"point_types", {1, 1, 3, 5}}};
  nlohmann::json mpk = {{"name", "edgetam_mask_decoder"},
                        {"sequence", 1},
                        {"model_sdk_version", "3.0.0"},
                        {"input_nodes", nlohmann::json::array()},
                        {"plugins", nlohmann::json::array()}};
  int sequence = 1;
  std::vector<std::pair<std::string, std::int64_t>> mla_inputs;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& [name, shape] = inputs[i];
    const auto cast = "cast_" + std::to_string(i);
    mpk["input_nodes"].push_back(
        {{"name", name}, {"type", "buffer"}, {"size", sam_numel(shape) * 4}});
    mpk["plugins"].push_back(sam_plugin(cast, sequence++, "cast_transform",
                                        {{"in_dtype", "float32"},
                                         {"out_dtype", "bfloat16"},
                                         {"input_shapes", {shape}},
                                         {"output_shapes", {shape}}},
                                        {{name, sam_numel(shape) * 4}},
                                        {{cast, sam_numel(shape) * 2}}));
    mla_inputs.push_back({cast, sam_numel(shape) * 2});
  }
  const std::vector<std::tuple<std::string, std::vector<std::int64_t>, std::vector<std::int64_t>>>
      outputs = {{"mask_logits", {1, 256, 256, 4}, {1, 256, 256, 8}},
                 {"iou_scores", {1, 4}, {1, 8}},
                 {"object_score_logits", {1, 1}, {1, 8}},
                 {"object_pointers", {1, 4, 256}, {1, 4, 256}}};
  std::vector<std::pair<std::string, std::int64_t>> unpacked, terminal;
  nlohmann::json types = nlohmann::json::array(), shapes = nlohmann::json::array();
  std::int64_t packed = 0;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& padded = std::get<2>(outputs[i]);
    unpacked.push_back({"MLA_0_ofm_unpack_transform_" + std::to_string(i), sam_numel(padded) * 2});
    packed += unpacked.back().second;
    types.push_back("bfloat16");
    shapes.push_back(padded);
  }
  auto mla = sam_plugin("MLA_0", sequence++, "", {}, mla_inputs, {{"MLA_0", packed}});
  mla["resources"]["executable"] = "edgetam_mask_decoder_stage1_mla.elf";
  mpk["plugins"].push_back(std::move(mla));
  mpk["plugins"].push_back(sam_plugin("MLA_0_ofm_unpack_transform", sequence++, "unpack_transform",
                                      {{"tensor_types", types},
                                       {"tensor_shapes", shapes},
                                       {"input_shapes", {{1, packed}}},
                                       {"output_shapes", shapes}},
                                      {{"MLA_0", packed}}, unpacked));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& [name, shape, padded] = outputs[i];
    std::string producer = unpacked[i].first;
    if (shape != padded) {
      const auto slice = "slice_MLA_0/tuple_get_item_" + std::to_string(i) + "_slice_transform";
      mpk["plugins"].push_back(sam_plugin(slice, sequence++, "slice_transform",
                                          {{"begin", std::vector<std::int64_t>(shape.size(), 0)},
                                           {"end", shape},
                                           {"input_shape", padded},
                                           {"output_shape", shape},
                                           {"input_shapes", {padded}},
                                           {"output_shapes", {shape}}},
                                          {{producer, sam_numel(padded) * 2}},
                                          {{slice, sam_numel(shape) * 2}}));
      producer = slice;
    }
    const auto cast = "cast_" + std::to_string(inputs.size() + i) + "/" + name;
    mpk["plugins"].push_back(sam_plugin(cast, sequence++, "cast_transform",
                                        {{"in_dtype", "bfloat16"},
                                         {"out_dtype", "float32"},
                                         {"input_shapes", {shape}},
                                         {"output_shapes", {shape}}},
                                        {{producer, sam_numel(shape) * 2}},
                                        {{cast, sam_numel(shape) * 4}}));
    terminal.push_back({cast, sam_numel(shape) * 4});
  }
  std::vector<std::pair<std::string, std::int64_t>> passed;
  for (std::size_t i = 0; i < terminal.size(); ++i)
    passed.push_back({"pass_through_out_" + std::to_string(i), terminal[i].second});
  mpk["plugins"].push_back(sam_plugin("PassThrough", sequence, "pass_through",
                                      nlohmann::json::object(), terminal, passed));
  return mpk.dump();
}

// EfficientSAM ViT-T prompt_decoder (compile.py COMPONENT_GRAPHS, W8A8):
// iou_predictions [1,1,3] first, then low_res_masks [1,256,256,3], packed in
// one INT8 MLA buffer with channels padded to 16, unpacked, sliced, dequantized.
inline std::string efficientsam_decoder_mpk() {
  nlohmann::json mpk = {{"name", "efficientsam_vit_t_prompt_decoder"},
                        {"sequence", 1},
                        {"model_sdk_version", "3.0.0"},
                        {"input_nodes",
                         {{{"name", "image_embeddings"}, {"type", "buffer"}, {"size", 4194304}},
                          {{"name", "points"}, {"type", "buffer"}, {"size", 192}}}},
                        {"plugins", nlohmann::json::array()}};
  mpk["plugins"].push_back(sam_plugin("quantize_0", 1, "quantization_transform",
                                      {{"channel_params", {{216.0, -2}}},
                                       {"num_bits", 8},
                                       {"rounding", "TONEAREST"},
                                       {"output_data_type", "int8"},
                                       {"input_shapes", {{1, 64, 64, 256}}},
                                       {"output_shapes", {{1, 64, 64, 256}}}},
                                      {{"image_embeddings", 4194304}}, {{"quantize_0", 1048576}}));
  mpk["plugins"].push_back(sam_plugin("quantize_1", 2, "quantization_transform",
                                      {{"channel_params", {{64.0, 0}}},
                                       {"num_bits", 8},
                                       {"rounding", "TONEAREST"},
                                       {"output_data_type", "int8"},
                                       {"input_shapes", {{1, 1, 6, 8}}},
                                       {"output_shapes", {{1, 1, 6, 8}}}},
                                      {{"points", 192}}, {{"quantize_1", 48}}));
  auto mla = sam_plugin("MLA_0", 3, "", {}, {{"quantize_0", 1048576}, {"quantize_1", 48}},
                        {{"MLA_0", 1048592}});
  // A package's MLA program is an .elf; the ModelPack loader requires one.
  mla["resources"]["executable"] = "efficientsam_prompt_decoder_stage1_mla.elf";
  mpk["plugins"].push_back(std::move(mla));
  mpk["plugins"].push_back(sam_plugin(
      "MLA_0_ofm_unpack_transform", 4, "unpack_transform",
      {{"tensor_types", {"int8", "int8"}},
       {"tensor_shapes", {{1, 1, 16}, {1, 256, 256, 16}}},
       {"input_shapes", {{1, 1048592}}},
       {"output_shapes", {{1, 1, 16}, {1, 256, 256, 16}}}},
      {{"MLA_0", 1048592}},
      {{"MLA_0_ofm_unpack_transform_0", 16}, {"MLA_0_ofm_unpack_transform_1", 1048576}}));
  const std::vector<
      std::tuple<std::string, std::vector<std::int64_t>, std::vector<std::int64_t>, double, int>>
      outputs = {{"iou_predictions", {1, 1, 3}, {1, 1, 16}, 120.5, -128},
                 {"low_res_masks", {1, 256, 256, 3}, {1, 256, 256, 16}, 2.25, 64}};
  std::vector<std::pair<std::string, std::int64_t>> terminal, passed;
  int sequence = 5;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& [name, shape, padded, scale, zero_point] = outputs[i];
    const auto slice = "slice_MLA_0/tuple_get_item_" + std::to_string(i) + "_slice_transform";
    mpk["plugins"].push_back(
        sam_plugin(slice, sequence++, "slice_transform",
                   {{"begin", std::vector<std::int64_t>(shape.size(), 0)},
                    {"end", shape},
                    {"input_shape", padded},
                    {"output_shape", shape},
                    {"input_shapes", {padded}},
                    {"output_shapes", {shape}}},
                   {{"MLA_0_ofm_unpack_transform_" + std::to_string(i), sam_numel(padded)}},
                   {{slice, sam_numel(shape)}}));
    const auto dequant = "dequantize_" + std::to_string(40 + i);
    const auto out = dequant + "/prompt_decoder_" + name;
    mpk["plugins"].push_back(sam_plugin(dequant, sequence++, "dequantization_transform",
                                        {{"channel_params", {{scale, zero_point}}},
                                         {"input_data_type", "int8"},
                                         {"input_shapes", {shape}},
                                         {"output_shapes", {shape}}},
                                        {{slice, sam_numel(shape)}},
                                        {{out, sam_numel(shape) * 4}}));
    terminal.push_back({out, sam_numel(shape) * 4});
    passed.push_back({"pass_through_out_" + std::to_string(i), sam_numel(shape) * 4});
  }
  mpk["plugins"].push_back(sam_plugin("PassThrough", sequence, "pass_through",
                                      nlohmann::json::object(), terminal, passed));
  return mpk.dump();
}

} // namespace sam_test
