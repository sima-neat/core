#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>
#include <cstdlib>

RUN_TEST(
    "unit_sima_boxdecode_static_contract_extractor_test", ([] {
      using namespace simaai::neat::pipeline_internal::sima;
      using namespace simaai::neat::pipeline_internal::sima::plugin_contracts;

      auto shape_desc_matches = [](const sima_ev_shape_desc& desc, const std::vector<int>& dims) {
        if (desc.rank != dims.size()) {
          return false;
        }
        for (std::size_t i = 0; i < dims.size(); ++i) {
          if (desc.sizes[i] != dims[i]) {
            return false;
          }
        }
        return true;
      };

      auto make_flags = [](bool quant_needed, bool tess_needed) {
        return ModelManagedRouteFlags{
            .quant_needed = quant_needed,
            .tess_needed = tess_needed,
            .pre_cast_needed = false,
            .quant_contract_required = quant_needed,
            .include_pre_stage = false,
            .boxdecode_selected = true,
        };
      };

      auto require_subset_matches_static_contract = [shape_desc_matches](
                                                        const BoxDecodeContractSubset& subset,
                                                        const BoxDecodeStaticContract& contract,
                                                        const std::string& context) {
        require(subset.decode_type == contract.decode_type,
                context + ": decode_type should match extracted contract");
        require(subset.logical_inputs.size() == contract.tensors.size(),
                context + ": logical input count should match extracted tensors");
        require(subset.input_bindings.size() == contract.tensors.size(),
                context + ": binding count should match extracted tensors");
        require(subset.slice_shapes.size() == contract.tensors.size(),
                context + ": slice shapes count should match extracted tensors");
        require(subset.tess_needed == contract.tess_needed,
                context + ": tess route flag should match extracted contract");
        require(subset.quant_needed == contract.quant_needed,
                context + ": quant route flag should match extracted contract");
        if (contract.decode_type_option == simaai::neat::BoxDecodeTypeOption::Auto) {
          require(!subset.decode_type_option.has_value(),
                  context + ": decode type option should stay unset when the extracted contract "
                            "leaves it auto");
        } else {
          require(subset.decode_type_option == contract.decode_type_option,
                  context + ": decode type option should match extracted contract");
        }
        require(subset.score_activation == contract.score_activation,
                context + ": score activation should match extracted contract");
        for (std::size_t i = 0; i < contract.tensors.size(); ++i) {
          const auto& tensor = contract.tensors[i];
          const auto& logical = subset.logical_inputs[i];
          const auto& binding = subset.input_bindings[i];
          const std::string expected_logical_name =
              !tensor.logical_name.empty()
                  ? tensor.logical_name
                  : (i < contract.tensor_names.size()
                         ? contract.tensor_names[i]
                         : std::string("input_tensor_") + std::to_string(i));
          const std::string expected_backend_name =
              !tensor.backend_name.empty() ? tensor.backend_name : expected_logical_name;
          std::vector<std::int64_t> expected_shape;
          for (const auto dim : tensor.input_shape) {
            expected_shape.push_back(static_cast<std::int64_t>(dim));
          }
          require(logical.shape == expected_shape,
                  context + ": logical input shape should preserve extracted geometry");
          require(logical.dtype ==
                      (!tensor.data_type.empty() ? tensor.data_type : contract.input_dtype),
                  context + ": logical input dtype should preserve extracted dtype");
          require(logical.logical_name == expected_logical_name,
                  context + ": logical input name should preserve extracted name");
          require(logical.backend_name == expected_backend_name,
                  context + ": backend name should preserve extracted name");
          require(logical.segment_name == tensor.source_segment_name,
                  context + ": logical segment should preserve source segment");
          require(logical.byte_offset == tensor.source_byte_offset,
                  context + ": logical byte offset should preserve source offset");
          require(logical.size_bytes == tensor.source_size_bytes,
                  context + ": logical size bytes should preserve source size");
          require(binding.cm_input_name == logical.backend_name,
                  context + ": binding input name should match logical backend name");
          require(binding.source_segment_name ==
                      (!contract.physical_inputs.empty() && i < contract.physical_inputs.size() &&
                               !contract.physical_inputs[i].name.empty()
                           ? contract.physical_inputs[i].name
                           : tensor.source_segment_name),
                  context + ": binding source segment should preserve runtime segment");
          require(binding.src_logical_output_index == tensor.source_logical_output_index,
                  context + ": binding should preserve logical output index");
          require(binding.src_output_slot == tensor.source_output_slot,
                  context + ": binding should preserve output slot");
          require(binding.src_physical_output_index ==
                      (i < contract.physical_inputs.size() &&
                               contract.physical_inputs[i].physical_index >= 0
                           ? contract.physical_inputs[i].physical_index
                           : tensor.source_physical_index),
                  context + ": binding should preserve physical output index");
          require(binding.src_physical_byte_offset == (i < contract.physical_inputs.size()
                                                           ? contract.physical_inputs[i].byte_offset
                                                           : tensor.source_byte_offset),
                  context + ": binding should preserve physical byte offset");
          require(binding.src_physical_size_bytes == (i < contract.physical_inputs.size()
                                                          ? contract.physical_inputs[i].size_bytes
                                                          : 0U),
                  context + ": binding should preserve physical size bytes");
          require(shape_desc_matches(subset.slice_shapes[i], tensor.slice_shape),
                  context + ": slice shape should preserve extracted geometry");
          if (contract.quant_needed) {
            require(logical.quant.has_value(),
                    context + ": quantized route should preserve per-input quant");
            require(logical.quant->scales == std::vector<double>{contract.dq_scale[i]},
                    context + ": quantized route should preserve dq_scale");
            require(logical.quant->zero_points == std::vector<std::int64_t>{contract.dq_zp[i]},
                    context + ": quantized route should preserve dq_zp");
          }
        }
      };

      const auto& boxdecode_decl = plugin_contract_family_declaration("boxdecode");
      require(boxdecode_decl.family == "boxdecode",
              "boxdecode family declaration should be registered");
      require(boxdecode_decl.required_fields.size() == 3U,
              "boxdecode family declaration should expose the typed required field set");
      require(std::string(plugin_contract_field_key_name(boxdecode_decl.required_fields[0])) ==
                  "logical_inputs",
              "boxdecode family declaration should require logical_inputs");
      require(std::string(plugin_contract_field_key_name(boxdecode_decl.required_fields[1])) ==
                  "input_bindings",
              "boxdecode family declaration should require input_bindings");
      require(std::string(plugin_contract_field_key_name(boxdecode_decl.required_fields[2])) ==
                  "slice_geometry",
              "boxdecode family declaration should require slice_geometry");
      require(boxdecode_decl.optional_fields.size() == 3U,
              "boxdecode family declaration should expose optional overlay fields");

      bool threw_missing_binding_field = false;
      try {
        BoxDecodeContractSubset invalid_subset;
        invalid_subset.logical_inputs.push_back(LogicalInputStaticSpec{
            .logical_index = 0,
            .backend_input_index = 0,
            .physical_index = 0,
            .shape = {84, 80, 80},
            .dtype = "INT8",
            .layout = "CHW",
            .logical_name = "bbox_0",
            .backend_name = "bbox_0",
            .segment_name = "MLA_0",
        });
        sima_ev_shape_desc invalid_slice_shape{};
        invalid_slice_shape.rank = 3;
        invalid_slice_shape.sizes[0] = 80;
        invalid_slice_shape.sizes[1] = 80;
        invalid_slice_shape.sizes[2] = 84;
        invalid_subset.slice_shapes.push_back(invalid_slice_shape);
        invalid_subset.quant_needed = true;
        (void)stagesemantics::build_boxdecode_compiled_contract_from_subset(invalid_subset);
      } catch (const std::exception& e) {
        threw_missing_binding_field = true;
        require_contains(std::string(e.what()), "missing required field 'input_bindings'",
                         "boxdecode subset validation should name the missing binding field");
      }
      require(threw_missing_binding_field,
              "boxdecode compiled contract builder should validate required declared fields");

      ModelBoxdecodeSemantics semantics;
      semantics.tess_needed = true;
      semantics.quant_needed = false;
      semantics.quant_contract_required = false;
      const auto from_semantics = model_route_flags_from_boxdecode_semantics(semantics);
      require(from_semantics.tess_needed, "semantic route flags should preserve tess_needed");
      require(!from_semantics.quant_needed, "semantic route flags should preserve quant_needed");
      require(!from_semantics.quant_contract_required,
              "semantic route flags should preserve quant_contract_required");
      require(from_semantics.boxdecode_selected,
              "semantic route flags should mark boxdecode_selected");

      BoxDecodeStaticContract route_contract;
      route_contract.tess_needed = false;
      route_contract.quant_needed = true;
      const auto from_contract = model_route_flags_from_boxdecode_contract(route_contract);
      require(!from_contract.tess_needed, "contract route flags should preserve tess_needed");
      require(from_contract.quant_needed, "contract route flags should preserve quant_needed");
      require(from_contract.quant_contract_required,
              "contract route flags should require quant contract when quantized");
      require(from_contract.boxdecode_selected,
              "contract route flags should mark boxdecode_selected");

      MpkContract mpk;
      MpkPluginIoContract mla;
      mla.name = "MLA_0";
      mla.processor = "MLA";
      mla.kernel = "mla";
      mla.canonical_output_dtype = "INT8";
      mla.quant = MpkQuantContract{{0.25, 0.125}, {4, 5}, -1};
      mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "MLA_0",
          .dtype = "INT8",
          .mpk_shape = {1, 921600},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 921600,
      });
      mpk.plugins.push_back(std::move(mla));

      MpkPluginIoContract unpack;
      unpack.name = "MLA_0_ofm_unpack";
      unpack.kernel = "ofm_unpack";
      unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "bbox_0",
          .dtype = "INT8",
          .mpk_shape = {64, 80, 80},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 409600,
          .logical_shape = {64, 80, 80},
      });
      unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 1,
          .name = "class_logit_0",
          .dtype = "INT8",
          .mpk_shape = {80, 80, 80},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 512000,
          .logical_shape = {80, 80, 80},
      });
      mpk.plugins.push_back(std::move(unpack));

      mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack",
          .tensor_name = "MLA_0",
      });

      std::string error;
      const auto extracted =
          build_boxdecode_static_contract_from_mpk(mpk, make_flags(true, false), &error);
      require(extracted.has_value(),
              "mpk boxdecode extraction should preserve per-head source facts: " + error);
      const auto extracted_subset =
          extract_boxdecode_contract_subset_from_mpk(mpk, make_flags(true, false), nullptr, &error);
      require(extracted_subset.has_value(),
              "mpk boxdecode subset extraction should preserve per-head source facts: " + error);
      require_subset_matches_static_contract(*extracted_subset, *extracted,
                                             "simple model-managed route");
      require(!extracted->tess_needed,
              "explicit unpacked MLA boundary must compile as non-tess for external boxdecode");
      require(extracted->tensors.size() == 2U,
              "mpk boxdecode extraction should keep both logical tensors");
      require(extracted->physical_inputs.size() == 2U,
              "mpk boxdecode extraction should publish per-head source segments");
      require(extracted->score_activation == BoxDecodeScoreActivation::Sigmoid,
              "simple semantic unpack route should preserve class-logit score activation");
      require(extracted->tensors[0].logical_name == "bbox_0",
              "boxdecode extractor should preserve upstream logical tensor name");
      require(extracted->tensors[0].source_segment_name == "MLA_0",
              "boxdecode extractor should preserve the upstream MLA parent segment for runtime "
              "binding");
      require(extracted->tensors[0].source_logical_output_index == 0,
              "boxdecode extractor should preserve upstream logical output index");
      require(extracted->tensors[0].source_output_slot == 0,
              "boxdecode extractor should preserve upstream output slot");
      require(
          extracted->physical_inputs[0].name == "MLA_0",
          "boxdecode extractor should publish the upstream MLA parent segment on physical inputs");
      require(extracted->physical_inputs[0].size_bytes == extracted->tensors[0].source_size_bytes,
              "boxdecode extractor should preserve physical source size");

      const auto extracted_missing_detess =
          build_boxdecode_static_contract_from_mpk(mpk, make_flags(true, true), &error);
      require(!extracted_missing_detess.has_value(),
              "external boxdecode should hard-fail when route flags advertise tess but no upstream "
              "detess slice facts exist");
      require(error.find("detess slice facts") != std::string::npos,
              "missing tess lineage should explain the upstream detess slice requirement");

      MpkContract decoupled_mpk;
      MpkPluginIoContract decoupled_mla;
      decoupled_mla.name = "MLA_0";
      decoupled_mla.processor = "MLA";
      decoupled_mla.kernel = "mla";
      decoupled_mla.canonical_output_dtype = "INT8";
      decoupled_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 2419200},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 2419200,
      });
      decoupled_mpk.plugins.push_back(std::move(decoupled_mla));

      MpkPluginIoContract decoupled_unpack;
      decoupled_unpack.name = "MLA_0_ofm_unpack";
      decoupled_unpack.kernel = "ofm_unpack";
      for (int i = 0; i < 6; ++i) {
        const bool score = i >= 3;
        const int width = (i % 3 == 0) ? 80 : ((i % 3 == 1) ? 40 : 20);
        const int channels = score ? 80 : 64;
        decoupled_unpack.output_tensors.push_back(MpkTensorContract{
            .tensor_index = i,
            .name = std::string("MLA_0_ofm_unpack_transform_") + std::to_string(i),
            .dtype = "INT8",
            .mpk_shape = {width, width, channels},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(width * width * channels),
            .logical_shape = {width, width, channels},
        });
      }
      decoupled_mpk.plugins.push_back(std::move(decoupled_unpack));
      decoupled_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack",
          .tensor_name = "output_tensor",
      });

      const std::array<int, 6> widths = {80, 40, 20, 80, 40, 20};
      const std::array<int, 6> channels = {64, 64, 64, 80, 80, 80};
      const std::array<double, 6> scales = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
      const std::array<std::int64_t, 6> zps = {11, 12, 13, 14, 15, 16};
      const std::array<const char*, 6> semantic_names = {
          "opaque_regression_head_a", "opaque_regression_head_b", "opaque_regression_head_c",
          "opaque_scores_head_a",     "opaque_scores_head_b",     "opaque_scores_head_c",
      };

      for (std::size_t i = 0; i < semantic_names.size(); ++i) {
        MpkPluginIoContract detess_stage;
        detess_stage.name = "detessellate_" + std::to_string(i);
        detess_stage.kernel = "detessellate";
        detess_stage.slice_shape = {1, 1, widths[i], channels[i]};
        detess_stage.output_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = std::string("detessellate_") + std::to_string(i) + "_out",
            .dtype = "INT8",
            .mpk_shape = {widths[i], widths[i], channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(widths[i] * widths[i] * channels[i]),
            .logical_shape = {widths[i], widths[i], channels[i]},
        });
        decoupled_mpk.plugins.push_back(std::move(detess_stage));
        const std::size_t detess_index = decoupled_mpk.plugins.size() - 1U;
        decoupled_mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = 1U,
            .src_output_index = static_cast<int>(i),
            .dst_plugin_index = detess_index,
            .dst_input_index = 0,
            .src_plugin = "MLA_0_ofm_unpack",
            .dst_plugin = decoupled_mpk.plugins[detess_index].name,
            .tensor_name = std::string("MLA_0_ofm_unpack_transform_") + std::to_string(i),
        });

        MpkPluginIoContract dequant_stage;
        dequant_stage.name = "dequantize_" + std::to_string(i);
        dequant_stage.kernel = "dequantize";
        dequant_stage.quant = MpkQuantContract{{scales[i]}, {zps[i]}, -1};
        dequant_stage.output_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = semantic_names[i],
            .dtype = "FP32",
            .mpk_shape = {widths[i], widths[i], channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 4),
            .logical_shape = {widths[i], widths[i], channels[i]},
        });
        decoupled_mpk.plugins.push_back(std::move(dequant_stage));
        const std::size_t dequant_index = decoupled_mpk.plugins.size() - 1U;
        decoupled_mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = detess_index,
            .src_output_index = 0,
            .dst_plugin_index = dequant_index,
            .dst_input_index = 0,
            .src_plugin = decoupled_mpk.plugins[detess_index].name,
            .dst_plugin = decoupled_mpk.plugins[dequant_index].name,
            .tensor_name = decoupled_mpk.plugins[detess_index].output_tensors.front().name,
        });
      }

      const auto extracted_decoupled =
          build_boxdecode_static_contract_from_mpk(decoupled_mpk, make_flags(true, true), &error);
      require(extracted_decoupled.has_value(),
              "decoupled YOLO route should resolve a typed tensor order without semantic names: " +
                  error);
      const auto extracted_decoupled_subset = extract_boxdecode_contract_subset_from_mpk(
          decoupled_mpk, make_flags(true, true), nullptr, &error);
      require(extracted_decoupled_subset.has_value(),
              "decoupled YOLO subset extraction should resolve a typed tensor order: " + error);
      require_subset_matches_static_contract(*extracted_decoupled_subset, *extracted_decoupled,
                                             "decoupled YOLO route");
      require(extracted_decoupled->decode_type == simaai::neat::BoxDecodeType::YoloV8,
              "decoupled YOLO route should normalize grouped multi-head tensors to yolov8");
      require(extracted_decoupled->tess_needed,
              "external/raw unpack route must preserve downstream detess lineage in the typed "
              "route facts");
      require(extracted_decoupled->tensors.size() == 6U,
              "decoupled route should keep all six logical inputs");
      require(extracted_decoupled->tensors[0].logical_name == "bbox_0" &&
                  extracted_decoupled->tensors[3].logical_name == "class_prob_0",
              "grouped YOLO normalization should synthesize canonical head names");
      require(extracted_decoupled->tensors[0].source_output_slot == 0 &&
                  extracted_decoupled->tensors[1].source_output_slot == 1 &&
                  extracted_decoupled->tensors[2].source_output_slot == 2 &&
                  extracted_decoupled->tensors[3].source_output_slot == 3 &&
                  extracted_decoupled->tensors[4].source_output_slot == 4 &&
                  extracted_decoupled->tensors[5].source_output_slot == 5,
              "decoupled route should preserve upstream source slots in extracted order");
      require(
          extracted_decoupled->dq_scale.size() == 6U && extracted_decoupled->dq_scale[0] == 1.0 &&
              extracted_decoupled->dq_scale[1] == 2.0 && extracted_decoupled->dq_scale[2] == 3.0 &&
              extracted_decoupled->dq_scale[3] == 4.0 && extracted_decoupled->dq_scale[4] == 5.0 &&
              extracted_decoupled->dq_scale[5] == 6.0,
          "decoupled route should preserve per-branch quant scales in extracted order");
      require(extracted_decoupled->dq_zp.size() == 6U && extracted_decoupled->dq_zp[0] == 11 &&
                  extracted_decoupled->dq_zp[1] == 12 && extracted_decoupled->dq_zp[2] == 13 &&
                  extracted_decoupled->dq_zp[3] == 14 && extracted_decoupled->dq_zp[4] == 15 &&
                  extracted_decoupled->dq_zp[5] == 16,
              "decoupled route should preserve per-branch quant zero-points in extracted order");
      require(extracted_decoupled->tensors[0].slice_shape == std::vector<int>({1, 80, 64}) &&
                  extracted_decoupled->tensors[3].slice_shape == std::vector<int>({1, 80, 80}),
              "decoupled route should preserve upstream detess slice geometry in extracted order");

      MpkContract c16_padded_mpk;
      MpkPluginIoContract c16_mla;
      c16_mla.name = "MLA_0";
      c16_mla.processor = "MLA";
      c16_mla.kernel = "mla";
      c16_mla.canonical_output_dtype = "BF16";
      c16_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "MLA_0",
          .dtype = "BF16",
          .mpk_shape = {1, 1612800},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 1612800,
      });
      c16_padded_mpk.plugins.push_back(std::move(c16_mla));

      MpkPluginIoContract c16_unpack;
      c16_unpack.name = "MLA_0_ofm_unpack_transform";
      c16_unpack.kernel = "ofm_unpack";
      const std::array<int, 6> c16_widths = {80, 40, 20, 80, 40, 20};
      const std::array<int, 6> c16_logical_channels = {4, 4, 4, 80, 80, 80};
      const std::array<const char*, 6> c16_names = {
          "bbox_0", "bbox_1", "bbox_2", "class_logit_0", "class_logit_1", "class_logit_2",
      };
      for (std::size_t i = 0; i < c16_names.size(); ++i) {
        c16_unpack.output_tensors.push_back(MpkTensorContract{
            .tensor_index = static_cast<int>(i),
            .name = c16_names[i],
            .segment_name = "MLA_0",
            .dtype = "BF16",
            .mpk_shape = {c16_widths[i], c16_widths[i], c16_logical_channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(c16_widths[i] * c16_widths[i] *
                                                   c16_logical_channels[i] * 2),
            .byte_offset = 0,
            .logical_shape = {c16_widths[i], c16_widths[i], c16_logical_channels[i]},
        });
      }
      c16_padded_mpk.plugins.push_back(std::move(c16_unpack));
      c16_padded_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack_transform",
          .tensor_name = "MLA_0",
      });
      for (std::size_t i = 0; i < c16_names.size(); ++i) {
        MpkPluginIoContract detess_stage;
        detess_stage.name = std::string("detessellate_") + c16_names[i];
        detess_stage.kernel = "detessellation_transform";
        detess_stage.slice_shape = {(i < 3U) ? c16_widths[i] : 1, c16_widths[i],
                                    c16_logical_channels[i]};
        detess_stage.has_align_c16 = true;
        detess_stage.align_c16 = true;
        detess_stage.has_cblock = true;
        detess_stage.cblock = true;
        detess_stage.output_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = std::string(c16_names[i]) + "_detess",
            .dtype = "BF16",
            .mpk_shape = {c16_widths[i], c16_widths[i], c16_logical_channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(c16_widths[i] * c16_widths[i] *
                                                   c16_logical_channels[i] * 2),
            .logical_shape = {c16_widths[i], c16_widths[i], c16_logical_channels[i]},
        });
        c16_padded_mpk.plugins.push_back(std::move(detess_stage));
        const std::size_t detess_index = c16_padded_mpk.plugins.size() - 1U;
        c16_padded_mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = 1U,
            .src_output_index = static_cast<int>(i),
            .dst_plugin_index = detess_index,
            .dst_input_index = 0,
            .src_plugin = "MLA_0_ofm_unpack_transform",
            .dst_plugin = c16_padded_mpk.plugins[detess_index].name,
            .tensor_name = c16_names[i],
        });
      }
      const auto extracted_c16_padded =
          build_boxdecode_static_contract_from_mpk(c16_padded_mpk, make_flags(false, true), &error);
      require(extracted_c16_padded.has_value(),
              "BF16 MLATess C16 padded MLA route should resolve a boxdecode contract: " + error);
      const auto extracted_c16_padded_subset = extract_boxdecode_contract_subset_from_mpk(
          c16_padded_mpk, make_flags(false, true), nullptr, &error);
      require(extracted_c16_padded_subset.has_value(),
              "BF16 MLATess C16 padded MLA subset should preserve padded source facts: " + error);
      require_subset_matches_static_contract(*extracted_c16_padded_subset, *extracted_c16_padded,
                                             "BF16 MLATess C16 padded MLA route");
      (void)stagesemantics::build_boxdecode_compiled_contract_from_subset(
          *extracted_c16_padded_subset);
      require(extracted_c16_padded->tensors.size() == 6U,
              "BF16 MLATess C16 padded route should preserve all YOLO heads");
      require(
          extracted_c16_padded->tensors[0].input_shape == std::vector<int>({80, 80, 16}) &&
              extracted_c16_padded->tensors[0].slice_shape == std::vector<int>({80, 80, 4}),
          "BF16 MLATess C16 padded bbox head should expose physical C16 input and logical slice");
      require(extracted_c16_padded->tensors[1].input_shape == std::vector<int>({40, 40, 16}) &&
                  extracted_c16_padded->tensors[2].input_shape == std::vector<int>({20, 20, 16}),
              "all bbox heads should round physical channels to C16");
      require(extracted_c16_padded->tensors[3].input_shape == std::vector<int>({80, 80, 80}) &&
                  extracted_c16_padded->tensors[3].slice_shape == std::vector<int>({1, 80, 80}),
              "already-aligned score head should preserve its physical channel count");
      std::uint64_t c16_expected_total = 0U;
      for (const auto& tensor : extracted_c16_padded->tensors) {
        c16_expected_total += tensor.source_size_bytes;
      }
      require(c16_expected_total == 1612800U,
              "boxdecode C16 packed view sizes should add up to the BF16 raw MLA buffer size");
      require(extracted_c16_padded->physical_inputs[0].size_bytes == 204800U,
              "first bbox physical binding should use padded 80x80x16 BF16 transport bytes");

      MpkContract probability_quant_mpk = decoupled_mpk;
      for (auto& plugin : probability_quant_mpk.plugins) {
        if (plugin.name == "dequantize_3" || plugin.name == "dequantize_4" ||
            plugin.name == "dequantize_5") {
          plugin.quant = MpkQuantContract{{255.0}, {-128}, -1};
        }
      }
      const auto extracted_probability_quant = build_boxdecode_static_contract_from_mpk(
          probability_quant_mpk, make_flags(true, true), &error);
      require(
          extracted_probability_quant.has_value(),
          "probability-domain quantized YOLO route should still resolve a typed tensor order: " +
              error);
      const auto extracted_probability_quant_subset = extract_boxdecode_contract_subset_from_mpk(
          probability_quant_mpk, make_flags(true, true), nullptr, &error);
      require(extracted_probability_quant_subset.has_value(),
              "probability-domain quantized YOLO subset extraction should preserve typed route "
              "facts: " +
                  error);
      require_subset_matches_static_contract(*extracted_probability_quant_subset,
                                             *extracted_probability_quant,
                                             "probability-domain quantized route");
      require(extracted_probability_quant->score_activation == BoxDecodeScoreActivation::Identity,
              "core extractor should infer probability-domain score activation generically from "
              "quant range semantics");
      require(extracted_probability_quant->decode_type_option ==
                  simaai::neat::BoxDecodeTypeOption::GroupedByRoleProbability,
              "probability-domain quantized YOLO route should infer grouped-by-role-probability");

      MpkContract packed_parent_mpk;
      MpkPluginIoContract packed_parent_mla;
      packed_parent_mla.name = "MLA_0";
      packed_parent_mla.processor = "MLA";
      packed_parent_mla.kernel = "mla";
      packed_parent_mla.canonical_output_dtype = "INT8";
      packed_parent_mla.quant = MpkQuantContract{{0.25, 0.125}, {4, 5}, -1};
      packed_parent_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 2419200},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 2419200,
      });
      packed_parent_mpk.plugins.push_back(std::move(packed_parent_mla));

      MpkPluginIoContract packed_parent_unpack;
      packed_parent_unpack.name = "MLA_0_ofm_unpack";
      packed_parent_unpack.kernel = "ofm_unpack";
      packed_parent_unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "bbox_0",
          .segment_name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {80, 80, 64},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 409600,
          .byte_offset = 1209600,
          .logical_shape = {80, 80, 64},
      });
      packed_parent_unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 1,
          .physical_index = 0,
          .name = "opaque_score_0",
          .segment_name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {80, 80, 80},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 512000,
          .byte_offset = 1747200,
          .logical_shape = {80, 80, 80},
      });
      packed_parent_mpk.plugins.push_back(std::move(packed_parent_unpack));
      packed_parent_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack",
          .tensor_name = "output_tensor",
      });
      for (int i = 0; i < 2; ++i) {
        MpkPluginIoContract detess_stage;
        detess_stage.name = "packed_parent_detess_" + std::to_string(i);
        detess_stage.kernel = "detess";
        detess_stage.slice_shape = {1, 80, i == 0 ? 64 : 80};
        detess_stage.output_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = i == 0 ? "bbox_0_detess" : "opaque_score_0_detess",
            .dtype = "INT8",
            .mpk_shape = {80, 80, i == 0 ? 64 : 80},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(80 * 80 * (i == 0 ? 64 : 80)),
            .logical_shape = {80, 80, i == 0 ? 64 : 80},
        });
        packed_parent_mpk.plugins.push_back(std::move(detess_stage));
        const std::size_t detess_index = packed_parent_mpk.plugins.size() - 1U;
        packed_parent_mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = 1U,
            .src_output_index = i,
            .dst_plugin_index = detess_index,
            .dst_input_index = 0,
            .src_plugin = "MLA_0_ofm_unpack",
            .dst_plugin = packed_parent_mpk.plugins[detess_index].name,
            .tensor_name = i == 0 ? "bbox_0" : "opaque_score_0",
        });
      }

      const auto extracted_packed_parent = build_boxdecode_static_contract_from_mpk(
          packed_parent_mpk, make_flags(true, false), &error);
      require(extracted_packed_parent.has_value(),
              "packed-parent MLA route should preserve both logical and physical source facts: " +
                  error);
      const auto extracted_packed_parent_subset = extract_boxdecode_contract_subset_from_mpk(
          packed_parent_mpk, make_flags(true, false), nullptr, &error);
      require(extracted_packed_parent_subset.has_value(),
              "packed-parent MLA subset extraction should preserve both logical and physical "
              "source facts: " +
                  error);
      require_subset_matches_static_contract(*extracted_packed_parent_subset,
                                             *extracted_packed_parent, "packed-parent MLA route");
      require(extracted_packed_parent->tensors.size() == 2U,
              "packed-parent route should keep both logical inputs");
      require(extracted_packed_parent->tensors[0].source_segment_name == "output_tensor",
              "explicit unpack boundary should still bind through the packed MLA parent segment");
      require(extracted_packed_parent->tensors[0].source_byte_offset == 1209600,
              "packed-parent route should preserve the first logical view byte offset");
      require(extracted_packed_parent->tensors[1].source_byte_offset == 1747200,
              "packed-parent route should preserve the second logical view byte offset");
      require(extracted_packed_parent->physical_inputs.size() == 2U,
              "packed-parent route should publish explicit physical inputs");
      require(extracted_packed_parent->physical_inputs[0].name == "output_tensor",
              "explicit unpack boundary should publish the packed MLA parent as the physical input "
              "name");
      require(extracted_packed_parent->physical_inputs[0].byte_offset == 1209600,
              "packed-parent physical input should preserve the first logical view offset");
      require(extracted_packed_parent->physical_inputs[1].byte_offset == 1747200,
              "packed-parent physical input should preserve the second logical view offset");

      setenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK", "1", 1);
      const auto extracted_packed_parent_bypass = build_boxdecode_static_contract_from_mpk(
          packed_parent_mpk, make_flags(true, true), &error);
      require(extracted_packed_parent_bypass.has_value(),
              "raw-parent bypass should allow external boxdecode to source directly from packed "
              "MLA parent: " +
                  error);
      const auto extracted_packed_parent_bypass_subset = extract_boxdecode_contract_subset_from_mpk(
          packed_parent_mpk, make_flags(true, true), nullptr, &error);
      unsetenv("SIMA_BOXDECODE_BYPASS_MLA_UNPACK");
      require(extracted_packed_parent_bypass_subset.has_value(),
              "raw-parent bypass subset extraction should preserve direct packed-parent facts: " +
                  error);
      require_subset_matches_static_contract(*extracted_packed_parent_bypass_subset,
                                             *extracted_packed_parent_bypass,
                                             "raw-parent bypass route");
      require(extracted_packed_parent_bypass->tess_needed,
              "raw-parent bypass should preserve tess decode semantics");
      require(extracted_packed_parent_bypass->tensors[0].source_segment_name == "output_tensor",
              "raw-parent bypass should bind logical tensors to the packed MLA parent");
      require(extracted_packed_parent_bypass->tensors[0].source_byte_offset == 1209600,
              "raw-parent bypass should preserve the first packed MLA head offset");
      require(extracted_packed_parent_bypass->tensors[1].source_byte_offset == 1747200,
              "raw-parent bypass should preserve the second packed MLA head offset");
      require(extracted_packed_parent_bypass->physical_inputs[0].name == "output_tensor",
              "raw-parent bypass should publish the packed MLA parent as the TensorBuffer source");
      require(extracted_packed_parent_bypass->physical_inputs[0].byte_offset == 1209600,
              "raw-parent bypass physical input should preserve the first packed head offset");
      require(extracted_packed_parent_bypass->physical_inputs[1].byte_offset == 1747200,
              "raw-parent bypass physical input should preserve the second packed head offset");
    }));
