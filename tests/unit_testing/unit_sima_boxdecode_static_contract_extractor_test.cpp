#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>

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

      auto add_packed_detess_facts = [](MpkPluginIoContract& detess, const std::string& input_name,
                                        int height, int width, int channels,
                                        const std::string& dtype = "INT8",
                                        std::uint64_t source_size_bytes = 0U) {
        const int elem_bytes = (dtype == "BF16" || dtype == "BFLOAT16" || dtype == "bfloat16" ||
                                dtype == "EVXX_BFLOAT16")
                                   ? 2
                                   : 1;
        const auto fallback_source_size =
            static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(channels) * static_cast<std::uint64_t>(elem_bytes);
        const auto source_bytes = source_size_bytes > 0U ? source_size_bytes : fallback_source_size;
        detess.frame_shape = {1, height, width, channels};
        detess.frame_type = dtype;
        detess.has_cblock = true;
        detess.cblock = true;
        detess.has_align_c16 = true;
        detess.align_c16 = true;
        detess.input_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = input_name,
            .dtype = dtype,
            .mpk_shape = {1, static_cast<std::int64_t>(source_bytes)},
            .shape_semantics = MpkShapeSemantics::PackedExtent,
            .size_bytes = static_cast<std::size_t>(source_bytes),
            .logical_shape = {height, width, channels},
        });
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
        require(subset.tensor_storage_kind.size() == contract.tensors.size(),
                context + ": storage-kind count should match extracted tensors");
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
          require(subset.tensor_storage_kind[i] == static_cast<int>(tensor.source_storage_kind),
                  context + ": source storage kind should preserve extracted MPK fact");
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

      ModelManagedRouteFlags stale_planner_flags;
      stale_planner_flags.tess_needed = false;
      stale_planner_flags.quant_needed = true;
      stale_planner_flags.quant_contract_required = true;
      stale_planner_flags.pre_cast_needed = true;
      stale_planner_flags.include_pre_stage = true;
      ModelManagedRouteFlags exact_bf16_flags;
      exact_bf16_flags.tess_needed = true;
      exact_bf16_flags.quant_needed = false;
      exact_bf16_flags.quant_contract_required = false;
      exact_bf16_flags.boxdecode_selected = true;
      const auto reconciled_bf16 =
          reconcile_exact_boxdecode_route_flags(stale_planner_flags, exact_bf16_flags);
      require(reconciled_bf16.tess_needed && !reconciled_bf16.quant_needed &&
                  !reconciled_bf16.quant_contract_required &&
                  reconciled_bf16.boxdecode_selected,
              "exact BF16 terminal facts must override stale planner quant flags");
      require(reconciled_bf16.pre_cast_needed && reconciled_bf16.include_pre_stage,
              "planner-owned pre-route fields must survive exact terminal reconciliation");

      ModelManagedRouteFlags permissive_planner_flags;
      permissive_planner_flags.tess_needed = false;
      permissive_planner_flags.quant_needed = false;
      permissive_planner_flags.quant_contract_required = false;
      ModelManagedRouteFlags exact_int8_flags;
      exact_int8_flags.tess_needed = true;
      exact_int8_flags.quant_needed = true;
      exact_int8_flags.quant_contract_required = true;
      exact_int8_flags.boxdecode_selected = true;
      const auto reconciled_int8 =
          reconcile_exact_boxdecode_route_flags(permissive_planner_flags, exact_int8_flags);
      require(reconciled_int8.tess_needed && reconciled_int8.quant_needed &&
                  reconciled_int8.quant_contract_required &&
                  reconciled_int8.boxdecode_selected,
              "exact INT8 terminal facts must override permissive planner flags");
      const auto reconciled_sync =
          reconcile_exact_boxdecode_route_flags(stale_planner_flags, exact_bf16_flags);
      require(reconciled_sync.quant_needed == reconciled_bf16.quant_needed &&
                  reconciled_sync.tess_needed == reconciled_bf16.tess_needed &&
                  reconciled_sync.quant_contract_required ==
                      reconciled_bf16.quant_contract_required &&
                  reconciled_sync.pre_cast_needed == reconciled_bf16.pre_cast_needed &&
                  reconciled_sync.include_pre_stage == reconciled_bf16.include_pre_stage &&
                  reconciled_sync.boxdecode_selected == reconciled_bf16.boxdecode_selected,
              "sync and async BoxDecode issuance must reconcile identical exact route facts");

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
            .shape_semantics = MpkShapeSemantics::PackedExtent,
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
        detess_stage.processor = "EV74";
        detess_stage.kernel = "detessellate";
        detess_stage.slice_shape = {1, 1, widths[i], channels[i]};
        add_packed_detess_facts(detess_stage,
                                std::string("MLA_0_ofm_unpack_transform_") + std::to_string(i),
                                widths[i], widths[i], channels[i]);
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
        dequant_stage.processor = "EV74";
        dequant_stage.kernel = "dequantize";
        dequant_stage.quant = MpkQuantContract{{scales[i]}, {zps[i]}, -1};
        dequant_stage.input_tensors.push_back(MpkTensorContract{
            .tensor_index = 0,
            .name = decoupled_mpk.plugins[detess_index].output_tensors.front().name,
            .dtype = "INT8",
            .mpk_shape = {widths[i], widths[i], channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(widths[i] * widths[i] * channels[i]),
            .logical_shape = {widths[i], widths[i], channels[i]},
        });
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
      require(extracted_decoupled->decode_type == simaai::neat::BoxDecodeType::Unspecified,
              "decoupled YOLO route should not infer a BoxDecode family without an explicit "
              "user/MPK decode_type");
      require(extracted_decoupled->tess_needed,
              "external/raw unpack route must preserve downstream detess lineage in the typed "
              "route facts");
      require(extracted_decoupled->tensors.size() == 6U,
              "decoupled route should keep all six logical inputs");
      require(extracted_decoupled->tensors[0].logical_name != "bbox_0" &&
                  extracted_decoupled->tensors[3].logical_name != "class_prob_0",
              "MPK extraction should not synthesize YOLO head names without an explicit "
              "user/MPK decode_type");
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

      // Exact YOLOv8s compiler contract (artifact
      // 029ddb603ceb96ac94d8e8f3647f333334e6c70566cc8e5e55efafb6f38c42d2): one
      // packed INT8 MLA parent carries three regression heads followed by three
      // probability heads. Core retargets the post-MLA Detess/Dequant nodes into
      // BoxDecode, so their authored tile geometry and qparams must survive even
      // though the route no longer renders separate stages.
      MpkContract exact_yolov8_mpk = decoupled_mpk;
      constexpr std::array<std::int64_t, 6> exact_offsets = {
          0, 409600, 512000, 537600, 1049600, 1177600,
      };
      constexpr std::array<std::size_t, 6> exact_spans = {
          409600U, 102400U, 25600U, 512000U, 128000U, 32000U,
      };
      constexpr std::array<double, 6> exact_scales = {
          13.470379316726476, 14.793570210081732, 14.266452350921979,
          651.989127865681,  330.2695539904479,   282.32459069196017,
      };
      constexpr std::array<std::int64_t, 6> exact_zps = {
          -66, -58, -45, -128, -128, -128,
      };
      constexpr std::array<const char*, 6> exact_names = {
          "bbox_0", "bbox_1", "bbox_2", "class_prob_0", "class_prob_1", "class_prob_2",
      };
      auto& exact_parent = exact_yolov8_mpk.plugins[0].output_tensors[0];
      exact_parent.mpk_shape = {1, 1209600};
      exact_parent.size_bytes = 1209600U;
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        auto& unpack_output = exact_yolov8_mpk.plugins[1].output_tensors[i];
        unpack_output.physical_index = 0;
        unpack_output.source_physical_index = 0;
        unpack_output.segment_name = "output_tensor";
        unpack_output.byte_offset = exact_offsets[i];
        unpack_output.source_byte_offset = exact_offsets[i];
        unpack_output.size_bytes = exact_spans[i];

        auto& detess_stage = exact_yolov8_mpk.plugins[2U + (2U * i)];
        require(detess_stage.has_cblock && detess_stage.cblock && detess_stage.has_align_c16 &&
                    detess_stage.align_c16,
                "exact YOLO fixture must carry compiler-authored CBlock/C16 storage facts");
        auto& dequant_stage = exact_yolov8_mpk.plugins[3U + (2U * i)];
        dequant_stage.quant = MpkQuantContract{{exact_scales[i]}, {exact_zps[i]}, -1};
        dequant_stage.output_tensors.front().name = exact_names[i];
      }

      const auto extracted_exact_yolov8 = build_boxdecode_static_contract_from_mpk(
          exact_yolov8_mpk, make_flags(true, false), &error);
      require(extracted_exact_yolov8.has_value(),
              "exact packed YOLO route should retarget compiler Detess/Dequant facts into "
              "BoxDecode without an environment bypass: " +
                  error);
      require(extracted_exact_yolov8->tess_needed && extracted_exact_yolov8->quant_needed,
              "exact packed YOLO route must select BoxDecode internal detess and dequant");
      require(extracted_exact_yolov8->score_activation == BoxDecodeScoreActivation::Identity,
              "exact class_prob heads must preserve probability-domain activation");
      require(extracted_exact_yolov8->tensors.size() == exact_names.size(),
              "exact packed YOLO route must preserve all six ordered heads");
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        const auto& tensor = extracted_exact_yolov8->tensors[i];
        require(tensor.logical_name == exact_names[i] && tensor.backend_name == exact_names[i],
                "exact packed YOLO route must preserve compiler-authored head order and roles");
        require(tensor.input_shape ==
                    std::vector<int>({widths[i], widths[i], channels[i]}) &&
                    tensor.slice_shape == std::vector<int>({1, widths[i], channels[i]}),
                "exact packed YOLO route must preserve full frame and tile geometry");
        require(tensor.data_type == "INT8" &&
                    tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock,
                "exact packed YOLO route must preserve INT8 CBlock carrier storage");
        require(tensor.source_segment_name == "output_tensor" &&
                    tensor.source_physical_index == 0 &&
                    tensor.source_byte_offset == exact_offsets[i] &&
                    tensor.source_size_bytes == exact_spans[i],
                "exact packed YOLO route must preserve the shared-parent carrier view");
        require(extracted_exact_yolov8->dq_scale[i] == exact_scales[i] &&
                    extracted_exact_yolov8->dq_zp[i] == exact_zps[i],
                "exact packed YOLO route must preserve per-head dequantization parameters");
      }

      const auto finalized_exact_yolov8 =
          stagesemantics::finalize_boxdecode_static_contract(
              *extracted_exact_yolov8, simaai::neat::BoxDecodeType::YoloV8, std::nullopt,
              make_flags(true, false),
              simaai::neat::BoxDecodeTypeOption::GroupedByRoleProbability, 0.25, 0.55, 100, 80,
              {"orig_width", "orig_height"});
      const auto compiled_exact_yolov8 =
          stagesemantics::build_boxdecode_compiled_contract(finalized_exact_yolov8);
      const auto& exact_payload = compiled_exact_yolov8.payload;
      require(exact_payload.decode_type == simaai::neat::BoxDecodeType::YoloV8 &&
                  exact_payload.decode_type_option ==
                      simaai::neat::BoxDecodeTypeOption::GroupedByRoleProbability &&
                  exact_payload.score_activation == BoxDecodeScoreActivation::Identity,
              "BoxDecode backend payload must receive the exact YOLO probability semantics");
      require(exact_payload.tess_needed && exact_payload.quant_needed &&
                  exact_payload.model_owned_flags && exact_payload.quant_contract_required &&
                  exact_payload.input_dtype == "INT8" && exact_payload.num_classes == 80,
              "BoxDecode backend payload must enable internal INT8 detess/dequant");
      require(exact_payload.slice_shapes.size() == exact_names.size() &&
                  exact_payload.tensor_storage_kind.size() == exact_names.size(),
              "BoxDecode backend payload must carry one tile/storage descriptor per head");
      require(compiled_exact_yolov8.runtime_contract.logical_inputs.size() == exact_names.size() &&
                  compiled_exact_yolov8.runtime_contract.input_bindings.size() ==
                      exact_names.size(),
              "BoxDecode runtime ABI must carry one logical input and binding per exact head");
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        require(shape_desc_matches(exact_payload.slice_shapes[i],
                                   {1, widths[i], channels[i]}) &&
                    exact_payload.tensor_storage_kind[i] ==
                        static_cast<int>(BoxDecodeSourceStorageKind::PackedCBlock),
                "BoxDecode backend payload must preserve exact CBlock tile geometry");
        const auto& logical = compiled_exact_yolov8.runtime_contract.logical_inputs[i];
        const auto& binding = compiled_exact_yolov8.runtime_contract.input_bindings[i];
        require(logical.shape == std::vector<std::int64_t>({widths[i], widths[i], channels[i]}) &&
                    logical.dtype == "INT8" && logical.logical_name == exact_names[i] &&
                    logical.segment_name == "output_tensor" &&
                    logical.byte_offset == exact_offsets[i] && logical.size_bytes == exact_spans[i],
                "BoxDecode runtime ABI must preserve exact ordered logical carrier views");
        require(logical.quant.has_value() &&
                    logical.quant->scales == std::vector<double>{exact_scales[i]} &&
                    logical.quant->zero_points == std::vector<std::int64_t>{exact_zps[i]},
                "BoxDecode runtime ABI must preserve exact per-head dequantization parameters");
        require(binding.source_segment_name == "output_tensor" &&
                    binding.src_physical_output_index == 0 &&
                    binding.src_physical_byte_offset == exact_offsets[i] &&
                    binding.src_physical_size_bytes == exact_spans[i],
                "BoxDecode runtime ABI must preserve exact same-parent offsets and spans");
      }

      const auto exact_int8_route_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(exact_yolov8_mpk, nullptr, &error);
      require(exact_int8_route_flags.has_value() && exact_int8_route_flags->tess_needed &&
                  exact_int8_route_flags->quant_needed &&
                  exact_int8_route_flags->quant_contract_required,
              "exact packed INT8 Detess/Dequant lineage must select internal detess/dequant: " +
                  error);
      const auto exact_int8_after_permissive_planner = reconcile_exact_boxdecode_route_flags(
          permissive_planner_flags, *exact_int8_route_flags);
      const auto extracted_exact_int8_after_planner = build_boxdecode_static_contract_from_mpk(
          exact_yolov8_mpk, exact_int8_after_permissive_planner, &error);
      require(extracted_exact_int8_after_planner.has_value() &&
                  extracted_exact_int8_after_planner->dq_scale.size() == exact_names.size() &&
                  extracted_exact_int8_after_planner->dq_zp.size() == exact_names.size(),
              "exact INT8 route must retain all six qparams despite permissive planner flags: " +
                  error);

      // Some packed BF16 packages retain the historical unpack tensor_types=INT8 token even
      // though the selected carriers are unambiguously BF16: every typed Detess owns a BF16
      // frame, its source span is two bytes per element, and the only following value transform
      // is BF16->FP32 Cast. The selected carrier lineage, not that advisory unpack token, owns the
      // BoxDecode quant decision.
      auto exact_packed_bf16_mpk = exact_yolov8_mpk;
      exact_packed_bf16_mpk.plugins[0].canonical_output_dtype = "BF16";
      exact_packed_bf16_mpk.plugins[0].output_tensors[0].dtype = "BF16";
      exact_packed_bf16_mpk.plugins[0].output_tensors[0].mpk_shape = {1, 2419200};
      exact_packed_bf16_mpk.plugins[0].output_tensors[0].size_bytes = 2419200U;
      // Stale compiler-derived MLA qparams are not carrier authority. The exact selected Detess
      // lineages below prove BF16 source views, so this metadata must be ignored rather than
      // coercing the route back to INT8.
      exact_packed_bf16_mpk.plugins[0].quant =
          MpkQuantContract{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {0, 0, 0, 0, 0, 0}, -1};
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        auto& unpack_output = exact_packed_bf16_mpk.plugins[1].output_tensors[i];
        // Deliberately retain INT8 here: it is the contradictory advisory token in the exact
        // package that motivated this regression.
        unpack_output.dtype = "INT8";
        unpack_output.byte_offset = exact_offsets[i] * 2;
        unpack_output.source_byte_offset = exact_offsets[i] * 2;
        unpack_output.size_bytes = exact_spans[i] * 2U;

        auto& detess_stage = exact_packed_bf16_mpk.plugins[2U + (2U * i)];
        detess_stage.frame_type = "BF16";
        detess_stage.input_tensors.front().dtype = "BF16";
        detess_stage.input_tensors.front().size_bytes = exact_spans[i] * 2U;
        detess_stage.output_tensors.front().dtype = "BF16";
        detess_stage.output_tensors.front().size_bytes = exact_spans[i] * 2U;

        auto& cast_stage = exact_packed_bf16_mpk.plugins[3U + (2U * i)];
        cast_stage.name = "cast_" + std::to_string(i + 2U);
        cast_stage.processor = "EV74";
        cast_stage.kernel = "cast_transform";
        cast_stage.quant.reset();
        cast_stage.input_tensors.front().dtype = "BF16";
        cast_stage.input_tensors.front().size_bytes = exact_spans[i] * 2U;
        cast_stage.output_tensors.front().dtype = "FP32";
        cast_stage.output_tensors.front().size_bytes = exact_spans[i] * 4U;
        cast_stage.output_tensors.front().name = exact_names[i];
      }

      const auto exact_bf16_route_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          exact_packed_bf16_mpk, nullptr, &error);
      require(exact_bf16_route_flags.has_value() && exact_bf16_route_flags->tess_needed &&
                  !exact_bf16_route_flags->quant_needed &&
                  !exact_bf16_route_flags->quant_contract_required,
              "exact packed BF16 Detess/Cast lineage must ignore advisory INT8 unpack tokens: " +
                  error);
      const auto exact_bf16_after_stale_planner = reconcile_exact_boxdecode_route_flags(
          stale_planner_flags, *exact_bf16_route_flags);
      require(exact_bf16_after_stale_planner.tess_needed &&
                  !exact_bf16_after_stale_planner.quant_needed &&
                  !exact_bf16_after_stale_planner.quant_contract_required &&
                  exact_bf16_after_stale_planner.pre_cast_needed &&
                  exact_bf16_after_stale_planner.include_pre_stage,
              "exact BF16 carrier facts must own post flags while preserving planner pre fields");
      const auto extracted_exact_packed_bf16 = build_boxdecode_static_contract_from_mpk(
          exact_packed_bf16_mpk, exact_bf16_after_stale_planner, &error);
      require(extracted_exact_packed_bf16.has_value(),
              "exact packed BF16 Detess/Cast lineage should build without quant facts: " + error);
      require(extracted_exact_packed_bf16->tess_needed &&
                  !extracted_exact_packed_bf16->quant_needed &&
                  extracted_exact_packed_bf16->input_dtype == "BF16",
              "exact packed BF16 route must retain typed detess and disable dequant");
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        const auto& tensor = extracted_exact_packed_bf16->tensors[i];
        require(tensor.data_type == "BF16" &&
                    tensor.source_storage_kind == BoxDecodeSourceStorageKind::PackedCBlock &&
                    tensor.source_byte_offset == exact_offsets[i] * 2 &&
                    tensor.source_size_bytes == exact_spans[i] * 2U,
                "exact packed BF16 route must preserve each ordered BF16 carrier view");
      }

      auto bf16_with_dequant_mpk = exact_packed_bf16_mpk;
      auto& contradictory_bf16_dequant = bf16_with_dequant_mpk.plugins[3];
      contradictory_bf16_dequant.name = "contradictory_bf16_dequant";
      contradictory_bf16_dequant.processor = "EV74";
      contradictory_bf16_dequant.kernel = "dequantization_transform";
      contradictory_bf16_dequant.quant = MpkQuantContract{{0.25}, {0}, -1};
      const auto bf16_with_dequant_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          bf16_with_dequant_mpk, nullptr, &error);
      require(!bf16_with_dequant_flags.has_value(),
              "typed Dequant qparams on an exact BF16 carrier must fail closed");
      require_contains(error, "BF16/float carrier lineage contradicts typed Dequant qparams",
                       "BF16/Dequant conflict should identify carrier-domain authority");

      // A typed INT8 Detess branch can be decoded directly without retaining a materialized
      // Dequant stage. In that form the canonical MLA qparam vector is accepted only as one exact
      // scalar contract per selected logical source slot.
      auto exact_int8_mla_fallback_mpk = exact_yolov8_mpk;
      exact_int8_mla_fallback_mpk.plugins[0].quant = MpkQuantContract{
          std::vector<double>(exact_scales.begin(), exact_scales.end()),
          std::vector<std::int64_t>(exact_zps.begin(), exact_zps.end()), -1};
      for (std::size_t i = 0; i < exact_names.size(); ++i) {
        auto& pass = exact_int8_mla_fallback_mpk.plugins[3U + (2U * i)];
        pass.name = "pass_through_" + std::to_string(i);
        pass.processor = "EV74";
        pass.kernel = "pass_through";
        pass.quant.reset();
        pass.output_tensors.front().dtype = "INT8";
        pass.output_tensors.front().size_bytes = pass.input_tensors.front().size_bytes;
        pass.output_tensors.front().name = exact_names[i];
      }
      const auto exact_int8_mla_fallback_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(
              exact_int8_mla_fallback_mpk, nullptr, &error);
      require(exact_int8_mla_fallback_flags.has_value() &&
                  exact_int8_mla_fallback_flags->tess_needed &&
                  exact_int8_mla_fallback_flags->quant_needed,
              "typed INT8 Detess without Dequant should accept exact selected-slot MLA qparams: " +
                  error);
      const auto extracted_int8_mla_fallback = build_boxdecode_static_contract_from_mpk(
          exact_int8_mla_fallback_mpk, *exact_int8_mla_fallback_flags, &error);
      require(extracted_int8_mla_fallback.has_value(),
              "exact selected-slot MLA qparam fallback should build: " + error);
      require(extracted_int8_mla_fallback->dq_scale ==
                      std::vector<double>(exact_scales.begin(), exact_scales.end()) &&
                  extracted_int8_mla_fallback->dq_zp ==
                      std::vector<std::int64_t>(exact_zps.begin(), exact_zps.end()),
              "MLA qparam fallback must preserve exact selected-slot ordering");

      auto extra_mla_qparam_mpk = exact_int8_mla_fallback_mpk;
      extra_mla_qparam_mpk.plugins[0].quant->scales.push_back(7.0);
      extra_mla_qparam_mpk.plugins[0].quant->zero_points.push_back(0);
      const auto extra_mla_qparam_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          extra_mla_qparam_mpk, nullptr, &error);
      require(!extra_mla_qparam_flags.has_value(),
              "oversized MLA qparam vectors must not pass selected-slot validation");
      require_contains(error, "exact per-logical-source-slot MLA qparam arity",
                       "oversized MLA qparam rejection should identify exact arity");

      auto duplicate_mla_slot_mpk = exact_int8_mla_fallback_mpk;
      duplicate_mla_slot_mpk.plugins[1].output_tensors[1].tensor_index = 0;
      const auto duplicate_mla_slot_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          duplicate_mla_slot_mpk, nullptr, &error);
      require(!duplicate_mla_slot_flags.has_value(),
              "duplicate selected logical source slots must fail closed");
      require_contains(error, "bijection over logical source slots",
                       "duplicate selected-slot rejection should identify the bijection");

      // An unrelated post-MLA Dequant stage that is not reachable from a selected head cannot
      // change the BF16 route domain or qparam requirement.
      auto bf16_with_off_route_dequant_mpk = exact_packed_bf16_mpk;
      MpkPluginIoContract off_route_dequant;
      off_route_dequant.name = "off_route_dequant";
      off_route_dequant.sequence = 500;
      off_route_dequant.processor = "EV74";
      off_route_dequant.kernel = "dequantization_transform";
      off_route_dequant.quant = MpkQuantContract{{0.5}, {0}, -1};
      off_route_dequant.input_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "unrelated_int8_input",
          .dtype = "INT8",
          .mpk_shape = {1, 16},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 16,
      });
      off_route_dequant.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "unrelated_fp32_output",
          .dtype = "FP32",
          .mpk_shape = {1, 16},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 64,
      });
      bf16_with_off_route_dequant_mpk.plugins.push_back(std::move(off_route_dequant));
      const auto bf16_with_off_route_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          bf16_with_off_route_dequant_mpk, nullptr, &error);
      require(bf16_with_off_route_flags.has_value() &&
                  !bf16_with_off_route_flags->quant_needed,
              "off-route Dequant metadata must not change an exact BF16 selected carrier: " +
                  error);

      auto bf16_leaf_fanout_mpk = exact_packed_bf16_mpk;
      for (int branch = 0; branch < 2; ++branch) {
        MpkPluginIoContract pass;
        pass.name = "leaf_sibling_pass_" + std::to_string(branch);
        pass.sequence = 600 + branch;
        pass.processor = "EV74";
        pass.kernel = "pass_through";
        pass.input_tensors.push_back(bf16_leaf_fanout_mpk.plugins[3].output_tensors[0]);
        pass.output_tensors.push_back(pass.input_tensors.front());
        pass.output_tensors.front().name = "leaf_sibling_" + std::to_string(branch);
        bf16_leaf_fanout_mpk.plugins.push_back(std::move(pass));
        const std::size_t pass_index = bf16_leaf_fanout_mpk.plugins.size() - 1U;
        bf16_leaf_fanout_mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = 3U,
            .src_output_index = 0,
            .dst_plugin_index = pass_index,
            .dst_input_index = 0,
            .src_plugin = bf16_leaf_fanout_mpk.plugins[3].name,
            .dst_plugin = bf16_leaf_fanout_mpk.plugins[pass_index].name,
            .tensor_name = bf16_leaf_fanout_mpk.plugins[3].output_tensors[0].name,
        });
      }
      const auto bf16_leaf_fanout_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          bf16_leaf_fanout_mpk, nullptr, &error);
      require(!bf16_leaf_fanout_flags.has_value(),
              "an external publication leaf with typed sibling consumers must fail closed");
      require_contains(error, "ambiguous post-MLA fanout",
                       "external leaf sibling rejection should identify fanout");

      auto bf16_cycle_mpk = exact_packed_bf16_mpk;
      bf16_cycle_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 3U,
          .src_output_index = 0,
          .dst_plugin_index = 2U,
          .dst_input_index = 0,
          .src_plugin = bf16_cycle_mpk.plugins[3].name,
          .dst_plugin = bf16_cycle_mpk.plugins[2].name,
          .tensor_name = bf16_cycle_mpk.plugins[3].output_tensors[0].name,
      });
      const auto bf16_cycle_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          bf16_cycle_mpk, nullptr, &error);
      require(!bf16_cycle_flags.has_value(),
              "a typed selected-lineage cycle must fail closed");
      require_contains(error, "contains a cycle",
                       "typed selected-lineage cycle rejection should identify the cycle");

      auto make_exact_terminal_mpk = [&]() {
        auto terminal_mpk = exact_yolov8_mpk;
        MpkPluginIoContract terminal;
        terminal.name = "objectdecode_terminal";
        terminal.processor = "A65";
        terminal.kernel = "boxdecode";
        terminal.decode_type = "yolov8";
        for (std::size_t i = 0; i < exact_names.size(); ++i) {
          terminal.input_tensors.push_back(terminal_mpk.plugins[3U + (2U * i)].output_tensors[0]);
        }
        terminal_mpk.plugins.push_back(std::move(terminal));
        const std::size_t terminal_index = terminal_mpk.plugins.size() - 1U;
        for (std::size_t i = 0; i < exact_names.size(); ++i) {
          terminal_mpk.edges.push_back(MpkContractEdge{
              .src_plugin_index = 3U + (2U * i),
              .src_output_index = 0,
              .dst_plugin_index = terminal_index,
              .dst_input_index = static_cast<int>(i),
              .src_plugin = terminal_mpk.plugins[3U + (2U * i)].name,
              .dst_plugin = terminal_mpk.plugins[terminal_index].name,
              .tensor_name = terminal_mpk.plugins[3U + (2U * i)].output_tensors[0].name,
          });
        }
        return terminal_mpk;
      };

      auto exact_terminal_mpk = make_exact_terminal_mpk();
      const auto* exact_terminal = &exact_terminal_mpk.plugins.back();
      const auto exact_terminal_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          exact_terminal_mpk, exact_terminal, &error);
      require(exact_terminal_flags.has_value() && exact_terminal_flags->tess_needed &&
                  exact_terminal_flags->quant_needed,
              "model-owned BoxDecode must accept one exact typed lineage per terminal input: " +
                  error);
      const auto extracted_exact_terminal = build_boxdecode_static_contract_from_mpk(
          exact_terminal_mpk, *exact_terminal_flags, exact_terminal, &error);
      require(extracted_exact_terminal.has_value(),
              "model-owned exact terminal binding should build: " + error);

      auto terminal_sibling_mpk = make_exact_terminal_mpk();
      const std::size_t terminal_sibling_terminal_index = terminal_sibling_mpk.plugins.size() - 1U;
      MpkPluginIoContract terminal_sibling;
      terminal_sibling.name = "typed_sibling_before_terminal";
      terminal_sibling.processor = "EV74";
      terminal_sibling.kernel = "pass_through";
      terminal_sibling.input_tensors.push_back(terminal_sibling_mpk.plugins[3].output_tensors[0]);
      terminal_sibling.output_tensors.push_back(terminal_sibling.input_tensors.front());
      terminal_sibling.output_tensors.front().name = "typed_terminal_sibling_output";
      terminal_sibling_mpk.plugins.push_back(std::move(terminal_sibling));
      const std::size_t terminal_sibling_index = terminal_sibling_mpk.plugins.size() - 1U;
      terminal_sibling_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 3U,
          .src_output_index = 0,
          .dst_plugin_index = terminal_sibling_index,
          .dst_input_index = 0,
          .src_plugin = terminal_sibling_mpk.plugins[3].name,
          .dst_plugin = terminal_sibling_mpk.plugins[terminal_sibling_index].name,
          .tensor_name = terminal_sibling_mpk.plugins[3].output_tensors[0].name,
      });
      const auto terminal_sibling_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          terminal_sibling_mpk,
          &terminal_sibling_mpk.plugins[terminal_sibling_terminal_index], &error);
      require(!terminal_sibling_flags.has_value(),
              "a terminal edge plus typed sibling edge must fail closed");
      require_contains(error, "ambiguous post-MLA fanout",
                       "terminal sibling rejection should identify fanout");

      auto wrong_terminal_input_mpk = make_exact_terminal_mpk();
      const std::size_t wrong_input_terminal_index = wrong_terminal_input_mpk.plugins.size() - 1U;
      wrong_terminal_input_mpk.edges[wrong_terminal_input_mpk.edges.size() - exact_names.size()]
          .dst_input_index = 1;
      const auto wrong_terminal_input_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(
              wrong_terminal_input_mpk,
              &wrong_terminal_input_mpk.plugins[wrong_input_terminal_index], &error);
      require(!wrong_terminal_input_flags.has_value(),
              "a selected head bound to the wrong terminal input must fail closed");
      require_contains(error, "incorrect terminal input binding",
                       "wrong terminal input rejection should identify binding");

      auto duplicate_terminal_input_mpk = make_exact_terminal_mpk();
      const std::size_t duplicate_input_terminal_index =
          duplicate_terminal_input_mpk.plugins.size() - 1U;
      duplicate_terminal_input_mpk.edges[
          duplicate_terminal_input_mpk.edges.size() - exact_names.size() + 1U]
          .dst_input_index = 0;
      const auto duplicate_terminal_input_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(
              duplicate_terminal_input_mpk,
              &duplicate_terminal_input_mpk.plugins[duplicate_input_terminal_index], &error);
      require(!duplicate_terminal_input_flags.has_value(),
              "duplicate model-owned terminal input bindings must fail closed");
      require_contains(error, "incorrect terminal input binding",
                       "duplicate terminal binding should fail the exact per-head bijection");

      auto wrong_terminal_mpk = make_exact_terminal_mpk();
      const std::size_t exact_wrong_terminal_index = wrong_terminal_mpk.plugins.size() - 1U;
      MpkPluginIoContract later_wrong_terminal;
      later_wrong_terminal.name = "later_wrong_terminal";
      later_wrong_terminal.processor = "A65";
      later_wrong_terminal.kernel = "boxdecode";
      later_wrong_terminal.input_tensors.push_back(wrong_terminal_mpk.plugins[3].output_tensors[0]);
      wrong_terminal_mpk.plugins.push_back(std::move(later_wrong_terminal));
      const std::size_t later_wrong_terminal_index = wrong_terminal_mpk.plugins.size() - 1U;
      auto& first_terminal_edge =
          wrong_terminal_mpk.edges[wrong_terminal_mpk.edges.size() - exact_names.size()];
      first_terminal_edge.dst_plugin_index = later_wrong_terminal_index;
      first_terminal_edge.dst_plugin = wrong_terminal_mpk.plugins[later_wrong_terminal_index].name;
      const auto wrong_terminal_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          wrong_terminal_mpk, &wrong_terminal_mpk.plugins[exact_wrong_terminal_index], &error);
      require(!wrong_terminal_flags.has_value(),
              "a selected head routed to a different terminal plugin must fail closed");
      require_contains(error, "wrong terminal destination",
                       "wrong terminal-plugin rejection should identify destination");

      auto bf16_with_sibling_dequant_mpk = exact_packed_bf16_mpk;
      MpkPluginIoContract sibling_dequant;
      sibling_dequant.name = "sibling_dequant";
      sibling_dequant.sequence = 501;
      sibling_dequant.processor = "EV74";
      sibling_dequant.kernel = "dequantization_transform";
      sibling_dequant.quant = MpkQuantContract{{0.5}, {0}, -1};
      sibling_dequant.input_tensors.push_back(
          bf16_with_sibling_dequant_mpk.plugins[1].output_tensors[0]);
      sibling_dequant.input_tensors.front().dtype = "INT8";
      sibling_dequant.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "sibling_fp32_output",
          .dtype = "FP32",
          .mpk_shape = {80, 80, 64},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 80U * 80U * 64U * 4U,
      });
      bf16_with_sibling_dequant_mpk.plugins.push_back(std::move(sibling_dequant));
      const std::size_t sibling_dequant_index =
          bf16_with_sibling_dequant_mpk.plugins.size() - 1U;
      bf16_with_sibling_dequant_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 1U,
          .src_output_index = 0,
          .dst_plugin_index = sibling_dequant_index,
          .dst_input_index = 0,
          .src_plugin = bf16_with_sibling_dequant_mpk.plugins[1].name,
          .dst_plugin = bf16_with_sibling_dequant_mpk.plugins[sibling_dequant_index].name,
          .tensor_name = bf16_with_sibling_dequant_mpk.plugins[1].output_tensors[0].name,
      });
      const auto sibling_dequant_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          bf16_with_sibling_dequant_mpk, nullptr, &error);
      require(!sibling_dequant_flags.has_value(),
              "a selected raw head with a sibling typed-Dequant fanout must fail closed");
      require_contains(error, "ambiguous post-MLA fanout",
                       "sibling typed-Dequant rejection should identify ambiguous topology");

      auto unknown_then_dequant_mpk = exact_packed_bf16_mpk;
      auto& unknown_transform = unknown_then_dequant_mpk.plugins[2];
      unknown_transform.name = "opaque_value_transform";
      unknown_transform.processor = "A65";
      unknown_transform.kernel = "opaque_tvm";
      auto& downstream_dequant = unknown_then_dequant_mpk.plugins[3];
      downstream_dequant.name = "dequant_after_opaque";
      downstream_dequant.processor = "EV74";
      downstream_dequant.kernel = "dequantization_transform";
      downstream_dequant.quant = MpkQuantContract{{0.5}, {0}, -1};
      const auto unknown_then_dequant_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          unknown_then_dequant_mpk, nullptr, &error);
      require(!unknown_then_dequant_flags.has_value(),
              "an opaque transform before typed Dequant must block numeric authority");
      require_contains(error, "unsupported or ambiguous post-MLA transform",
                       "opaque-transform rejection should occur before downstream qparams");

      auto mixed_packed_source_mpk = exact_packed_bf16_mpk;
      mixed_packed_source_mpk.plugins[2].frame_type = "INT8";
      mixed_packed_source_mpk.plugins[2].input_tensors.front().dtype = "INT8";
      mixed_packed_source_mpk.plugins[2].input_tensors.front().size_bytes = exact_spans[0];
      mixed_packed_source_mpk.plugins[2].output_tensors.front().dtype = "INT8";
      mixed_packed_source_mpk.plugins[2].output_tensors.front().size_bytes = exact_spans[0];
      mixed_packed_source_mpk.plugins[3].input_tensors.front().dtype = "INT8";
      mixed_packed_source_mpk.plugins[3].input_tensors.front().size_bytes = exact_spans[0];
      const auto mixed_packed_source_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          mixed_packed_source_mpk, nullptr, &error);
      require(!mixed_packed_source_flags.has_value(),
              "mixed BF16/INT8 selected carrier lineages must fail closed");
      require_contains(error, "mix incompatible source dtypes",
                       "mixed selected-carrier rejection should identify the domain conflict");

      auto contradictory_small_c_bf16_mpk = exact_packed_bf16_mpk;
      auto& contradictory_detess = contradictory_small_c_bf16_mpk.plugins[2];
      contradictory_detess.frame_shape = {1, 80, 80, 4};
      contradictory_detess.runtime_frame_shape = {1, 80, 80, 4};
      // This is exactly an INT8 C16 carrier span. It must not be accepted as BF16 merely because
      // 8 derived physical channels are greater than the four logical channels.
      contradictory_detess.input_tensors.front().size_bytes = 80U * 80U * 16U;
      const auto contradictory_small_c_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(
              contradictory_small_c_bf16_mpk, nullptr, &error);
      require(!contradictory_small_c_flags.has_value(),
              "an INT8 C16 span must not masquerade as a BF16 C4 carrier");
      require_contains(error, "exact C16-aligned frame_shape*dtype extent",
                       "small-C BF16 dtype/span rejection should identify the exact C16 byte "
                       "contract");

      auto incomplete_dequant_mpk = exact_yolov8_mpk;
      incomplete_dequant_mpk.plugins[3].quant.reset();
      const auto incomplete_dequant_flags = resolve_model_managed_boxdecode_route_flags_from_mpk(
          incomplete_dequant_mpk, nullptr, &error);
      require(!incomplete_dequant_flags.has_value(),
              "typed Dequant without complete per-branch qparams must fail closed");
      require_contains(error, "typed Dequant stage requires exact scalar per-branch quant facts",
                       "incomplete typed Dequant rejection should identify missing qparams");

      // BF16 YOLO packages publish one dense MLA port per head and use Cast only to expose the
      // final FP32 API tensors.  Cast/pass-through preserve the score value domain, so their
      // compiler-authored class_prob/class_logit names remain the authority when BoxDecode
      // retargets the route to the raw BF16 MLA ports.
      auto make_bf16_cast_yolov8_mpk = [&](const std::string& score_domain) {
        MpkContract contract;
        MpkPluginIoContract mla_stage;
        mla_stage.name = "MLA_0";
        mla_stage.sequence = 1;
        mla_stage.processor = "MLA";
        mla_stage.kernel = "mla";
        mla_stage.canonical_output_dtype = "BF16";
        for (std::size_t i = 0; i < widths.size(); ++i) {
          const auto size_bytes = static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 2);
          mla_stage.output_tensors.push_back(MpkTensorContract{
              .tensor_index = static_cast<int>(i),
              .physical_index = static_cast<int>(i),
              .name = "MLA_0_" + std::to_string(i),
              .dtype = "BF16",
              .mpk_shape = {widths[i], widths[i], channels[i]},
              .shape_semantics = MpkShapeSemantics::Geometry,
              .size_bytes = size_bytes,
              .logical_shape = {widths[i], widths[i], channels[i]},
          });
        }
        contract.plugins.push_back(std::move(mla_stage));

        std::array<std::string, 6> cast_output_names;
        for (std::size_t i = 0; i < widths.size(); ++i) {
          const bool score = i >= 3U;
          const std::size_t head = score ? i - 3U : i;
          const std::string semantic_name =
              score ? ("class_" + score_domain + "_" + std::to_string(head))
                    : ("bbox_" + std::to_string(head));
          cast_output_names[i] = "cast_" + std::to_string(i + 2U) + "/" + semantic_name;

          MpkPluginIoContract cast_stage;
          cast_stage.name = "cast_" + std::to_string(i + 2U);
          cast_stage.sequence = static_cast<int>(i + 2U);
          cast_stage.processor = "EV74";
          cast_stage.kernel = "cast_transform";
          cast_stage.input_tensors.push_back(MpkTensorContract{
              .tensor_index = 0,
              .name = "MLA_0_" + std::to_string(i),
              .dtype = "BF16",
              .mpk_shape = {widths[i], widths[i], channels[i]},
              .shape_semantics = MpkShapeSemantics::Geometry,
              .size_bytes =
                  static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 2),
              .logical_shape = {widths[i], widths[i], channels[i]},
          });
          cast_stage.output_tensors.push_back(MpkTensorContract{
              .tensor_index = 0,
              .name = cast_output_names[i],
              .dtype = "FP32",
              .mpk_shape = {widths[i], widths[i], channels[i]},
              .shape_semantics = MpkShapeSemantics::Geometry,
              .size_bytes =
                  static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 4),
              .logical_shape = {widths[i], widths[i], channels[i]},
          });
          contract.plugins.push_back(std::move(cast_stage));
          contract.edges.push_back(MpkContractEdge{
              .src_plugin_index = 0U,
              .src_output_index = static_cast<int>(i),
              .dst_plugin_index = i + 1U,
              .dst_input_index = 0,
              .src_plugin = "MLA_0",
              .dst_plugin = contract.plugins.back().name,
              .tensor_name = "MLA_0_" + std::to_string(i),
          });
        }

        MpkPluginIoContract pass_through;
        pass_through.name = "PassThrough";
        pass_through.sequence = 8;
        pass_through.processor = "EV74";
        pass_through.kernel = "pass_through";
        for (std::size_t i = 0; i < widths.size(); ++i) {
          pass_through.input_tensors.push_back(MpkTensorContract{
              .tensor_index = static_cast<int>(i),
              .name = cast_output_names[i],
              .dtype = "FP32",
              .mpk_shape = {widths[i], widths[i], channels[i]},
              .shape_semantics = MpkShapeSemantics::Geometry,
              .size_bytes =
                  static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 4),
              .logical_shape = {widths[i], widths[i], channels[i]},
          });
          pass_through.output_tensors.push_back(MpkTensorContract{
              .tensor_index = static_cast<int>(i),
              .name = "pass_through_out_" + std::to_string(i),
              .dtype = "FP32",
              .mpk_shape = {widths[i], widths[i], channels[i]},
              .shape_semantics = MpkShapeSemantics::Geometry,
              .size_bytes =
                  static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 4),
              .logical_shape = {widths[i], widths[i], channels[i]},
          });
          contract.edges.push_back(MpkContractEdge{
              .src_plugin_index = i + 1U,
              .src_output_index = 0,
              .dst_plugin_index = 7U,
              .dst_input_index = static_cast<int>(i),
              .src_plugin = contract.plugins[i + 1U].name,
              .dst_plugin = "PassThrough",
              .tensor_name = cast_output_names[i],
          });
        }
        contract.plugins.push_back(std::move(pass_through));
        return contract;
      };

      const auto require_bf16_score_domain = [&](const std::string& domain,
                                                  BoxDecodeScoreActivation activation,
                                                  simaai::neat::BoxDecodeTypeOption option) {
        auto bf16_mpk = make_bf16_cast_yolov8_mpk(domain);
        const auto extracted_bf16 = build_boxdecode_static_contract_from_mpk(
            bf16_mpk, make_flags(false, false), &error);
        require(extracted_bf16.has_value(),
                "BF16 value-preserving Cast lineage should preserve its score domain: " + error);
        require(extracted_bf16->tensors.size() == 6U &&
                    extracted_bf16->score_activation == activation,
                "BF16 raw MLA contract should recover the compiler-authored score activation");
        for (std::size_t i = 0; i < 3U; ++i) {
          require(extracted_bf16->tensors[i].logical_name == "bbox_" + std::to_string(i) &&
                      extracted_bf16->tensors[i + 3U].logical_name ==
                          "class_" + domain + "_" + std::to_string(i),
                  "BF16 raw MLA ports should recover exact grouped semantic names through Cast");
          require(extracted_bf16->tensors[i].data_type == "BF16" &&
                      extracted_bf16->tensors[i + 3U].data_type == "BF16",
                  "BF16 route must keep the raw MLA dtype instead of the downstream FP32 Cast");
        }
        for (std::size_t i = 0; i < extracted_bf16->tensors.size(); ++i) {
          const auto expected_span =
              static_cast<std::size_t>(widths[i] * widths[i] * channels[i] * 2);
          const auto& tensor = extracted_bf16->tensors[i];
          require(tensor.source_output_slot == static_cast<int>(i) &&
                      tensor.source_physical_index == static_cast<int>(i) &&
                      tensor.source_segment_name == "MLA_0_" + std::to_string(i) &&
                      tensor.source_byte_offset == 0 && tensor.source_size_bytes == expected_span,
                  "BF16 semantic propagation must preserve each exact raw MLA port and span");
        }

        auto finalized_bf16 = stagesemantics::finalize_boxdecode_static_contract(
            *extracted_bf16, simaai::neat::BoxDecodeType::YoloV8, std::nullopt,
            make_flags(false, false), simaai::neat::BoxDecodeTypeOption::Auto, 0.25, 0.55, 100,
            80, {"orig_width", "orig_height"});
        stagesemantics::resolve_grouped_yolo_dfl_score_domain(&finalized_bf16);
        require(finalized_bf16.decode_type_option == option &&
                    finalized_bf16.score_activation == activation,
                "BF16 YOLO grouped decoder option must follow the exact Cast score domain");
        const auto compiled_bf16 =
            stagesemantics::build_boxdecode_compiled_contract(finalized_bf16);
        require(compiled_bf16.payload.input_dtype == "BF16" &&
                    compiled_bf16.runtime_contract.logical_inputs.size() == 6U &&
                    compiled_bf16.runtime_contract.input_bindings.size() == 6U,
                "compiled BF16 payload must canonicalize source_bf16 and preserve all six "
                "exact MLA ports");
        for (std::size_t i = 0; i < 6U; ++i) {
          const auto& logical = compiled_bf16.runtime_contract.logical_inputs[i];
          const auto& binding = compiled_bf16.runtime_contract.input_bindings[i];
          require(logical.logical_name == finalized_bf16.tensors[i].logical_name &&
                      logical.dtype == "BF16" &&
                      binding.src_output_slot == static_cast<int>(i) &&
                      binding.src_physical_output_index == static_cast<int>(i) &&
                      binding.source_segment_name == "MLA_0_" + std::to_string(i),
                  "compiled BF16 bindings must stay in exact role/head/raw-port order");
        }
      };

      require_bf16_score_domain(
          "prob", BoxDecodeScoreActivation::Identity,
          simaai::neat::BoxDecodeTypeOption::GroupedByRoleProbability);
      require_bf16_score_domain("logit", BoxDecodeScoreActivation::Sigmoid,
                                simaai::neat::BoxDecodeTypeOption::GroupedByRoleLogit);

      auto conflicting_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      conflicting_bf16_mpk.plugins[7].output_tensors[3].name =
          "pass_through/class_logit_0";
      const auto conflicting_bf16 = build_boxdecode_static_contract_from_mpk(
          conflicting_bf16_mpk, make_flags(false, false), &error);
      require(!conflicting_bf16.has_value(),
              "one value-preserving branch must reject mixed probability/logit evidence");
      require_contains(error, "conflicting score domains",
                       "mixed BF16 score-domain rejection should name the exact conflict");

      auto cross_head_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      cross_head_bf16_mpk.plugins[5].output_tensors[0].name = "cast_6/class_logit_1";
      cross_head_bf16_mpk.plugins[7].input_tensors[4].name = "cast_6/class_logit_1";
      cross_head_bf16_mpk.edges[10].tensor_name = "cast_6/class_logit_1";
      const auto cross_head_bf16 = build_boxdecode_static_contract_from_mpk(
          cross_head_bf16_mpk, make_flags(false, false), &error);
      require(!cross_head_bf16.has_value(),
              "grouped BF16 heads must reject cross-head probability/logit mixtures");
      require_contains(error, "mixed probability/logit domains",
                       "cross-head rejection should identify the aggregate domain conflict");

      auto direct_conflict_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      direct_conflict_bf16_mpk.plugins[0].output_tensors[3].name = "class_logit_0";
      direct_conflict_bf16_mpk.plugins[4].input_tensors[0].name = "class_logit_0";
      direct_conflict_bf16_mpk.edges[3].tensor_name = "class_logit_0";
      const auto direct_conflict_bf16 = build_boxdecode_static_contract_from_mpk(
          direct_conflict_bf16_mpk, make_flags(false, false), &error);
      require(!direct_conflict_bf16.has_value(),
              "direct MLA logit semantics must reject a value-preserving Cast prob rename");
      require_contains(error, "conflicting score domains",
                       "raw-MLA/Cast conflict should identify one-branch domain disagreement");

      auto direct_probability_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      auto direct_logit_bf16_mpk = make_bf16_cast_yolov8_mpk("logit");
      for (std::size_t i = 0; i < 3U; ++i) {
        direct_probability_bf16_mpk.plugins[0].output_tensors[i].name =
            "bbox_" + std::to_string(i);
        direct_probability_bf16_mpk.plugins[0].output_tensors[i + 3U].name =
            "class_prob_" + std::to_string(i);
        direct_logit_bf16_mpk.plugins[0].output_tensors[i].name = "bbox_" + std::to_string(i);
        direct_logit_bf16_mpk.plugins[0].output_tensors[i + 3U].name =
            "class_logit_" + std::to_string(i);
      }
      const auto direct_probability_bf16 = build_boxdecode_static_contract_from_mpk(
          direct_probability_bf16_mpk, make_flags(false, false), &error);
      require(direct_probability_bf16.has_value() &&
                  direct_probability_bf16->score_activation ==
                      BoxDecodeScoreActivation::Identity,
              "direct MLA class_prob names must remain exact probability authority");
      const auto direct_logit_bf16 = build_boxdecode_static_contract_from_mpk(
          direct_logit_bf16_mpk, make_flags(false, false), &error);
      require(direct_logit_bf16.has_value() &&
                  direct_logit_bf16->score_activation == BoxDecodeScoreActivation::Sigmoid,
              "direct MLA class_logit names must remain exact logit authority");

      auto missing_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      for (std::size_t i = 3U; i < 6U; ++i) {
        const std::string opaque = "cast_opaque_score_" + std::to_string(i - 3U);
        missing_bf16_mpk.plugins[i + 1U].output_tensors[0].name = opaque;
        missing_bf16_mpk.plugins[7].input_tensors[i].name = opaque;
        missing_bf16_mpk.edges[6U + i].tensor_name = opaque;
      }
      const auto missing_bf16 = build_boxdecode_static_contract_from_mpk(
          missing_bf16_mpk, make_flags(false, false), &error);
      require(missing_bf16.has_value() &&
                  missing_bf16->score_activation == BoxDecodeScoreActivation::Unknown &&
                  missing_bf16->decode_type_option == simaai::neat::BoxDecodeTypeOption::Auto,
              "missing BF16 semantic domains must remain unknown without dtype fallbacks");
      bool missing_bf16_rejected = false;
      try {
        auto finalized_missing = stagesemantics::finalize_boxdecode_static_contract(
            *missing_bf16, simaai::neat::BoxDecodeType::YoloV8, std::nullopt,
            make_flags(false, false), simaai::neat::BoxDecodeTypeOption::Auto, 0.25, 0.55, 100,
            80, {"orig_width", "orig_height"});
        stagesemantics::resolve_grouped_yolo_dfl_score_domain(&finalized_missing);
      } catch (const std::exception& e) {
        missing_bf16_rejected = true;
        require_contains(std::string(e.what()), "score domain is ambiguous",
                         "missing BF16 domain should fail at grouped-YOLO resolution");
      }
      require(missing_bf16_rejected,
              "missing BF16 domain must fail closed once grouped YOLO is requested");

      auto barrier_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      for (std::size_t i = 3U; i < 6U; ++i) {
        barrier_bf16_mpk.plugins[i + 1U].kernel = "opaque_nonlinear_transform";
      }
      const auto barrier_bf16 = build_boxdecode_static_contract_from_mpk(
          barrier_bf16_mpk, make_flags(false, false), &error);
      require(!barrier_bf16.has_value(),
              "unknown nonlinear stages must fail closed before semantic-name evidence");
      require_contains(error, "unsupported or ambiguous post-MLA transform",
                       "unknown nonlinear barrier should identify the strict path violation");

      auto multi_cast_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      for (std::size_t i = 3U; i < 6U; ++i) {
        const std::string opaque = "reshape_score_" + std::to_string(i - 3U);
        multi_cast_bf16_mpk.plugins[i + 1U].kernel = "reshape_transform";
        multi_cast_bf16_mpk.plugins[i + 1U].output_tensors[0].name = opaque;
        multi_cast_bf16_mpk.plugins[7].input_tensors[i].name = opaque;
        multi_cast_bf16_mpk.edges[6U + i].tensor_name = opaque;
        multi_cast_bf16_mpk.plugins[7].output_tensors[i].name =
            "cast_many/class_prob_" + std::to_string(i - 3U);
      }
      multi_cast_bf16_mpk.plugins[7].kernel = "cast_transform";
      const auto multi_cast_bf16 = build_boxdecode_static_contract_from_mpk(
          multi_cast_bf16_mpk, make_flags(false, false), &error);
      require(!multi_cast_bf16.has_value(),
              "a multi-input/multi-output stage named Cast must fail closed");
      require_contains(error, "unsupported or ambiguous post-MLA transform",
                       "multi-I/O fake Cast should identify the strict path violation");

      auto sigmoid_bf16_mpk = make_bf16_cast_yolov8_mpk("prob");
      for (std::size_t i = 3U; i < 6U; ++i) {
        sigmoid_bf16_mpk.plugins[i + 1U].kernel = "sigmoid_transform";
      }
      const auto sigmoid_bf16 = build_boxdecode_static_contract_from_mpk(
          sigmoid_bf16_mpk, make_flags(false, false), &error);
      require(sigmoid_bf16.has_value() &&
                  sigmoid_bf16->score_activation == BoxDecodeScoreActivation::Sigmoid &&
                  sigmoid_bf16->decode_type_option ==
                      simaai::neat::BoxDecodeTypeOption::GroupedByRoleLogit,
              "typed Sigmoid lineage must map published probability back to raw MLA logits");

      auto missing_int8_mpk = exact_yolov8_mpk;
      for (std::size_t i = 3U; i < 6U; ++i) {
        missing_int8_mpk.plugins[3U + (2U * i)].output_tensors[0].name =
            "opaque_quant_score_" + std::to_string(i - 3U);
      }
      const auto missing_int8 = build_boxdecode_static_contract_from_mpk(
          missing_int8_mpk, make_flags(true, false), &error);
      require(missing_int8.has_value() &&
                  missing_int8->score_activation == BoxDecodeScoreActivation::Unknown &&
                  missing_int8->decode_type_option == simaai::neat::BoxDecodeTypeOption::Auto,
              "missing INT8 semantic domains must not be inferred from dtype/qparam range");
      bool missing_int8_rejected = false;
      try {
        auto finalized_missing = stagesemantics::finalize_boxdecode_static_contract(
            *missing_int8, simaai::neat::BoxDecodeType::YoloV8, std::nullopt,
            make_flags(true, false), simaai::neat::BoxDecodeTypeOption::Auto, 0.25, 0.55, 100,
            80, {"orig_width", "orig_height"});
        stagesemantics::resolve_grouped_yolo_dfl_score_domain(&finalized_missing);
      } catch (const std::exception& e) {
        missing_int8_rejected = true;
        require_contains(std::string(e.what()), "score domain is ambiguous",
                         "missing INT8 domain should fail at grouped-YOLO resolution");
      }
      require(missing_int8_rejected,
              "missing INT8 domain must fail closed once grouped YOLO is requested");

      auto interleaved_unspecified_mpk = make_bf16_cast_yolov8_mpk("prob");
      constexpr std::array<const char*, 6> interleaved_names = {
          "bbox_0", "class_prob_0", "bbox_1", "class_prob_1", "bbox_2", "class_prob_2",
      };
      for (std::size_t i = 0; i < interleaved_names.size(); ++i) {
        const std::string wrapped =
            "cast_" + std::to_string(i + 2U) + "/" + interleaved_names[i];
        interleaved_unspecified_mpk.plugins[i + 1U].output_tensors[0].name = wrapped;
        interleaved_unspecified_mpk.plugins[7].input_tensors[i].name = wrapped;
        interleaved_unspecified_mpk.edges[6U + i].tensor_name = wrapped;
      }
      const auto interleaved_unspecified = build_boxdecode_static_contract_from_mpk(
          interleaved_unspecified_mpk, make_flags(false, false), &error);
      require(interleaved_unspecified.has_value() &&
                  interleaved_unspecified->decode_type_option ==
                      simaai::neat::BoxDecodeTypeOption::Auto,
              "Unspecified interleaved lineage must not be claimed as grouped YOLO");

      auto spoofed_dequant_mpk = exact_yolov8_mpk;
      for (std::size_t i = 0; i < 6U; ++i) {
        auto& stage = spoofed_dequant_mpk.plugins[3U + (2U * i)];
        stage.name = "opaque_dequant_proxy_" + std::to_string(i);
        stage.processor = "A65";
        stage.kernel = "opaque_transform";
      }
      const auto spoofed_dequant = build_boxdecode_static_contract_from_mpk(
          spoofed_dequant_mpk, make_flags(true, false), &error);
      require(!spoofed_dequant.has_value(),
              "opaque stages with dequant in their names must not authorize qparams");
      require_contains(error, "unsupported or ambiguous post-MLA transform",
                       "spoofed dequant qparam rejection should identify the strict path "
                       "violation");

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
      require(extracted_probability_quant->score_activation == BoxDecodeScoreActivation::Unknown,
              "core extractor should not infer score activation before an explicit BoxDecode "
              "family is selected");
      require(extracted_probability_quant->decode_type_option ==
                  simaai::neat::BoxDecodeTypeOption::Auto,
              "core extractor should not infer grouped-by-role score domain before an explicit "
              "BoxDecode family is selected");

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
          .shape_semantics = MpkShapeSemantics::PackedExtent,
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
          .shape_semantics = MpkShapeSemantics::PackedExtent,
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
        detess_stage.processor = "EV74";
        detess_stage.kernel = "detess";
        detess_stage.slice_shape = {1, 80, i == 0 ? 64 : 80};
        add_packed_detess_facts(detess_stage, i == 0 ? "bbox_0" : "opaque_score_0", 80, 80,
                                i == 0 ? 64 : 80);
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
      require(extracted_packed_parent->tess_needed,
              "typed packed-parent storage must select BoxDecode's internal detess path even "
              "when the retargeted route flag omits a separate Detess stage");
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

      MpkContract dense_unpack_parent = packed_parent_mpk;
      dense_unpack_parent.plugins.resize(2U);
      dense_unpack_parent.edges.resize(1U);
      for (auto& output : dense_unpack_parent.plugins[1].output_tensors) {
        output.shape_semantics = MpkShapeSemantics::Geometry;
      }
      const auto extracted_dense_unpack_parent = build_boxdecode_static_contract_from_mpk(
          dense_unpack_parent, make_flags(true, false), &error);
      require(extracted_dense_unpack_parent.has_value(),
              "dense unpack outputs should remain externally decodable: " + error);
      require(!extracted_dense_unpack_parent->tess_needed,
              "dense unpack storage must not be classified as a direct packed route");

      const auto extracted_packed_parent_bypass = build_boxdecode_static_contract_from_mpk(
          packed_parent_mpk, make_flags(true, true), &error);
      require(extracted_packed_parent_bypass.has_value(),
              "typed packed route should allow external boxdecode to source from the packed MLA "
              "parent: " +
                  error);
      const auto extracted_packed_parent_bypass_subset = extract_boxdecode_contract_subset_from_mpk(
          packed_parent_mpk, make_flags(true, true), nullptr, &error);
      require(extracted_packed_parent_bypass_subset.has_value(),
              "typed packed subset extraction should preserve direct packed-parent facts: " +
                  error);
      require_subset_matches_static_contract(*extracted_packed_parent_bypass_subset,
                                             *extracted_packed_parent_bypass,
                                             "typed packed-parent route");
      require(extracted_packed_parent_bypass->tess_needed,
              "typed packed-parent route should preserve internal detess semantics");
      require(extracted_packed_parent_bypass->tensors[0].source_segment_name == "output_tensor",
              "typed packed-parent route should bind logical tensors to the packed MLA parent");
      require(extracted_packed_parent_bypass->tensors[0].source_byte_offset == 1209600,
              "typed packed-parent route should preserve the first packed MLA head offset");
      require(extracted_packed_parent_bypass->tensors[1].source_byte_offset == 1747200,
              "typed packed-parent route should preserve the second packed MLA head offset");
      require(extracted_packed_parent_bypass->physical_inputs[0].name == "output_tensor",
              "typed packed-parent route should publish the packed MLA parent as the "
              "TensorBuffer source");
      require(extracted_packed_parent_bypass->physical_inputs[0].byte_offset == 1209600,
              "typed packed-parent physical input should preserve the first packed head offset");
      require(extracted_packed_parent_bypass->physical_inputs[1].byte_offset == 1747200,
              "typed packed-parent physical input should preserve the second packed head offset");

      // AFE emits cblock=false outputs as tiled HWC with C16 padding. The
      // logical C65 detector therefore occupies 60*80*80 bytes in the shared
      // MLA carrier, and the descriptor must begin after that physical span.
      MpkContract superpoint_hwc_parent = packed_parent_mpk;
      auto& hwc_mla = superpoint_hwc_parent.plugins[0];
      hwc_mla.output_tensors[0].mpk_shape = {1, 1612800};
      hwc_mla.output_tensors[0].size_bytes = 1612800U;
      hwc_mla.quant = MpkQuantContract{{4.0, 0.25}, {0, 0}, -1};

      auto& hwc_unpack = superpoint_hwc_parent.plugins[1];
      hwc_unpack.output_tensors[0].name = "detector";
      hwc_unpack.output_tensors[0].mpk_shape = {1, 384000};
      hwc_unpack.output_tensors[0].logical_shape = {60, 80, 65};
      hwc_unpack.output_tensors[0].size_bytes = 384000U;
      hwc_unpack.output_tensors[0].byte_offset = 0;
      hwc_unpack.output_tensors[0].source_byte_offset = 0;
      hwc_unpack.output_tensors[1].name = "descriptor";
      hwc_unpack.output_tensors[1].mpk_shape = {1, 1228800};
      hwc_unpack.output_tensors[1].logical_shape = {60, 80, 256};
      hwc_unpack.output_tensors[1].size_bytes = 1228800U;
      hwc_unpack.output_tensors[1].byte_offset = 0;
      hwc_unpack.output_tensors[1].source_byte_offset = 0;

      const std::array<const char*, 2> hwc_names = {"detector", "descriptor"};
      const std::array<int, 2> hwc_channels = {65, 256};
      const std::array<std::size_t, 2> hwc_sizes = {384000U, 1228800U};
      for (std::size_t i = 0; i < 2U; ++i) {
        auto& detess = superpoint_hwc_parent.plugins[2U + i];
        detess.name = std::string("superpoint_detess_") + std::to_string(i);
        detess.processor = "EV74";
        detess.kernel = "detessellation_transform";
        detess.slice_shape = {80, 1, hwc_channels[i]};
        detess.frame_shape = {1, 60, 80, hwc_channels[i]};
        detess.frame_type = "INT8";
        detess.has_cblock = true;
        detess.cblock = false;
        detess.has_align_c16 = true;
        detess.align_c16 = true;
        detess.input_tensors = {MpkTensorContract{
            .tensor_index = 0,
            .name = hwc_names[i],
            .dtype = "INT8",
            .mpk_shape = {1, static_cast<std::int64_t>(hwc_sizes[i])},
            .shape_semantics = MpkShapeSemantics::PackedExtent,
            .size_bytes = hwc_sizes[i],
            .logical_shape = {60, 80, hwc_channels[i]},
        }};
        detess.output_tensors = {MpkTensorContract{
            .tensor_index = 0,
            .name = std::string(hwc_names[i]) + "_detess",
            .dtype = "INT8",
            .mpk_shape = {60, 80, hwc_channels[i]},
            .shape_semantics = MpkShapeSemantics::Geometry,
            .size_bytes = static_cast<std::size_t>(60 * 80 * hwc_channels[i]),
            .logical_shape = {60, 80, hwc_channels[i]},
        }};
        auto& edge = superpoint_hwc_parent.edges[1U + i];
        edge.tensor_name = hwc_names[i];
        edge.dst_plugin = detess.name;
      }

      MpkPluginIoContract hwc_boxdecode;
      hwc_boxdecode.name = "boxdecode_superpoint";
      hwc_boxdecode.kernel = "boxdecode";
      hwc_boxdecode.decode_type = "superpoint";
      hwc_boxdecode.superpoint.schema_version = 0;
      hwc_boxdecode.superpoint.descriptor_dim = 256;
      hwc_boxdecode.input_tensors = {
          superpoint_hwc_parent.plugins[2].output_tensors[0],
          superpoint_hwc_parent.plugins[3].output_tensors[0],
      };
      superpoint_hwc_parent.plugins.push_back(std::move(hwc_boxdecode));
      for (std::size_t i = 0; i < 2U; ++i) {
        superpoint_hwc_parent.edges.push_back(MpkContractEdge{
            .src_plugin_index = 2U + i,
            .src_output_index = 0,
            .dst_plugin_index = 4U,
            .dst_input_index = static_cast<int>(i),
            .src_plugin = superpoint_hwc_parent.plugins[2U + i].name,
            .dst_plugin = "boxdecode_superpoint",
            .tensor_name = superpoint_hwc_parent.plugins[2U + i].output_tensors[0].name,
        });
      }

      const auto extracted_superpoint_hwc = build_boxdecode_static_contract_from_mpk(
          superpoint_hwc_parent, make_flags(true, true),
          &superpoint_hwc_parent.plugins[4], &error);
      require(extracted_superpoint_hwc.has_value(),
              "cblock=false SuperPoint route should extract tiled-HWC source facts: " + error);
      require(extracted_superpoint_hwc->decode_type == simaai::neat::BoxDecodeType::SuperPoint &&
                  extracted_superpoint_hwc->tensors.size() == 2U,
              "cblock=false SuperPoint route should retain both semantic heads");
      require(extracted_superpoint_hwc->tensors[0].source_storage_kind ==
                      BoxDecodeSourceStorageKind::PackedHwcC16 &&
                  extracted_superpoint_hwc->tensors[1].source_storage_kind ==
                      BoxDecodeSourceStorageKind::PackedHwcC16,
              "cblock=false align_c16 sources should publish PackedHwcC16 storage");
      require(extracted_superpoint_hwc->tensors[0].source_byte_offset == 0 &&
                  extracted_superpoint_hwc->tensors[1].source_byte_offset == 384000,
              "shared SuperPoint parent offsets must use the padded detector byte span");
      require(extracted_superpoint_hwc->tensors[0].source_size_bytes == 384000U &&
                  extracted_superpoint_hwc->tensors[1].source_size_bytes == 1228800U,
              "cblock=false SuperPoint heads must preserve physical source spans");

      // Stage-by-stage storage regression: direct route has full-frame detess slice but the source
      // is still packed/cblock. This is the YOLO26-pose INT8 direct failure mode.
      MpkContract direct_int8_mpk;
      MpkPluginIoContract direct_mla;
      direct_mla.name = "MLA_0";
      direct_mla.sequence = 1;
      direct_mla.processor = "MLA";
      direct_mla.kernel = "mla";
      direct_mla.canonical_output_dtype = "INT8";
      direct_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 102400},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 102400,
      });
      direct_int8_mpk.plugins.push_back(std::move(direct_mla));

      MpkPluginIoContract direct_unpack;
      direct_unpack.name = "MLA_0_ofm_unpack_transform";
      direct_unpack.sequence = 2;
      direct_unpack.kernel = "ofm_unpack";
      direct_unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "score_head_packed",
          .segment_name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 102400},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 102400,
          .logical_shape = {80, 80, 1},
      });
      direct_int8_mpk.plugins.push_back(std::move(direct_unpack));
      direct_int8_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack_transform",
          .tensor_name = "output_tensor",
      });

      MpkPluginIoContract direct_detess;
      direct_detess.name = "detessellation_transform";
      direct_detess.sequence = 3;
      direct_detess.processor = "EV74";
      direct_detess.kernel = "detessellation_transform";
      direct_detess.slice_shape = {80, 80, 1};
      add_packed_detess_facts(direct_detess, "score_head_packed", 80, 80, 1, "INT8", 102400U);
      direct_detess.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "score_head_detess",
          .dtype = "INT8",
          .mpk_shape = {80, 80, 1},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 6400,
          .logical_shape = {80, 80, 1},
      });
      direct_int8_mpk.plugins.push_back(std::move(direct_detess));
      direct_int8_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 1U,
          .src_output_index = 0,
          .dst_plugin_index = 2U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0_ofm_unpack_transform",
          .dst_plugin = "detessellation_transform",
          .tensor_name = "score_head_packed",
      });

      MpkPluginIoContract direct_dequant;
      direct_dequant.name = "dequantize";
      direct_dequant.sequence = 4;
      direct_dequant.processor = "EV74";
      direct_dequant.kernel = "dequantize";
      direct_dequant.quant = MpkQuantContract{{0.5}, {3}, -1};
      direct_dequant.input_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "score_head_detess",
          .dtype = "INT8",
          .mpk_shape = {80, 80, 1},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 6400U,
          .logical_shape = {80, 80, 1},
      });
      direct_dequant.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "class_prob_0",
          .dtype = "FP32",
          .mpk_shape = {80, 80, 1},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 80U * 80U * 4U,
          .logical_shape = {80, 80, 1},
      });
      direct_int8_mpk.plugins.push_back(std::move(direct_dequant));
      direct_int8_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 2U,
          .src_output_index = 0,
          .dst_plugin_index = 3U,
          .dst_input_index = 0,
          .src_plugin = "detessellation_transform",
          .dst_plugin = "dequantize",
          .tensor_name = "score_head_detess",
      });

      const auto extracted_direct_int8 =
          build_boxdecode_static_contract_from_mpk(direct_int8_mpk, make_flags(true, true), &error);
      require(extracted_direct_int8.has_value(),
              "INT8 direct packed/cblock route should extract storage facts: " + error);
      require(extracted_direct_int8->tensors.size() == 1U,
              "INT8 direct route should keep one test head");
      require(extracted_direct_int8->tensors[0].source_storage_kind ==
                  BoxDecodeSourceStorageKind::PackedCBlock,
              "INT8 direct full-frame detess slice must still use packed/cblock storage");
      require(extracted_direct_int8->tensors[0].input_shape == std::vector<int>({80, 80, 1}),
              "INT8 direct packed route should use detess frame_shape as logical decode shape");
      require(extracted_direct_int8->tensors[0].slice_shape == std::vector<int>({80, 80, 1}),
              "INT8 direct packed route should preserve full-frame detess slice shape");
      require(extracted_direct_int8->tensors[0].source_size_bytes == 102400U,
              "INT8 direct packed route should preserve packed source byte span");
      const auto extracted_direct_int8_subset = extract_boxdecode_contract_subset_from_mpk(
          direct_int8_mpk, make_flags(true, true), nullptr, &error);
      require(extracted_direct_int8_subset.has_value(),
              "INT8 direct subset should carry packed/cblock storage kind: " + error);
      require(extracted_direct_int8_subset->tensor_storage_kind[0] ==
                  static_cast<int>(BoxDecodeSourceStorageKind::PackedCBlock),
              "INT8 direct subset should expose PackedCBlock=0");

      MpkContract direct_bf16_mpk;
      MpkPluginIoContract bf16_mla;
      bf16_mla.name = "MLA_0";
      bf16_mla.sequence = 1;
      bf16_mla.processor = "MLA";
      bf16_mla.kernel = "mla";
      bf16_mla.canonical_output_dtype = "BF16";
      bf16_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "bf16_output_tensor",
          .dtype = "BF16",
          .mpk_shape = {1, 204800},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 204800,
      });
      direct_bf16_mpk.plugins.push_back(std::move(bf16_mla));

      MpkPluginIoContract bf16_unpack;
      bf16_unpack.name = "MLA_0_ofm_unpack_transform";
      bf16_unpack.sequence = 2;
      bf16_unpack.kernel = "ofm_unpack";
      bf16_unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "bbox_head_packed_bf16",
          .segment_name = "bf16_output_tensor",
          .dtype = "BF16",
          .mpk_shape = {1, 204800},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 204800,
          .logical_shape = {80, 80, 4},
      });
      direct_bf16_mpk.plugins.push_back(std::move(bf16_unpack));
      direct_bf16_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack_transform",
          .tensor_name = "bf16_output_tensor",
      });

      MpkPluginIoContract bf16_detess;
      bf16_detess.name = "bf16_detessellation_transform";
      bf16_detess.sequence = 3;
      bf16_detess.processor = "EV74";
      bf16_detess.kernel = "detessellation_transform";
      bf16_detess.slice_shape = {16, 4, 4};
      add_packed_detess_facts(bf16_detess, "bbox_head_packed_bf16", 80, 80, 4, "BF16", 204800U);
      bf16_detess.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "bbox_head_detess_bf16",
          .dtype = "BF16",
          .mpk_shape = {80, 80, 4},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 80U * 80U * 4U * 2U,
          .logical_shape = {80, 80, 4},
      });
      direct_bf16_mpk.plugins.push_back(std::move(bf16_detess));
      direct_bf16_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 1U,
          .src_output_index = 0,
          .dst_plugin_index = 2U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0_ofm_unpack_transform",
          .dst_plugin = "bf16_detessellation_transform",
          .tensor_name = "bbox_head_packed_bf16",
      });

      MpkPluginIoContract bf16_cast;
      bf16_cast.name = "cast_bf16_to_fp32";
      bf16_cast.sequence = 4;
      bf16_cast.processor = "EV74";
      bf16_cast.kernel = "cast";
      bf16_cast.input_tensors.push_back(direct_bf16_mpk.plugins[2].output_tensors.front());
      bf16_cast.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "bbox_0",
          .dtype = "FP32",
          .mpk_shape = {80, 80, 4},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 80U * 80U * 4U * 4U,
          .logical_shape = {80, 80, 4},
      });
      direct_bf16_mpk.plugins.push_back(std::move(bf16_cast));
      direct_bf16_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 2U,
          .src_output_index = 0,
          .dst_plugin_index = 3U,
          .dst_input_index = 0,
          .src_plugin = "bf16_detessellation_transform",
          .dst_plugin = "cast_bf16_to_fp32",
          .tensor_name = "bbox_head_detess_bf16",
      });

      const auto extracted_direct_bf16 = build_boxdecode_static_contract_from_mpk(
          direct_bf16_mpk, make_flags(false, true), &error);
      require(extracted_direct_bf16.has_value(),
              "BF16 direct packed/cblock route should extract storage facts: " + error);
      require(extracted_direct_bf16->tensors[0].source_storage_kind ==
                  BoxDecodeSourceStorageKind::PackedCBlock,
              "BF16 direct route should use packed/cblock storage");
      require(extracted_direct_bf16->tensors[0].data_type == "BF16",
              "BF16 direct route should take source dtype from detess frame_type, not cast output");
      require(extracted_direct_bf16->tensors[0].source_size_bytes == 204800U,
              "BF16 direct route should preserve packed source byte span");

      auto rank2_bf16_mpk = direct_bf16_mpk;
      auto& rank2_mla_output = rank2_bf16_mpk.plugins[0].output_tensors.front();
      rank2_mla_output.mpk_shape = {1, 448};
      rank2_mla_output.size_bytes = 448U;
      auto& rank2_unpack_output = rank2_bf16_mpk.plugins[1].output_tensors.front();
      rank2_unpack_output.mpk_shape = {1, 448};
      rank2_unpack_output.logical_shape = {1, 213};
      rank2_unpack_output.size_bytes = 448U;
      auto& rank2_detess = rank2_bf16_mpk.plugins[2];
      rank2_detess.frame_shape = {1, 213};
      rank2_detess.runtime_frame_shape = {1, 1, 1, 213};
      rank2_detess.slice_shape = {1, 1, 64};
      auto& rank2_detess_input = rank2_detess.input_tensors.front();
      rank2_detess_input.mpk_shape = {1, 448};
      rank2_detess_input.logical_shape = {1, 213};
      rank2_detess_input.size_bytes = 448U;
      auto& rank2_detess_output = rank2_detess.output_tensors.front();
      rank2_detess_output.mpk_shape = {1, 213};
      rank2_detess_output.logical_shape = {1, 213};
      rank2_detess_output.size_bytes = 426U;
      auto& rank2_cast_output = rank2_bf16_mpk.plugins[3].output_tensors.front();
      rank2_cast_output.mpk_shape = {1, 213};
      rank2_cast_output.logical_shape = {1, 213};
      rank2_cast_output.size_bytes = 852U;

      const auto extracted_rank2_bf16 =
          build_boxdecode_static_contract_from_mpk(rank2_bf16_mpk, make_flags(false, true), &error);
      require(extracted_rank2_bf16.has_value(),
              "BF16 rank-2 NC route should use resolved detess geometry: " + error);
      require(extracted_rank2_bf16->tensors[0].input_shape == std::vector<int>({1, 1, 213}),
              "BF16 rank-2 NC route should expose H=1, W=1, C=213");
      require(extracted_rank2_bf16->tensors[0].source_size_bytes == 448U,
              "BF16 rank-2 NC route should preserve the C16-packed source byte span");

      // Tess route: unpack publishes dense physical HWC and slice selects logical C=1.
      MpkContract dense_slice_mpk;
      MpkPluginIoContract dense_mla;
      dense_mla.name = "MLA_0";
      dense_mla.sequence = 1;
      dense_mla.processor = "MLA";
      dense_mla.kernel = "mla";
      dense_mla.canonical_output_dtype = "INT8";
      dense_mla.quant = MpkQuantContract{{0.5}, {3}, -1};
      dense_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 102400},
          .shape_semantics = MpkShapeSemantics::PackedExtent,
          .size_bytes = 102400,
      });
      dense_slice_mpk.plugins.push_back(std::move(dense_mla));

      MpkPluginIoContract dense_unpack;
      dense_unpack.name = "MLA_0_ofm_unpack_transform";
      dense_unpack.sequence = 2;
      dense_unpack.kernel = "ofm_unpack";
      dense_unpack.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "score_head_dense",
          .segment_name = "output_tensor",
          .dtype = "INT8",
          .mpk_shape = {1, 80, 80, 16},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 102400,
          .logical_shape = {1, 80, 80, 1},
      });
      dense_slice_mpk.plugins.push_back(std::move(dense_unpack));
      dense_slice_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 0U,
          .src_output_index = 0,
          .dst_plugin_index = 1U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0",
          .dst_plugin = "MLA_0_ofm_unpack_transform",
          .tensor_name = "output_tensor",
      });

      MpkPluginIoContract slice_stage;
      slice_stage.name = "slice_transform";
      slice_stage.sequence = 3;
      slice_stage.processor = "EV74";
      slice_stage.kernel = "slice_transform";
      slice_stage.slice_begin = {0, 0, 0, 0};
      slice_stage.input_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "score_head_dense",
          .dtype = "INT8",
          .mpk_shape = {1, 80, 80, 16},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 102400,
          .logical_shape = {1, 80, 80, 1},
      });
      slice_stage.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .name = "class_logit_0",
          .dtype = "INT8",
          .mpk_shape = {1, 80, 80, 1},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 6400,
          .logical_shape = {1, 80, 80, 1},
      });
      dense_slice_mpk.plugins.push_back(std::move(slice_stage));
      dense_slice_mpk.edges.push_back(MpkContractEdge{
          .src_plugin_index = 1U,
          .src_output_index = 0,
          .dst_plugin_index = 2U,
          .dst_input_index = 0,
          .src_plugin = "MLA_0_ofm_unpack_transform",
          .dst_plugin = "slice_transform",
          .tensor_name = "score_head_dense",
      });

      const auto dense_slice_flags =
          resolve_model_managed_boxdecode_route_flags_from_mpk(dense_slice_mpk, nullptr, &error);
      require(dense_slice_flags.has_value() && dense_slice_flags->quant_needed &&
                  dense_slice_flags->quant_contract_required,
              "dense INT8 slice route must select its exact MLA quant contract: " + error);
      const auto extracted_dense_slice = build_boxdecode_static_contract_from_mpk(
          dense_slice_mpk, *dense_slice_flags, &error);
      require(extracted_dense_slice.has_value(),
              "dense HWC slice route should extract storage facts: " + error);
      require(extracted_dense_slice->tensors[0].source_storage_kind ==
                  BoxDecodeSourceStorageKind::DenseHwcPhysical,
              "dense slice route should use DenseHwcPhysical storage");
      require(extracted_dense_slice->tensors[0].input_shape == std::vector<int>({80, 80, 16}),
              "dense slice route should preserve physical C stride");
      require(extracted_dense_slice->tensors[0].slice_shape == std::vector<int>({80, 80, 1}),
              "dense slice route should preserve logical sliced channels");
      require(extracted_dense_slice->tensors[0].source_size_bytes == 102400U,
              "dense slice route should preserve physical source byte span even when "
              "logical_shape is sliced");
      require(extracted_dense_slice->dq_scale == std::vector<double>({0.5}) &&
                  extracted_dense_slice->dq_zp == std::vector<std::int64_t>({3}),
              "dense INT8 slice route must preserve its exact MLA qparams");
      const auto extracted_dense_slice_subset = extract_boxdecode_contract_subset_from_mpk(
          dense_slice_mpk, *dense_slice_flags, nullptr, &error);
      require(extracted_dense_slice_subset.has_value(),
              "dense HWC slice subset should carry storage kind: " + error);
      require(extracted_dense_slice_subset->tensor_storage_kind[0] ==
                  static_cast<int>(BoxDecodeSourceStorageKind::DenseHwcPhysical),
              "dense HWC slice subset should expose DenseHwcPhysical=1");
      require(!extracted_dense_slice_subset->logical_inputs.empty() &&
                  extracted_dense_slice_subset->logical_inputs[0].size_bytes == 102400U,
              "dense HWC slice subset should publish physical source byte span");

      // AFE single-MLA packages can expose logical dense heads directly from
      // MLA with no unpack/detess stage. The exact shape*dtype byte count is an
      // authoritative dense-storage fact; packed rank-2 MLA carriers remain
      // rejected by dense_hwc_source_fact_from_mpk_tensor_local().
      MpkContract direct_dense_mpk;
      MpkPluginIoContract direct_dense_mla;
      direct_dense_mla.name = "MLA_0";
      direct_dense_mla.sequence = 1;
      direct_dense_mla.processor = "MLA";
      direct_dense_mla.kernel = "mla";
      direct_dense_mla.canonical_output_dtype = "BF16";
      direct_dense_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 0,
          .physical_index = 0,
          .name = "detector",
          .dtype = "BF16",
          .mpk_shape = {1, 60, 80, 65},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 60U * 80U * 65U * 2U,
          .logical_shape = {1, 60, 80, 65},
      });
      direct_dense_mla.output_tensors.push_back(MpkTensorContract{
          .tensor_index = 1,
          .physical_index = 1,
          .name = "descriptor",
          .dtype = "BF16",
          .mpk_shape = {1, 60, 80, 256},
          .shape_semantics = MpkShapeSemantics::Geometry,
          .size_bytes = 60U * 80U * 256U * 2U,
          .logical_shape = {1, 60, 80, 256},
      });
      direct_dense_mpk.plugins.push_back(std::move(direct_dense_mla));

      const auto extracted_direct_dense = build_boxdecode_static_contract_from_mpk(
          direct_dense_mpk, make_flags(false, false), &error);
      require(extracted_direct_dense.has_value(),
              "direct dense MLA route should extract storage facts: " + error);
      require(extracted_direct_dense->tensors.size() == 2U,
              "direct dense MLA route should preserve both heads");
      require(extracted_direct_dense->tensors[0].source_storage_kind ==
                      BoxDecodeSourceStorageKind::DenseHwcPhysical &&
                  extracted_direct_dense->tensors[1].source_storage_kind ==
                      BoxDecodeSourceStorageKind::DenseHwcPhysical,
              "direct dense MLA outputs should use DenseHwcPhysical storage");
      require(extracted_direct_dense->tensors[0].input_shape == std::vector<int>({60, 80, 65}) &&
                  extracted_direct_dense->tensors[1].input_shape == std::vector<int>({60, 80, 256}),
              "direct dense MLA outputs should preserve SuperPoint head geometry");
    }));
