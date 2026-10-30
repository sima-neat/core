// Exercise the real cold publication contract and InputStream materializer.
// No Graph/MLA, firmware or DMA allocation: numerical wire bytes are an
// independent fixture; existing recipe tests cover numerical serialization.
void oriented_publication_sample_bridge(BoxDecodeType type = BoxDecodeType::YoloV26Obb) {
  using namespace simaai::neat::pipeline_internal;
  constexpr std::size_t bytes = 96; // Header32 + two records32.
  // Reproduce the production lowering boundary: authored typed contract ->
  // checked kernel -> discard typed payload -> generic native-stage renderer.
  // A manually tagged StageStaticSpec bypasses the bug this test must catch.
  auto typed = source(type, 2);
  typed.payload.topk = 2;
  typed.payload.nms_iou_threshold = 0;
  typed.payload.decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
  for (int channels : {type == BoxDecodeType::YoloV8Obb ? 64 : 4, 2, 1})
    for (int side : {8, 4, 2})
      input(typed, side, side, channels, "FP32");
  for (std::size_t i = 0; i < typed.runtime_contract.logical_inputs.size(); ++i) {
    const auto& logical = typed.runtime_contract.logical_inputs[i];
    const auto span = logical.shape[0] * logical.shape[1] * logical.shape[2] * 4;
    typed.runtime_contract.physical_inputs.push_back(
        specbuilders::build_physical_buffer_static_spec(i, i, span, DeviceKind::Cpu, "head"));
    typed.runtime_contract.input_bindings.push_back(
        specbuilders::build_input_binding_static_spec(i, i, "head", "head"));
  }
  BoxDecodeGeometryOptions geometry{64, 64, 64, 64, ResizeMode::Letterbox};
  geometry.model_to_original = std::array<float, 4>{1, 1, 0, 0};
  ContractCompileInput context;
  CompiledNodeContract node;
  node.element_name = node.logical_stage_id = "obb_publication";
  node.node_kind = "SimaBoxDecode";
  node.boxdecode = typed;
  node.kernel = compile_boxdecode_kernel(*node.boxdecode, context, geometry);
  node.boxdecode.reset(); // Same ownership transition as SimaBoxDecode/ModelPack.
  CompiledPipelineContracts compiled;
  compiled.stages.push_back(node);
  compiled.fully_renderable = true;
  ManifestBuildDiagnostics diagnostics;
  auto manifest = render_manifest_from_compiled_contracts(compiled, context, &diagnostics);
  require(manifest && manifest->stages.size() == 1 && diagnostics.errors.empty(),
          "OBB canonical kernel render failed");
  auto prepared = prepare_kernel_stage_recipes(&*manifest, std::nullopt, {}, {}, {});
  require(!prepared, "OBB checked recipe preparation failed");
  finalize_pipeline_storage(*manifest);
  require(!finalize_stage_publications(&*manifest), "OBB final publication failed");
  const auto& stage = manifest->stages.front();
  require(stage.payload_kind == StagePayloadKind::None && stage.native_kernel &&
              stage.native_kernel->output_format == kDetectionFormatBboxObbV1 &&
              stage.frame_layout && stage.tensor_publication,
          "OBB wire semantics did not survive typed-payload retirement");
  std::string error;
  c::TensorStagePublication fresh = *stage.tensor_publication;
  auto generic_stage = stage;
  auto generic_recipe = std::make_shared<KernelStageRecipe>(*stage.native_kernel);
  generic_recipe->output_format.clear();
  generic_stage.native_kernel = generic_recipe;
  generic_stage.tensor_publication.reset();
  c::TensorStagePublication generic;
  require(build_kernel_stage_publication(generic_stage, &generic, &error), error.c_str());
  require(!gst_structure_get_string(gst_caps_get_structure(generic.output_caps.get(), 0), "format"),
          "OBB fix changed empty-format generic caps");
  c::TensorStagePublication cached;
  require(build_kernel_stage_publication(stage, &cached, &error), error.c_str());
  require(
      cached.output_caps == fresh.output_caps &&
          !gst_structure_get_string(gst_caps_get_structure(generic.output_caps.get(), 0), "format"),
      "cached OBB publication reconstructed or mutated shared caps");
  for (const auto* publication : {&fresh, &cached}) {
    const auto* format = gst_structure_get_string(
        gst_caps_get_structure(publication->output_caps.get(), 0), "format");
    require(format && std::string(format) == kDetectionFormatBboxObbV1,
            "OBB typed producer omitted wire format");
    const auto& public_contract = *publication->contract;
    require(public_contract.logical_outputs.size() == 1 &&
                public_contract.logical_outputs[0].shape == std::vector<std::int64_t>{bytes} &&
                public_contract.logical_outputs[0].stride_bytes == std::vector<std::int64_t>{1},
            "OBB publication changed rank-one descriptor");
    for (const bool copy_output : {false, true}) {
      InputStream::State state;
      state.opt.copy_output = copy_output;
      state.lifetime_token = std::make_shared<int>(0);
      for (unsigned count : {1U, 0U, 1U}) {
        auto buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
        // Host-owned stand-ins exercise metadata/materialization, not heap admission.
        for (const auto& allocation : stage.frame_layout->allocations) {
          auto* memory = gst_allocator_alloc(nullptr, allocation.size_bytes, nullptr);
          require(memory, "OBB fixture allocation failed");
          gst_buffer_append_memory(buffer.get(), memory);
        }
        std::array<std::uint32_t, bytes / 4> wire{};
        // Independent little-endian V1 wire fixture (no producer serializer).
        wire[0] = 0x3156424f;
        wire[1] = 0x00200001;
        wire[2] = bytes;
        wire[3] = count;
        wire[4] = 2;
        wire[5] = 32;
        wire[6] = 32;
        if (count) {
          for (unsigned i = 0; i < 6; ++i)
            wire[8 + i] = std::bit_cast<std::uint32_t>(
                std::array<float, 6>{12.f, 12.f, -32.f, 48.f, .25f, .75f}[i]);
          wire[14] = 3;
        }
        const auto& selected = public_contract.logical_outputs.front();
        GstMapInfo mapped{};
        auto* memory = gst_buffer_peek_memory(buffer.get(), selected.memory_index);
        require(gst_memory_map(memory, &mapped, GST_MAP_WRITE) && selected.byte_offset >= 0 &&
                    std::size_t(selected.byte_offset) <= mapped.size &&
                    bytes <= mapped.size - std::size_t(selected.byte_offset),
                "OBB fixture selected span invalid");
        std::memcpy(mapped.data + selected.byte_offset, wire.data(), bytes);
        gst_memory_unmap(memory, &mapped);
        simaai::gst::TensorBufferPreparedMetaTemplate metadata;
        require(simaai::gst::tensor_buffer_prepare_meta_template_from_contract(public_contract,
                                                                               &metadata, &error),
                error.c_str());
        auto frozen = simaai::gst::tensor_buffer_freeze_publication(metadata, &error);
        require(frozen &&
                    simaai::gst::tensor_buffer_attach_publication(buffer.get(), frozen, &error),
                error.c_str());
        auto input = std::shared_ptr<GstSample>(
            gst_sample_new(buffer.get(), publication->output_caps.get(), nullptr, nullptr),
            gst_sample_unref);
        auto sample = decode_sample_from_inputstream_state(state, input.get(), "OBB publication");
        require(sample.kind == SampleKind::TensorSet && sample.tensors.size() == 1,
                "OBB publication did not materialize one public Tensor");
        // Raw pull exposes Sample.tensors without Run.run normalization. Decode
        // that exact producer output first; never manually tag the fixture.
        const auto raw_decoded = decode_obb(sample.tensors);
        require(raw_decoded.size() == 1 &&
                    raw_decoded[0].geometry.shape == std::vector<std::int64_t>{count, 5},
                "raw pulled Sample tensors lost authored OBB semantics");
        auto tensors = tensors_from_sample(sample, true);
        require(tensors[0].shape == std::vector<std::int64_t>{bytes} &&
                    tensors[0].dtype == TensorDType::UInt8 && tensors[0].is_dense() &&
                    tensors[0].semantic.detection &&
                    tensors[0].semantic.detection->format == kDetectionFormatBboxObbV1,
                "OBB public tensor lost exact rank/dtype/format");
        const auto decoded = decode_obb(tensors);
        require(decoded.size() == 1 &&
                    decoded[0].geometry.shape == std::vector<std::int64_t>{count, 5},
                "OBB strict public decode failed");
        if (count) {
          auto map = decoded[0].geometry.storage->map(MapMode::Read);
          const auto* values = static_cast<const float*>(map.data);
          require(values[0] == 12 && values[2] == -32 && values[4] == .25f,
                  "OBB public geometry changed");
        }
        auto invalid = tensors[0];
        invalid.shape = {1, bytes};
        invalid.strides_bytes = {bytes, 1};
        bool rejected = false;
        try {
          (void)decode_obb_tensor(invalid);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "OBB strict decoder admitted rank-two carrier");
        invalid = tensors[0];
        invalid.semantic.detection.reset();
        rejected = false;
        try {
          (void)decode_obb_tensor(invalid);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "OBB strict decoder admitted untagged bytes");
      }
    }
  }
}
