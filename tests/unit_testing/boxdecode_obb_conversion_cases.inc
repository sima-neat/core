// Model-managed OBB reads checked native carriers without eager FP32 staging.
void oriented_carrier_conversions() {
  const auto take = []<class T>(c::Result<T> result) -> T {
    if (const auto* error = std::get_if<c::Error>(&result))
      throw std::runtime_error(error->field + ": " + error->message);
    return std::get<T>(std::move(result));
  };
  BoxDecodeGeometryOptions geometry{64, 64, 64, 64, ResizeMode::Letterbox};
  geometry.model_to_original = std::array<float, 4>{1, 1, 0, 0};
  require(1.0F / static_cast<float>(3.1) != static_cast<float>(1.0 / 3.1),
          "FP32 reciprocal ordering witness is vacuous");
  for (const std::string type : {"FP32", "BF16", "INT8"}) {
    // Exact registry unpack+slice depths, before any physical execution. The
    // public FP32 contract is not the MLA raw INT8/BF16 carrier contract.
    auto registry = source(BoxDecodeType::YoloV26Obb, 15);
    registry.payload.decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
    registry.payload.nms_iou_threshold = 0;
    for (int channels : {4, 15, 1})
      for (int side : {128, 64, 32}) {
        const int physical = type == "INT8"   ? 16
                             : type == "BF16" ? ((channels + 7) / 8) * 8
                                              : channels;
        input(registry, side, side, physical, type);
        registry.payload.slice_shapes.back().sizes[2] = channels;
      }
    auto registry_geometry = geometry;
    registry_geometry.model_width = registry_geometry.model_height = 1024;
    registry_geometry.original_width = registry_geometry.original_height = 1024;
    require(check(registry, registry_geometry).decode_member == 0U,
            "registry OBB raw carrier lowering introduced eager FP32 conversion");
    for (int layout = 0; layout < 6; ++layout) {
      if (layout == 5 && type == "FP32")
        continue; // Existing Detess deliberately admits only INT8/BF16.
      std::cerr << "DIAG_CASE type=" << type << " layout=" << layout << std::endl;
      auto s = source(BoxDecodeType::YoloV26Obb, 15);
      s.payload.decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
      s.payload.detection_threshold = .001;
      s.payload.nms_iou_threshold = 0;
      auto reference = s;
      std::vector<std::shared_ptr<HostStorage>> carriers, floats;
      std::vector<std::size_t> carrier_sizes;
      const std::size_t width = type == "INT8" ? 1 : type == "BF16" ? 2 : 4;
      for (int role = 0; role < 3; ++role) {
        const int channels = role == 0 ? 4 : role == 1 ? 15 : 1;
        for (int side : {8, 4, 2}) {
          const std::size_t port = carriers.size();
          const int physical_channels = layout == 3 || layout == 5 ? 16
                                        : layout == 4              ? ((channels + 7) / 8) * 8
                                                                   : channels;
          input(s, side, side, layout == 5 ? channels : physical_channels, type,
                layout == 5 ? 2 : 1, 16, layout == 2);
          s.payload.slice_shapes.back().sizes[2] = layout == 5 ? 16 : channels;
          auto& logical = s.runtime_contract.logical_inputs.back();
          const std::size_t pixel = physical_channels * width + (layout == 1 ? 4 : 0);
          const std::size_t row = side * pixel + (layout == 1 ? 32 : 0);
          const std::size_t plane = side * side * width;
          logical.stride_bytes =
              layout == 2
                  ? std::vector<std::int64_t>{std::int64_t(plane), std::int64_t(side * width),
                                              std::int64_t(width)}
                  : std::vector<std::int64_t>{std::int64_t(row), std::int64_t(pixel),
                                              std::int64_t(width)};
          const double divisor = role == 0   ? 37.55733764560728
                                 : role == 1 ? 4.1070431934107985
                                             : 3.1;
          const int zp = role == 0 ? -128 : role == 1 ? 115 : 0;
          if (type == "INT8")
            logical.quant = QuantStaticSpec{QuantGranularity::PerTensor, -1, {divisor}, {zp}};
          const std::size_t bytes = 16 + (layout == 2 ? channels * plane : side * row);
          auto storage = std::make_shared<HostStorage>(bytes + 128);
          std::memset(storage->memory.get(), 0xa5, storage->size);
          auto fp = std::make_shared<HostStorage>(side * side * channels * 4 + 128);
          std::memset(fp->memory.get(), 0xa5, fp->size);
          input(reference, side, side, channels, "FP32", 1, 0);
          for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x)
              for (int ch = 0; ch < channels; ++ch) {
                const int cell = y * side + x;
                const std::size_t at =
                    64 + 16 +
                    (layout == 2 ? ch * plane + cell * width : y * row + x * pixel + ch * width);
                float value = role == 0   ? 1.f + float((cell + ch) % 7) * .25f
                              : role == 1 ? float((cell + ch) % 13 - 6) * .25f
                                          : float(cell % 3 - 1) * .25f;
                if (type == "INT8") {
                  const auto q = std::int8_t(role == 0   ? -127 + (cell + ch) % 15
                                             : role == 1 ? 109 + (cell + ch) % 13
                                                         : cell % 3 - 1);
                  std::memcpy(static_cast<std::byte*>(storage->memory.get()) + at, &q, 1);
                  const float scale = 1.0F / static_cast<float>(divisor);
                  const float centered = float(q) - float(zp);
                  value = centered * scale;
                } else if (type == "BF16") {
                  const auto bits = std::uint16_t(std::bit_cast<std::uint32_t>(value) >> 16);
                  if (layout == 5) {
                    // MLA BF16 pixel-major storage has two byte planes per
                    // padded C16 pixel, unlike dense padded BF16 words.
                    auto* bytes = static_cast<std::uint8_t*>(storage->memory.get());
                    const auto pixel_base = 64 + 16 + std::size_t(cell) * 32;
                    bytes[pixel_base + ch] = std::uint8_t(bits);
                    bytes[pixel_base + 16 + ch] = std::uint8_t(bits >> 8);
                  } else {
                    std::memcpy(static_cast<std::byte*>(storage->memory.get()) + at, &bits, 2);
                  }
                  value = std::bit_cast<float>(std::uint32_t(bits) << 16);
                } else {
                  std::memcpy(static_cast<std::byte*>(storage->memory.get()) + at, &value, 4);
                }
                std::memcpy(static_cast<std::byte*>(fp->memory.get()) + 64 +
                                (cell * channels + ch) * 4,
                            &value, 4);
              }
          carriers.push_back(std::move(storage));
          floats.push_back(std::move(fp));
          carrier_sizes.push_back(bytes);
          require(port + 1 == carriers.size(), "carrier ordering drift");
        }
      }
      const auto prepared = check(s, geometry);
      const auto expected = check(reference, geometry);
      const auto& recipe = *prepared.compiled.recipe;
      require(prepared.decode_member == (layout == 5 ? 9U : 0U),
              "OBB native carrier route has unexpected layout conversion members");
      for (std::size_t i = 0; i < 9; ++i) {
        require(prepared.compiled.runtime_contract.logical_inputs[i].dtype == type &&
                    prepared.compiled.runtime_contract.logical_inputs[i].byte_offset == 16,
                "OBB conversion changed source carrier authority");
        const auto decoded = inspect(recipe.members[prepared.decode_member].executable);
        const auto expected_dtype = type == "INT8" ? c::DType::Int8
                                    : type == "BF16" ? c::DType::BFloat16
                                                     : c::DType::Float32;
        require(decoded.inputs()[i].dtype == expected_dtype,
                "OBB native carrier dtype was retagged or eagerly converted");
        if (layout == 5) {
          const auto& conversion = recipe.members[i];
          const auto converted = inspect(conversion.executable);
          require(converted.inputs()[0].dtype == expected_dtype &&
                      converted.outputs()[0].dtype == expected_dtype &&
                      conversion.inputs[0].source == c::TensorStagePortSource::Input &&
                      recipe.members[prepared.decode_member].inputs[i].tensor ==
                          conversion.outputs[0].tensor,
                  "OBB tiled carrier must retain its checked dtype-preserving Detess");
        } else {
          require(recipe.members[prepared.decode_member].inputs[i].source ==
                      c::TensorStagePortSource::Input,
                  "OBB linear carrier introduced a private staging dependency");
        }
      }
      const auto run = [&](const BoxDecodeKernelRecipeResult& compiled,
                           const std::vector<std::shared_ptr<HostStorage>>& source_storage,
                           bool bounds_check, bool expect_failure = false) {
        const auto& r = *compiled.compiled.recipe;
        std::vector<std::shared_ptr<HostStorage>> outputs(r.private_buffers.size() + 1);
        outputs[0] = std::make_shared<HostStorage>(
            compiled.compiled.runtime_contract.logical_outputs[0].size_bytes + 128);
        for (const auto& b : r.private_buffers)
          outputs[b.physical_index] = std::make_shared<HostStorage>(b.size_bytes + 128);
        for (const auto& b : outputs)
          std::memset(b->memory.get(), 0xa5, b->size);
        std::size_t maximum_dependencies = 0;
        for (const auto& member : r.members)
          maximum_dependencies = std::max(maximum_dependencies, member.dependencies.size());
        auto domain = std::make_shared<test::Domain>(nullptr, maximum_dependencies);
        test::Options probe_options;
        probe_options.domain = domain;
        std::vector<test::Completion> completed;
        std::vector<std::unique_ptr<test::OperationProbe>> probes;
        for (std::size_t member = 0; member < r.members.size(); ++member) {
          const auto& m = r.members[member];
          auto probe = std::make_unique<test::OperationProbe>();
          require(!probe->prepare(m.executable, probe_options), "OBB checked member preparation failed");
          const auto contract = inspect(m.executable);
          std::vector<c::ConstBytes> in;
          std::vector<c::MutableBytes> out;
          for (const auto& p : m.inputs) {
            const auto& buffer = p.source == c::TensorStagePortSource::Input
                                     ? source_storage[p.tensor]
                                     : outputs[p.tensor];
            const auto offset =
                p.byte_offset +
                (p.source == c::TensorStagePortSource::Input
                     ? compiled.compiled.runtime_contract.logical_inputs[p.tensor].byte_offset
                     : 0);
            in.push_back({static_cast<const std::byte*>(buffer->memory.get()) + 64 + offset,
                          buffer->size - 128 - offset});
          }
          for (const auto& p : m.outputs) {
            const auto& buffer = outputs[p.tensor];
            out.push_back({static_cast<std::byte*>(buffer->memory.get()) + 64 + p.byte_offset,
                           buffer->size - 128 - p.byte_offset});
          }
          if (bounds_check && member == 0) {
            auto short_input = in;
            short_input[0].size_bytes = contract.input_requirements()[0].storage_bytes - 1;
            require(std::holds_alternative<c::Error>(
                        probe->bind_host(probe->operation(), short_input, out)),
                    "OBB conversion accepted undersized source storage");
          }
          auto binding = take(probe->bind_host(probe->operation(), in, out));
          std::vector<test::Completion> dependencies;
          for (auto d : m.dependencies)
            dependencies.push_back(std::move(completed.at(d)));
          if (type == "BF16" && layout == 5 && !expect_failure &&
              &source_storage == &carriers && member == compiled.decode_member) {
            require(maximum_dependencies == 9 && dependencies.size() == 9 &&
                        m.dependencies.size() == 9,
                    "OBB nine-predecessor capacity witness changed");
            // Preserve the original default-eight failure as an explicit
            // negative, and verify the same bound through the new overload.
            for (bool explicit_budget : {false, true}) {
              test::OperationProbe too_small;
              test::Options limited;
              if (explicit_budget)
                limited.domain = std::make_shared<test::Domain>(nullptr, maximum_dependencies - 1);
              require(!too_small.prepare(m.executable, limited), "capacity negative preparation");
              auto limited_binding = take(too_small.bind_host(too_small.operation(), in, out));
              const auto rejected = too_small.submit(limited_binding, dependencies).wait();
              require(rejected && rejected->field == "dependencies" &&
                          rejected->message == "dependency capacity exceeded",
                      "bounded dependency context did not reject nine-over-eight fan-in");
            }
            bool bounded = false;
            try { (void)std::make_shared<test::Domain>(nullptr, 65); }
            catch (const std::invalid_argument&) { bounded = true; }
            require(bounded, "test-only dependency constructor lost its explicit bound");
            std::cerr << "NINE_DEPENDENCY_DEFAULT_AND_EXPLICIT_EIGHT_REJECTION_PASS" << std::endl;
          }
          auto done = probe->submit(binding, dependencies);
          const auto status = done.wait();
          if (bool(status) != (expect_failure && member >= compiled.decode_member)) {
            std::cerr << "DIAG_WAIT type=" << type << " layout=" << layout
                      << " source=" << (&source_storage == &carriers ? "actual" : "reference")
                      << " member=" << member << " decode_member=" << compiled.decode_member
                      << " expected_failure=" << expect_failure
                      << " operation=" << (member == compiled.decode_member ? "boxdecode" :
                          member > compiled.decode_member ? "publication" :
                          contract.outputs()[0].dtype != c::DType::Float32 ? "detess" :
                          type == "INT8" ? "dequant" : "cast") << std::endl;
            if (status) std::cerr << "DIAG_ERROR code=" << int(status->code)
                                  << " field=" << status->field
                                  << " message=" << status->message << std::endl;
            else std::cerr << "DIAG_ERROR success where failure expected" << std::endl;
          }
          require(bool(status) == (expect_failure && member >= compiled.decode_member),
                  "OBB conversion/decode/publication completion mismatch");
          for (std::size_t d = 0; d < m.dependencies.size(); ++d)
            completed.at(m.dependencies[d]) = std::move(dependencies[d]);
          completed.push_back(std::move(done));
          probes.push_back(std::move(probe));

        }
        for (const auto& b : outputs)
          for (std::size_t i = 0; i < 64; ++i)
            require(b->read<std::uint8_t>(i) == 0xa5 &&
                        b->read<std::uint8_t>(b->size - 64 + i) == 0xa5,
                    "OBB conversion/publication output canary changed");
        if (expect_failure)
          for (std::size_t i = 64; i < outputs[0]->size - 64; ++i)
            require(outputs[0]->read<std::uint8_t>(i) == 0xa5,
                    "OBB malformed converted angle published a wire buffer");
        return outputs[0];
      };
      std::vector<std::vector<std::byte>> source_snapshots;
      for (const auto& b : carriers)
        source_snapshots.emplace_back(static_cast<const std::byte*>(b->memory.get()),
                                      static_cast<const std::byte*>(b->memory.get()) + b->size);
      const auto actual = run(prepared, carriers, true);
      require(actual->read<std::uint32_t>(64 + 12) == 4,
              "OBB conversion wire comparison had no populated detections");
      for (std::size_t i = 0; i < carriers.size(); ++i)
        require(std::memcmp(carriers[i]->memory.get(), source_snapshots[i].data(),
                            carriers[i]->size) == 0,
                "OBB conversion modified raw source bytes");
      const auto oracle = run(expected, floats, false);
      require(actual->size == oracle->size &&
                  std::memcmp(actual->memory.get(), oracle->memory.get(), actual->size) == 0,
              "native OBB wire differs from ordered scalar FP32 reference recipe");
      for (const auto& b : carriers)
        for (std::size_t i = 0; i < 64; ++i)
          require(b->read<std::uint8_t>(i) == 0xa5 &&
                      b->read<std::uint8_t>(b->size - 64 + i) == 0xa5,
                  "OBB conversion modified source canaries");
      if (type == "BF16" && layout == 0) {
        const std::uint16_t nan = 0x7fc0;
        std::memcpy(static_cast<std::byte*>(carriers[8]->memory.get()) + 64 + 16, &nan, 2);
        (void)run(prepared, carriers, false, true);
      }
      if (type == "INT8" && layout == 0) {
        for (int fault = 0; fault < 6; ++fault) {
          auto bad = s;
          auto& q = bad.runtime_contract.logical_inputs[0].quant;
          if (fault == 0)
            q.reset();
          if (fault == 1)
            q->scales = {0.};
          if (fault == 2)
            q->zero_points = {128};
          if (fault == 3)
            q->scales = {std::numeric_limits<double>::infinity()};
          if (fault == 4)
            bad.payload.slice_shapes[0].sizes[2] = 3;
          if (fault == 5)
            bad.payload.tensor_storage_kind[0] = 0; // No checked C-block converter exists.
          bool rejected = false;
          try {
            (void)check(bad, geometry);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          require(rejected, "malformed OBB conversion contract accepted");
        }
      }
    }
  }
  // SEG retains its native INT8 route; this fix must not insert conversion copies there.
  auto seg = source(BoxDecodeType::YoloV26Seg, 1);
  seg.payload.nms_iou_threshold = 0;
  for (int side : {80, 40, 20}) {
    input(seg, side, side, 4, "INT8");
    input(seg, side, side, 1, "INT8");
    input(seg, side, side, 32, "INT8");
  }
  input(seg, 160, 160, 32, "INT8");
  require(check(seg, {640, 640, 640, 640, ResizeMode::Letterbox}).decode_member == 0,
          "OBB conversion changed native SEG route");
}
