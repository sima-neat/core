// Included by unit_boxdecode_kernel_recipe_test.cpp, inside its anonymous namespace.

struct RoiFrame {
  int original_width = 1280, original_height = 720, canvas = 640;
  std::array<std::int64_t, 4> primary{0, 0, 1280, 720}, placed{0, 140, 640, 360};
  std::array<double, 6> affine{2., 0., 0., 0., 2., -280.};
  const char* mode = "letterbox";
  bool roi_list = false;
  std::int64_t capacity = 1;
};
// 1280x720 full frame, centered letterbox into 640x640: scale 2, pad top 140.
RoiFrame full_letterbox() {
  return {};
}
// The issue's ROI: x=400 y=200 w=640 h=360 letterboxed into 640x640 at scale 1, pad top 140.
RoiFrame roi_letterbox() {
  RoiFrame frame;
  frame.primary = {400, 200, 640, 360};
  frame.affine = {1., 0., 400., 0., 1., 60.};
  return frame;
}
// The same ROI stretched into 640x640: scale (1, 0.5625), offset (400, 200).
RoiFrame roi_stretch() {
  RoiFrame frame;
  frame.mode = "stretch";
  frame.primary = {400, 200, 640, 360};
  frame.placed = {0, 0, 640, 640};
  frame.affine = {1., 0., 400., 0., 0.5625, 200.};
  return frame;
}
RoiFrame full_stretch() {
  RoiFrame frame;
  frame.mode = "stretch";
  frame.placed = {0, 0, 640, 640};
  frame.affine = {2., 0., 0., 0., 1.125, 0.};
  return frame;
}
test::GstBufferOwner roi_carrier(const RoiFrame& frame) {
  namespace tr = simaai::gst;
  test::GstBufferOwner buffer(gst_buffer_new_allocate(nullptr, 4, nullptr));
  simaai::neat::test_support::attach_compact_frame_carrier(buffer.get());
  tr::TensorBufferPreprocessFacts facts;
  facts.original_width = frame.original_width;
  facts.original_height = frame.original_height;
  facts.resized_width = facts.resized_height = frame.canvas;
  facts.primary = frame.primary;
  facts.placed = frame.placed;
  facts.affine = frame.affine;
  facts.source_stride = std::int64_t(frame.original_width) * 3;
  facts.roi_list = frame.roi_list;
  facts.capacity = frame.capacity;
  if (frame.roi_list) {
    // One ROI per slot, each with its own affine; every slot repeats this frame's ROI here.
    facts.has_roi = true;
    facts.counts = {std::uint64_t(frame.capacity), std::uint64_t(frame.capacity), 0, 0};
    for (std::int64_t slot = 0; slot < frame.capacity; ++slot) {
      facts.roi_batch_indices.push_back(0);
      facts.roi_rects.insert(facts.roi_rects.end(), frame.primary.begin(), frame.primary.end());
      facts.roi_affines.insert(facts.roi_affines.end(), frame.affine.begin(), frame.affine.end());
    }
  }
  facts.resize_mode = frame.mode;
  facts.color_in = facts.color_out = "RGB";
  std::string error;
  tr::TensorBufferFrameMetadata metadata;
  metadata.preprocess = tr::tensor_buffer_freeze_preprocess_facts(std::move(facts), &error);
  require(bool(metadata.preprocess), error.c_str());
  require(tr::tensor_buffer_attach_frame_metadata(buffer.get(), std::move(metadata), &error),
          error.c_str());
  return buffer;
}
// The head tensors a profile decodes, with their physical carriers.
CompiledBoxDecodeContract roi_heads(CompiledBoxDecodeContract heads) {
  for (auto& logical : heads.runtime_contract.logical_inputs) {
    const std::int64_t width = logical.dtype == "FP32" ? 4 : logical.dtype == "INT8" ? 1 : 2;
    std::int64_t bytes = width;
    for (const auto extent : logical.shape)
      bytes *= extent;
    heads.runtime_contract.physical_inputs.push_back(
        specbuilders::build_physical_buffer_static_spec(logical.physical_index,
                                                        logical.physical_index, bytes,
                                                        DeviceKind::Cpu, logical.segment_name));
    InputBindingStaticSpec binding;
    binding.local_logical_input_index = logical.logical_index;
    binding.src_physical_output_index = logical.physical_index;
    heads.runtime_contract.input_bindings.push_back(std::move(binding));
  }
  return heads;
}
// The compiled Preproc that precedes BoxDecode: 1280x720 RGB into 640x640.
CompiledNodeContract roi_preproc(ResizeMode mode) {
  PreprocOptions options;
  options.set_input_shape({720, 1280, 3});
  options.input_img_type = options.output_img_type = "RGB";
  options.set_output_shape({640, 640, 3});
  options.resize_mode = mode;
  options.aspect_ratio = mode == ResizeMode::Letterbox;
  options.scaled_width = mode == ResizeMode::Letterbox ? 640 : 640;
  options.scaled_height = mode == ResizeMode::Letterbox ? 360 : 640;
  options.output_dtype = "BF16";
  options.tessellate = false;
  options.single_output_handoff = true;
  CompiledNodeContract producer;
  std::string error;
  require(Preproc(options).compile_node_contract({}, &producer, &error), error.c_str());
  return producer;
}
// One admission: the resolver the Scheduler instantiates per run, and the cold executable.
struct RoiRun {
  test::OperationProbe executor;
  std::shared_ptr<const KernelStageRecipe> recipe;
  KernelStageRecipe::Resolver resolve;
};
std::unique_ptr<RoiRun> admit_roi(const CompiledBoxDecodeContract& heads, ResizeMode preproc_mode,
                                  const BoxDecodeGeometryOptions& geometry,
                                  std::optional<ResizeMode> resize = std::nullopt) {
  auto preproc = std::make_shared<CompiledNodeContract>(roi_preproc(preproc_mode));
  auto producer = std::make_shared<CompiledNodeContract>();
  producer->logical_stage_id = "roi_head_producer";
  producer->kernel.emplace();
  producer->kernel->runtime_contract.physical_outputs = heads.runtime_contract.physical_inputs;
  ContractCompileInput context;
  context.immediate_upstream = producer.get();
  context.upstream_preprocess = &*preproc->processcvu;
  auto admitted = compile_boxdecode_kernel(heads, context, geometry, resize);
  auto run = std::make_unique<RoiRun>();
  run->recipe = admitted.recipe;
  require(run->recipe->resolver_factories.size() == 1,
          "an admitted BoxDecode must install its per-frame geometry resolver");
  require(!run->executor.prepare(run->recipe->members[0].executable),
          "cold BoxDecode preparation failed");
  run->resolve = run->recipe->resolver_factories.front().instantiate();
  return run;
}
c::Result<c::PreparedSelection> select_frame(RoiRun& run, GstBuffer* buffer) {
  const std::array<GstBuffer*, 1> carriers{buffer};
  return run.resolve(run.executor.preparer(), carriers);
}
bool is_error(const c::Result<c::PreparedSelection>& result, const char* fragment = "") {
  const auto* error = std::get_if<c::Error>(&result);
  return error && error->message.find(fragment) != std::string::npos;
}
struct RoiBox {
  std::int32_t count = 0, x = 0, y = 0, w = 0, h = 0;
  bool operator==(const RoiBox&) const = default;
};
// Runs the selected numerical operation on `fill`ed heads and reads the first published box.
RoiBox execute_roi(RoiRun& run, const c::PreparedSelection& selected,
                   const std::function<void(std::vector<std::shared_ptr<HostStorage>>&)>& fill,
                   std::vector<std::uint8_t>* masks = nullptr) {
  const auto description = inspect(selected.executable());
  std::vector<std::shared_ptr<HostStorage>> inputs, outputs;
  std::vector<c::ConstBytes> in, publication_inputs;
  std::vector<c::MutableBytes> out;
  for (const auto& req : description.input_requirements()) {
    inputs.push_back(std::make_shared<HostStorage>(req.storage_bytes));
    in.push_back({static_cast<const std::byte*>(inputs.back()->memory.get()), req.storage_bytes});
  }
  fill(inputs);
  for (const auto& req : description.output_requirements()) {
    outputs.push_back(std::make_shared<HostStorage>(req.storage_bytes));
    auto* data = static_cast<std::byte*>(outputs.back()->memory.get());
    out.push_back({data, req.storage_bytes});
    publication_inputs.push_back({data, req.storage_bytes});
  }
  require(!run.executor.execute_host(selected, in, out), "BoxDecode numerical execution failed");
  const auto& wire_operation = run.recipe->members.back().executable;
  // Direct masks (YOLO26) are not publication inputs.
  publication_inputs.resize(inspect(wire_operation).input_requirements().size());
  if (masks) {
    const auto* raw = static_cast<const std::uint8_t*>(outputs.back()->memory.get());
    masks->assign(raw, raw + outputs.back()->size);
  }
  test::OperationProbe publisher;
  require(!publisher.prepare(wire_operation), "BoxDecode publication preparation failed");
  const auto bytes = inspect(wire_operation).output_requirements()[0].storage_bytes;
  HostStorage wire(bytes);
  const std::array<c::MutableBytes, 1> wire_out{
      {{static_cast<std::byte*>(wire.memory.get()), bytes}}};
  require(!publisher.execute_host(publisher.operation(), publication_inputs, wire_out),
          "BoxDecode publication failed");
  RoiBox box;
  box.count = wire.read<std::int32_t>(0);
  if (box.count > 0) {
    box.x = wire.read<std::int32_t>(4);
    box.y = wire.read<std::int32_t>(8);
    box.w = wire.read<std::int32_t>(12);
    box.h = wire.read<std::int32_t>(16);
  }
  return box;
}
void set_bf16(HostStorage& storage, std::size_t element, float value) {
  const auto bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
  std::memcpy(static_cast<std::byte*>(storage.memory.get()) + element * 2, &bits, 2);
}
const auto expect_box = [](const RoiBox& got, const RoiBox& want, const char* message) {
  if (!(got == want))
    throw std::runtime_error(std::string(message) + ": got count " + std::to_string(got.count) +
                             " box " + std::to_string(got.x) + "," + std::to_string(got.y) + "," +
                             std::to_string(got.w) + "," + std::to_string(got.h) + ", expected " +
                             std::to_string(want.count) + " box " + std::to_string(want.x) + "," +
                             std::to_string(want.y) + "," + std::to_string(want.w) + "," +
                             std::to_string(want.h));
};

// Decoded (BoxDecodeType::Yolo) centered letterbox: one box (cx,cy,w,h)=(324,324,16,16) in model
// pixels. Independent oracle: the ROI frame maps it by (x + 400, y + 60) to 716,376,16,16; the
// whole frame at scale 2 with pad top 140 maps it to 632,352,32,32 (what the size-only route
// produced for the ROI frame as well). Cache, rejection and metadata-format variations live in
// unit_boxdecode_geometry_test.
void roi_decoded_letterbox() {
  auto heads = source(BoxDecodeType::Yolo, 2);
  heads.payload.topk = 4;
  input(heads, 8, 8, 4);
  input(heads, 8, 8, 2);
  heads = roi_heads(std::move(heads));
  const BoxDecodeGeometryOptions geometry{640, 640, 0, 0, ResizeMode::Letterbox};
  auto run = admit_roi(heads, ResizeMode::Letterbox, geometry);
  const auto fill = [](std::vector<std::shared_ptr<HostStorage>>& inputs) {
    for (std::size_t i = 0; i < 8 * 8 * 2; ++i)
      set_bf16(*inputs[1], i, -80.f);
    const float box[4] = {324.f, 324.f, 16.f, 16.f};
    for (int i = 0; i < 4; ++i)
      set_bf16(*inputs[0], i, box[i]);
    set_bf16(*inputs[1], 0, 8.f);
  };
  const auto decode = [&](const RoiFrame& frame) {
    auto carrier = roi_carrier(frame);
    auto selected = select_frame(*run, carrier.get());
    require(std::holds_alternative<c::PreparedSelection>(selected),
            "an ordinary frame was rejected");
    return execute_roi(*run, std::get<c::PreparedSelection>(selected), fill);
  };
  expect_box(decode(roi_letterbox()), {1, 716, 376, 16, 16},
             "an ROI letterbox frame was decoded as if the whole frame had been resized");
  // Full frames keep the boxes they have always had.
  expect_box(decode(full_letterbox()), {1, 632, 352, 32, 32}, "full-frame letterbox changed");
}

// DETR stretch: normalized cxcywh (.5,.5,.5,.5). The ROI frame (scale 1, .5625, offset 400,200)
// maps it to 560,290,320,180; the whole frame to 320,180,640,360.
void roi_detr_stretch() {
  auto heads = detr_geometry_heads(BoxDecodeType::Detr);
  heads.payload.topk = 2;
  heads.payload.detection_threshold = .6;
  const BoxDecodeGeometryOptions geometry{640, 640, 0, 0, ResizeMode::Stretch};
  auto run = admit_roi(heads, ResizeMode::Stretch, geometry);
  const auto fill = [](std::vector<std::shared_ptr<HostStorage>>& inputs) {
    // Raw DETR box logits: sigmoid(0) = .5, so every query is normalized cxcywh (.5, .5, .5, .5).
    for (std::size_t i = 0; i < 8; ++i)
      inputs[0]->set_float(i, 0.f);
    for (std::size_t i = 0; i < 6; ++i)
      inputs[1]->set_float(i, i % 3 == 0 ? std::log(4.f) : 0.f);
  };
  const auto decode = [&](const RoiFrame& frame) {
    auto carrier = roi_carrier(frame);
    auto selected = select_frame(*run, carrier.get());
    require(std::holds_alternative<c::PreparedSelection>(selected),
            "an ordinary DETR frame was rejected");
    return execute_roi(*run, std::get<c::PreparedSelection>(selected), fill);
  };
  expect_box(decode(roi_stretch()), {2, 560, 290, 320, 180},
             "a stretched ROI frame was decoded as if the whole frame had been stretched");
  expect_box(decode(full_stretch()), {2, 320, 180, 640, 360}, "full-frame stretch changed");
}

// RCNN proposals feed a second stage whose coordinate contract is not settled: the kernel accepts
// no transform for them, whatever its release. Admission installs the guard: a whole frame decodes,
// an ROI frame fails naming the profile, a requested Crop fails at build. The per-frame rules
// themselves are covered by unit_boxdecode_geometry_test.
void roi_no_transform_profiles() {
  auto heads = source(BoxDecodeType::RcnnStage1, 1);
  for (int side : {136, 68, 34, 17, 9}) {
    input(heads, side, side, 12);
    input(heads, side, side, 3);
  }
  heads = roi_heads(std::move(heads));
  const BoxDecodeGeometryOptions geometry{544, 544, 0, 0, ResizeMode::Letterbox};
  const std::string token = box_decode_type_token(BoxDecodeType::RcnnStage1);
  RoiFrame whole = full_letterbox();
  RoiFrame roi = roi_letterbox();
  whole.canvas = roi.canvas = 544;
  whole.placed = {0, 112, 544, 306};
  whole.affine = {1280. / 544, 0., 0., 0., 1280. / 544, -112. * 1280. / 544};
  roi.placed = {0, 112, 544, 306};
  roi.affine = {640. / 544, 0., 400., 0., 640. / 544, 200. - 112. * 640. / 544};
  auto run = admit_roi(heads, ResizeMode::Letterbox, geometry);
  auto carrier = roi_carrier(whole);
  require(std::holds_alternative<c::PreparedSelection>(select_frame(*run, carrier.get())),
          "RCNN: a whole frame was rejected");
  carrier = roi_carrier(roi);
  require(is_error(select_frame(*run, carrier.get()), token.c_str()),
          "RCNN: an ROI frame decoded in full-frame coordinates");
  bool rejected = false;
  try {
    auto producer = std::make_shared<CompiledNodeContract>();
    producer->logical_stage_id = "roi_head_producer";
    producer->kernel.emplace();
    producer->kernel->runtime_contract.physical_outputs = heads.runtime_contract.physical_inputs;
    ContractCompileInput context;
    context.immediate_upstream = producer.get();
    (void)compile_boxdecode_kernel(heads, context, geometry, ResizeMode::Crop);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find(token) != std::string::npos ||
               std::string(error.what()).find("crop") != std::string::npos;
  }
  require(rejected, "RCNN: a requested Crop was not refused");
}

// A ROI frame whose model-to-source scale is not 1: ROI 100,50,320,180 letterboxed into 640x640
// (x2, pad top 140): source = model * 0.5 + (100, -20).
RoiFrame roi_half_letterbox() {
  RoiFrame frame;
  frame.primary = {100, 50, 320, 180};
  frame.affine = {.5, 0., 100., 0., .5, -20.};
  return frame;
}
// Pose and segmentation (Kernels with the transform for them): the box moves with the ROI affine.
// Oracle: the whole frame (scale 2, pad top 140) decodes the same model-space box, so its model
// box is ((x), (y + 280)) / 2 and every ROI frame must report model * scale + offset. Masks stay in
// model space: they are byte-identical for every frame.
void roi_pose_seg_known_point() {
  struct Case {
    BoxDecodeType type;
    CompiledBoxDecodeContract heads;
    bool masks;
  };
  std::vector<Case> cases;
  {
    auto s = source(BoxDecodeType::YoloV26Pose, 1);
    for (int n : {80, 40, 20}) {
      input(s, n, n, 4, "FP32");
      input(s, n, n, 1, "FP32");
      input(s, n, n, 51, "FP32");
    }
    cases.push_back({BoxDecodeType::YoloV26Pose, roi_heads(std::move(s)), false});
  }
  {
    auto s = source(BoxDecodeType::YoloV26Seg, 1);
    s.payload.nms_iou_threshold = 0.;
    for (int n : {80, 40, 20}) {
      input(s, n, n, 4, "FP32");
      input(s, n, n, 1, "FP32");
      input(s, n, n, 32, "FP32");
    }
    input(s, 160, 160, 32, "FP32");
    cases.push_back({BoxDecodeType::YoloV26Seg, roi_heads(std::move(s)), true});
  }
  const BoxDecodeGeometryOptions geometry{640, 640, 0, 0, ResizeMode::Letterbox};
  for (auto& item : cases) {
    const std::string token = box_decode_type_token(item.type);
    const bool masks = item.masks;
    auto run = admit_roi(item.heads, ResizeMode::Letterbox, geometry);
    const auto fill = [masks](std::vector<std::shared_ptr<HostStorage>>& inputs) {
      for (int port : {1, 4, 7})
        for (std::size_t i = 0; i < inputs[port]->size / 4; ++i)
          inputs[port]->set_float(i, -80.f);
      // The first level's anchor at row 40, column 40 centers on model pixel (324, 324).
      constexpr std::size_t anchor = 40 * 80 + 40;
      for (int coordinate = 0; coordinate < 4; ++coordinate)
        inputs[0]->set_float(anchor * 4 + coordinate, 1.f);
      inputs[1]->set_float(anchor, 2.f);
      if (masks) {
        inputs[2]->set_float(anchor * 32, 1.f);
        for (std::size_t pixel = 0; pixel < 160 * 160; ++pixel)
          inputs.back()->set_float(pixel * 32, 1.f);
      }
    };
    std::vector<std::uint8_t> whole_masks, first_masks, second_masks;
    const auto decode = [&](const RoiFrame& frame, std::vector<std::uint8_t>* out) {
      auto carrier = roi_carrier(frame);
      auto selected = select_frame(*run, carrier.get());
      require(
          std::holds_alternative<c::PreparedSelection>(selected),
          (token + ": an ROI frame was refused although the kernel takes the transform").c_str());
      return execute_roi(*run, std::get<c::PreparedSelection>(selected), fill, out);
    };
    const auto whole = decode(full_letterbox(), &whole_masks);
    require(whole.count == 1 && whole.w > 0 && whole.h > 0,
            (token + ": the whole frame did not decode one box").c_str());
    const auto model = [&](int whole_value, int offset) { return (whole_value + offset) / 2.; };
    const auto near = [&](const RoiBox& got, double sx, double sy, double ox, double oy) {
      const double want[4] = {model(whole.x, 0) * sx + ox, model(whole.y, 280) * sy + oy,
                              model(whole.w, 0) * sx, model(whole.h, 0) * sy};
      const std::int32_t have[4] = {got.x, got.y, got.w, got.h};
      for (int i = 0; i < 4; ++i)
        if (got.count != 1 || std::abs(have[i] - want[i]) > 2.5)
          throw std::runtime_error(token + ": ROI box " + std::to_string(i) + " is " +
                                   std::to_string(have[i]) + ", expected " +
                                   std::to_string(want[i]));
    };
    near(decode(roi_letterbox(), &first_masks), 1., 1., 400., 60.);
    near(decode(roi_half_letterbox(), &second_masks), .5, .5, 100., -20.);
    if (masks) {
      require(std::any_of(whole_masks.begin(), whole_masks.end(), [](auto v) { return v != 0; }),
              (token + ": the mask fixture decoded an empty mask").c_str());
      require(whole_masks == first_masks && whole_masks == second_masks,
              (token + ": masks changed with the ROI; they stay in model space").c_str());
    }
  }
}

// Which profiles take the preprocess transform: ordinary detection yes; multiclass SSD shares
// BoxDecodeType::Ssd with the detection recipe, but its kernel profile accepts none.
void roi_transform_support() {
  BoxDecodeStagePayload payload;
  payload.decode_type = BoxDecodeType::Ssd;
  payload.num_classes = 81;
  payload.ssd_recipe_id = stagesemantics::SsdRecipeId::Ssd300V1;
  require(boxdecode_payload_supports_detection_transform(payload),
          "legacy SSD must accept the preprocess transform");
  for (const auto id : {stagesemantics::SsdRecipeId::SsdMobile300V1,
                        stagesemantics::SsdRecipeId::SsdLiteMobile320V1}) {
    payload.num_classes = 91;
    payload.ssd_recipe_id = id;
    require(!boxdecode_payload_supports_detection_transform(payload),
            "multiclass SSD joined the unconditional set");
  }
  for (const auto type :
       {BoxDecodeType::Yolo, BoxDecodeType::YoloV5, BoxDecodeType::YoloV6, BoxDecodeType::YoloV7,
        BoxDecodeType::YoloV8, BoxDecodeType::YoloV9, BoxDecodeType::YoloV10,
        BoxDecodeType::YoloV26, BoxDecodeType::YoloX, BoxDecodeType::EffDet, BoxDecodeType::Detr,
        BoxDecodeType::DetrFocal, BoxDecodeType::Centernet}) {
    payload.decode_type = type;
    require(boxdecode_payload_supports_detection_transform(payload),
            "detection profile lost the preprocess transform");
  }
  // Pose, segmentation and 91-class SSD take the transform only from kernels that have it:
  // admission tries it; they are never part of the unconditional set. RCNN never takes it.
  for (const auto type :
       {BoxDecodeType::YoloV5Seg, BoxDecodeType::YoloV7Seg, BoxDecodeType::YoloV8Seg,
        BoxDecodeType::YoloV9Seg, BoxDecodeType::YoloV10Seg, BoxDecodeType::YoloV8Pose,
        BoxDecodeType::YoloV26Pose, BoxDecodeType::YoloV26Seg}) {
    payload.decode_type = type;
    require(!boxdecode_payload_supports_detection_transform(payload) &&
                boxdecode_payload_may_take_transform(payload),
            "pose/segmentation must be tried against the installed kernel");
  }
  payload.decode_type = BoxDecodeType::RcnnStage1;
  require(!boxdecode_payload_supports_detection_transform(payload) &&
              !boxdecode_payload_may_take_transform(payload),
          "RCNN proposals must never take a transform");
  payload.decode_type = BoxDecodeType::Ssd;
  payload.ssd_recipe_id = stagesemantics::SsdRecipeId::Ssd300V1;
  payload.num_classes = 81;
  require(!boxdecode_payload_may_take_transform(payload), "legacy SSD is not capability-gated");
  for (const auto id : {stagesemantics::SsdRecipeId::SsdMobile300V1,
                        stagesemantics::SsdRecipeId::SsdLiteMobile320V1}) {
    payload.num_classes = 91;
    payload.ssd_recipe_id = id;
    require(boxdecode_payload_may_take_transform(payload), "multiclass SSD is capability-gated");
  }
}

// An explicit image size with no preprocess publisher (no upstream Preproc, no ingress facts)
// keeps the static recipe it had before #1507: nothing can publish a per-frame affine. The same
// explicit size behind a Preproc installs the per-frame resolver.
void roi_explicit_size_static() {
  auto heads = source(BoxDecodeType::Yolo, 2);
  heads.payload.topk = 4;
  input(heads, 8, 8, 4);
  input(heads, 8, 8, 2);
  heads = roi_heads(std::move(heads));
  const BoxDecodeGeometryOptions geometry{640, 640, 1280, 720, ResizeMode::Letterbox};
  auto producer = std::make_shared<CompiledNodeContract>();
  producer->logical_stage_id = "roi_head_producer";
  producer->kernel.emplace();
  producer->kernel->runtime_contract.physical_outputs = heads.runtime_contract.physical_inputs;
  ContractCompileInput alone;
  alone.immediate_upstream = producer.get();
  const auto admitted = compile_boxdecode_kernel(heads, alone, geometry, ResizeMode::Letterbox);
  require(admitted.recipe && admitted.recipe->resolver_factories.empty(),
          "an explicit image size without a preprocess publisher gained per-frame resolution");
  (void)admit_roi(heads, ResizeMode::Letterbox, geometry);
}

// A kernel that refuses the transform of a capability-gated profile (an older release) must give
// BoxDecodeTransformUnavailable, which admission turns into the whole-frame guard; any other
// refusal stays an ordinary error. A non-finite scale stands in for the refusal: the kernel rejects
// it with the transform and accepts the same request without one.
void roi_transform_unavailable_trigger() {
  const auto refuse = [](BoxDecodeType type, int levels_channels, bool& unavailable, bool& other) {
    auto s = source(type, 1);
    for (int n : {80, 40, 20}) {
      input(s, n, n, 4, "FP32");
      input(s, n, n, 1, "FP32");
      if (levels_channels)
        input(s, n, n, levels_channels, "FP32");
    }
    BoxDecodeGeometryOptions geometry{640, 640, 1280, 720, ResizeMode::Letterbox};
    geometry.model_to_original = std::array<float, 4>{std::nanf(""), 1.f, 0.f, 0.f};
    try {
      (void)prepare_boxdecode_kernel(s, geometry);
    } catch (const BoxDecodeTransformUnavailable&) {
      unavailable = true;
    } catch (const std::invalid_argument&) {
      other = true;
    }
  };
  bool unavailable = false, other = false;
  refuse(BoxDecodeType::YoloV26Pose, 51, unavailable, other);
  require(unavailable && !other, "a refused transform of a gated profile was not reported as such");
  unavailable = other = false;
  refuse(BoxDecodeType::YoloV26, 0, unavailable, other);
  require(!unavailable && other, "a refused transform of an ungated profile became a capability");
}

void roi_affine_geometry() {
  roi_transform_support();
  roi_explicit_size_static();
  roi_decoded_letterbox();
  roi_detr_stretch();
  roi_no_transform_profiles();
  roi_pose_seg_known_point();
  roi_transform_unavailable_trigger();
}
