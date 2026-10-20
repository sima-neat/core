#include "pipeline/internal/sima/BoxDecodeKernelRecipe.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/BoxDecodeKernelAdmission.h"
#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "gst/GstInit.h"
#include <simaai/scheduler/testing/probe.h>
#include <iostream>
#include <simaai/scheduler/client/kernel.h>
#include <cmath>
#include <cstring>
#include <new>
#include <stdexcept>

namespace test = simaai::scheduler::testing;
namespace c = simaai::scheduler::client;
namespace {
c::KernelInspection inspect(const c::KernelExecutable& value) {
  auto result = c::inspect_kernel(value);
  if (const auto* error = std::get_if<c::Error>(&result))
    throw std::runtime_error(error->message);
  return std::get<c::KernelInspection>(std::move(result));
}

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal::sima;
namespace gs = simaai::scheduler::client;
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
CompiledBoxDecodeContract source(BoxDecodeType type, int classes = 2) {
  CompiledBoxDecodeContract s;
  s.payload.decode_type = type;
  s.payload.num_classes = classes;
  s.payload.topk = 4;
  s.payload.detection_threshold = .5;
  s.payload.nms_iou_threshold = .5;
  s.payload.score_activation = BoxDecodeScoreActivation::Sigmoid;
  s.payload.decode_type_option = BoxDecodeTypeOption::InterleavedByHead;
  return s;
}
void input(CompiledBoxDecodeContract& s, int h, int w, int c, const std::string& dtype = "BF16",
           int kind = 1, int offset = 0, bool chw = false) {
  const int i = s.runtime_contract.logical_inputs.size();
  auto shape = chw ? std::vector<std::int64_t>{c, h, w} : std::vector<std::int64_t>{h, w, c};
  auto logical = specbuilders::build_logical_input_static_spec(
      i, i, i, shape, dtype, chw ? "CHW" : "HWC", "input" + std::to_string(i),
      "input" + std::to_string(i), "input" + std::to_string(i), offset);
  if (dtype == "INT8")
    logical.quant = QuantStaticSpec{QuantGranularity::PerTensor, -1, {3.1}, {-3}};
  s.runtime_contract.logical_inputs.push_back(std::move(logical));
  s.payload.tensor_storage_kind.push_back(kind);
  sima_ev_shape_desc slice{};
  slice.rank = 3;
  slice.sizes[0] = h;
  slice.sizes[1] = w;
  slice.sizes[2] = c;
  s.payload.slice_shapes.push_back(slice);
}
BoxDecodeKernelRecipeResult check(const CompiledBoxDecodeContract& s,
                                  BoxDecodeGeometryOptions geometry = {64, 64, 64, 64,
                                                                       ResizeMode::Letterbox}) {
  auto r = prepare_boxdecode_kernel(s, geometry);
  const auto& recipe = *r.compiled.recipe;
  require(recipe.members.size() >= 2, "decode/publication members missing");
  require(recipe.members.back().dependencies == std::vector<std::size_t>{r.decode_member},
          "publication must depend on decode");
  require(recipe.metrics && recipe.metrics->version == 1 && recipe.metrics->phase == "Exec" &&
              recipe.metrics->component == "boxdecode_stage_wall_v2",
          "recipe must declare the versioned outer stage wall scope");
  require(recipe.metrics->regions.size() == 1, "one backend region per frame, not per member");
  const auto& region = recipe.metrics->regions.front();
  require(region.phase == "Run" && region.component == "boxdecode_backend_wall_v2" &&
              region.first_member == r.decode_member &&
              region.last_member == recipe.members.size() - 1,
          "backend timing must include wire publication and exclude preceding conversion helpers");
  require(r.compiled.runtime_contract.logical_outputs.size() == 1, "wire output count");
  for (auto port : recipe.members.back().inputs)
    require(port.source == gs::TensorStagePortSource::Output, "publication reads private outputs");
  return r;
}
void families() {
  auto rcnn = source(BoxDecodeType::RcnnStage1, 1);
  for (int side : {136, 68, 34, 17, 9}) {
    input(rcnn, side, side, 12);
    input(rcnn, side, side, 3);
  }
  const auto proposals = check(rcnn, {544, 544, 544, 544, ResizeMode::Letterbox});
  const auto contract =
      inspect(proposals.compiled.recipe->members[proposals.decode_member].executable);
  const auto specs = contract.outputs();
  require(specs.size() == 5 && specs[0].dtype == simaai::scheduler::client::DType::Float32 &&
              specs[0].shape == std::vector<std::int64_t>({16, 4}),
          "RCNN needs four per-level capacities and unrounded FP32 coordinates");

  for (auto type : {BoxDecodeType::YoloV26, BoxDecodeType::YoloV6}) {
    auto s = source(type);
    input(s, 8, 8, 4);
    input(s, 8, 8, 2);
    check(s);
  }
  for (auto type : {BoxDecodeType::YoloV5, BoxDecodeType::YoloV7}) {
    auto s = source(type);
    s.payload.decode_type_option = BoxDecodeTypeOption::Auto;
    for (int n : {8, 4, 2})
      input(s, n, n, 32);
    check(s);
  }
  for (auto type : {BoxDecodeType::YoloV8, BoxDecodeType::YoloV9, BoxDecodeType::YoloV10}) {
    auto s = source(type);
    for (int n : {8, 4, 2}) {
      input(s, n, n, 64);
      input(s, n, n, 2);
    }
    check(s);
  }
  auto x = source(BoxDecodeType::YoloX);
  x.payload.decode_type_option = BoxDecodeTypeOption::Split3Interleaved;
  for (int n : {8, 4, 2}) {
    input(x, n, n, 4);
    input(x, n, n, 1);
    input(x, n, n, 2);
  }
  check(x);
  auto decoded = source(BoxDecodeType::Yolo);
  input(decoded, 8, 8, 4);
  input(decoded, 8, 8, 2);
  check(decoded);
  auto det = source(BoxDecodeType::Detr);
  input(det, 1, 8, 4);
  input(det, 1, 8, 3);
  det.payload.score_activation = BoxDecodeScoreActivation::Softmax;
  check(det, {64, 64, 64, 64, ResizeMode::Stretch});
  auto center = source(BoxDecodeType::Centernet);
  center.payload.decode_type_option = BoxDecodeTypeOption::Split3Interleaved;
  input(center, 8, 8, 2);
  input(center, 8, 8, 2);
  input(center, 8, 8, 2);
  check(center);
  auto eff = source(BoxDecodeType::EffDet);
  for (int n : {32, 16, 8, 4, 2}) {
    input(eff, n, n, 36);
    input(eff, n, n, 32);
  }
  check(eff, {256, 256, 256, 256, ResizeMode::Letterbox});
  auto ssd = source(BoxDecodeType::Ssd, 81);
  ssd.payload.ssd_recipe_id = stagesemantics::SsdRecipeId::Ssd300V1;
  ssd.payload.score_activation = BoxDecodeScoreActivation::Softmax;
  const int dims[]{38, 19, 10, 5, 3, 1}, anchors[]{4, 6, 6, 6, 4, 4};
  for (int i = 0; i < 6; ++i)
    input(ssd, dims[i], dims[i], anchors[i] * 4);
  for (int i = 0; i < 6; ++i)
    input(ssd, dims[i], dims[i], anchors[i] * 81);
  check(ssd, {300, 300, 300, 300, ResizeMode::Stretch});
}
void auxiliary_tasks() {
  for (auto type : {BoxDecodeType::YoloV26Pose, BoxDecodeType::YoloV8Pose}) {
    auto s = source(type, 1);
    s.payload.decode_type_option = BoxDecodeTypeOption::InterleavedByHead;
    const bool dfl = type == BoxDecodeType::YoloV8Pose;
    for (int n : {8, 4, 2}) {
      input(s, n, n, dfl ? 64 : 4);
      input(s, n, n, 1);
      input(s, n, n, 51);
    }
    check(s);
  }
  for (auto type :
       {BoxDecodeType::YoloV26Seg, BoxDecodeType::YoloV5Seg, BoxDecodeType::YoloV7Seg,
        BoxDecodeType::YoloV8Seg, BoxDecodeType::YoloV9Seg, BoxDecodeType::YoloV10Seg}) {
    auto s = source(type);
    const bool combined = type == BoxDecodeType::YoloV5Seg || type == BoxDecodeType::YoloV7Seg;
    const bool raw = type == BoxDecodeType::YoloV26Seg;
    for (int n : {80, 40, 20}) {
      input(s, n, n, combined ? 32 : raw ? 4 : 64);
      if (!combined)
        input(s, n, n, 2);
      input(s, n, n, combined ? 96 : 32);
    }
    input(s, 160, 160, 32);
    check(s, {640, 640, 640, 640, ResizeMode::Letterbox});
  }
}

void combined_task() {
  namespace k = simaai::scheduler::client;
  for (const auto* dtype : {"BF16", "INT8", "FP32"}) {
    for (const int points : {1, 13, 17}) {
      for (const bool by_role : {false, true}) {
        for (const bool rectangular : {false, true}) {
          auto s = source(BoxDecodeType::YoloXSegPose, 5);
          s.payload.pose_classes = {0, 4};
          s.payload.decode_type_option = by_role ? BoxDecodeTypeOption::GroupedByRoleLogit
                                                 : BoxDecodeTypeOption::InterleavedByHeadLogit;
          const int depths[]{4, 6, 32, points * 3};
          auto add = [&](int head, int role) {
            input(s, 80 >> head, (rectangular ? 60 : 80) >> head, depths[role], dtype);
            // Physical channel padding must not change score C or keypoint P.
            if (std::string(dtype) != "FP32") {
              auto& logical = s.runtime_contract.logical_inputs.back();
              auto& shape = logical.shape;
              shape.back() = ((depths[role] + 15) / 16) * 16;
              const std::int64_t width = std::string(dtype) == "BF16" ? 2 : 1;
              logical.stride_bytes = {shape[1] * shape[2] * width, shape[2] * width, width};
              logical.size_bytes = shape[0] * logical.stride_bytes[0];
            }
          };
          if (by_role) {
            for (int role = 0; role < 4; ++role)
              for (int head = 0; head < 3; ++head)
                add(head, role);
          } else {
            for (int head = 0; head < 3; ++head)
              for (int role = 0; role < 4; ++role)
                add(head, role);
          }
          // Prototype dtype is independent of per-head dtype.
          input(s, 160, 160, 32, std::string(dtype) == "INT8" ? "BF16" : "INT8");
          const auto r = check(s, {rectangular ? 480 : 640, 640, 960, 720, ResizeMode::Letterbox});
          require(r.decode_member == 0 && r.compiled.recipe->members.size() == 2,
                  "combined decode duplicated computation or introduced source copies");
          const auto& numerical = r.compiled.recipe->members.front();
          const auto contract = inspect(numerical.executable);
          const auto specs = contract.outputs();
          require(numerical.inputs.size() == 13 && specs.size() == 6 &&
                      specs[4].shape == std::vector<std::int64_t>({4, 160, 160}) &&
                      specs[5].shape == std::vector<std::int64_t>({4, points, 3}),
                  "combined canonical roles/output contract mismatch");
          require(r.compiled.runtime_contract.logical_outputs.front().size_bytes ==
                      4 + 4 * (24 + 160 * 160 + 204),
                  "combined public wire must retain17-point capacity");
        }
      }
    }
  }
  auto make = [] {
    auto s = source(BoxDecodeType::YoloXSegPose, 2);
    for (int side : {80, 40, 20}) {
      input(s, side, side, 4);
      input(s, side, side, 3);
      input(s, side, side, 32);
      input(s, side, side, 39);
    }
    input(s, 160, 160, 32);
    return s;
  };
  for (int failure = 0; failure < 5; ++failure) {
    auto s = make();
    if (failure == 0)
      s.payload.pose_classes = {2};
    if (failure == 1)
      s.runtime_contract.logical_inputs[0].dtype = "UINT8";
    if (failure == 2)
      s.payload.slice_shapes[7].sizes[2] = 36;
    if (failure == 3)
      s.payload.slice_shapes[1].sizes[2] = 4;
    if (failure == 4)
      s.payload.score_activation = BoxDecodeScoreActivation::Identity;
    bool rejected = false;
    try {
      check(s, {640, 640, 640, 640, ResizeMode::Letterbox});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "malformed combined contract accepted");
  }
  // Decoding an existing crop is defined by its actual affine, not by
  // inventing center-crop preprocessing from the resize-mode token.
  BoxDecodeGeometryOptions crop{640, 640, 960, 720, ResizeMode::Crop};
  crop.model_to_original = std::array<float, 4>{1.1f, 1.1f, 120.f, 8.f};
  require(check(make(), crop).decode_member == 0,
          "explicit crop geometry must not introduce an adapter");
  crop.model_to_original.reset();
  bool missing_crop_rejected = false;
  try {
    (void)check(make(), crop);
  } catch (const std::invalid_argument&) {
    missing_crop_rejected = true;
  }
  require(missing_crop_rejected, "crop without real preprocessing authority was fabricated");
  // Packed slice depth is a transport tile, not the keypoint count.
  auto packed = make();
  for (std::size_t i = 0; i < packed.payload.tensor_storage_kind.size(); ++i) {
    packed.payload.tensor_storage_kind[i] = 0;
    packed.payload.slice_shapes[i].sizes[2] = 16;
  }
  require(check(packed, {640, 640, 640, 640, ResizeMode::Letterbox}).decode_member == 0,
          "native C-block combined reader introduced source copies");
}

// A real host allocation adapter, not a numerical mock. Runner executes the
// canonical checked kernel and Core publication using ordinary retained leases.
struct HostStorage final {
  std::size_t size;
  std::shared_ptr<void> memory;
  explicit HostStorage(std::size_t bytes)
      : size(bytes), memory(::operator new(bytes, std::align_val_t{64}),
                            [](void* p) { ::operator delete(p, std::align_val_t{64}); }) {
    std::memset(memory.get(), 0, size);
  }
  template <class T> T read(std::size_t at) const {
    T value;
    std::memcpy(&value, static_cast<const std::byte*>(memory.get()) + at, sizeof(T));
    return value;
  }
  void set_float(std::size_t element, float value) {
    std::memcpy(static_cast<std::byte*>(memory.get()) + element * 4, &value, 4);
  }
};
void combined_execution(bool planar, int ingress_route = 0, bool gaps = false) {
  namespace rt = simaai::scheduler::client;
  const auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };
  auto s = source(BoxDecodeType::YoloXSegPose, 2);
  s.payload.pose_classes = {0};
  const float boundary_score = sigmoid(2.f) * sigmoid(2.001f);
  const float narrowed_score = sigmoid(2.f) * sigmoid(2.f);
  s.payload.detection_threshold = (boundary_score + narrowed_score) / 2.f;
  require(boundary_score > s.payload.detection_threshold &&
              narrowed_score < s.payload.detection_threshold,
          "FP32 threshold negative control does not separate narrowing");
  for (int side : {80, 40, 20}) {
    input(s, side, side, 4, "FP32", 1, 0, planar);
    input(s, side, side, 3, "FP32", 1, 0, planar);
    input(s, side, side, 32, "FP32", 1, 0, planar);
    input(s, side, side, 3, "FP32", 1, 0, planar);
  }
  input(s, 160, 160, 32, "FP32", 1, 0, planar);
  if (gaps) {
    require(!planar, "permutation fixture uses declared HWC logical axes");
    for (auto& logical : s.runtime_contract.logical_inputs) {
      const auto& shape = logical.shape;
      const std::int64_t channel_stride = shape[0] * 4 + 4;
      logical.stride_bytes = {4, shape[2] * channel_stride + 4, channel_stride};
    }
  }
  const auto authored_inputs = s.runtime_contract.logical_inputs;
  auto execution_inputs = authored_inputs;
  std::optional<CompiledKernelContract> admitted_recipe;
  // Public combined extraction uses grouped semantic roles. Actual storage
  // remains independent: CHW and padded permutations must not become HWC just
  // because the extracted logical shape is canonicalized.
  constexpr std::array<std::size_t, 13> grouped{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12};
  std::array<std::size_t, 13> port_map{};
  for (std::size_t i = 0; i < port_map.size(); ++i)
    port_map[ingress_route ? grouped[i] : i] = i;
  if (ingress_route) {
    TensorList tensors;
    CompiledNodeContract upstream;
    upstream.kernel.emplace();
    for (std::size_t i = 0; i < grouped.size(); ++i) {
      const auto& original = authored_inputs[grouped[i]];
      Tensor tensor;
      tensor.dtype = TensorDType::Float32;
      tensor.layout = planar ? TensorLayout::CHW : TensorLayout::HWC;
      tensor.shape = original.shape;
      tensor.strides_bytes = original.stride_bytes;
      if (planar && ingress_route == 1)
        tensor.strides_bytes.clear(); // Public Tensor means compact declared CHW, not HWC.
      const auto linear =
          boxdecode_linear_storage(original.shape, original.stride_bytes, original.layout, 4);
      require(linear.has_value(), "fixture linear geometry invalid");
      tensor.storage = make_cpu_owned_storage(linear->span_bytes);
      tensor.route.logical_index = tensor.route.physical_index = tensor.route.route_slot = i;
      const auto role = i == 12 ? "mask_proto"
                        : i < 3 ? "bbox"
                        : i < 6 ? "class"
                        : i < 9 ? "mask_coeff"
                                : "kpt";
      tensor.route.name = std::string(role) + "_" + std::to_string(i % 3);
      tensors.push_back(std::move(tensor));
      auto output = specbuilders::build_logical_output_static_spec(
          i, i, i, i, i, original.shape, "FP32", original.layout, tensors.back().route.name,
          tensors.back().route.name, tensors.back().route.name);
      output.stride_bytes = original.stride_bytes;
      output.size_bytes = linear->span_bytes;
      upstream.kernel->runtime_contract.logical_outputs.push_back(std::move(output));
      upstream.kernel->runtime_contract.physical_outputs.push_back(
          specbuilders::build_physical_buffer_static_spec(i, i, linear->span_bytes, DeviceKind::Cpu,
                                                          tensors.back().route.name));
    }
    BoxDecodeStandaloneContractOverrides overrides;
    overrides.source_storage_kind = BoxDecodeSourceStorageKind::DenseHwcPhysical;
    std::string error;
    auto extracted = ingress_route == 1
                         ? build_boxdecode_static_contract_from_sample(
                               sample_from_tensors(tensors), BoxDecodeType::YoloXSegPose,
                               std::nullopt, overrides, &error)
                         : build_boxdecode_static_contract_from_compiled_upstream(
                               upstream, BoxDecodeType::YoloXSegPose, overrides, &error);
    if (!extracted)
      throw std::runtime_error("public strided fixture extraction: " + error);
    extracted->pose_classes = {0};
    auto finalized = stagesemantics::finalize_boxdecode_static_contract(
        *extracted, BoxDecodeType::YoloXSegPose, {}, {}, BoxDecodeTypeOption::Auto,
        s.payload.detection_threshold, .5, 4, 0, {});
    s = stagesemantics::build_boxdecode_compiled_contract(finalized);
    for (std::size_t i = 0; i < grouped.size(); ++i) {
      const auto& original = authored_inputs[grouped[i]];
      const auto expected =
          boxdecode_linear_storage(original.shape, original.stride_bytes, original.layout, 4);
      require(s.runtime_contract.logical_inputs[i].stride_bytes == expected->stride_bytes,
              "public sample/upstream extraction dropped physical strides");
    }
    ContractCompileInput context;
    if (ingress_route == 1)
      context.ingress.ingress_sample = sample_from_tensors(tensors);
    else
      context.immediate_upstream = &upstream;
    admitted_recipe = compile_boxdecode_kernel(
        s, context, {640, 640, 640, 640, ResizeMode::Letterbox}, ResizeMode::Letterbox);
    if (ingress_route == 1 && gaps) {
      // Existing public CPU transport packs non-contiguous views. Admission
      // must use that new carrier's strides, not the pre-copy source layout.
      for (auto& logical : execution_inputs) {
        std::int64_t stride = 4;
        for (std::size_t axis = logical.shape.size(); axis-- > 0;) {
          logical.stride_bytes[axis] = stride;
          stride *= logical.shape[axis];
        }
      }
    }
    for (std::size_t i = 0; i < grouped.size(); ++i) {
      const auto& physical = execution_inputs[grouped[i]];
      const auto expected =
          boxdecode_linear_storage(physical.shape, physical.stride_bytes, physical.layout, 4);
      require(admitted_recipe->runtime_contract.logical_inputs[i].stride_bytes ==
                  expected->stride_bytes,
              "admission did not preserve the actual envelope/upstream storage authority");
    }
  }
  const auto recipe = admitted_recipe ? BoxDecodeKernelRecipeResult{std::move(*admitted_recipe), 0}
                                      : check(s, {640, 640, 640, 640, ResizeMode::Letterbox});
  require(recipe.decode_member == 0 && recipe.compiled.recipe->members.size() == 2,
          "native FP32 execution gained an adapter or staging copy");
  const auto& members = recipe.compiled.recipe->members;
  test::OperationProbe executor, publisher;
  const auto take = []<class T>(rt::Result<T> result) -> T {
    if (const auto* error = std::get_if<rt::Error>(&result))
      throw std::runtime_error(error->field + ": " + error->message);
    return std::get<T>(std::move(result));
  };
  if (auto error = executor.prepare(members[0].executable))
    throw std::runtime_error(error->message);
  if (auto error = publisher.prepare(members[1].executable))
    throw std::runtime_error(error->message);
  const auto base = executor.operation();
  const auto publication = publisher.operation();
  auto dynamic_recipe = *recipe.compiled.recipe;
  BoxDecodeGeometryOptions geometry{640, 640, 640, 640, ResizeMode::Letterbox};
  geometry.complete_geometry = true;
  attach_boxdecode_geometry_resolver(dynamic_recipe, 0, geometry, true, true);
  auto resolve = dynamic_recipe.resolver_factories.front().instantiate();
  const gchar* tags[] = {nullptr};
  if (!gst_meta_get_info("GstSimaMeta"))
    require(gst_meta_register_custom("GstSimaMeta", tags, nullptr, nullptr, nullptr),
            "combined execution metadata registration failed");
  test::GstBufferOwner carrier(gst_buffer_new());
  auto* metadata =
      gst_custom_meta_get_structure(gst_buffer_add_custom_meta(carrier.get(), "GstSimaMeta"));
  const std::array<GstBuffer*, 1> carriers{carrier.get()};
  rt::PreparedSelection previous;
  for (const bool rebind : {false, true}) {
    const float offset_x = rebind ? 14.f : 10.f;
    gst_structure_set(
        metadata, "preproc_original_width", G_TYPE_INT, 640, "preproc_original_height", G_TYPE_INT,
        640, "preproc_resized_width", G_TYPE_INT, 640, "preproc_resized_height", G_TYPE_INT, 640,
        "preproc_resize_mode", G_TYPE_STRING, "letterbox", "preproc_affine_scale_x", G_TYPE_DOUBLE,
        2., "preproc_affine_scale_y", G_TYPE_DOUBLE, 3., "preproc_affine_offset_x", G_TYPE_DOUBLE,
        double(offset_x), "preproc_affine_offset_y", G_TYPE_DOUBLE, -10., nullptr);
    const auto numerical = take(resolve(executor.preparer(), carriers));
    require(numerical != base && numerical != previous &&
                c::same_kernel_contract_resources(numerical.executable(), base.executable()),
            "actual changed-affine execution lost immutable resource reuse");
    previous = numerical;
    std::vector<std::shared_ptr<HostStorage>> inputs, results;
    std::vector<rt::ConstBytes> in;
    std::vector<rt::MutableBytes> out;
    std::vector<rt::ConstBytes> publication_inputs;
    const auto contract = inspect(members[0].executable);
    for (const auto requirement : contract.input_requirements()) {
      inputs.push_back(std::make_shared<HostStorage>(requirement.storage_bytes));
      in.push_back(
          {static_cast<const std::byte*>(inputs.back()->memory.get()), requirement.storage_bytes});
    }
    for (const auto requirement : contract.output_requirements()) {
      results.push_back(std::make_shared<HostStorage>(requirement.storage_bytes));
      out.push_back(
          {static_cast<std::byte*>(results.back()->memory.get()), requirement.storage_bytes});
      publication_inputs.push_back({out.back().data, out.back().size_bytes});
    }
    const auto bytes = recipe.compiled.runtime_contract.logical_outputs.front().size_bytes;
    auto wire = std::make_shared<HostStorage>(bytes);
    const std::array<rt::MutableBytes, 1> published{
        {{static_cast<std::byte*>(wire->memory.get()), bytes}}};
    auto numerical_binding = take(executor.bind_host(numerical, in, out));
    auto publication_binding =
        take(publisher.bind_host(publication, publication_inputs, published));
    for (std::size_t port : {1U, 5U, 9U})
      for (std::size_t element = 0; element < inputs[port_map[port]]->size / 4; ++element)
        inputs[port_map[port]]->set_float(element, -8.f);
    // Author values from independent logical HWC coordinates. The planar
    // fixture uses genuinely different bytes, including its mask prototype;
    // changing only the layout token would not detect an incorrect HWC reader.
    const auto write = [&](std::size_t port, std::size_t element, float value) {
      const auto& original = execution_inputs[port];
      const auto& shape = original.shape;
      const auto& strides = original.stride_bytes;
      const std::size_t channels = planar ? shape[0] : shape[2];
      const std::size_t width = planar ? shape[2] : shape[1];
      const std::size_t channel = element % channels, cell = element / channels;
      const std::size_t offset =
          planar ? channel * strides[0] + (cell / width) * strides[1] + (cell % width) * strides[2]
                 : (cell / width) * strides[0] + (cell % width) * strides[1] + channel * strides[2];
      inputs[port_map[port]]->set_float(offset / 4, value);
    };
    for (std::size_t pixel = 0; pixel < 160 * 160; ++pixel)
      write(12, pixel * 32, 1.f);
    constexpr int low_cell = 10 * 80 + 10, high_cell = 20 * 80 + 20;
    for (const int cell : {low_cell, high_cell}) {
      write(1, cell * 3, 2.f);
      write(2, cell * 32, cell == low_cell ? 1.f : 2.f);
      write(3, cell * 3, .25f);
      write(3, cell * 3 + 1, .5f);
      write(3, cell * 3 + 2, 2.f);
    }
    for (const int frame : {0, 1, 2, 3}) {
      const int expected_count = frame < 2 ? 2 : frame == 2 ? 1 : 0;
      write(1, low_cell * 3 + 1, frame < 2 ? 2.001f : 2.f);
      write(1, high_cell * 3 + 2, frame == 3 ? -8.f : 3.f);
      std::memset(wire->memory.get(), 0xa5, bytes);
      const auto decoded = executor.submit(numerical_binding);
      const std::span<const test::Completion> dependencies{&decoded, 1};
      auto complete = publisher.submit(publication_binding, dependencies);
      require(!complete.wait(), "Runner combined numerical/publication execution failed");
      require(wire->read<std::int32_t>(0) == expected_count,
              "combined survivor count or FP32 threshold precision changed");
      constexpr std::size_t mask_base = 4 + 4 * 24;
      constexpr std::size_t pose_base = mask_base + 4 * 160 * 160;
      for (int row = 0; row < 4; ++row) {
        if (row < expected_count) {
          const int cell_axis = row == 0 ? 20 : 10;
          require(wire->read<std::int32_t>(4 + row * 24) ==
                          (cell_axis * 8 - 4) * 2 + int(offset_x) &&
                      wire->read<std::int32_t>(8 + row * 24) == (cell_axis * 8 - 4) * 3 - 10 &&
                      wire->read<std::int32_t>(12 + row * 24) == 16 &&
                      wire->read<std::int32_t>(16 + row * 24) == 24 &&
                      wire->read<std::int32_t>(24 + row * 24) == (row == 0 ? 1 : 0),
                  "independent combined XYWH/class oracle failed");
          const float expected_score = row == 0 ? sigmoid(2.f) * sigmoid(3.f) : boundary_score;
          require(std::abs(wire->read<float>(20 + row * 24) - expected_score) < 2e-6f,
                  "packed objectness score oracle failed");
          const auto expected_mask = std::uint8_t(sigmoid(row == 0 ? 2.f : 1.f) * 255.f);
          for (int y = 0; y < 160; ++y)
            for (int x = 0; x < 160; ++x) {
              const int origin = cell_axis * 2 - 1;
              const auto expected = x >= origin && x < origin + 2 && y >= origin && y < origin + 2
                                        ? expected_mask
                                        : std::uint8_t{0};
              require(wire->read<std::uint8_t>(mask_base + row * 25600 + y * 160 + x) == expected,
                      "independent combined mask/survivor oracle failed");
            }
          require(wire->read<std::uint32_t>(pose_base + row * 204) ==
                          (row == 0 ? 0U : 168U + unsigned(offset_x)) &&
                      wire->read<std::uint32_t>(pose_base + row * 204 + 4) ==
                          (row == 0 ? 0U : 254U) &&
                      std::abs(wire->read<float>(pose_base + row * 204 + 8) -
                               (row == 0 ? 0.f : sigmoid(2.f))) < 1e-7f,
                  "combined pose class gate/survivor oracle failed");
        } else {
          for (int byte = 0; byte < 24; ++byte)
            require(wire->read<std::uint8_t>(4 + row * 24 + byte) == 0, "stale box tail");
          for (int byte = 0; byte < 25600; ++byte)
            require(wire->read<std::uint8_t>(mask_base + row * 25600 + byte) == 0,
                    "stale mask tail");
        }
        for (int byte = row < expected_count ? 12 : 0; byte < 204; ++byte)
          require(wire->read<std::uint8_t>(pose_base + row * 204 + byte) == 0,
                  "combined pose padding/tail was not cleared");
      }
    }
  }
}

void normalization() {
  for (auto dtype : {"BF16", "INT8", "FP32"}) {
    auto s = source(BoxDecodeType::YoloV26);
    input(s, 8, 8, 4, dtype, 1, 64);
    input(s, 8, 8, 2, dtype);
    const auto r = check(s);
    require(r.compiled.recipe->members.front().inputs.front().byte_offset == 64,
            "source offset lost");
    require(r.decode_member == 0, "dense heads acquired a hidden conversion");
    if (std::string(dtype) == "FP32")
      require(inspect(r.compiled.recipe->members.front().executable).inputs()[0].dtype ==
                  simaai::scheduler::client::DType::Float32,
              "native FP32 precision was narrowed");
  }
  auto rank6 = source(BoxDecodeType::YoloV26);
  input(rank6, 8, 8, 4);
  input(rank6, 8, 8, 2);
  for (auto& logical : rank6.runtime_contract.logical_inputs) {
    const auto bytes = static_cast<std::int64_t>(logical.size_bytes);
    logical.shape.insert(logical.shape.begin(), 3, 1);
    logical.stride_bytes.insert(logical.stride_bytes.begin(), 3, bytes);
  }
  check(rank6);
  auto chw = source(BoxDecodeType::YoloV26);
  input(chw, 8, 8, 4, "FP32", 1, 0, true);
  input(chw, 8, 8, 2, "FP32", 1, 0, true);
  require(check(chw).decode_member == 0,
          "planar FP32 must preserve precision without a staging copy");
  for (auto& logical : chw.runtime_contract.logical_inputs) {
    const auto channels = logical.shape[0], width = logical.shape[2];
    logical.stride_bytes = {4, width * channels * 4, channels * 4};
  }
  require(check(chw).decode_member == 0,
          "CHW logical axes over explicit HWC FP32 storage added an unnecessary copy");
  auto packed = source(BoxDecodeType::YoloV26);
  input(packed, 8, 8, 16, "INT8", 0);
  input(packed, 8, 8, 16, "INT8", 0);
  require(check(packed).decode_member == 0, "C-block coordinates must not add a detess pass");
  auto hwc16 = source(BoxDecodeType::YoloV26);
  input(hwc16, 8, 8, 16, "BF16", 2);
  input(hwc16, 8, 8, 16, "BF16", 2);
  require(check(hwc16).decode_member == 2, "HWC-C16 normalization missing");
}
void superpoint() {
  for (auto profile : {SuperPointProfile::A65V1, SuperPointProfile::LightGlueV1,
                       SuperPointProfile::MagicLeapDemoV1}) {
    auto s = source(BoxDecodeType::SuperPoint);
    s.payload.superpoint.profile = profile;
    s.payload.superpoint.nms_radius = 4;
    s.payload.superpoint.border_margin = 4;
    s.payload.superpoint.descriptor_dim = 256;
    s.payload.tensor_roles = {BoxDecodeTensorRole::DescriptorGrid,
                              BoxDecodeTensorRole::DetectorLogits};
    input(s, 8, 8, 256, "FP32");
    input(s, 8, 8, 65, "FP32");
    const auto r = check(s);
    require(r.compiled.recipe->members[r.decode_member].inputs[0].tensor == 1,
            "SuperPoint explicit role ignored");
  }
  // MLA BF16 pixel-major storage is byte-planar, not dense padded BF16.
  // Keep the explicit native layout and avoid an unnecessary detess member.
  auto tiled = source(BoxDecodeType::SuperPoint);
  tiled.payload.superpoint.profile = SuperPointProfile::A65V1;
  tiled.payload.superpoint.nms_radius = 4;
  tiled.payload.superpoint.border_margin = 4;
  tiled.payload.superpoint.descriptor_dim = 256;
  tiled.payload.tensor_roles = {BoxDecodeTensorRole::DetectorLogits,
                                BoxDecodeTensorRole::DescriptorGrid};
  input(tiled, 8, 8, 65, "BF16", 2);
  input(tiled, 8, 8, 256, "BF16", 2);
  require(check(tiled).decode_member == 0, "native SuperPoint tiled storage was copied");
  require(check(tiled, {64, 64, 64, 64, ResizeMode::Crop}).decode_member == 0,
          "model-space SuperPoint inherited BoxDecode inverse-crop restrictions");
  for (const auto& dtype : {"BF16", "INT8", "FP32"}) {
    auto padded = tiled;
    padded.payload.tensor_storage_kind = {1, 1};
    auto& detector = padded.runtime_contract.logical_inputs[0];
    detector.shape = {8, 8, 80};
    detector.dtype = dtype;
    detector.stride_bytes.clear();
    if (detector.dtype == "INT8")
      detector.quant = QuantStaticSpec{QuantGranularity::PerTensor, -1, {3.1}, {-3}};
    const auto prepared = check(padded);
    require(prepared.decode_member == 0, "padded SuperPoint must use the native dense reader");
    const auto numerical = inspect(prepared.compiled.recipe->members.front().executable);
    const auto expected = detector.dtype == "INT8"   ? simaai::scheduler::client::DType::Int8
                          : detector.dtype == "BF16" ? simaai::scheduler::client::DType::BFloat16
                                                     : simaai::scheduler::client::DType::Float32;
    require(numerical.inputs()[0].dtype == expected && numerical.inputs()[0].shape.back() == 65 &&
                std::holds_alternative<simaai::scheduler::client::Strided>(
                    numerical.inputs()[0].layout),
            "padded SuperPoint must retain its scalar dtype and explicit pixel stride");
  }
}
void output_capacity() {
  // Prove the same physical heads and geometry are admitted with explicit capacity,
  // so another input-admission error cannot satisfy the negative cases below.
  auto s = source(BoxDecodeType::YoloV26);
  input(s, 8, 8, 4);
  input(s, 8, 8, 2);
  for (int capacity : {1, 4}) {
    s.payload.topk = capacity;
    const auto prepared = check(s);
    const auto numerical =
        inspect(prepared.compiled.recipe->members[prepared.decode_member].executable);
    require(numerical.outputs()[0].shape.front() == capacity,
            "explicit output capacity was replaced by a default");
  }
  for (int capacity : {0, -1, 65537}) {
    s.payload.topk = capacity;
    bool rejected = false;
    try {
      check(s);
    } catch (const std::invalid_argument& error) {
      const std::string message = error.what();
      rejected = capacity == 0
                     ? message.find("top_k is unresolved") != std::string::npos &&
                           message.find("positive output capacity") != std::string::npos
                     : message.find("top_k output capacity must be 1..65536") != std::string::npos;
    }
    require(rejected, "invalid output capacity did not produce its specific diagnostic");
  }

  auto rcnn = source(BoxDecodeType::RcnnStage1, 1);
  for (int side : {136, 68, 34, 17, 9}) {
    input(rcnn, side, side, 12);
    input(rcnn, side, side, 3);
  }
  check(rcnn, {544, 544, 544, 544, ResizeMode::Letterbox});
  rcnn.payload.topk = 16385;
  bool rejected = false;
  try {
    check(rcnn, {544, 544, 544, 544, ResizeMode::Letterbox});
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("RCNN topk is per level") != std::string::npos;
  }
  require(rejected, "RCNN per-level capacity bound was lost");
}
void rejects() {
  auto s = source(BoxDecodeType::YoloV26);
  input(s, 8, 8, 4, "INT8");
  input(s, 8, 8, 2, "INT8");
  s.runtime_contract.logical_inputs[0].quant.reset();
  try {
    check(s);
    throw std::runtime_error("missing quantization accepted");
  } catch (const std::invalid_argument&) {
  }
  s = source(BoxDecodeType::YoloV26);
  input(s, 8, 8, 4);
  input(s, 8, 8, 2);
  try {
    check(s, {64, 64, 64, 64, ResizeMode::Crop});
    throw std::runtime_error("crop accepted");
  } catch (const std::invalid_argument&) {
  }
}
void linear_metadata_contracts() {
  require(boxdecode_linear_storage({1, 4, 8, 6}, {}, "NCHW", 4)->stride_bytes ==
              std::vector<std::int64_t>{24, 4, 192},
          "singleton-batch CHW byte strides were not normalized once");
  for (const auto& strides : {std::vector<std::int64_t>{24, 4},
                              {24, -4, 4},
                              {24, 4, 6},
                              {24, 0, 4},
                              {INT64_MAX - 3, 16, 4}})
    require(!boxdecode_linear_storage({8, 6, 4}, strides, "HWC", 4),
            "invalid linear byte strides accepted");
  require(!boxdecode_linear_storage({2, 4, 8, 6}, {}, "NCHW", 4),
          "non-singleton batch was silently flattened");
  require(boxdecode_linear_storage({1, 6, 4}, {0, 16, 4}, "HWC", 4).has_value(),
          "zero singleton stride rejected");
  for (const auto dtype : {TensorDType::Int8, TensorDType::BFloat16}) {
    const std::size_t width = dtype == TensorDType::Int8 ? 1 : 2;
    Tensor tensor;
    tensor.dtype = dtype;
    tensor.layout = TensorLayout::HWC;
    tensor.shape = {8, 8, 16};
    tensor.strides_bytes = {8 * 16 * std::int64_t(width), 16 * std::int64_t(width),
                            std::int64_t(width)};
    tensor.storage = make_cpu_owned_storage(8 * 8 * 16 * width);
    if (dtype == TensorDType::Int8) {
      QuantSpec quant;
      quant.scales = {1.f};
      quant.zero_points = {0};
      tensor.semantic.quant = quant;
    }
    BoxDecodeStandaloneContractOverrides overrides;
    overrides.source_storage_kind = BoxDecodeSourceStorageKind::PackedCBlock;
    std::string error;
    const auto packed = build_boxdecode_static_contract_from_sample(
        sample_from_tensors({tensor}), BoxDecodeType::YoloV26, {}, overrides, &error);
    require(packed && packed->tensors.front().stride_bytes.empty(),
            "packed INT8/BF16 acquired misleading linear storage strides");
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    simaai::neat::gst_init_once();
    linear_metadata_contracts();
    families();
    auxiliary_tasks();
    combined_task();
    combined_execution(false);
    combined_execution(true);
    for (int route : {1, 2}) {
      combined_execution(true, route);
      combined_execution(false, route, true);
    }
    normalization();
    superpoint();
    output_capacity();
    rejects();
    std::cout << "BoxDecode recipe families, normalization, roles and rejection contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
