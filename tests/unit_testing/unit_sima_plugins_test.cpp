#include "gst/GstHelpers.h"
#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/FeatureTypes.h"
#include "nodes/sima/SimaRender.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "pipeline/Graph.h"
#include "pipeline/Run.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorUtil.h"
#include <gstsimaaitensorbuffer_meta.h>
#include "pipeline/internal/sima/StaticSpecBuilders.h"

#include "test_utils.h"

#include <algorithm>
#include <iostream>
#include <array>
#include <cstring>
#include <cmath>
#include <bit>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
void check_argmax_ingress_projection() {
  using namespace simaai::neat;
  namespace pi = pipeline_internal;
  SimaArgMax node;
  const auto compile = [&](const Sample& sample) {
    ContractCompileInput input;
    input.ingress.ingress_sample = sample;
    CompiledNodeContract compiled;
    std::string error;
    const bool admitted = node.compile_node_contract(input, &compiled, &error);
    require(admitted, "ArgMax current-storage admission: " + error);
    auto holder = pi::sample_to_gst_envelope_holder(sample, &error);
    pi::TensorBufferView expected;
    require(holder && pi::tensor_buffer_descriptor_from_sample(
                          static_cast<GstSample*>(holder.get()), &expected, &error),
            "ArgMax ingress reference projection: " + error);
    const auto& runtime = compiled.kernel->runtime_contract;
    require(runtime.logical_inputs.size() == expected.tensors.size(),
            "ArgMax lost a projected ingress tensor");
    for (std::size_t i = 0; i < expected.tensors.size(); ++i) {
      const auto& actual = runtime.logical_inputs[i];
      const auto& projected = expected.tensors[i];
      require(actual.physical_index == projected.memory_index &&
                  actual.byte_offset == projected.byte_offset && actual.shape == projected.shape &&
                  actual.stride_bytes == projected.stride_bytes &&
                  compiled.kernel->recipe->members[i].inputs[0].byte_offset ==
                      static_cast<std::size_t>(projected.byte_offset),
              "ArgMax disagrees with the execution ingress descriptor");
      const auto carrier = std::find_if(
          runtime.physical_inputs.begin(), runtime.physical_inputs.end(),
          [&](const auto& value) { return value.physical_index == projected.memory_index; });
      require(carrier != runtime.physical_inputs.end() &&
                  carrier->size_bytes ==
                      gst_memory_get_sizes(
                          gst_buffer_peek_memory(expected.buffer, projected.memory_index), nullptr,
                          nullptr),
              "ArgMax lost the actual ingress carrier extent");
    }
    return compiled;
  };
  Tensor source;
  source.storage = make_cpu_owned_storage(128);
  source.dtype = TensorDType::Float32;
  source.shape = {2, 3};
  source.strides_bytes = {12, 4};
  source.byte_offset = 64;
  source.route.logical_index = 0;
  source.route.physical_index = 7;
  source.route.memory_index = 0;
  source.route.physical_byte_offset = 64;
  source.route.name = "scores";
  source.route.segment_name = "carrier";
  Tensor cloned = source.clone();
  require(cloned.byte_offset == 0 && cloned.storage->size_bytes == 24 &&
              cloned.route.physical_byte_offset == 64,
          "clone fixture must retain provenance but replace current storage");
  auto current = compile(sample_from_tensors({cloned}));
  require(current.kernel->runtime_contract.logical_inputs[0].byte_offset == 0,
          "cloned ingress reapplied the producer's offset");

  Tensor independent = cloned.clone();
  independent.route.logical_index = 1;
  independent.route.name = "other_scores";
  current = compile(sample_from_tensors({cloned, independent}));
  const auto& separate = current.kernel->runtime_contract.logical_inputs;
  require(separate[0].physical_index != separate[1].physical_index ||
              separate[0].byte_offset != separate[1].byte_offset,
          "independently owned tensors with historical index collisions were aliased");

  Tensor first = source;
  first.byte_offset = first.route.physical_byte_offset = 16;
  first.route.physical_index = 0;
  Tensor second = first;
  second.byte_offset = second.route.physical_byte_offset = 64;
  second.route.logical_index = 1;
  second.route.name = "other_scores";
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 128, nullptr);
  require(buffer != nullptr, "could not allocate borrowed ArgMax carrier");
  pi::attach_tensor_set_meta_from_tensors(buffer, {first, second});
  std::string error;
  require(simaai::gst::tensor_buffer_attach_physical_bindings(buffer, {{0, 0, 128}}, &error),
          "could not describe borrowed ArgMax carrier: " + error);
  GstCaps* caps = gst_caps_from_string("application/vnd.simaai.tensor");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_buffer_unref(buffer);
  gst_caps_unref(caps);
  first.storage = second.storage = pi::make_gst_sample_storage(sample);
  gst_sample_unref(sample);
  current = compile(sample_from_tensors({first, second}));
  const auto& borrowed = current.kernel->runtime_contract;
  require(borrowed.physical_inputs.size() == 1 && borrowed.physical_inputs[0].size_bytes == 128 &&
              borrowed.logical_inputs[0].byte_offset == 16 &&
              borrowed.logical_inputs[1].byte_offset == 64,
          "borrowed shared carrier lost its actual offsets or extent");

  // OutputMemory::Owned materializes each packed logical output independently,
  // retaining its route provenance just as clone() does.
  current = compile(sample_from_tensors({first.clone(), second.clone()}));
  const auto& owned = current.kernel->runtime_contract.logical_inputs;
  require(owned[0].byte_offset == 0 &&
              (owned[1].physical_index != owned[0].physical_index || owned[1].byte_offset == 24),
          "Owned packed outputs used stale parent offsets");
  // A single contiguous CPU view remains zero-copy, but its visible GstMemory
  // starts at the view, not the retained producer offset. The envelope, not the
  // input Tensor wrapper, must retain that allocation.
  for (const bool external : {false, true}) {
    Tensor view = source;
    view.storage = make_cpu_owned_storage(128);
    if (external) {
      const auto owner = view.storage->holder;
      view.storage = make_cpu_external_storage(view.storage->data, 128, owner, false);
    }
    std::weak_ptr<void> owner = view.storage->holder;
    const auto* address = static_cast<const std::uint8_t*>(view.storage->data) + 64;
    auto envelope = pi::sample_to_gst_envelope_holder(sample_from_tensors({view}), &error);
    require(envelope != nullptr, "could not wrap CPU view: " + error);
    GstBuffer* visible = gst_sample_get_buffer(static_cast<GstSample*>(envelope.get()));
    pi::TensorBufferView descriptor;
    require(pi::tensor_buffer_descriptor_from_sample(static_cast<GstSample*>(envelope.get()),
                                                     &descriptor, &error) &&
                descriptor.tensors[0].byte_offset == 0 &&
                descriptor.tensors[0].stride_bytes == std::vector<std::int64_t>({12, 4}) &&
                gst_buffer_get_size(visible) == 24,
            "CPU view descriptor did not describe the visible carrier");
    GstMapInfo mapped{};
    require(gst_buffer_map(visible, &mapped, GST_MAP_READ), "could not inspect CPU view");
    require(mapped.data == address && mapped.size == 24, "contiguous CPU view was copied");
    gst_buffer_unmap(visible, &mapped);
    view.storage.reset();
    require(!owner.expired(), "CPU envelope did not retain its allocation");
    descriptor = {};
    envelope.reset();
    require(owner.expired(), "CPU envelope leaked its allocation");
  }

  source.byte_offset = 16;
  source.strides_bytes = {16, 4};
  auto* source_values = static_cast<float*>(source.storage->data);
  std::fill_n(source_values, 32, -1000.f);
  const std::array<float, 6> expected_values{1.f, 3.f, 2.f, 4.f, 6.f, 5.f};
  std::copy_n(expected_values.data(), 3, source_values + 4);
  std::copy_n(expected_values.data() + 3, 3, source_values + 8);
  auto compacted = pi::sample_to_gst_envelope_holder(sample_from_tensors({source}), &error);
  require(compacted != nullptr, "could not compact strided CPU view: " + error);
  std::array<float, 6> actual_values{};
  require(gst_buffer_extract(gst_sample_get_buffer(static_cast<GstSample*>(compacted.get())), 0,
                             actual_values.data(),
                             sizeof(actual_values)) == sizeof(actual_values) &&
              actual_values == expected_values,
          "singleton CPU materialization copied padding instead of logical rows");
  current = compile(sample_from_tensors({source}));
  require(current.kernel->runtime_contract.logical_inputs[0].stride_bytes ==
              std::vector<std::int64_t>({12, 4}),
          "ArgMax retained pre-materialization CPU strides");

  for (const auto offset :
       {std::int64_t{-1}, std::int64_t{120}, std::numeric_limits<std::int64_t>::max()}) {
    source.byte_offset = offset;
    ContractCompileInput invalid;
    invalid.ingress.ingress_sample = sample_from_tensors({source});
    require(!node.compile_node_contract(invalid, &current, &error),
            "ArgMax accepted an ingress view extending beyond its current storage");
  }
}

void execute_argmax_graph(bool scalar) {
  using namespace simaai::neat;
  const auto shape = scalar ? std::vector<std::int64_t>{6} : std::vector<std::int64_t>{2, 3};
  // GStreamer transport imports DMA-BUFs independently of the A65 execution
  // target. Request shared CMA explicitly through the public Tensor API.
  auto input = Tensor::from_vector(std::vector<float>(6, 0.f), shape, TensorMemory::EV74);
  Graph graph;
  InputOptions source;
  source.payload_type = PayloadType::Tensor;
  source.memory_policy = InputMemoryPolicy::Ev74; // Shared CMA, also A65-mappable.
  graph.add(nodes::Input(source));
  graph.add(nodes::SimaArgMax());
  graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
  RunOptions options;
  options.queue_depth = 2;
  options.output_memory = OutputMemory::Owned;
  options.advanced.copy_input = false;
  auto run = graph.build(TensorList{input}, options);
  for (int frame = 0; frame < 3; ++frame) {
    const int first = frame % 3;
    const int second = (frame + 1) % 3;
    std::array<float, 6> values{-5, -5, -5, -7, -7, -7};
    values[first] = 10;
    values[3 + second] = 20;
    input = Tensor::from_vector(std::vector<float>(values.begin(), values.end()), shape,
                                TensorMemory::EV74);
    require(run.push(TensorList{input}), "Core ArgMax input was not accepted");
    auto sample = run.pull(5000);
    require(sample.has_value(), "Core ArgMax public output timed out");
    const auto outputs = tensors_from_sample(*sample, true);
    require(outputs.size() == 1 && outputs[0].dtype == TensorDType::Int32,
            "Core ArgMax did not publish one INT32 tensor");
    const auto expected_shape = scalar ? std::vector<std::int64_t>{} : std::vector<std::int64_t>{2};
    require(outputs[0].shape == expected_shape, "Core ArgMax public reduction shape changed");
    const auto host = outputs[0].cpu().contiguous();
    const auto* actual = host.data_ptr<std::int32_t>();
    require(actual &&
                (scalar ? actual[0] == 3 + second : actual[0] == first && actual[1] == second),
            "Core ArgMax numerical result disagrees with the submitted frame");
    const auto bytes = outputs[0].copy_dense_bytes_tight();
    const auto cloned = outputs[0].clone();
    require(bytes.size() == (scalar ? 1U : 2U) * sizeof(std::int32_t) &&
                std::memcmp(bytes.data(), actual, bytes.size()) == 0 &&
                outputs[0].copy_payload_bytes() == bytes && cloned.shape == expected_shape &&
                cloned.byte_offset == 0 && cloned.copy_dense_bytes_tight() == bytes &&
                cloned.storage != outputs[0].storage,
            "Core ArgMax public copy/clone lost the reduction result or scalar rank");
  }
  run.close();
}
void execute_decode_graph(bool features,
                          std::optional<simaai::neat::TensorDType> padded_type = std::nullopt) {
  using namespace simaai::neat;
  const int side = features ? 2 : 8;
  const int image_side = side * 8;
  const auto make_heads = [&](int frame) {
    const int cell = frame % (side * side);
    const bool active = frame != 1;
    std::vector<float> first(side * side * (features ? 65 : 4), features ? -20.f : 1.f);
    std::vector<float> second(side * side * (features ? 256 : 2), features ? 1.f : -20.f);
    if (features) {
      for (int i = 0; i < side * side; ++i)
        first[i * 65 + 64] = 20.f;
      if (active)
        first[cell * 65 + 36] = 30.f;
    } else if (active)
      second[cell * 2] = 10.f;
    TensorList tensors{
        Tensor::from_vector(first, {side, side, features ? 65 : 4}, TensorMemory::EV74),
        Tensor::from_vector(second, {side, side, features ? 256 : 2}, TensorMemory::EV74)};
    if (padded_type) {
      // Poison physical-only channels. Native dense readers must ignore them,
      // not reinterpret interleaved scalar storage as MLA byte planes.
      if (*padded_type == TensorDType::BFloat16) {
        std::vector<std::uint16_t> packed(side * side * 80, 0x7fc1);
        for (int pixel = 0; pixel < side * side; ++pixel)
          for (int channel = 0; channel < 65; ++channel)
            packed[pixel * 80 + channel] = static_cast<std::uint16_t>(
                std::bit_cast<std::uint32_t>(first[pixel * 65 + channel]) >> 16);
        tensors[0] = Tensor::from_vector(packed, {side, side, 80}, TensorMemory::EV74);
        tensors[0].dtype = TensorDType::BFloat16;
      } else if (*padded_type == TensorDType::Int8) {
        std::vector<std::int8_t> packed(side * side * 80, 127);
        for (int pixel = 0; pixel < side * side; ++pixel)
          for (int channel = 0; channel < 65; ++channel)
            packed[pixel * 80 + channel] = static_cast<std::int8_t>(first[pixel * 65 + channel]);
        tensors[0] = Tensor::from_vector(packed, {side, side, 80}, TensorMemory::EV74);
        QuantSpec quant;
        quant.scales = {1.f};
        quant.zero_points = {0};
        tensors[0].semantic.quant = quant;
      } else {
        std::vector<float> packed(side * side * 80, std::numeric_limits<float>::quiet_NaN());
        for (int pixel = 0; pixel < side * side; ++pixel)
          std::copy_n(first.data() + pixel * 65, 65, packed.data() + pixel * 80);
        tensors[0] = Tensor::from_vector(packed, {side, side, 80}, TensorMemory::EV74);
      }
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      tensors[i].layout = TensorLayout::HWC;
      tensors[i].route.name = std::string("head_") + std::to_string(i);
      PreprocessRuntimeMeta provenance;
      provenance.original_width = provenance.resized_width = provenance.scaled_width = image_side;
      provenance.original_height = provenance.resized_height = provenance.scaled_height =
          image_side;
      provenance.resize_mode = "letterbox";
      provenance.color_in = provenance.color_out = "rgb";
      provenance.axis_perm = {0, 1, 2};
      if (!features)
        tensors[i].semantic.preprocess = std::move(provenance);
    }
    return tensors;
  };
  BoxDecodeOptions decode(features ? BoxDecodeType::SuperPoint : BoxDecodeType::YoloV26);
  decode.top_k = 4;
  decode.detection_threshold = .5;
  decode.nms_iou_threshold = features ? 0.0 : .5;
  if (features) {
    decode.superpoint.profile = SuperPointProfile::A65V1;
    decode.superpoint.nms_radius = 0;
    decode.superpoint.border_margin = 0;
  }
  Graph graph;
  InputOptions source;
  source.payload_type = PayloadType::Tensor;
  source.memory_policy = InputMemoryPolicy::Ev74;
  graph.add(nodes::Input(source));
  if (padded_type) {
    namespace si = simaai::neat::pipeline_internal::sima;
    auto logical = make_heads(0);
    logical[0].shape.back() = 65;
    std::string error;
    auto contract = si::build_boxdecode_static_contract_from_sample(
        sample_from_tensors(logical), BoxDecodeType::SuperPoint, std::nullopt, {}, &error);
    require(contract.has_value(), "padded SuperPoint fixture contract: " + error);
    // This is the same authored physical/slice distinction supplied by MPK.
    contract->tensors[0].input_shape.back() = 80;
    contract->tensors[0].slice_shape.back() = 65;
    const auto physical_bytes =
        static_cast<std::uint64_t>(side * side * 80 *
                                   (*padded_type == TensorDType::Int8       ? 1
                                    : *padded_type == TensorDType::BFloat16 ? 2
                                                                            : 4));
    contract->tensors[0].source_size_bytes = physical_bytes;
    contract->physical_inputs[0].size_bytes = physical_bytes;
    contract->superpoint.profile = SuperPointProfile::A65V1;
    contract->superpoint.nms_radius = 0;
    contract->superpoint.border_margin = 0;
    graph.add(nodes::SimaBoxDecode(*contract, BoxDecodeType::SuperPoint, .5, 0, 4, "decode", {},
                                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                                   std::nullopt, std::nullopt, image_side, image_side, image_side,
                                   image_side));
  } else {
    graph.add(nodes::SimaBoxDecode(decode, "decode", image_side, image_side, image_side, image_side,
                                   BoxDecodeTypeOption::InterleavedByHeadLogit,
                                   BoxDecodeSourceStorage::DenseHwc, false, false));
  }
  graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
  RunOptions options;
  options.queue_depth = 2;
  options.output_memory = OutputMemory::Owned;
  auto run = graph.build(make_heads(0), options);
  for (int frame = 0; frame < 3; ++frame) {
    require(run.push(make_heads(frame)), "Core decode input was not accepted");
    auto sample = run.pull(5000);
    require(sample.has_value(), "Core decode public output timed out");
    const auto tensors = tensors_from_sample(*sample, true);
    require(tensors.size() == 1 && tensors[0].dtype == TensorDType::UInt8,
            "Core decode must publish exactly one public wire tensor, no private intermediates");
    const std::size_t expected = frame == 1 ? 0 : 1;
    if (features) {
      const auto output = decode_superpoint_tensor(tensors[0]);
      require(output.keypoints.shape ==
                  std::vector<std::int64_t>({static_cast<std::int64_t>(expected), 2}),
              "Core SuperPoint wire count/shape mismatch");
      require(output.descriptors.shape ==
                  std::vector<std::int64_t>({static_cast<std::int64_t>(expected), 256}),
              "Core SuperPoint descriptor shape mismatch");
      if (expected) {
        const auto points = output.keypoints.cpu().contiguous();
        const auto* xy = points.data_ptr<float>();
        const auto descriptors = output.descriptors.cpu().contiguous();
        const auto* d = descriptors.data_ptr<float>();
        require(xy && xy[0] == (frame % side) * 8 + 4 && xy[1] == (frame / side) * 8 + 4,
                "Core SuperPoint keypoint coordinates changed");
        require(d && std::abs(d[0] - .0625f) < 1e-5f,
                "Core SuperPoint normalized descriptor changed");
      }
    } else {
      const auto output = decode_bbox_tensor(tensors[0], image_side, image_side, 4, true);
      require(output.boxes.size() == expected, "Core BoxDecode wire count mismatch");
      if (expected)
        require(output.boxes[0].class_id == 0 && output.boxes[0].score > .999f &&
                    output.boxes[0].x2 > output.boxes[0].x1,
                "Core BoxDecode candidate values changed");
    }
  }
  run.close();
}
void execute_rcnn_stage1_graph() {
  using namespace simaai::neat;
  const auto heads = [](bool active) {
    TensorList result;
    int level = 0;
    for (int side : {136, 68, 34, 17, 9}) {
      std::vector<float> scores(side * side * 3, -20.f);
      if (active)
        scores[(side / 2 * side + side / 2) * 3 + 1] = float(6 - level);
      result.push_back(Tensor::from_vector(std::vector<float>(side * side * 12, 0.f),
                                           {side, side, 12}, TensorMemory::EV74));
      result.push_back(Tensor::from_vector(scores, {side, side, 3}, TensorMemory::EV74));
      ++level;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i].layout = TensorLayout::HWC;
      result[i].route.name = "proposal_head_" + std::to_string(i);
      PreprocessRuntimeMeta meta;
      meta.original_width = meta.original_height = meta.resized_width = meta.resized_height =
          meta.scaled_width = meta.scaled_height = 544;
      meta.resize_mode = "letterbox";
      meta.color_in = meta.color_out = "rgb";
      meta.axis_perm = {0, 1, 2};
      result[i].semantic.preprocess = meta;
    }
    return result;
  };
  Graph graph;
  InputOptions source;
  source.payload_type = PayloadType::Tensor;
  source.memory_policy = InputMemoryPolicy::Ev74;
  graph.add(nodes::Input(source));
  BoxDecodeOptions options(BoxDecodeType::RcnnStage1);
  options.top_k = 1;
  options.detection_threshold = .5;
  options.nms_iou_threshold = 1.;
  graph.add(nodes::SimaBoxDecode(options, "proposals", 544, 544, 544, 544,
                                 BoxDecodeTypeOption::InterleavedByHeadLogit,
                                 BoxDecodeSourceStorage::DenseHwc, false, false));
  graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
  RunOptions run_options;
  run_options.output_memory = OutputMemory::Owned;
  auto run = graph.build(heads(true), run_options);
  for (bool active : {true, false, true}) {
    require(run.push(heads(active)), "RCNN input was not accepted");
    auto sample = run.pull(5000);
    require(sample.has_value(), "RCNN public graph timed out");
    const auto outputs = tensors_from_sample(*sample, true);
    require(outputs.size() == 1, "RCNN private intermediates leaked into public output");
    const auto host = outputs[0].cpu().contiguous();
    const auto* bytes = host.data_ptr<std::uint8_t>();
    require(host.shape == std::vector<std::int64_t>{64}, "RCNN four-level wire size");
    for (int level = 0; level < 4; ++level) {
      std::int32_t count;
      std::memcpy(&count, bytes + level * 16, 4);
      require(count == (active ? 1 : 0), "RCNN per-level top-K public count");
      if (active) {
        // Independent IEEE-half goldens for the four square-anchor winners:
        // score, reserved label, clipped x1,y1,x2,y2. Level-zero's second
        // higher-scoring anchor must not replace the lower-scoring level three.
        constexpr std::array<std::array<std::uint16_t, 6>, 4> expected{
            {{0x3bfb, 0, 0x5c00, 0x5c00, 0x5c80, 0x5c80},
             {0x3bdb, 0, 0x5a80, 0x5a80, 0x5d40, 0x5d40},
             {0x3b9f, 0, 0x5800, 0x5800, 0x5e00, 0x5e00},
             {0x3b0c, 0, 0, 0, 0x5fc0, 0x5fc0}}};
        for (int field = 0; field < 6; ++field) {
          std::uint16_t value;
          std::memcpy(&value, bytes + level * 16 + 4 + field * 2, 2);
          require(value == expected[level][field], "RCNN public proposal wire differs from oracle");
        }
      } else
        for (int field = 4; field < 16; ++field)
          require(bytes[level * 16 + field] == 0, "RCNN stale wire proposal");
    }
  }
  run.close();
}

} // namespace

int main() {
  try {
    const bool has_neatargmax = simaai::neat::element_exists("neatkernelrunner");
    const bool has_neatrender = simaai::neat::element_exists("neatrender");
    if (has_neatargmax) {
      simaai::neat::require_element("neatkernelrunner", "unit_sima_plugins_test");
    }
    if (has_neatrender) {
      simaai::neat::require_element("neatrender", "unit_sima_plugins_test");
    }

    {
      simaai::neat::SimaArgMaxOptions opt;
      opt.axis = -1;
      opt.keepdims = false;
      auto node = simaai::neat::nodes::SimaArgMax(opt);
      require(has_neatargmax, "required generic kernel element missing");
      const std::string frag = node->backend_fragment(2);
      require(frag == "neatkernelrunner name=n2_argmax", "ArgMax must use typed Runner admission");
      namespace si = simaai::neat::pipeline_internal::sima;
      namespace specs = si::specbuilders;
      simaai::neat::CompiledNodeContract upstream;
      upstream.processcvu.emplace();
      auto& runtime = upstream.processcvu->runtime_contract;
      runtime.physical_outputs.push_back(
          specs::build_physical_buffer_static_spec(7, 0, 128, si::DeviceKind::Cpu, "scores"));
      runtime.logical_outputs.push_back(specs::build_logical_output_static_spec(
          3, 4, 7, 0, 0, {2, 3}, "FP32", "", "scores", "scores", "scores", 16, 24));
      runtime.logical_outputs.back().stride_bytes = {16, 4};
      simaai::neat::ContractCompileInput input;
      input.node_index = 2;
      input.immediate_upstream = &upstream;
      simaai::neat::CompiledNodeContract compiled;
      std::string error;
      auto* compiler = dynamic_cast<simaai::neat::NodeContractProvider*>(node.get());
      require(compiler && compiler->compile_node_contract(input, &compiled, &error),
              "ArgMax compile: " + error);
      require(compiled.kernel && compiled.kernel->recipe->members.size() == 1,
              "ArgMax needs one checked native member, not a synthetic graph ID");
      const auto& native = *compiled.kernel;
      require(native.runtime_contract.logical_outputs[0].shape == std::vector<std::int64_t>{2} &&
                  native.runtime_contract.logical_outputs[0].dtype == "INT32",
              "ArgMax output contract mismatch");
      require(native.recipe->members[0].inputs[0].byte_offset == 16 &&
                  native.runtime_contract.physical_inputs[0].physical_index == 7,
              "ArgMax lost packed source identity or tensor offset");
      opt.keepdims = true;
      simaai::neat::SimaArgMax keepdims(opt);
      require(keepdims.compile_node_contract(input, &compiled, &error),
              "ArgMax keepdims compile: " + error);
      require(compiled.kernel->runtime_contract.logical_outputs[0].shape ==
                  std::vector<std::int64_t>({2, 1}),
              "ArgMax keepdims shape mismatch");
      opt.axis = 2;
      simaai::neat::SimaArgMax invalid_axis(opt);
      require(!invalid_axis.compile_node_contract(input, &compiled, &error),
              "ArgMax accepted axis beyond rank");
      input.immediate_upstream = nullptr;
      require(!compiler->compile_node_contract(input, &compiled, &error),
              "ArgMax accepted absent typed input");
      simaai::neat::Tensor ingress;
      ingress.storage = simaai::neat::make_cpu_owned_storage(64);
      ingress.dtype = simaai::neat::TensorDType::Float32;
      ingress.shape = {6};
      ingress.byte_offset = 16;
      input.ingress.ingress_sample = simaai::neat::make_tensor_sample("scores", ingress);
      require(compiler->compile_node_contract(input, &compiled, &error),
              "ArgMax typed ingress compile: " + error);
      require(compiled.kernel->runtime_contract.logical_outputs[0].shape.empty() &&
                  compiled.kernel->runtime_contract.logical_outputs[0].size_bytes == 4 &&
                  compiled.kernel->recipe->members[0].inputs[0].byte_offset == 0,
              "ArgMax scalar reduction or materialized ingress offset changed");
      simaai::neat::CompiledPipelineContracts pipeline;
      pipeline.stages = {compiled};
      pipeline.fully_renderable = true;
      si::ManifestBuildDiagnostics diagnostics;
      auto manifest =
          simaai::neat::render_manifest_from_compiled_contracts(pipeline, input, &diagnostics);
      require(manifest && diagnostics.errors.empty(), "ArgMax native scalar manifest failed");
      simaai::scheduler::client::TensorStagePublication publication;
      require(si::build_kernel_stage_publication(manifest->stages[0], &publication, &error),
              "ArgMax scalar publication: " + error);
      require(publication.contract->logical_outputs.size() == 1 &&
                  publication.contract->logical_outputs[0].shape.empty() &&
                  publication.contract->logical_outputs[0].size_bytes == 4,
              "ArgMax scalar publication invented an axis or lost its storage extent");
      ingress.byte_offset = 64;
      input.ingress.ingress_sample = simaai::neat::make_tensor_sample("scores", ingress);
      require(!compiler->compile_node_contract(input, &compiled, &error),
              "ArgMax accepted out-of-storage ingress view");
      opt.config_path = "/tmp/argmax.json";
      bool rejected = false;
      try {
        simaai::neat::SimaArgMax unsupported(opt);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "ArgMax silently ignored retired JSON configuration");

      auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
      require(provider != nullptr, "argmax OutputSpecProvider missing");
      simaai::neat::OutputSpec spec = provider->output_spec({});
      require(spec.media_type == "application/vnd.simaai.tensor",
              "argmax output media_type mismatch");
      require(spec.format == "ARGMAX", "argmax output format mismatch");
      check_argmax_ingress_projection();
    }

    {
      simaai::neat::SimaRenderOptions opt;
      opt.config_path = "/tmp/render.json";
      opt.sima_allocator_type = 2;
      opt.silent = false;
      opt.emit_signals = true;
      opt.transmit = false;

      auto node = simaai::neat::nodes::SimaRender(opt);
      if (has_neatrender) {
        const std::string frag = node->backend_fragment(3);
        require_contains(frag, "neatrender name=n3_render", "render fragment name mismatch");
        require_contains(frag, "config=\"/tmp/render.json\"", "render config missing");
        require_contains(frag, "sima-allocator-type=2", "render allocator missing");
        require_contains(frag, "silent=false", "render silent missing");
        require_contains(frag, "emit-signals=true", "render emit-signals missing");
        require_contains(frag, "transmit=false", "render transmit missing");
      }

      auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
      require(provider != nullptr, "render OutputSpecProvider missing");
      simaai::neat::OutputSpec spec = provider->output_spec({});
      require(spec.note == "neatrender", "render output note mismatch");
    }

    execute_argmax_graph(false);
    execute_argmax_graph(true);
    execute_decode_graph(false);
    execute_rcnn_stage1_graph();
    execute_decode_graph(true);
    for (auto type : {simaai::neat::TensorDType::BFloat16, simaai::neat::TensorDType::Int8,
                      simaai::neat::TensorDType::Float32})
      execute_decode_graph(true, type);
    std::cout << "[OK] unit_sima_plugins_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
