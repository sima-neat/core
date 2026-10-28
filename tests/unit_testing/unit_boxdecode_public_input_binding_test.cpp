#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"

#include <gst/gst.h>
#include <gstsimaaitensorbuffer_meta.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
using namespace simaai::neat;
namespace pi = simaai::neat::pipeline_internal;
namespace ps = simaai::neat::pipeline_internal::sima;
namespace client = simaai::scheduler::client;
namespace transport = simaai::gst;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

enum class Packing { Separate, Shared, TwoGroups };
struct Fixture {
  Sample sample;
  std::vector<std::size_t> memory, offset, bytes, capacity;
};

// Real public views over ordinary CPU GstMemory. A shared GstSample holder is
// model-like transport, not a model dependency or a fabricated DMA allocation.
Fixture fixture(Packing packing, std::size_t prefix) {
  Fixture result;
  result.capacity.resize(packing == Packing::Separate ? 6
                         : packing == Packing::Shared ? 1
                                                      : 2,
                         prefix);
  std::vector<std::array<std::int64_t, 3>> shapes;
  for (int side : {8, 4, 2}) {
    for (int channels : {64, 80}) {
      const auto i = shapes.size();
      const auto memory = packing == Packing::Separate ? i : packing == Packing::Shared ? 0 : i / 3;
      const std::size_t bytes = side * side * channels * sizeof(float);
      shapes.push_back({side, side, channels});
      result.memory.push_back(memory);
      result.offset.push_back(result.capacity[memory]);
      result.bytes.push_back(bytes);
      result.capacity[memory] += bytes;
    }
  }
  auto buffer = std::shared_ptr<GstBuffer>(gst_buffer_new(), gst_buffer_unref);
  require(bool(buffer), "CPU fixture buffer allocation failed");
  for (const auto bytes : result.capacity) {
    auto* memory = gst_allocator_alloc(nullptr, bytes, nullptr);
    require(memory != nullptr, "CPU fixture memory allocation failed");
    gst_buffer_append_memory(buffer.get(), memory);
    GstMapInfo mapped{};
    require(gst_memory_map(memory, &mapped, GST_MAP_WRITE), "CPU fixture initialization failed");
    std::memset(mapped.data, 0, mapped.size);
    gst_memory_unmap(memory, &mapped);
  }
  auto caps = std::shared_ptr<GstCaps>(gst_caps_new_empty_simple("application/vnd.simaai.tensor"),
                                       gst_caps_unref);
  auto holder = std::shared_ptr<void>(
      gst_sample_new(buffer.get(), caps.get(), nullptr, nullptr),
      [](void* sample) { gst_sample_unref(static_cast<GstSample*>(sample)); });
  require(bool(holder), "CPU fixture sample allocation failed");
  std::vector<std::shared_ptr<TensorBuffer>> storage;
  for (const auto bytes : result.capacity) {
    auto selected = std::make_shared<TensorBuffer>();
    selected->kind = StorageKind::GstSample;
    selected->holder = holder;
    selected->size_bytes = bytes;
    const auto memory_index = storage.size();
    selected->map_fn = [holder, memory_index](MapMode mode) {
      auto* source = gst_sample_get_buffer(static_cast<GstSample*>(holder.get()));
      auto* memory = gst_buffer_peek_memory(source, memory_index);
      auto info = std::make_shared<GstMapInfo>();
      require(
          gst_memory_map(memory, info.get(), mode == MapMode::Read ? GST_MAP_READ : GST_MAP_WRITE),
          "CPU fixture mapping failed");
      Mapping mapping;
      mapping.data = info->data;
      mapping.size_bytes = info->size;
      mapping.keepalive = holder;
      mapping.unmap = [holder, memory, info] { gst_memory_unmap(memory, info.get()); };
      return mapping;
    };
    storage.push_back(std::move(selected));
  }
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    Tensor tensor;
    tensor.storage = storage[result.memory[i]];
    tensor.dtype = TensorDType::Float32;
    tensor.layout = TensorLayout::HWC;
    tensor.shape.assign(shapes[i].begin(), shapes[i].end());
    tensor.strides_bytes = {shapes[i][1] * shapes[i][2] * 4, shapes[i][2] * 4, 4};
    tensor.byte_offset = result.offset[i];
    tensor.route.stage_key = "producer";
    tensor.route.logical_index = tensor.route.backend_output_index = tensor.route.route_slot =
        static_cast<int>(i);
    tensor.route.physical_index = static_cast<int>(i);
    tensor.route.memory_index = static_cast<int>(result.memory[i]);
    tensor.route.physical_byte_offset = result.offset[i];
    tensor.route.name = tensor.route.backend_name = "head" + std::to_string(i);
    tensor.route.segment_name = "memory" + std::to_string(result.memory[i]);
    result.sample.tensors.push_back(std::move(tensor));
  }
  result.sample.kind = SampleKind::TensorSet;
  result.sample.payload_type = PayloadType::Tensor;
  result.sample.media_type = "application/vnd.simaai.tensor";
  return result;
}

std::string errors(const ps::ManifestBuildDiagnostics& diagnostics) {
  std::string result;
  for (const auto& error : diagnostics.errors)
    result += error + "\n";
  return result;
}

ps::SimaPluginStaticManifest compile(const Sample& sample) {
  ContractCompileInput input;
  input.ingress.ingress_sample = sample;
  const std::vector<std::shared_ptr<Node>> nodes{
      nodes::Input(),
      nodes::SimaBoxDecode(BoxDecodeType::YoloV8, .5, .5, 4, "decode", 64, 64, 64, 64,
                           BoxDecodeTypeOption::InterleavedByHead),
      nodes::Output()};
  ps::ManifestBuildDiagnostics diagnostics;
  const auto compiled = compile_node_contracts(nodes, input, &diagnostics);
  require(compiled.fully_renderable && diagnostics.errors.empty(),
          "BoxDecode compile: " + errors(diagnostics));
  auto manifest = render_manifest_from_compiled_contracts(compiled, input, &diagnostics);
  require(manifest.has_value() && diagnostics.errors.empty(),
          "BoxDecode render: " + errors(diagnostics));
  if (const auto error = ps::prepare_kernel_stage_recipes(&*manifest, std::nullopt, {}, {}, {}))
    throw std::runtime_error("BoxDecode recipe: " + error->message);
  ps::finalize_pipeline_storage(*manifest);
  require(manifest->stages.size() == 1 && manifest->stages.front().frame_layout,
          "standalone BoxDecode lost its finalized frame layout");
  return std::move(*manifest);
}

void check(const Fixture& fixture) {
  std::string error;
  const auto envelope = pi::sample_to_gst_envelope_holder(fixture.sample, &error);
  require(bool(envelope), "public envelope: " + error);
  auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(envelope.get()));
  require(gst_buffer_n_memory(buffer) == fixture.capacity.size(),
          "public envelope changed the expected allocation count");
  for (std::size_t i = 0; i < fixture.capacity.size(); ++i)
    require(gst_memory_get_sizes(gst_buffer_peek_memory(buffer, i), nullptr, nullptr) ==
                fixture.capacity[i],
            "public envelope changed the expected carrier capacity");

  const auto manifest = compile(fixture.sample);
  const auto& inputs = manifest.stages.front().frame_layout->inputs;
  require(inputs.size() == fixture.memory.size(), "BoxDecode lost a head input");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    // The public envelope uses tensor-position bindings. The compiled BoxDecode
    // contract has already resolved those into memory-coordinate carriers.
    transport::TensorBufferPhysicalBinding binding;
    require(transport::tensor_buffer_physical_binding_at(buffer, i, &binding, &error),
            "public binding: " + error);
    require(binding.memory_index == static_cast<int>(fixture.memory[i]) &&
                binding.byte_offset == fixture.offset[i] &&
                binding.size_bytes == fixture.capacity[fixture.memory[i]] - fixture.offset[i],
            "public tensor binding does not describe its actual view");
    const auto* source = std::get_if<client::DirectMemory>(&inputs[i].source);
    require(source && source->pad == 0 && source->memory_index == fixture.memory[i] &&
                source->byte_offset == 0 &&
                source->size_bytes == fixture.capacity[fixture.memory[i]] &&
                inputs[i].view_offset == fixture.offset[i],
            "BoxDecode relabeled a resolved memory span as a tensor binding or changed its view");
  }
}

void owned_materialization() {
  auto copied = fixture(Packing::Shared, 64);
  copied.capacity = copied.bytes;
  for (std::size_t i = 0; i < copied.sample.tensors.size(); ++i) {
    auto& tensor = copied.sample.tensors[i];
    tensor = tensor.clone();
    require(tensor.storage->kind == StorageKind::CpuOwned && tensor.byte_offset == 0 &&
                tensor.route.physical_byte_offset == static_cast<std::int64_t>(copied.offset[i]),
            "owned fixture must retain producer lineage but not producer storage offsets");
    copied.memory[i] = i;
    copied.offset[i] = 0;
  }
  copied.sample.owned = true;
  check(copied);
}
} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    require(transport::tensor_buffer_register_meta(), "TensorBuffer metadata registration failed");
    // Cold contract preparation only: no Graph/Model, DMA allocation, provider
    // frame binding or numerical submission. No Scheduler predicate is copied.
    for (const auto packing : {Packing::Separate, Packing::Shared, Packing::TwoGroups})
      for (const std::size_t prefix : {0U, 64U})
        check(fixture(packing, prefix));
    owned_materialization();
    std::cout << "BoxDecode public-input carrier coordinates PASS (cold contracts only)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
