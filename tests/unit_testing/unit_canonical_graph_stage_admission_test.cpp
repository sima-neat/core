#include "pipeline/internal/sima/CanonicalGraphStageContract.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"

#include <gst/SimaPluginStaticManifestAbi.h>
#include <simaai/scheduler/client/dma_memory.h>
#include <simaai/scheduler/client/stage.h>
#include <simaai/scheduler/testing/route_probe.h>
#include <gstsimaaitensorbuffer_meta.h>

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// Standalone host builds compile the real TensorBuffer provider without its
// platform logger DSO. This is its externally declared GStreamer category.
#ifdef SIMA_CORE_GRAPH_ADMISSION_HOST_TEST
GST_DEBUG_CATEGORY(GST_CAT_DEFAULT);
#endif

namespace {
namespace ps = simaai::neat::pipeline_internal::sima;
namespace c = simaai::scheduler::client;
namespace test = simaai::scheduler::testing;
namespace tr = simaai::gst;
namespace dma = c::dma;

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T take(c::Result<T> result) {
  if (const auto* error = std::get_if<c::Error>(&result))
    throw std::runtime_error(error->field + ": " + error->message);
  return std::get<T>(std::move(result));
}
void okay(c::Status status) {
  if (status)
    throw std::runtime_error(status->field + ": " + status->message);
}

ps::MpkTensorContract tensor(const char* name, const char* dtype, std::uint64_t bytes) {
  ps::MpkTensorContract result;
  result.tensor_index = result.physical_index = result.source_physical_index = 0;
  result.name = result.segment_name = name;
  result.dtype = dtype;
  result.dtype_source = ps::DTypeSource::ExplicitMpk;
  result.logical_shape = {2, 3, 4};
  result.size_bytes = bytes;
  return result;
}

ps::StageStaticSpec complete(std::int64_t logical_offset, bool retained_physical) {
  ps::StageStaticSpec stage;
  stage.element_name = stage.logical_stage_id = "graph-cast";
  sima_ev_tensor_desc input{}, output_descriptor{};
  input.dtype = SIMA_EV_DTYPE_FP32;
  input.storage.nbytes = 96;
  output_descriptor.dtype = SIMA_EV_DTYPE_BF16;
  output_descriptor.storage.nbytes = 48;
  stage.processcvu.input_tensors = {{input, "FP32"}};
  stage.processcvu.output_tensors = {{output_descriptor, "BF16"}};
  stage.frame_arena_size_bytes = 4096;
  stage.frame_arena_role = ps::FrameArenaRole::Allocate;
  stage.frame_arena_storage_domain = ps::static_contract::ArenaStorageDomain::Cma;
  stage.frame_arena_provenance = ps::static_contract::ArenaAllocationProvenance::CoreAllocated;
  stage.frame_arena_required_device_access =
      static_cast<std::uint32_t>(ps::static_contract::ArenaDeviceAccess::CpuA65);
  ps::PhysicalBufferStaticSpec output;
  output.physical_index = output.allocator_index = 0;
  output.size_bytes = 48;
  output.source_byte_offset = 256;
  output.required_alignment_bytes = 64;
  output.segment_name = "result";
  stage.physical_outputs.push_back(output);
  if (retained_physical) {
    auto physical = output;
    physical.source_physical_index = 7;
    physical.source_byte_offset = 512;
    physical.size_bytes = 2048;
    physical.segment_name = "compiled-parent";
    physical.address_source = ps::PhysicalAddressSource::RuntimePhysicalBinding;
    stage.physical_inputs.push_back(physical);
  }
  std::array inputs{tensor("source", "FP32", 96)};
  inputs[0].source_byte_offset = 128;
  inputs[0].byte_offset = logical_offset;
  const std::array outputs{tensor("result", "BF16", 48)};
  std::string error;
  const bool completed = ps::complete_graph_kernel_contract(&stage, inputs, outputs, "FP32", "BF16",
                                                            "source", {}, &error);
  require(completed, error.c_str());
  return stage;
}

// Scheduler's test-only probe observes the actual carrier provider; Core owns
// only the graph facts and public request, not Storage/Frame implementation.
c::StageRequest request(const ps::StageStaticSpec& stage) {
  ps::SimaPluginStaticManifest manifest;
  manifest.stages = {stage};
  ps::finalize_pipeline_storage(manifest);
  c::StageRequest result;
  result.layout = manifest.stages.front().frame_layout;
  result.options.capacity = 1;
  const std::array input{c::TensorSpec{{2, 3, 4}, c::DType::Float32}};
  const std::array output{c::TensorSpec{{2, 3, 4}, c::DType::BFloat16}};
  c::KernelProgram program;
  program.target = c::Hardware::A65;
  auto executable = take(c::check_kernel({"cast", program.target, input, output, {}}));
  program.members.push_back({std::move(executable), {{0, 0}}, {{0, 0}}, {}});
  result.program = std::move(program);
  std::string error;
  require(ps::build_kernel_stage_publication(manifest.stages.front(), &result.publication, &error),
          error.c_str());
  return result;
}

void admitted_coordinates(std::int64_t logical_offset, bool retained_physical) {
  const auto model = complete(logical_offset, retained_physical);
  const auto& physical = model.physical_inputs.at(0);
  const auto& route = model.input_bindings.at(0);
  require(route.src_physical_output_index == (retained_physical ? 7 : 0) &&
              route.src_physical_byte_offset == 0 &&
              route.src_physical_size_bytes == physical.size_bytes &&
              route.source_segment_name == physical.segment_name,
          "inferred binding contradicts authoritative carrier or relative logical offset");
  const auto recipe = request(model);
  const auto& input_route = recipe.layout->inputs.at(0);
  const auto matches_carrier = [&](const auto& source) {
    return source.byte_offset == std::size_t(physical.source_byte_offset) &&
           source.size_bytes == physical.size_bytes;
  };
  require((retained_physical ? matches_carrier(std::get<c::PhysicalBinding>(input_route.source))
                             : matches_carrier(std::get<c::DirectMemory>(input_route.source))) &&
              input_route.view_offset == std::size_t(logical_offset),
          "final input layout must retain the carrier base and logical view exactly once");
  test::RouteProbe probe;
  okay(probe.prepare(recipe));
  require(probe.member_count() == 1, "cold Cast admission failed");
  require(probe.read_offsets().empty(), "cold preparation imported a current input carrier");

  dma::Error allocation_error;
  std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> input(
      dma::allocateDmaBufBuffer(dma::HeapKind::Cma, 4096, {}, &allocation_error), gst_buffer_unref);
  require(bool(input), allocation_error.message().c_str());
  constexpr std::size_t runtime_base = 64;
  if (retained_physical) {
    std::vector<tr::TensorBufferPhysicalBinding> bindings(8, {0, 0, 4096});
    bindings[7] = {0, runtime_base, 4096 - runtime_base};
    tr::TensorBufferPublishContract publication;
    publication.stage_key = "producer";
    publication.physical_outputs.push_back({0, 4096, "source"});
    auto logical = recipe.publication.contract->logical_outputs[0];
    logical.logical_name = logical.backend_name = logical.segment_name = "source";
    logical.dtype = SIMA_TENSOR_SET_DTYPE_FP32_V1;
    logical.byte_offset = runtime_base + physical.source_byte_offset + logical_offset;
    logical.size_bytes = 96;
    logical.stride_bytes = {48, 16, 4};
    publication.logical_outputs.push_back(logical);
    publication.output_order.push_back({0, 0, "source", "source"});
    std::string error;
    const bool published = tr::tensor_buffer_publish_outputs_from_contract(
        input.get(), publication, tr::TensorBufferProducerKind::Transport, &error);
    require(published, error.c_str());
    const bool attached = tr::tensor_buffer_attach_physical_bindings(input.get(), bindings, &error);
    require(attached, error.c_str());
  }
  // No numerical member is submitted. The probe performs the original
  // acquire/bind/retire sequence and releases its actual current producer loans.
  okay(probe.bind_and_retire(input.get(), 17, 123, "route-test"));
  require(
      probe.read_offsets() ==
              std::vector<std::size_t>{std::size_t(physical.source_byte_offset + logical_offset) +
                                       (retained_physical ? runtime_base : 0)} &&
          probe.read_lengths() == std::vector<std::size_t>{96},
      "actual Scheduler binding doubled the upstream base, lost the logical offset, or chose the "
      "wrong source ID");
  require(gst_buffer_is_writable(input.get()),
          "unsubmitted route retained its current producer loan");
}

void rejects_double_base() {
  auto model = complete(64, false);
  model.input_bindings[0].src_physical_byte_offset = model.physical_inputs[0].source_byte_offset;
  bool rejected = false;
  try {
    (void)request(model);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "doubled-base route was not rejected during cold Core layout finalization");
}
} // namespace

int main() {
  try {
    gst_init(nullptr, nullptr);
#ifdef SIMA_CORE_GRAPH_ADMISSION_HOST_TEST
    GST_DEBUG_CATEGORY_INIT(GST_CAT_DEFAULT, "core-graph-stage-admission", 0, "route admission");
#endif
    require(tr::tensor_buffer_register_meta(), "TensorBuffer metadata registration failed");
    rejects_double_base();
    admitted_coordinates(0, false);
    admitted_coordinates(64, false);
    admitted_coordinates(64, true);
    std::cout << "Core graph projection through generic Scheduler admission/binding PASS (no "
                 "submission)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
