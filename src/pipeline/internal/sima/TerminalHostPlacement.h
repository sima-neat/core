#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <algorithm>
#include <limits>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {

// Cold eligibility only. A separate typed public endpoint and actual stopped
// direct-sink proof are mandatory before applying this placement.
inline bool terminal_host_arena_eligible(const StageStaticSpec& stage,
                                         const SimaPluginStaticManifest& manifest) {
  using namespace static_contract;
  namespace client = ::simaai::scheduler::client;
  if (!stage.core_fresh_default_arena || !stage.native_kernel || stage.image_publication ||
      stage.native_kernel->image_publication || stage.native_kernel->frame_adapter_factory ||
      stage.frame_arena_role != FrameArenaRole::Allocate ||
      stage.frame_arena_provenance != ArenaAllocationProvenance::CoreAllocated ||
      stage.frame_arena_storage_domain != ArenaStorageDomain::Cma ||
      stage.frame_arena_allocation_policy != ArenaAllocationPolicy::RequiredDomain ||
      stage.frame_arena_required_device_access !=
          static_cast<std::uint32_t>(ArenaDeviceAccess::CpuA65) ||
      stage.frame_arena_escape_policy != ArenaEscapePolicy::CpuMappablePublic ||
      stage.native_kernel->target != client::Hardware::A65 ||
      stage.native_kernel->members.empty() || stage.physical_outputs.empty() ||
      stage.logical_outputs.empty() || stage.output_order.empty() || !stage.frame_arena_size_bytes)
    return false;
  for (const auto& other : manifest.stages)
    for (const auto& input : other.input_bindings)
      if (!input.src_stage_id.empty() && (input.src_stage_id == stage.element_name ||
                                          input.src_stage_id == stage.logical_stage_id))
        return false;
  for (std::size_t i = 0; i < stage.physical_outputs.size(); ++i) {
    const auto& region = stage.physical_outputs[i];
    if (region.physical_index != static_cast<int>(i) || region.source_physical_index != -1 ||
        region.device_kind != DeviceKind::Cpu || region.memory_flags || !region.size_bytes ||
        region.source_byte_offset < 0 ||
        region.address_source != PhysicalAddressSource::DirectCarrierSpan ||
        region.required_alignment_bytes == 0 || region.required_alignment_bytes > 64 ||
        static_cast<std::uint64_t>(region.source_byte_offset) > stage.frame_arena_size_bytes ||
        region.size_bytes >
            stage.frame_arena_size_bytes - static_cast<std::uint64_t>(region.source_byte_offset))
      return false;
    for (std::size_t k = 0; k < i; ++k) {
      const auto& other = stage.physical_outputs[k];
      if (static_cast<std::uint64_t>(region.source_byte_offset) <
              static_cast<std::uint64_t>(other.source_byte_offset) + other.size_bytes &&
          static_cast<std::uint64_t>(other.source_byte_offset) <
              static_cast<std::uint64_t>(region.source_byte_offset) + region.size_bytes)
        return false;
    }
  }
  std::vector<bool> written(stage.physical_outputs.size(), false);
  struct Write {
    std::size_t tensor, offset, bytes, member;
  };
  std::vector<Write> writes;
  std::vector<std::vector<bool>> ancestors(
      stage.native_kernel->members.size(),
      std::vector<bool>(stage.native_kernel->members.size(), false));
  for (std::size_t member_index = 0; member_index < stage.native_kernel->members.size();
       ++member_index) {
    const auto& member = stage.native_kernel->members[member_index];
    for (auto dependency : member.dependencies) {
      if (dependency >= member_index)
        return false;
      ancestors[member_index][dependency] = true;
      for (std::size_t i = 0; i < dependency; ++i)
        ancestors[member_index][i] = ancestors[member_index][i] || ancestors[dependency][i];
    }
    const auto inspected = client::inspect_kernel(member.executable);
    const auto* checked = std::get_if<client::KernelInspection>(&inspected);
    if (!checked || checked->target() != client::Target::A65 ||
        checked->outputs().size() != member.outputs.size() ||
        checked->output_requirements().size() != member.outputs.size() ||
        checked->input_requirements().size() != member.inputs.size())
      return false;
    for (std::size_t i = 0; i < member.inputs.size(); ++i) {
      const auto& port = member.inputs[i];
      if (port.source != client::TensorStagePortSource::Output)
        continue;
      const auto bytes = checked->input_requirements()[i].storage_bytes;
      if (port.tensor >= stage.physical_outputs.size() ||
          port.byte_offset > stage.physical_outputs[port.tensor].size_bytes ||
          bytes > stage.physical_outputs[port.tensor].size_bytes - port.byte_offset)
        return false;
      if (std::none_of(writes.begin(), writes.end(), [&](const auto& write) {
            return write.tensor == port.tensor && ancestors[member_index][write.member] &&
                   write.offset <= port.byte_offset && bytes <= write.bytes &&
                   port.byte_offset - write.offset <= write.bytes - bytes;
          }))
        return false;
    }
    for (std::size_t i = 0; i < member.outputs.size(); ++i) {
      const auto& port = member.outputs[i];
      if (port.tensor >= stage.physical_outputs.size())
        return false;
      const auto& region = stage.physical_outputs[port.tensor];
      const auto& required = checked->output_requirements()[i];
      if (required.alignment_bytes > region.required_alignment_bytes ||
          port.byte_offset > region.size_bytes ||
          required.storage_bytes > region.size_bytes - port.byte_offset)
        return false;
      for (const auto& prior : writes)
        if (prior.tensor == port.tensor &&
            prior.offset < port.byte_offset + required.storage_bytes &&
            port.byte_offset < prior.offset + prior.bytes &&
            (prior.member == member_index || !ancestors[member_index][prior.member]))
          return false;
      writes.push_back({port.tensor, port.byte_offset, required.storage_bytes, member_index});
      written[port.tensor] = true;
    }
  }
  return std::all_of(written.begin(), written.end(), [](bool value) { return value; });
}

inline bool terminal_host_direct_peer(GstElement* pipeline, GstElement* stage, GstElement* sink) {
  if (!pipeline || !stage || !sink || !GST_IS_BIN(pipeline) ||
      !gst_object_has_as_ancestor(GST_OBJECT(stage), GST_OBJECT(pipeline)) ||
      !gst_object_has_as_ancestor(GST_OBJECT(sink), GST_OBJECT(pipeline)))
    return false;
  auto* factory = gst_element_get_factory(sink);
  if (!factory || !g_str_equal(gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)), "appsink"))
    return false;
  GstState state = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
  gst_element_get_state(pipeline, &state, &pending, 0);
  if (state > GST_STATE_READY || pending > GST_STATE_READY)
    return false;
  auto* src = gst_element_get_static_pad(stage, "src");
  auto* peer = src ? gst_pad_get_peer(src) : nullptr;
  auto* parent = peer ? gst_pad_get_parent_element(peer) : nullptr;
  const bool valid = parent == sink;
  if (parent)
    gst_object_unref(parent);
  if (peer)
    gst_object_unref(peer);
  if (src)
    gst_object_unref(src);
  return valid;
}

} // namespace simaai::neat::pipeline_internal::sima
