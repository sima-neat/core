/**
 * @file
 * @ingroup internal_sima
 * @brief **Framework internal — reach-through tier.** Helpers for constructing static contract
 *        specs from raw geometry/dtype/layout inputs.
 *
 * Used during MPK ingestion and route planning to build the canonical
 * `LogicalInputStaticSpec` / `LogicalTensorStaticSpec` / `PhysicalBufferStaticSpec` / etc.
 * structs that go into a `SimaPluginStaticManifest`. Centralizing the construction here keeps
 * default sizing (e.g., shape×dtype byte size) and field-finalization rules in one place.
 *
 * @see SimaPluginStaticManifest.h for the spec types these builders produce.
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::specbuilders {

/// Bytes per element for a dtype token (e.g., `"FP32"` -> 4); 0 for unknown tokens.
std::uint64_t dtype_size_bytes_from_token(const std::string& raw_dtype);

/// Total tensor byte-size from shape and dtype (product of dims × element size).
std::uint64_t tensor_size_bytes_from_shape_dtype(const std::vector<std::int64_t>& shape,
                                                 const std::string& dtype);

/// Build a dense shape vector from `(width,height,depth)` ordered for the given layout token.
std::vector<std::int64_t> dense_shape_from_dims(int width,
                                                int height,
                                                int depth,
                                                const std::string& layout);

/**
 * @brief Build a `LogicalInputStaticSpec` with sensible defaults.
 *
 * Computes `size_bytes` from shape×dtype if `size_bytes_override` is 0; otherwise uses the
 * override. Other fields are populated directly from the arguments.
 */
LogicalInputStaticSpec build_logical_input_static_spec(
    int logical_index,
    int backend_input_index,
    int physical_index,
    const std::vector<std::int64_t>& shape,
    const std::string& dtype,
    const std::string& layout,
    const std::string& logical_name,
    const std::string& backend_name,
    const std::string& segment_name,
    std::int64_t byte_offset = 0,
    std::uint64_t size_bytes_override = 0,
    TensorMaterializationKind materialization_kind = TensorMaterializationKind::Direct,
    const std::optional<QuantStaticSpec>& quant = std::nullopt);

/**
 * @brief Build a `LogicalTensorStaticSpec` (logical output) with sensible defaults.
 *
 * Same sizing convention as `build_logical_input_static_spec`.
 */
LogicalTensorStaticSpec build_logical_output_static_spec(
    int logical_index,
    int backend_output_index,
    int physical_index,
    int output_slot,
    int tensor_index,
    const std::vector<std::int64_t>& shape,
    const std::string& dtype,
    const std::string& layout,
    const std::string& logical_name,
    const std::string& backend_name,
    const std::string& segment_name,
    std::int64_t byte_offset = 0,
    std::uint64_t size_bytes_override = 0,
    const std::optional<QuantStaticSpec>& quant = std::nullopt);

/// Build a `PhysicalBufferStaticSpec` for a stage's physical input or output buffer.
PhysicalBufferStaticSpec build_physical_buffer_static_spec(int physical_index,
                                                           int allocator_index,
                                                           std::uint64_t size_bytes,
                                                           DeviceKind device_kind,
                                                           const std::string& segment_name,
                                                           int source_physical_index = -1,
                                                           std::int64_t source_byte_offset = 0);

/// Build an `InputBindingStaticSpec` connecting one of this stage's inputs to an upstream output.
InputBindingStaticSpec build_input_binding_static_spec(int sink_pad_index,
                                                       int local_logical_input_index,
                                                       const std::string& cm_input_name,
                                                       const std::string& source_segment_name,
                                                       int src_logical_output_index = -1,
                                                       int src_output_slot = -1,
                                                       int src_physical_output_index = -1,
                                                       std::uint64_t src_physical_size_bytes = 0,
                                                       std::int64_t src_physical_byte_offset = 0,
                                                       bool required = true);

/// Build a `StageOutputRoute` describing one logical output's routing slot/tensor identity.
StageOutputRoute build_output_route_static_spec(int output_slot,
                                                int logical_output_index,
                                                int tensor_index,
                                                const std::string& cm_output_name,
                                                const std::string& segment_name);

/**
 * @brief Finalize a `LogicalInputStaticSpec` after all physical inputs are known.
 *
 * Resolves the spec's segment / backend names against the supplied physical-input name table.
 *
 * @param logical              In/out spec to finalize.
 * @param index                The spec's position in the parent stage's logical-input list.
 * @param physical_input_names Name table for the parent stage's physical inputs.
 */
void finalize_logical_input_spec(LogicalInputStaticSpec* logical,
                                 std::size_t index,
                                 const std::vector<std::string>& physical_input_names);

/**
 * @brief Finalize a `LogicalTensorStaticSpec` (logical output) after physical outputs are known.
 *
 * @param logical               In/out spec to finalize.
 * @param index                 The spec's position in the parent stage's logical-output list.
 * @param physical_output_names Name table for the parent stage's physical outputs.
 */
void finalize_logical_output_spec(LogicalTensorStaticSpec* logical,
                                  std::size_t index,
                                  const std::vector<std::string>& physical_output_names);

} // namespace simaai::neat::pipeline_internal::sima::specbuilders
