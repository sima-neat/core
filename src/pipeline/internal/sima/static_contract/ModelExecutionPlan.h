#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

using ValueId = std::uint32_t;
using OpId = std::uint32_t;
using CarrierId = std::uint32_t;
using TensorShape = std::vector<std::int64_t>;

enum class ValueRepresentation {
  Dense,
  Tessellated,
  Packed,
  BackendNative,
};

struct QuantizationSpec {
  double scale = 0.0;
  std::int64_t zero_point = 0;
};

// A compile-time tensor view over one canonical materialized carrier.  This is
// an address expression consumed by the next real kernel, not an execution
// operation and not a request to copy/repack bytes.  `byte_offset` is relative
// to the root carrier and `stride_bytes` describes how the logical shape is
// read from that address.
struct ReadExpression {
  ValueId source_value_id = 0;
  std::uint64_t byte_offset = 0;
  std::vector<std::int64_t> stride_bytes;
};

enum class StorageBindingKind {
  External,
  Root,
  View,
};

enum class StorageAccess {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

// Target-physical storage identity is independent of logical ValueId. Several
// semantic values may bind disjoint or aliased spans of one carrier; the arena
// allocates the carrier once.
struct StorageBinding {
  StorageBindingKind kind = StorageBindingKind::Root;
  CarrierId carrier_id = 0;
  std::uint64_t byte_offset = 0;
  std::uint64_t physical_span = 0;
  std::vector<std::int64_t> stride_bytes;
  StorageAccess access = StorageAccess::ReadWrite;
  std::optional<ValueId> source_value_id;
};

struct CarrierSpec {
  CarrierId id = 0;
  std::uint64_t required_bytes = 0;
  std::size_t required_alignment_bytes = 0;
  ValueRepresentation representation = ValueRepresentation::Dense;
};

struct ValueSpec {
  ValueId id = 0;
  std::string name;
  std::uint64_t required_bytes = 0;
  std::optional<std::string> logical_dtype;
  std::optional<TensorShape> logical_shape;
  std::optional<std::string> logical_layout;
  std::vector<QuantizationSpec> quantization;
  ValueRepresentation representation = ValueRepresentation::Dense;
  // Stock-AFE normalization initially expresses exact views this way;
  // create() lowers them into the single carrier/storage-binding model.
  std::optional<ReadExpression> read_expression;
  std::optional<StorageBinding> storage_binding;
};

enum class OpKind {
  Cast,
  Quantize,
  Tessellate,
  Pack,
  Mla,
  Unpack,
  Slice,
  Reshape,
  Detessellate,
  Dequantize,
  HostTvm,
  PassThrough,
};

struct CastOpConfig {
  std::string output_dtype;
};

struct QuantizeOpConfig {
  std::string output_dtype;
  std::int64_t num_bits = 0;
  std::string rounding;
  std::vector<QuantizationSpec> channel_params;
};

struct TessellateOpConfig {
  TensorShape slice_shape;
  bool align_c16 = false;
  bool cblock = false;
  std::string frame_type;
};

// Exact batch-1 placement of one Pack input in its materialized parent. New
// compiler contracts should emit this directly. The quarantined legacy EVO
// decoder derives it only from exact MPK order/bytes and AFE's 16-byte Pack
// rule; an incomplete parent remains executable only through a real Pack.
struct PackComponentPlacement {
  ValueId value_id = 0;
  std::uint64_t parent_offset = 0;
  std::uint64_t stored_bytes = 0;
};

struct PackSpan {
  ValueId value_id = 0;
  std::uint32_t batch_index = 0;
  std::uint64_t source_byte_offset = 0;
  std::uint64_t parent_offset = 0;
  std::uint64_t logical_bytes = 0;
  std::uint64_t stored_bytes = 0;
  std::string padding_policy;
};

struct PackOpConfig {
  // Frozen batch-one direct-placement adapter. New contracts use spans.
  std::vector<PackComponentPlacement> components;
  std::uint32_t batch_count = 1;
  std::uint64_t parent_required_bytes = 0;
  std::vector<PackSpan> spans;
  bool materializes = false;
};

struct HostTensorTypeSpec {
  std::string scalar;
  TensorShape shape;

  bool operator==(const HostTensorTypeSpec&) const = default;
};

struct MlaOpConfig {
  std::string executable;
  std::int64_t number_of_quads = 0;
  // Stock AFE 2.1 physical-port type facts. They describe logical values at
  // the MLA boundary; the ELF remains authority only for IFM/OFM topology.
  std::vector<HostTensorTypeSpec> input_types;
  std::vector<HostTensorTypeSpec> output_types;
  std::uint64_t executable_bytes = 0;
  std::string executable_sha256;
};

struct UnpackOpConfig {
  std::vector<std::string> tensor_types;
  std::vector<TensorShape> tensor_shapes;
};

struct SliceOpConfig {
  TensorShape begin;
  TensorShape end;
  TensorShape input_shape;
  TensorShape output_shape;
};

// A reshape is an address-only reinterpretation.  It never materializes bytes;
// validation proves that the input and output dense byte equations are equal.
struct ReshapeOpConfig {
  TensorShape new_shape;
};

struct DetessellateOpConfig {
  TensorShape slice_shape;
  TensorShape frame_shape;
  bool align_c16 = false;
  bool cblock = false;
  std::string frame_type;
};

struct DequantizeOpConfig {
  std::string input_dtype;
  std::vector<QuantizationSpec> channel_params;
};

// Exact AFE host-module contract.  The runtime loads `executable` once and
// binds these ordered DLTensor ports directly to the frame arena.  There is no
// dispatcher/config-manager interpretation and no staging tensor.
struct HostTvmOpConfig {
  std::string executable;
  std::vector<std::string> input_names;
  std::vector<HostTensorTypeSpec> input_types;
  std::vector<HostTensorTypeSpec> output_types;
  // -1 means a materialized graph-executor output.  Otherwise the output is
  // the exact compiler-authored __nop view of this input port.
  std::vector<std::int32_t> output_alias_input;
  // TVM GraphExecutor lists linked constants alongside external inputs as
  // null/arg nodes. The compiler authors this disjoint set explicitly; these
  // names are never bound from the frame arena.
  std::vector<std::string> linked_parameter_names;
  std::uint64_t executable_bytes = 0;
  std::string executable_sha256;
};

struct PassThroughOpConfig {};

using OpConfig =
    std::variant<CastOpConfig, QuantizeOpConfig, TessellateOpConfig, PackOpConfig, MlaOpConfig,
                 UnpackOpConfig, SliceOpConfig, ReshapeOpConfig, DetessellateOpConfig,
                 DequantizeOpConfig, HostTvmOpConfig, PassThroughOpConfig>;

struct OpSpec {
  OpId id = 0;
  std::uint64_t sequence = 0;
  std::string name;
  OpKind kind = OpKind::PassThrough;
  std::string processor;
  // Exact registry token. MLA uses the explicitly registered empty token
  // because its legacy MPK entry has no config_params.kernel member.
  std::string kernel;
  std::string implementation_id;
  std::uint32_t implementation_abi_version = 0;
  std::vector<OpId> dependencies;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  std::vector<TensorShape> input_shapes;
  std::vector<TensorShape> output_shapes;
  OpConfig config = PassThroughOpConfig{};
};

enum class BackendPortDirection { Input, Output };
enum class BackendPortAccess { ReadOnly, WriteOnly };
enum class BackendPortAlignmentAuthority { Contract, LegacyPolicy };

// Frozen legacy EVO manifests contain no port-alignment field. Section 6.2 of
// the migration contract deliberately over-aligns their common CMA regions to
// a page. This is policy provenance, never represented as an MPK/ELF fact.
inline constexpr std::size_t kLegacyEvoCmaRegionAlignmentBytes = 4096U;

struct BackendPortSpec {
  std::size_t stage_index = 0;
  BackendPortDirection direction = BackendPortDirection::Input;
  std::size_t port_index = 0;
  std::string elf_symbol;
  ValueId value_id = 0;
  // Exact compiler-authored physical address extent for this backend port.
  // ValueSpec::required_bytes remains the logical tensor byte count.
  std::uint64_t physical_extent_bytes = 0;
  std::size_t required_alignment_bytes = 0;
  BackendPortAlignmentAuthority alignment_authority = BackendPortAlignmentAuthority::Contract;
  BackendPortAccess access = BackendPortAccess::ReadOnly;
};

// Immutable identity of one MLA operation in the compiler-authored graph.
// The dense index is useful for setup-sized arrays; op_id/logical_stage_id and
// executable make accidental positional rebinding observable and rejectable.
struct MlaStageKey {
  std::size_t stage_index = 0;
  OpId op_id = 0;
  std::string logical_stage_id;
  std::string executable;
};

struct MlaStageSpec {
  MlaStageKey key;
  std::size_t input_port_begin = 0;
  std::size_t input_port_count = 0;
  std::size_t output_port_begin = 0;
  std::size_t output_port_count = 0;
};

struct ModelOutputSpec {
  std::size_t public_index = 0;
  std::string name;
  ValueId value_id = 0;
};

// Mutable construction payload. It is consumed by ModelExecutionPlan::create;
// successful plans expose only const access to an immutable shared snapshot.
struct ModelExecutionPlanData {
  std::string contract_version;
  std::vector<ValueSpec> values;
  std::vector<CarrierSpec> carriers;
  std::vector<ValueId> model_inputs;
  std::vector<OpSpec> ops;
  std::vector<BackendPortSpec> backend_ports;
  std::vector<ModelOutputSpec> model_outputs;
  // Derived by create(). Callers never author or mutate this index.
  std::vector<MlaStageSpec> mla_stages;
};

class ModelExecutionPlan final {
public:
  static std::optional<ModelExecutionPlan> create(ModelExecutionPlanData data,
                                                  std::string* error = nullptr);

  const std::string& contract_version() const noexcept;
  const std::vector<ValueSpec>& values() const noexcept;
  const std::vector<CarrierSpec>& carriers() const noexcept;
  const std::vector<ValueId>& model_inputs() const noexcept;
  const std::vector<OpSpec>& ops() const noexcept;
  const std::vector<BackendPortSpec>& backend_ports() const noexcept;
  std::size_t mla_stage_count() const noexcept;
  const MlaStageSpec* mla_stage(std::size_t stage_index) const noexcept;
  const MlaStageSpec* mla_stage_for_op(OpId op_id) const noexcept;
  const MlaStageSpec* mla_stage_for_identity(std::string_view logical_stage_id,
                                             std::string_view executable) const noexcept;
  std::span<const BackendPortSpec> backend_ports(std::size_t stage_index,
                                                 BackendPortDirection direction) const noexcept;
  const std::vector<ModelOutputSpec>& model_outputs() const noexcept;
  const ValueSpec* value(ValueId id) const noexcept;
  const CarrierSpec* carrier(CarrierId id) const noexcept;

private:
  explicit ModelExecutionPlan(std::shared_ptr<const ModelExecutionPlanData> data);
  std::shared_ptr<const ModelExecutionPlanData> data_;
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract
