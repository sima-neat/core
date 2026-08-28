#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima::static_contract {

using PhysicalCommandId = std::uint32_t;
using PhysicalCohortId = std::uint32_t;

enum class PhysicalEngine : std::uint8_t { Cvu, Mla, A65 };

// Exact semantic position authored while lowering the compiler graph. CVU
// placement policy consumes this authority directly; it must never recover a
// role from a generated cohort name or tensor dtype.
enum class PhysicalCommandRole : std::uint8_t {
  NonCvu = 0,
  Ingress,
  Egress,
  Interstitial,
};

// One independently bound member of a prepared backend call.  A standalone
// transform has one semantic origin; a fused transform has the complete
// ordered chain.  Only the outer values are physical bindings -- values on an
// edge internal to semantic_chain are deliberately not materialized.
//
// outer_inputs/outer_outputs are vectors rather than singular ids so the same
// representation remains exact for the singleton MLA and A65 commands, whose
// backend calls can have several ports.  Registered CVU transform members are
// required to have exactly one outer input and one outer output.
struct PhysicalCommandMember {
  std::uint32_t ordinal = 0;
  std::vector<OpId> semantic_chain;
  std::vector<ValueId> outer_inputs;
  std::vector<ValueId> outer_outputs;
};

// One prepared backend call. A CVU command can contain several horizontally
// grouped members; member order is canonical MLA-port/Unpack order.
struct PhysicalCommand {
  PhysicalCommandId id = 0;
  // Commands with the same cohort id are capacity chunks of one logical
  // transform frontier.  They remain distinct backend submissions in this
  // plan, but the compatibility pipeline renders the cohort as one element so
  // every chunk sees the original ordered input set and one shared arena.
  PhysicalCohortId cohort_id = 0;
  std::uint64_t topological_rank = 0;
  PhysicalEngine engine = PhysicalEngine::Cvu;
  PhysicalCommandRole role = PhysicalCommandRole::NonCvu;
  std::string implementation_id;
  // Generated CVU graph identity. Zero is reserved for non-CVU commands.
  std::uint32_t graph_id = 0U;
  // Uniform model batch contract for every member in this invocation.
  std::uint32_t batch_size = 1U;
  // Generated implementation ABI capacity. Zero for non-CVU commands.
  std::uint32_t maximum_members = 0U;
  std::vector<PhysicalCommandMember> members;
  // Canonical flattened outer bindings. These are retained because the arena
  // and prepared backend ABI operate on a command-wide ordered binding list.
  // They are derived from members and never contain a fused internal edge.
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  std::vector<PhysicalCommandId> predecessors;
  std::vector<PhysicalCommandId> successors;
};

struct PhysicalExecutionPlan {
  std::vector<PhysicalCommand> commands;
  std::vector<std::optional<PhysicalCommandId>> command_for_semantic_op;
  std::string deterministic_digest_material;
};

// True only for semantic relations which carry no independently scheduled
// backend work. Route selection may retain these operations to preserve an
// exact logical publication frontier, but must never maintain a second local
// list of address-only operation kinds.
[[nodiscard]] bool is_address_relation_op(const OpSpec& op) noexcept;

// Proves that `first` reaches `second` either directly or through only private,
// one-to-one address views which preserve the complete byte sequence at offset
// zero. `internal_values`, when requested, receives every value on that path,
// including the direct edge value and all relation outputs.
[[nodiscard]] bool resolve_exact_private_ordered_relation_path(
    const ModelExecutionPlan& plan, OpId first, OpId second,
    std::vector<ValueId>* internal_values = nullptr);

[[nodiscard]] std::optional<std::uint32_t>
minimum_cvu_member_capacity(const PhysicalExecutionPlan& plan) noexcept;

enum class PhysicalCommandState : std::uint8_t {
  Pending,
  Submitted,
  Completed,
  Failed,
  Blocked,
};

// Fixed-size per-frame dependency state. Stable topological rank is only a
// deterministic ready-queue tie breaker: independent commands can be claimed
// concurrently by the setup-sized backend worker pools. No storage is grown
// after create().
class PhysicalExecutionTracker final {
public:
  static std::optional<PhysicalExecutionTracker>
  create(const PhysicalExecutionPlan& plan, std::string* error = nullptr);

  [[nodiscard]] bool ready(PhysicalCommandId id) const noexcept;
  [[nodiscard]] std::optional<PhysicalCommandId> next_ready() const noexcept;
  [[nodiscard]] bool claim(PhysicalCommandId id, std::string* error = nullptr) noexcept;
  [[nodiscard]] bool complete(PhysicalCommandId id, std::string* error = nullptr) noexcept;
  [[nodiscard]] bool fail(PhysicalCommandId id, std::string* error = nullptr) noexcept;
  [[nodiscard]] PhysicalCommandState state(PhysicalCommandId id) const noexcept;
  [[nodiscard]] bool succeeded() const noexcept;
  [[nodiscard]] bool terminal() const noexcept;
  void reset() noexcept;

private:
  explicit PhysicalExecutionTracker(const PhysicalExecutionPlan* plan);
  const PhysicalExecutionPlan* plan_ = nullptr;
  std::vector<PhysicalCommandState> states_;
};

class PhysicalExecutionLowerer final {
public:
  static std::optional<PhysicalExecutionPlan> lower(const ModelExecutionPlan& semantic,
                                                    std::string* error = nullptr);
};

} // namespace simaai::neat::pipeline_internal::sima::static_contract
