#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

// Represents the I/O topology declared by an MLA-stage compiled ELF binary.
// Populated by parsing the ELF section name table for symbols of the form:
//   data.ifm.persistent.input_NN/<stage>/placeholder_N_0.b0
//   data.ofm.persistent.output_NN/<stage>/<op>.b0
//   data.ifm.persistent.qmla_ifm_N.b0
//   data.ifm.persistent.afe_direct_input_N.b0
//   data.ofm.persistent.afe_mla_output_N.b0
//   data.ifm.b0           (legacy monolithic IFM)
//   data.ofm.b0           (legacy monolithic OFM)
//
// Two compilation strategies coexist in the model toolchain:
//   - "monolithic": one IFM blob, one OFM blob; runtime delivers a packed
//     parent buffer. Models with an explicit canonical_op == "pack" producer
//     in the MPK fall here. .elf has data.ifm.b0 / data.ofm.b0 only.
//   - "multi-IFM/multi-OFM": per-tensor placeholders. Runtime must deliver
//     each input as a distinct physical segment; firmware reads each from
//     its own base address. .elf carries data.ifm.persistent.input_NN or
//     data.ifm.persistent.qmla_ifm_N or afe_direct_input_N slots.
//
// The two strategies are mutually exclusive within a single ELF.
struct MlaElfIoTopology {
  // True when only data.ifm.b0 is present (no per-input placeholders).
  bool monolithic_ifm = false;
  // True when only data.ofm.b0 is present (no per-output placeholders).
  bool monolithic_ofm = false;
  // Full section names for IFM placeholders, ordered by input index. Empty if
  // monolithic. Example entry:
  //   "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0"
  // or:
  //   "data.ifm.persistent.qmla_ifm_0.b0"
  std::vector<std::string> ifm_symbol_names;
  std::vector<std::uint64_t> ifm_extent_bytes;
  // Full section names for OFM placeholders, ordered by output index. Empty if
  // monolithic.
  std::vector<std::string> ofm_symbol_names;
  std::vector<std::uint64_t> ofm_extent_bytes;
  std::uint64_t monolithic_ifm_extent_bytes = 0;
  std::uint64_t monolithic_ofm_extent_bytes = 0;
  // Evidence retained for strict consumers.  The legacy parser continues to
  // prefer indexed symbols when an ELF declares both layouts, but strict
  // validation rejects that ambiguity instead of silently choosing one.
  bool ifm_layout_conflict = false;
  bool ofm_layout_conflict = false;
  std::vector<std::size_t> duplicate_ifm_indices;
  std::vector<std::size_t> duplicate_ofm_indices;
  // True when the file was successfully parsed as an ELF and at least one of
  // the four section types above was found. False on parse failure or when
  // the .elf has neither monolithic nor placeholder sections; the topology is
  // unknown and no policy decision should be made from this instance.
  bool valid = false;
  // Diagnostic context for telemetry / error reporting.
  std::string source_path;
  std::string error;
};

// Stable, machine-readable failure reasons for the strict topology contract.
// This API intentionally coexists with the permissive read API above so
// legacy callers keep their behavior while new static-plan code can fail
// closed on incomplete or ambiguous ELF evidence.
enum class MlaElfIoTopologyError {
  None,
  InvalidTopology,
  MissingIfm,
  MissingOfm,
  ConflictingIfmLayouts,
  ConflictingOfmLayouts,
  DuplicateIfmIndex,
  DuplicateOfmIndex,
  NonContiguousIfmIndices,
  NonContiguousOfmIndices,
  MissingIfmExtent,
  MissingOfmExtent,
  IfmPortCountMismatch,
  OfmPortCountMismatch,
};

struct MlaElfIoTopologyValidation {
  bool ok = false;
  MlaElfIoTopologyError code = MlaElfIoTopologyError::InvalidTopology;
  std::size_t expected = 0;
  std::size_t actual = 0;
  std::string detail;
};

// Parse the ELF section name table at `elf_path` and return the I/O topology.
// Sets out->valid=true on success and on partial success when at least one
// recognized section is found. On hard failure (file unreadable, ELF header
// invalid), sets out->valid=false and populates out->error. Never throws;
// callers must check out->valid before using the topology.
//
// Implementation also reads the 16-byte QMLA header of each recognized I/O
// section; it never loads bulk code/data payloads.
bool read_mla_elf_io_topology(const std::filesystem::path& elf_path, MlaElfIoTopology* out);

// Validate that both directions are present, exactly one layout is declared
// per direction, indexed layouts contain no duplicate or missing slots, and
// the topology is therefore safe to bind without a fallback heuristic.
MlaElfIoTopologyValidation validate_mla_elf_io_topology_strict(const MlaElfIoTopology& topology);

// Apply strict validation and then prove that the ELF physical port arity is
// exactly the arity declared by the MPK MLA operation.
MlaElfIoTopologyValidation reconcile_mla_elf_io_topology_strict(const MlaElfIoTopology& topology,
                                                                std::size_t expected_ifm_count,
                                                                std::size_t expected_ofm_count);

// Physical port counts. Call validate_mla_elf_io_topology_strict first when a
// result is used for binding; these helpers deliberately perform no fallback.
std::size_t mla_elf_ifm_port_count(const MlaElfIoTopology& topology);
std::size_t mla_elf_ofm_port_count(const MlaElfIoTopology& topology);
std::uint64_t mla_elf_ifm_extent_bytes(const MlaElfIoTopology& topology,
                                       std::size_t port_index);
std::uint64_t mla_elf_ofm_extent_bytes(const MlaElfIoTopology& topology,
                                       std::size_t port_index);

// True iff the .elf's IFM layout demands per-physical-input dispatch (i.e.
// there are >=2 placeholder slots and no monolithic data.ifm.b0 carrier).
// Returns false on parse failure (caller should fall back to MPK heuristic).
bool elf_topology_requires_distinct_ifm_segments(const MlaElfIoTopology& topology);

} // namespace simaai::neat::pipeline_internal::sima
