#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

// Represents the I/O topology declared by an MLA-stage compiled ELF binary.
// Populated by parsing the ELF section name table for symbols of the form:
//   data.ifm.persistent.input_NN/<stage>/placeholder_N_0.b0
//   data.ofm.persistent.output_NN/<stage>/<op>.b0
//   data.ifm.persistent.qmla_ifm_N.b0
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
//     data.ifm.persistent.qmla_ifm_N slots.
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
  // Full section names for OFM placeholders, ordered by output index. Empty if
  // monolithic.
  std::vector<std::string> ofm_symbol_names;
  // True when the file was successfully parsed as an ELF and at least one of
  // the four section types above was found. False on parse failure or when
  // the .elf has neither monolithic nor placeholder sections; the topology is
  // unknown and no policy decision should be made from this instance.
  bool valid = false;
  // Diagnostic context for telemetry / error reporting.
  std::string source_path;
  std::string error;
};

// Parse the ELF section name table at `elf_path` and return the I/O topology.
// Sets out->valid=true on success and on partial success when at least one
// recognized section is found. On hard failure (file unreadable, ELF header
// invalid), sets out->valid=false and populates out->error. Never throws;
// callers must check out->valid before using the topology.
//
// Implementation reads only the ELF header, the section header table, and
// the .shstrtab — it never loads the bulk code/data sections. For a typical
// ~23 MB SiMa MLA .elf the on-disk read footprint is ~10 KB.
bool read_mla_elf_io_topology(const std::filesystem::path& elf_path, MlaElfIoTopology* out);

// True iff the .elf's IFM layout demands per-physical-input dispatch (i.e.
// there are >=2 placeholder slots and no monolithic data.ifm.b0 carrier).
// Returns false on parse failure (caller should fall back to MPK heuristic).
bool elf_topology_requires_distinct_ifm_segments(const MlaElfIoTopology& topology);

} // namespace simaai::neat::pipeline_internal::sima
