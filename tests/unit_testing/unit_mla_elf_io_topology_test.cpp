// Unit test for MlaElfIoTopology. Synthesizes minimal ELF64 byte streams
// in-memory to exercise the parser without requiring large binary fixtures.
//
// Three scenarios:
//   1. Multi-IFM .elf — sections data.ifm.persistent.input_NN/...
//   2. Monolithic .elf — sections data.ifm.b0 / data.ofm.b0
//   3. Mixed .elf — a bare data.{ifm,ofm}.b0 beside placeholder sections is
//      one more placeholder port when its position is unambiguous, and a
//      layout conflict otherwise.

#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/MlaElfIoTopology.h"
#include "pipeline/internal/sima/static_contract/MpkDecoder.h"
#include "model_archive_fixture_utils.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Elf64Header {
  std::uint8_t e_ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::uint16_t e_type = 1;        // ET_REL
  std::uint16_t e_machine = 0x109; // SiMa custom
  std::uint32_t e_version = 1;
  std::uint64_t e_entry = 0;
  std::uint64_t e_phoff = 0;
  std::uint64_t e_shoff = 0;
  std::uint32_t e_flags = 0;
  std::uint16_t e_ehsize = 64;
  std::uint16_t e_phentsize = 0;
  std::uint16_t e_phnum = 0;
  std::uint16_t e_shentsize = 64;
  std::uint16_t e_shnum = 0;
  std::uint16_t e_shstrndx = 0;
};

struct Elf64SectionHeader {
  std::uint32_t sh_name = 0;
  std::uint32_t sh_type = 1; // SHT_PROGBITS
  std::uint64_t sh_flags = 0;
  std::uint64_t sh_addr = 0;
  std::uint64_t sh_offset = 0;
  std::uint64_t sh_size = 0;
  std::uint32_t sh_link = 0;
  std::uint32_t sh_info = 0;
  std::uint64_t sh_addralign = 8;
  std::uint64_t sh_entsize = 0;
};
#pragma pack(pop)

constexpr std::uint32_t kQmlaShtData = 0x71ba0002U;

bool is_mla_io_section(const std::string& name) {
  return name.starts_with("data.ifm.b") || name.starts_with("data.ofm.b") ||
         name.starts_with("data.ifm.persistent.") || name.starts_with("data.ofm.persistent.");
}

void append_u64_le(std::vector<std::uint8_t>& bytes, const std::uint64_t value) {
  for (unsigned shift = 0; shift < 64U; shift += 8U) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

// Build a minimal ELF64 file containing compiler-authored 16-byte QMLA
// SHT_DATA headers for every recognized I/O section plus a shstrtab.
std::filesystem::path
write_minimal_elf(const std::string& tag, const std::vector<std::string>& section_names,
                  const std::unordered_map<std::string, std::uint64_t>& extent_overrides = {}) {
  // First section is always the NULL section (name index 0). We append the
  // requested names, then append ".shstrtab" as the final section so its name
  // is also represented in the table.
  std::vector<std::string> names = {""};
  for (const auto& n : section_names) {
    names.push_back(n);
  }
  names.push_back(".shstrtab");

  // Build shstrtab contents (NUL-separated, leading NUL).
  std::vector<char> shstrtab;
  std::vector<std::uint32_t> name_offsets(names.size(), 0);
  for (std::size_t i = 0; i < names.size(); ++i) {
    name_offsets[i] = static_cast<std::uint32_t>(shstrtab.size());
    shstrtab.insert(shstrtab.end(), names[i].begin(), names[i].end());
    shstrtab.push_back('\0');
  }

  const std::uint16_t shnum = static_cast<std::uint16_t>(names.size());
  const std::uint64_t header_bytes = sizeof(Elf64Header);
  std::vector<std::uint8_t> qmla_headers;
  std::vector<std::uint64_t> payload_offsets(names.size(), 0U);
  for (std::size_t i = 1U; i + 1U < names.size(); ++i) {
    if (!is_mla_io_section(names[i])) {
      continue;
    }
    payload_offsets[i] = header_bytes + qmla_headers.size();
    const auto override = extent_overrides.find(names[i]);
    const auto extent =
        override == extent_overrides.end() ? static_cast<std::uint64_t>(i) * 16U : override->second;
    append_u64_le(qmla_headers, extent);
    append_u64_le(qmla_headers, 1U); // one address segment
  }
  const std::uint64_t shstrtab_offset = header_bytes + qmla_headers.size();
  const std::uint64_t shoff = shstrtab_offset + shstrtab.size();

  Elf64Header hdr;
  hdr.e_shoff = shoff;
  hdr.e_shnum = shnum;
  hdr.e_shstrndx = static_cast<std::uint16_t>(shnum - 1U); // last section is shstrtab

  std::vector<Elf64SectionHeader> sections(shnum);
  sections[0].sh_type = 0; // SHT_NULL
  // Section 0: NULL.
  // Sections 1..shnum-2: requested user sections (names[1..shnum-2]).
  // Section shnum-1: .shstrtab itself, offset/size into the file.
  for (std::size_t i = 1U; i < sections.size(); ++i) {
    sections[i].sh_name = name_offsets[i];
    if (i + 1U == sections.size()) {
      sections[i].sh_type = 3; // SHT_STRTAB
      sections[i].sh_offset = shstrtab_offset;
      sections[i].sh_size = static_cast<std::uint64_t>(shstrtab.size());
    } else if (payload_offsets[i] != 0U) {
      sections[i].sh_type = kQmlaShtData;
      sections[i].sh_offset = payload_offsets[i];
      sections[i].sh_size = 16U;
    }
  }

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("mla_elf_io_topology_test_" + tag + ".elf");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char*>(qmla_headers.data()),
            static_cast<std::streamsize>(qmla_headers.size()));
  out.write(shstrtab.data(), static_cast<std::streamsize>(shstrtab.size()));
  out.write(reinterpret_cast<const char*>(sections.data()),
            static_cast<std::streamsize>(sections.size() * sizeof(Elf64SectionHeader)));
  out.close();
  return path;
}

void check(bool cond, const char* what) {
  if (!cond) {
    std::cerr << "FAIL: " << what << "\n";
    std::exit(1);
  }
}

void test_multi_ifm_topology() {
  const auto path =
      write_minimal_elf("multi_ifm", {
                                         "code.r0.c0",
                                         "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0",
                                         "data.ifm.persistent.input_01/MLA_0/placeholder_1_0.b0",
                                         "data.ofm.persistent.output_00/MLA_0/sigmoid_64.b0",
                                         "data.ofm.persistent.output_01/MLA_0/conv2d_add_68.b0",
                                     });
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(ok, "multi_ifm: parser returned ok");
  check(topology.valid, "multi_ifm: topology.valid");
  check(!topology.monolithic_ifm, "multi_ifm: !monolithic_ifm");
  check(!topology.monolithic_ofm, "multi_ifm: !monolithic_ofm");
  check(topology.ifm_symbol_names.size() == 2U, "multi_ifm: 2 IFM slots");
  check(topology.ofm_symbol_names.size() == 2U, "multi_ifm: 2 OFM slots");
  check(topology.ifm_symbol_names[0].find("placeholder_0_0") != std::string::npos,
        "multi_ifm: ifm[0] is placeholder_0_0");
  check(topology.ifm_symbol_names[1].find("placeholder_1_0") != std::string::npos,
        "multi_ifm: ifm[1] is placeholder_1_0");
  check(
      simaai::neat::pipeline_internal::sima::elf_topology_requires_distinct_ifm_segments(topology),
      "multi_ifm: requires_distinct_ifm_segments == true");
  std::filesystem::remove(path);
}

void test_qmla_flat_topology() {
  std::vector<std::string> names{"code.r0.c0"};
  std::unordered_map<std::string, std::uint64_t> extents;
  // Deliberately put slot 10 before slots 0..9. Numeric port authority must
  // never depend on lexical order or ELF section order.
  names.push_back("data.ifm.persistent.afe_direct_input_10.b0");
  extents.emplace(names.back(), 1010U);
  for (std::size_t index = 0U; index < 10U; ++index) {
    names.push_back("data.ifm.persistent.afe_direct_input_" + std::to_string(index) + ".b0");
    extents.emplace(names.back(), 1000U + index);
  }
  names.push_back("data.ofm.persistent.afe_mla_output_1.b0");
  extents.emplace(names.back(), 2001U);
  names.push_back("data.ofm.persistent.afe_mla_output_0.b0");
  extents.emplace(names.back(), 2000U);
  const auto path = write_minimal_elf("qmla_flat", names, extents);
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(ok, "qmla_flat: parser returned ok");
  check(topology.valid, "qmla_flat: topology.valid");
  check(!topology.monolithic_ifm, "qmla_flat: !monolithic_ifm");
  check(!topology.monolithic_ofm, "qmla_flat: !monolithic_ofm");
  check(topology.ifm_symbol_names.size() == 11U, "qmla_flat: 11 IFM slots");
  check(topology.ofm_symbol_names.size() == 2U, "qmla_flat: 2 OFM slots");
  check(topology.ifm_symbol_names[0] == "data.ifm.persistent.afe_direct_input_0.b0",
        "qmla_flat: ifm[0]");
  check(topology.ifm_symbol_names[10] == "data.ifm.persistent.afe_direct_input_10.b0",
        "qmla_flat: ifm[10]");
  check(topology.ofm_symbol_names[1] == "data.ofm.persistent.afe_mla_output_1.b0",
        "qmla_flat: ofm[1]");
  check(topology.ifm_extent_bytes.size() == 11U && topology.ifm_extent_bytes[0] == 1000U &&
            topology.ifm_extent_bytes[10] == 1010U,
        "qmla_flat: IFM extents preserve numeric slot order");
  check(topology.ofm_extent_bytes.size() == 2U && topology.ofm_extent_bytes[0] == 2000U &&
            topology.ofm_extent_bytes[1] == 2001U,
        "qmla_flat: OFM extents preserve numeric slot order");
  check(
      simaai::neat::pipeline_internal::sima::elf_topology_requires_distinct_ifm_segments(topology),
      "qmla_flat: requires_distinct_ifm_segments == true");
  std::filesystem::remove(path);
}

void test_afe_direct_input_topology() {
  const auto path = write_minimal_elf("afe_direct_input",
                                      {"code.r0.c0", "data.ifm.persistent.afe_direct_input_0.b0",
                                       "data.ifm.persistent.afe_direct_input_1.b0",
                                       "data.ofm.persistent.afe_mla_output_0.b0"});
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(ok && topology.valid, "afe_direct_input: parser accepted current AFE symbols");
  check(topology.ifm_symbol_names.size() == 2U, "afe_direct_input: two IFM ports");
  check(topology.ofm_symbol_names.size() == 1U, "afe_direct_input: one OFM port");
  check(topology.ifm_symbol_names[1] == "data.ifm.persistent.afe_direct_input_1.b0",
        "afe_direct_input: stable indexed order");
  std::filesystem::remove(path);
}

void test_monolithic_topology() {
  const auto path = write_minimal_elf("monolithic", {
                                                        "code.r0.c0",
                                                        "data.ifm.b0",
                                                        "data.ofm.b0",
                                                    });
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(ok, "monolithic: parser returned ok");
  check(topology.valid, "monolithic: topology.valid");
  check(topology.monolithic_ifm, "monolithic: monolithic_ifm");
  check(topology.monolithic_ofm, "monolithic: monolithic_ofm");
  check(topology.ifm_symbol_names.empty(), "monolithic: no IFM placeholders");
  check(topology.ofm_symbol_names.empty(), "monolithic: no OFM placeholders");
  check(
      !simaai::neat::pipeline_internal::sima::elf_topology_requires_distinct_ifm_segments(topology),
      "monolithic: requires_distinct_ifm_segments == false");
  std::filesystem::remove(path);
}

void test_unknown_topology_fails_cleanly() {
  const auto path = write_minimal_elf("unknown", {
                                                     "code.r0.c0",
                                                     "checksums",
                                                     "tile.latencies",
                                                 });
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(!ok, "unknown: parser reports failure");
  check(!topology.valid, "unknown: topology.valid is false");
  check(!topology.error.empty(), "unknown: error message populated");
  std::filesystem::remove(path);
}

void test_missing_file_fails_cleanly() {
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok = simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(
      "/tmp/does_not_exist_mla_elf_io_topology_test.elf", &topology);
  check(!ok, "missing_file: parser reports failure");
  check(!topology.valid, "missing_file: topology.valid is false");
}

void test_unindexed_persistent_topology() {
  using namespace simaai::neat::pipeline_internal::sima;
  const std::string ifm = "data.ifm.persistent.MLA_0/placeholder_0_0.b0";
  const std::string ofm = "data.ofm.persistent.MLA_0/conv2d_add_0_output.b0";
  const auto path = write_minimal_elf("unindexed", {ofm, ifm}, {{ifm, 112U}, {ofm, 240U}});
  MlaElfIoTopology topology;
  check(read_mla_elf_io_topology(path, &topology), "unindexed: recognized compiler tensor names");
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 1U).ok,
        "unindexed: single port in each direction reconciles");
  check(topology.ifm_symbol_names == std::vector<std::string>{ifm} &&
            topology.ofm_symbol_names == std::vector<std::string>{ofm},
        "unindexed: preserves exact names independent of section order");
  check(mla_elf_ifm_extent_bytes(topology, 0U) == 112U &&
            mla_elf_ofm_extent_bytes(topology, 0U) == 240U,
        "unindexed: preserves compiler storage extents");
  check(!reconcile_mla_elf_io_topology_strict(topology, 2U, 1U).ok,
        "unindexed: cannot satisfy a multi-input MPK");
  std::filesystem::remove(path);

  for (const auto& name : {ifm, ofm}) {
    const auto missing_path =
        write_minimal_elf("unindexed_missing_extent", {ifm, ofm}, {{name, 0U}});
    check(read_mla_elf_io_topology(missing_path, &topology),
          "unindexed: missing extent retains topology evidence");
    const auto result = validate_mla_elf_io_topology_strict(topology);
    check(!result.ok && result.code == (name == ifm ? MlaElfIoTopologyError::MissingIfmExtent
                                                    : MlaElfIoTopologyError::MissingOfmExtent),
          "unindexed: non-zero QMLA extent remains required");
    std::filesystem::remove(missing_path);
  }

  for (const auto& extra :
       {"data.ifm.persistent.afe_direct_input_0.b0", "data.ofm.persistent.afe_mla_output_0.b0"}) {
    const auto ambiguous_path = write_minimal_elf("unindexed_ambiguous", {ifm, ofm, extra});
    check(!read_mla_elf_io_topology(ambiguous_path, &topology) && !topology.valid,
          "unindexed: mixed indexed sections have conflicting port orders");
    check(topology.error.find("mixed indexed and unindexed") != std::string::npos,
          "unindexed: ambiguous mapping has an actionable diagnostic");
    std::filesystem::remove(ambiguous_path);
  }

  for (const std::string extra : {"data.ifm.b0", "data.ofm.b0"}) {
    const bool input = extra == "data.ifm.b0";
    const auto bare_path = write_minimal_elf("unindexed_bare", {ifm, ofm, extra});
    check(read_mla_elf_io_topology(bare_path, &topology),
          "unindexed: bare section beside a native section is recognized");
    check(reconcile_mla_elf_io_topology_strict(topology, input ? 2U : 1U, input ? 1U : 2U).ok,
          "unindexed: bare section is one more per-tensor port");
    check((input ? topology.ifm_symbol_names : topology.ofm_symbol_names) ==
              std::vector<std::string>{input ? ifm : ofm, extra},
          "unindexed: bare section keeps its ELF encounter position");
    std::filesystem::remove(bare_path);
  }
}

// The compiler names a tensor's section only when it has a persistent name and
// otherwise falls back to the bare data.{ifm,ofm}.b<batch> section.
void test_bare_section_beside_per_tensor_sections() {
  using namespace simaai::neat::pipeline_internal::sima;
  const std::string ifm = "data.ifm.persistent.MLA_0/placeholder_0_0.b0";
  const std::string native_ofm = "data.ofm.persistent.MLA_0/add_0_output.b0";
  const auto read = [](const std::string& tag, const std::vector<std::string>& names,
                       const std::unordered_map<std::string, std::uint64_t>& extents = {}) {
    const auto path = write_minimal_elf(tag, names, extents);
    MlaElfIoTopology topology;
    check(read_mla_elf_io_topology(path, &topology), "bare: parser retains topology evidence");
    std::filesystem::remove(path);
    return topology;
  };

  auto topology = read("bare_first", {ifm, "data.ofm.b0", native_ofm},
                       {{"data.ofm.b0", 3200U}, {native_ofm, 14745600U}});
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 2U).ok && !topology.monolithic_ofm &&
            !topology.ofm_layout_conflict,
        "bare first: admitted as a per-tensor OFM port");
  check(topology.ofm_symbol_names == std::vector<std::string>{"data.ofm.b0", native_ofm} &&
            topology.ofm_extent_bytes == std::vector<std::uint64_t>{3200U, 14745600U},
        "bare first: ELF encounter order and extents are preserved");
  check(topology.ofm_slots.size() == 2U && topology.ofm_slots[0].logical_index == 0U &&
            topology.ofm_slots[1].logical_index == 1U,
        "bare first: each slot owns its port");

  topology = read("bare_ifm", {"data.ifm.b0", ifm, native_ofm});
  check(reconcile_mla_elf_io_topology_strict(topology, 2U, 1U).ok &&
            topology.ifm_symbol_names == std::vector<std::string>{"data.ifm.b0", ifm},
        "bare IFM: admitted symmetrically");

  topology = read("bare_batch", {ifm, "data.ofm.b0", native_ofm, "data.ofm.b1",
                                 "data.ofm.persistent.MLA_0/add_0_output.b1"});
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 2U).ok,
        "bare batch: samples of both ports reconcile");
  for (const auto& slot : topology.ofm_slots) {
    check(slot.logical_index == (slot.symbol.starts_with("data.ofm.b") ? 0U : 1U),
          "bare batch: every sample keeps its tensor's port");
  }

  const std::string output_0 = "data.ofm.persistent.output_0/MLA_0/a.b0";
  const std::string output_1 = "data.ofm.persistent.output_1/MLA_0/b.b0";
  topology = read("bare_indexed_gap", {ifm, "data.ofm.b0", output_1});
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 2U).ok &&
            topology.ofm_symbol_names == std::vector<std::string>{"data.ofm.b0", output_1},
        "bare indexed: fills the unclaimed index at its encounter position");
  topology = read("bare_indexed_tail", {ifm, output_0, "data.ofm.b0"});
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 2U).ok &&
            topology.ofm_symbol_names == std::vector<std::string>{output_0, "data.ofm.b0"},
        "bare indexed: follows every claimed index");

  for (const auto& names :
       {std::vector<std::string>{ifm, output_1, "data.ofm.b0"},
        std::vector<std::string>{ifm, "data.ofm.b0", "data.ofm.persistent.output_2/MLA_0/c.b0"}}) {
    topology = read("bare_indexed_conflict", names);
    const auto result = validate_mla_elf_io_topology_strict(topology);
    check(!result.ok && result.code == MlaElfIoTopologyError::ConflictingOfmLayouts,
          "bare indexed: position disagreeing with the unclaimed index stays rejected");
  }

  const auto mixed_path =
      write_minimal_elf("bare_mixed", {ifm, "data.ofm.b0", output_0, native_ofm});
  check(!read_mla_elf_io_topology(mixed_path, &topology) && !topology.valid,
        "bare mixed: indexed and native siblings remain rejected");
  std::filesystem::remove(mixed_path);
}

void test_native_ports_preserve_encounter_order() {
  using namespace simaai::neat::pipeline_internal::sima;
  const std::string ifm_first = "data.ifm.persistent.MLA_0/placeholder_20_0.b0";
  const std::string ifm_second = "data.ifm.persistent.MLA_0/placeholder_10_0.b0";
  const std::string ofm_first = "data.ofm.persistent.MLA_0/z_output.b0";
  const std::string ofm_second = "data.ofm.persistent.MLA_0/a_output.b0";
  // Interleave directions and use equal extents. Neither lexical sorting nor
  // size matching can recover the authored port order.
  const auto path = write_minimal_elf(
      "native_order", {ofm_first, ifm_first, "code.r0.c0", ofm_second, ifm_second},
      {{ifm_first, 128U}, {ifm_second, 128U}, {ofm_first, 240U}, {ofm_second, 240U}});
  MlaElfIoTopology topology;
  check(read_mla_elf_io_topology(path, &topology), "native order: parser succeeds");
  check(reconcile_mla_elf_io_topology_strict(topology, 2U, 2U).ok,
        "native order: exact MPK arity reconciles");
  check(topology.ifm_symbol_names == std::vector<std::string>{ifm_first, ifm_second},
        "native order: IFM encounter order survives equal sizes and nonlexical names");
  check(topology.ofm_symbol_names == std::vector<std::string>{ofm_first, ofm_second},
        "native order: OFM encounter order survives equal sizes and nonlexical names");
  check(topology.ifm_extent_bytes == std::vector<std::uint64_t>{128U, 128U} &&
            topology.ofm_extent_bytes == std::vector<std::uint64_t>{240U, 240U},
        "native order: extents remain associated with their ports");
  check(reconcile_mla_elf_io_topology_strict(topology, 1U, 2U).code ==
                MlaElfIoTopologyError::IfmPortCountMismatch &&
            reconcile_mla_elf_io_topology_strict(topology, 2U, 1U).code ==
                MlaElfIoTopologyError::OfmPortCountMismatch,
        "native order: both MPK port counts remain enforced");
  std::filesystem::remove(path);

  for (const auto& duplicate : {ifm_first, ofm_first}) {
    const auto duplicate_path =
        write_minimal_elf("native_duplicate", {ifm_first, ofm_first, duplicate});
    check(!read_mla_elf_io_topology(duplicate_path, &topology) && !topology.valid,
          "native duplicate: repeated section identity rejected");
    check(topology.error.find("duplicate unindexed") != std::string::npos,
          "native duplicate: diagnostic identifies duplicate section");
    std::filesystem::remove(duplicate_path);
  }

  for (const auto& malformed :
       {"data.ifm.persistent.MLA_bad/placeholder_0_0.b0", "data.ifm.persistent.MLA_0/.b0",
        "data.ifm.persistent.MLA_0/placeholder_0_0.b1"}) {
    const auto malformed_path = write_minimal_elf("native_malformed", {malformed, ofm_first});
    check(read_mla_elf_io_topology(malformed_path, &topology),
          "native malformed: parser retains the valid OFM evidence");
    check(validate_mla_elf_io_topology_strict(topology).code == MlaElfIoTopologyError::MissingIfm,
          "native malformed: invalid native IFM name cannot establish a port");
    std::filesystem::remove(malformed_path);
  }
}

void test_strict_validation_and_reconciliation() {
  using namespace simaai::neat::pipeline_internal::sima;

  const auto valid_path =
      write_minimal_elf("strict_valid", {"data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0",
                                         "data.ifm.persistent.input_01/MLA_0/placeholder_1_0.b0",
                                         "data.ofm.persistent.output_00/MLA_0/out0.b0"});
  MlaElfIoTopology valid;
  check(read_mla_elf_io_topology(valid_path, &valid), "strict valid: parser returned ok");
  check(validate_mla_elf_io_topology_strict(valid).ok, "strict valid: validation succeeds");
  check(reconcile_mla_elf_io_topology_strict(valid, 2U, 1U).ok,
        "strict valid: exact arity reconciles");
  const auto mismatch = reconcile_mla_elf_io_topology_strict(valid, 1U, 1U);
  check(!mismatch.ok && mismatch.code == MlaElfIoTopologyError::IfmPortCountMismatch &&
            mismatch.expected == 1U && mismatch.actual == 2U,
        "strict valid: mismatch reports exact IFM counts");
  std::filesystem::remove(valid_path);

  const std::string missing_extent_ifm = "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0";
  const auto missing_extent_path = write_minimal_elf(
      "strict_missing_extent", {missing_extent_ifm, "data.ofm.b0"}, {{missing_extent_ifm, 0U}});
  MlaElfIoTopology missing_extent;
  check(read_mla_elf_io_topology(missing_extent_path, &missing_extent),
        "strict missing extent: parser retains topology evidence");
  const auto missing_extent_result = validate_mla_elf_io_topology_strict(missing_extent);
  check(!missing_extent_result.ok &&
            missing_extent_result.code == MlaElfIoTopologyError::MissingIfmExtent,
        "strict missing extent: zero QMLA extent rejected");
  std::filesystem::remove(missing_extent_path);

  const auto conflict_path =
      write_minimal_elf("strict_conflict", {"data.ifm.persistent.input_01/MLA_0/placeholder_1_0.b0",
                                            "data.ifm.b0", "data.ofm.b0"});
  MlaElfIoTopology conflict;
  check(read_mla_elf_io_topology(conflict_path, &conflict),
        "strict conflict: permissive parser remains compatible");
  const auto conflict_result = validate_mla_elf_io_topology_strict(conflict);
  check(!conflict_result.ok && conflict_result.code == MlaElfIoTopologyError::ConflictingIfmLayouts,
        "strict conflict: ambiguity rejected");
  std::filesystem::remove(conflict_path);

  const auto gap_path =
      write_minimal_elf("strict_gap", {"data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0",
                                       "data.ifm.persistent.input_02/MLA_0/placeholder_2_0.b0",
                                       "data.ofm.persistent.output_00/MLA_0/out0.b0"});
  MlaElfIoTopology gap;
  check(read_mla_elf_io_topology(gap_path, &gap), "strict gap: parser returned ok");
  const auto gap_result = validate_mla_elf_io_topology_strict(gap);
  check(!gap_result.ok && gap_result.code == MlaElfIoTopologyError::NonContiguousIfmIndices,
        "strict gap: missing indexed slot rejected");
  std::filesystem::remove(gap_path);

  const auto duplicate_path = write_minimal_elf(
      "strict_duplicate",
      {"data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0", "data.ifm.persistent.qmla_ifm_0.b0",
       "data.ofm.persistent.output_00/MLA_0/out0.b0"});
  MlaElfIoTopology duplicate;
  check(read_mla_elf_io_topology(duplicate_path, &duplicate),
        "strict duplicate: parser returned ok");
  const auto duplicate_result = validate_mla_elf_io_topology_strict(duplicate);
  check(!duplicate_result.ok && duplicate_result.code == MlaElfIoTopologyError::DuplicateIfmIndex,
        "strict duplicate: repeated indexed slot rejected");
  std::filesystem::remove(duplicate_path);
}

void test_batch_slots_preserve_physical_order() {
  using namespace simaai::neat::pipeline_internal::sima;
  for (const std::string prefix : {"data.ifm.persistent.MLA_0/left.b",
                                   "data.ifm.persistent.afe_direct_input_0.b", "data.ifm.b"}) {
    const bool monolithic = prefix == "data.ifm.b";
    const std::string output = monolithic ? "data.ofm.b" : "data.ofm.persistent.MLA_0/out.b";
    std::vector<std::string> names{prefix + "0", prefix + "1", output + "0", output + "1"};
    auto path = write_minimal_elf("batch", names);
    MlaElfIoTopology topology;
    check(read_mla_elf_io_topology(path, &topology), "batched ELF is readable");
    check(validate_mla_elf_io_topology_strict(topology).ok, "complete batches validate");
    check(topology.ifm_slots.size() == 2U && topology.ofm_slots.size() == 2U &&
              topology.ifm_slots[1].symbol == prefix + "1" &&
              topology.ifm_slots[1].logical_index == 0U &&
              topology.ifm_slots[1].batch_index == 1U && topology.ifm_slots[1].extent_bytes == 32U,
          "batch sections retain their own extent and physical order");
    std::filesystem::remove(path);
    for (const auto& invalid : {prefix + "1", prefix + "3"}) {
      auto bad_names = names;
      bad_names.push_back(invalid);
      path = write_minimal_elf("bad_batch", bad_names);
      check(read_mla_elf_io_topology(path, &topology), "invalid batch can be diagnosed");
      check(!validate_mla_elf_io_topology_strict(topology).ok,
            "duplicate samples and holes are rejected");
      std::filesystem::remove(path);
    }
  }
  const std::string left = "data.ifm.persistent.MLA_0/left.b";
  const std::string right = "data.ifm.persistent.MLA_0/right.b";
  const auto path = write_minimal_elf(
      "batch_multi_input", {right + "0", left + "0", left + "1", right + "1", "data.ofm.b0"});
  MlaElfIoTopology topology;
  check(read_mla_elf_io_topology(path, &topology) &&
            validate_mla_elf_io_topology_strict(topology).ok,
        "multi-input batches validate independently of encounter interleaving");
  check(topology.ifm_slots.size() == 4U && topology.ifm_slots[2].logical_index == 1U &&
            topology.ifm_slots[3].logical_index == 0U && topology.ifm_slots[2].batch_index == 1U,
        "input identity and sample identity are retained separately");
  std::filesystem::remove(path);
}

} // namespace

// EfficientSAM3 image model compiled by Model Compiler 3.0: the detections
// output has no persistent section name, the masks output does. The ELF is
// synthesized with the compiled stage's exact I/O section names, extents and
// header order; the MPK is the compiler's unmodified mpk.json.
void test_efficientsam3_fixture_decodes() {
  using namespace simaai::neat::pipeline_internal::sima;
  const auto manifest =
      sima_test::test_model_archive_fixture_root_path() / "strict-seeds" / "efficientsam3_mpk.json";
  std::error_code ec;
  check(std::filesystem::file_size(manifest, ec) == 9734U && !ec &&
            sima_test::fixture_file_sha256(manifest) ==
                "8fef526843a31fea965b309a13b47060052295e7d336821f0c025a552ea6baad",
        "efficientsam3: exact MPK fixture");

  const std::string text = "data.ifm.persistent.MLA_0/placeholder_12_0.b0";
  const std::string image = "data.ifm.persistent.MLA_0/placeholder_14_0.b0";
  const std::string masks = "data.ofm.persistent.MLA_0/add_3728_output.b0";
  const auto path = write_minimal_elf(
      "efficientsam3", {"code.r0.c0", text, image, "data.ofm.b0", masks},
      {{text, 8704U}, {image, 6096384U}, {"data.ofm.b0", 3200U}, {masks, 14745600U}});
  MlaElfIoTopology topology;
  check(read_mla_elf_io_topology(path, &topology), "efficientsam3: ELF topology parses");
  std::filesystem::remove(path);

  const auto result = static_contract::MpkDecoder{}.decode_file(manifest, topology);
  if (!result && result.error) {
    std::cerr << result.error->json_path << ": " << result.error->detail << "\n";
  }
  check(static_cast<bool>(result), "efficientsam3: MPK decodes against its ELF topology");
  const std::unordered_map<std::string, std::string> expected = {
      {text, "cast_0"}, {image, "cast_1"}, {"data.ofm.b0", "MLA_0_0"}, {masks, "MLA_0_1"}};
  const auto& ports = result.plan->backend_ports();
  check(ports.size() == expected.size(), "efficientsam3: one backend port per ELF section");
  for (const auto& port : ports) {
    const auto found = expected.find(port.elf_symbol);
    check(found != expected.end() && result.plan->value(port.value_id)->name == found->second,
          "efficientsam3: each ELF section binds its MPK tensor");
  }
}

int main() {
  test_batch_slots_preserve_physical_order();
  test_multi_ifm_topology();
  test_qmla_flat_topology();
  test_afe_direct_input_topology();
  test_monolithic_topology();
  test_unknown_topology_fails_cleanly();
  test_missing_file_fails_cleanly();
  test_unindexed_persistent_topology();
  test_native_ports_preserve_encounter_order();
  test_bare_section_beside_per_tensor_sections();
  test_efficientsam3_fixture_decodes();
  test_strict_validation_and_reconciliation();
  std::cout << "unit_mla_elf_io_topology_test: PASS\n";
  return 0;
}
