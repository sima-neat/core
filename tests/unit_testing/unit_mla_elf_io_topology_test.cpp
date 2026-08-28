// Unit test for MlaElfIoTopology. Synthesizes minimal ELF64 byte streams
// in-memory to exercise the parser without requiring large binary fixtures.
//
// Three scenarios:
//   1. Multi-IFM .elf — sections data.ifm.persistent.input_NN/...
//   2. Monolithic .elf — sections data.ifm.b0 / data.ofm.b0
//   3. Inconsistent .elf — both monolithic and placeholder sections present;
//      parser must accept and prefer placeholders, surfacing a warning.

#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/MlaElfIoTopology.h"

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
  return name == "data.ifm.b0" || name == "data.ofm.b0" ||
         name.starts_with("data.ifm.persistent.") ||
         name.starts_with("data.ofm.persistent.");
}

void append_u64_le(std::vector<std::uint8_t>& bytes, const std::uint64_t value) {
  for (unsigned shift = 0; shift < 64U; shift += 8U) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

// Build a minimal ELF64 file containing compiler-authored 16-byte QMLA
// SHT_DATA headers for every recognized I/O section plus a shstrtab.
std::filesystem::path write_minimal_elf(const std::string& tag,
                                        const std::vector<std::string>& section_names,
                                        const std::unordered_map<std::string, std::uint64_t>&
                                            extent_overrides = {}) {
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
    const auto extent = override == extent_overrides.end()
                            ? static_cast<std::uint64_t>(i) * 16U
                            : override->second;
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
  const auto path = write_minimal_elf(
      "afe_direct_input",
      {"code.r0.c0", "data.ifm.persistent.afe_direct_input_0.b0",
       "data.ifm.persistent.afe_direct_input_1.b0",
       "data.ofm.persistent.afe_mla_output_0.b0"});
  simaai::neat::pipeline_internal::sima::MlaElfIoTopology topology;
  const bool ok =
      simaai::neat::pipeline_internal::sima::read_mla_elf_io_topology(path, &topology);
  check(ok && topology.valid, "afe_direct_input: parser accepted current AFE symbols");
  check(topology.ifm_symbol_names.size() == 2U, "afe_direct_input: two IFM ports");
  check(topology.ofm_symbol_names.size() == 1U, "afe_direct_input: one OFM port");
  check(topology.ifm_symbol_names[1] ==
            "data.ifm.persistent.afe_direct_input_1.b0",
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

  const std::string missing_extent_ifm =
      "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0";
  const auto missing_extent_path = write_minimal_elf(
      "strict_missing_extent", {missing_extent_ifm, "data.ofm.b0"},
      {{missing_extent_ifm, 0U}});
  MlaElfIoTopology missing_extent;
  check(read_mla_elf_io_topology(missing_extent_path, &missing_extent),
        "strict missing extent: parser retains topology evidence");
  const auto missing_extent_result = validate_mla_elf_io_topology_strict(missing_extent);
  check(!missing_extent_result.ok &&
            missing_extent_result.code == MlaElfIoTopologyError::MissingIfmExtent,
        "strict missing extent: zero QMLA extent rejected");
  std::filesystem::remove(missing_extent_path);

  const auto conflict_path = write_minimal_elf(
      "strict_conflict",
      {"data.ifm.b0", "data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0", "data.ofm.b0"});
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

} // namespace

int main() {
  test_multi_ifm_topology();
  test_qmla_flat_topology();
  test_afe_direct_input_topology();
  test_monolithic_topology();
  test_unknown_topology_fails_cleanly();
  test_missing_file_fails_cleanly();
  test_strict_validation_and_reconciliation();
  std::cout << "unit_mla_elf_io_topology_test: PASS\n";
  return 0;
}
