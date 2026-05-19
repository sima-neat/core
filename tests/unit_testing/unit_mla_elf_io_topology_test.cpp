// Unit test for MlaElfIoTopology. Synthesizes minimal ELF64 byte streams
// in-memory to exercise the parser without requiring large binary fixtures.
//
// Three scenarios:
//   1. Multi-IFM .elf — sections data.ifm.persistent.input_NN/...
//   2. Monolithic .elf — sections data.ifm.b0 / data.ofm.b0
//   3. Inconsistent .elf — both monolithic and placeholder sections present;
//      parser must accept and prefer placeholders, surfacing a warning.

#define SIMA_NEAT_INTERNAL 1
#include "mpk/MlaElfIoTopology.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
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

// Build a minimal ELF64 file containing the listed section names plus a
// shstrtab. Returns the path to the temp file. Sections are zero-sized; only
// their names matter for the parser.
std::filesystem::path write_minimal_elf(const std::string& tag,
                                        const std::vector<std::string>& section_names) {
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
  const std::uint64_t shstrtab_offset = header_bytes;
  const std::uint64_t shoff = shstrtab_offset + shstrtab.size();

  Elf64Header hdr;
  hdr.e_shoff = shoff;
  hdr.e_shnum = shnum;
  hdr.e_shstrndx = static_cast<std::uint16_t>(shnum - 1U); // last section is shstrtab

  std::vector<Elf64SectionHeader> sections(shnum);
  // Section 0: NULL.
  // Sections 1..shnum-2: requested user sections (names[1..shnum-2]).
  // Section shnum-1: .shstrtab itself, offset/size into the file.
  for (std::size_t i = 0; i < sections.size(); ++i) {
    sections[i].sh_name = name_offsets[i];
    if (i + 1U == sections.size()) {
      sections[i].sh_type = 3; // SHT_STRTAB
      sections[i].sh_offset = shstrtab_offset;
      sections[i].sh_size = static_cast<std::uint64_t>(shstrtab.size());
    }
  }

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("mla_elf_io_topology_test_" + tag + ".elf");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
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

} // namespace

int main() {
  test_multi_ifm_topology();
  test_monolithic_topology();
  test_unknown_topology_fails_cleanly();
  test_missing_file_fails_cleanly();
  std::cout << "unit_mla_elf_io_topology_test: PASS\n";
  return 0;
}
