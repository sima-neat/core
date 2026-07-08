#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/MlaElfIoTopology.h"

#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal::sima {

namespace {

// Minimal ELF64 layout types. We don't include <elf.h> so this code is
// portable across platforms (e.g. macOS host builds, sysroot-less CI).
struct Elf64Header {
  std::uint8_t e_ident[16];
  std::uint16_t e_type;
  std::uint16_t e_machine;
  std::uint32_t e_version;
  std::uint64_t e_entry;
  std::uint64_t e_phoff;
  std::uint64_t e_shoff;
  std::uint32_t e_flags;
  std::uint16_t e_ehsize;
  std::uint16_t e_phentsize;
  std::uint16_t e_phnum;
  std::uint16_t e_shentsize;
  std::uint16_t e_shnum;
  std::uint16_t e_shstrndx;
};
static_assert(sizeof(Elf64Header) == 64, "Elf64 header must be 64 bytes");

struct Elf64SectionHeader {
  std::uint32_t sh_name;
  std::uint32_t sh_type;
  std::uint64_t sh_flags;
  std::uint64_t sh_addr;
  std::uint64_t sh_offset;
  std::uint64_t sh_size;
  std::uint32_t sh_link;
  std::uint32_t sh_info;
  std::uint64_t sh_addralign;
  std::uint64_t sh_entsize;
};
static_assert(sizeof(Elf64SectionHeader) == 64, "Elf64 section header must be 64 bytes");

constexpr std::uint8_t kElfMagic[4] = {0x7f, 'E', 'L', 'F'};
constexpr std::uint8_t kElfClass64 = 2;
constexpr std::uint8_t kElfData2Lsb = 1;

// Read N bytes at a given absolute offset. Returns true on success.
bool read_at(std::ifstream& in, std::uint64_t offset, void* dst, std::size_t n) {
  in.clear();
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!in.good()) {
    return false;
  }
  in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
  return in.good() && static_cast<std::size_t>(in.gcount()) == n;
}

// Extract section name from the loaded shstrtab buffer at index `name_offset`.
// Returns empty string if out of bounds.
std::string section_name_at(const std::vector<char>& shstrtab, std::uint32_t name_offset) {
  if (name_offset >= shstrtab.size()) {
    return {};
  }
  // Names are NUL-terminated within the table.
  const char* base = shstrtab.data() + name_offset;
  const std::size_t max_len = shstrtab.size() - name_offset;
  const std::size_t len = ::strnlen(base, max_len);
  return std::string(base, len);
}

// Match patterns for I/O section symbols. Anchored regexes; ordered scan keeps
// the implementation stable against future suffix additions.
//
// Older compiler drops used nested placeholder names:
//   data.ifm.persistent.input_00/MLA_0/placeholder_0_0.b0
//   data.ofm.persistent.output_00/MLA_0/sigmoid_64.b0
//
// QMLA/public-IO artifacts produced by newer compiler drops use flat section
// names:
//   data.ifm.persistent.qmla_ifm_0.b0
//   data.ofm.persistent.afe_mla_output_0.b0
//
// Treat both forms as the same physical topology signal.  The index capture is
// deliberately the only semantic dependency; the full section name is preserved
// for downstream diagnostics / binding names.
constexpr const char* kMonolithicIfmName = "data.ifm.b0";
constexpr const char* kMonolithicOfmName = "data.ofm.b0";

bool parse_uint_at(const std::string& name, std::size_t pos, std::size_t* value,
                   std::size_t* end_pos) {
  if (!value || pos >= name.size() || !std::isdigit(static_cast<unsigned char>(name[pos]))) {
    return false;
  }
  std::size_t out = 0U;
  std::size_t cur = pos;
  while (cur < name.size() && std::isdigit(static_cast<unsigned char>(name[cur]))) {
    const unsigned digit = static_cast<unsigned>(name[cur] - '0');
    if (out > ((std::numeric_limits<std::size_t>::max() - digit) / 10U)) {
      return false;
    }
    out = out * 10U + digit;
    ++cur;
  }
  *value = out;
  if (end_pos) {
    *end_pos = cur;
  }
  return true;
}

bool parse_section_index_after_prefix(const std::string& name, const std::string& prefix,
                                      bool require_slash_after_index, std::size_t* index) {
  if (name.rfind(prefix, 0U) != 0U) {
    return false;
  }
  std::size_t end = 0U;
  if (!parse_uint_at(name, prefix.size(), index, &end)) {
    return false;
  }
  if (require_slash_after_index) {
    return end < name.size() && name[end] == '/';
  }
  return end + 2U <= name.size() && name[end] == '.' && name[end + 1U] == 'b';
}

std::optional<std::size_t> parse_ifm_section_index(const std::string& name) {
  std::size_t index = 0U;
  if (parse_section_index_after_prefix(name, "data.ifm.persistent.input_", true, &index) ||
      parse_section_index_after_prefix(name, "data.ifm.persistent.qmla_ifm_", false, &index)) {
    return index;
  }
  return std::nullopt;
}

std::optional<std::size_t> parse_ofm_section_index(const std::string& name) {
  std::size_t index = 0U;
  if (parse_section_index_after_prefix(name, "data.ofm.persistent.output_", true, &index) ||
      parse_section_index_after_prefix(name, "data.ofm.persistent.afe_mla_output_", false,
                                       &index)) {
    return index;
  }
  return std::nullopt;
}

// Insert `name` at slot `index` in `dst`, growing the vector as needed. If
// `dst` already has a value at that index, prefer the existing one (keeps the
// first-seen entry on duplicate scan; multi-section ELFs sometimes mention the
// same logical placeholder in multiple sections).
void place_at_index(std::vector<std::string>* dst, std::size_t index, const std::string& name) {
  if (dst->size() <= index) {
    dst->resize(index + 1U);
  }
  if ((*dst)[index].empty()) {
    (*dst)[index] = name;
  }
}

} // namespace

bool read_mla_elf_io_topology(const std::filesystem::path& elf_path, MlaElfIoTopology* out) {
  if (!out) {
    return false;
  }
  *out = MlaElfIoTopology{};
  out->source_path = elf_path.string();

  std::ifstream in(elf_path, std::ios::binary);
  if (!in.is_open()) {
    out->error = "elf-io-topology: cannot open file";
    return false;
  }

  Elf64Header hdr{};
  if (!read_at(in, 0, &hdr, sizeof(hdr))) {
    out->error = "elf-io-topology: short read on ELF header";
    return false;
  }
  if (std::memcmp(hdr.e_ident, kElfMagic, sizeof(kElfMagic)) != 0) {
    out->error = "elf-io-topology: ELF magic mismatch";
    return false;
  }
  if (hdr.e_ident[4] != kElfClass64) {
    out->error = "elf-io-topology: only ELF64 supported";
    return false;
  }
  if (hdr.e_ident[5] != kElfData2Lsb) {
    out->error = "elf-io-topology: only little-endian ELF supported";
    return false;
  }
  if (hdr.e_shentsize != sizeof(Elf64SectionHeader)) {
    out->error = "elf-io-topology: unexpected section header size";
    return false;
  }
  if (hdr.e_shnum == 0 || hdr.e_shstrndx >= hdr.e_shnum) {
    out->error = "elf-io-topology: missing or invalid section headers";
    return false;
  }

  // Read the section header table.
  std::vector<Elf64SectionHeader> sections(hdr.e_shnum);
  if (!read_at(in, hdr.e_shoff, sections.data(),
               static_cast<std::size_t>(hdr.e_shnum) * sizeof(Elf64SectionHeader))) {
    out->error = "elf-io-topology: short read on section header table";
    return false;
  }

  // Read the section-header string table.
  const auto& shstrtab_hdr = sections[hdr.e_shstrndx];
  if (shstrtab_hdr.sh_size == 0 ||
      shstrtab_hdr.sh_size > static_cast<std::uint64_t>(64) * 1024 * 1024) {
    out->error = "elf-io-topology: invalid shstrtab size";
    return false;
  }
  std::vector<char> shstrtab(static_cast<std::size_t>(shstrtab_hdr.sh_size));
  if (!read_at(in, shstrtab_hdr.sh_offset, shstrtab.data(), shstrtab.size())) {
    out->error = "elf-io-topology: short read on shstrtab";
    return false;
  }

  // Walk every section name and classify.
  std::size_t recognized = 0U;
  for (const auto& s : sections) {
    const std::string name = section_name_at(shstrtab, s.sh_name);
    if (name.empty()) {
      continue;
    }
    if (name == kMonolithicIfmName) {
      out->monolithic_ifm = true;
      ++recognized;
      continue;
    }
    if (name == kMonolithicOfmName) {
      out->monolithic_ofm = true;
      ++recognized;
      continue;
    }
    if (const auto index = parse_ifm_section_index(name); index.has_value()) {
      place_at_index(&out->ifm_symbol_names, *index, name);
      ++recognized;
      continue;
    }
    if (const auto index = parse_ofm_section_index(name); index.has_value()) {
      place_at_index(&out->ofm_symbol_names, *index, name);
      ++recognized;
      continue;
    }
  }

  if (recognized == 0U) {
    out->error = "elf-io-topology: no IFM/OFM sections recognized";
    return false;
  }

  // A well-formed ELF declares either monolithic OR placeholders, never both
  // for the same direction. Prefer the placeholder list (it's strictly more
  // expressive) but flag the inconsistency for diagnostics.
  if (out->monolithic_ifm && !out->ifm_symbol_names.empty()) {
    out->error += "; warning: both monolithic and placeholder IFM sections present";
    out->monolithic_ifm = false;
  }
  if (out->monolithic_ofm && !out->ofm_symbol_names.empty()) {
    out->error += "; warning: both monolithic and placeholder OFM sections present";
    out->monolithic_ofm = false;
  }

  out->valid = true;
  return true;
}

bool elf_topology_requires_distinct_ifm_segments(const MlaElfIoTopology& topology) {
  if (!topology.valid || topology.monolithic_ifm) {
    return false;
  }
  return topology.ifm_symbol_names.size() >= 2U;
}

} // namespace simaai::neat::pipeline_internal::sima
