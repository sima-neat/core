#pragma once

#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <link.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::test_support {

struct LoadedPackageLibrary {
  std::string soname;
  std::string path;
};

namespace loaded_identity_detail {

struct Range {
  std::uintptr_t begin;
  std::uintptr_t end;
};

inline std::uintptr_t checked_add(std::uintptr_t left, std::uintptr_t right) {
  if (right > std::numeric_limits<std::uintptr_t>::max() - left)
    throw std::runtime_error("overflow in loaded ELF address");
  return left + right;
}

inline std::vector<Range> readable_loads(const dl_phdr_info& info) {
  std::vector<Range> ranges;
  for (std::size_t i = 0; i < info.dlpi_phnum; ++i) {
    const auto& header = info.dlpi_phdr[i];
    if (header.p_type == PT_LOAD && (header.p_flags & PF_R)) {
      const auto begin = checked_add(info.dlpi_addr, header.p_vaddr);
      ranges.push_back({begin, checked_add(begin, header.p_memsz)});
    }
  }
  return ranges;
}

inline std::size_t readable_bytes(const std::vector<Range>& ranges, std::uintptr_t address) {
  for (const auto& range : ranges)
    if (address >= range.begin && address < range.end)
      return range.end - address;
  return 0;
}

// Read the loaded image, not a basename or a replacement file at the same path.
// glibc relocates DT_STRTAB in ordinary DSOs; the vDSO and some loaders retain a
// relative virtual address. Accept either only when the complete read is bounded
// by a readable PT_LOAD range and DT_STRSZ.
inline std::string loaded_soname(const dl_phdr_info& info) {
  const auto ranges = readable_loads(info);
  for (std::size_t i = 0; i < info.dlpi_phnum; ++i) {
    const auto& header = info.dlpi_phdr[i];
    if (header.p_type != PT_DYNAMIC)
      continue;
    const auto address = checked_add(info.dlpi_addr, header.p_vaddr);
    if (address % alignof(ElfW(Dyn)) != 0 || header.p_memsz > readable_bytes(ranges, address))
      throw std::runtime_error("loaded ELF dynamic section is outside readable PT_LOAD");
    const auto* entries = reinterpret_cast<const ElfW(Dyn)*>(address);
    std::uintptr_t strings = 0;
    std::size_t string_bytes = 0;
    std::size_t name_offset = 0;
    bool has_strings = false;
    bool has_string_bytes = false;
    bool has_name = false;
    bool terminated = false;
    for (std::size_t j = 0; j < header.p_memsz / sizeof(ElfW(Dyn)); ++j) {
      const auto& entry = entries[j];
      if (entry.d_tag == DT_NULL) {
        terminated = true;
        break;
      }
      if (entry.d_tag == DT_STRTAB) {
        strings = entry.d_un.d_ptr;
        has_strings = true;
      } else if (entry.d_tag == DT_STRSZ) {
        string_bytes = entry.d_un.d_val;
        has_string_bytes = true;
      } else if (entry.d_tag == DT_SONAME) {
        name_offset = entry.d_un.d_val;
        has_name = true;
      }
    }
    if (!terminated)
      throw std::runtime_error("unterminated loaded ELF dynamic section");
    if (!has_name)
      return {};
    if (!has_strings || !has_string_bytes)
      throw std::runtime_error("loaded ELF SONAME lacks string-table metadata");
    if (readable_bytes(ranges, strings) == 0)
      strings = checked_add(info.dlpi_addr, strings);
    if (name_offset >= string_bytes || string_bytes > readable_bytes(ranges, strings))
      throw std::runtime_error("loaded ELF SONAME string table is outside readable PT_LOAD");
    const auto* name = reinterpret_cast<const char*>(checked_add(strings, name_offset));
    const auto* end = static_cast<const char*>(std::memchr(name, '\0', string_bytes - name_offset));
    if (!end || end == name)
      throw std::runtime_error("invalid loaded ELF SONAME");
    return {name, end};
  }
  return {};
}

struct Mapping {
  Range range;
  std::string path;
};

inline std::vector<Mapping> mappings() {
  std::ifstream maps("/proc/self/maps");
  if (!maps)
    throw std::runtime_error("cannot inspect actual loaded libraries");
  std::vector<Mapping> result;
  std::string line;
  while (std::getline(maps, line)) {
    const auto slash = line.find('/');
    const auto dash = line.find('-');
    const auto space = line.find(' ');
    if (slash == std::string::npos)
      continue;
    if (dash == std::string::npos || space == std::string::npos)
      throw std::runtime_error("invalid process mapping record");
    result.push_back({{std::stoull(line.substr(0, dash), nullptr, 16),
                       std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16)},
                      line.substr(slash)});
  }
  return result;
}

} // namespace loaded_identity_detail

inline std::vector<LoadedPackageLibrary>
selected_loaded_libraries(const std::map<std::string, std::set<std::string>>& expected_by_soname) {
  using namespace loaded_identity_detail;
  struct Context {
    const std::map<std::string, std::set<std::string>>& expected;
    std::vector<Mapping> mappings;
    std::vector<LoadedPackageLibrary> selected;
    std::exception_ptr failure;
  } context{expected_by_soname, mappings(), {}, {}};
  // Never propagate a C++ exception through the loader's callback ABI.
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* opaque) noexcept {
        auto& state = *static_cast<Context*>(opaque);
        try {
          const auto soname = loaded_soname(*info);
          const auto expected = state.expected.find(soname);
          if (expected == state.expected.end())
            return 0;
          const auto ranges = readable_loads(*info);
          for (const auto& range : ranges) {
            for (const auto& mapping : state.mappings) {
              if (range.begin < mapping.range.begin || range.begin >= mapping.range.end)
                continue;
              // A deleted mapping cannot be attributed to the currently installed
              // archive, even if a replacement now exists at the same pathname.
              if (!expected->second.count(mapping.path))
                throw std::runtime_error("loaded selected SONAME " + soname +
                                         " is outside its installed package: " + mapping.path);
              state.selected.push_back({soname, mapping.path});
              return 0;
            }
          }
          throw std::runtime_error("cannot attribute selected loaded SONAME: " + soname);
        } catch (...) {
          state.failure = std::current_exception();
          return 1;
        }
      },
      &context);
  if (context.failure)
    std::rethrow_exception(context.failure);
  return context.selected;
}

} // namespace simaai::neat::test_support
