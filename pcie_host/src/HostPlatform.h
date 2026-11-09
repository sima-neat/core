#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace simaai::neat::pcie::internal {

std::optional<std::filesystem::path> host_home_directory();
std::string host_path_string(const std::filesystem::path& path);
std::string open_ssh_executable(const char* command);
std::uint64_t host_process_id();

} // namespace simaai::neat::pcie::internal
