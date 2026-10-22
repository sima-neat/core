#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace simaai::neat::internal {

/// Longest one mla-rt run may take before Core stops it.
inline constexpr std::chrono::milliseconds kMlaRtTimeout{120000};

// Runs `mla_rt args...` and returns its combined stdout and stderr, keeping the last 4 KiB.
// Core starts mla-rt, so Core also stops it: after `timeout`, mla-rt's process group is killed.
// Throws when mla-rt cannot start, exits unsuccessfully, or times out.
std::string run_mla_rt(const std::filesystem::path& mla_rt, const std::vector<std::string>& args,
                       std::chrono::milliseconds timeout = kMlaRtTimeout);

} // namespace simaai::neat::internal
