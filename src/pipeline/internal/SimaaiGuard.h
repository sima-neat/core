#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <memory>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

bool pipeline_uses_simaai(const std::string& pipeline);
bool pipeline_uses_mla(const std::string& pipeline);

void enforce_single_mla_pipeline(std::string_view where, const std::string& pipeline,
                                 const void* owner, const char* owner_desc);

std::shared_ptr<void> acquire_simaai_guard(std::string_view where, const std::string& pipeline,
                                           bool force, std::string* err_out);

// Test-only hook to reset the guard to a known state (no-op when guard disabled).
void reset_simaai_guard_for_test();

} // namespace simaai::neat::pipeline_internal
