// src/pipeline/internal/SimaaiGuard.cpp
#include "pipeline/internal/SimaaiGuard.h"

#include <string>

namespace simaai::neat::pipeline_internal {

bool pipeline_uses_simaai(const std::string& pipeline) {
  return pipeline.find("neat") != std::string::npos;
}

bool pipeline_uses_mla(const std::string& pipeline) {
  return pipeline.find("neatprocessmla") != std::string::npos;
}

void enforce_single_mla_pipeline(std::string_view where, const std::string& pipeline,
                                 const void* owner, const char* owner_desc) {
  // Guard disabled: allow multiple MLA pipelines per process.
  (void)where;
  (void)pipeline;
  (void)owner;
  (void)owner_desc;
}

std::shared_ptr<void> acquire_simaai_guard(std::string_view where, const std::string& pipeline,
                                           bool force, std::string* err_out) {
  (void)where;
  if (err_out)
    err_out->clear();
  if (!force && !pipeline_uses_simaai(pipeline))
    return {};
  return std::shared_ptr<void>(new int(0), [](void* p) { delete static_cast<int*>(p); });
}

void reset_simaai_guard_for_test() {
  // No-op: guard enforcement is disabled.
}

} // namespace simaai::neat::pipeline_internal
