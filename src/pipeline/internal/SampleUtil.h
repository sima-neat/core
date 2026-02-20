#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionOptions.h"

#include <memory>
#include <string>

namespace simaai::neat::pipeline_internal {

std::shared_ptr<void> make_sample_holder_from_bundle(const Sample& bundle, std::string* err);

} // namespace simaai::neat::pipeline_internal
